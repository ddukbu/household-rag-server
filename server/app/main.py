from datetime import datetime
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.auth import verify_firebase_token
from app.firebase_client import get_firestore_client
from app.rag_engine import answer_question, build_rag_record, process_expense_change, ExpenseIn, IncomeIn, SummaryIn
from app.budget import (
    BudgetOut,
    FixedIncomeBudget,
    FixedExpenseBudget,
    SavingUpdateRequest,
    BudgetDetailsUpdateRequest,
    load_budgets,
    load_budget,
    load_fixed_incomes,
    load_fixed_expenses,
    update_saving,
    update_budget_details,
    create_fixed_income,
    update_fixed_income,
    delete_fixed_income,
    create_fixed_expense,
    update_fixed_expense,
    delete_fixed_expense,
    carry_over_budget_raw,
    BudgetDraftRequest,
    create_budget_draft,
    apply_budget_draft,
    cancel_budget_draft,
    refresh_total_budget,
)
from app.asset import (
    AssetOut,
    InitialAssetRequest,
    save_initial_asset,
    load_asset_status,
)

app = FastAPI(title="HouseHold RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = get_firestore_client()

#프로필 전용 클래스 추가
class SignUpProfile(BaseModel):
    email: str

class Expense(ExpenseIn):
    id: str

class Income(IncomeIn):
    id: str

class Summary(SummaryIn):
    id: str



class AskRequest(BaseModel):
    question: str


#응답 클래스에 측정 시간 멤버변수 추가
class AskResponse(BaseModel):
    answer: str
    references: List[str]
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float

@app.get("/")
def root():
    return {"message": "HouseHold RAG server is running"}


@app.get("/health")
def health():
    return {"status": "ok"}

#로그인한 사용자의 기본 사용자 문서를 Firestore에 만들어주는 함수, 사용자별 데이터 공간을 준비하는 초기화 API
@app.post("/profile/init")
def init_profile(profile: SignUpProfile, uid: str = Depends(verify_firebase_token)):
    user_ref = db.collection("users").document(uid)

    if not user_ref.get().exists:
        user_ref.set({
            "email": profile.email,
            "created_at": datetime.utcnow().isoformat()
        })

    return {"message": "profile initialized", "uid": uid}

# 지출
@app.get("/expenses", response_model=List[Expense])
def get_expenses(uid: str = Depends(verify_firebase_token)):
    docs = db.collection("users").document(uid).collection("expenses").stream()
    expenses = []

    for doc in docs:
        data = doc.to_dict()
        expenses.append({
            "id": doc.id,
            "date": data["date"],
            "time": data["time"],
            "is_fixed_expense": data["is_fixed_expense"],
            "category": data["category"],
            "amount": data["amount"],
            "payment_method": data["payment_method"],
            "place": data["place"],
            "memo": data["memo"],
        })

    return expenses

@app.post("/expenses", response_model=Expense)
def create_expense(expense_in: ExpenseIn, uid: str = Depends(verify_firebase_token)):
    print("POST /expenses called", flush=True)
    print("uid =", uid, flush=True)
    print("expense_in =", expense_in.model_dump(), flush=True)

    try:
        expenses_ref = db.collection("users").document(uid).collection("expenses")
        doc_ref = expenses_ref.document()

        record = build_rag_record(expense_in.model_dump())
        print("record built successfully", flush=True)

        doc_ref.set(record)
        print("saved to firestore", flush=True)

        # [수정] 요약본 업데이트 호출
        process_expense_change(uid, expense_in, mode="add")

        # 변동 지출 입력시 예산안 업데이트
        if not expense_in.is_fixed_expense:
            year_month = expense_in.date[:7]
            refresh_total_budget(uid, year_month)

        return {
            "id": doc_ref.id,
            **expense_in.model_dump()
        }
    except Exception as e:
        print("create_expense error =", str(e), flush=True)
        raise

@app.put("/expenses/{expense_id}", response_model=Expense)
def update_expense(expense_id: str, expense_in: ExpenseIn, uid: str = Depends(verify_firebase_token)):
    doc_ref = db.collection("users").document(uid).collection("expenses").document(expense_id)

    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="Expense not found")

    # [수정] 요약본 업데이트 호출(로직: 기존 값 차감 -> 새 값 반영)
    # 기존 데이터 차감
    old_data = doc_ref.get().to_dict()
    old_expense = ExpenseIn(**{k: v for k, v in old_data.items() if k in ExpenseIn.__fields__})
    process_expense_change(uid, old_expense, mode="delete")

    record = build_rag_record(expense_in.model_dump())
    doc_ref.set(record)

    # [수정] 요약본 업데이트 호출(로직: 기존 값 차감 -> 새 값 반영)
    # 새로운 데이터 요약본에 합산
    process_expense_change(uid, expense_in, mode="add")


    # 변동 지출 수정시 예산안 업데이트
    # 기존 지출의 월과 새 지출의 월이 다른 경우 둘 다 갱신 -> set이라서 year_month가 중복되면 한번만 갱신됨
    months_to_refresh = set()

    if not old_expense.is_fixed_expense:
        months_to_refresh.add(old_expense.date[:7])

    if not expense_in.is_fixed_expense:
        months_to_refresh.add(expense_in.date[:7])

    for year_month in months_to_refresh:
        refresh_total_budget(uid, year_month)

    
    return {
        "id": expense_id,
        **expense_in.model_dump()
    }

@app.delete("/expenses/{expense_id}")
def delete_expense(expense_id: str, uid: str = Depends(verify_firebase_token)):
    doc_ref = db.collection("users").document(uid).collection("expenses").document(expense_id)

    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="Expense not found")

    # [수정] 삭제 전 데이터를 가져와 요약본에서 차감
    expense_data = doc_ref.get().to_dict()
    # Pydantic 모델로 변환 (필요한 필드만 추출)
    expense_in = ExpenseIn(**{k: v for k, v in expense_data.items() if k in ExpenseIn.__fields__})
    process_expense_change(uid, expense_in, mode="delete")

    # 변동 지출 삭제시 예산안 업데이트
    if not expense_in.is_fixed_expense:
        refresh_total_budget(uid, expense_in.date[:7])

    doc_ref.delete()
    return {"message": "deleted"}

# 수익
@app.get("/Incomes", response_model=List[Income])
def get_Incomes(uid: str = Depends(verify_firebase_token)):
    docs = db.collection("users").document(uid).collection("Incomes").stream()
    incomes = []

    for doc in docs:
        data = doc.to_dict()
        incomes.append({
            "id": doc.id,
            "date": data["date"],
            "time": data["time"],
            "is_fixed_income": data["is_fixed_income"],
            "category": data["category"],
            "amount": data["amount"],
            "deposit_method": data["deposit_method"],
            "deposit_source": data["deposit_source"],
            "memo": data["memo"],
        })

    return incomes

@app.post("/Incomes", response_model=Income)
def create_Income(income_in: IncomeIn, uid: str = Depends(verify_firebase_token)):
    print("POST /Incomes called", flush=True)
    print("uid =", uid, flush=True)
    print("income_in =", income_in.model_dump(), flush=True)

    try:
        incomes_ref = db.collection("users").document(uid).collection("Incomes")
        doc_ref = incomes_ref.document()

        record = build_rag_record(income_in.model_dump())
        print("record built successfully", flush=True)

        doc_ref.set(record)
        print("saved to firestore", flush=True)

        # [수정] 요약본 업데이트 호출
        process_expense_change(uid, income_in, mode="add")

        return {
            "id": doc_ref.id,
            **income_in.model_dump()
        }
    except Exception as e:
        print("create_expense error =", str(e), flush=True)
        raise

@app.put("/Incomes/{income_id}", response_model=Income)
def update_Income(income_id: str, income_in: IncomeIn, uid: str = Depends(verify_firebase_token)):
    doc_ref = db.collection("users").document(uid).collection("Incomes").document(income_id)

    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="Income not found")

    # [수정] 요약본 업데이트 호출(로직: 기존 값 차감 -> 새 값 반영)
    # 기존 데이터 차감
    old_data = doc_ref.get().to_dict()
    old_income = IncomeIn(**{k: v for k, v in old_data.items() if k in IncomeIn.__fields__})
    process_expense_change(uid, old_income, mode="delete")

    record = build_rag_record(income_in.model_dump())
    doc_ref.set(record)

    # [수정] 요약본 업데이트 호출(로직: 기존 값 차감 -> 새 값 반영)
    # 새로운 데이터 요약본에 합산
    process_expense_change(uid, income_in, mode="add")
    
    return {
        "id": income_id,
        **income_in.model_dump()
    }

@app.delete("/Incomes/{income_id}")
def delete_Income(income_id: str, uid: str = Depends(verify_firebase_token)):
    doc_ref = db.collection("users").document(uid).collection("Incomes").document(income_id)

    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="Income not found")

    # [수정] 삭제 전 데이터를 가져와 요약본에서 차감
    Income_data = doc_ref.get().to_dict()
    # Pydantic 모델로 변환 (필요한 필드만 추출)
    income_in = IncomeIn(**{k: v for k, v in Income_data.items() if k in IncomeIn.__fields__})
    process_expense_change(uid, income_in, mode="delete")

    doc_ref.delete()
    return {"message": "deleted"}

# 월별 요약본
@app.get("/summaries", response_model=List[Summary])
def get_summary(uid: str = Depends(verify_firebase_token)):
    docs = db.collection("users").document(uid).collection("summaries").stream()
    summaries = []
    for doc in docs:
        data = doc.to_dict()
        summaries.append({
            "id": doc.id,
            "year_month": data["year_month"],
            "total_income": data["total_income"],
            "total_expense": data["total_expense"],
            "fixed_income_details": data["fixed_income_details"],
            "variable_income_details": data["variable_income_details"],
            "fixed_expense_details": data["fixed_expense_details"],
            "variable_expense_details": data["variable_expense_details"],
        })

    return summaries


@app.get("/chat-history")
def get_chat_history_api(uid: str = Depends(verify_firebase_token)):
    docs = (
        db.collection("users")
        .document(uid)
        .collection("chat_history")
        .order_by("created_at")
        .stream()
    )

    return [
        {
            "id": doc.id,
            **doc.to_dict()
        }
        for doc in docs
    ]


@app.post("/ask", response_model=AskResponse)
def ask_api(request: AskRequest, uid: str = Depends(verify_firebase_token)):
    return answer_question(uid, request.question)


@app.post("/analysis")
def analyze_spending(
    uid: str = Depends(verify_firebase_token)
):
    question = "최근 카테고리별 소비 패턴을 분석해줘."
    return answer_question(uid, question)


#앞으로 예산안 업데이트는 요약본에서 실행하지 않음.
"""
@app.put("/summaries/{summary_id}", response_model=Summary)
def update_summary_budget(summary_id: str, summary_in: SummaryIn, uid: str = Depends(verify_firebase_token)):
    doc_ref = db.collection("users").document(uid).collection("summaries").document(summary_id)
    doc_snapshot = doc_ref.get()

    if not doc_snapshot.exists:
        raise HTTPException(status_code=404, detail="Summary not found")

    # 1. 특정 필드(budget_details)만 업데이트
    update_data = {"budget_details": summary_in.budget_details}
    doc_ref.update(update_data)

    # 2. 업데이트된 최신 데이터를 가져와서 반환
    updated_doc = doc_ref.get().to_dict()
    
    return {
        "id": summary_id,
        **updated_doc
    }
"""


# =========================
# Assets
# =========================


@app.put("/assets/initial", response_model=AssetOut)
def update_initial_asset_api(
    request: InitialAssetRequest,
    uid: str = Depends(verify_firebase_token)
):
    return save_initial_asset(
        uid=uid,
        initial_asset=request.initial_asset
    )


@app.get("/assets", response_model=AssetOut)
def get_asset_status_api(
    uid: str = Depends(verify_firebase_token)
):
    return load_asset_status(uid)






# =========================
# Budgets
# =========================

@app.get("/budgets", response_model=List[BudgetOut])
def get_budgets(uid: str = Depends(verify_firebase_token)):
    return load_budgets(uid)


@app.get("/budgets/{year_month}", response_model=BudgetOut)
def get_budget(
    year_month: str,
    uid: str = Depends(verify_firebase_token)
):
    return load_budget(uid, year_month)


@app.put("/budgets/{year_month}/saving", response_model=BudgetOut)
def update_budget_saving_api(
    year_month: str,
    request: SavingUpdateRequest,
    uid: str = Depends(verify_firebase_token)
):
    return update_saving(
        uid=uid,
        year_month=year_month,
        saving=request.saving
    )


@app.put("/budgets/{year_month}/details", response_model=BudgetOut)
def update_budget_details_api(
    year_month: str,
    request: BudgetDetailsUpdateRequest,
    uid: str = Depends(verify_firebase_token)
):
    return update_budget_details(
        uid=uid,
        year_month=year_month,
        budget_details=request.budget_details,
        created_by="user"
    )


@app.post("/budgets/{year_month}/draft")
def create_budget_draft_api(
    year_month: str,
    request: BudgetDraftRequest,
    uid: str = Depends(verify_firebase_token)
):
    return create_budget_draft(
        uid=uid,
        year_month=year_month,
        request=request
    )


@app.post("/budgets/{year_month}/apply-draft")
def apply_budget_draft_api(
    year_month: str,
    uid: str = Depends(verify_firebase_token)
):
    return apply_budget_draft(
        uid=uid,
        year_month=year_month
    )


@app.delete("/budgets/{year_month}/draft")
def cancel_budget_draft_api(
    year_month: str,
    uid: str = Depends(verify_firebase_token)
):
    return cancel_budget_draft(
        uid=uid,
        year_month=year_month
    )


"""
#기존의 제안 즉시 예산안으로 적용하는 api는 사용x
@app.post("/budgets/{year_month}/recommend")
def recommend_budget_api(
    year_month: str,
    uid: str = Depends(verify_firebase_token)
):
    return recommend_and_save_budget(
        uid=uid,
        year_month=year_month
    )
"""


#없어도 get_budget에서 자동으로 기존 예산안을 이월 해주지만, 확정성을 위해서 놔둠.
@app.post("/budgets/{from_year_month}/carry-over/{to_year_month}")
def carry_over_budget_api(
    from_year_month: str,
    to_year_month: str,
    uid: str = Depends(verify_firebase_token)
):
    return carry_over_budget_raw(
        uid=uid,
        from_year_month=from_year_month,
        to_year_month=to_year_month
    )
    

# =========================
# Fixed incomes for budget
# =========================

@app.get("/budgets/{year_month}/fixed-incomes")
def get_fixed_incomes_api(
    year_month: str,
    uid: str = Depends(verify_firebase_token)
):
    return load_fixed_incomes(uid, year_month)


@app.post("/budgets/{year_month}/fixed-incomes")
def create_fixed_income_api(
    year_month: str,
    request: FixedIncomeBudget,
    uid: str = Depends(verify_firebase_token)
):
    return create_fixed_income(
        uid=uid,
        year_month=year_month,
        fixed_income=request
    )


@app.put("/budgets/{year_month}/fixed-incomes/{fixed_income_id}")
def update_fixed_income_api(
    year_month: str,
    fixed_income_id: str,
    request: FixedIncomeBudget,
    uid: str = Depends(verify_firebase_token)
):
    return update_fixed_income(
        uid=uid,
        year_month=year_month,
        fixed_income_id=fixed_income_id,
        fixed_income=request
    )


@app.delete("/budgets/{year_month}/fixed-incomes/{fixed_income_id}")
def delete_fixed_income_api(
    year_month: str,
    fixed_income_id: str,
    uid: str = Depends(verify_firebase_token)
):
    return delete_fixed_income(
        uid=uid,
        year_month=year_month,
        fixed_income_id=fixed_income_id
    )


# =========================
# Fixed expenses for budget
# =========================

@app.get("/budgets/{year_month}/fixed-expenses")
def get_fixed_expenses_api(
    year_month: str,
    uid: str = Depends(verify_firebase_token)
):
    return load_fixed_expenses(uid, year_month)


@app.post("/budgets/{year_month}/fixed-expenses")
def create_fixed_expense_api(
    year_month: str,
    request: FixedExpenseBudget,
    uid: str = Depends(verify_firebase_token)
):
    return create_fixed_expense(
        uid=uid,
        year_month=year_month,
        fixed_expense=request
    )


@app.put("/budgets/{year_month}/fixed-expenses/{fixed_expense_id}")
def update_fixed_expense_api(
    year_month: str,
    fixed_expense_id: str,
    request: FixedExpenseBudget,
    uid: str = Depends(verify_firebase_token)
):
    return update_fixed_expense(
        uid=uid,
        year_month=year_month,
        fixed_expense_id=fixed_expense_id,
        fixed_expense=request
    )


@app.delete("/budgets/{year_month}/fixed-expenses/{fixed_expense_id}")
def delete_fixed_expense_api(
    year_month: str,
    fixed_expense_id: str,
    uid: str = Depends(verify_firebase_token)
):
    return delete_fixed_expense(
        uid=uid,
        year_month=year_month,
        fixed_expense_id=fixed_expense_id
    )