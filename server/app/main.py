from datetime import datetime
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.auth import verify_firebase_token
from app.firebase_client import get_firestore_client
from app.rag_engine import answer_question, build_rag_record, process_expense_change, ExpenseIn, IncomeIn, SummaryIn

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

@app.put("/Incomes/{Income_id}", response_model=Income)
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

@app.delete("/Incomes/{Income_id}")
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
            "total_budget": data["total_budget"],
            "saving": data["saving"],
            "budget_details": data["budget_details"],
        })

    return summaries

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

# 질문
@app.post("/ask", response_model=AskResponse)
def ask_api(request: AskRequest, uid: str = Depends(verify_firebase_token)):
    return answer_question(uid, request.question)