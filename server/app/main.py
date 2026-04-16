from datetime import datetime
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.auth import verify_firebase_token
from app.firebase_client import get_firestore_client
from app.rag_engine import answer_question, build_expense_rag_record, process_expense_change, ExpenseIn

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


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    references: List[str]

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

@app.get("/expenses", response_model=List[Expense])
def get_expenses(uid: str = Depends(verify_firebase_token)):
    docs = db.collection("users").document(uid).collection("expenses").stream()
    expenses = []

    for doc in docs:
        data = doc.to_dict()
        expenses.append({
            "id": doc.id,
            "date": data["date"],
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

        record = build_expense_rag_record(expense_in.model_dump())
        print("record built successfully", flush=True)

        doc_ref.set(record)
        print("saved to firestore", flush=True)

        # [수정] 요약본 업데이트 호출
        process_expense_change(expense_in, mode="add")

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
    process_expense_change(old_expense, mode="delete")

    record = build_expense_rag_record(expense_in.model_dump())
    doc_ref.set(record)

    # [수정] 요약본 업데이트 호출(로직: 기존 값 차감 -> 새 값 반영)
    # 새로운 데이터 요약본에 합산
    process_expense_change(expense_in, mode="add")
    
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
    process_expense_change(expense_in, mode="delete")

    doc_ref.delete()
    return {"message": "deleted"}

@app.post("/ask", response_model=AskResponse)
def ask_api(request: AskRequest, uid: str = Depends(verify_firebase_token)):
    return answer_question(uid, request.question)