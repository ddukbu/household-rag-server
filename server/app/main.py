from datetime import datetime
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.auth import verify_firebase_token
from app.firebase_client import get_firestore_client
from app.rag_engine import answer_question, build_expense_rag_record

app = FastAPI(title="HouseHold RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = get_firestore_client()
#expenses_ref = db.collection("expenses")
#-> 각 함수에서 uid 인자에 맞는 expenses를 따로 호출

#프로필 전용 클래스 추가
class SignUpProfile(BaseModel):
    email: str


class ExpenseIn(BaseModel):
    date: str
    category: str
    amount: int
    payment_method: str
    place: str
    memo: str


class Expense(ExpenseIn):
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
    expenses_ref = db.collection("users").document(uid).collection("expenses")
    doc_ref = expenses_ref.document()

    record = build_expense_rag_record(expense_in.model_dump())
    doc_ref.set(record)

    return {
        "id": doc_ref.id,
        **expense_in.model_dump()
    }


@app.put("/expenses/{expense_id}", response_model=Expense)
def update_expense(expense_id: str, expense_in: ExpenseIn, uid: str = Depends(verify_firebase_token)):
    doc_ref = db.collection("users").document(uid).collection("expenses").document(expense_id)

    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="Expense not found")

    record = build_expense_rag_record(expense_in.model_dump())
    doc_ref.set(record)

    return {
        "id": expense_id,
        **expense_in.model_dump()
    }


@app.delete("/expenses/{expense_id}")
def delete_expense(expense_id: str, uid: str = Depends(verify_firebase_token)):
    doc_ref = db.collection("users").document(uid).collection("expenses").document(expense_id)

    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="Expense not found")

    doc_ref.delete()
    return {"message": "deleted"}


@app.post("/ask", response_model=AskResponse)
def ask_api(request: AskRequest, uid: str = Depends(verify_firebase_token)):
    return answer_question(uid, request.question)