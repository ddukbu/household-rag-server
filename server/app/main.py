from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.firebase_client import get_firestore_client
from app.rag_engine import answer_question

app = FastAPI(title="HouseHold RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = get_firestore_client()
expenses_ref = db.collection("expenses")


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


class AskResponse(BaseModel):
    answer: str
    references: List[str]


@app.get("/")
def root():
    return {"message": "HouseHold RAG server is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/expenses", response_model=List[Expense])
def get_expenses():
    docs = expenses_ref.stream()
    expenses = []
    for doc in docs:
        data = doc.to_dict()
        expenses.append({
            "id": doc.id,
            **data
        })
    return expenses


@app.post("/expenses", response_model=Expense)
def create_expense(expense_in: ExpenseIn):
    doc_ref = expenses_ref.document()
    doc_ref.set(expense_in.model_dump())
    return {
        "id": doc_ref.id,
        **expense_in.model_dump()
    }


@app.put("/expenses/{expense_id}", response_model=Expense)
def update_expense(expense_id: str, expense_in: ExpenseIn):
    doc_ref = expenses_ref.document(expense_id)
    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="Expense not found")

    doc_ref.set(expense_in.model_dump())
    return {
        "id": expense_id,
        **expense_in.model_dump()
    }


@app.delete("/expenses/{expense_id}")
def delete_expense(expense_id: str):
    doc_ref = expenses_ref.document(expense_id)
    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="Expense not found")

    doc_ref.delete()
    return {"message": "deleted"}


@app.post("/ask", response_model=AskResponse)
def ask_api(request: AskRequest):
    return answer_question(request.question)