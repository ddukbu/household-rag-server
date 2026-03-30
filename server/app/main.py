import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag_engine import (
    ensure_data_files,
    load_expenses,
    rebuild_rag_texts,
    answer_question,
)

app = FastAPI(title="HouseHold RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 후엔 필요한 도메인만 허용하는 게 더 안전함
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("data")
EXPENSES_FILE = DATA_DIR / "expenses.json"

DATA_DIR.mkdir(exist_ok=True)
if not EXPENSES_FILE.exists():
    EXPENSES_FILE.write_text("[]", encoding="utf-8")


class ExpenseIn(BaseModel):
    date: str
    category: str
    amount: int
    payment_method: str
    place: str
    memo: str


class Expense(ExpenseIn):
    id: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    references: List[str]


def save_expenses(expenses):
    EXPENSES_FILE.write_text(
        json.dumps(expenses, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    rebuild_rag_texts(expenses)


@app.on_event("startup")
def startup_event():
    ensure_data_files()
    rebuild_rag_texts(load_expenses())


@app.get("/")
def root():
    return {"message": "HouseHold RAG server is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/expenses", response_model=List[Expense])
def get_expenses():
    return load_expenses()


@app.post("/expenses", response_model=Expense)
def create_expense(expense_in: ExpenseIn):
    expenses = load_expenses()
    next_id = max([e["id"] for e in expenses], default=0) + 1

    expense = {
        "id": next_id,
        **expense_in.model_dump()
    }

    expenses.append(expense)
    save_expenses(expenses)
    return expense


@app.put("/expenses/{expense_id}", response_model=Expense)
def update_expense(expense_id: int, expense_in: ExpenseIn):
    expenses = load_expenses()

    for i, e in enumerate(expenses):
        if e["id"] == expense_id:
            updated = {
                "id": expense_id,
                **expense_in.model_dump()
            }
            expenses[i] = updated
            save_expenses(expenses)
            return updated

    raise HTTPException(status_code=404, detail="Expense not found")


@app.delete("/expenses/{expense_id}")
def delete_expense(expense_id: int):
    expenses = load_expenses()
    new_expenses = [e for e in expenses if e["id"] != expense_id]

    if len(new_expenses) == len(expenses):
        raise HTTPException(status_code=404, detail="Expense not found")

    save_expenses(new_expenses)
    return {"message": "deleted"}


@app.post("/ask", response_model=AskResponse)
def ask_api(request: AskRequest):
    return answer_question(request.question)