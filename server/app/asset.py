from datetime import datetime
from typing import Dict, Any

from fastapi import HTTPException
from pydantic import BaseModel

from app.firebase_client import get_firestore_client

db = get_firestore_client()


class InitialAssetRequest(BaseModel):
    initial_asset: int


class AssetOut(BaseModel):
    initial_asset: int
    current_asset: int
    total_income: int
    total_expense: int
    updated_at: str = ""


def asset_ref(uid: str):
    return (
        db.collection("users")
        .document(uid)
        .collection("assetStatus")
        .document("current")
    )


def save_initial_asset(uid: str, initial_asset: int) -> Dict[str, Any]:
    if initial_asset < 0:
        raise HTTPException(
            status_code=400,
            detail="기존 자산은 음수일 수 없습니다."
        )

    data = {
        "initial_asset": initial_asset,
        "updated_at": datetime.utcnow().isoformat(),
    }

    asset_ref(uid).set(data, merge=True)

    return load_asset_status(uid)


def calculate_total_income(uid: str) -> int:
    docs = (
        db.collection("users")
        .document(uid)
        .collection("Incomes")
        .stream()
    )

    return sum(doc.to_dict().get("amount", 0) for doc in docs)


def calculate_total_expense(uid: str) -> int:
    docs = (
        db.collection("users")
        .document(uid)
        .collection("expenses")
        .stream()
    )

    return sum(doc.to_dict().get("amount", 0) for doc in docs)


def load_asset_status(uid: str) -> Dict[str, Any]:
    doc = asset_ref(uid).get()

    if not doc.exists:
        raise HTTPException(
            status_code=404,
            detail="초기 자산이 설정되지 않았습니다."
        )

    data = doc.to_dict()

    initial_asset = data.get("initial_asset", 0)

    total_income = calculate_total_income(uid)
    total_expense = calculate_total_expense(uid)

    current_asset = initial_asset + total_income - total_expense

    update_data = {
        "current_asset": current_asset,
        "total_income": total_income,
        "total_expense": total_expense,
        "updated_at": datetime.utcnow().isoformat(),
    }

    asset_ref(uid).set(update_data, merge=True)

    return {
        "initial_asset": initial_asset,
        "current_asset": current_asset,
        "total_income": total_income,
        "total_expense": total_expense,
        "updated_at": update_data["updated_at"],
    }