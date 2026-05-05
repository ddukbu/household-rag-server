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


def asset_history_ref(uid: str, year_month: str):
    return (
        db.collection("users")
        .document(uid)
        .collection("assetHistory")
        .document(year_month)
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


def rebuild_asset_history(uid: str) -> Dict[str, Any]:
    current = load_asset_status(uid)
    current_asset = current["current_asset"]

    summaries = load_summaries(uid)

    if not summaries:
        return {
            "message": "월별 요약본이 없어 자산 히스토리를 생성하지 않았습니다.",
            "current_asset": current_asset,
            "asset_history": []
        }

    summaries.sort(key=lambda x: x["year_month"], reverse=True)

    next_asset = current_asset
    history = []

    for summary in summaries:
        year_month = summary["year_month"]
        total_income = summary.get("total_income", 0)
        total_expense = summary.get("total_expense", 0)
        net_change = total_income - total_expense

        month_asset = next_asset

        data = {
            "year_month": year_month,
            "asset": month_asset,
            "total_income": total_income,
            "total_expense": total_expense,
            "net_change": net_change,
            "updated_at": datetime.utcnow().isoformat(),
        }

        asset_history_ref(uid, year_month).set(data, merge=True)
        history.append(data)

        next_asset = month_asset - net_change

    history.sort(key=lambda x: x["year_month"])

    return {
        "message": "월별 자산 히스토리를 갱신했습니다.",
        "current_asset": current_asset,
        "asset_history": history
    }


def load_summaries(uid: str) -> List[Dict[str, Any]]:
    docs = (
        db.collection("users")
        .document(uid)
        .collection("summaries")
        .stream()
    )

    result = []

    for doc in docs:
        data = doc.to_dict()
        result.append({
            "year_month": data.get("year_month", doc.id),
            "total_income": data.get("total_income", 0),
            "total_expense": data.get("total_expense", 0),
        })

    return result


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