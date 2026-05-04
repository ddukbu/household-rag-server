import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel

from app.firebase_client import get_firestore_client
from app.llm_client import call_gemini

db = get_firestore_client()


# =========================
# Models
# =========================

class FixedIncomeBudget(BaseModel):
    category: str
    amount: int
    memo: str = ""


class FixedExpenseBudget(BaseModel):
    category: str
    amount: int
    memo: str = ""


class SavingUpdateRequest(BaseModel):
    saving: int


class BudgetDetailsUpdateRequest(BaseModel):
    budget_details: Dict[str, int]


#class BudgetRecommendRequest(BaseModel):
#    saving: int


class BudgetDraftRequest(BaseModel):
    mode: str = "balanced"  # balanced, saving, relaxed
    user_message: str = ""


class BudgetDraftOut(BaseModel):
    type: str = "budget_draft"
    message: str
    year_month: str
    mode: str
    saving: int
    total_budget: int
    budget_details: Dict[str, int]
    remaining_budget_details: Dict[str, int]
    state: str


class BudgetOut(BaseModel):
    id: str
    year_month: str
    saving: int = 0
    total_budget: int = 0
    budget_details: Dict[str, int] = {}
    remaining_budget_details: Dict[str, int] = {}
    state: str = "good"
    created_by: str = "user"
    updated_at: str = ""


# =========================
# Helper refs
# =========================

def budget_ref(uid: str, year_month: str):
    return (
        db.collection("users")
        .document(uid)
        .collection("budgets")
        .document(year_month)
    )


def fixed_incomes_ref(uid: str, year_month: str):
    return budget_ref(uid, year_month).collection("fixedIncomes")


def fixed_expenses_ref(uid: str, year_month: str):
    return budget_ref(uid, year_month).collection("fixedExpenses")


def summary_ref(uid: str, year_month: str):
    return (
        db.collection("users")
        .document(uid)
        .collection("summaries")
        .document(year_month)
    )


def budget_draft_ref(uid: str, year_month: str):
    return (
        db.collection("users")
        .document(uid)
        .collection("budgetDrafts")
        .document(year_month)
    )

# =========================
# Load functions
# =========================

def load_budgets(uid: str) -> List[Dict[str, Any]]:
    docs = (
        db.collection("users")
        .document(uid)
        .collection("budgets")
        .stream()
    )

    result = []

    for doc in docs:
        data = doc.to_dict()
        result.append({
            "id": doc.id,
            "year_month": data.get("year_month", doc.id),
            "saving": data.get("saving", 0),
            "total_budget": data.get("total_budget", 0),
            "budget_details": data.get("budget_details", {}),
            "remaining_budget_details": data.get("remaining_budget_details", {}),
            "state": data.get("state", "good"),
            "created_by": data.get("created_by", "user"),
            "updated_at": data.get("updated_at", ""),
        })

    return result


def load_budget(uid: str, year_month: str) -> Dict[str, Any]:
    return load_or_create_budget(uid, year_month)


def load_fixed_incomes(uid: str, year_month: str) -> List[Dict[str, Any]]:
    docs = fixed_incomes_ref(uid, year_month).stream()

    return [
        {
            "id": doc.id,
            **doc.to_dict()
        }
        for doc in docs
    ]


def load_fixed_expenses(uid: str, year_month: str) -> List[Dict[str, Any]]:
    docs = fixed_expenses_ref(uid, year_month).stream()

    return [
        {
            "id": doc.id,
            **doc.to_dict()
        }
        for doc in docs
    ]


def load_summary(uid: str, year_month: str) -> Dict[str, Any]:
    doc = summary_ref(uid, year_month).get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Summary not found")

    return doc.to_dict()


# =========================
# Calculation / validation
# =========================

def calculate_total_budget(uid: str, year_month: str, saving: int) -> int:
    fixed_incomes = load_fixed_incomes(uid, year_month)
    fixed_expenses = load_fixed_expenses(uid, year_month)

    fixed_income_total = sum(item.get("amount", 0) for item in fixed_incomes)
    fixed_expense_total = sum(item.get("amount", 0) for item in fixed_expenses)

    return fixed_income_total - fixed_expense_total - saving


def calculate_remaining_budget_details(
    uid: str,
    year_month: str,
    total_budget: int,
    budget_details: Dict[str, int]
) -> Dict[str, int]:

    try:
        summary = load_summary(uid, year_month)
        variable_expense_details = summary.get("variable_expense_details", {})
    except HTTPException:
        variable_expense_details = {}

    remaining_budget_details = {}

    total_used_variable_expense = sum(variable_expense_details.values())

    remaining_budget_details["전체"] = total_budget - total_used_variable_expense

    for category, budget_amount in budget_details.items():
        spent_amount = variable_expense_details.get(category, 0)
        remaining_budget_details[category] = budget_amount - spent_amount

    return remaining_budget_details


def calculate_budget_state(
    saving: int,
    remaining_budget_details: Dict[str, int]
) -> str:
    remaining_total = remaining_budget_details.get("전체", 0)

    if remaining_total >= 0:
        return "good"
    elif saving + remaining_total >= 0:
        return "warning"
    else:
        return "bad"


def validate_budget_details(
    budget_details: Dict[str, int],
    total_budget: int
):
    if total_budget < 0:
        raise HTTPException(
            status_code=400,
            detail="가용 예산이 음수입니다."
        )

    for category, amount in budget_details.items():
        if not isinstance(amount, int):
            raise HTTPException(
                status_code=400,
                detail=f"{category} 예산 금액은 정수여야 합니다."
            )

        if amount < 0:
            raise HTTPException(
                status_code=400,
                detail=f"{category} 예산 금액은 음수일 수 없습니다."
            )

    if sum(budget_details.values()) > total_budget:
        raise HTTPException(
            status_code=400,
            detail="카테고리별 예산 합계가 가용 예산을 초과했습니다."
        )


def refresh_total_budget(uid: str, year_month: str) -> Dict[str, Any]:
    budget = load_budget(uid, year_month)
    saving = budget.get("saving", 0)
    budget_details = budget.get("budget_details", {})

    total_budget = calculate_total_budget(uid, year_month, saving)

    validate_budget_details(budget_details, total_budget)

    remaining_budget_details = calculate_remaining_budget_details(
        uid=uid,
        year_month=year_month,
        total_budget=total_budget,
        budget_details=budget_details
    )

    state = calculate_budget_state(
        saving=saving,
        remaining_budget_details=remaining_budget_details
    )

    update_data = {
        "year_month": year_month,
        "saving": saving,
        "total_budget": total_budget,
        "budget_details": budget_details,
        "remaining_budget_details": remaining_budget_details,
        "state": state,
        "updated_at": datetime.utcnow().isoformat(),
    }

    budget_ref(uid, year_month).set(update_data, merge=True)

    return load_budget(uid, year_month)


# =========================
# Budget main document
# =========================

def update_saving(
    uid: str,
    year_month: str,
    saving: int
) -> Dict[str, Any]:
    current_budget = load_budget(uid, year_month)
    budget_details = current_budget.get("budget_details", {})

    total_budget = calculate_total_budget(uid, year_month, saving)

    validate_budget_details(budget_details, total_budget)

    remaining_budget_details = calculate_remaining_budget_details(
        uid=uid,
        year_month=year_month,
        total_budget=total_budget,
        budget_details=budget_details
    )

    state = calculate_budget_state(
        saving=saving,
        remaining_budget_details=remaining_budget_details
    )

    data = {
        "year_month": year_month,
        "saving": saving,
        "total_budget": total_budget,
        "budget_details": budget_details,
        "remaining_budget_details": remaining_budget_details,
        "state": state,
        "updated_at": datetime.utcnow().isoformat(),
    }

    budget_ref(uid, year_month).set(data, merge=True)

    return load_budget(uid, year_month)


def update_budget_details(
    uid: str,
    year_month: str,
    budget_details: Dict[str, int],
    created_by: str = "user"
) -> Dict[str, Any]:
    budget = load_budget(uid, year_month)
    saving = budget.get("saving", 0)

    total_budget = calculate_total_budget(uid, year_month, saving)

    validate_budget_details(budget_details, total_budget)

    remaining_budget_details = calculate_remaining_budget_details(
        uid=uid,
        year_month=year_month,
        total_budget=total_budget,
        budget_details=budget_details
    )

    state = calculate_budget_state(
        saving=saving,
        remaining_budget_details=remaining_budget_details
    )

    data = {
        "year_month": year_month,
        "saving": saving,
        "total_budget": total_budget,
        "budget_details": budget_details,
        "remaining_budget_details": remaining_budget_details,
        "state": state,
        "created_by": created_by,
        "updated_at": datetime.utcnow().isoformat(),
    }

    budget_ref(uid, year_month).set(data, merge=True)

    return load_budget(uid, year_month)


# =========================
# Carry over budget or create budget
# =========================

def get_previous_year_month(year_month: str) -> str:
    dt = datetime.strptime(year_month, "%Y-%m")
    prev = dt - relativedelta(months=1)
    return prev.strftime("%Y-%m")

def get_budget_if_exists(uid: str, year_month: str) -> Optional[Dict[str, Any]]:
    doc = budget_ref(uid, year_month).get()

    if not doc.exists:
        return None

    data = doc.to_dict()

    return {
        "id": doc.id,
        "year_month": data.get("year_month", doc.id),
        "saving": data.get("saving", 0),
        "total_budget": data.get("total_budget", 0),
        "budget_details": data.get("budget_details", {}),
        "remaining_budget_details": data.get("remaining_budget_details", {}),
        "state": data.get("state", "good"),
        "created_by": data.get("created_by", "user"),
        "updated_at": data.get("updated_at", ""),
    }


def create_empty_budget(uid: str, year_month: str) -> Dict[str, Any]:
    data = {
        "year_month": year_month,
        "saving": 0,
        "total_budget": 0,
        "budget_details": {},
        "remaining_budget_details": {},
        "state": "good",
        "created_by": "user",
        "updated_at": datetime.utcnow().isoformat(),
    }

    budget_ref(uid, year_month).set(data, merge=True)

    return {
        "id": year_month,
        **data
    }


def carry_over_budget_raw(
    uid: str,
    from_year_month: str,
    to_year_month: str
) -> Dict[str, Any]:

    from_budget = get_budget_if_exists(uid, from_year_month)

    if from_budget is None:
        raise HTTPException(
            status_code=404,
            detail="이월할 이전 달 예산안이 없습니다."
        )

    saving = from_budget.get("saving", 0)
    budget_details = from_budget.get("budget_details", {})

    if not budget_details:
        raise HTTPException(
            status_code=400,
            detail="이월할 카테고리별 예산안이 없습니다."
        )

    # 기존 다음 달 고정 수입/지출 제거
    for doc in fixed_incomes_ref(uid, to_year_month).stream():
        doc.reference.delete()

    for doc in fixed_expenses_ref(uid, to_year_month).stream():
        doc.reference.delete()

    # 이전 달 고정 수입 복사
    for item in load_fixed_incomes(uid, from_year_month):
        fixed_incomes_ref(uid, to_year_month).document().set({
            "category": item.get("category", ""),
            "amount": item.get("amount", 0),
            "memo": item.get("memo", ""),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        })

    # 이전 달 고정 지출 복사
    for item in load_fixed_expenses(uid, from_year_month):
        fixed_expenses_ref(uid, to_year_month).document().set({
            "category": item.get("category", ""),
            "amount": item.get("amount", 0),
            "memo": item.get("memo", ""),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        })

    total_budget = calculate_total_budget(
        uid=uid,
        year_month=to_year_month,
        saving=saving
    )

    validate_budget_details(
        budget_details=budget_details,
        total_budget=total_budget
    )

    # 다음 달은 변동 지출이 0원이라고 보고 초기화
    remaining_budget_details = {
        "전체": total_budget
    }

    for category, amount in budget_details.items():
        remaining_budget_details[category] = amount

    state = calculate_budget_state(
        saving=saving,
        remaining_budget_details=remaining_budget_details
    )

    data = {
        "year_month": to_year_month,
        "saving": saving,
        "total_budget": total_budget,
        "budget_details": budget_details,
        "remaining_budget_details": remaining_budget_details,
        "state": state,
        "created_by": "carry_over",
        "updated_at": datetime.utcnow().isoformat(),
    }

    budget_ref(uid, to_year_month).set(data, merge=True)

    return {
        "id": to_year_month,
        **data
    }


def load_or_create_budget(uid: str, year_month: str) -> Dict[str, Any]:
    current_budget = get_budget_if_exists(uid, year_month)

    if current_budget is not None:
        return current_budget

    previous_year_month = get_previous_year_month(year_month)
    previous_budget = get_budget_if_exists(uid, previous_year_month)

    if previous_budget is not None and previous_budget.get("budget_details"):
        return carry_over_budget_raw(
            uid=uid,
            from_year_month=previous_year_month,
            to_year_month=year_month
        )

    return create_empty_budget(uid, year_month)

# =========================
# Fixed incomes CRUD
# =========================

def create_fixed_income(
    uid: str,
    year_month: str,
    fixed_income: FixedIncomeBudget
) -> Dict[str, Any]:
    doc_ref = fixed_incomes_ref(uid, year_month).document()

    doc_ref.set({
        **fixed_income.model_dump(),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    })

    budget = refresh_total_budget(uid, year_month)
    fixed_incomes = load_fixed_incomes(uid, year_month)

    return {
        "budget": budget,
        "fixed_incomes": fixed_incomes,
        "created_fixed_income": {
            "id": doc_ref.id,
            **fixed_income.model_dump()
        }
    }


def update_fixed_income(
    uid: str,
    year_month: str,
    fixed_income_id: str,
    fixed_income: FixedIncomeBudget
) -> Dict[str, Any]:
    doc_ref = fixed_incomes_ref(uid, year_month).document(fixed_income_id)

    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="Fixed income not found")

    doc_ref.update({
        **fixed_income.model_dump(),
        "updated_at": datetime.utcnow().isoformat(),
    })

    budget = refresh_total_budget(uid, year_month)
    fixed_incomes = load_fixed_incomes(uid, year_month)

    return {
        "budget": budget,
        "fixed_incomes": fixed_incomes,
        "updated_fixed_income": {
            "id": fixed_income_id,
            **fixed_income.model_dump()
        }
    }


def delete_fixed_income(
    uid: str,
    year_month: str,
    fixed_income_id: str
) -> Dict[str, Any]:
    doc_ref = fixed_incomes_ref(uid, year_month).document(fixed_income_id)

    doc = doc_ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Fixed income not found")

    deleted_data = doc.to_dict()

    doc_ref.delete()

    budget = refresh_total_budget(uid, year_month)
    fixed_incomes = load_fixed_incomes(uid, year_month)

    return {
        "budget": budget,
        "fixed_incomes": fixed_incomes,
        "deleted_fixed_income": {
            "id": fixed_income_id,
            **deleted_data
        }
    }


# =========================
# Fixed expenses CRUD
# =========================

def create_fixed_expense(
    uid: str,
    year_month: str,
    fixed_expense: FixedExpenseBudget
) -> Dict[str, Any]:
    doc_ref = fixed_expenses_ref(uid, year_month).document()

    doc_ref.set({
        **fixed_expense.model_dump(),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    })

    budget = refresh_total_budget(uid, year_month)
    fixed_expenses = load_fixed_expenses(uid, year_month)

    return {
        "budget": budget,
        "fixed_expenses": fixed_expenses,
        "created_fixed_expense": {
            "id": doc_ref.id,
            **fixed_expense.model_dump()
        }
    }


def update_fixed_expense(
    uid: str,
    year_month: str,
    fixed_expense_id: str,
    fixed_expense: FixedExpenseBudget
) -> Dict[str, Any]:
    doc_ref = fixed_expenses_ref(uid, year_month).document(fixed_expense_id)

    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="Fixed expense not found")

    doc_ref.update({
        **fixed_expense.model_dump(),
        "updated_at": datetime.utcnow().isoformat(),
    })

    budget = refresh_total_budget(uid, year_month)
    fixed_expenses = load_fixed_expenses(uid, year_month)

    return {
        "budget": budget,
        "fixed_expenses": fixed_expenses,
        "updated_fixed_expense": {
            "id": fixed_expense_id,
            **fixed_expense.model_dump()
        }
    }


def delete_fixed_expense(
    uid: str,
    year_month: str,
    fixed_expense_id: str
) -> Dict[str, Any]:
    doc_ref = fixed_expenses_ref(uid, year_month).document(fixed_expense_id)

    doc = doc_ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Fixed expense not found")

    deleted_data = doc.to_dict()

    doc_ref.delete()

    budget = refresh_total_budget(uid, year_month)
    fixed_expenses = load_fixed_expenses(uid, year_month)

    return {
        "budget": budget,
        "fixed_expenses": fixed_expenses,
        "deleted_fixed_expense": {
            "id": fixed_expense_id,
            **deleted_data
        }
    }


# =========================
# AI recommendation
# =========================

def parse_ai_budget_response(answer: str) -> Dict[str, Any]:
    try:
        result = json.loads(answer)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="AI 응답을 JSON으로 변환할 수 없습니다."
        )

    if result.get("type") != "budget_recommendation":
        raise HTTPException(
            status_code=500,
            detail="AI 응답이 예산 추천 형식이 아닙니다."
        )

    if "budget_details" not in result:
        raise HTTPException(
            status_code=500,
            detail="AI 응답에 budget_details가 없습니다."
        )

    return result


def get_budget_mode_instruction(mode: str) -> str:
    if mode == "saving":
        return """
[예산 성향]
절약 모드:
- 최소 저금 금액을 우선 보호한다.
- 필수 생활비인 식비, 교통비는 너무 무리하게 줄이지 않는다.
- 쇼핑, 문화생활, 기타 같은 선택 지출은 보수적으로 배정한다.
"""
    elif mode == "relaxed":
        return """
[예산 성향]
여유 모드:
- 최소 저금 금액은 유지하되 생활 만족도를 고려한다.
- 식비, 문화생활, 여가성 지출에 조금 더 여유를 둔다.
- 과도한 절약보다는 실천 가능한 예산을 우선한다.
"""
    else:
        return """
[예산 성향]
균형 모드:
- 기본 생활비와 여가비를 적당히 유지한다.
- 최근 소비 패턴을 참고하되 과소비 카테고리는 완만하게 조정한다.
- 일반적인 기본 예산안으로 추천한다.
"""


def recommend_budget_with_ai(
    uid: str,
    year_month: str,
    total_budget: int,
    saving: int,
    cur_budget_details: Dict[str, Any],
    cur_remaining_budget_details: Dict[str, Any]
) -> Dict[str, Any]:
    summary = load_summary(uid, year_month)
    fixed_incomes = load_fixed_incomes(uid, year_month)
    fixed_expenses = load_fixed_expenses(uid, year_month)


    variable_expense_details = summary.get("variable_expense_details", {})
    variable_categories = list(variable_expense_details.keys())

    prompt = f"""
너는 사용자의 소비 계획을 도와주는 예산 추천 AI이다.

중요:
- [월별 요약]은 실제 발생한 수입/지출 통계이다.
- [예산안용 고정 수입/지출]은 사용자가 예산 계획을 위해 따로 입력한 데이터이다.
- 가용 예산 계산에는 [예산안용 고정 수입/지출]과 [저축 금액]만 사용한다.
- 너는 이미 계산된 [예산 가용 금액]을 변동 지출 카테고리별로 나누면 된다.

[월]
{year_month}

[월별 요약 - 실제 변동 지출 기록]
{variable_expense_details}

[예산안용 고정 수입]
{fixed_incomes}

[예산안용 고정 지출]
{fixed_expenses}

[저축 금액]
{saving}

[예산 가용 금액]
{total_budget}

[현재 산정된 변동 지출 카테고리 별 예산안]
{cur_budget_details}

[변동 지출 카테고리별 남은 예산]
{cur_remaining_budget_details}

[사용 가능한 변동 지출 카테고리]
{variable_categories}

반드시 아래 JSON 형식으로만 답해라.
JSON 바깥에 설명 문장을 절대 쓰지 마라.

{{
  "type": "budget_recommendation",
  "message": "사용자에게 보여줄 예산 추천 설명",
  "year_month": "{year_month}",
  "saving": {saving},
  "total_budget": {total_budget},
  "budget_details": {{
    "카테고리명": 예산금액
  }}
}}

조건:
1. type은 반드시 "budget_recommendation"이어야 한다.
2. budget_details의 key는 사용 가능한 변동 지출 카테고리 중에서만 선택해라.
3. budget_details의 value는 반드시 정수여야 한다.
4. budget_details의 총합은 total_budget을 넘으면 안 된다.
5. JSON만 출력해라.
6. 현재 산정된 예산안이 비어 있지 않다면, 기존 예산안을 참고하여 더 현실적인 방향으로 조정해라.
7. 현재 예산안이 비어 있다면, 실제 변동 지출 기록을 기준으로 새 예산안을 만들어라.
"""

    answer = call_gemini(prompt)
    return parse_ai_budget_response(answer)


def recommend_and_save_budget(
    uid: str,
    year_month: str
) -> Dict[str, Any]:

    budget = load_budget(uid, year_month)

    #saving = request.saving
    saving = budget.get("saving", 0)

    total_budget = calculate_total_budget(uid, year_month, saving)
    #total_budget = budget.get("total_budget")

    cur_budget_details = budget.get("budget_details", {})
    cur_remaining_budget_details = budget.get("remaining_budget_details", {})

    ai_result = recommend_budget_with_ai(
        uid=uid,
        year_month=year_month,
        total_budget=total_budget,
        saving=saving,
        cur_budget_details=cur_budget_details,
        cur_remaining_budget_details=cur_remaining_budget_details
    )

    budget_details = ai_result["budget_details"]

    validate_budget_details(
        budget_details=budget_details,
        total_budget=total_budget
    )

    remaining_budget_details = calculate_remaining_budget_details(
        uid=uid,
        year_month=year_month,
        total_budget=total_budget,
        budget_details=budget_details
    )

    state = calculate_budget_state(
        saving=saving,
        remaining_budget_details=remaining_budget_details
    )

    data = {
        "year_month": year_month,
        "saving": saving,
        "total_budget": total_budget,
        "budget_details": budget_details,
        "remaining_budget_details": remaining_budget_details,
        "state": state,
        "created_by": "ai",
        "updated_at": datetime.utcnow().isoformat(),
    }

    budget_ref(uid, year_month).set(data, merge=True)

    return {
        "type": "budget_recommendation",
        "message": ai_result.get("message", "AI 추천 예산안이 저장되었습니다."),
        "budget": load_budget(uid, year_month)
    }