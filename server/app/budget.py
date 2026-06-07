import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel

from app.firebase_client import get_firestore_client
from app.llm_client import call_gemini, call_embed_api
from app.rag_utils import load_chat_history, retrieve_relevant_chat_history, save_chat_history

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

    try:
        summary = load_summary(uid, year_month)
        variable_expense_details = summary.get("variable_expense_details", {})
    except HTTPException:
        variable_expense_details = {}

    # summary의 변동 지출 카테고리가 budget_details에 없으면 0원으로 추가
    for category in variable_expense_details.keys():
        if category not in budget_details:
            budget_details[category] = 0

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
    text = f"{year_month}"
    embedding = call_embed_api(text)

    data = {
        "year_month": year_month,
        "saving": 0,
        "total_budget": 0,
        "budget_details": {},
        "remaining_budget_details": {},
        "state": "good",
        "created_by": "user",
        "updated_at": datetime.utcnow().isoformat(),
        "embedding": embedding,
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
        new_budget = carry_over_budget_raw(
            uid=uid,
            from_year_month=previous_year_month,
            to_year_month=year_month
        )
        text = f"{year_month}"
        embedding = call_embed_api(text)
        new_budget["embedding"] = embedding
        return new_budget

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

"""
#AI 예산안 추천 프롬프트에 넣을 채팅 기록을 RAG 방식으로 전달하는 것으로 대체 -> 추후 이 방식이 더 정확도 높을 경우 변경?
def load_budget_chat_history(uid: str) -> List[Dict[str, Any]]:
    docs = (
        db.collection("users")
        .document(uid)
        .collection("chat_history")
        .stream()
    )

    histories = []

    for doc in docs:
        data = doc.to_dict()
        histories.append({
            "id": doc.id,
            **data
        })

    return histories[-10:]
"""


def parse_ai_budget_response(answer: str) -> Dict[str, Any]:
    cleaned = answer.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1).strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "", 1).strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="AI 응답을 JSON으로 변환할 수 없습니다."
        )

    if result.get("type") != "budget_recommendation":
        raise HTTPException(status_code=500, detail="AI 응답이 예산 추천 형식이 아닙니다.")

    if "budget_details" not in result:
        raise HTTPException(status_code=500, detail="AI 응답에 budget_details가 없습니다.")

    return result


def get_budget_mode_instruction(mode: str) -> str:
    if mode == "saving":
        return """
[예산 성향: 절약 모드]
- 목표: 지출을 극단적으로 통제하고, 자산의 절반 이상을 강제 저축/비상금으로 격리합니다.
- 핵심 연산 규칙:
  1. [미할당 여유 자금]을 최우선으로 '총 가용 금액의 최소 50%' 이상 무조건 확보하세요.
  2. 필수 생활비(식비, 교통비)는 사용자의 과거 평균 소비액의 '최대 70%' 수준으로 방어적 배정합니다.
  3. 선택 지출(쇼핑, 문화 등)은 최하위 우선순위이며, 필수 생활비를 배정하고 남은 잔액이 없다면 '0원'으로 배정해도 무방합니다. 
- 유연성 예외 조항: 과거 고정 지출(월세, 보험료 등)이 총액의 50%를 초과하는 불가능한 상황에 한해서만, 미할당 여유 자금을 최하 40%까지 단계적으로 하향 조정할 수 있습니다.
"""
    elif mode == "relaxed":
        return """
[예산 성향: 균형 모드]
- 목표: 일상적인 삶의 질을 유지하면서도, 자산의 3분의 1이상을 안정적으로 저축합니다.
- 핵심 연산 규칙:
  1. [미할당 여유 자금]으로 '총 가용 금액의 30% ~ 40%'를 안정적으로 격리하세요.
  2. 필수 생활비(식비, 교통비)는 사용자의 최근 소비 패턴 평균치(100%)를 그대로 유지하여 일상의 불편함을 최소화합니다.
  3. 선택 지출(쇼핑, 문화 등)은 과거 과소비했던 카테고리 위주로 '10~20% 하향 조정'하여 배정합니다.
- 유연성 예외 조항: 당월에 경조사나 세금 등 일시적 대형 지출이 예상되는 데이터가 있다면, 미할당 여유 자금을 20%까지 줄이고 해당 카테고리에 일시 반영할 수 있습니다.
"""
    else:
        return """
[예산 성향: 여유 모드]
- 목표: 스트레스 없는 현실적인 예산을 수립하여 실천 가능성과 소비 만족도를 극대화합니다.
- 핵심 연산 규칙:
  1. [미할당 여유 자금]은 최소한의 안전장치인 '총 가용 금액의 10% ~ 15%'만 남겨둡니다.
  2. 필수 생활비는 사용자 평균치의 110% 수준으로 넉넉하게 잡아 예산 초과 스트레스를 방지합니다.
  3. 선택 지출(쇼핑, 문화, 여가 등)에 가장 높은 유연성을 부여하여, 사용자의 핵심 관심사 카테고리에 예산을 우선 증액 배정합니다.
- 유연성 예외 조항: 사용자가 당월 특별한 절약 목표를 추가 입력하지 않는 한, 미할당 자금보다는 당월의 대형 소비 계획(여행, 가전 구매 등)에 예산을 우선 유연하게 할당합니다.
"""


def get_budget_mode_question(mode: str, user_message: str = "") -> str:
    button_text = {
        "saving": "특수 버튼 : 절약",
        "relaxed": "특수 버튼 : 여유",
        "balanced": "특수 버튼 : 균형",
    }.get(mode, "특수 버튼 : 균형")

    if user_message.strip():
        return f"{button_text}\n{user_message.strip()}"

    return button_text


def recommend_budget_with_ai(
    uid: str,
    year_month: str,
    total_budget: int,
    saving: int,
    cur_budget_details: Dict[str, Any],
    cur_remaining_budget_details: Dict[str, Any],
    mode: str,
    user_message: str = ""
) -> Dict[str, Any]:
    
    try:
        summary = load_summary(uid, year_month)
    except HTTPException:
        summary = {
            "variable_expense_details": {}
        }
    
    fixed_incomes = load_fixed_incomes(uid, year_month)
    fixed_expenses = load_fixed_expenses(uid, year_month)
    chat_histories = load_chat_history(uid)

    #현재 사용자가 요청한 예산안 추천에 대해 RAG 방식으로 유사한 채팅 기록을 찾아 AI에게 전달하기 위한 임시 쿼리
    query_for_history = f"""
    예산안 추천 요청
    월: {year_month}
    모드: {mode}
    사용자 요청: {user_message}
    현재 예산안: {cur_budget_details}
    현재 남은 예산: {cur_remaining_budget_details}
    """

    relevant_chat_histories = retrieve_relevant_chat_history(
        query=query_for_history,
        histories=chat_histories,
        threshold=0.75,
        min_k=0,
        max_k=3
    )


    variable_expense_details = summary.get("variable_expense_details", {})
    variable_categories = list(variable_expense_details.keys())

    mode_instruction = get_budget_mode_instruction(mode)

    prompt = f"""
너는 사용자의 소비 계획을 도와주는 예산 추천 AI이다.

중요:
- [월별 요약]은 실제 발생한 수입/지출 통계이다.
- [예산안용 고정 수입/지출]은 사용자가 예산 계획을 위해 따로 입력한 데이터이다.
- 가용 예산 계산에는 [예산안용 고정 수입/지출]과 [저축 금액]만 사용한다.
- 너는 이미 계산된 [예산 가용 금액]을 변동 지출 카테고리별로 나누면 된다.
- 이 응답은 임시 예산안이다. 사용자가 확인하기 전까지 실제 예산안으로 적용되지 않는다.

[시스템 프롬프트: 지능형 역산(Bottom-up) 예산 수립 가이드라인]

★핵심 연산 메커니즘★:
1. 모든 모드에서 연산 시작 전에 [미할당 여유 자금]의 비율이나 금액을 먼저 정해두지 마세요.
2. AI는 아래의 '단계별 예산 차감 프로세스'를 엄격히 준수하여 순서대로 예산을 깎아 나가야 합니다.
3. 1단계와 2단계 배정이 모두 끝난 후, 【최종적으로 남은 모든 잔액】은 다른 어떤 카테고리(기타 등)에도 절대 임의 분배하지 말고, 오직 [미할당 여유 자금]으로 귀속시켜 연산을 마감하세요. 
4. 사용자의 추가 요청 사항이 있다면 연산 시 반드시 반영해야 합니다.

[단계별 예산 차감 프로세스]

1단계: 필수 생활비 선할당 (최우선순위)
- 필수 생활비(식비, 교통비)를 현재 선택된 [예산 성향 모드]의 기준에 맞춰 할당하고 [예산 가용 금액]에서 차감합니다.
- 이전 달의 소비내역이 충분하다면 이를 기반으로 삼고, 소비내역이 부족하다면 한국 직장인의 소득 평균에 맞춘 예산을 추천합니다.

2단계: 선택 지출 카테고리 유연 할당 (차순위 우선순위)
- 1단계 연산 후 '남은 잔액'을 가지고 쇼핑, 문화, 여가 등의 카테고리 예산을 모드별 규칙에 맞게 능동적으로 책정하고 차감합니다.
- 이전 달의 소비내역 또는 직장인의 평균을 고려하여 필요한 만큼만 현실적으로 할당하세요. 임의의 돈 남기기나 몰아주기를 하지 마세요.
- 만약 1단계를 거치고 남은 금액이 선택 지출을 주기에 부족하다면, 선택 지출 카테고리의 예산은 AI의 판단하에 과감히 0원으로 축소해야 합니다. 절대 총 가용 금액을 초과하는 오버 버젯 예산안을 만들 수 없습니다.

3단계: 최종 마감 및 미할당 자금 확정 (최종 프로세스)
- 1단계와 2단계 카테고리 배정을 모두 마친 후, 남은 최종 금액을 계산하여 가계부 데이터의 '미할당 여유 자금' 항목으로 최종 확정합니다. 임의의 '기타' 카테고리를 생성하여 남은 자돈을 강제로 소진하지 마십시오.

[예산 추천 유형별 프롬프트]
{mode_instruction}

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

[관련 이전 대화 기록]
{relevant_chat_histories}

[사용자 추가 요청]
{user_message}

[사용 가능한 변동 지출 카테고리]
{variable_categories}

----------------------------------------------------------------------
[출력 형식 및 작성 가이드라인]
반드시 아래 지정된 JSON 형식으로만 답해라. JSON 바깥에 설명 문장을 절대 쓰지 마라.

특히 "message" 필드는 사용자가 화면에서 읽을 최종 텍스트이므로, 아래 규칙을 반드시 준수하여 작성해라:
1. JSON이나 백슬래시(\n)가 그대로 노출되는 듯한 개발용 텍스트를 절대 포함하지 마라.
2. 기획서 UI에 맞게 직관적이고 친절한 한글 문장과 이모지(📌, ➔)를 사용하여 단락을 나누어 작성해라.
3. "message"에 그렇게 예산안을 설계한 이유를 포함하여라.
    작성 포맷 예시 (아래 서식을 참고하여 실제 데이터 수치를 대입해 작성할 것):
   [모드이름] 추천
   
   📌 최근 [카테고리명] 사용량 변화 내용 (예: 최근 식비 사용량 증가(+32%))
   📌 [카테고리명] 예산 여유 상태 (예: 교통비 예산 여유 있음(-40%))
   
   ➔ [카테고리명] 예산 [+/-금액]원
   ➔ [카테고리명] 예산 [+/-금액]원
   
   해당 내용으로 예산안을 적용할까요?

----------------------------------------------------------------------

{{
  "type": "budget_recommendation",
  "message": "위 가이드라인의 3번 포맷을 기반으로 작성된 친절하고 직관적인 설명 텍스트",
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


def create_budget_draft(
    uid: str,
    year_month: str,
    request: BudgetDraftRequest
) -> Dict[str, Any]:
    budget = load_budget(uid, year_month)

    saving = budget.get("saving", 0)
    total_budget = calculate_total_budget(uid, year_month, saving)

    cur_budget_details = budget.get("budget_details", {})
    cur_remaining_budget_details = budget.get("remaining_budget_details", {})

    ai_result = recommend_budget_with_ai(
        uid=uid,
        year_month=year_month,
        total_budget=total_budget,
        saving=saving,
        cur_budget_details=cur_budget_details,
        cur_remaining_budget_details=cur_remaining_budget_details,
        mode=request.mode,
        user_message=request.user_message
    )

    draft_budget_details = ai_result["budget_details"]

    validate_budget_details(
        budget_details=draft_budget_details,
        total_budget=total_budget
    )

    draft_remaining_budget_details = calculate_remaining_budget_details(
        uid=uid,
        year_month=year_month,
        total_budget=total_budget,
        budget_details=draft_budget_details
    )

    draft_state = calculate_budget_state(
        saving=saving,
        remaining_budget_details=draft_remaining_budget_details
    )

    draft_data = {
        "type": "budget_draft",
        "message": ai_result.get("message", "AI가 임시 예산안을 추천했습니다."),
        "year_month": year_month,
        "mode": request.mode,
        "saving": saving,
        "total_budget": total_budget,
        "budget_details": draft_budget_details,
        "remaining_budget_details": draft_remaining_budget_details,
        "state": draft_state,
        "updated_at": datetime.utcnow().isoformat(),
    }

    budget_draft_ref(uid, year_month).set(draft_data, merge=True)


    # 여기서부터는 AI가 생성한 추천 예산안과 사용자 요청을 채팅 기록에 저장.
    question_text = get_budget_mode_question(
        mode=request.mode,
        user_message=request.user_message
    )

    answer_text = f"""
    {draft_data["message"]}
    """.strip()

    save_chat_history(
        uid=uid,
        question=question_text,
        answer=answer_text,
        mode="budget"
    )

    return draft_data


def apply_budget_draft(
    uid: str,
    year_month: str
) -> Dict[str, Any]:
    draft_doc = budget_draft_ref(uid, year_month).get()

    if not draft_doc.exists:
        raise HTTPException(
            status_code=404,
            detail="적용할 임시 예산안이 없습니다."
        )

    draft = draft_doc.to_dict()

    budget_details = draft.get("budget_details", {})
    saving = draft.get("saving", 0)
    total_budget = calculate_total_budget(uid, year_month, saving)

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

    # 적용 후 draft 삭제
    budget_draft_ref(uid, year_month).delete()

    result = "\n".join([f"✅{key} 예산: {value}원" for key, value in budget_details.items()])
    answer_text = f"""
AI 추천 예산안이 실제 예산안으로 적용되었습니다.
적용된 예산안:
{result}
"""

    save_chat_history(
        uid=uid,
        question="특수 버튼 : 확인",
        answer=answer_text,
        mode="budget"
    )

    return {
        "message": answer_text,
        "budget": load_budget(uid, year_month)
    }


def cancel_budget_draft(
    uid: str,
    year_month: str
) -> Dict[str, Any]:
    draft_doc = budget_draft_ref(uid, year_month).get()

    if draft_doc.exists:
        draft = draft_doc.to_dict()
        budget_draft_ref(uid, year_month).delete()

        answer_text = f"""
임시 예산안이 취소되었습니다.
기존 예산안은 변경되지 않았습니다.

취소된 임시 예산안:
{json.dumps(draft.get("budget_details", {}), ensure_ascii=False, indent=2)}
""".strip()
    else:
        answer_text = "취소할 임시 예산안이 없었습니다. 기존 예산안은 변경되지 않았습니다."

    save_chat_history(
        uid=uid,
        question="특수 버튼 : 취소",
        answer=answer_text,
        mode="budget"
    )

    return {
        "message": "임시 예산안이 취소되었습니다.",
        "budget": load_budget(uid, year_month)
    }


"""
#AI의 예산안 제안과 동시에 예산안을 수정해버리는 문제가 있어, 이 함수는 사용하지 않음.
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
"""