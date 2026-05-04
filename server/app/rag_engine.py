import os
import time
from collections import defaultdict
from typing import Union, Dict, Any, List
import numpy as np
import requests
import json
from app.firebase_client import get_firestore_client
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from typing import Dict
from app.budget import load_budgets
from app.llm_client import call_gemini
#app.budget과 app.rag_engine 사이 순환 import 관계를 끊기 위해 call_gemini 함수를 따로 app.llm_client에 분리.


# db
db = get_firestore_client()

# api key, embedding model

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

EMBEDDING_MODEL = "gemini-embedding-001"
EMBED_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{EMBEDDING_MODEL}:embedContent"
)

class SummaryIn(BaseModel):
    year_month: str
    embedding: List[float] = []
    # 통계
    total_income: int = 0                           # 총소득
    total_expense: int = 0                          # 총지출
    # 수익 섹션
    fixed_income_details: Dict[str, int] = {}       # 고정 수입
    variable_income_details: Dict[str, int] = {}    # 변동 수입
    # 지출 섹션
    fixed_expense_details: Dict[str, int] = {}      # 고정 지출
    variable_expense_details: Dict[str, int] = {}   # 변동 지출
    """
    # 예산 설정 섹션 -> budget.py로 구조 이동, 요약본은 실제 수입/지출 통계만 관리
    total_budget: int = 0                           # 가용 예산 (고정 수익 - 고정 지출 - 저축)
    saving: int = 0                                 # 저축
    budget_details: Dict[str, int] = {}             # 카테고리별 예산 설정
    """


class ExpenseIn(BaseModel):
    date: str               # 날짜
    time: str               # 시간
    is_fixed_expense: bool  # 고정지출 여부
    category: str           # 카테고리
    amount: int             # 금액
    payment_method: str     # 결제방법
    place: str              # 사용처
    memo: str               # 메모

class IncomeIn(BaseModel):
    date: str               # 날짜
    time: str               # 시간
    is_fixed_income: bool   # 고정지출 여부
    category: str           # 카테고리
    amount: int             # 금액
    deposit_method: str     # 입금방법
    deposit_source: str     # 입금처
    memo: str               # 메모

def create_sentence(data: Dict[str, Any]) -> str:
    # 고정 지출 여부에 따른 텍스트 변환
    if "is_fixed_expense" in data:
        expense_type = "고정" if data['is_fixed_expense'] else "변동"
        return (
            f"{data['date']} {data['time']}에 {data['category']} 카테고리로 "
            f"{data['amount']}원을 지출하였다. "
            f"이 지출은 {expense_type} 지출이며, "
            f"결제수단은 {data['payment_method']}이고, "
            f"사용처는 {data['place']}이다. "
            f"메모: {data['memo']}."
        )
    elif "is_fixed_income" in data:
        income_type = "고정" if data['is_fixed_income'] else "변동"
    
        return (
            f"{data['date']} {data['time']}에 {data['amount']}원의 "
            f"수입이 발생하였다. "
            f"이 수입은 {income_type} 수입이며, "
            f"입금방법은 {data['deposit_method']}이고, "
            f"입금처는 {data['deposit_source']}이다. "
            f"메모: {data['memo']}."
    )
    raise ValueError(f"지원하지 않는 데이터 형식입니다: {data}")


def build_rag_record(data: Dict[str, Any]) -> Dict[str, Any]:
    rag_text = create_sentence(data)
    embedding = call_embed_api(rag_text)

    return {
        **data,
        "embedding": embedding
    }

def update_summary(summary: SummaryIn, data: Union[ExpenseIn, IncomeIn], mode: str = "add"):
    """
    수익/지출 변동분을 요약본(SummaryIn)의 고정/변동 섹션에 각각 반영
    """
    multiplier = 1 if mode == "add" else -1
    amount_change = data.amount * multiplier

    # 지출(ExpenseIn) 처리
    if isinstance(data, ExpenseIn):
        # 총 지출 업데이트
        summary.total_expense += amount_change
        
        # 고정 vs 변동 구분하여 상세 내역 업데이트
        if data.is_fixed_expense:
            target_dict = summary.fixed_expense_details
        else:
            target_dict = summary.variable_expense_details
        
        # 카테고리별 합산
        cat = data.category
        target_dict[cat] = target_dict.get(cat, 0) + amount_change
        if target_dict[cat] <= 0: del target_dict[cat]

    # 수익(IncomeIn) 처리
    elif isinstance(data, IncomeIn):
        # 총 수익 업데이트
        summary.total_income += amount_change
        
        # 고정 vs 변동 구분하여 상세 내역 업데이트
        if data.is_fixed_income:
            target_dict = summary.fixed_income_details
        else:
            target_dict = summary.variable_income_details
            
        # 카테고리별 합산
        cat = data.category
        target_dict[cat] = target_dict.get(cat, 0) + amount_change
        if target_dict[cat] <= 0: del target_dict[cat]

    """
    # 3. 가용 예산(total_budget) 및 저축(saving) 자동 계산 (필요 시)
    # 예: 가용 예산 = 고정 수익 - 고정 지출 - 저축
    fixed_inc = sum(summary.fixed_income_details.values())
    fixed_exp = sum(summary.fixed_expense_details.values())
    summary.total_budget = fixed_inc - fixed_exp - summary.saving
    """

    return summary

def process_expense_change(uid: str, data: Union[ExpenseIn, IncomeIn], mode: str = "add"):
    """
    DB 요약본 업데이트 처리
    """
    year_month = data.date[:7]
    doc_ref = db.collection("users").document(uid).collection("summaries").document(year_month)
    doc = doc_ref.get()

    if doc.exists:
        summary = SummaryIn(**doc.to_dict())
    else:
        # 텍스트 생성 후 즉시 임베딩화하여 저장
        text = f"{year_month} 지출 요약 및 통계 내역"
        embedding = call_embed_api(text)
        summary = SummaryIn(
            year_month=year_month,
            embedding=embedding
        )

    updated_summary = update_summary(summary, data, mode)
    doc_ref.set(updated_summary.dict(), merge=True)

def call_embed_api(text: str) -> List[float]:
    """
    전달받은 텍스트(질문 또는 지출 내역)를 Gemini 모델에 보내 
    의미적 특징이 담긴 숫자 리스트(임베딩)로 변환
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    payload = {
        "content": {
            "parts": [{"text": text}]
        }
    }

    response = requests.post(
        EMBED_URL,
        headers=headers,
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    data = response.json()
    return data["embedding"]["values"]

#기존 call_gemini를 llm_client.py로 이동.

def transform_query(question: str) -> str:
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    
    prompt = f"""
현재 날짜는 {current_date}이다. 
사용자의 질문에서 날짜 관련 표현(지난달, 이번달, 3월 등)이 있다면 이를 YYYY-MM 형식 또는 YYYY-MM-DD 형식으로 변환하여 질문에 포함시켜라.
만약 날짜 언급이 없다면 질문을 그대로 반환해라.

질문: {question}
변환된 질문:"""
    transformed = call_gemini(prompt)

    return transformed if transformed else question

def load_monthly_summaries(uid: str) -> List[Dict[str, Any]]:
    """
    Firestore의 'summaries' 컬렉션에서 모든 월별 요약본을 로드합니다.
    """
    # 'summaries' 컬렉션 내의 모든 문서를 스트림 형태로 로드
    docs = db.collection("users").document(uid).collection("summaries").stream()
    # 결과 데이터를 담을 빈 리스트 초기화
    summaries = []
    # 가져온 문서들을 하나씩 순회하며 처리
    # 문서를 id를 추가한 딕셔너리로 변환하여 리스트에 추가
    for doc in docs:
        # 문서를 딕셔너리로 변환
        data = doc.to_dict()
        # id를 추가한 딕셔너리 리스트에 추가
        summaries.append({
            "id": doc.id,   # Firestore가 자동으로 생성한 문서 고유 ID
            **data
        })
    return summaries


def load_expenses(uid: str) -> List[Dict[str, Any]]:
    """
    Firestore의 'expenses' 컬렉션에 있는 모든 문서를 읽어와 
    파이썬 딕셔너리 리스트 형태로 반환합니다.
    """
    # 'expenses' 컬렉션 내의 모든 문서를 스트림 형태로 로드
    docs = db.collection("users").document(uid).collection("expenses").stream()
    # 결과 데이터를 담을 빈 리스트 초기화
    expenses = []
    # 가져온 문서들을 하나씩 순회하며 처리
    # 문서를 id를 추가한 딕셔너리로 변환하여 리스트에 추가
    for doc in docs:
        # 문서를 딕셔너리로 변환
        data = doc.to_dict()
        # id를 추가한 딕셔너리 리스트에 추가
        expenses.append({
            "id": doc.id,  # Firestore가 자동으로 생성한 문서 고유 ID
            **data         
        })
    # 모든 문서가 담긴 리스트 반환
    return expenses

def load_chat_history(uid: str) -> List[Dict[str, Any]]:
    """
    Firestore의 'chat_history' 컬렉션에 있는 모든 대화 기록을 읽어와 
    파이썬 딕셔너리 리스트 형태로 반환합니다.
    """
    # 'chat_history' 컬렉션 내의 모든 문서를 스트림 형태로 로드
    docs = db.collection("users").document(uid).collection("chat_history").stream()
    # 결과 데이터를 담을 빈 리스트 초기화
    chat_history = []
    # 가져온 문서들을 하나씩 순회하며 처리
    # 문서를 id를 추가한 딕셔너리로 변환하여 리스트에 추가
    for doc in docs:
        # 문서를 딕셔너리로 변환
        data = doc.to_dict()
        # id를 추가한 딕셔너리 리스트에 추가
        chat_history.append({
            "id": doc.id,
            **data         
        })
    return chat_history

def get_expenses_json(expenses: List[Dict[str, Any]]) -> str:
    """
    전체 데이터 리스트에서 RAG 관련 필드(score, embedding)를 제외하고
    JSON 텍스트로 변환합니다.
    """
    clean_list = []
    
    for exp in expenses:
        # 제외하고 싶은 필드 목록
        exclude_keys = {"score", "embedding"}
        # 제외 필드를 제외한 나머지 데이터만 추출
        clean_data = {k: v for k, v in exp.items() if k not in exclude_keys}
        clean_list.append(clean_data)

    # 로그 출력
    print("json 형태의 텍스트 생성")
    print(json.dumps(clean_list, ensure_ascii=False, indent=4))
    
    # 한글 깨짐 방지를 위해 ensure_ascii=False 설정
    return json.dumps(clean_list, ensure_ascii=False, indent=4)
  
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    벡터 a와 b 사이의 각도 코사인 값을 계산하여 유사도를 측정합니다.
    결과값은 1.0에 가까울수록 매우 유사하고, 0.0에 가까울수록 관련이 없음을 의미합니다.
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)

def retrieve_relevant_docs(
        question: str, 
        summaries: List[Dict[str, Any]],
        expenses: List[Dict[str, Any]], 
        chat_histories: List[Dict[str, Any]], 
        ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    사용자 질문과 가장 관련성이 높은 상위 k개의 문서를 반환
    """

    # 사용자 질문 임베딩
    query_embedding = call_embed_api(question)

    def filter_docs(
            docs: List[Dict[str, Any]], 
            threshold: float, 
            min_k: int, 
            max_k: int
            )-> List[Dict[str, Any]]:
        # 유사도를 포함한 문서를 담을 리스트
        scored: List[Dict[str, Any]] = []
        # 모든 문서의 유사도 계산
        for doc in docs:
            # 검색을 위한 임베딩 데이터가 없는 경우 계산에서 제외
            if "embedding" not in doc or not doc["embedding"]: 
                continue
            # 유사도 계산
            score = cosine_similarity(query_embedding, doc["embedding"])
            # 유사도를 추가한 문서 리스트에 추가
            scored.append({
                **doc, 
                "score": score
            })
        # 유사도 순 정렬
        scored.sort(key=lambda x: x["score"], reverse=True)
        
        # 임계치를 넘는 문서들 필터링
        passed = [d for d in scored if d["score"] >= threshold]
        
        # 최소 개수(min_k) 보장: 임계치 못 넘어도 상위권은 가져옴
        if len(passed) < min_k:
            passed = scored[:min_k]
        
        # 최대 개수(max_k) 제한
        return passed[:max_k]

    # 요약본: 날짜 매칭이 중요하므로 임계치를 높게 잡되, 최소 1개는 보장
    relevant_summaries = filter_docs(summaries, threshold=0.8, min_k=0, max_k=3)
    
    # 개별 항목: 상세 내역은 관련 있는 것 위주로 최대 15개
    relevant_expenses = filter_docs(expenses, threshold=0.7, min_k=0, max_k=30)
    
    # 대화 내역: 문맥 파악용으로 최대 3개
    relevant_histories = filter_docs(chat_histories, threshold=0.75, min_k=0, max_k=3)

    return relevant_summaries, relevant_expenses, relevant_histories

def build_prompt(
        question: str, 
        summaries: List[Dict[str, Any]], 
        budgets: List[Dict[str, Any]],
        docs: List[Dict[str, Any]], 
        histories: List[Dict[str, Any]]
        ) -> str:
    """
    데이터를 바탕으로 LLM에게 보낼 최종 프롬프트를 생성하여 반환
    """
    
    # 검색된 문서 리스트를 JSON 문자열로 변환
    summary_context = get_expenses_json(summaries)
    budget_context = get_expenses_json(budgets)
    expense_context = get_expenses_json(docs)
    history_context = get_expenses_json(histories)

    # LLM에게 전달할 시스템 프롬프트 및 컨텍스트 구성
    #return f"""
    """
너는 개인 가계부 소비 분석 도우미다.
제공된 [월별 요약]을 통해 전체적인 흐름을 파악하고, [상세 지출 내역]을 참고하여 답변해라.
반드시 아래 참고 데이터만 근거로 답변하고, 없는 내용은 "확인되지 않습니다"라고 답해라.

[질문]
{question}

[참고 내역]
{summary_context}
{expense_context}

[이전 대화 내역]
{history_context}

[답변 가이드]
1. 질문한 달의 전체 지출 현황(카테고리/결제방식 별)을 요약본을 근거로 먼저 설명해줘.
2. 구체적인 내역을 물었다면 상세 지출 내역의 장소와 금액을 언급해줘.
3. 한국어로 구체적이고 친절하게 답변해라.
"""
#""".strip()
    return f"""
너는 사용자의 자산 관리를 돕는 [스마트 가계부 분석가]이다. 아래 지침에 따라 답변해라.

### [데이터 활용 가이드]
1. **통계의 출처:** - 카테고리/결제 수단 총액은 **[월별 요약]**을 최우선 근거로 답변해라.
   - 요약본에 없는 구체적인 통계(예: 특정 식당 방문 횟수, 특정 시간대 지출 등)는 **[상세 지출 내역]**을 바탕으로 직접 계산하되, "검색된 내역을 바탕으로 확인한 결과~"와 같은 표현을 사용하여 데이터가 일부일 수 있음을 암시해라.
2. **상세 내역의 유연성:** 상세 내역을 단순히 나열하지 말고, 질문의 맥락에 맞게 분석하여 답변에 녹여내라. (예: "주로 점심시간에 편의점 지출이 많으시네요")
3. **인사이트 제공 (중요):** 데이터 분석 후에는 반드시 사용자의 소비 습관에 도움이 될 만한 **팁이나 조언**을 한 문장 이상 포함해라.
4. **예산안 추천:** 
    - [예산안]은 사용자가 설정했거나 AI가 추천한 계획 데이터이다.
    - 예산 초과/잔여 예산을 판단할 때는 [예산안]의 budget_details와 [월별 요약]의 variable_expense_details를 비교해라.

### [참고 데이터]
* [월별 요약]: {summary_context}
* [예산안]: {budget_context}
* [상세 지출 내역]: {expense_context}
* [이전 대화]: {history_context}

### [질문]
"{question}"

답변 (핵심 위주로 친절하게):
""".strip()

def save_chat_history(uid: str, question: str, answer: str):
    """
    질문, 답변, 그리고 '대화 시점'을 하나의 문장으로 묶어 임베딩합니다.
    """
    # 대화 시점 계산
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst)
    time_str = now.strftime('%Y년 %m월 %d일 %H시 %M분')
    
    # 검색을 위해 대화 시점과 질문, 답변을 합친 텍스트를 임베딩
    context_text = f"대화 시점: {time_str}\n질문: {question}\n답변: {answer}"
    embedding = call_embed_api(context_text)
    
    # Firestore 저장
    db.collection("users").document(uid).collection("chat_history").add({
        "context_text": context_text,
        "embedding": embedding,
    })

def answer_question(uid: str, question: str) -> Dict[str, Any]:
    # 사용자 질문의 날짜 관련 표현을 YYYY-MM or YYYY-MM-DD 형식으로 변환
    transformed_query = transform_query(question)
    time.sleep(1.0)
    # 로그 출력
    print("변환된 질문: " + transformed_query)

    # 월별 요약본 로드
    summaries = load_monthly_summaries(uid)
    # 설정한 예산안 로드
    budgets = load_budgets(uid)
    # 데이터 로드
    expenses = load_expenses(uid)
    # 대화 내역 로드
    chat_histories = load_chat_history(uid)

    # 시간 측정
    start = time.time()
    # 데이터 추출
    summaries, docs, histories = retrieve_relevant_docs(transformed_query, summaries, expenses, chat_histories)
    # 시간측정
    retrieval_elapsed = time.time() - start

    # 프롬프트 생성
    prompt = build_prompt(question, summaries, budgets, docs, histories)

    # 시간 측정
    gen_start = time.time()
    # api 호출
    answer = call_gemini(prompt)
    # 시간 측정
    generation_elapsed = time.time() - gen_start
    total_elapsed = retrieval_elapsed + generation_elapsed
    # 로그 출력
    print("답변")
    print(answer)

    # 대화 내용 저장
    save_chat_history(uid, question, answer)

    # 답변 반환
    return {
        "answer": answer,
        "references": [doc["id"] for doc in docs],
        "retrieval_seconds": round(retrieval_elapsed, 3),
        "generation_seconds": round(generation_elapsed, 3),
        "total_seconds": round(total_elapsed, 3),
    }