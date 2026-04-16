import os
import time
from collections import defaultdict
from typing import Any, Dict, List
import numpy as np
import requests
import json
from app.firebase_client import get_firestore_client
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from typing import Dict


# db
db = get_firestore_client()

# api model
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GENERATION_MODEL = "gemini-2.5-flash"
GENERATE_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GENERATION_MODEL}:generateContent"
)

EMBEDDING_MODEL = "gemini-embedding-001"
EMBED_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{EMBEDDING_MODEL}:embedContent"
)

# [Model] 월별 지출 통계 구조
class MonthlySummary(BaseModel):
    year_month: str
    category_totals: Dict[str, int] = {}
    payment_method_totals: Dict[str, int] = {}
    embedding: List[float] = []

class ExpenseIn(BaseModel):
    date: str
    category: str
    amount: int
    payment_method: str
    place: str
    memo: str

def expense_to_sentence(expense: Dict[str, Any]) -> str:
    return (
        f"{expense['date']}에 {expense['category']} 카테고리로 "
        f"{expense['amount']}원을 지출하였다. "
        f"결제수단은 {expense['payment_method']}이며, "
        f"사용처는 {expense['place']}이다. "
        f"메모: {expense['memo']}."
    )

def build_expense_rag_record(expense: Dict[str, Any]) -> Dict[str, Any]:
    rag_text = expense_to_sentence(expense)
    embedding = call_embed_api(rag_text)

    return {
        **expense,
        "embedding": embedding
    }

def update_monthly_summary(summary: MonthlySummary, expense: ExpenseIn, mode: str = "add"):
    """
    지출 변동분을 요약본에 반영
    """
    multiplier = 1 if mode == "add" else -1
    change = expense.amount * multiplier

    # 카테고리 업데이트
    cat = expense.category
    summary.category_totals[cat] = summary.category_totals.get(cat, 0) + change

    # 결제 수단 업데이트
    pay = expense.payment_method
    summary.payment_method_totals[pay] = summary.payment_method_totals.get(pay, 0) + change
    
    # 0원 항목 정리
    if summary.category_totals.get(cat) == 0:
        del summary.category_totals[cat]
    
    return summary

def process_expense_change(uid: str, expense: ExpenseIn, mode: str = "add"):
    """
    DB 요약본 업데이트 처리
    """
    year_month = expense.date[:7]
    doc_ref = db.collection("users").document(uid).collection("summaries").document(year_month)
    doc = doc_ref.get()

    if doc.exists:
        summary = MonthlySummary(**doc.to_dict())
    else:
        # 텍스트 생성 후 즉시 임베딩화하여 저장
        text = f"{year_month} 지출 요약 및 통계 내역"
        embedding = call_embed_api(text)
        summary = MonthlySummary(
            year_month=year_month,
            embedding=embedding
        )

    updated_summary = update_monthly_summary(summary, expense, mode)
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

def call_gemini(prompt: str) -> str:
    """
    작성된 프롬프트를 Gemini 모델에 전달하고 AI의 답변 반환
    네트워크 오류 발생 시 최대 3번까지 재시도
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

    time.sleep(3.0)
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    max_retries = 3
    delay = 2

    for attempt in range(max_retries):
        response = requests.post(
            GENERATE_URL,
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return "응답을 생성하지 못했습니다."

            parts = candidates[0].get("content", {}).get("parts", [])
            texts = [part.get("text", "") for part in parts if "text" in part]
            return "\n".join(texts).strip()

        if response.status_code in (429, 500, 503) and attempt < max_retries - 1:
            time.sleep(delay)
            delay *= 2
            continue

        response.raise_for_status()

    return "응답을 생성하지 못했습니다."

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

    # 로그 출력
    print("변환된 질문 " + transformed)

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

    # 로그 출력
    print("개별 항목 불러오기")
    for expense in expenses:
        print(f"{expense["date"]}, {expense["category"]}, {expense["amount"]}, {expense["payment_method"]}, {expense["place"]}, {expense["memo"]}")

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

    # 로그 출력
    print("이전 대화 내역 불러오기")
    for chat_history_item in chat_history:
        print(chat_history_item)
        
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
    print(json.dumps(clean_list, ensure_ascii=False, indent=4) + " json 형태의 텍스트 생성")
    
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
    relevant_summaries = filter_docs(summaries, threshold=0.8, min_k=1, max_k=3)
    
    # 개별 항목: 상세 내역은 관련 있는 것 위주로 최대 15개
    relevant_expenses = filter_docs(expenses, threshold=0.7, min_k=0, max_k=15)
    
    # 대화 내역: 문맥 파악용으로 최대 3개
    relevant_histories = filter_docs(chat_histories, threshold=0.75, min_k=0, max_k=3)

    return relevant_summaries, relevant_expenses, relevant_histories

def build_prompt(
        question: str, 
        summaries: List[Dict[str, Any]], 
        docs: List[Dict[str, Any]], 
        histories: List[Dict[str, Any]]
        ) -> str:
    """
    데이터를 바탕으로 LLM에게 보낼 최종 프롬프트를 생성하여 반환
    """
    
    # 검색된 문서 리스트를 JSON 문자열로 변환
    summary_context = get_expenses_json(summaries)
    expense_context = get_expenses_json(docs)
    history_context = get_expenses_json(histories)

    # LLM에게 전달할 시스템 프롬프트 및 컨텍스트 구성
    return f"""
너는 개인 가계부 소비 분석 도우미다.
제공된 [월별 요약]을 통해 전체적인 흐름을 파악하고, [상세 지출 내역]을 참고하여 답변해라.
반드시 아래 참고 데이터만 근거로 답변하고, 없는 내용은 "확인되지 않습니다"라고 답해라.

[질문]
{question}

[월별 요약 통계]
{summary_context}

[상세 지출 내역]
{expense_context}

[이전 대화 내역]
{history_context}

[답변 가이드]
1. 질문한 달의 전체 지출 현황(카테고리/결제방식 별)을 요약본을 근거로 먼저 설명해줘.
2. 구체적인 내역을 물었다면 상세 지출 내역의 장소와 금액을 언급해줘.
3. 한국어로 구체적이고 친절하게 답변해라.
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
    
    # 로그 출력
    print(context_text + " 대화 내용 저장")

def answer_question(uid: str, question: str) -> Dict[str, Any]:
    # 사용자 질문의 날짜 관련 표현을 YYYY-MM or YYYY-MM-DD 형식으로 변환
    transformed_query = transform_query(question)
    # 월별 요약본 로드
    summaries = load_monthly_summaries(uid)
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
    prompt = build_prompt(question, summaries, docs, histories)
    # 시간 측정
    gen_start = time.time()
    # api 호출
    answer = call_gemini(prompt)
    # 시간 측정
    generation_elapsed = time.time() - gen_start
    total_elapsed = retrieval_elapsed + generation_elapsed
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