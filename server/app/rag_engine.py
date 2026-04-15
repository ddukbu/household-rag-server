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


# db
db = get_firestore_client()
expenses_ref = db.collection("expenses")
chat_history_ref = db.collection("chat_history")
summaries_ref = db.collection("summaries")

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
            "parts": [
                {"text": text}
            ]
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

def load_expenses() -> List[Dict[str, Any]]:
    """
    Firestore의 'expenses' 컬렉션에 있는 모든 문서를 읽어와 
    파이썬 딕셔너리 리스트 형태로 반환합니다.
    """
    # 'expenses' 컬렉션 내의 모든 문서를 스트림 형태로 로드
    docs = expenses_ref.stream()
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

def load_chat_history() -> List[Dict[str, Any]]:
    """
    Firestore의 'chat_history' 컬렉션에 있는 모든 대화 기록을 읽어와 
    파이썬 딕셔너리 리스트 형태로 반환합니다.
    """
    # 'chat_history' 컬렉션 내의 모든 문서를 스트림 형태로 로드
    docs = chat_history_ref.stream()
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
    전체 데이터 리스트에서 RAG 관련 필드(rag_text, embedding)를 제외하고
    JSON 텍스트로 변환합니다.
    """
    clean_list = []
    
    for exp in expenses:
        # 제외하고 싶은 필드 목록
        exclude_keys = {"rag_text", "embedding"}
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
        expenses: List[Dict[str, Any]], 
        chat_histories: List[Dict[str, Any]], 
        top_k: List[int] = [3, 1]
        ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    사용자 질문과 가장 관련성이 높은 상위 k개의 문서를 반환
    """
    k_expense, k_history = top_k

    # 사용자 질문 임베딩
    query_embedding = call_embed_api(question)

    # 1. 개별 항목에 대하여
    # 유사도를 포함한 문서를 담을 리스트
    scored_docs: List[Dict[str, Any]] = []
    # 모든 문서의 유사도 계산
    for expense in expenses:
        # 검색을 위한 임베딩 데이터나 텍스트가 없는 경우 계산에서 제외
        if "embedding" not in expense:
            continue
        # 유사도 계산
        score = cosine_similarity(query_embedding, expense["embedding"])
        
        # 유사도를 추가한 문서 리스트에 추가
        scored_docs.append({
            **expense,     # date, amount, category, place, id 등이 모두 들어감
            "score": score # 유사도 점수 추가
        })
    # 유사도를 기준으로 내림차순 정렬
    scored_docs.sort(key=lambda x: x["score"], reverse=True)

    # 2. 이전 대화 기록에 대하여
    # 유사도를 포함한 문서를 담을 리스트
    scored_history: List[Dict[str, Any]] = []
    # 모든 문서의 유사도 계산
    for chat_history in chat_histories:
        # 검색을 위한 임베딩 데이터나 텍스트가 없는 경우 계산에서 제외
        if "embedding" not in chat_history:
            continue
        # 유사도 계산
        score = cosine_similarity(query_embedding, chat_history["embedding"])
        
        # 유사도를 추가한 문서 리스트에 추가
        scored_history.append({
            **chat_history,     # date, amount, category, place, id 등이 모두 들어감
            "score": score # 유사도 점수 추가
        })
    # 유사도를 기준으로 내림차순 정렬
    scored_history.sort(key=lambda x: x["score"], reverse=True)

    # 로그 출력
    print("검색된 개별 항목")
    for expense in scored_docs[:k_expense]:
        print(f"{expense["date"]}, {expense["category"]}, {expense["amount"]}, {expense["payment_method"]}, {expense["place"]}, {expense["memo"]}")
    print("검색된 이전 대화 내역")
    for i in scored_history[:k_history]:
        print(f"{i["context_text"]}")

    # 유사도가 높은 상위 top_k개의 문서 반환
    return scored_docs[:k_expense], scored_history[:k_history]

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
    
    # 한글 깨짐 방지를 위해 ensure_ascii=False 설정
    return json.dumps(clean_list, ensure_ascii=False, indent=4)

def build_prompt(question: str, docs: List[Dict[str, Any]], histories: List[Dict[str, Any]]) -> str:
    """
    데이터를 바탕으로 LLM에게 보낼 최종 프롬프트를 생성하여 반환
    """
    
    # 검색된 문서 리스트를 JSON 문자열로 변환
    context = get_expenses_json(docs)
    history = get_expenses_json(histories)

    # [STEP 2] LLM에게 전달할 시스템 프롬프트 및 컨텍스트 구성
    return f"""
너는 개인 가계부 소비 분석 도우미다.
반드시 아래 참고 문서, 이전 대화 내역만 근거로 답변해라.
문서에 없는 내용은 추측하지 말고 "문서에서 확인되지 않습니다."라고 답해라.
답변은 한국어로 작성하라.
가능하면 날짜, 금액, 카테고리, 사용처를 구체적으로 언급하라.

[질문]
{question}

[참고 문서]
{context}

[이전 대화 내역]
{history}

[답변 형식]
1. 먼저 질문에 직접 답변
2. 필요한 경우 핵심 근거 요약
3. 마지막에 "참고:" 아래에 사용한 id 나열
""".strip() # 앞뒤 불필요한 공백을 제거

def save_chat_history(question: str, answer: str):
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
    chat_history_ref.add({
        "context_text": context_text,
        "embedding": embedding,
    })
    
    # 로그 출력
    print(context_text + " 대화 내용 저장")

def answer_question(question: str) -> Dict[str, Any]:
    # 데이터 로드
    expenses = load_expenses()
    # 대화 내역 로드
    chat_history = load_chat_history()
    # 데이터 추출
    docs, histories = retrieve_relevant_docs(question, expenses, chat_history, top_k=[3, 1])
    # 프롬프트 생성
    prompt = build_prompt(question, docs, histories)
    # api 호출
    answer = call_gemini(prompt)
    # 대화 내용 저장
    save_chat_history(question, answer)

    # 로그 출력
    print(answer + " 답변 반환")

    # 답변 반환
    return {
        "answer": answer,
        "references": [doc["id"] for doc in docs]
    }

class ExpenseIn(BaseModel):
    date: str
    category: str
    amount: int
    payment_method: str
    place: str
    memo: str