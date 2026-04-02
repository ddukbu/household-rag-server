import os
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import requests

from app.firebase_client import get_firestore_client

db = get_firestore_client()
expenses_ref = db.collection("expenses")

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


def month_of(date_str: str) -> str:
    return date_str[:7]


def build_monthly_summary(expenses: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    grouped = defaultdict(list)
    for expense in expenses:
        grouped[month_of(expense["date"])].append(expense)

    summaries = []
    months = sorted(grouped.keys())
    previous_total = None

    for month in months:
        items = grouped[month]
        total = sum(int(x["amount"]) for x in items)

        category_totals = defaultdict(int)
        for x in items:
            category_totals[x["category"]] += int(x["amount"])

        top_category = max(category_totals.items(), key=lambda x: x[1])[0]
        top_amount = category_totals[top_category]

        if previous_total is None:
            diff_text = "이전 달 데이터가 없어 증감 비교는 불가능하다."
        else:
            diff = total - previous_total
            if diff > 0:
                diff_text = f"전월 대비 총지출이 {diff}원 증가하였다."
            elif diff < 0:
                diff_text = f"전월 대비 총지출이 {abs(diff)}원 감소하였다."
            else:
                diff_text = "전월 대비 총지출 변화가 없다."

        summaries.append({
            "ref": f"summary:{month}",
            "text": (
                f"{month} 소비 요약이다. "
                f"총지출은 {total}원이다. "
                f"가장 큰 지출 카테고리는 {top_category}이며 해당 지출은 {top_amount}원이다. "
                f"{diff_text}"
            )
        })
        previous_total = total

    return summaries


def call_embed_api(text: str) -> List[float]:
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


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def build_expense_rag_record(expense: Dict[str, Any]) -> Dict[str, Any]:
    rag_text = expense_to_sentence(expense)
    embedding = call_embed_api(rag_text)

    return {
        **expense,
        "rag_text": rag_text,
        "embedding": embedding
    }


def load_expenses() -> List[Dict[str, Any]]:
    docs = expenses_ref.stream()
    expenses = []
    for doc in docs:
        data = doc.to_dict()
        expenses.append({
            "id": doc.id,
            **data
        })
    return expenses


def retrieve_relevant_docs(question: str, expenses: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, str]]:
    query_embedding = call_embed_api(question)

    scored_docs: List[Dict[str, Any]] = []

    for expense in expenses:
        if "embedding" not in expense or "rag_text" not in expense:
            continue

        score = cosine_similarity(query_embedding, expense["embedding"])
        scored_docs.append({
            "ref": f"expense:{expense['id']}",
            "text": expense["rag_text"],
            "score": score
        })

    summaries = build_monthly_summary(expenses)
    for summary in summaries:
        summary_embedding = call_embed_api(summary["text"])
        score = cosine_similarity(query_embedding, summary_embedding)
        scored_docs.append({
            "ref": summary["ref"],
            "text": summary["text"],
            "score": score
        })

    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    return scored_docs[:top_k]


def build_prompt(question: str, docs: List[Dict[str, str]]) -> str:
    context = "\n\n".join(
        [f"[{doc['ref']}]\n{doc['text']}" for doc in docs]
    )

    return f"""
너는 개인 가계부 소비 분석 도우미다.
반드시 아래 참고 문서만 근거로 답변해라.
문서에 없는 내용은 추측하지 말고 "문서에서 확인되지 않습니다."라고 답해라.
답변은 한국어로 작성하라.
가능하면 날짜, 금액, 카테고리, 사용처를 구체적으로 언급하라.

[질문]
{question}

[참고 문서]
{context}

[답변 형식]
1. 먼저 질문에 직접 답변
2. 필요한 경우 핵심 근거 요약
3. 마지막에 "참고:" 아래에 사용한 ref 나열
""".strip()


def call_gemini(prompt: str) -> str:
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


def answer_question(question: str) -> Dict[str, Any]:
    expenses = load_expenses()
    docs = retrieve_relevant_docs(question, expenses, top_k=3)
    prompt = build_prompt(question, docs)
    answer = call_gemini(prompt)

    return {
        "answer": answer,
        "references": [doc["ref"] for doc in docs]
    }