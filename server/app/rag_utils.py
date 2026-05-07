# app/rag_utils.py

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
import numpy as np

from app.firebase_client import get_firestore_client
from app.llm_client import call_embed_api

db = get_firestore_client()


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


def save_chat_history(uid: str, question: str, answer: str, mode: str = "general"):
    """
    모드, 질문, 답변, 그리고 '대화 시점'을 하나의 문장으로 묶어 임베딩합니다.
    앱에서 바로 채팅창에 표시하기 위한 자연어 질문, 답변 또한 저장합니다.
    """
    # 대화 시점 계산
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst)
    time_str = now.strftime('%Y년 %m월 %d일 %H시 %M분')
    
    # 검색을 위해 대화 모드와 대화 시점, 질문, 답변을 합친 텍스트를 임베딩
    context_text = f"대화 모드: {mode}\n대화 시점: {time_str}\n질문: {question}\n답변: {answer}"
    embedding = call_embed_api(context_text)
    
    # Firestore 저장
    db.collection("users").document(uid).collection("chat_history").add({
        "mode": mode,
        "question": question,
        "answer": answer,
        "context_text": context_text,
        "embedding": embedding,
        "created_at": now.isoformat(),
    })


#예산안 모드에서 사용자 요청과 유사한 채팅 기록을 RAG방식으로 찾아서 AI에게 프롬프트로 전달하기 위한 함수. rag_engine.py에 있는 retrieve_relevant_docs함수와 유사하게 작동하나, 채팅 기록만 다룸.
def retrieve_relevant_chat_history(
    query: str,
    histories: List[Dict[str, Any]],
    threshold: float = 0.75,
    min_k: int = 0,
    max_k: int = 3
) -> List[Dict[str, Any]]:
    query_embedding = call_embed_api(query)

    scored = []

    for history in histories:
        if "embedding" not in history or not history["embedding"]:
            continue

        score = cosine_similarity(query_embedding, history["embedding"])

        scored.append({
            **history,
            "score": score
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    passed = [item for item in scored if item["score"] >= threshold]

    if len(passed) < min_k:
        passed = scored[:min_k]

    return passed[:max_k]