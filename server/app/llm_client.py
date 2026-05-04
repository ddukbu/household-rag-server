import os
import time
import requests

#api model

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GENERATION_MODEL = "gemini-2.5-flash"
GENERATE_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GENERATION_MODEL}:generateContent"
)


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