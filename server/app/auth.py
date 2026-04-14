from fastapi import Header, HTTPException
from firebase_admin import auth as firebase_auth


def verify_firebase_token(authorization: str | None = Header(default=None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization 헤더가 없습니다.")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer 토큰 형식이 아닙니다.")

    id_token = authorization.split(" ", 1)[1].strip()
    if not id_token:
        raise HTTPException(status_code=401, detail="토큰이 비어 있습니다.")

    try:
        decoded_token = firebase_auth.verify_id_token(id_token)
        uid = decoded_token["uid"]
        return uid
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"유효하지 않은 Firebase 토큰입니다: {str(e)}")