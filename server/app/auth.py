from fastapi import HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth as firebase_auth

security = HTTPBearer()


def verify_firebase_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization 정보가 없습니다.")

    id_token = credentials.credentials

    try:
        decoded_token = firebase_auth.verify_id_token(id_token)
        return decoded_token["uid"]
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"유효하지 않은 Firebase 토큰입니다: {str(e)}")