import json
import os
import firebase_admin
from firebase_admin import credentials, firestore

def get_firestore_client():
    """
    Firebase Admin SDK를 초기화하고 Firestore 클라이언트 인스턴스를 반환하는 함수
    중복 초기화를 방지하며, 환경 변수에서 인증 정보를 읽어옵니다.
    """
    
    # 1. Firebase Admin SDK를 초기화 여부 확인
    # Firebase Admin SDK: 서버(백엔드)에서 관리자 권한으로 Firebase의 데이터나 사용자 계정을 제약 없이 직접 제어할 수 있게 해주는 서버 전용 도구 모음
    # firebase_admin._apps: Firebase Admin SDK가 내부적으로 사용하는 "현재 활성화된 Firebase 앱 인스턴스들의 리스트
    # firebase_admin._apps가 비어있을 경우 == Firebase Admin SDK가 초기화되지 않은 상태
    if not firebase_admin._apps:
        
        # 2. 환경 변수에서 서비스 계정 JSON 문자열 로드
        # "FIREBASE_SERVICE_ACCOUNT_JSON"이라는 이름의 환경 변수 값 로드
        raw_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "")
        
        # 3. 환경 변수 존재 여부 체크
        # 환경 변수 값(JSON 문자열)이 없을 경우 에러 발생
        if not raw_json:
            raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON 환경변수가 설정되지 않았습니다.")

        # 4. JSON 문자열을 JSON 객체(파이썬 딕셔너리)로 변환
        # SDK에서 환경 변수를 사용하기 위해 JSON 문자열 형태에서 JSON 객체 파싱
        service_account_info = json.loads(raw_json)
        
        # 5. 인증 정보 생성
        # 파싱된 JSON 객체(서비스 계정 정보)를 바탕으로 Firebase 인증 인증서를 생성
        cred = credentials.Certificate(service_account_info)
        
        # 6. Firebase Admin SDK 초기화
        # 생성된 인증서를 사용하여 앱을 시스템 전체에 초기화
        firebase_admin.initialize_app(cred)

    # 7. Firestore 클라이언트 반환
    # 이미 초기화되었거나 위에서 초기화를 마친 후
    # Firestore 데이터베이스에 접근할 수 있는 클라이언트를 반환
    return firestore.client()