import json
import os

import firebase_admin
from firebase_admin import credentials, firestore


def get_firestore_client():
    if not firebase_admin._apps:
        raw_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "")
        if not raw_json:
            raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON 환경변수가 설정되지 않았습니다.")

        service_account_info = json.loads(raw_json)
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)

    return firestore.client()