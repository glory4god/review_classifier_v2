"""
설정 파일
- 모델 파라미터
- 스팸 키워드 및 패턴
- 전처리 옵션
- API 설정
"""

import os

# 프로젝트 디렉토리
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# 학습 설정
TRAIN_CONFIG = {
    'model_name': 'monologg/koelectra-base-v3-discriminator',  # KoELECTRA 모델
    'max_length': 256,        # 최대 시퀀스 길이
    'batch_size': 16,         # 배치 크기
    'epochs': 5,              # 학습 에폭
    'learning_rate': 2e-5,    # 학습률
    'dropout_rate': 0.3,      # 드롭아웃 비율
    'validation_split': 0.1,  # 검증 데이터 비율
    'test_split': 0.1,        # 테스트 데이터 비율
    'random_seed': 42,        # 랜덤 시드
    'early_stopping': True,   # 조기 종료 활성화
    'patience': 3,            # 조기 종료 인내심
    'l2_reg': 0.01,           # L2 정규화 강도
    'cross_validation': False, # 교차 검증 활성화
    'n_folds': 5              # 교차 검증 폴드 수
}

# 모델 경로
MODEL_PATH = os.path.join(MODEL_DIR, 'koelectra_classifier.pt')

# 스팸 패턴 파일
SPAM_PATTERNS_FILE = os.path.join(MODEL_DIR, 'spam_patterns.json')

# API 설정
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': True,
    'debug': True,
    'workers': 4,
    'timeout': 60
}

# 특성 추출 설정
FEATURES = {
    # 스팸 키워드
    'spam_keywords': [
        "글자수", "100자", "100글자", "백글자", "100자이상", "백자", "글자달성", 
        "글자제한", "많이 쓰라고", "몇자를쓰라", "글자수채우기", "채우기용", 
        "리뷰쓰기귀찮", "리뷰작성귀찮", "그냥쓰는", "쓸말이없", "더이상 쓸말",
        "쓸게없", "뭐라쓸지", "아무말", "아무거나", "아무소리"
    ],
    
    # 욕설 단어
    'profanity_words': [
        "ㅅㅂ", "ㅂㅅ", "ㅈㄹ", "ㄱㅅㄲ", "ㅅㄲ", "ㅈ", "ㅆㅂ",
        "시발", "병신", "지랄", "개새끼", "미친", "존나", "꺼져", 
        "썅", "시바", "시팔", "쌍놈", "개같은", "씨발", "씨팔", "개새끼"
    ],
    
    # 이상한 문자 패턴
    'weird_chars': [
        "ㄴㅇㄹ", "ㅋㅋㅋㅋ", "ㅎㅎㅎㅎ", "ㅡㅡ", "ㅠㅠㅠ", 
        ";;;", "...", "!!!!", "???", ",,,,,", 
        "~~~~~", "_____", "-----"
    ],
    
    # 비정상 시간대
    'abnormal_time_ranges': [
        {"start": "01:00", "end": "05:00"}  # 새벽 1시 ~ 5시
    ],
    
    # 최소 한국어 비율
    'min_korean_ratio': 0.5,
    
    # 최소 리뷰 길이
    'min_length': 20,
    
    # 최대 반복 비율
    'max_repetition_ratio': 0.3,
    
    # 문장 분석 임계값
    'segment_abnormal_threshold': 0.6,
    
    # 퍼지 매칭 임계값
    'fuzzy_threshold': 0.8
}

# 앙상블 설정
ENSEMBLE_CONFIG = {
    'model_weight': 0.5,    # 모델 가중치
    'rule_weight': 0.3,     # 규칙 가중치
    'segment_weight': 0.2,  # 세그먼트 가중치
    'threshold': 0.5        # 최종 판정 임계값
}

# 스팸 텍스트 리스트 (데이터 생성용)
SPAM_TEXTS = [
    "진짤ㄴㅇ쓰기 시랃ㅇ다아아하아",
    "몇자를쓰라는겨,,ㄴㅇ란ㅇ",
    "ㄴㅇ란어쩌구저쩌구",
    "100자를 써야해서 아무말 적음",
    "글자수 채우기용 텍스트",
    "가나다라마바사아자차카타파하",
    "글자수 채우기 위한 내용입니다",
    "특별히 할말은 없지만 글자 수 맞추려고요",
    "이건 글자수 채우기일 뿐이에요",
    "글자수 채우기 위한 내용",
    "100자 이상 작성하라해서 적는중",
    "리뷰쓰기 귀찮은데 어쩔수없네요",
    "쓸말이 없어요ㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠ",
    "아무거나 적는 중이에요ㅋㅋㅋㅋㅋㅋㅋ",
    "이제 모르겠고 글자수나 채우자",
    "글자수 채워야 한다니까 아무말이나 쓰는중",
    "더이상 쓸말이 없네요 글자수 채우기",
    "더 적을 내용이 없어요",
    "동해물과 백두산이 마르고 닳도록",
    "이 정도면 충분할까요?",
    "100자 이상 쓰기 너무 어렵네요"
]
