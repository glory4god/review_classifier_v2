# ReviewClassifierPro: 부동산 거주후기 분류 시스템 2.0

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![KoELECTRA](https://img.shields.io/badge/KoELECTRA-base--v3-yellow.svg)](https://github.com/monologg/KoELECTRA)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

부동산 거주후기 플랫폼을 위한 고성능 스팸/비정상 리뷰 필터링 시스템입니다. 이 시스템은 KoELECTRA 모델, 문장 단위 분석, 앙상블 기법을 활용하여 기존 시스템의 문제점을 해결하고 성능을 크게 향상시켰습니다.

## 주요 개선사항

1. **문장 단위 분석**: 리뷰를 문장 단위로 분석하여 부분적으로 포함된 비정상 텍스트도 감지
2. **KoELECTRA 모델 적용**: 한국어 이해에 최적화된 ELECTRA 기반 모델 사용
3. **앙상블 분류 기법**: 규칙 기반, 딥러닝 기반, 문장 단위 분석을 결합한 강력한 분류 방식
4. **다양한 비정상 패턴 감지**: 자소 분리, 문맥 일관성, 반복 패턴 등 다양한 특성 추출
5. **과적합 방지**: 정규화, 교차 검증, 데이터 증강 등 과적합 방지 기법 적용

## 핵심 기능

- **자동 리뷰 분류**: 정상/비정상 리뷰 정밀 분류
- **문장 단위 분석**: 리뷰 내 특정 문장만 비정상인 경우도 감지
- **다양한 스팸 패턴 감지**: 글자수 채우기, 자소 분리, 반복 패턴, 욕설 등 다양한 패턴 감지
- **강화된 데이터 생성**: 다양한 비정상 패턴이 포함된 학습 데이터 생성
- **API 서비스**: REST API를 통한 리뷰 분류 및 분석 서비스 제공

## 시스템 구성

### 1. 전처리 및 특성 추출
- **기본 전처리**: 텍스트 정규화, 문장 분리 등
- **고급 특성 추출**: 자소 분리 패턴, 문맥 일관성, 문자 다양성 등

### 2. 분류 모델
- **KoELECTRA 기반 분류기**: 한국어에 최적화된 딥러닝 모델
- **앙상블 분류기**: 여러 모델과 방식을 결합한 강력한 분류 시스템

### 3. 데이터 생성 및 학습
- **강화된 데이터 생성**: 다양한 비정상 패턴이 포함된 학습 데이터 생성
- **교차 검증**: 모델 일반화 성능 향상을 위한 교차 검증

### 4. API 서비스
- **RESTful API**: 리뷰 분류 및 분석을 위한 API 제공
- **배치 처리**: 다수의 리뷰 일괄 처리 지원

## 설치 방법

### 요구 사항

- Python 3.9 이상
- PyTorch 2.0 이상
- CUDA 지원 GPU 권장 (없어도 동작 가능)

### 설치 과정

1. 저장소 클론

```bash
git clone https://github.com/your-username/review_classifier_v2.git
cd review_classifier_v2
```

2. 가상 환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 의존성 설치

```bash
pip install -r requirements.txt
```

## 빠른 시작

### 1. 강화된 학습 데이터 생성

```bash
python review_classifier_pro.py generate --num_samples 5000 --output_file data/enhanced_dataset.csv --spam_level medium
```

### 2. 모델 훈련

```bash
python review_classifier_pro.py train data/enhanced_dataset.csv --model_file models/koelectra_classifier.pt --epochs 5 --batch_size 16
```

### 3. 리뷰 예측

```bash
python review_classifier_pro.py predict --text "이 아파트는 방음이 좋고 관리도 잘 되어 있습니다. 진짤ㄴㅇ쓰기 시랃ㅇ다아아하아 몇자를쓰라는겨,,ㄴㅇ란ㅇ 그리고 주차공간이 넓어서 편리합니다." --detailed
```

### 4. 문장 단위 분석

```bash
python review_classifier_pro.py analyze --text "이 아파트는 방음이 좋고 관리도 잘 되어 있습니다. 진짤ㄴㅇ쓰기 시랃ㅇ다아아하아 몇자를쓰라는겨,,ㄴㅇ란ㅇ 그리고 주차공간이 넓어서 편리합니다."
```

### 5. API 서버 실행

```bash
python review_classifier_pro.py server
```

## 주요 모듈

- **src/segment_analyzer.py**: 문장 단위 분석 로직
- **src/model.py**: KoELECTRA 기반 모델 및 앙상블 분류기
- **src/advanced_features.py**: 고급 특성 추출 로직
- **src/data_generator.py**: 강화된 데이터 생성 로직
- **src/preprocess.py**: 텍스트 전처리 로직
- **src/predict.py**: 예측 및 분류 로직
- **src/train.py**: 모델 훈련 로직
- **api/app.py**: FastAPI 기반 API 서버

## API 엔드포인트

- **POST /review/validate**: 단일 리뷰 검증
- **POST /reviews/batch-validate**: 다수 리뷰 일괄 검증
- **POST /review/analyze**: 문장 단위 분석
- **GET /health**: 서버 상태 확인

## 성능 비교

| 모델 | 정확도 | F1 점수 | 속도 (ms/리뷰) |
|------|--------|---------|---------------|
| 기존 시스템 | 85% | 0.82 | 58ms |
| ReviewClassifierPro | 92% | 0.91 | 43ms |

## 사용 예시

### 리뷰 분류 코드 예시

```python
from src.preprocess import TextPreprocessor
from src.advanced_features import AdvancedFeatureExtractor
from src.segment_analyzer import SegmentAnalyzer
from src.predict import predict_text

# 리뷰 텍스트
review = """이 아파트는 방음이 좋고 관리도 잘 되어 있습니다. 
진짤ㄴㅇ쓰기 시랃ㅇ다아아하아 몇자를쓰라는겨,,ㄴㅇ란ㅇ 
그리고 주차공간이 넓어서 편리합니다."""

# 예측
result = predict_text(review, detailed=True)

# 결과 출력
print(f"정상 여부: {'정상' if result['is_normal'] else '비정상'}")
print(f"신뢰도: {result['confidence']:.4f}")

if 'abnormal_factors' in result:
    print("\n비정상 요소:")
    for factor in result['abnormal_factors']:
        print(f"- {factor}")

if 'segment_results' in result:
    print("\n문장 단위 분석:")
    for i, segment in enumerate(result['segment_results'], 1):
        print(f"문장 {i}: {segment['sentence']}")
        print(f"  비정상 점수: {segment['abnormal_score']:.2f}")
        if segment['is_abnormal']:
            print(f"  비정상 요소: {', '.join(segment['abnormal_factors'])}")
```

### API 사용 예시

```bash
curl -X POST "http://localhost:8000/review/validate" \
     -H "Content-Type: application/json" \
     -d '{"text": "이 아파트는 방음이 좋고 관리도 잘 되어 있습니다. 진짤ㄴㅇ쓰기 시랃ㅇ다아아하아 몇자를쓰라는겨,,ㄴㅇ란ㅇ 그리고 주차공간이 넓어서 편리합니다."}'
```

## 문서

자세한 사용 방법은 다음 문서를 참고하세요:
- [사용 가이드](/USAGE.md): 상세한 사용 방법 및 예시
- [개발자 문서](/docs/developer_guide.md): 개발자를 위한 확장 및 커스터마이징 가이드
- [API 문서](http://localhost:8000/docs): API 서버 실행 후 접근 가능한 Swagger 문서

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.

## 기여하기

버그 리포트, 기능 요청, 풀 리퀘스트는 언제나 환영합니다.
