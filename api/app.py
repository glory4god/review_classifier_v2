"""
API 서버 모듈
- FastAPI 기반 REST API 서비스
- 리뷰 분류, 문장 단위 분석 등 API 제공
"""

import sys
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Union

from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import torch
import numpy as np

# 상위 디렉토리 추가 (상대 임포트를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    ReviewRequest, ReviewResponse, ErrorResponse, HealthResponse,
    BatchReviewRequest, BatchReviewResponse, AnalysisResponse
)
from src.preprocess import TextPreprocessor
from src.advanced_features import AdvancedFeatureExtractor
from src.model import ReviewClassifierPro, EnsembleClassifier
from src.segment_analyzer import SegmentAnalyzer
from config import MODEL_PATH, API_CONFIG

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger("review-classifier-api")

# 시작 시간
START_TIME = time.time()

# FastAPI 앱 초기화
app = FastAPI(
    title="ReviewClassifierPro API",
    description="부동산 거주후기 리뷰 분류 시스템 2.0 API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 특정 출처만 허용하도록 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 모델 및 처리기 초기화
preprocessor = TextPreprocessor()
feature_extractor = AdvancedFeatureExtractor()
segment_analyzer = None
classifier = None
ensemble = None

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_segment_analyzer():
    """세그먼트 분석기 가져오기 (지연 초기화)"""
    global segment_analyzer
    if segment_analyzer is None:
        logger.info("세그먼트 분석기 초기화 중...")
        segment_analyzer = SegmentAnalyzer(
            preprocessor=preprocessor,
            feature_extractor=feature_extractor
        )
    return segment_analyzer


def get_classifier():
    """분류기 가져오기 (지연 초기화)"""
    global classifier, ensemble

    if classifier is None:
        try:
            logger.info("분류기 초기화 중...")

            # 모델 디렉토리 확인
            model_dir = os.path.dirname(MODEL_PATH)
            os.makedirs(model_dir, exist_ok=True)

            # 실제 ModelClassifier 객체 생성 (딕셔너리 대신)
            from src.predict import ModelClassifier, RuleBasedClassifier

            try:
                # 모델 파일 존재 확인 및 분류기 초기화
                if os.path.exists(MODEL_PATH):
                    logger.info(f"모델 로딩: {MODEL_PATH}")
                    classifier = ModelClassifier(model_path=MODEL_PATH)

                    # 앙상블 분류기 초기화
                    segment_analyzer = get_segment_analyzer()
                    rule_classifier = RuleBasedClassifier(feature_extractor)

                    ensemble = EnsembleClassifier(
                        model_classifier=classifier,
                        rule_classifier=rule_classifier,
                        segment_analyzer=segment_analyzer
                    )

                    logger.info(f"모델 로드 완료 (장치: {classifier.device})")
                else:
                    logger.warning(f"모델 파일이 존재하지 않아 규칙 기반 분류만 사용합니다: {MODEL_PATH}")
                    # 규칙 기반 분류만 사용
                    classifier = None  # 모델 없음을 명시
                    segment_analyzer = get_segment_analyzer()
                    rule_classifier = RuleBasedClassifier(feature_extractor)

                    # 규칙 기반 분류와 세그먼트 분석만 사용하는 앙상블 생성
                    ensemble = EnsembleClassifier(
                        model_classifier=None,
                        rule_classifier=rule_classifier,
                        segment_analyzer=segment_analyzer,
                        model_weight=0.0,
                        rule_weight=0.7,
                        segment_weight=0.3
                    )
            except Exception as e:
                logger.error(f"모델 로드 실패, 규칙 기반 분류만 사용합니다: {str(e)}")
                # 규칙 기반 분류만 사용
                classifier = None  # 모델 없음을 명시
                segment_analyzer = get_segment_analyzer()
                rule_classifier = RuleBasedClassifier(feature_extractor)

                # 규칙 기반 분류와 세그먼트 분석만 사용하는 앙상블 생성
                ensemble = EnsembleClassifier(
                    model_classifier=None,
                    rule_classifier=rule_classifier,
                    segment_analyzer=segment_analyzer,
                    model_weight=0.0,
                    rule_weight=0.7,
                    segment_weight=0.3
                )
        except Exception as e:
            logger.error(f"분류기 초기화 실패: {str(e)}")
            raise HTTPException(status_code=500, detail=f"분류기 초기화 실패: {str(e)}")

    return classifier, ensemble

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """요청 로깅 미들웨어"""
    request_id = id(request)
    logger.info(f"요청 시작: {request_id} - {request.method} {request.url.path}")
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"요청 완료: {request_id} - {process_time:.4f}초 소요")
    
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 핸들러"""
    error_detail = str(exc)
    logger.error(f"예외 발생: {error_detail}")
    
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        error_msg = exc.detail
    else:
        status_code = 500
        error_msg = "내부 서버 오류"
    
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=error_msg, detail=error_detail).dict()
    )

@app.get("/", response_model=Dict)
async def root():
    """루트 엔드포인트"""
    return {
        "service": "ReviewClassifierPro API",
        "version": "2.0.0",
        "status": "정상 작동 중",
        "docs_url": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """상태 확인 엔드포인트"""
    # 모델 로드 여부 확인 (실제 로드하지는 않음)
    model_loaded = classifier is not None or os.path.exists(MODEL_PATH)
    
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        model_loaded=model_loaded,
        uptime=time.time() - START_TIME,
        device=str(device)
    )

@app.post("/review/analyze", response_model=AnalysisResponse)
async def analyze_review(review: ReviewRequest):
    """
    리뷰 문장 단위 분석 엔드포인트
    
    Args:
        review: 분석할 리뷰 데이터
    
    Returns:
        문장 단위 분석 결과
    """
    start_time = time.time()
    
    try:
        # 세그먼트 분석기 가져오기
        analyzer = get_segment_analyzer()
        
        # 분석 수행
        result = analyzer.analyze(review.text)
        
        # 결과 생성
        processing_time = time.time() - start_time
        
        return AnalysisResponse(
            is_abnormal=result['is_abnormal'],
            abnormal_score=result['max_abnormal_score'],
            abnormal_ratio=result['abnormal_ratio'],
            context_coherence=result.get('context_coherence', 1.0),
            segments=[{
                'text': segment['sentence'],
                'is_abnormal': segment['is_abnormal'],
                'abnormal_score': segment['abnormal_score'],
                'abnormal_factors': segment['abnormal_factors'] if segment['is_abnormal'] else []
            } for segment in result['segment_results']],
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"리뷰 분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def simple_predict(text):
    """모델 없이 간단한 규칙 기반 예측"""
    preprocessor = TextPreprocessor()
    feature_extractor = AdvancedFeatureExtractor()

    # 전처리
    preprocessed = preprocessor.preprocess(text)

    # 특성 추출
    features = feature_extractor.extract_features(preprocessed)

    # 비정상 점수 기반 판정 (임시)
    abnormal_score = features['abnormal_score']
    is_normal = abnormal_score < 0.5
    confidence = 1 - abnormal_score if is_normal else abnormal_score

    # 비정상 요소 추출
    abnormal_factors = []
    if features.get('has_spam_keywords'):
        abnormal_factors.append(f"스팸 키워드 포함: {', '.join(features['spam_keywords_found'])}")
    if features.get('has_profanity'):
        abnormal_factors.append(f"욕설 포함: {', '.join(features['profanity_words_found'])}")
    if features.get('has_repetition'):
        abnormal_factors.append("반복 패턴 포함")

    return {
        'is_normal': is_normal,
        'confidence': confidence,
        'abnormal_factors': abnormal_factors,
        'method': 'rule_based'
    }


@app.post("/review/validate", response_model=ReviewResponse)
async def validate_review(review: ReviewRequest):
    """단일 리뷰 검증 엔드포인트 (임시 규칙 기반)"""
    start_time = time.time()

    try:
        # 규칙 기반 예측 사용
        prediction = simple_predict(review.text)

        processing_time = time.time() - start_time

        return ReviewResponse(
            is_normal=prediction['is_normal'],
            confidence=prediction['confidence'],
            abnormal_factors=prediction.get('abnormal_factors', []),
            method=prediction.get('method', 'rule_based'),
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"리뷰 검증 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/review/validate", response_model=ReviewResponse)
# async def validate_review(review: ReviewRequest, background_tasks: BackgroundTasks):
#     """
#     단일 리뷰 검증 엔드포인트
#
#     Args:
#         review: 검증할 리뷰 데이터
#
#     Returns:
#         검증 결과
#     """
#     start_time = time.time()
#
#     try:
#         # 앙상블 분류기 가져오기
#         _, ensemble_classifier = get_classifier()
#
#         # 예측 수행
#         logger.info(f"리뷰 검증 시작: {review.text[:50]}...")
#
#         # 전처리
#         preprocessed = preprocessor.preprocess(review.text)
#
#         # 특성 추출
#         features = feature_extractor.extract_features(preprocessed)
#
#         # 앙상블 예측
#         prediction = ensemble_classifier.predict(
#             text=review.text,
#             preprocessed_data=preprocessed,
#             features=features
#         )
#
#         # 결과 생성
#         processing_time = time.time() - start_time
#
#         # 빈번한 비정상 리뷰 로깅 (백그라운드 작업)
#         if not prediction['is_normal'] and prediction['confidence'] > 0.8:
#             background_tasks.add_task(log_abnormal_review, review.text, prediction)
#
#         return ReviewResponse(
#             is_normal=prediction['is_normal'],
#             confidence=prediction['confidence'],
#             abnormal_factors=prediction.get('abnormal_factors', []),
#             method=prediction.get('method', 'ensemble'),
#             processing_time=processing_time
#         )
#     except Exception as e:
#         logger.error(f"리뷰 검증 중 오류 발생: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/reviews/batch-validate", response_model=BatchReviewResponse)
async def batch_validate_reviews(batch: BatchReviewRequest):
    """
    일괄 리뷰 검증 엔드포인트
    
    Args:
        batch: 검증할 리뷰 목록
    
    Returns:
        검증 결과 목록
    """
    start_time = time.time()
    
    try:
        # 앙상블 분류기 가져오기
        _, ensemble_classifier = get_classifier()
        
        results = []
        normal_count = 0
        abnormal_count = 0
        
        logger.info(f"일괄 검증 시작: {len(batch.reviews)}개 리뷰")
        
        for review in batch.reviews:
            review_start = time.time()
            
            # 앙상블 예측
            prediction = ensemble_classifier.predict(review.text)
            
            # 결과 저장
            review_time = time.time() - review_start
            result = ReviewResponse(
                is_normal=prediction['is_normal'],
                confidence=prediction['confidence'],
                abnormal_factors=prediction.get('abnormal_factors', []),
                method=prediction.get('method', 'ensemble'),
                processing_time=review_time
            )
            results.append(result)
            
            # 카운트 증가
            if prediction['is_normal']:
                normal_count += 1
            else:
                abnormal_count += 1
        
        # 총 처리 시간
        processing_time = time.time() - start_time
        logger.info(f"일괄 검증 완료: {len(batch.reviews)}개 리뷰, {processing_time:.4f}초 소요")
        
        return BatchReviewResponse(
            results=results,
            processing_time=processing_time,
            total_reviews=len(batch.reviews),
            normal_count=normal_count,
            abnormal_count=abnormal_count
        )
    except Exception as e:
        logger.error(f"일괄 검증 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def log_abnormal_review(text, prediction):
    """비정상 리뷰 로깅 (백그라운드 작업)"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 로그 파일에 비정상 리뷰 기록
        with open('abnormal_reviews.log', 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] 비정상 리뷰 감지 (신뢰도: {prediction['confidence']:.2f})\n")
            f.write(f"텍스트: {text}\n")
            f.write(f"비정상 요소: {', '.join(prediction.get('abnormal_factors', []))}\n")
            f.write(f"판정 방법: {prediction.get('method', 'ensemble')}\n")
            f.write("-" * 80 + "\n")
    except Exception as e:
        logger.error(f"비정상 리뷰 로깅 중 오류 발생: {str(e)}")

def start_server(host=None, port=None, reload=None):
    """API 서버 시작"""
    logger.info("API 서버 시작 중...")
    
    # 설정값 사용
    _host = host or API_CONFIG.get('host', '0.0.0.0')
    _port = port or API_CONFIG.get('port', 8000)
    _reload = reload if reload is not None else API_CONFIG.get('reload', False)
    
    uvicorn.run(
        "api.app:app",
        host=_host,
        port=_port,
        reload=_reload,
        workers=API_CONFIG.get('workers', 1)
    )

if __name__ == "__main__":
    start_server()
