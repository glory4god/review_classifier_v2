"""
예측 모듈
- 모델 기반 예측
- 규칙 기반 예측
- 단일 리뷰 및 배치 예측
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Union, Any, Optional
from datetime import datetime

# 상위 디렉토리 추가 (상대 임포트를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, TRAIN_CONFIG

logger = logging.getLogger("predict")

class RuleBasedClassifier:
    """
    규칙 기반 분류기
    - 빠른 필터링을 위한 규칙 기반 분류
    """
    def __init__(self, feature_extractor, threshold: float = 0.7):
        """
        Args:
            feature_extractor: 특성 추출기
            threshold: 비정상 판정 임계값
        """
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        
    def predict(self, text: str, features: Optional[Dict] = None) -> Dict:
        """
        규칙 기반 예측
        
        Args:
            text: 예측할 텍스트
            features: 이미 추출된 특성 (없으면 직접 추출)
            
        Returns:
            예측 결과 딕셔너리
        """
        from src.preprocess import TextPreprocessor
        
        # 특성이 없으면 추출
        if features is None:
            preprocessor = TextPreprocessor()
            preprocessed = preprocessor.preprocess(text)
            features = self.feature_extractor.extract_features(preprocessed)
        
        # 비정상 점수
        abnormal_score = features.get('abnormal_score', 0)
        
        # 임계값 기반 판정
        is_normal = abnormal_score < self.threshold
        confidence = 1 - abnormal_score if is_normal else abnormal_score
        
        # 비정상 요소 추출
        abnormal_factors = []
        if features.get('has_spam_keywords'):
            abnormal_factors.append(f"스팸 키워드 포함: {', '.join(features['spam_keywords_found'])}")
        if features.get('has_profanity'):
            abnormal_factors.append(f"욕설 포함: {', '.join(features['profanity_words_found'])}")
        if features.get('is_english_review'):
            abnormal_factors.append("영어 리뷰")
        if features.get('has_repetition'):
            abnormal_factors.append("반복 패턴 포함")
        if features.get('is_korean_insufficient'):
            abnormal_factors.append(f"한국어 비율 낮음: {features['korean_ratio']:.2f}")
        if features.get('is_abnormal_time'):
            abnormal_factors.append("비정상 시간대 작성")
        if features.get('has_jamo_pattern'):
            abnormal_factors.append("자소 분리 패턴 포함")
        if features.get('has_weird_punctuation'):
            abnormal_factors.append("이상한 구두점 패턴")
        
        return {
            'is_normal': is_normal,
            'confidence': confidence,
            'abnormal_factors': abnormal_factors,
            'method': 'rule_based',
            'features': features
        }

class ModelClassifier:
    """
    모델 기반 분류기
    - KoELECTRA 모델을 사용한 분류
    """
    def __init__(self, model_path: str = MODEL_PATH):
        """
        Args:
            model_path: 모델 파일 경로
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        self._load_model()
        
    def _load_model(self):
        """모델 로드"""
        try:
            from transformers import ElectraTokenizer
            from src.model import ReviewClassifierPro
            
            # 토크나이저 로드
            self.tokenizer = ElectraTokenizer.from_pretrained(TRAIN_CONFIG['model_name'])
            
            # 모델 초기화 및 로드
            self.model = ReviewClassifierPro().to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            
            logger.info(f"모델 로드 완료: {self.model_path}")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise RuntimeError(f"모델 로드 실패: {e}")
    
    def predict(self, text: str) -> Dict:
        """
        모델 기반 예측
        
        Args:
            text: 예측할 텍스트
            
        Returns:
            예측 결과 딕셔너리
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        
        # 토크나이징
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=TRAIN_CONFIG['max_length'],
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # 텐서를 장치로 이동
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)
        
        # 예측
        with torch.no_grad():
            # 로짓 계산
            logits = self.model(input_ids, attention_mask, token_type_ids)
            
            # 확률 계산
            probs = torch.softmax(logits, dim=1)
            
            # 클래스 및 확률 추출 (0: 정상, 1: 비정상)
            normal_prob = probs[0, 0].item()
            abnormal_prob = probs[0, 1].item()
            
            is_normal = normal_prob >= 0.5
            confidence = normal_prob if is_normal else abnormal_prob
        
        return {
            'is_normal': is_normal,
            'confidence': confidence,
            'method': 'model_based',
            'normal_prob': normal_prob,
            'abnormal_prob': abnormal_prob
        }

def predict_text(text: str, time: Optional[str] = None, model_file: str = MODEL_PATH, detailed: bool = False) -> Dict:
    """
    단일 텍스트 예측
    
    Args:
        text: 예측할 텍스트
        time: 작성 시간 (형식: YYYY-MM-DD HH:MM:SS)
        model_file: 모델 파일 경로
        detailed: 상세 결과 반환 여부
        
    Returns:
        예측 결과 딕셔너리
    """
    from src.preprocess import TextPreprocessor
    from src.advanced_features import AdvancedFeatureExtractor
    from src.segment_analyzer import SegmentAnalyzer
    from src.model import EnsembleClassifier
    
    # 초기화
    preprocessor = TextPreprocessor()
    feature_extractor = AdvancedFeatureExtractor()
    rule_classifier = RuleBasedClassifier(feature_extractor)
    model_classifier = ModelClassifier(model_file)
    segment_analyzer = SegmentAnalyzer(preprocessor, feature_extractor)
    
    # 앙상블 분류기 생성
    ensemble = EnsembleClassifier(
        model_classifier=model_classifier,
        rule_classifier=rule_classifier,
        segment_analyzer=segment_analyzer
    )
    
    # 전처리
    preprocessed = preprocessor.preprocess(text)
    
    # 작성 시간 추가
    if time:
        preprocessed['created_at'] = time
    
    # 특성 추출
    features = feature_extractor.extract_features(preprocessed)
    
    # 앙상블 예측
    prediction = ensemble.predict(text, preprocessed, features)
    
    # 상세 결과가 필요 없으면 축약 버전 반환
    if not detailed:
        return {
            'is_normal': prediction['is_normal'],
            'confidence': prediction['confidence'],
            'abnormal_factors': prediction.get('abnormal_factors', []),
            'method': prediction.get('method', 'ensemble'),
            'text': text
        }
    
    # 상세 결과 반환
    return prediction

def predict_batch(input_file: str, output_file: str, model_file: str = MODEL_PATH) -> pd.DataFrame:
    """
    배치 예측
    
    Args:
        input_file: 입력 파일 경로 (CSV)
        output_file: 출력 파일 경로 (CSV)
        model_file: 모델 파일 경로
        
    Returns:
        예측 결과 데이터프레임
    """
    # 입력 파일 로드
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"입력 파일 로드 실패: {e}")
        raise RuntimeError(f"입력 파일 로드 실패: {e}")
    
    # 필수 컬럼 확인
    if 'text' not in df.columns:
        raise ValueError("입력 파일에 'text' 컬럼이 없습니다.")
    
    # 결과 저장용 리스트
    results = []
    
    # 모델 초기화 (한 번만)
    from src.preprocess import TextPreprocessor
    from src.advanced_features import AdvancedFeatureExtractor
    from src.segment_analyzer import SegmentAnalyzer
    from src.model import EnsembleClassifier
    
    preprocessor = TextPreprocessor()
    feature_extractor = AdvancedFeatureExtractor()
    rule_classifier = RuleBasedClassifier(feature_extractor)
    model_classifier = ModelClassifier(model_file)
    segment_analyzer = SegmentAnalyzer(preprocessor, feature_extractor)
    
    ensemble = EnsembleClassifier(
        model_classifier=model_classifier,
        rule_classifier=rule_classifier,
        segment_analyzer=segment_analyzer
    )
    
    # 각 행 처리
    total = len(df)
    logger.info(f"배치 예측 시작: {total}개 리뷰")
    
    for i, row in df.iterrows():
        # 진행 로그 (10% 단위)
        if i % max(1, total // 10) == 0:
            logger.info(f"배치 예측 진행 중: {i}/{total} ({i/total*100:.1f}%)")
        
        # 예측
        text = row['text']
        time = row.get('created_at') if 'created_at' in row else None
        
        try:
            # 앙상블 예측
            prediction = ensemble.predict(text)
            
            # 결과 저장
            result = row.to_dict()
            result['is_normal'] = prediction['is_normal']
            result['confidence'] = prediction['confidence']
            result['abnormal_factors'] = ';'.join(prediction.get('abnormal_factors', []))
            result['method'] = prediction.get('method', 'ensemble')
            
            results.append(result)
        except Exception as e:
            logger.error(f"행 {i} 예측 실패: {e}")
            # 실패 시에도 원본 행 유지
            result = row.to_dict()
            result['is_normal'] = None
            result['confidence'] = None
            result['abnormal_factors'] = f"오류: {str(e)}"
            result['method'] = 'error'
            
            results.append(result)
    
    # 결과 데이터프레임 생성
    result_df = pd.DataFrame(results)
    
    # 출력 파일에 저장
    try:
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # CSV 파일로 저장
        result_df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"배치 예측 결과 저장 완료: {output_file}")
    except Exception as e:
        logger.error(f"결과 파일 저장 실패: {e}")
    
    return result_df
