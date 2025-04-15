"""
문장 단위 분석기 모듈
- 리뷰를 문장 단위로 분리하여 각 문장의 비정상 점수 계산
- 부분적 스팸 텍스트 감지 강화
"""

import re
import sys
import os
import logging
from typing import List, Dict, Tuple, Any, Optional

# 상위 디렉토리 추가 (상대 임포트를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES

logger = logging.getLogger("segment-analyzer")

class SegmentAnalyzer:
    """
    문장 단위 분석기 클래스
    - 리뷰를 문장 단위로 분리하여 각 문장의 비정상 점수 계산
    - 문장 내 스팸 패턴, 욕설, 비정상 요소 감지
    """
    def __init__(self, preprocessor, feature_extractor, threshold=None):
        """
        Args:
            preprocessor: 텍스트 전처리기
            feature_extractor: 특성 추출기
            threshold: 비정상 판정 임계값 (None이면 기본값 사용)
        """
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.threshold = threshold or FEATURES.get('segment_abnormal_threshold', 0.6)
        
    def analyze(self, text):
        """
        리뷰를 문장 단위로 분석
        
        Args:
            text: 분석할 리뷰 텍스트
            
        Returns:
            분석 결과 딕셔너리
        """
        # 문장 분리
        segments = self._split_into_segments(text)
        
        # 분석 결과
        results = []
        
        # 각 문장 분석
        for segment in segments:
            segment_result = self._analyze_segment(segment)
            results.append(segment_result)
        
        # 전체 판정
        is_abnormal = any(r['is_abnormal'] for r in results)
        
        # 최대 점수
        max_score = max(r['abnormal_score'] for r in results) if results else 0
        
        # 비정상 문장 비율
        abnormal_ratio = sum(1 for r in results if r['is_abnormal']) / len(results) if results else 0
        
        # 문맥 일관성 검사
        context_coherence_score = self._check_context_coherence(segments, results)
        
        return {
            'is_abnormal': is_abnormal,
            'max_abnormal_score': max_score,
            'abnormal_ratio': abnormal_ratio,
            'context_coherence': context_coherence_score,
            'segment_results': results
        }
    
    def _split_into_segments(self, text):
        """
        리뷰를 문장 단위로 분리
        
        Args:
            text: 리뷰 텍스트
            
        Returns:
            문장 리스트
        """
        # 문장 분리 (마침표, 느낌표, 물음표로 구분)
        segments = re.split(r'([.!?]\s)', text)
        
        # 홀수 인덱스의 항목(구분자)을 앞의 문장에 추가
        cleaned_segments = []
        for i in range(0, len(segments) - 1, 2):
            if i + 1 < len(segments):
                cleaned_segments.append(segments[i] + segments[i + 1])
            else:
                cleaned_segments.append(segments[i])
                
        # 마지막 항목 처리
        if len(segments) % 2 == 1:
            if segments[-1].strip():
                cleaned_segments.append(segments[-1])
        
        # 빈 문장 제거 및 공백 제거
        return [s.strip() for s in cleaned_segments if s.strip()]
    
    def _analyze_segment(self, segment):
        """
        개별 문장 분석
        
        Args:
            segment: 분석할 문장
            
        Returns:
            문장 분석 결과
        """
        # 전처리
        preprocessed = self.preprocessor.preprocess(segment)
        
        # 특성 추출
        features = self.feature_extractor.extract_features(preprocessed)
        
        # 비정상 판정
        is_abnormal = features['abnormal_score'] > self.threshold
        
        # 비정상 요소 추출
        abnormal_factors = []
        if features['has_spam_keywords']:
            abnormal_factors.append(f"스팸 키워드: {', '.join(features['spam_keywords_found'])}")
        if features['has_profanity']:
            abnormal_factors.append(f"욕설: {', '.join(features['profanity_words_found'])}")
        if features['has_repetition']:
            abnormal_factors.append("반복 패턴")
        if features['is_korean_insufficient']:
            abnormal_factors.append(f"한국어 비율 낮음: {features['korean_ratio']:.2f}")
        if 'has_jamo_pattern' in features and features['has_jamo_pattern']:
            abnormal_factors.append("자소 분리 패턴")
        if 'has_weird_punctuation' in features and features['has_weird_punctuation']:
            abnormal_factors.append("이상한 구두점")
        if 'context_coherence' in features and features['context_coherence'] < 0.5:
            abnormal_factors.append("낮은 문맥 일관성")
        
        return {
            'sentence': segment,
            'abnormal_score': features['abnormal_score'],
            'is_abnormal': is_abnormal,
            'abnormal_factors': abnormal_factors,
            'features': features
        }
    
    def _check_context_coherence(self, segments, results):
        """
        문맥 일관성 검사
        
        Args:
            segments: 문장 리스트
            results: 문장별 분석 결과
            
        Returns:
            문맥 일관성 점수 (0.0 ~ 1.0)
        """
        if len(segments) <= 1:
            return 1.0
        
        # 문장 간 주제 일관성 측정
        # (간단한 구현: 연속된 문장의 비정상 점수 차이가 크면 일관성이 낮다고 판단)
        coherence_scores = []
        for i in range(1, len(results)):
            score_diff = abs(results[i]['abnormal_score'] - results[i-1]['abnormal_score'])
            # 점수 차이를 0~1 사이로 정규화 (차이가 클수록 일관성 낮음)
            coherence = max(0, 1 - score_diff)
            coherence_scores.append(coherence)
        
        # 평균 일관성 점수
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 1.0

    def get_abnormal_segments(self, text):
        """
        비정상으로 판정된 문장만 추출
        
        Args:
            text: 분석할 리뷰 텍스트
            
        Returns:
            비정상 문장 리스트
        """
        result = self.analyze(text)
        return [segment for segment in result['segment_results'] if segment['is_abnormal']]
