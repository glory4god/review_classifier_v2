"""
고급 특성 추출기 모듈
- 자소 분리, 문맥 일관성 등 고급 특성 추출
- 기존 특성 추출기 확장
"""

import re
import sys
import os
from typing import List, Dict, Set, Any, Tuple
import logging

# 상위 디렉토리 추가 (상대 임포트를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES

logger = logging.getLogger("advanced-features")

class AdvancedFeatureExtractor:
    """
    고급 특성 추출기
    - 자소 분리, 문맥 일관성 등 고급 특성 추출
    - 기존 특성 추출기 확장
    """
    def __init__(self, base_extractor=None):
        """
        Args:
            base_extractor: 기본 특성 추출기 (None이면 내부적으로 생성)
        """
        # 기본 특성 추출기가 없으면 생성
        if base_extractor is None:
            from src.features import FeatureExtractor
            self.base_extractor = FeatureExtractor()
        else:
            self.base_extractor = base_extractor
            
    def extract_features(self, preprocessed_data):
        """
        고급 특성 추출
        
        Args:
            preprocessed_data: 전처리된 데이터
            
        Returns:
            추출된 특성 딕셔너리
        """
        # 기본 특성 추출
        base_features = self.base_extractor.extract_features(preprocessed_data)
        
        text = preprocessed_data.get('normalized_text', '')
        
        # 추가 특성 계산
        additional_features = {
            # 자소 분리 패턴 감지
            'has_jamo_pattern': self._detect_jamo_pattern(text),
            
            # 이상한 구두점 감지
            'has_weird_punctuation': self._detect_weird_punctuation(text),
            
            # 문맥 일관성 점수
            'context_coherence': self._calculate_context_coherence(text),
            
            # 단어 복잡도 (정상 리뷰는 일반적으로 복잡도가 높음)
            'word_complexity': self._calculate_word_complexity(text),
            
            # 문자 다양성 (비정상 리뷰는 종종 문자 다양성이 낮음)
            'char_diversity': self._calculate_char_diversity(text)
        }
        
        # 추가 특성에 따른 비정상 점수 조정
        abnormal_score = base_features['abnormal_score']
        
        if additional_features['has_jamo_pattern']:
            abnormal_score += 0.3
        
        if additional_features['has_weird_punctuation']:
            abnormal_score += 0.2
        
        if additional_features['context_coherence'] < 0.5:
            abnormal_score += 0.2
        
        if additional_features['word_complexity'] < 0.3:
            abnormal_score += 0.15
        
        if additional_features['char_diversity'] < 0.4:
            abnormal_score += 0.15
        
        # 최종 점수는 1.0을 초과하지 않도록 제한
        additional_features['adjusted_abnormal_score'] = min(abnormal_score, 1.0)
        
        # 결과 병합
        return {**base_features, **additional_features, 'abnormal_score': additional_features['adjusted_abnormal_score']}
    
    def _detect_jamo_pattern(self, text):
        """
        자소 분리 패턴 감지 (예: ㄴㅇㄹ, ㅋㅋㅋ)
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            자소 분리 패턴 존재 여부
        """
        jamo_pattern = re.compile(r'[ㄱ-ㅎ]{2,}')
        return bool(jamo_pattern.search(text))
    
    def _detect_weird_punctuation(self, text):
        """
        이상한 구두점 패턴 감지
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            이상한 구두점 패턴 존재 여부
        """
        punct_pattern = re.compile(r'[.,!?;:~-]{3,}')
        return bool(punct_pattern.search(text))
    
    def _calculate_context_coherence(self, text):
        """
        문맥 일관성 점수 계산
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            문맥 일관성 점수 (0.0 ~ 1.0)
        """
        # 간단한 구현: 문장 간 어휘 유사도
        sentences = re.split(r'[.!?]\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return 1.0  # 문장이 하나면 일관성 있다고 간주
        
        # 문장 간 유사도 계산
        similarities = []
        for i in range(1, len(sentences)):
            prev_words = set(sentences[i-1].split())
            curr_words = set(sentences[i].split())
            
            # 자카드 유사도
            if prev_words and curr_words:
                similarity = len(prev_words & curr_words) / len(prev_words | curr_words)
                similarities.append(similarity)
        
        # 평균 유사도 반환
        return sum(similarities) / len(similarities) if similarities else 1.0
    
    def _calculate_word_complexity(self, text):
        """
        단어 복잡도 계산 - 평균 단어 길이, 다양성 등
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            단어 복잡도 점수 (0.0 ~ 1.0)
        """
        words = text.split()
        if not words:
            return 0.0
        
        # 평균 단어 길이
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # 고유 단어 비율
        unique_words = len(set(words)) / len(words)
        
        # 복잡도 점수 계산
        # 정규화 (일반적으로 2-5자가 정상적인 한국어 단어)
        length_score = min(max((avg_word_length - 1) / 4, 0), 1)
        
        # 최종 복잡도 점수 (길이와 다양성 가중 평균)
        return 0.7 * length_score + 0.3 * unique_words
    
    def _calculate_char_diversity(self, text):
        """
        문자 다양성 계산
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            문자 다양성 점수 (0.0 ~ 1.0)
        """
        if not text:
            return 0.0
        
        # 중복 제외 문자 비율
        unique_chars = len(set(text))
        total_chars = len(text)
        
        return unique_chars / total_chars
    
    def enhanced_abnormal_check(self, features):
        """
        고급 비정상 리뷰 확인
        
        Args:
            features: 추출된 특성
            
        Returns:
            (비정상 여부, 비정상 점수)
        """
        abnormal_score = features.get('abnormal_score', 0)
        
        # 추가 특성 기반 조정
        if features.get('has_jamo_pattern', False):
            abnormal_score += 0.3
        
        if features.get('has_weird_punctuation', False):
            abnormal_score += 0.2
        
        if features.get('context_coherence', 1.0) < 0.5:
            abnormal_score += 0.2
        
        # 최종 점수는 1.0을 초과하지 않도록 제한
        abnormal_score = min(abnormal_score, 1.0)
        
        # 임계값으로 판정
        threshold = 0.6  # 비정상 판정 임계값
        is_abnormal = abnormal_score >= threshold
        
        return is_abnormal, abnormal_score
