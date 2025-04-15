"""
텍스트 전처리 모듈
- 텍스트 정규화
- 형태소 분석
- 한국어 비율 계산 등
"""

import re
import logging
from typing import List, Dict, Tuple, Union, Optional
import sys
import os

# 상위 디렉토리 추가 (상대 임포트를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES

logger = logging.getLogger("preprocess")

class TextPreprocessor:
    """
    텍스트 전처리 클래스
    - 텍스트 정규화
    - 한국어 비율 계산
    - 영어 리뷰 탐지
    - 반복 패턴 감지 등
    """
    def __init__(self, use_morphs: bool = True):
        """
        Args:
            use_morphs: 형태소 분석 사용 여부
        """
        self.use_morphs = use_morphs
        
        # 형태소 분석기 초기화 (Mecab)
        if use_morphs:
            try:
                from konlpy.tag import Mecab
                self.mecab = Mecab()
                logger.info("Mecab 형태소 분석기 초기화 완료")
            except Exception as e:
                logger.warning(f"Mecab 초기화 실패: {e}")
                logger.warning("형태소 분석을 사용하지 않습니다.")
                self.use_morphs = False
    
    def normalize(self, text: str) -> str:
        """
        텍스트 정규화
        
        Args:
            text: 원본 텍스트
            
        Returns:
            정규화된 텍스트
        """
        if not isinstance(text, str):
            return ""
            
        # 중복 공백 제거
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 중복 문장부호 제거 (3개 이상은 2개로 변환)
        text = re.sub(r'[.]{3,}', '..', text)
        text = re.sub(r'[!]{3,}', '!!', text)
        text = re.sub(r'[?]{3,}', '??', text)
        
        # 이메일, URL 마스킹
        text = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL]', text)
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        
        # 이모티콘 제거
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        
        return text
    
    def get_morphs(self, text: str) -> List[str]:
        """
        형태소 분석 수행
        
        Args:
            text: 원본 텍스트
            
        Returns:
            형태소 리스트
        """
        if not self.use_morphs:
            return text.split()
            
        try:
            return self.mecab.morphs(text)
        except Exception as e:
            logger.warning(f"형태소 분석 중 오류 발생: {e}")
            return text.split()
    
    def calculate_korean_ratio(self, text: str) -> float:
        """
        텍스트 내 한국어 비율 계산
        
        Args:
            text: 원본 텍스트
            
        Returns:
            한국어 문자 비율 (0.0 ~ 1.0)
        """
        if not text:
            return 0.0
            
        # 한글 문자 패턴
        korean_pattern = re.compile('[가-힣]')
        # 공백 제외 문자 수
        total_chars = len(text.replace(" ", ""))
        
        if total_chars == 0:
            return 0.0
            
        # 한글 문자 수
        korean_chars = len(korean_pattern.findall(text))
        
        return korean_chars / total_chars
    
    def is_english_review(self, text: str, threshold: float = 0.6) -> bool:
        """
        영어 리뷰 판별
        
        Args:
            text: 원본 텍스트
            threshold: 영어 비율 임계값
            
        Returns:
            영어 리뷰 여부
        """
        if not text:
            return False
            
        # 영어 문자 패턴
        english_pattern = re.compile('[a-zA-Z]')
        # 공백 제외 문자 수
        total_chars = len(text.replace(" ", ""))
        
        if total_chars == 0:
            return False
            
        # 영어 문자 수
        english_chars = len(english_pattern.findall(text))
        english_ratio = english_chars / total_chars
        
        return english_ratio > threshold
    
    def detect_repetition(self, text: str, threshold: int = 3) -> Tuple[bool, float]:
        """
        반복 패턴 감지
        
        Args:
            text: 원본 텍스트
            threshold: 반복 임계값
            
        Returns:
            (반복 패턴 존재 여부, 반복 비율)
        """
        if not text:
            return False, 0.0
            
        # 단일 문자 반복 (예: 'ㅋㅋㅋㅋㅋㅋ')
        char_repetition = re.search(r'(.)\1{' + str(threshold) + ',}', text)
        
        # 단어 반복
        words = text.split()
        if not words:
            return False, 0.0
            
        word_counts = {}
        
        for word in words:
            if len(word) > 1:  # 1글자 단어는 제외
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # 최대 반복 횟수 및 단어
        max_repeat = 0
        max_repeat_word = ""
        
        for word, count in word_counts.items():
            if count > max_repeat:
                max_repeat = count
                max_repeat_word = word
        
        # 반복 비율 (전체 단어 중 최대 반복 단어의 비율)
        repetition_ratio = max_repeat / len(words) if words else 0.0
        
        # 단어 반복 여부
        word_repetition = max_repeat >= threshold
        
        # 최종 결과
        is_repetition = bool(char_repetition) or word_repetition
        
        return is_repetition, repetition_ratio
    
    def segment_text(self, text: str) -> List[str]:
        """
        텍스트를 문장 단위로 분리
        
        Args:
            text: 원본 텍스트
            
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
    
    def preprocess(self, text: str) -> Dict:
        """
        전체 전처리 수행
        
        Args:
            text: 원본 텍스트
            
        Returns:
            전처리 결과 및 특성
        """
        if not isinstance(text, str):
            text = str(text)
            
        # 기본 정규화
        normalized_text = self.normalize(text)
        
        # 형태소 분석
        morphs = self.get_morphs(normalized_text) if self.use_morphs else normalized_text.split()
        
        # 문장 분리
        segments = self.segment_text(normalized_text)
        
        # 특성 추출
        korean_ratio = self.calculate_korean_ratio(normalized_text)
        is_english = self.is_english_review(normalized_text)
        has_repetition, repetition_ratio = self.detect_repetition(normalized_text)
        
        return {
            "original_text": text,
            "normalized_text": normalized_text,
            "morphs": morphs,
            "segments": segments,
            "korean_ratio": korean_ratio,
            "is_english_review": is_english,
            "has_repetition": has_repetition,
            "repetition_ratio": repetition_ratio,
            "segment_count": len(segments)
        }
