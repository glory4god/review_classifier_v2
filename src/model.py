"""
KoELECTRA 기반 분류 모델
- 한국어에 최적화된 모델 구현
- 정규화 기법 적용으로 과적합 방지
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraModel, ElectraTokenizer, ElectraConfig
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import sys
import os
import logging

# 상위 디렉토리 추가 (상대 임포트를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_CONFIG

logger = logging.getLogger("model")

class ReviewDataset(Dataset):
    """
    리뷰 데이터셋 클래스
    """

    def __init__(self, model_name: str = TRAIN_CONFIG['model_name'], num_classes: int = 2,
                 dropout_rate: float = TRAIN_CONFIG['dropout_rate']):
        """
        Args:
            model_name: 사전학습 모델 이름 (예: 'monologg/koelectra-base-v3-discriminator')
            num_classes: 클래스 수
            dropout_rate: 드롭아웃 비율
        """
        super(ReviewClassifierPro, self).__init__()
        # config만 불러오고 직접 모델을 로드하지 않음
        self.config = ElectraConfig.from_pretrained(model_name)

        # 직접 config로부터 모델 초기화
        self.electra = ElectraModel(self.config)

        # 나머지 레이어 초기화
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        # L2 정규화 강도
        self.l2_reg = TRAIN_CONFIG.get('l2_reg', 0.01)
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 토크나이징
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class ReviewClassifierPro(nn.Module):
    """
    KoELECTRA 기반 리뷰 분류 모델
    """
    def __init__(self, model_name: str = TRAIN_CONFIG['model_name'], num_classes: int = 2, dropout_rate: float = TRAIN_CONFIG['dropout_rate']):
        """
        Args:
            model_name: 사전학습 모델 이름 (예: 'monologg/koelectra-base-v3-discriminator')
            num_classes: 클래스 수
            dropout_rate: 드롭아웃 비율
        """
        super(ReviewClassifierPro, self).__init__()
        self.config = ElectraConfig.from_pretrained(model_name)
        self.electra = ElectraModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 분류기 - 더 깊은 네트워크로 구성
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # L2 정규화 강도
        self.l2_reg = TRAIN_CONFIG.get('l2_reg', 0.01)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        순전파
        
        Args:
            input_ids: 입력 ID
            attention_mask: 어텐션 마스크
            token_type_ids: 토큰 타입 ID
            
        Returns:
            logits
        """
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # ELECTRA는 pooler_output이 없으므로 마지막 hidden_state의 [CLS] 토큰 사용
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_l2_loss(self):
        """
        L2 정규화 손실 계산
        
        Returns:
            L2 손실
        """
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, 2)
        return self.l2_reg * l2_loss
    
    def predict_proba(self, input_ids, attention_mask, token_type_ids=None):
        """
        확률 예측
        
        Args:
            input_ids: 입력 ID
            attention_mask: 어텐션 마스크
            token_type_ids: 토큰 타입 ID
            
        Returns:
            클래스별 확률
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, token_type_ids)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def predict(self, input_ids, attention_mask, token_type_ids=None):
        """
        클래스 예측
        
        Args:
            input_ids: 입력 ID
            attention_mask: 어텐션 마스크
            token_type_ids: 토큰 타입 ID
            
        Returns:
            예측 클래스
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, token_type_ids)
            _, preds = torch.max(logits, dim=1)
        return preds

class EnsembleClassifier:
    """
    앙상블 분류기
    - 규칙 기반, 모델 기반, 문장 분석 결합
    """
    def __init__(self, 
                 model_classifier, 
                 rule_classifier, 
                 segment_analyzer,
                 model_weight=None, 
                 rule_weight=None, 
                 segment_weight=None):
        """
        Args:
            model_classifier: 모델 기반 분류기
            rule_classifier: 규칙 기반 분류기
            segment_analyzer: 문장 단위 분석기
            model_weight: 모델 가중치
            rule_weight: 규칙 가중치
            segment_weight: 세그먼트 가중치
        """
        from config import ENSEMBLE_CONFIG
        
        self.model_classifier = model_classifier
        self.rule_classifier = rule_classifier
        self.segment_analyzer = segment_analyzer
        
        # 가중치 설정
        self.model_weight = model_weight or ENSEMBLE_CONFIG.get('model_weight', 0.5)
        self.rule_weight = rule_weight or ENSEMBLE_CONFIG.get('rule_weight', 0.3)
        self.segment_weight = segment_weight or ENSEMBLE_CONFIG.get('segment_weight', 0.2)
        self.threshold = ENSEMBLE_CONFIG.get('threshold', 0.5)
        
        logger.info(f"앙상블 분류기 초기화 - 모델: {self.model_weight}, 규칙: {self.rule_weight}, 세그먼트: {self.segment_weight}")
    
    def predict(self, text, preprocessed_data=None, features=None):
        """
        앙상블 예측 수행
        
        Args:
            text: 리뷰 텍스트
            preprocessed_data: 전처리된 데이터 (없으면 생성)
            features: 특성 (없으면 추출)
            
        Returns:
            예측 결과 딕셔너리
        """
        # 1. 규칙 기반 분류 (빠른 필터링)
        rule_result = self.rule_classifier.predict(text, features)
        
        # 빠른 판단: 규칙에서 매우 높은 확신도로 비정상이면 바로 반환
        if not rule_result['is_normal'] and rule_result['confidence'] > 0.9:
            return {
                'is_normal': False,
                'confidence': rule_result['confidence'],
                'method': 'rule_based_fast',
                'abnormal_factors': rule_result.get('abnormal_factors', []),
                'text': text
            }
        
        # 2. 모델 기반 분류
        model_result = self.model_classifier.predict(text)
        
        # 3. 문장 단위 분석
        segment_result = self.segment_analyzer.analyze(text)
        
        # 4. 앙상블 결정
        # 각 분류기의 정상 확률 (0에 가까울수록 비정상, 1에 가까울수록 정상)
        rule_normal_prob = rule_result['confidence'] if rule_result['is_normal'] else 1 - rule_result['confidence']
        model_normal_prob = model_result['confidence'] if model_result['is_normal'] else 1 - model_result['confidence']
        segment_normal_prob = 1 - segment_result['max_abnormal_score']  # 비정상 점수의 반대
        
        # 가중 평균 계산
        weighted_normal_prob = (
            self.model_weight * model_normal_prob +
            self.rule_weight * rule_normal_prob +
            self.segment_weight * segment_normal_prob
        )
        
        # 최종 판정
        is_normal = weighted_normal_prob >= self.threshold
        confidence = weighted_normal_prob if is_normal else 1 - weighted_normal_prob
        
        # 비정상 요소 수집
        abnormal_factors = []
        
        # 규칙 기반에서 감지된 비정상 요소
        if not rule_result['is_normal'] and 'abnormal_factors' in rule_result:
            abnormal_factors.extend(rule_result['abnormal_factors'])
        
        # 세그먼트 분석에서 감지된 비정상 요소
        if segment_result['is_abnormal']:
            for segment in segment_result['segment_results']:
                if segment['is_abnormal']:
                    sentence_preview = segment['sentence'][:30] + '...' if len(segment['sentence']) > 30 else segment['sentence']
                    abnormal_factors.append(f"비정상 문장 (점수: {segment['abnormal_score']:.2f}): '{sentence_preview}'")
        
        # 결과 반환
        return {
            'is_normal': is_normal,
            'confidence': confidence,
            'method': 'ensemble',
            'abnormal_factors': abnormal_factors,
            'text': text,
            'segment_results': segment_result['segment_results'],
            'component_results': {
                'rule': rule_result,
                'model': model_result,
                'segment': segment_result
            }
        }
