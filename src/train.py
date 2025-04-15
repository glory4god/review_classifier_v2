"""
모델 훈련 모듈
- KoELECTRA 기반 모델 훈련
- 교차 검증 옵션
- 조기 종료 등 정규화 기법 적용
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import ElectraTokenizer, get_linear_schedule_with_warmup

# 상위 디렉토리 추가 (상대 임포트를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_CONFIG, MODEL_PATH
from src.model import ReviewClassifierPro, ReviewDataset

# 로깅 설정
logger = logging.getLogger("train")

def train_model(data_file: str, model_file: str = MODEL_PATH, epochs: int = None, batch_size: int = None,
                learning_rate: float = None, use_early_stopping: bool = None, cross_validation: bool = None):
    """
    모델 훈련
    
    Args:
        data_file: 훈련 데이터 파일 경로
        model_file: 모델 저장 경로
        epochs: 학습 에폭 수
        batch_size: 배치 크기
        learning_rate: 학습률
        use_early_stopping: 조기 종료 사용 여부
        cross_validation: 교차 검증 사용 여부
    """
    # 훈련 설정
    _epochs = epochs or TRAIN_CONFIG['epochs']
    _batch_size = batch_size or TRAIN_CONFIG['batch_size']
    _learning_rate = learning_rate or TRAIN_CONFIG['learning_rate']
    _use_early_stopping = use_early_stopping if use_early_stopping is not None else TRAIN_CONFIG['early_stopping']
    _cross_validation = cross_validation if cross_validation is not None else TRAIN_CONFIG['cross_validation']
    _patience = TRAIN_CONFIG['patience']
    _max_length = TRAIN_CONFIG['max_length']
    _model_name = TRAIN_CONFIG['model_name']
    
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"훈련 장치: {device}")
    
    # 데이터 로드
    try:
        df = pd.read_csv(data_file)
        logger.info(f"데이터 로드: {len(df)}개 샘플")
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        raise RuntimeError(f"데이터 로드 실패: {e}")
    
    # 필수 열 확인
    required_columns = ['text', 'is_abnormal']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"데이터 파일에 필수 열이 없습니다: {col}")
    
    # 토크나이저 로드
    tokenizer = ElectraTokenizer.from_pretrained(_model_name)
    
    # 레이블 분포 확인
    normal_count = len(df[df['is_abnormal'] == 0])
    abnormal_count = len(df[df['is_abnormal'] == 1])
    logger.info(f"레이블 분포: 정상={normal_count}개 ({normal_count/len(df)*100:.1f}%), "
                f"비정상={abnormal_count}개 ({abnormal_count/len(df)*100:.1f}%)")
    
    if _cross_validation:
        # 교차 검증 수행
        logger.info(f"{TRAIN_CONFIG['n_folds']}중 교차 검증 시작")
        kf = KFold(n_splits=TRAIN_CONFIG['n_folds'], shuffle=True, random_state=TRAIN_CONFIG['random_seed'])
        
        fold_results = []
        best_model_state = None
        best_val_f1 = 0.0
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
            logger.info(f"Fold {fold}/{TRAIN_CONFIG['n_folds']} 훈련 시작")
            
            # 훈련 및 검증 데이터 분할
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            
            # 데이터셋 생성
            train_dataset = ReviewDataset(
                texts=train_df['text'].tolist(),
                labels=train_df['is_abnormal'].tolist(),
                tokenizer=tokenizer,
                max_length=_max_length
            )
            
            val_dataset = ReviewDataset(
                texts=val_df['text'].tolist(),
                labels=val_df['is_abnormal'].tolist(),
                tokenizer=tokenizer,
                max_length=_max_length
            )
            
            # 데이터 로더 생성
            train_loader = DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=_batch_size)
            
            # 모델 초기화
            model = ReviewClassifierPro(model_name=_model_name).to(device)
            
            # 최적화기 설정
            optimizer = torch.optim.AdamW(model.parameters(), lr=_learning_rate)
            
            # 손실 함수
            criterion = torch.nn.CrossEntropyLoss()
            
            # 스케줄러
            total_steps = len(train_loader) * _epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(total_steps * 0.1),
                num_training_steps=total_steps
            )
            
            # 훈련 실행
            fold_result = _train_fold(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                device=device,
                epochs=_epochs,
                patience=_patience,
                use_early_stopping=_use_early_stopping
            )
            
            fold_results.append(fold_result)
            
            # 현재 폴드의 성능이 이전 최고 성능보다 좋으면 모델 저장
            if fold_result['best_val_f1'] > best_val_f1:
                best_val_f1 = fold_result['best_val_f1']
                best_model_state = model.state_dict()
        
        # 최종 모델 및 결과 저장
        if best_model_state is not None:
            # 최종 모델 초기화
            final_model = ReviewClassifierPro(model_name=_model_name).to(device)
            final_model.load_state_dict(best_model_state)
            torch.save(final_model.state_dict(), model_file)
            
            # 결과 저장
            avg_train_loss = np.mean([r['train_losses'][-1] for r in fold_results])
            avg_val_loss = np.mean([r['val_losses'][-1] for r in fold_results])
            avg_val_accuracy = np.mean([r['val_accuracies'][-1] for r in fold_results])
            avg_val_f1 = np.mean([r['best_val_f1'] for r in fold_results])
            
            logger.info(f"교차 검증 완료: 평균 검증 F1={avg_val_f1:.4f}, "
                        f"평균 검증 정확도={avg_val_accuracy:.4f}")
            
            # 훈련 히스토리 저장
            _save_training_history(
                fold_results=fold_results,
                avg_metrics={
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_accuracy': avg_val_accuracy,
                    'val_f1': avg_val_f1
                },
                model_file=model_file
            )
    else:
        # 일반 훈련 (단일 훈련/검증 분할)
        
        # 훈련/검증/테스트 분할
        train_size = int(len(df) * (1 - TRAIN_CONFIG['validation_split'] - TRAIN_CONFIG['test_split']))
        val_size = int(len(df) * TRAIN_CONFIG['validation_split'])
        test_size = len(df) - train_size - val_size
        
        train_df, val_df, test_df = np.split(
            df.sample(frac=1, random_state=TRAIN_CONFIG['random_seed']), 
            [train_size, train_size + val_size]
        )
        
        logger.info(f"데이터 분할: 훈련={len(train_df)}개, 검증={len(val_df)}개, 테스트={len(test_df)}개")
        
        # 데이터셋 생성
        train_dataset = ReviewDataset(
            texts=train_df['text'].tolist(),
            labels=train_df['is_abnormal'].tolist(),
            tokenizer=tokenizer,
            max_length=_max_length
        )
        
        val_dataset = ReviewDataset(
            texts=val_df['text'].tolist(),
            labels=val_df['is_abnormal'].tolist(),
            tokenizer=tokenizer,
            max_length=_max_length
        )
        
        test_dataset = ReviewDataset(
            texts=test_df['text'].tolist(),
            labels=test_df['is_abnormal'].tolist(),
            tokenizer=tokenizer,
            max_length=_max_length
        )
        
        # 데이터 로더 생성
        train_loader = DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=_batch_size)
        test_loader = DataLoader(test_dataset, batch_size=_batch_size)
        
        # 모델 초기화
        model = ReviewClassifierPro(model_name=_model_name).to(device)
        
        # 최적화기 설정
        optimizer = torch.optim.AdamW(model.parameters(), lr=_learning_rate)
        
        # 손실 함수
        criterion = torch.nn.CrossEntropyLoss()
        
        # 스케줄러
        total_steps = len(train_loader) * _epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        # 훈련 실행
        result = _train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            epochs=_epochs,
            patience=_patience,
            use_early_stopping=_use_early_stopping
        )
        
        # 테스트셋 평가
        test_loss, test_acc, test_f1 = _evaluate(model, test_loader, criterion, device)
        logger.info(f"테스트 결과: 손실={test_loss:.4f}, 정확도={test_acc:.4f}, F1={test_f1:.4f}")
        
        # 모델 저장
        torch.save(model.state_dict(), model_file)
        logger.info(f"모델 저장 완료: {model_file}")
        
        # 훈련 히스토리 저장
        result['test_loss'] = test_loss
        result['test_accuracy'] = test_acc
        result['test_f1'] = test_f1
        
        _save_training_history([result], None, model_file)

def _train_fold(model, train_loader, val_loader, optimizer, criterion, scheduler, device, 
               epochs, patience, use_early_stopping):
    """
    한 폴드 훈련
    
    Args:
        model: 모델
        train_loader: 훈련 데이터 로더
        val_loader: 검증 데이터 로더
        optimizer: 최적화기
        criterion: 손실 함수
        scheduler: 학습률 스케줄러
        device: 장치
        epochs: 에폭 수
        patience: 조기 종료 인내심
        use_early_stopping: 조기 종료 사용 여부
        
    Returns:
        훈련 결과 딕셔너리
    """
    # 결과 저장용 변수
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1s = []
    
    # 조기 종료 변수
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_model_state = None
    early_stop_counter = 0
    
    # 에폭 반복
    for epoch in range(epochs):
        # 훈련 모드
        model.train()
        train_loss = 0.0
        
        # 훈련 배치 반복
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (훈련)"):
            # 배치 데이터를 장치로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(input_ids, attention_mask, token_type_ids)
            
            # 손실 계산 (교차 엔트로피 + L2 정규화)
            loss = criterion(outputs, labels)
            l2_loss = model.get_l2_loss()
            total_loss = loss + l2_loss
            
            # 역전파
            total_loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 가중치 업데이트
            optimizer.step()
            
            # 스케줄러 업데이트
            scheduler.step()
            
            # 손실 누적
            train_loss += loss.item()
        
        # 평균 훈련 손실
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # 검증
        val_loss, val_acc, val_f1 = _evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)
        
        # 로그 출력
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                    f"훈련 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}, "
                    f"검증 정확도: {val_acc:.4f}, 검증 F1: {val_f1:.4f}")
        
        # 최고 성능 모델 저장
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # 조기 종료 확인
        if use_early_stopping and early_stop_counter >= patience:
            logger.info(f"Epoch {epoch+1}: 조기 종료 (성능 향상 없음: {patience}번 연속)")
            break
    
    # 최고 성능 모델 상태로 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_f1s': val_f1s,
        'best_val_f1': best_val_f1
    }

def _evaluate(model, data_loader, criterion, device):
    """
    모델 평가
    
    Args:
        model: 모델
        data_loader: 데이터 로더
        criterion: 손실 함수
        device: 장치
        
    Returns:
        (손실, 정확도, F1 점수)
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    # 혼동 행렬을 위한 변수
    tp = 0  # True Positive
    fp = 0  # False Positive
    fn = 0  # False Negative
    
    with torch.no_grad():
        for batch in data_loader:
            # 배치 데이터를 장치로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            # 순전파
            outputs = model(input_ids, attention_mask, token_type_ids)
            
            # 손실 계산
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # 예측값 계산
            _, predicted = torch.max(outputs, 1)
            
            # 정확도 계산
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 혼동 행렬 업데이트 (비정상=1이 양성 클래스)
            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()
    
    # 평균 손실
    val_loss = val_loss / len(data_loader)
    
    # 정확도
    accuracy = correct / total
    
    # F1 점수 계산
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return val_loss, accuracy, f1

def _save_training_history(fold_results, avg_metrics=None, model_file=None):
    """
    훈련 히스토리 저장
    
    Args:
        fold_results: 폴드별 훈련 결과
        avg_metrics: 평균 지표 (교차 검증 사용 시)
        model_file: 모델 파일 경로
    """
    # 저장 경로 설정
    history_file = os.path.join(os.path.dirname(model_file), 'training_history.json')
    
    # 훈련 히스토리 데이터
    history_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': TRAIN_CONFIG,
        'fold_results': fold_results
    }
    
    if avg_metrics:
        history_data['avg_metrics'] = avg_metrics
    
    # JSON 파일로 저장
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"훈련 히스토리 저장 완료: {history_file}")
    
    # 학습 곡선 그래프 생성 및 저장
    _plot_learning_curves(fold_results, model_file)

def _plot_learning_curves(fold_results, model_file):
    """
    학습 곡선 그래프 생성 및 저장
    
    Args:
        fold_results: 폴드별 훈련 결과
        model_file: 모델 파일 경로
    """
    # 저장 경로 설정
    plot_file = os.path.join(os.path.dirname(model_file), 'learning_curves.png')
    
    plt.figure(figsize=(15, 10))
    
    # 손실 그래프
    plt.subplot(2, 2, 1)
    for i, result in enumerate(fold_results):
        plt.plot(result['train_losses'], label=f'Fold {i+1} Train' if len(fold_results) > 1 else 'Train')
        plt.plot(result['val_losses'], label=f'Fold {i+1} Validation' if len(fold_results) > 1 else 'Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 정확도 그래프
    plt.subplot(2, 2, 2)
    for i, result in enumerate(fold_results):
        plt.plot(result['val_accuracies'], label=f'Fold {i+1}' if len(fold_results) > 1 else 'Validation')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # F1 그래프
    plt.subplot(2, 2, 3)
    for i, result in enumerate(fold_results):
        plt.plot(result['val_f1s'], label=f'Fold {i+1}' if len(fold_results) > 1 else 'Validation')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # 저장
    plt.tight_layout()
    plt.savefig(plot_file)
    logger.info(f"학습 곡선 저장 완료: {plot_file}")

if __name__ == "__main__":
    # 단독 실행 시 테스트
    logging.basicConfig(level=logging.INFO)
    train_model('data/sample_reviews.csv')
