#!/usr/bin/env python3
"""
부동산 거주후기 리뷰 분류 시스템 2.0 (ReviewClassifierPro)
- 문장 단위 분석, KoELECTRA 모델, 앙상블 기법 등을 활용한 개선된 버전
- 정상 리뷰 내 부분적 비정상 텍스트 감지 능력 강화
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('review_classifier.log')
    ]
)
logger = logging.getLogger("review-classifier-pro")

def train_command(args):
    """모델 훈련 명령"""
    from src.train import train_model
    
    logger.info(f"모델 훈련 시작: {args.data_file}")
    train_model(
        data_file=args.data_file,
        model_file=args.model_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_early_stopping=args.early_stopping,
        cross_validation=args.cross_validation
    )

def predict_command(args):
    """예측 명령"""
    from src.predict import predict_text, predict_batch
    
    if args.text:
        # 단일 텍스트 예측
        logger.info(f"단일 텍스트 예측: {args.text[:50]}...")
        result = predict_text(
            text=args.text,
            time=args.time,
            model_file=args.model_file,
            detailed=args.detailed
        )
        
        # 결과 출력
        print("\n예측 결과:")
        print(f"텍스트: {result['text'][:70]}..." if len(result['text']) > 70 else f"텍스트: {result['text']}")
        print(f"정상 여부: {'정상' if result['is_normal'] else '비정상'}")
        print(f"신뢰도: {result['confidence']:.4f}")
        
        if not result['is_normal'] and 'abnormal_factors' in result:
            print("\n비정상 요소:")
            for factor in result['abnormal_factors']:
                print(f"- {factor}")
        
        if args.detailed and 'segment_results' in result:
            print("\n문장 단위 분석 결과:")
            for i, segment in enumerate(result['segment_results'], 1):
                print(f"문장 {i}: \"{segment['sentence'][:50]}...\"" if len(segment['sentence']) > 50 else f"문장 {i}: \"{segment['sentence']}\"")
                print(f"  - 비정상 점수: {segment['abnormal_score']:.2f}")
                print(f"  - 판정: {'비정상' if segment['is_abnormal'] else '정상'}")
                if segment['is_abnormal'] and 'abnormal_factors' in segment:
                    print(f"  - 비정상 요소: {', '.join(segment['abnormal_factors'])}")
    elif args.file:
        # 배치 예측
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            args.output = f"results_prediction_{timestamp}.csv"
        
        logger.info(f"배치 예측 시작: {args.file} -> {args.output}")
        predict_batch(
            input_file=args.file,
            output_file=args.output,
            model_file=args.model_file
        )
        print(f"예측 결과가 {args.output}에 저장되었습니다.")
    else:
        print("오류: 텍스트나 파일 중 하나를 지정해야 합니다.")

def analyze_command(args):
    """문장 단위 분석 명령"""
    from src.segment_analyzer import SegmentAnalyzer
    from src.preprocess import TextPreprocessor
    from src.advanced_features import AdvancedFeatureExtractor
    
    # 전처리기 및 특성 추출기 초기화
    preprocessor = TextPreprocessor()
    feature_extractor = AdvancedFeatureExtractor()
    
    # 문장 단위 분석기 초기화
    analyzer = SegmentAnalyzer(
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        threshold=args.threshold
    )
    
    # 분석 수행
    result = analyzer.analyze(args.text)
    
    # 결과 출력
    print("\n문장 단위 분석 결과:")
    for i, segment in enumerate(result['segment_results'], 1):
        print(f"문장 {i}: \"{segment['sentence'][:50]}...\"" if len(segment['sentence']) > 50 else f"문장 {i}: \"{segment['sentence']}\"")
        print(f"  - 비정상 점수: {segment['abnormal_score']:.2f}")
        print(f"  - 판정: {'비정상' if segment['is_abnormal'] else '정상'}")
        if segment['is_abnormal'] and 'abnormal_factors' in segment:
            print(f"  - 비정상 요소: {', '.join(segment['abnormal_factors'])}")
    
    print(f"\n종합 판정: {'비정상' if result['is_abnormal'] else '정상'} 리뷰")
    print(f"최대 비정상 점수: {result['max_abnormal_score']:.2f}")
    print(f"비정상 문장 비율: {result['abnormal_ratio']*100:.1f}%")

def evaluate_command(args):
    """평가 명령"""
    from src.evaluate import evaluate_model, evaluate_rule_based, evaluate_ensemble
    
    if args.rule_based:
        # 규칙 기반만 평가
        logger.info(f"규칙 기반 평가: {args.data_file}")
        evaluate_rule_based(
            data_file=args.data_file,
            output_dir=args.output_dir
        )
    elif args.ensemble:
        # 앙상블 평가
        logger.info(f"앙상블 평가: {args.data_file}")
        evaluate_ensemble(
            data_file=args.data_file,
            model_file=args.model_file,
            output_dir=args.output_dir,
            model_weight=args.model_weight,
            rule_weight=args.rule_weight,
            segment_weight=args.segment_weight
        )
    else:
        # 모델 평가
        logger.info(f"모델 평가: {args.data_file}")
        evaluate_model(
            data_file=args.data_file,
            model_file=args.model_file,
            output_dir=args.output_dir
        )

def generate_data_command(args):
    """데이터 생성 명령"""
    from src.data_generator import generate_enhanced_dataset
    
    logger.info(f"강화된 데이터 생성 시작: {args.num_samples}개")
    
    # 강화된 데이터 생성
    generate_enhanced_dataset(
        num_samples=args.num_samples,
        normal_ratio=args.normal_ratio,
        output_file=args.output_file,
        spam_level=args.spam_level,
        weird_chars_prob=args.weird_chars_prob
    )

def pattern_command(args):
    """패턴 관리 명령"""
    from tools.pattern_manager import PatternManager
    
    # 패턴 관리자 초기화
    manager = PatternManager(pattern_file=args.pattern_file)
    
    # 명령어에 따라 실행
    if args.list:
        manager.list_patterns(group=args.group, detailed=args.detailed)
    elif args.add:
        manager.add_pattern(args.add, args.group)
    elif args.add_file:
        manager.add_patterns_from_file(args.add_file, args.group)
    elif args.remove:
        manager.remove_pattern(args.remove)
    elif args.test:
        manager.test_pattern(args.test)
    elif args.test_file:
        manager.test_patterns_from_file(args.test_file)
    elif args.export:
        manager.export_patterns(args.export)
    elif args.import_file:
        manager.import_patterns(args.import_file, args.overwrite)
    elif args.create_default:
        manager.create_default_patterns()
        print("기본 패턴이 생성되었습니다.")
    else:
        print("패턴 관리 명령을 지정해주세요. 도움말은 'pattern -h'를 참조하세요.")

def server_command(args):
    """서버 실행 명령"""
    from api.app import start_server
    
    logger.info(f"API 서버 시작: {args.host}:{args.port}")
    start_server(
        host=args.host,
        port=args.port,
        reload=args.reload
    )

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='부동산 거주후기 리뷰 분류 시스템 2.0')
    subparsers = parser.add_subparsers(dest='command', help='명령')
    
    # 훈련 명령
    train_parser = subparsers.add_parser('train', help='모델 훈련')
    train_parser.add_argument('data_file', type=str, help='훈련 데이터 파일')
    train_parser.add_argument('--model_file', type=str, default='models/koelectra_classifier.pt', help='모델 저장 경로')
    train_parser.add_argument('--epochs', type=int, default=5, help='학습 에폭')
    train_parser.add_argument('--batch_size', type=int, default=16, help='배치 크기')
    train_parser.add_argument('--learning_rate', type=float, default=2e-5, help='학습률')
    train_parser.add_argument('--early_stopping', action='store_true', help='조기 종료 사용')
    train_parser.add_argument('--cross_validation', action='store_true', help='교차 검증 사용')
    train_parser.set_defaults(func=train_command)
    
    # 예측 명령
    predict_parser = subparsers.add_parser('predict', help='리뷰 예측')
    predict_parser.add_argument('--text', type=str, help='예측할 텍스트')
    predict_parser.add_argument('--file', type=str, help='예측할 파일')
    predict_parser.add_argument('--time', type=str, help='작성 시간 (형식: YYYY-MM-DD HH:MM:SS)')
    predict_parser.add_argument('--output', type=str, help='결과 출력 파일')
    predict_parser.add_argument('--model_file', type=str, default='models/koelectra_classifier.pt', help='모델 파일 경로')
    predict_parser.add_argument('--detailed', action='store_true', help='상세 결과 출력')
    predict_parser.set_defaults(func=predict_command)
    
    # 문장 단위 분석 명령
    analyze_parser = subparsers.add_parser('analyze', help='문장 단위 분석')
    analyze_parser.add_argument('--text', type=str, required=True, help='분석할 텍스트')
    analyze_parser.add_argument('--threshold', type=float, default=0.6, help='비정상 판정 임계값')
    analyze_parser.set_defaults(func=analyze_command)
    
    # 평가 명령
    evaluate_parser = subparsers.add_parser('evaluate', help='모델 평가')
    evaluate_parser.add_argument('data_file', type=str, help='평가 데이터 파일')
    evaluate_parser.add_argument('--model_file', type=str, default='models/koelectra_classifier.pt', help='모델 파일 경로')
    evaluate_parser.add_argument('--rule_based', action='store_true', help='규칙 기반만 평가')
    evaluate_parser.add_argument('--ensemble', action='store_true', help='앙상블 평가')
    evaluate_parser.add_argument('--output_dir', type=str, default='results', help='결과 출력 디렉토리')
    evaluate_parser.add_argument('--model_weight', type=float, default=0.5, help='모델 가중치')
    evaluate_parser.add_argument('--rule_weight', type=float, default=0.3, help='규칙 가중치')
    evaluate_parser.add_argument('--segment_weight', type=float, default=0.2, help='세그먼트 가중치')
    evaluate_parser.set_defaults(func=evaluate_command)
    
    # 데이터 생성 명령
    generate_parser = subparsers.add_parser('generate', help='강화된 데이터 생성')
    generate_parser.add_argument('--num_samples', type=int, default=5000, help='생성할 샘플 수')
    generate_parser.add_argument('--normal_ratio', type=float, default=0.7, help='정상 리뷰 비율 (0.0 ~ 1.0)')
    generate_parser.add_argument('--output_file', type=str, default='data/enhanced_dataset.csv', help='출력 파일 경로')
    generate_parser.add_argument('--spam_level', type=str, choices=['low', 'medium', 'high'], default='medium', 
                              help='스팸 텍스트 삽입 수준 (low: 문장 끝에만, medium: 랜덤 위치 1곳, high: 여러 위치)')
    generate_parser.add_argument('--weird_chars_prob', type=float, default=0.3, 
                              help='이상한 문자 추가 확률 (0.0 ~ 1.0)')
    generate_parser.set_defaults(func=generate_data_command)
    
    # 패턴 관리 명령
    pattern_parser = subparsers.add_parser('pattern', help='패턴 관리')
    pattern_parser.add_argument('--pattern_file', type=str, default='models/spam_patterns.json', help='패턴 파일 경로')
    pattern_parser.add_argument('--list', action='store_true', help='패턴 목록 조회')
    pattern_parser.add_argument('--group', type=str, help='패턴 그룹')
    pattern_parser.add_argument('--detailed', action='store_true', help='상세 정보 표시')
    pattern_parser.add_argument('--add', type=str, help='패턴 추가')
    pattern_parser.add_argument('--add_file', type=str, help='파일에서 패턴 추가')
    pattern_parser.add_argument('--remove', type=str, help='패턴 삭제')
    pattern_parser.add_argument('--test', type=str, help='패턴 테스트')
    pattern_parser.add_argument('--test_file', type=str, help='파일에서 패턴 테스트')
    pattern_parser.add_argument('--export', type=str, help='패턴 내보내기')
    pattern_parser.add_argument('--import_file', type=str, help='패턴 가져오기')
    pattern_parser.add_argument('--overwrite', action='store_true', help='기존 패턴 덮어쓰기')
    pattern_parser.add_argument('--create_default', action='store_true', help='기본 패턴 생성')
    pattern_parser.set_defaults(func=pattern_command)
    
    # 서버 실행 명령
    server_parser = subparsers.add_parser('server', help='API 서버 실행')
    server_parser.add_argument('--host', type=str, default='0.0.0.0', help='서버 호스트')
    server_parser.add_argument('--port', type=int, default=8000, help='서버 포트')
    server_parser.add_argument('--reload', action='store_true', help='코드 변경 시 자동 리로드')
    server_parser.set_defaults(func=server_command)
    
    # 명령행 인자 파싱
    args = parser.parse_args()
    
    # 기본 디렉토리 생성
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 명령 실행
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
