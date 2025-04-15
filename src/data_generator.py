"""
강화된 데이터 생성기
- 다양한 비정상 패턴을 포함한 학습 데이터 생성
- 정상 리뷰에 스팸 텍스트 삽입 기능
"""

import os
import random
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys

# 상위 디렉토리 추가 (상대 임포트를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPAM_TEXTS, FEATURES

logger = logging.getLogger("data-generator")

# 부동산 거주후기 관련 단어 사전
RESIDENCE_DICT = {
    "building_type": ["아파트", "빌라", "오피스텔", "주택", "원룸", "투룸", "쓰리룸", "오피스텔", "도시형생활주택", "단독주택", "다세대주택"],
    "environment": ["소음", "방음", "단열", "환기", "채광", "통풍", "결로", "곰팡이", "해충", "바퀴벌레", "햇빛", "환경", "사생활", "보안"],
    "facility": ["엘리베이터", "주차장", "현관", "CCTV", "인터폰", "난방", "에어컨", "수도", "전기", "가스", "인터넷", "관리실", "경비실", "택배함"],
    "location": ["역", "버스정류장", "지하철", "마트", "편의점", "병원", "학교", "공원", "주변", "상가", "접근성", "도로", "교통"],
    "cost": ["관리비", "월세", "전세", "보증금", "난방비", "전기세", "수도세", "가스비", "인터넷비", "주차비"],
    
    "quality": ["좋", "훌륭", "괜찮", "나쁘", "별로", "최악", "적당", "만족스럽", "불만족스럽", "쾌적", "불쾌", "조용", "시끄럽", "따뜻", "춥", "쾌적"],
    "management": ["잘", "제대로", "엉망", "깔끔", "지저분", "청결", "불결", "신속", "느리", "정확", "부정확", "친절", "불친절", "철저", "허술"],
    "accessibility": ["좋", "편리", "불편", "가깝", "멀", "뛰어나", "떨어지", "편", "불편", "만족스럽", "불만족스럽", "훌륭", "나쁘"],
    "price_opinion": ["적당", "합리적", "비싸", "저렴", "부담스럽", "감당할 만", "과도", "적정", "만족스럽", "불만족스럽"],
    
    "advantage": [
        "방음이 잘 되어 있어요",
        "난방이 잘 돼요",
        "채광이 좋아요",
        "관리가 잘 되고 있어요",
        "교통이 편리해요",
        "주변 시설이 잘 갖춰져 있어요",
        "주차 공간이 충분해요",
        "보안이 철저해요",
        "넓고 쾌적해요",
        "환기가 잘 돼요"
    ],
    "disadvantage": [
        "소음이 심해요",
        "단열이 잘 안돼요",
        "주차 공간이 부족해요",
        "엘리베이터가 느려요",
        "관리비가 비싸요",
        "햇빛이 잘 안 들어와요",
        "결로 현상이 있어요",
        "곰팡이가 잘 생겨요",
        "방이 좁아요",
        "주변에 편의 시설이 적어요"
    ]
}

# 정상 리뷰 템플릿
NORMAL_REVIEW_TEMPLATES = [
    "{building_type} {quality}한 편이에요. {facility} {management}게 관리되고 있어요. {advantage}. 다만 {disadvantage} 점은 아쉽습니다.",
    "이 {building_type}은 {quality}고 {facility}도 {management}해요. {advantage}, 하지만 {disadvantage}.",
    "{quality}은 {building_type}입니다. {facility}가 {management}고, {advantage}. 단점은 {disadvantage}입니다.",
    "여기 살면서 좋은 점은 {advantage}이고, 아쉬운 점은 {disadvantage}예요. 전반적으로 {quality}은 {building_type}입니다.",
    "{building_type}이 {quality}고 {facility}도 {management}한 편이에요. {advantage}, 그래도 {disadvantage} 점이 좀 아쉬워요."
]

# 이상한 문자 패턴
WEIRD_CHARS = [
    "ㄴㅇㄹ", "ㅋㅋㅋㅋ", "ㅎㅎㅎㅎ", "ㅡㅡ", "ㅠㅠㅠ", 
    ";;;", "...", "!!!!", "???", ",,,,,", 
    "~~~~~", "_____", "-----"
]

def generate_base_reviews(num_samples):
    """
    기본 정상 리뷰 생성
    
    Args:
        num_samples: 생성할 샘플 수
    
    Returns:
        생성된 리뷰 목록
    """
    reviews = []
    
    # 현재 시간 기준
    now = datetime.now()
    
    for i in range(num_samples):
        # 리뷰 템플릿 선택
        template = random.choice(NORMAL_REVIEW_TEMPLATES)
        
        # 템플릿 채우기
        text = fill_template(template, RESIDENCE_DICT)
        
        # 추가 세부 정보로 리뷰 확장
        text = add_more_details(text, min_length=100)
        
        # 현실적인 느낌을 위해 아주 가끔 오타 추가
        if random.random() < 0.1:
            chars = list(text)
            typo_idx = random.randint(0, len(chars) - 1)
            chars[typo_idx] = random.choice("가나다라마바사아자차카타파하")
            text = "".join(chars)
        
        # 메타데이터 생성
        # 리뷰 작성 시간 (최근 30일 내 랜덤)
        days_ago = random.randint(0, 30)
        
        # 정상 시간대
        hours = random.randint(8, 22)  # 정상 시간대 (오전 8시 ~ 오후 10시)
        minutes = random.randint(0, 59)
        created_at = (now - timedelta(days=days_ago)).replace(hour=hours, minute=minutes)
        
        # 사용자 ID
        user_id = f"user_{random.randint(1000, 9999)}"
        
        # 건물 유형
        building_type = random.choice(RESIDENCE_DICT["building_type"])
        
        # 평가 점수 (정규 분포에 가까운 분포)
        weights = [0.05, 0.15, 0.45, 0.25, 0.1]  # 대략적인 정규 분포 가중치
        rating = random.choices([1, 2, 3, 4, 5], weights=weights)[0]
        
        # 리뷰 길이
        length = len(text)
        
        # 데이터 추가
        reviews.append({
            'text': text,
            'created_at': created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
            'building_type': building_type,
            'rating': rating,
            'length': length
        })
    
    return reviews

def fill_template(template, word_dict):
    """
    템플릿을 채워 리뷰 생성
    
    Args:
        template: 리뷰 템플릿 문자열
        word_dict: 단어 사전
    
    Returns:
        생성된 리뷰 텍스트
    """
    # 템플릿에서 모든 변수 찾기
    variables = re.findall(r'\{([^}]+)\}', template)
    
    # 변수를 실제 단어로 대체
    result = template
    
    for var in variables:
        if var in word_dict:
            replacement = random.choice(word_dict[var])
            result = result.replace(f"{{{var}}}", replacement)
    
    return result

def add_more_details(review, min_length=100):
    """
    리뷰에 추가 세부 정보 추가 (최소 길이 충족을 위함)
    
    Args:
        review: 기본 리뷰 텍스트
        min_length: 최소 길이
    
    Returns:
        세부 정보가 추가된 리뷰 텍스트
    """
    if len(review) >= min_length:
        return review
    
    additional_details = [
        f"창문은 {random.choice(['크', '작', '적당한 크기'])}고 {random.choice(['환기가 잘 돼요', '채광이 좋아요', '뷰가 좋아요', '소음이 잘 차단돼요'])}.",
        f"화장실은 {random.choice(['넓', '작', '적당한 크기'])}고 {random.choice(['깨끗해요', '환기가 잘 돼요', '수압이 좋아요', '온수가 잘 나와요'])}.",
        f"부엌은 {random.choice(['넓', '작', '적당한 크기'])}고 {random.choice(['사용하기 편해요', '수납공간이 많아요', '환기가 잘 돼요', '깨끗해요'])}.",
        f"방은 {random.choice(['넓', '작', '적당한 크기'])}고 {random.choice(['채광이 좋아요', '수납공간이 많아요', '따뜻해요', '시원해요'])}.",
        f"복도는 {random.choice(['넓', '좁', '적당한 크기'])}고 {random.choice(['깨끗해요', '조용해요', '밝아요', '안전해요'])}.",
        f"주차장은 {random.choice(['넓', '좁', '적당한 크기'])}고 {random.choice(['주차하기 편해요', '항상 자리가 있어요', '안전해요', '밝아요'])}.",
        f"이웃들은 {random.choice(['조용', '시끄럽', '친절', '무관심'])}해요.",
        f"관리인은 {random.choice(['친절', '불친절', '책임감 있', '무책임'])}해요.",
        f"택배 보관은 {random.choice(['편리', '불편', '안전', '걱정'])}해요.",
        f"여름에는 {random.choice(['시원', '덥', '쾌적', '불쾌'])}해요.",
        f"겨울에는 {random.choice(['따뜻', '춥', '쾌적', '불쾌'])}해요."
    ]
    
    # 리뷰에 추가 세부 정보 더하기
    while len(review) < min_length and additional_details:
        detail = random.choice(additional_details)
        additional_details.remove(detail)
        review += " " + detail
    
    return review

def insert_spam_text(text, position='middle'):
    """
    정상 리뷰에 스팸 텍스트 삽입
    
    Args:
        text: 정상 리뷰 텍스트
        position: 삽입 위치 ('start', 'middle', 'end', 'random')
    
    Returns:
        스팸 텍스트가 삽입된 리뷰
    """
    # 스팸 텍스트 선택
    spam_text = random.choice(SPAM_TEXTS)
    
    # 위치에 따라 삽입
    if position == 'start':
        return spam_text + " " + text
    elif position == 'end':
        return text + " " + spam_text
    elif position == 'middle':
        # 문장 분리
        sentences = re.split(r'([.!?]\s)', text)
        sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''])]
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            return text
        
        # 중간에 삽입
        insert_position = len(sentences) // 2
        
        result = []
        for i, sentence in enumerate(sentences):
            result.append(sentence)
            if i == insert_position:
                result.append(spam_text)
        
        return " ".join(result)
    else:  # random
        # 문장 분리
        sentences = re.split(r'([.!?]\s)', text)
        sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''])]
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            return text
        
        # 랜덤 위치에 삽입
        insert_position = random.randint(0, len(sentences))
        
        result = []
        for i, sentence in enumerate(sentences):
            if i == insert_position:
                result.append(spam_text)
            result.append(sentence)
        
        # 마지막 위치인 경우
        if insert_position == len(sentences):
            result.append(spam_text)
        
        return " ".join(result)

def insert_repetition_pattern(text):
    """
    반복 패턴 삽입
    
    Args:
        text: 정상 리뷰 텍스트
    
    Returns:
        반복 패턴이 삽입된 리뷰
    """
    # 반복할 단어/구 선택
    repeat_words = ["좋아요", "싫어요", "그냥그래요", "괜찮아요", "별로에요", "최악이에요"]
    repeat_word = random.choice(repeat_words)
    
    # 반복 횟수
    repeat_count = random.randint(3, 8)
    
    # 반복 패턴 생성
    repetition = (repeat_word + " ") * repeat_count
    
    # 랜덤 위치에 삽입
    sentences = text.split(". ")
    if not sentences:
        return text
    
    insert_position = random.randint(0, len(sentences) - 1)
    sentences[insert_position] = sentences[insert_position] + " " + repetition
    
    return ". ".join(sentences)

def insert_jamo_text(text):
    """
    자소 분리 텍스트 삽입 (ㄴㅇㄹ 등)
    
    Args:
        text: 정상 리뷰 텍스트
    
    Returns:
        자소 분리 텍스트가 삽입된 리뷰
    """
    # 자소 분리 텍스트 생성
    jamo_patterns = ["ㄴㅇㄹ", "ㅂㄷㅈ", "ㄱㅅㄴ", "ㅁㄴㅇㄹ", "ㅎㄴㄱ", "ㅈㄱㄷ"]
    jamo_text = random.choice(jamo_patterns) * random.randint(1, 3)
    
    # 랜덤 위치에 삽입
    words = text.split()
    if not words:
        return text
    
    insert_position = random.randint(0, len(words) - 1)
    words[insert_position] = words[insert_position] + " " + jamo_text
    
    return " ".join(words)

def insert_filler_text(text):
    """
    글자수 채우기 텍스트 삽입
    
    Args:
        text: 정상 리뷰 텍스트
    
    Returns:
        글자수 채우기 텍스트가 삽입된 리뷰
    """
    # 글자수 채우기 텍스트
    filler_texts = [
        "글자수 채우기용 텍스트입니다.",
        "100자 이상 써야 해서 이렇게 적어요.",
        "더 적을 내용이 없어서 그냥 쓰는 중이에요.",
        "리뷰 글자수 채우기 위한 내용입니다.",
        "특별히 할 말은 없지만 글자수를 맞추려고요."
    ]
    filler_text = random.choice(filler_texts)
    
    # 문장 끝에 추가
    return text + " " + filler_text

def insert_weird_punctuation(text):
    """
    이상한 구두점 패턴 삽입
    
    Args:
        text: 정상 리뷰 텍스트
    
    Returns:
        이상한 구두점 패턴이 삽입된 리뷰
    """
    # 이상한 구두점 패턴
    weird_punct = random.choice(["...", "!!!", "???", ",,,,", "~~~~", "-----"])
    
    # 랜덤 위치에 삽입
    words = text.split()
    if not words:
        return text
    
    insert_position = random.randint(0, len(words) - 1)
    words[insert_position] = words[insert_position] + weird_punct
    
    return " ".join(words)

def insert_english_text(text):
    """
    영어 텍스트 삽입
    
    Args:
        text: 정상 리뷰 텍스트
    
    Returns:
        영어 텍스트가 삽입된 리뷰
    """
    # 영어 텍스트
    english_texts = [
        "I don't know what to write",
        "Just filling up the characters",
        "This is so annoying",
        "I hate writing reviews",
        "Why do I need to write so much"
    ]
    english_text = random.choice(english_texts)
    
    # 랜덤 위치에 삽입
    sentences = text.split(". ")
    if not sentences:
        return text
    
    insert_position = random.randint(0, len(sentences) - 1)
    sentences[insert_position] = sentences[insert_position] + " " + english_text
    
    return ". ".join(sentences)

def insert_irrelevant_text(text):
    """
    문맥 무관 텍스트 삽입
    
    Args:
        text: 정상 리뷰 텍스트
    
    Returns:
        문맥 무관 텍스트가 삽입된 리뷰
    """
    # 문맥 무관 텍스트
    irrelevant_texts = [
        "동해물과 백두산이 마르고 닳도록",
        "까치 까치 설날은 어저께고요 매화 매화 피는 건 올해에요",
        "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679",
        "abcdefghijklmnopqrstuvwxyz",
        "가나다라마바사아자차카타파하"
    ]
    irrelevant_text = random.choice(irrelevant_texts)
    
    # 랜덤 위치에 삽입
    sentences = text.split(". ")
    if not sentences:
        return text
    
    insert_position = random.randint(0, len(sentences) - 1)
    sentences[insert_position] = sentences[insert_position] + " " + irrelevant_text
    
    return ". ".join(sentences)

def apply_multiple_abnormal_patterns(text):
    """
    여러 비정상 패턴 적용
    
    Args:
        text: 정상 리뷰 텍스트
    
    Returns:
        여러 비정상 패턴이 적용된 리뷰
    """
    # 적용할 패턴 수 (2~3개)
    num_patterns = random.randint(2, 3)
    
    # 사용 가능한 패턴 함수들
    pattern_functions = [
        insert_spam_text,
        insert_repetition_pattern,
        insert_jamo_text,
        insert_filler_text,
        insert_weird_punctuation,
        insert_english_text,
        insert_irrelevant_text
    ]
    
    # 선택된 패턴 함수들
    selected_functions = random.sample(pattern_functions, num_patterns)
    
    # 순차적으로 패턴 적용
    modified_text = text
    for func in selected_functions:
        modified_text = func(modified_text)
    
    return modified_text

def generate_enhanced_dataset(num_samples=5000, normal_ratio=0.7, output_file='data/enhanced_dataset.csv', spam_level='medium', weird_chars_prob=0.3):
    """
    강화된 학습 데이터셋 생성
    
    Args:
        num_samples: 생성할 총 샘플 수
        normal_ratio: 정상 리뷰 비율
        output_file: 출력 파일 경로
        spam_level: 스팸 텍스트 삽입 수준 ('low', 'medium', 'high')
        weird_chars_prob: 이상한 문자 추가 확률 (0.0 ~ 1.0)
    
    Returns:
        생성된 데이터프레임
    """
    logger.info(f"강화된 학습 데이터 생성 시작: {num_samples}개 샘플")
    
    # 기본 데이터 생성 (전부 정상 리뷰로 먼저 생성)
    base_reviews = generate_base_reviews(num_samples)
    
    # 일부를 비정상 리뷰로 변환
    abnormal_count = int(num_samples * (1 - normal_ratio))
    abnormal_indices = random.sample(range(num_samples), abnormal_count)
    
    # 스팸 수준에 따른 삽입 함수 결정
    if spam_level == 'low':
        spam_insert_func = lambda text: insert_spam_text(text, 'end')
    elif spam_level == 'medium':
        spam_insert_func = lambda text: insert_spam_text(text, 'middle')
    else:  # high
        spam_insert_func = apply_multiple_abnormal_patterns
    
    # 변환 적용
    enhanced_data = []
    
    for i, review in enumerate(base_reviews):
        if i in abnormal_indices:
            # 비정상 패턴 적용
            modified_text = spam_insert_func(review['text'])
            
            # 이상한 문자 추가
            if random.random() < weird_chars_prob:
                weird_chars = random.choice(WEIRD_CHARS)
                position = random.randint(0, len(modified_text))
                modified_text = modified_text[:position] + " " + weird_chars + " " + modified_text[position:]
            
            # 행 업데이트
            review_dict = review.copy()
            review_dict['text'] = modified_text
            review_dict['is_abnormal'] = 1  # 비정상으로 표시
            enhanced_data.append(review_dict)
        else:
            # 정상 리뷰는 그대로 유지
            review_dict = review.copy()
            review_dict['is_abnormal'] = 0  # 정상으로 표시
            enhanced_data.append(review_dict)
    
    # 데이터프레임 생성
    df = pd.DataFrame(enhanced_data)
    
    # 출력 디렉토리 확인
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # CSV 파일로 저장
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    # 결과 출력
    normal_count = sum(1 for row in enhanced_data if row['is_abnormal'] == 0)
    abnormal_count = len(enhanced_data) - normal_count
    
    logger.info(f"강화된 학습 데이터 생성 완료:")
    logger.info(f"- 총 샘플 수: {len(enhanced_data)}")
    logger.info(f"- 정상 리뷰: {normal_count}개 ({normal_count/len(enhanced_data)*100:.1f}%)")
    logger.info(f"- 비정상 리뷰: {abnormal_count}개 ({abnormal_count/len(enhanced_data)*100:.1f}%)")
    logger.info(f"- 저장 위치: {output_file}")
    
    return df

if __name__ == "__main__":
    # 단독 실행 시 테스트
    generate_enhanced_dataset(num_samples=100, output_file='data/test_dataset.csv')
