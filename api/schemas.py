"""
API 스키마 정의
- 요청 및 응답 모델
- 데이터 검증
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Any
from datetime import datetime

class ReviewRequest(BaseModel):
    """단일 리뷰 검증 요청"""
    text: str = Field(..., min_length=1, description="검증할 리뷰 텍스트")
    created_at: Optional[str] = Field(None, description="리뷰 작성 시간 (형식: YYYY-MM-DD HH:MM:SS)")
    user_id: Optional[str] = Field(None, description="사용자 ID")
    meta: Optional[Dict[str, Any]] = Field(None, description="추가 메타데이터")
    
    @validator('created_at')
    def validate_created_at(cls, v):
        """작성 시간 유효성 검사"""
        if v is not None:
            try:
                datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                raise ValueError("작성 시간 형식이 잘못되었습니다 (YYYY-MM-DD HH:MM:SS)")
        return v

class ReviewResponse(BaseModel):
    """리뷰 검증 응답"""
    is_normal: bool = Field(..., description="정상 리뷰 여부")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 (0.0 ~ 1.0)")
    abnormal_factors: List[str] = Field([], description="비정상 요소 목록")
    method: str = Field("ensemble", description="사용된 판정 방법")
    processing_time: float = Field(..., description="처리 시간 (초)")

class BatchReviewRequest(BaseModel):
    """일괄 리뷰 검증 요청"""
    reviews: List[ReviewRequest] = Field(..., min_items=1, max_items=100, description="검증할 리뷰 목록")

class BatchReviewResponse(BaseModel):
    """일괄 리뷰 검증 응답"""
    results: List[ReviewResponse] = Field(..., description="리뷰별 검증 결과")
    processing_time: float = Field(..., description="전체 처리 시간 (초)")
    total_reviews: int = Field(..., description="전체 리뷰 수")
    normal_count: int = Field(..., description="정상 리뷰 수")
    abnormal_count: int = Field(..., description="비정상 리뷰 수")

class SegmentInfo(BaseModel):
    """문장 분석 정보"""
    text: str = Field(..., description="문장 텍스트")
    is_abnormal: bool = Field(..., description="비정상 여부")
    abnormal_score: float = Field(..., ge=0.0, le=1.0, description="비정상 점수 (0.0 ~ 1.0)")
    abnormal_factors: List[str] = Field([], description="비정상 요소 목록")

class AnalysisResponse(BaseModel):
    """문장 단위 분석 응답"""
    is_abnormal: bool = Field(..., description="전체 비정상 여부")
    abnormal_score: float = Field(..., ge=0.0, le=1.0, description="최대 비정상 점수")
    abnormal_ratio: float = Field(..., ge=0.0, le=1.0, description="비정상 문장 비율")
    context_coherence: float = Field(1.0, ge=0.0, le=1.0, description="문맥 일관성 점수")
    segments: List[SegmentInfo] = Field(..., description="문장별 분석 결과")
    processing_time: float = Field(..., description="처리 시간 (초)")

class ErrorResponse(BaseModel):
    """오류 응답"""
    error: str = Field(..., description="오류 메시지")
    detail: Optional[str] = Field(None, description="상세 오류 정보")

class HealthResponse(BaseModel):
    """상태 확인 응답"""
    status: str = Field(..., description="시스템 상태")
    version: str = Field(..., description="API 버전")
    model_loaded: bool = Field(..., description="모델 로드 여부")
    uptime: float = Field(..., description="가동 시간 (초)")
    device: Optional[str] = Field(None, description="사용 중인 장치 (CPU/GPU)")
