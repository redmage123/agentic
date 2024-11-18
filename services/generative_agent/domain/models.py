# services/generative_agent/domain/models.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any

class AnalysisType(Enum):
    """Types of analysis the service can perform"""
    FINANCIAL = "financial"
    RISK = "risk"
    PATTERN = "pattern"
    MARKET = "market"
    COMPOSITE = "composite"

class ConfidenceLevel(Enum):
    """Confidence levels for analysis results"""
    VERY_LOW = "very_low"    # < 20%
    LOW = "low"              # 20-40%
    MODERATE = "moderate"    # 40-60%
    HIGH = "high"            # 60-80%
    VERY_HIGH = "very_high"  # > 80%

@dataclass(frozen=True)
class AnalysisRequest:
    """Incoming request for analysis"""
    request_id: str
    query: str
    analysis_type: AnalysisType
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    required_confidence: Optional[float] = None

@dataclass(frozen=True)
class AnalysisEvidence:
    """Supporting evidence for analysis"""
    source: str
    content: str
    relevance_score: float
    timestamp: datetime

@dataclass(frozen=True)
class ReasoningStep:
    """Individual step in analysis reasoning"""
    step_number: int
    description: str
    conclusion: str
    confidence: float
    evidence: List[AnalysisEvidence]

@dataclass(frozen=True)
class AnalysisResponse:
    """Response containing analysis results"""
    request_id: str
    analysis_type: AnalysisType
    summary: str
    detailed_analysis: str
    reasoning_chain: List[ReasoningStep]
    recommendations: List[str]
    confidence_score: float
    confidence_level: ConfidenceLevel
    metadata: Dict[str, Any]
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass(frozen=True)
class PromptMetadata:
    """Metadata about a prompt"""
    name: str
    description: str
    version: str
    required_context: List[str]
    optional_context: List[str]
    chain_compatible: List[str]

@dataclass(frozen=True)
class PromptChain:
    """Sequence of prompts for analysis"""
    chain_id: str
    prompts: List[str]
    metadata: Dict[str, Any]
    total_tokens: int
