# services/generative_agent/domain/interfaces/sentiment_protocols.py

from typing import List, Dict, Any, Protocol
from datetime import datetime
from dataclasses import dataclass

@dataclass
class NewsItem:
    """Value object representing a news item"""
    source: str
    title: str
    summary: str
    url: str
    timestamp: datetime
    symbols: List[str]
    sentiment_score: float = 0.0
    relevance_score: float = 0.0

@dataclass
class SocialPost:
    """Value object representing a social media post"""
    source: str
    platform: str
    content: str
    author: str
    timestamp: datetime
    engagement_metrics: Dict[str, int]
    symbols: List[str]
    sentiment_score: float = 0.0
    relevance_score: float = 0.0

@dataclass
class MarketMetrics:
    """Value object representing market sentiment metrics"""
    symbol: str
    timestamp: datetime
    put_call_ratio: float
    trading_volume: int
    price_momentum: float
    volume_trend: float
    technical_indicators: Dict[str, float]

class NewsSource(Protocol):
    """Protocol for news data sources"""
    
    async def get_news(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> List[NewsItem]:
        """Get news articles for given symbols and date range"""
        pass

    async def validate_source(self) -> bool:
        """Validate source availability and credentials"""
        pass

class SocialMediaSource(Protocol):
    """Protocol for social media data sources"""
    
    async def get_social_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> List[SocialPost]:
        """Get social media data for given symbols and date range"""
        pass

    async def validate_source(self) -> bool:
        """Validate source availability and credentials"""
        pass

    def get_supported_platforms(self) -> List[str]:
        """Get list of supported social media platforms"""
        pass

class MarketDataSource(Protocol):
    """Protocol for market sentiment data sources"""
    
    async def get_market_sentiment(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> List[MarketMetrics]:
        """Get market sentiment data for given symbols and date range"""
        pass

    async def validate_source(self) -> bool:
        """Validate source availability and credentials"""
        pass

    def get_supported_metrics(self) -> List[str]:
        """Get list of supported market metrics"""
        pass
