# services/generative_agent/infrastructure/sentiment_sources/stocktwits_adapter.py

from typing import List, Dict, Any, Optional
from datetime import datetime
import aiohttp
from ...domain.interfaces.sentiment_protocols import SocialMediaSource, SocialPost
from ...domain.exceptions import DataSourceError
from dataclasses import dataclass
from enum import Enum


class StockTwitsSentiment(Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"


@dataclass
class StockTwitsConfig:
    """Configuration for StockTwits API"""

    base_url: str = "https://api.stocktwits.com/api/2"
    max_messages_per_request: int = 30
    request_timeout: int = 30
    rate_limit_pause: float = 1.0  # seconds between requests


class StockTwitsAdapter(SocialMediaSource):
    """
    Adapter for StockTwits social sentiment data.
    Uses the public StockTwits API which doesn't require authentication for basic endpoints.
    """

    def __init__(self, config: Optional[StockTwitsConfig] = None):
        self.config = config or StockTwitsConfig()
        self._session: Optional[aiohttp.ClientSession] = None

    async def get_social_data(
        self, symbols: List[str], start_date: datetime, end_date: datetime
    ) -> List[SocialPost]:
        """Get social sentiment data from StockTwits"""
        try:
            posts = []
            async with aiohttp.ClientSession() as session:
                self._session = session
                for symbol in symbols:
                    # Get symbol streams
                    symbol_posts = await self._get_symbol_stream(symbol)
                    # Filter and convert posts
                    for post in symbol_posts:
                        if self._is_within_timerange(post, start_date, end_date):
                            processed_post = await self._convert_to_social_post(
                                post, symbol
                            )
                            if processed_post:
                                posts.append(processed_post)
            return posts
        except Exception as e:
            raise DataSourceError(f"Error fetching StockTwits data: {str(e)}")
        finally:
            self._session = None

    async def validate_source(self) -> bool:
        """Validate StockTwits API availability"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.base_url}/streams/symbol/SPY"
                ) as response:
                    return response.status == 200
        except Exception:
            return False

    def get_supported_platforms(self) -> List[str]:
        """Get supported platforms"""
        return ["StockTwits"]

    async def _get_symbol_stream(self, symbol: str) -> List[Dict[str, Any]]:
        """Get message stream for a symbol"""
        if not self._session:
            raise DataSourceError("No active session")

        messages = []
        try:
            url = f"{self.config.base_url}/streams/symbol/{symbol}"
            async with self._session.get(
                url, params={"limit": self.config.max_messages_per_request}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    messages = data.get("messages", [])
                elif response.status == 429:
                    raise DataSourceError("Rate limit exceeded")
                else:
                    raise DataSourceError(
                        f"Failed to fetch StockTwits data: {response.status}"
                    )
        except aiohttp.ClientError as e:
            raise DataSourceError(f"Network error: {str(e)}")

        return messages

    async def _get_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment data for a symbol"""
        if not self._session:
            raise DataSourceError("No active session")

        try:
            url = f"{self.config.base_url}/symbols/{symbol}/sentiment"
            async with self._session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
        except Exception:
            return {}

    def _is_within_timerange(
        self, post: Dict[str, Any], start_date: datetime, end_date: datetime
    ) -> bool:
        """Check if post is within specified timerange"""
        try:
            post_date = datetime.fromisoformat(
                post["created_at"].replace("Z", "+00:00")
            )
            return start_date <= post_date <= end_date
        except (KeyError, ValueError):
            return False

    async def _convert_to_social_post(
        self, raw_post: Dict[str, Any], symbol: str
    ) -> Optional[SocialPost]:
        """Convert StockTwits post to SocialPost"""
        try:
            # Extract sentiment
            sentiment_score = self._calculate_sentiment_score(raw_post)

            # Extract engagement metrics
            engagement_metrics = {
                "likes": raw_post.get("likes", {}).get("total", 0),
                "reshares": raw_post.get("reshares", {}).get("total", 0),
                "replies": raw_post.get("conversation", {}).get("total", 0),
            }

            # Calculate relevance
            relevance_score = self._calculate_relevance(raw_post, symbol)

            return SocialPost(
                source="StockTwits",
                platform="StockTwits",
                content=raw_post.get("body", ""),
                author=raw_post.get("user", {}).get("username", "unknown"),
                timestamp=datetime.fromisoformat(
                    raw_post["created_at"].replace("Z", "+00:00")
                ),
                engagement_metrics=engagement_metrics,
                symbols=[symbol],
                sentiment_score=sentiment_score,
                relevance_score=relevance_score,
            )
        except (KeyError, ValueError):
            return None

    def _calculate_sentiment_score(self, post: Dict[str, Any]) -> float:
        """Calculate sentiment score from StockTwits post"""
        sentiment = post.get("entities", {}).get("sentiment", {}).get("basic", "")

        if sentiment == StockTwitsSentiment.BULLISH.value:
            return 1.0
        elif sentiment == StockTwitsSentiment.BEARISH.value:
            return -1.0
        return 0.0

    def _calculate_relevance(self, post: Dict[str, Any], symbol: str) -> float:
        """Calculate relevance score for post"""
        relevance = 0.0

        # Check if symbol is mentioned in cashtags
        cashtags = post.get("symbols", [])
        if any(tag.get("symbol") == symbol for tag in cashtags):
            relevance += 0.6

        # Check user metrics for credibility
        user = post.get("user", {})
        followers = user.get("followers", 0)
        following = user.get("following", 0)
        ideas = user.get("ideas", 0)

        # Calculate user credibility score
        if followers > 1000 and ideas > 100:
            relevance += 0.2
        elif followers > 500 and ideas > 50:
            relevance += 0.1

        # Check engagement
        likes = post.get("likes", {}).get("total", 0)
        reshares = post.get("reshares", {}).get("total", 0)

        if likes > 20 or reshares > 5:
            relevance += 0.2

        return min(relevance, 1.0)

    async def get_symbol_sentiment_summary(self, symbol: str) -> Dict[str, float]:
        """Get overall sentiment summary for a symbol"""
        sentiment_data = await self._get_sentiment_data(symbol)

        try:
            sentiment = sentiment_data.get("sentiment", {})
            return {
                "bullish_percentage": sentiment.get("bullish", 0) / 100,
                "bearish_percentage": sentiment.get("bearish", 0) / 100,
                "neutral_percentage": sentiment.get("neutral", 0) / 100,
            }
        except (KeyError, TypeError, ZeroDivisionError):
            return {
                "bullish_percentage": 0,
                "bearish_percentage": 0,
                "neutral_percentage": 0,
            }
