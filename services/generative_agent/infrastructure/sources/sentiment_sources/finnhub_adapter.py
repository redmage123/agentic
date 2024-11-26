# services/generative_agent/infrastructure/sentiment_sources/finnhub_adapter.py

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import aiohttp
import asyncio
from dataclasses import dataclass
from enum import Enum
import json

from ...domain.interfaces.sentiment_protocols import (
    NewsSource,
    MarketDataSource,
    NewsItem,
    MarketMetrics,
)
from ...domain.exceptions import DataSourceError


class FinnhubSentimentCategory(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class FinnhubConfig:
    """Configuration for Finnhub API"""

    api_key: str
    base_url: str = "https://finnhub.io/api/v1"
    request_timeout: int = 30
    rate_limit_per_minute: int = 30  # Free tier limit
    max_retries: int = 3
    retry_delay: float = 1.0


class FinnhubAdapter(NewsSource, MarketDataSource):
    """
    Adapter for Finnhub financial data API.
    Provides access to news, company filings, and market sentiment data.
    """

    def __init__(self, api_key: str, config: Optional[FinnhubConfig] = None):
        self.config = config or FinnhubConfig(api_key=api_key)
        self._session: Optional[aiohttp.ClientSession] = None
        self._request_timestamps: List[datetime] = []
        self._lock = asyncio.Lock()

    async def get_news(
        self, symbols: List[str], start_date: datetime, end_date: datetime
    ) -> List[NewsItem]:
        """Get company news and sentiment from Finnhub"""
        try:
            news_items = []
            async with aiohttp.ClientSession() as session:
                self._session = session
                for symbol in symbols:
                    # Get company news
                    company_news = await self._get_company_news(
                        symbol, start_date, end_date
                    )
                    # Get sentiment data
                    sentiment_data = await self._get_news_sentiment(symbol)

                    # Combine news with sentiment
                    for news in company_news:
                        sentiment_score = self._match_sentiment_score(
                            news, sentiment_data
                        )
                        news_item = await self._convert_to_news_item(
                            news, symbol, sentiment_score
                        )
                        if news_item:
                            news_items.append(news_item)

            return news_items

        except Exception as e:
            raise DataSourceError(f"Error fetching Finnhub news: {str(e)}")
        finally:
            self._session = None

    async def get_market_sentiment(
        self, symbols: List[str], start_date: datetime, end_date: datetime
    ) -> List[MarketMetrics]:
        """Get market sentiment metrics from Finnhub"""
        try:
            metrics = []
            async with aiohttp.ClientSession() as session:
                self._session = session
                for symbol in symbols:
                    # Get recommendation trends
                    recommendations = await self._get_recommendation_trends(symbol)

                    # Get technical indicators
                    technicals = await self._get_technical_indicators(symbol)

                    # Get earnings sentiment
                    earnings = await self._get_earnings_sentiment(symbol)

                    # Combine all metrics
                    metric = await self._create_market_metrics(
                        symbol, recommendations, technicals, earnings
                    )
                    metrics.append(metric)

            return metrics

        except Exception as e:
            raise DataSourceError(f"Error fetching Finnhub market sentiment: {str(e)}")
        finally:
            self._session = None

    async def validate_source(self) -> bool:
        """Validate Finnhub API availability and credentials"""
        try:
            async with aiohttp.ClientSession() as session:
                self._session = session
                # Test with a simple API call
                await self._make_request("/stock/symbol", {"exchange": "US"})
                return True
        except Exception:
            return False
        finally:
            self._session = None

    def get_supported_metrics(self) -> List[str]:
        """Get list of supported market metrics"""
        return [
            "recommendation_trends",
            "technical_indicators",
            "earnings_sentiment",
            "news_sentiment",
            "social_sentiment",
        ]

    async def _make_request(
        self, endpoint: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make rate-limited request to Finnhub API"""
        if not self._session:
            raise DataSourceError("No active session")

        await self._enforce_rate_limit()

        params["token"] = self.config.api_key
        url = f"{self.config.base_url}{endpoint}"

        for attempt in range(self.config.max_retries):
            try:
                async with self._session.get(
                    url, params=params, timeout=self.config.request_timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        # Rate limit exceeded
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    else:
                        raise DataSourceError(f"Finnhub API error: {response.status}")
            except asyncio.TimeoutError:
                if attempt == self.config.max_retries - 1:
                    raise DataSourceError("Finnhub API timeout")
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
            except Exception as e:
                raise DataSourceError(f"Finnhub API request failed: {str(e)}")

    async def _enforce_rate_limit(self):
        """Enforce API rate limits"""
        async with self._lock:
            current_time = datetime.now()

            # Remove timestamps older than 1 minute
            self._request_timestamps = [
                ts
                for ts in self._request_timestamps
                if current_time - ts < timedelta(minutes=1)
            ]

            # Check if we're at the rate limit
            if len(self._request_timestamps) >= self.config.rate_limit_per_minute:
                # Calculate sleep time
                sleep_time = (
                    60 - (current_time - self._request_timestamps[0]).total_seconds()
                )
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            # Add current timestamp
            self._request_timestamps.append(current_time)

    async def _get_company_news(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get company-specific news"""
        params = {
            "symbol": symbol,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
        }
        return await self._make_request("/company-news", params)

    async def _get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment data"""
        params = {"symbol": symbol}
        return await self._make_request("/news-sentiment", params)

    async def _get_recommendation_trends(self, symbol: str) -> List[Dict[str, Any]]:
        """Get analyst recommendation trends"""
        params = {"symbol": symbol}
        return await self._make_request("/stock/recommendation", params)

    async def _get_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Get technical indicators"""
        params = {"symbol": symbol, "resolution": "D"}  # Daily resolution
        return await self._make_request("/indicator", params)

    async def _get_earnings_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get earnings sentiment"""
        params = {"symbol": symbol}
        return await self._make_request("/stock/earnings", params)

    async def _convert_to_news_item(
        self, raw_news: Dict[str, Any], symbol: str, sentiment_score: float
    ) -> Optional[NewsItem]:
        """Convert Finnhub news item to NewsItem"""
        try:
            return NewsItem(
                source="Finnhub",
                title=raw_news["headline"],
                summary=raw_news.get("summary", ""),
                url=raw_news.get("url", ""),
                timestamp=datetime.fromtimestamp(raw_news["datetime"]),
                symbols=[symbol],
                sentiment_score=sentiment_score,
                relevance_score=self._calculate_relevance(raw_news, symbol),
            )
        except (KeyError, ValueError):
            return None

    def _match_sentiment_score(
        self, news: Dict[str, Any], sentiment_data: Dict[str, Any]
    ) -> float:
        """Match news item with sentiment score"""
        try:
            # Try to find matching sentiment in sentiment data
            sentiment = sentiment_data.get("sentiment", {})
            if sentiment:
                return sentiment.get("score", 0.0)
        except Exception:
            pass
        return 0.0

    def _calculate_relevance(self, news: Dict[str, Any], symbol: str) -> float:
        """Calculate relevance score for news item"""
        relevance = 0.0

        # Check related symbols
        if symbol in news.get("related", ""):
            relevance += 0.4

        # Check categories
        categories = news.get("categories", "")
        if "earnings" in categories or "company news" in categories:
            relevance += 0.3

        # Check source quality
        source = news.get("source", "")
        if source in ["Reuters", "Bloomberg", "CNBC"]:
            relevance += 0.3

        return min(relevance, 1.0)

    async def _create_market_metrics(
        self,
        symbol: str,
        recommendations: List[Dict[str, Any]],
        technicals: Dict[str, Any],
        earnings: Dict[str, Any],
    ) -> MarketMetrics:
        """Create MarketMetrics from various data sources"""
        current_time = datetime.now()

        # Calculate aggregated metrics
        technical_indicators = self._aggregate_technical_indicators(technicals)
        recommendation_score = self._calculate_recommendation_score(recommendations)
        earnings_score = self._calculate_earnings_score(earnings)

        return MarketMetrics(
            symbol=symbol,
            timestamp=current_time,
            put_call_ratio=technical_indicators.get("put_call_ratio", 0.0),
            trading_volume=technical_indicators.get("volume", 0),
            price_momentum=technical_indicators.get("momentum", 0.0),
            volume_trend=technical_indicators.get("volume_trend", 0.0),
            technical_indicators={
                "recommendation_score": recommendation_score,
                "earnings_score": earnings_score,
                **technical_indicators,
            },
        )

    def _aggregate_technical_indicators(
        self, technicals: Dict[str, Any]
    ) -> Dict[str, float]:
        """Aggregate technical indicators into a single score"""
        indicators = {}

        try:
            # Add relevant technical indicators
            if "technicalAnalysis" in technicals:
                analysis = technicals["technicalAnalysis"]
                indicators["trend_strength"] = analysis.get("trending", 0.0)
                indicators["oscillator_score"] = analysis.get("oscillating", 0.0)

            # Add momentum indicators
            if "trend" in technicals:
                trend = technicals["trend"]
                indicators["momentum"] = trend.get("momentum", 0.0)
                indicators["rsi"] = trend.get("rsi", 0.0)

        except Exception:
            pass

        return indicators

    def _calculate_recommendation_score(
        self, recommendations: List[Dict[str, Any]]
    ) -> float:
        """Calculate aggregate recommendation score"""
        if not recommendations:
            return 0.0

        try:
            latest = recommendations[0]
            total = sum(
                [
                    latest.get("strongBuy", 0),
                    latest.get("buy", 0),
                    latest.get("hold", 0),
                    latest.get("sell", 0),
                    latest.get("strongSell", 0),
                ]
            )

            if total == 0:
                return 0.0

            score = (
                latest.get("strongBuy", 0) * 1.0
                + latest.get("buy", 0) * 0.5
                + latest.get("hold", 0) * 0.0
                + latest.get("sell", 0) * -0.5
                + latest.get("strongSell", 0) * -1.0
            ) / total

            return score

        except (IndexError, KeyError):
            return 0.0

    def _calculate_earnings_score(self, earnings: Dict[str, Any]) -> float:
        """Calculate earnings sentiment score"""
        try:
            surprises = earnings.get("earningsSurprises", [])
            if not surprises:
                return 0.0

            # Calculate weighted average of recent surprises
            total_weight = 0
            weighted_sum = 0

            for i, surprise in enumerate(surprises[:4]):  # Last 4 quarters
                weight = 1 / (i + 1)  # More recent quarters have higher weight
                surprise_percent = surprise.get("surprisePercent", 0)
                weighted_sum += surprise_percent * weight
                total_weight += weight

            return weighted_sum / total_weight if total_weight > 0 else 0.0

        except Exception:
            return 0.0
