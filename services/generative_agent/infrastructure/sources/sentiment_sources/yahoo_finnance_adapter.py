# services/generative_agent/infrastructure/sentiment_sources/yahoo_finance_adapter.py

from typing import List, Dict, Any
from datetime import datetime
import yfinance as yf
from ...domain.interfaces.sentiment_protocols import NewsSource, NewsItem
from ...domain.exceptions import DataSourceError


class YahooFinanceNewsAdapter(NewsSource):
    """Adapter for Yahoo Finance news data"""

    def __init__(self, request_timeout: int = 30):
        self.request_timeout = request_timeout
        self._ticker_cache: Dict[str, yf.Ticker] = {}

    async def get_news(
        self, symbols: List[str], start_date: datetime, end_date: datetime
    ) -> List[NewsItem]:
        """Get news articles from Yahoo Finance"""
        try:
            news_items = []
            for symbol in symbols:
                ticker = await self._get_ticker(symbol)
                if ticker:
                    news = ticker.news
                    for item in news:
                        if self._is_within_timerange(item, start_date, end_date):
                            news_items.append(self._convert_to_news_item(item, symbol))
            return news_items
        except Exception as e:
            raise DataSourceError(f"Error fetching Yahoo Finance news: {str(e)}")

    async def validate_source(self) -> bool:
        """Validate Yahoo Finance availability"""
        try:
            # Test with a known valid symbol
            test_ticker = await self._get_ticker("AAPL")
            return bool(test_ticker is not None)
        except Exception:
            return False

    async def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create ticker object with caching"""
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)
        return self._ticker_cache[symbol]

    def _is_within_timerange(
        self, item: Dict[str, Any], start_date: datetime, end_date: datetime
    ) -> bool:
        """Check if news item is within specified timerange"""
        item_date = datetime.fromtimestamp(item["providerPublishTime"])
        return start_date <= item_date <= end_date

    def _convert_to_news_item(self, raw_item: Dict[str, Any], symbol: str) -> NewsItem:
        """Convert Yahoo Finance news item to NewsItem"""
        return NewsItem(
            source="Yahoo Finance",
            title=raw_item["title"],
            summary=raw_item.get("summary", ""),
            url=raw_item.get("link", ""),
            timestamp=datetime.fromtimestamp(raw_item["providerPublishTime"]),
            symbols=[symbol],
            relevance_score=self._calculate_relevance(raw_item, symbol),
        )

    def _calculate_relevance(self, item: Dict[str, Any], symbol: str) -> float:
        """Calculate relevance score for news item"""
        relevance = 0.0
        if symbol in item["title"]:
            relevance += 0.6
        if symbol in item.get("summary", ""):
            relevance += 0.4
        return min(relevance, 1.0)
