# services/generative_agent/infrastructure/event_sources/yahoo_event_adapter.py

from typing import List, Dict, Any, Optional
from datetime import datetime
import yfinance as yf
from ...domain.interfaces.event_protocols import EventSource, EventData, EventCategory
from ...domain.exceptions import DataSourceError

class YahooEventAdapter(EventSource):
    """
    Adapter for Yahoo Finance events including earnings,
    dividends, and stock splits.
    """
    
    def __init__(self):
        self._ticker_cache: Dict[str, yf.Ticker] = {}
        self._supported_categories = [
            EventCategory.EARNINGS,
            EventCategory.DIVIDEND,
            EventCategory.STOCK_SPLIT
        ]

    async def get_events(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        categories: Optional[List[EventCategory]] = None
    ) -> List[EventData]:
        """Get events from Yahoo Finance"""
        try:
            events = []
            for symbol in symbols:
                ticker = await self._get_ticker(symbol)
                if ticker:
                    # Get earnings events
                    if not categories or EventCategory.EARNINGS in categories:
                        earnings_events = await self._get_earnings_events(
                            ticker, symbol, start_date, end_date
                        )
                        events.extend(earnings_events)
                    
                    # Get dividend events
                    if not categories or EventCategory.DIVIDEND in categories:
                        dividend_events = await self._get_dividend_events(
                            ticker, symbol, start_date, end_date
                        )
                        events.extend(dividend_events)
                    
                    # Get stock split events
                    if not categories or EventCategory.STOCK_SPLIT in categories:
                        split_events = await self._get_split_events(
                            ticker, symbol, start_date, end_date
                        )
                        events.extend(split_events)
            
            return events

        except Exception as e:
            raise DataSourceError(f"Error fetching Yahoo Finance events: {str(e)}")

    async def validate_source(self) -> bool:
        """Validate Yahoo Finance availability"""
        try:
            test_ticker = await self._get_ticker("SPY")
            return test_ticker is not None
        except Exception:
            return False

    def get_supported_categories(self) -> List[EventCategory]:
        """Get supported event categories"""
        return self._supported_categories

    async def _get_ticker(self, symbol: str) -> Optional[yf.Ticker]:
        """Get or create ticker object with caching"""
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)
        return self._ticker_cache[symbol]

    async def _get_earnings_events(
        self,
        ticker: yf.Ticker,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[EventData]:
        """Get earnings events"""
        events = []
        calendar = ticker.calendar
        if calendar is not None:
            earnings_date = calendar.get('Earnings Date')
            if earnings_date and start_date <= earnings_date <= end_date:
                events.append(EventData(
                    event_id=f"earnings_{symbol}_{earnings_date}",
                    category=EventCategory.EARNINGS,
                    title=f"{symbol} Earnings Release",
                    description=f"Expected EPS: {calendar.get('EPS Estimate', 'N/A')}",
                    timestamp=earnings_date,
                    source="Yahoo Finance",
                    url=f"https://finance.yahoo.com/quote/{symbol}",
                    symbols=[symbol],
                    metadata={
                        'eps_estimate': calendar.get('EPS Estimate'),
                        'revenue_estimate': calendar.get('Revenue Estimate')
                    }
                ))
        return events

    async def _get_dividend_events(
        self,
        ticker: yf.Ticker,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[EventData]:
        """Get dividend events"""
        events = []
        dividends = ticker.dividends
        if dividends is not None:
            for date, amount in dividends.items():
                if start_date <= date <= end_date:
                    events.append(EventData(
                        event_id=f"dividend_{symbol}_{date}",
                        category=EventCategory.DIVIDEND,
                        title=f"{symbol} Dividend Payment",
                        description=f"Dividend amount: ${amount:.2f}",
                        timestamp=date,
                        source="Yahoo Finance",
                        url=f"https://finance.yahoo.com/quote/{symbol}",
                        symbols=[symbol],
                        metadata={'dividend_amount': amount}
                    ))
        return events

    async def _get_split_events(
        self,
        ticker: yf.Ticker,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[EventData]:
        """Get stock split events"""
        events = []
        splits = ticker.splits
        if splits is not None:
            for date, ratio in splits.items():
                if start_date <= date <= end_date:
                    events.append(EventData(
                        event_id=f"split_{symbol}_{date}",
                        category=EventCategory.STOCK_SPLIT,
                        title=f"{symbol} Stock Split",
                        description=f"Split ratio: {ratio}:1",
                        timestamp=date,
                        source="Yahoo Finance",
                        url=f"https://finance.yahoo.com/quote/{symbol}",
                        symbols=[symbol],
                        metadata={'split_ratio': ratio}
                    ))
        return events
