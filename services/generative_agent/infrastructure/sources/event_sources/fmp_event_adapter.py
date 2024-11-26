# services/generative_agent/infrastructure/event_sources/fmp_event_adapter.py

from typing import List, Dict, Any, Optional
from datetime import datetime
import aiohttp
from ...domain.interfaces.event_protocols import EventSource, EventData, EventCategory
from ...domain.exceptions import DataSourceError

class FMPEventAdapter(EventSource):
    """
    Adapter for Financial Modeling Prep (FMP) API events.
    Uses free tier access for financial events data.
    """
    
    def __init__(self):
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self._supported_categories = [
            EventCategory.EARNINGS,
            EventCategory.MERGER_ACQUISITION,
            EventCategory.DIVIDEND
        ]

    async def get_events(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        categories: Optional[List[EventCategory]] = None
    ) -> List[EventData]:
        """Get events from FMP API"""
        try:
            events = []
            async with aiohttp.ClientSession() as session:
                for symbol in symbols:
                    # Get earnings calendars
                    if not categories or EventCategory.EARNINGS in categories:
                        earnings_events = await self._get_earnings_calendar(
                            session, symbol, start_date, end_date
                        )
                        events.extend(earnings_events)
                    
                    # Get M&A events
                    if not categories or EventCategory.MERGER_ACQUISITION in categories:
                        ma_events = await self._get_merger_acquisitions(
                            session, symbol, start_date, end_date
                        )
                        events.extend(ma_events)
                    
                    # Get dividend events
                    if not categories or EventCategory.DIVIDEND in categories:
                        dividend_events = await self._get_dividend_calendar(
                            session, symbol, start_date, end_date
                        )
                        events.extend(dividend_events)
            
            return events

        except Exception as e:
            raise DataSourceError(f"Error fetching FMP events: {str(e)}")

    async def validate_source(self) -> bool:
        """Validate FMP API availability"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/is-the-market-open"
                ) as response:
                    return response.status == 200
        except Exception:
            return False

    def get_supported_categories(self) -> List[EventCategory]:
        """Get supported event categories"""
        return self._supported_categories

    async def _get_earnings_calendar(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[EventData]:
        """Get earnings calendar events"""
        url = f"{self.base_url}/historical/earning_calendar/{symbol}"
        params = {
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d')
        }
        
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                events = []
                
                for event in data:
                    event_date = datetime.strptime(
                        event['date'], '%Y-%m-%d'
                    )
                    
                    events.append(EventData(
                        event_id=f"fmp_earnings_{symbol}_{event_date}",
                        category=EventCategory.EARNINGS,
                        title=f"{symbol} Earnings Report",
                        description=self._format_earnings_description(event),
                        timestamp=event_date,
                        source="Financial Modeling Prep",
                        url=f"https://financialmodelingprep.com/financial-statements/{symbol}",
                        symbols=[symbol],
                        impact_score=self._calculate_earnings_impact(event),
                        metadata={
                            'eps': event.get('eps', None),
                            'estimated_eps': event.get('epsEstimated', None),
                            'revenue': event.get('revenue', None),
                            'estimated_revenue': event.get('revenueEstimated', None)
                        }
                    ))
                
                return events
        return []

    async def _get_merger_acquisitions(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[EventData]:
        """Get merger and acquisition events"""
        url = f"{self.base_url}/mergers-acquisitions-rss-feed"
        
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                events = []
                
                for event in data:
                    event_date = datetime.strptime(
                        event['publishedDate'], '%Y-%m-%d %H:%M:%S'
                    )
                    
                    if (start_date <= event_date <= end_date and
                        (symbol.lower() in event['title'].lower() or
                         symbol.lower() in event['text'].lower())):
                        
                        events.append(EventData(
                            event_id=f"fmp_ma_{symbol}_{event_date}",
                            category=EventCategory.MERGER_ACQUISITION,
                            title=event['title'],
                            description=event['text'],
                            timestamp=event_date,
                            source="Financial Modeling Prep",
                            url=event.get('link', ''),
                            symbols=[symbol],
                            impact_score=self._calculate_ma_impact(event),
                            metadata={
                                'deal_type': self._determine_deal_type(event),
                                'companies_involved': self._extract_companies(event)
                            }
                        ))
                
                return events
        return []

    async def _get_dividend_calendar(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[EventData]:
        """Get dividend calendar events"""
        url = f"{self.base_url}/historical-price-full/stock_dividend/{symbol}"
        
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                events = []
                
                for event in data.get('historical', []):
                    event_date = datetime.strptime(
                        event['date'], '%Y-%m-%d'
                    )
                    
                    if start_date <= event_date <= end_date:
                        events.append(EventData(
                            event_id=f"fmp_dividend_{symbol}_{event_date}",
                            category=EventCategory.DIVIDEND,
                            title=f"{symbol} Dividend Announcement",
                            description=self._format_dividend_description(event),
                            timestamp=event_date,
                            source="Financial Modeling Prep",
                            url=f"https://financialmodelingprep.com/financial-statements/{symbol}",
                            symbols=[symbol],
                            impact_score=self._calculate_dividend_impact(event),
                            metadata={
                                'dividend': event.get('dividend', None),
                                'adjusted_dividend': event.get('adjDividend', None),
                                'record_date': event.get('recordDate', None),
                                'payment_date': event.get('paymentDate', None)
                            }
                        ))
                
                return events
        return []

    def _format_earnings_description(self, event: Dict[str, Any]) -> str:
        """Format earnings event description"""
        eps = event.get('eps', 'N/A')
        eps_est = event.get('epsEstimated', 'N/A')
        rev = event.get('revenue', 'N/A')
        rev_est = event.get('revenueEstimated', 'N/A')
        
        return (
            f"EPS: ${eps} (Est: ${eps_est}), "
            f"Revenue: ${rev:,.2f} (Est: ${rev_est:,.2f})"
        )

    def _format_dividend_description(self, event: Dict[str, Any]) -> str:
        """Format dividend event description"""
        dividend = event.get('dividend', 'N/A')
        adj_dividend = event.get('adjDividend', 'N/A')
        return f"Dividend Amount: ${dividend} (Adjusted: ${adj_dividend})"

    def _calculate_earnings_impact(self, event: Dict[str, Any]) -> float:
        """Calculate impact score for earnings event"""
        try:
            eps = float(event.get('eps', 0))
            eps_est = float(event.get('epsEstimated', 0))
            
            if eps_est != 0:
                surprise = abs((eps - eps_est) / eps_est)
                return min(surprise, 1.0)
        except (ValueError, TypeError):
            pass
        return 0.0

    def _calculate_ma_impact(self, event: Dict[str, Any]) -> float:
        """Calculate impact score for M&A event"""
        # Simple keyword-based scoring
        impact_score = 0.0
        text = (event.get('title', '') + ' ' + event.get('text', '')).lower()
        
        if 'merger' in text:
            impact_score += 0.8
        elif 'acquisition' in text:
            impact_score += 0.6
        elif 'takeover' in text:
            impact_score += 0.7
        
        return min(impact_score, 1.0)

    def _calculate_dividend_impact(self, event: Dict[str, Any]) -> float:
        """Calculate impact score for dividend event"""
        try:
            current = float(event.get('dividend', 0))
            adjusted = float(event.get('adjDividend', 0))
            
            if adjusted != 0:
                change = abs((current - adjusted) / adjusted)
                return min(change, 1.0)
        except (ValueError, TypeError):
            pass
        return 0.0

    def _determine_deal_type(self, event: Dict[str, Any]) -> str:
        """Determine type of M&A deal"""
        text = (event.get('title', '') + ' ' + event.get('text', '')).lower()
        
        if 'merger' in text:
            return 'merger'
        elif 'acquisition' in text:
            return 'acquisition'
        elif 'takeover' in text:
            return 'takeover'
        return 'other'

    def _extract_companies(self, event: Dict[str, Any]) -> List[str]:
        """Extract company names from M&A event"""
        # This is a simplified implementation
        # Could be enhanced with NLP for better company name extraction
        companies = set()
        text = event.get('text', '')
        
        # Look for common company identifiers
        for word in text.split():
            if word.endswith('Corp.') or word.endswith('Inc.') or word.endswith('Ltd.'):
                companies.add(word)
        
        return list(companies)
