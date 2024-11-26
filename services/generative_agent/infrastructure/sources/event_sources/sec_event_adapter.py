# services/generative_agent/infrastructure/event_sources/sec_event_adapter.py

from typing import List, Dict, Any, Optional
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
from ...domain.interfaces.event_protocols import EventSource, EventData, EventCategory
from ...domain.exceptions import DataSourceError

class SECEventAdapter(EventSource):

    """
    Adapter for SEC EDGAR filing events.
    Uses public SEC EDGAR database.
    """
    
    def __init__(self):
        self.base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        self._supported_categories = [EventCategory.SEC_FILING]

    async def get_events(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        categories: Optional[List[EventCategory]] = None
    ) -> List[EventData]:
        """Get SEC filing events"""
        if categories and EventCategory.SEC_FILING not in categories:
            return []

        try:
            events = []
            async with aiohttp.ClientSession() as session:
                for symbol in symbols:
                    filings = await self._get_company_filings(
                        session, symbol, start_date, end_date
                    )
                    events.extend(filings)
            return events

        except Exception as e:
            raise DataSourceError(f"Error fetching SEC events: {str(e)}")

    async def validate_source(self) -> bool:
        """Validate SEC EDGAR availability"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url) as response:
                    return response.status == 200
        except Exception:
            return False

    def get_supported_categories(self) -> List[EventCategory]:
        """Get supported event categories"""
        return self._supported_categories

    async def _get_company_filings(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[EventData]:
        """Get SEC filings for a specific company"""
        params = {
            'CIK': symbol,
            'type': '10-K,10-Q,8-K',
            'dateb': end_date.strftime('%Y%m%d'),
            'datea': start_date.strftime('%Y%m%d'),
            'owner': 'exclude',
            'count': '100'
        }
        
        async with session.get(self.base_url, params=params) as response:
            if response.status == 200:
                html = await response.text()
                return await self._parse_edgar_response(html, symbol)
            return []

    async def _parse_edgar_response(
        self,
        html: str,
        symbol: str
    ) -> List[EventData]:
        """Parse SEC EDGAR HTML response"""
        events = []
        soup = BeautifulSoup(html, 'html.parser')
        
        for row in soup.find_all('tr'):
            try:
                cells = row.find_all('td')
                if len(cells) >= 4:
                    filing_type = cells[0].text.strip()
                    filing_date = datetime.strptime(
                        cells[3].text.strip(), '%Y-%m-%d'
                    )
                    description = cells[1].text.strip()
                    url = f"https://www.sec.gov{cells[1].find('a')['href']}"
                    
                    events.append(EventData(
                        event_id=f"sec_{symbol}_{filing_date}_{filing_type}",
                        category=EventCategory.SEC_FILING,
                        title=f"{symbol} {filing_type} Filing",
                        description=description,
                        timestamp=filing_date,
                        source="SEC EDGAR",
                        url=url,
                        symbols=[symbol],
                        metadata={
                            'filing_type': filing_type,
                            'accession_number': cells[4].text.strip()
                            if len(cells) > 4 else None
                        }
                    ))
            except Exception:
                continue
                
        return events
