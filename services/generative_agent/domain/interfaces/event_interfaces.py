# services/generative_agent/domain/interfaces/event_protocols.py

from typing import List, Dict, Any, Protocol, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class EventCategory(Enum):
    EARNINGS = "earnings"
    SEC_FILING = "sec_filing"
    FED_ANNOUNCEMENT = "fed_announcement"
    DIVIDEND = "dividend"
    STOCK_SPLIT = "stock_split"
    MERGER_ACQUISITION = "merger_acquisition"

@dataclass
class EventData:
    """Value object representing an event"""
    event_id: str
    category: EventCategory
    title: str
    description: str
    timestamp: datetime
    source: str
    url: Optional[str]
    symbols: List[str]
    impact_score: float = 0.0
    metadata: Dict[str, Any] = None

class EventSource(Protocol):
    """Protocol for event data sources"""
    
    async def get_events(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        categories: Optional[List[EventCategory]] = None
    ) -> List[EventData]:
        """Get events for given symbols and timeframe"""
        pass

    async def validate_source(self) -> bool:
        """Validate source availability"""
        pass

    def get_supported_categories(self) -> List[EventCategory]:
        """Get supported event categories"""
        pass
