# services/generative_agent/domain/interfaces/volatility_protocols.py

from typing import List, Dict, Any, Protocol, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class OptionData:
    """Represents option market data"""
    symbol: str
    strike: float
    expiry: datetime
    option_type: str  # 'call' or 'put'
    price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_vol: Optional[float] = None

@dataclass
class VIXData:
    """Represents VIX index data"""
    timestamp: datetime
    close: float
    open: float
    high: float
    low: float
    volume: int

class VolatilityDataSource(Protocol):
    """Protocol for volatility data sources"""
    
    async def get_historical_volatility(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, float]:
        """Get historical volatility data"""
        pass

    async def get_options_chain(
        self,
        symbol: str,
        expiry_range: Optional[tuple[datetime, datetime]] = None
    ) -> List[OptionData]:
        """Get options chain data"""
        pass

    async def get_vix_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[VIXData]:
        """Get VIX index data"""
        pass
