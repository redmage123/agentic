# services/generative_agent/infrastructure/volatility_sources/cboe_adapter.py

import aiohttp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

from .volatility_protocols import VolatilityDataSource, OptionData, VIXData
from ...domain.exceptions import DataSourceError

class CBOEVolatilityAdapter(VolatilityDataSource):
    """
    Adapter for CBOE (Chicago Board Options Exchange) data.
    Uses freely available data from CBOE's website and data feeds.
    """
    
    def __init__(self):
        self.base_url = "https://www.cboe.com/us/options/market_statistics"
        self.vix_url = "https://cdn.cboe.com/api/global/delayed_quotes/charts/historical"
        self.skew_url = "https://cdn.cboe.com/data/us/futures/market_statistics/daily_settlement/skew"
        self.session = None
        self._cache = {}
        self.cache_duration = timedelta(minutes=15)

    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Accept': 'application/json'
                }
            )

    async def get_historical_volatility(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, float]:
        """
        Get historical volatility data from CBOE.
        Includes VIX-style calculation for individual stocks.
        """
        try:
            await self._ensure_session()
            
            # Check cache
            cache_key = f"hist_vol_{symbol}_{start_date}_{end_date}"
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                return cached_data

            # Construct URL for historical data
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
            }

            async with self.session.get(
                f"{self.base_url}/historical_volatility",
                params=params
            ) as response:
                if response.status != 200:
                    raise DataSourceError(f"Failed to fetch CBOE data: {response.status}")
                
                data = await response.json()
                
                # Calculate volatilities for different windows
                vols = {}
                if data.get('data'):
                    df = pd.DataFrame(data['data'])
                    for window in [5, 10, 20, 30, 60]:
                        vols[f'{window}d'] = self._calculate_volatility(
                            df, window
                        )
                
                # Cache the results
                self._add_to_cache(cache_key, vols)
                return vols

        except Exception as e:
            raise DataSourceError(f"CBOE historical volatility error: {str(e)}")

    async def get_options_chain(
        self,
        symbol: str,
        expiry_range: Optional[tuple[datetime, datetime]] = None
    ) -> List[OptionData]:
        """Get options chain data from CBOE"""
        try:
            await self._ensure_session()
            
            # Check cache
            cache_key = f"options_{symbol}"
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                return cached_data

            # Fetch options chain
            params = {'symbol': symbol}
            async with self.session.get(
                f"{self.base_url}/options_chains",
                params=params
            ) as response:
                if response.status != 200:
                    raise DataSourceError(f"Failed to fetch options data: {response.status}")
                
                data = await response.json()
                options_data = []
                
                for option in data.get('data', []):
                    expiry_date = datetime.strptime(
                        option['expiration'],
                        '%Y-%m-%d'
                    )
                    
                    # Apply expiry filter if provided
                    if expiry_range:
                        if not (expiry_range[0] <= expiry_date <= expiry_range[1]):
                            continue
                    
                    # Create OptionData objects
                    options_data.append(
                        OptionData(
                            symbol=symbol,
                            strike=float(option['strike']),
                            expiry=expiry_date,
                            option_type=option['type'].lower(),
                            price=float(option['last']),
                            bid=float(option['bid']),
                            ask=float(option['ask']),
                            volume=int(option['volume']),
                            open_interest=int(option['openInterest']),
                            implied_vol=float(option['impliedVolatility'])
                        )
                    )
                
                # Cache the results
                self._add_to_cache(cache_key, options_data)
                return options_data

        except Exception as e:
            raise DataSourceError(f"CBOE options chain error: {str(e)}")

    async def get_vix_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[VIXData]:
        """Get VIX index data from CBOE"""
        try:
            await self._ensure_session()
            
            # Check cache
            cache_key = f"vix_{start_date}_{end_date}"
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                return cached_data

            # Fetch VIX data
            params = {
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
            }
            
            async with self.session.get(
                f"{self.vix_url}/VIX",
                params=params
            ) as response:
                if response.status != 200:
                    raise DataSourceError(f"Failed to fetch VIX data: {response.status}")
                
                data = await response.json()
                vix_data = []
                
                for point in data.get('data', []):
                    vix_data.append(
                        VIXData(
                            timestamp=datetime.fromtimestamp(point['timestamp']/1000),
                            close=float(point['close']),
                            open=float(point['open']),
                            high=float(point['high']),
                            low=float(point['low']),
                            volume=int(point['volume'])
                        )
                    )
                
                # Cache the results
                self._add_to_cache(cache_key, vix_data)
                return vix_data

        except Exception as e:
            raise DataSourceError(f"CBOE VIX data error: {str(e)}")

    async def get_volatility_skew(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        """Get volatility skew data"""
        try:
            await self._ensure_session()
            
            async with self.session.get(
                f"{self.skew_url}/{symbol}"
            ) as response:
                if response.status != 200:
                    raise DataSourceError(f"Failed to fetch skew data: {response.status}")
                
                data = await response.json()
                return self._process_skew_data(data)

        except Exception as e:
            raise DataSourceError(f"CBOE skew data error: {str(e)}")

    async def get_term_structure(
        self,
        symbol: str
    ) -> Dict[str, float]:
        """Get volatility term structure"""
        try:
            options_data = await self.get_options_chain(symbol)
            
            # Group by expiry and calculate average implied vol
            term_structure = {}
            grouped_data = {}
            
            for option in options_data:
                expiry = option.expiry
                if expiry not in grouped_data:
                    grouped_data[expiry] = []
                grouped_data[expiry].append(option.implied_vol)
                
            for expiry, vols in grouped_data.items():
                term_structure[expiry.strftime('%Y-%m-%d')] = np.mean(vols)
                
            return term_structure

        except Exception as e:
            raise DataSourceError(f"CBOE term structure error: {str(e)}")

    async def get_volatility_surface(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        """Get complete volatility surface"""
        try:
            options_data = await self.get_options_chain(symbol)
            
            # Create surface structure
            surface = {
                'strikes': [],
                'expiries': [],
                'implied_vols': [],
                'call_vols': [],
                'put_vols': []
            }
            
            # Organize data
            for option in options_data:
                if option.strike not in surface['strikes']:
                    surface['strikes'].append(option.strike)
                if option.expiry not in surface['expiries']:
                    surface['expiries'].append(option.expiry)
                    
            surface['strikes'].sort()
            surface['expiries'].sort()
            
            # Create vol matrices
            for i, expiry in enumerate(surface['expiries']):
                vol_row = []
                call_row = []
                put_row = []
                
                for strike in surface['strikes']:
                    # Find matching options
                    matching_options = [
                        opt for opt in options_data
                        if opt.expiry == expiry and opt.strike == strike
                    ]
                    
                    call_vol = next(
                        (opt.implied_vol for opt in matching_options 
                         if opt.option_type == 'call'),
                        None
                    )
                    put_vol = next(
                        (opt.implied_vol for opt in matching_options 
                         if opt.option_type == 'put'),
                        None
                    )
                    
                    # Average of call and put vols if both exist
                    implied_vol = np.mean([v for v in [call_vol, put_vol] if v is not None])
                    
                    vol_row.append(implied_vol)
                    call_row.append(call_vol)
                    put_row.append(put_vol)
                    
                surface['implied_vols'].append(vol_row)
                surface['call_vols'].append(call_row)
                surface['put_vols'].append(put_row)
                
            return surface

        except Exception as e:
            raise DataSourceError(f"CBOE volatility surface error: {str(e)}")

    def _calculate_volatility(
        self,
        df: pd.DataFrame,
        window: int
    ) -> float:
        """Calculate historical volatility for a given window"""
        returns = np.log(df['close'] / df['close'].shift(1))
        vol = returns.rolling(window=window).std() * np.sqrt(252)
        return vol.iloc[-1]

    def _process_skew_data(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process raw skew data"""
        skew_data = {
            'timestamp': [],
            'skew_value': [],
            'percentiles': {
                '25th': [],
                '50th': [],
                '75th': []
            }
        }
        
        for point in data:
            skew_data['timestamp'].append(point['timestamp'])
            skew_data['skew_value'].append(point['skew'])
            skew_data['percentiles']['25th'].append(point['percentile_25'])
            skew_data['percentiles']['50th'].append(point['percentile_50'])
            skew_data['percentiles']['75th'].append(point['percentile_75'])
            
        return skew_data

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache if not expired"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if datetime.now() - timestamp < self.cache_duration:
                return data
            else:
                del self._cache[key]
        return None

    def _add_to_cache(self, key: str, data: Any):
        """Add data to cache"""
        self._cache[key] = (data, datetime.now())

    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
            self.session = None
