# services/generative_agent/infrastructure/volatility_sources/yahoo_volatility_adapter.py

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from .volatility_protocols import VolatilityDataSource, OptionData, VIXData

class YahooVolatilityAdapter(VolatilityDataSource):
    """Adapter for Yahoo Finance volatility data"""
    
    def __init__(self):
        self._ticker_cache: Dict[str, yf.Ticker] = {}

    async def get_historical_volatility(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, float]:
        """Get historical volatility from price data"""
        try:
            # Get ticker
            ticker = await self._get_ticker(symbol)
            
            # Get historical data
            hist_data = ticker.history(
                start=start_date,
                end=end_date,
                interval='1d'
            )
            
            # Calculate returns
            returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
            
            # Calculate volatilities for different windows
            vols = {}
            for window in [5, 10, 20, 30, 60]:
                rolling_vol = returns.rolling(
                    window=window
                ).std() * np.sqrt(252)
                vols[f'{window}d'] = rolling_vol.iloc[-1]
                
            return vols

        except Exception as e:
            raise DataSourceError(f"Failed to get historical volatility: {str(e)}")

    async def get_options_chain(
        self,
        symbol: str,
        expiry_range: Optional[tuple[datetime, datetime]] = None
    ) -> List[OptionData]:
        """Get options chain from Yahoo Finance"""
        try:
            # Get ticker
            ticker = await self._get_ticker(symbol)
            
            # Get all expiries
            expirations = ticker.options
            
            if not expirations:
                return []
                
            options_data = []
            
            for expiry in expirations:
                expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
                
                # Check expiry range if provided
                if expiry_range:
                    if not (expiry_range[0] <= expiry_date <= expiry_range[1]):
                        continue
                
                # Get options chain for this expiry
                opt = ticker.option_chain(expiry)
                
                # Process calls
                for _, row in opt.calls.iterrows():
                    options_data.append(
                        OptionData(
                            symbol=symbol,
                            strike=row['strike'],
                            expiry=expiry_date,
                            option_type='call',
                            price=row['lastPrice'],
                            bid=row['bid'],
                            ask=row['ask'],
                            volume=row['volume'],
                            open_interest=row['openInterest'],
                            implied_vol=row['impliedVolatility']
                        )
                    )
                
                # Process puts
                for _, row in opt.puts.iterrows():
                    options_data.append(
                        OptionData(
                            symbol=symbol,
                            strike=row['strike'],
                            expiry=expiry_date,
                            option_type='put',
                            price=row['lastPrice'],
                            bid=row['bid'],
                            ask=row['ask'],
                            volume=row['volume'],
                            open_interest=row['openInterest'],
                            implied_vol=row['impliedVolatility']
                        )
                    )
            
            return options_data

        except Exception as e:
            raise DataSourceError(f"Failed to get options chain: {str(e)}")

    async def get_vix_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[VIXData]:
        """Get VIX index data from Yahoo Finance"""
        try:
            # Get VIX data
            vix = yf.Ticker("^VIX")
            hist_data = vix.history(
                start=start_date,
                end=end_date,
                interval='1d'
            )
            
            vix_data = []
            
            for index, row in hist_data.iterrows():
                vix_data.append(
                    VIXData(
                        timestamp=index.to_pydatetime(),
                        close=row['Close'],
                        open=row['Open'],
                        high=row['High'],
                        low=row['Low'],
                        volume=row['Volume']
                    )
                )
            
            return vix_data

        except Exception as e:
            raise DataSourceError(f"Failed to get VIX data: {str(e)}")

    async def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create cached ticker object"""
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)
        return self._ticker_cache[symbol]
