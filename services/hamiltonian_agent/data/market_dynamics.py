# services/hamiltonian_agent/data/market_dynamics.py

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass

@dataclass
class MarketState:
    """Represents market state in phase space"""
    position: np.ndarray  # Price levels, market caps, etc.
    momentum: np.ndarray  # Price changes, volume momentum, etc.
    energy: float        # Market "energy" level
    time: pd.Timestamp

class MarketDataCollector:
    """Collects and processes market data for HNN training"""
    
    def __init__(
        self,
        primary_symbols: List[str],     # Major indices/ETFs
        related_symbols: List[str],     # Related instruments
        lookback_period: str = "2y"     # Data lookback period
    ):
        self.primary_symbols = primary_symbols
        self.related_symbols = related_symbols
        self.lookback_period = lookback_period

    def collect_market_states(self) -> List[MarketState]:
        """Collect market states preserving conservation properties"""
        states = []
        
        # Collect raw data
        primary_data = self._collect_primary_data()
        related_data = self._collect_related_data()
        
        # Process into phase space
        for timestamp, row in primary_data.iterrows():
            # Position components (q)
            position = np.array([
                row['market_cap_normalized'],
                row['price_normalized'],
                row['volume_profile'],
                row['depth_imbalance']
            ])
            
            # Momentum components (p)
            momentum = np.array([
                row['price_velocity'],
                row['volume_momentum'],
                row['order_flow_momentum'],
                row['market_impact']
            ])
            
            # Calculate "energy"
            energy = self._calculate_market_energy(
                position, momentum, related_data.loc[timestamp]
            )
            
            states.append(MarketState(
                position=position,
                momentum=momentum,
                energy=energy,
                time=timestamp
            ))
            
        return states

    def _collect_primary_data(self) -> pd.DataFrame:
        """Collect primary market data"""
        dfs = []
        for symbol in self.primary_symbols:
            df = yf.download(
                symbol,
                period=self.lookback_period,
                interval="1d",
                auto_adjust=True
            )
            
            # Calculate conservation-related features
            df['market_cap_normalized'] = self._normalize_market_cap(df)
            df['price_normalized'] = self._normalize_prices(df)
            df['price_velocity'] = self._calculate_velocity(df['Close'])
            df['volume_momentum'] = self._calculate_volume_momentum(df)
            df['volume_profile'] = self._calculate_volume_profile(df)
            df['depth_imbalance'] = self._calculate_depth_imbalance(df)
            df['order_flow_momentum'] = self._calculate_order_flow(df)
            df['market_impact'] = self._calculate_market_impact(df)
            
            dfs.append(df)
            
        return pd.concat(dfs)

    def _normalize_market_cap(self, df: pd.DataFrame) -> pd.Series:
        """Normalize market capitalization to conserve scale"""
        market_cap = df['Close'] * df['Volume']
        return (market_cap - market_cap.mean()) / market_cap.std()

    def _normalize_prices(self, df: pd.DataFrame) -> pd.Series:
        """Normalize prices to conserve relative levels"""
        return (df['Close'] - df['Close'].mean()) / df['Close'].std()

    def _calculate_velocity(self, prices: pd.Series) -> pd.Series:
        """Calculate price velocity (momentum)"""
        return prices.pct_change()

    def _calculate_volume_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume momentum"""
        volume_changes = df['Volume'].pct_change()
        return volume_changes.rolling(window=5).mean()

    def _calculate_volume_profile(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume profile as conservation metric"""
        total_volume = df['Volume'].rolling(window=20).sum()
        return df['Volume'] / total_volume

    def _calculate_depth_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate market depth imbalance
        (This would ideally use order book data if available)
        """
        high_low_range = df['High'] - df['Low']
        volume_normalized = df['Volume'] / df['Volume'].rolling(window=20).mean()
        return high_low_range * volume_normalized

    def _calculate_order_flow(self, df: pd.DataFrame) -> pd.Series:
        """Calculate order flow momentum"""
        # Approximate using price * volume changes
        flow = (df['Close'] * df['Volume']).pct_change()
        return flow.rolling(window=5).mean()

    def _calculate_market_impact(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market impact as energy dissipation"""
        returns = df['Close'].pct_change()
        volumes = df['Volume'] / df['Volume'].rolling(window=20).mean()
        return returns.abs() / volumes

    def _calculate_market_energy(
        self,
        position: np.ndarray,
        momentum: np.ndarray,
        related_state: pd.Series
    ) -> float:
        """
        Calculate market 'energy' as a conservation quantity
        Combines kinetic (momentum) and potential (position) energies
        """
        # Kinetic energy (momentum-based)
        kinetic = 0.5 * np.sum(momentum ** 2)
        
        # Potential energy (position-based)
        potential = 0.5 * np.sum(position ** 2)
        
        # Add coupling terms with related markets
        coupling = self._calculate_coupling_energy(position, related_state)
        
        return kinetic + potential + coupling

    def _calculate_coupling_energy(
        self,
        position: np.ndarray,
        related_state: pd.Series
    ) -> float:
        """Calculate coupling energy with related markets"""
        # Implement market coupling terms
        coupling_strength = 0.1  # Coupling coefficient
        coupling = 0.0
        
        # Add correlations with related markets
        for key in ['price_normalized', 'volume_profile']:
            if key in related_state:
                coupling += coupling_strength * position[0] * related_state[key]
                
        return coupling

    def validate_conservation(
        self,
        states: List[MarketState],
        threshold: float = 0.1
    ) -> bool:
        """Validate energy conservation in collected states"""
        energies = [state.energy for state in states]
        energy_std = np.std(energies)
        mean_energy = np.mean(energies)
        
        # Check if energy is approximately conserved
        return energy_std / mean_energy < threshold
