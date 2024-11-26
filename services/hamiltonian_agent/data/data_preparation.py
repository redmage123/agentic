# services/hamiltonian_agent/data/data_preparation.py

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

class MarketDataset(Dataset):
    """
    Dataset for Hamiltonian NN training on market data.
    Transforms market data into phase space (position/momentum) representation.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = 10,
        transform: bool = True
    ):
        """
        Args:
            data: Raw market data (prices, volumes, etc.)
            sequence_length: Length of sequences to generate
            transform: Whether to apply phase space transformation
        """
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Prepare sequences
        self.sequences = self._prepare_sequences(data)
        if transform:
            self.sequences = self._transform_to_phase_space(self.sequences)
            
    def __len__(self) -> int:
        return len(self.sequences) - 1  # -1 for target state
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sequence and target state"""
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.sequences[idx + 1])
        )
        
    def _prepare_sequences(self, data: np.ndarray) -> np.ndarray:
        """Prepare sequences of market states"""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequence = data[i:i + self.sequence_length]
            sequences.append(sequence)
        return np.array(sequences)
        
    def _transform_to_phase_space(self, sequences: np.ndarray) -> np.ndarray:
        """
        Transform market data into phase space representation
        (position q and momentum p)
        """
        # Position (q) represents price levels
        q = sequences.copy()
        
        # Momentum (p) represents price changes
        p = np.gradient(sequences, axis=0)
        
        # Combine into phase space representation [q, p]
        phase_space = np.concatenate([q, p], axis=-1)
        return phase_space

class MarketDataPreprocessor:
    """
    Preprocesses market data for Hamiltonian NN training.
    Handles data loading, cleaning, and transformation.
    """
    
    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        sequence_length: int = 10,
        train_split: float = 0.8,
        batch_size: int = 32
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.train_split = train_split
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders
        """
        # Load data
        raw_data = self._load_market_data()
        
        # Preprocess
        processed_data = self._preprocess_data(raw_data)
        
        # Split data
        train_data, val_data = self._split_data(processed_data)
        
        # Create datasets
        train_dataset = MarketDataset(
            train_data,
            self.sequence_length
        )
        val_dataset = MarketDataset(
            val_data,
            self.sequence_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader
        
    def _load_market_data(self) -> pd.DataFrame:
        """Load market data using yfinance"""
        data_frames = []
        
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval='1d'
            )
            
            # Add symbol column
            df['Symbol'] = symbol
            data_frames.append(df)
            
        # Combine all data
        combined_data = pd.concat(data_frames)
        return combined_data
        
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess market data"""
        # Extract relevant features
        features = [
            'Close', 'Volume', 'High', 'Low',
            'Open', 'Close'
        ]
        
        # Calculate additional features
        data['Returns'] = data.groupby('Symbol')['Close'].pct_change()
        data['Volatility'] = data.groupby('Symbol')['Returns'].rolling(
            window=20
        ).std().reset_index(0, drop=True)
        
        # Add momentum indicators
        data['RSI'] = self._calculate_rsi(data['Close'])
        data['MACD'] = self._calculate_macd(data['Close'])
        
        # Add market microstructure features
        data['Price_Impact'] = self._calculate_price_impact(
            data['Returns'],
            data['Volume']
        )
        
        # Scale features
        feature_data = data[features].values
        scaled_data = self.scaler.fit_transform(feature_data)
        
        return scaled_data
        
    def _split_data(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split data into train and validation sets"""
        split_idx = int(len(data) * self.train_split)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        return train_data, val_data
        
    def _calculate_rsi(
        self,
        prices: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.Series:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line
        
    def _calculate_price_impact(
        self,
        returns: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Calculate price impact coefficient"""
        return returns.abs() / volume

def generate_synthetic_data(
    num_samples: int,
    num_features: int,
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Generate synthetic market data for testing
    """
    # Generate base patterns
    t = np.linspace(0, 8*np.pi, num_samples)
    base_pattern = np.sin(t) + np.sin(2*t) + np.sin(3*t)
    
    # Add features
    data = np.zeros((num_samples, num_features))
    for i in range(num_features):
        # Add phase shift and amplitude variation
        phase_shift = np.random.uniform(0, 2*np.pi)
        amplitude = np.random.uniform(0.5, 2.0)
        data[:, i] = amplitude * np.sin(t + phase_shift)
        
    # Add noise
    noise = np.random.normal(0, noise_level, (num_samples, num_features))
    data += noise
    
    return data
