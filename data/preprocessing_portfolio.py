#!C:/Users/bbrel/agentic/.venv/Scripts/python
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import logging
from typing import Dict, Tuple, Optional
import sys

# Configure logging for the preprocessing pipeline
logging.basicConfig(
    filename='preprocessing_log.txt',   # Specify a log file name
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# --- Module Level Documentation ---
"""
This script handles data preprocessing for a portfolio of stocks. It includes functions for 
loading historical stock data, normalizing it, and adding advanced feature engineering such as 
Fourier transforms and technical indicators. The preprocessing pipeline is tailored to support 
different neural network agents, making it suitable for an agent-based stock prediction application.

Main functions:
- load_config: Loads configuration settings from a YAML file.
- normalize_data: Normalizes price and volume data and saves or loads scalers as needed.
- apply_fourier_transform: Applies Fourier transform to capture frequency components.
- add_technical_indicators: Adds common technical indicators used in stock analysis.
- preprocess_portfolio: Iterates over a portfolio of stocks, applying all preprocessing steps.
"""
class PreprocessingException(Exception):
    """Custom exception for errors encountered during data preprocessing."""
    pass

def load_config(config_path: str = "config.yaml") -> Dict[str, any]:
    """
    Load configuration settings from a YAML file.

    Parameters:
    - config_path (str): File path to the YAML configuration file (default: "config.yaml").
    
    Returns:
    - Dict[str, any]: A dictionary containing configuration settings such as tickers, date range, and folder path.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def normalize_data(df: pd.DataFrame, scaler_path: Optional[str] = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalize stock data to scale values between 0 and 1 for model consistency. 
    Optionally saves the scaler to ensure consistent future scaling.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing stock data to normalize.
    - scaler_path (Optional[str]): File path to save the fitted scaler (if provided).
    
    Returns:
    - Tuple[pd.DataFrame, MinMaxScaler]: A tuple containing the normalized DataFrame and the fitted scaler.
    """
    scaler = MinMaxScaler()
    print ("Fuck you",file=sys.stderr)
    print ("columns = ",df.columns,file=sys.stderr)
    df[['Open', 'High', 'Low', 'Adjusted Close', 'Volume']] = scaler.fit_transform(
        df[['Open', 'High', 'Low', 'Adjusted Close', 'Volume']]
    )
    if scaler_path:
        pd.to_pickle(scaler, scaler_path)
    return df, scaler

def load_scaler(scaler_path: str) -> Optional[MinMaxScaler]:
    """
    Load a saved scaler from a given path. If the scaler file exists, it loads the file;
    otherwise, it returns None, indicating no scaler file is available.
    
    Parameters:
    - scaler_path (str): File path to load the scaler from.
    
    Returns:
    - Optional[MinMaxScaler]: The loaded scaler, or None if the file does not exist.
    """
    return pd.read_pickle(scaler_path) if os.path.exists(scaler_path) else None

def apply_fourier_transform(df: pd.DataFrame, column: str = 'Adjusted Close') -> pd.DataFrame:
    """
    Apply a Fourier transform on the specified column to capture frequency components 
    in the time-series data. This is useful for detecting cyclical patterns in stock prices.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing stock data.
    - column (str): Column name to apply Fourier transform on.
    
    Returns:
    - pd.DataFrame: The input DataFrame with an additional 'Fourier' column.
    """
    fourier_transform = np.fft.fft(df[column].values)
    df['Fourier'] = np.abs(fourier_transform)
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add commonly used technical indicators, including a 20-day moving average, 
    Relative Strength Index (RSI), and a 20-day rolling volatility.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing stock data.
    
    Returns:
    - pd.DataFrame: The input DataFrame with additional columns for technical indicators.
    """
    df['MA_20'] = df['Adjusted Close'].rolling(window=20).mean()
    df['RSI'] = 100 - (100 / (1 + df['Adjusted Close'].pct_change().apply(lambda x: (x + 1) / 1).rolling(window=14).mean()))
    df['Volatility'] = df['Adjusted Close'].rolling(window=20).std()
    return df

def preprocess_stock_data(self, ticker: str) -> None:
    """
    Preprocess data for a single stock, including loading, normalization, Fourier transformation, 
    and adding technical indicators. Saves the processed data to the specified directory.
    
    Parameters:
    - ticker (str): Stock ticker symbol.
    
    Raises:
    - PreprocessingException: If any step in the preprocessing fails.
    """
    try:
        # Load stock data for the current ticker, skipping the first two rows
        file_path = f"{self.folder_path}{ticker}_data.csv"

        # Check if the file exists
        if not os.path.exists(file_path):
            raise PreprocessingException(f"Data file for {ticker} not found at {file_path}")

        # Load CSV without setting the index to inspect the actual column names
        column_names = ['Date', 'Open', 'Adjusted Close', 'High', 'Low', 'Volume']
        df = pd.read_csv(file_path, skiprows=2,names=column_names)

        # Check if 'Date' is present in the columns; if so, set it as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is parsed as datetime
            df.set_index('Date', inplace=True)
        else:
            raise PreprocessingException(f"'Date' column missing in {file_path}")

        logging.info(f"Loaded data for {ticker}")

        # Normalize data and save scaler for future use
        print ("About to call normalize_data",file=sys.stderr)
        print ("Columns here are ",df.columns)
        scaler_path = os.path.join(self.folder_path, f"{ticker}_scaler.pkl")
        df, scaler = self.normalize_data(df, scaler_path=scaler_path)

        # Apply Fourier transform to detect cyclical behavior
        df = self.apply_fourier_transform(df)

        # Add technical indicators for enhanced analysis
        df = self.add_technical_indicators(df)

        # Fill any NaN values resulting from rolling calculations
        df.fillna(method='bfill', inplace=True)

        # Save the preprocessed data
        preprocessed_file = f"{self.folder_path}{ticker}_data_preprocessed.csv"
        df.to_csv(preprocessed_file)
        logging.info(f"Preprocessed data saved for {ticker}")

    except PreprocessingException as e:
        print ("Caught an error:  ")
        logging.error(f"Preprocessing error for {ticker}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error processing {ticker}: {e}")
        raise PreprocessingException(f"Failed to preprocess data for {ticker}") from e

def preprocess_portfolio(config: Dict[str, any]) -> None:
    """
    Preprocess a portfolio of stocks by iterating over each ticker, normalizing data, 
    applying Fourier transforms, and adding technical indicators. 
    Saves the processed data and scalers to the specified directory.
    
    Parameters:
    - config (Dict[str, any]): Configuration dictionary containing tickers, dates, 
      and folder path for data storage.
      
    Returns:
    - None
    """
    tickers = config['tickers']
    folder_path = config['folder_path']
    
    for ticker in tickers:
        try:
            # Load stock data for the current ticker
            df = pd.read_csv(f"{folder_path}{ticker}_data.csv", parse_dates=True,skiprows=2)

            # Verify if 'Date' was correctly set as the index
            if df.index.name != "Date":
            # If 'Date' is not the index, try setting it manually
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                else:
                    raise PreprocessingException(f"'Date' column missing in {file_path}")

            logging.info(f"Loaded data for {ticker}")
            
            
        except PreprocessingException as e:
            logging.error(f"Error processing {ticker}: {e}")

if __name__ == "__main__":
    # Load configuration settings from the config.yaml file
    config = load_config("config.yaml")
    preprocess_portfolio(config)


