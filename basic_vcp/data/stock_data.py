"""
Stock data loader for VCP Detection System
Reads from existing CSV files in the data directory
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
import sqlite3

class StockDataLoader:
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the stock data loader
        
        Args:
            data_dir: Path to the directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        
    def get_available_tickers(self) -> List[str]:
        """Get list of available ticker symbols"""
        csv_files = list(self.data_dir.glob("*.csv"))
        tickers = [f.stem for f in csv_files]
        return sorted(tickers)
    
    def load_stock_data(self, ticker: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load stock data for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLC data or None if file not found
        """
        file_path = self.data_dir / f"{ticker}.csv"
        
        if not file_path.exists():
            self.logger.warning(f"Data file not found for ticker: {ticker}")
            return None
        
        try:
            # Load CSV data
            df = pd.read_csv(file_path)
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # Filter by date range if specified
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            # Add technical indicators
            df = self._add_technical_indicators(df.copy())
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data for {ticker}: {str(e)}")
            return None
    
    def load_multiple_stocks(self, tickers: List[str], start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple stocks
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping tickers to their DataFrames
        """
        stock_data = {}
        
        for ticker in tickers:
            data = self.load_stock_data(ticker, start_date, end_date)
            if data is not None:
                stock_data[ticker] = data
        
        return stock_data
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with added technical indicators
        """
        df_copy = df.copy()
        
        # Calculate returns
        df_copy['returns'] = df_copy['close'].pct_change()
        
        # Calculate volatility (rolling standard deviation of returns)
        df_copy['volatility'] = df_copy['returns'].rolling(window=20).std()
        
        # Calculate moving averages
        df_copy['sma_20'] = df_copy['close'].rolling(window=20).mean()
        df_copy['sma_50'] = df_copy['close'].rolling(window=50).mean()
        df_copy['sma_200'] = df_copy['close'].rolling(window=200).mean()
        
        # Calculate Bollinger Bands
        df_copy['bb_middle'] = df_copy['close'].rolling(window=20).mean()
        bb_std = df_copy['close'].rolling(window=20).std()
        df_copy['bb_upper'] = df_copy['bb_middle'] + (bb_std * 2)
        df_copy['bb_lower'] = df_copy['bb_middle'] - (bb_std * 2)
        df_copy['bb_width'] = (df_copy['bb_upper'] - df_copy['bb_lower']) / df_copy['bb_middle']
        
        # Calculate RSI
        delta = df_copy['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_copy['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate price ranges
        df_copy['daily_range'] = (df_copy['high'] - df_copy['low']) / df_copy['close']
        df_copy['range_20'] = df_copy['daily_range'].rolling(window=20).mean()
        
        # Calculate volume indicators (if volume data available)
        if 'volume' in df_copy.columns:
            df_copy['volume_sma'] = df_copy['volume'].rolling(window=20).mean()
            df_copy['volume_ratio'] = df_copy['volume'] / df_copy['volume_sma']
        
        return df_copy
    
    def get_recent_data(self, ticker: str, days: int = 252) -> Optional[pd.DataFrame]:
        """
        Get recent data for a ticker
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            DataFrame with recent data
        """
        df = self.load_stock_data(ticker)
        if df is not None:
            return df.tail(days)
        return None
    
    def validate_data_quality(self, df: pd.DataFrame, min_days: int = 252) -> bool:
        """
        Validate data quality for VCP analysis - Optimized for breakout detection
        
        Args:
            df: DataFrame to validate
            min_days: Minimum number of days required
            
        Returns:
            True if data meets quality standards
        """
        if df is None or len(df) < min_days:
            return False
        
        # Check for sufficient price movement (further relaxed for breakout detection)
        price_range = (df['close'].max() - df['close'].min()) / df['close'].min()
        if price_range < 0.02:  # Less than 2% price range (very relaxed for breakouts)
            return False
        
        # Check for missing data (further relaxed for breakout detection)
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > 0.3:  # More than 30% missing data (very relaxed)
            return False
        
        return True 

class SQLiteStockDataLoader:
    def __init__(self, db_file: str = "stock_data.db", table_name: str = "stock_prices"):
        self.db_file = db_file
        self.table_name = table_name

    def get_available_tickers(self) -> List[str]:
        with sqlite3.connect(self.db_file) as conn:
            query = f"SELECT DISTINCT ticker FROM {self.table_name}"
            tickers = pd.read_sql_query(query, conn)["ticker"].tolist()
        return sorted(tickers)

    def load_stock_data(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        with sqlite3.connect(self.db_file) as conn:
            query = f"SELECT * FROM {self.table_name} WHERE ticker = ?"
            params = [ticker]
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            df = pd.read_sql_query(query, conn, params=params)
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        # Add technical indicators (reuse existing method)
        df = StockDataLoader()._add_technical_indicators(df)
        return df

    def validate_data_quality(self, df: pd.DataFrame, min_days: int = 252) -> bool:
        return StockDataLoader().validate_data_quality(df, min_days)

    def get_recent_data(self, ticker: str, days: int = 252) -> Optional[pd.DataFrame]:
        df = self.load_stock_data(ticker)
        if df is not None:
            return df.tail(days)
        return None 