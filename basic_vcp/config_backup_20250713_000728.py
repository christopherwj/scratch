"""
Configuration settings for the VCP Detection System
"""

import os
from typing import List, Dict, Any

class Config:
    # Data settings
    DEFAULT_PERIOD = "1y"  # Default data period
    DEFAULT_INTERVAL = "1d"  # Default interval
    MIN_DATA_POINTS = 252  # Minimum data points required (1 year)
    
    # VCP Pattern settings
    VCP_MIN_CONSOLIDATION_DAYS = 10  # Minimum days for consolidation
    VCP_MAX_CONSOLIDATION_DAYS = 60 # Maximum days for consolidation
    VOLATILITY_CONTRACTION_THRESHOLD = 0.5  # BB width contraction threshold
    VOLUME_DECLINE_THRESHOLD = 0.2  # Volume decline threshold during consolidation
    
    # Swing point settings
    SWING_POINT_LOOKBACK = 10  # Days to look back for swing points
    SWING_POINT_THRESHOLD = 0.02  # Minimum swing point threshold (2%)
    
    # Breakout settings
    BREAKOUT_CONFIRMATION_DAYS = 3  # Days to confirm breakout
    BREAKOUT_VOLUME_MULTIPLIER = 1  # Volume multiplier for breakout confirmation
    BREAKOUT_PERCENTAGE = 0.03  # Percentage above resistance for breakout
    
    # Risk management
    STOP_LOSS_PERCENTAGE = 0.05  # 5% stop loss
    PROFIT_TARGET_MULTIPLIER = 2.0  # 2:1 risk-reward ratio
    MAX_POSITION_SIZE = 0.02  # 2% of portfolio per trade
    
    # Technical indicators
    BOLLINGER_BANDS_PERIOD = 20
    BOLLINGER_BANDS_STD = 2
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    SMA_PERIODS = [20, 50, 200]
    
    # Stock universe
    DEFAULT_STOCKS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
        "AMD", "INTC", "CRM", "ADBE", "PYPL", "SQ", "ZM", "UBER", "LYFT",
        "SPOT", "SNAP", "TWTR", "SHOP", "ROKU", "ZM", "DOCU", "CRWD"
    ]
    
    # Database settings
    DATABASE_URL = "sqlite:///vcp_database.db"
    
    # API settings
    YAHOO_FINANCE_TIMEOUT = 30
    MAX_CONCURRENT_REQUESTS = 5
    
    # Dashboard settings
    DASHBOARD_PORT = 8050
    DASHBOARD_HOST = "0.0.0.0"
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "vcp_system.log"
    
    @classmethod
    def get_vcp_parameters(cls) -> Dict[str, Any]:
        """Get VCP detection parameters"""
        return {
            "min_consolidation_days": cls.VCP_MIN_CONSOLIDATION_DAYS,
            "max_consolidation_days": cls.VCP_MAX_CONSOLIDATION_DAYS,
            "volatility_contraction_threshold": cls.VOLATILITY_CONTRACTION_THRESHOLD,
            "volume_decline_threshold": cls.VOLUME_DECLINE_THRESHOLD
        }
    
    @classmethod
    def get_swing_point_parameters(cls) -> Dict[str, Any]:
        """Get swing point detection parameters"""
        return {
            "lookback_days": cls.SWING_POINT_LOOKBACK,
            "threshold": cls.SWING_POINT_THRESHOLD
        }
    
    @classmethod
    def get_breakout_parameters(cls) -> Dict[str, Any]:
        """Get breakout detection parameters"""
        return {
            "confirmation_days": cls.BREAKOUT_CONFIRMATION_DAYS,
            "volume_multiplier": cls.BREAKOUT_VOLUME_MULTIPLIER,
            "percentage": cls.BREAKOUT_PERCENTAGE
        }
    
    @classmethod
    def get_rsi_parameters(cls) -> Dict[str, Any]:
        """Get RSI parameters"""
        return {
            "period": cls.RSI_PERIOD,
            "overbought": cls.RSI_OVERBOUGHT,
            "oversold": cls.RSI_OVERSOLD
        } 