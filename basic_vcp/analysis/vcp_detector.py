"""
VCP (Volatility Contraction Pattern) Detection Engine
Identifies VCP patterns, swing points, and breakout levels
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import logging

@dataclass
class SwingPoint:
    """Represents a swing high or low point"""
    date: pd.Timestamp
    price: float
    point_type: str  # 'high' or 'low'
    strength: float  # How strong the swing point is (0-1)
    volume: float = 0.0  # Volume at this swing point (0.0 if not available)

@dataclass
class VCPPattern:
    """Represents a detected VCP pattern"""
    ticker: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    consolidation_days: int
    volatility_contraction: float
    volume_decline: float
    resistance_level: float
    support_level: float
    breakout_level: float
    pattern_strength: float  # 0-1 score
    swing_points: List[SwingPoint]
    status: str  # 'forming', 'breakout', 'failed'

@dataclass
class BreakoutSignal:
    """Represents a breakout signal"""
    ticker: str
    date: pd.Timestamp
    price: float
    volume: float
    signal_type: str  # 'breakout', 'breakdown', 'false_breakout'
    confidence: float  # 0-1 confidence score
    stop_loss: float
    profit_target: float
    risk_reward_ratio: float

class VCPDetector:
    def __init__(self, config: Dict = None):
        """
        Initialize VCP detector
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default parameters
        self.min_consolidation_days = self.config.get('min_consolidation_days', 20)
        self.max_consolidation_days = self.config.get('max_consolidation_days', 252)
        self.volatility_threshold = self.config.get('volatility_contraction_threshold', 0.3)
        self.volume_threshold = self.config.get('volume_decline_threshold', 0.5)
        self.swing_lookback = self.config.get('lookback_days', 10)
        self.swing_threshold = self.config.get('threshold', 0.02)
        
    def detect_vcp_patterns(self, df: pd.DataFrame, ticker: str) -> List[VCPPattern]:
        """
        Detect VCP patterns in the given data
        
        Args:
            df: DataFrame with OHLC and technical indicators
            ticker: Stock ticker symbol
            
        Returns:
            List of detected VCP patterns
        """
        patterns = []
        
        # Find consolidation periods
        consolidation_periods = self._find_consolidation_periods(df)
        
        for start_idx, end_idx in consolidation_periods:
            # Check if this period meets VCP criteria
            if self._is_vcp_pattern(df, start_idx, end_idx):
                pattern = self._create_vcp_pattern(df, ticker, start_idx, end_idx)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _find_consolidation_periods(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Find periods of price consolidation
        
        Args:
            df: DataFrame with price data
            
        Returns:
            List of (start_index, end_index) tuples for consolidation periods
        """
        consolidation_periods = []
        
        for i in range(len(df) - self.min_consolidation_days):
            # Check if we have a consolidation period starting at index i
            if self._is_consolidation_start(df, i):
                end_idx = self._find_consolidation_end(df, i)
                if end_idx and (end_idx - i) >= self.min_consolidation_days:
                    consolidation_periods.append((i, end_idx))
        
        return consolidation_periods
    
    def _is_consolidation_start(self, df: pd.DataFrame, start_idx: int) -> bool:
        """
        Check if a consolidation period starts at the given index
        
        Args:
            df: DataFrame with price data
            start_idx: Starting index to check
            
        Returns:
            True if consolidation starts here
        """
        if start_idx < 20:  # Need some history for comparison
            return False
        
        # Check if price movement has decreased
        recent_volatility = df['volatility'].iloc[start_idx-20:start_idx].mean()
        current_volatility = df['volatility'].iloc[start_idx]
        
        # Check if Bollinger Band width is contracting
        bb_width_recent = df['bb_width'].iloc[start_idx-20:start_idx].mean()
        bb_width_current = df['bb_width'].iloc[start_idx]
        
        return (current_volatility < recent_volatility * 0.9 and 
                bb_width_current < bb_width_recent * 0.9)
    
    def _find_consolidation_end(self, df: pd.DataFrame, start_idx: int) -> Optional[int]:
        """
        Find the end of a consolidation period
        
        Args:
            df: DataFrame with price data
            start_idx: Starting index of consolidation
            
        Returns:
            End index of consolidation or None
        """
        for i in range(start_idx + self.min_consolidation_days, 
                      min(start_idx + self.max_consolidation_days, len(df))):
            
            # Check if consolidation has ended (breakout or breakdown)
            if self._is_consolidation_end(df, start_idx, i):
                return i
        
        # If no breakout/breakdown found, return the end of the search period
        # This allows us to find consolidation periods that are still forming
        end_idx = min(start_idx + self.max_consolidation_days, len(df)) - 1
        if end_idx > start_idx + self.min_consolidation_days:
            return end_idx
        
        return None
    
    def _is_consolidation_end(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
        """
        Check if consolidation has ended
        
        Args:
            df: DataFrame with price data
            start_idx: Start of consolidation
            end_idx: Current index to check
            
        Returns:
            True if consolidation has ended
        """
        # Get price range during consolidation
        consolidation_data = df.iloc[start_idx:end_idx+1]
        high = consolidation_data['high'].max()
        low = consolidation_data['low'].min()
        current_price = df['close'].iloc[end_idx]
        
        # Check for breakout above resistance
        if current_price > high * 1.02:  # 2% above resistance
            return True
        
        # Check for breakdown below support
        if current_price < low * 0.98:  # 2% below support
            return True
        
        return False
    
    def _is_vcp_pattern(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
        """
        Check if a consolidation period qualifies as a VCP pattern
        
        Args:
            df: DataFrame with price data
            start_idx: Start index of consolidation
            end_idx: End index of consolidation
            
        Returns:
            True if this is a valid VCP pattern
        """
        consolidation_data = df.iloc[start_idx:end_idx+1]
        
        # Check volatility contraction
        bb_width_start = df['bb_width'].iloc[start_idx]
        bb_width_end = df['bb_width'].iloc[end_idx]
        
        # Avoid divide by zero
        if bb_width_start <= 0:
            return False
        
        volatility_contraction = bb_width_end / bb_width_start
        
        if volatility_contraction > self.volatility_threshold:
            return False
        
        # Check volume decline during consolidation
        if 'volume' in df.columns:
            volume_start = df['volume'].iloc[start_idx:start_idx+5].mean()
            volume_end = df['volume'].iloc[end_idx-4:end_idx+1].mean()
            volume_decline = volume_end / volume_start
            
            if volume_decline > self.volume_threshold:
                return False
        
        # Check for proper price structure (higher lows, lower highs)
        if not self._has_proper_structure(consolidation_data):
            return False
        
        return True
    
    def _has_proper_structure(self, df: pd.DataFrame) -> bool:
        """
        Check if price structure is proper for VCP (higher lows, lower highs)
        
        Args:
            df: DataFrame with price data
            
        Returns:
            True if structure is proper
        """
        # Find swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(1, len(df) - 1):
            if df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1]:
                swing_highs.append(df['high'].iloc[i])
            if df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1]:
                swing_lows.append(df['low'].iloc[i])
        
        # Check for higher lows and lower highs
        if len(swing_lows) >= 2:
            if swing_lows[-1] <= swing_lows[-2]:  # Not higher lows
                return False
        
        if len(swing_highs) >= 2:
            if swing_highs[-1] >= swing_highs[-2]:  # Not lower highs
                return False
        
        return True
    
    def _create_vcp_pattern(self, df: pd.DataFrame, ticker: str, 
                           start_idx: int, end_idx: int) -> Optional[VCPPattern]:
        """
        Create a VCP pattern object
        
        Args:
            df: DataFrame with price data
            ticker: Stock ticker
            start_idx: Start index of pattern
            end_idx: End index of pattern
            
        Returns:
            VCPPattern object or None
        """
        consolidation_data = df.iloc[start_idx:end_idx+1]
        
        # Calculate pattern metrics
        bb_width_start = df['bb_width'].iloc[start_idx]
        bb_width_end = df['bb_width'].iloc[end_idx]
        
        # Avoid divide by zero
        if bb_width_start <= 0:
            return None
        
        volatility_contraction = bb_width_end / bb_width_start
        
        volume_decline = 1.0
        if 'volume' in df.columns:
            volume_start = df['volume'].iloc[start_idx:start_idx+5].mean()
            volume_end = df['volume'].iloc[end_idx-4:end_idx+1].mean()
            volume_decline = volume_end / volume_start
        
        # Find resistance and support levels
        resistance = consolidation_data['high'].max()
        support = consolidation_data['low'].min()
        breakout_level = resistance * 1.02  # 2% above resistance
        
        # Calculate pattern strength
        pattern_strength = self._calculate_pattern_strength(
            volatility_contraction, volume_decline, end_idx - start_idx
        )
        
        # Find swing points
        swing_points = self._find_swing_points(consolidation_data)
        
        return VCPPattern(
            ticker=ticker,
            start_date=df.index[start_idx],
            end_date=df.index[end_idx],
            consolidation_days=end_idx - start_idx,
            volatility_contraction=volatility_contraction,
            volume_decline=volume_decline,
            resistance_level=resistance,
            support_level=support,
            breakout_level=breakout_level,
            pattern_strength=pattern_strength,
            swing_points=swing_points,
            status='forming'
        )
    
    def _calculate_pattern_strength(self, volatility_contraction: float, 
                                  volume_decline: float, days: int) -> float:
        """
        Calculate the strength of a VCP pattern (0-1)
        
        Args:
            volatility_contraction: Volatility contraction ratio
            volume_decline: Volume decline ratio
            days: Number of consolidation days
            
        Returns:
            Pattern strength score (0-1)
        """
        # Volatility contraction score (lower is better)
        vol_score = max(0, 1 - volatility_contraction)
        
        # Volume decline score (lower is better)
        vol_decline_score = max(0, 1 - volume_decline)
        
        # Duration score (optimal range)
        if 30 <= days <= 90:
            duration_score = 1.0
        elif 20 <= days <= 120:
            duration_score = 0.8
        else:
            duration_score = 0.5
        
        # Weighted average
        strength = (vol_score * 0.4 + vol_decline_score * 0.3 + duration_score * 0.3)
        return min(1.0, max(0.0, strength))
    
    def _find_swing_points(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Find swing high and low points in the data
        
        Args:
            df: DataFrame with price data
            
        Returns:
            List of SwingPoint objects
        """
        swing_points = []
        
        for i in range(1, len(df) - 1):
            # Swing high
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i+1]):
                
                strength = self._calculate_swing_strength(df, i, 'high')
                
                # Handle volume data - ensure it's never None
                if 'volume' in df.columns:
                    volume = df['volume'].iloc[i]
                    if volume is None or pd.isna(volume):
                        volume = 0.0
                else:
                    volume = 0.0
                
                swing_points.append(SwingPoint(
                    date=df.index[i],
                    price=df['high'].iloc[i],
                    point_type='high',
                    strength=strength,
                    volume=volume
                ))
            
            # Swing low
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i+1]):
                
                strength = self._calculate_swing_strength(df, i, 'low')
                
                # Handle volume data - ensure it's never None
                if 'volume' in df.columns:
                    volume = df['volume'].iloc[i]
                    if volume is None or pd.isna(volume):
                        volume = 0.0
                else:
                    volume = 0.0
                
                swing_points.append(SwingPoint(
                    date=df.index[i],
                    price=df['low'].iloc[i],
                    point_type='low',
                    strength=strength,
                    volume=volume
                ))
        
        return swing_points
    
    def _calculate_swing_strength(self, df: pd.DataFrame, idx: int, point_type: str) -> float:
        """
        Calculate the strength of a swing point (0-1)
        
        Args:
            df: DataFrame with price data
            idx: Index of the swing point
            point_type: 'high' or 'low'
            
        Returns:
            Swing point strength (0-1)
        """
        if point_type == 'high':
            current = df['high'].iloc[idx]
            left = df['high'].iloc[max(0, idx-5):idx].max()
            right = df['high'].iloc[idx+1:min(len(df), idx+6)].max()
            
            # Calculate how much higher this point is than surrounding points
            left_diff = (current - left) / left if left > 0 else 0
            right_diff = (current - right) / right if right > 0 else 0
            
            strength = (left_diff + right_diff) / 2
            return min(1.0, max(0.0, strength * 10))  # Scale to 0-1
            
        else:  # low
            current = df['low'].iloc[idx]
            left = df['low'].iloc[max(0, idx-5):idx].min()
            right = df['low'].iloc[idx+1:min(len(df), idx+6)].min()
            
            # Calculate how much lower this point is than surrounding points
            left_diff = (left - current) / current if current > 0 else 0
            right_diff = (right - current) / current if current > 0 else 0
            
            strength = (left_diff + right_diff) / 2
            return min(1.0, max(0.0, strength * 10))  # Scale to 0-1
    
    def detect_breakout_signals(self, df: pd.DataFrame, vcp_pattern: VCPPattern) -> List[BreakoutSignal]:
        """
        Detect breakout signals for a VCP pattern - More conservative approach
        
        Args:
            df: DataFrame with price data
            vcp_pattern: VCP pattern to analyze
            
        Returns:
            List of breakout signals
        """
        signals = []
        
        try:
            # Look for breakouts after the pattern end date
            pattern_end_idx = df.index.get_loc(vcp_pattern.end_date)
            
            # More conservative breakout detection
            breakout_threshold = vcp_pattern.resistance_level * 1.02  # 2% above resistance
            breakdown_threshold = vcp_pattern.support_level * 0.98   # 2% below support
            
            # Limit the number of signals per pattern to avoid spam
            max_signals_per_pattern = 5
            signals_found = 0
            
            for i in range(pattern_end_idx + 1, len(df)):
                if signals_found >= max_signals_per_pattern:
                    break
                    
                current_price = df['close'].iloc[i]
                
                # Handle None values in volume data
                if 'volume' in df.columns:
                    current_volume = df['volume'].iloc[i]
                    if current_volume is None or pd.isna(current_volume):
                        current_volume = 0
                else:
                    current_volume = 0
                
                # Check for breakout above resistance
                if current_price > breakout_threshold:
                    signal = self._create_breakout_signal(
                        df, i, vcp_pattern, 'breakout', current_price, current_volume
                    )
                    signals.append(signal)
                    signals_found += 1
                
                # Check for breakdown below support
                elif current_price < breakdown_threshold:
                    signal = self._create_breakout_signal(
                        df, i, vcp_pattern, 'breakdown', current_price, current_volume
                    )
                    signals.append(signal)
                    signals_found += 1
        
        except Exception as e:
            # Log error but don't crash
            self.logger.warning(f"Error detecting breakout signals: {e}")
        
        return signals
    
    def _create_breakout_signal(self, df: pd.DataFrame, idx: int, vcp_pattern: VCPPattern,
                               signal_type: str, price: float, volume: float) -> BreakoutSignal:
        """
        Create a breakout signal object
        
        Args:
            df: DataFrame with price data
            idx: Index of the signal
            vcp_pattern: Associated VCP pattern
            signal_type: Type of signal
            price: Current price
            volume: Current volume
            
        Returns:
            BreakoutSignal object
        """
        # Calculate stop loss and profit target based on actual market levels
        if signal_type == 'breakout':
            # For breakouts: stop loss below support, profit target at next resistance
            stop_loss = vcp_pattern.support_level * 0.95  # 5% below support
            
            # Find next resistance level (look for swing highs after pattern)
            next_resistance = self._find_next_resistance_level(df, idx, price)
            if next_resistance and next_resistance > price:
                profit_target = next_resistance
            else:
                # Fallback: use 1.5x the pattern's height
                pattern_height = vcp_pattern.resistance_level - vcp_pattern.support_level
                profit_target = price + pattern_height * 1.5
        else:  # breakdown
            # For breakdowns: stop loss above resistance, profit target at next support
            stop_loss = vcp_pattern.resistance_level * 1.05  # 5% above resistance
            
            # Find next support level (look for swing lows after pattern)
            next_support = self._find_next_support_level(df, idx, price)
            if next_support and next_support < price:
                profit_target = next_support
            else:
                # Fallback: use 1.5x the pattern's height
                pattern_height = vcp_pattern.resistance_level - vcp_pattern.support_level
                profit_target = price - pattern_height * 1.5
        
        # Calculate actual risk/reward ratio
        risk = abs(price - stop_loss)
        reward = abs(profit_target - price)
        risk_reward = reward / risk if risk > 0 else 1.0
        
        # Calculate confidence based on pattern strength and volume
        confidence = vcp_pattern.pattern_strength
        if volume > 0 and 'volume' in df.columns:
            try:
                # Get recent volume data, handling None values
                recent_volume = df['volume'].iloc[max(0, idx-20):idx]
                recent_volume = recent_volume.dropna()  # Remove None/NaN values
                
                if len(recent_volume) > 0:
                    avg_volume = recent_volume.mean()
                    if avg_volume > 0:
                        volume_factor = min(1.0, volume / avg_volume)
                        confidence = (confidence + volume_factor) / 2
            except Exception:
                # If volume calculation fails, just use pattern strength
                pass
        
        return BreakoutSignal(
            ticker=vcp_pattern.ticker,
            date=df.index[idx],
            price=price,
            volume=volume,
            signal_type=signal_type,
            confidence=confidence,
            stop_loss=stop_loss,
            profit_target=profit_target,
            risk_reward_ratio=risk_reward
        )
    
    def _find_next_resistance_level(self, df: pd.DataFrame, current_idx: int, current_price: float) -> Optional[float]:
        """
        Find the next resistance level after the current position
        
        Args:
            df: DataFrame with price data
            current_idx: Current position index
            current_price: Current price
            
        Returns:
            Next resistance level or None if not found
        """
        try:
            # Look ahead up to 50 bars for swing highs
            look_ahead = min(50, len(df) - current_idx - 1)
            if look_ahead <= 0:
                return None
            
            future_data = df.iloc[current_idx + 1:current_idx + 1 + look_ahead]
            
            # Find swing highs in the future data
            swing_highs = []
            for i in range(1, len(future_data) - 1):
                if (future_data['high'].iloc[i] > future_data['high'].iloc[i-1] and 
                    future_data['high'].iloc[i] > future_data['high'].iloc[i+1]):
                    swing_highs.append(future_data['high'].iloc[i])
            
            # Return the lowest swing high above current price
            valid_resistances = [h for h in swing_highs if h > current_price]
            return min(valid_resistances) if valid_resistances else None
            
        except Exception:
            return None
    
    def _find_next_support_level(self, df: pd.DataFrame, current_idx: int, current_price: float) -> Optional[float]:
        """
        Find the next support level after the current position
        
        Args:
            df: DataFrame with price data
            current_idx: Current position index
            current_price: Current price
            
        Returns:
            Next support level or None if not found
        """
        try:
            # Look ahead up to 50 bars for swing lows
            look_ahead = min(50, len(df) - current_idx - 1)
            if look_ahead <= 0:
                return None
            
            future_data = df.iloc[current_idx + 1:current_idx + 1 + look_ahead]
            
            # Find swing lows in the future data
            swing_lows = []
            for i in range(1, len(future_data) - 1):
                if (future_data['low'].iloc[i] < future_data['low'].iloc[i-1] and 
                    future_data['low'].iloc[i] < future_data['low'].iloc[i+1]):
                    swing_lows.append(future_data['low'].iloc[i])
            
            # Return the highest swing low below current price
            valid_supports = [l for l in swing_lows if l < current_price]
            return max(valid_supports) if valid_supports else None
            
        except Exception:
            return None 