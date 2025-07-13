#!/usr/bin/env python3
"""
Test script for VCP Parameter Tuner
Tests the chart generation functionality without running the full dashboard
"""

import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dashboard.vcp_parameter_tuner import VCPParameterTuner
from config import Config

def test_parameter_tuner():
    """Test the parameter tuner functionality"""
    print("🧪 Testing VCP Parameter Tuner...")
    
    try:
        # Create parameter tuner instance
        tuner = VCPParameterTuner()
        
        # Test with default parameters
        default_params = {
            'min_consolidation_days': Config.VCP_MIN_CONSOLIDATION_DAYS,
            'max_consolidation_days': Config.VCP_MAX_CONSOLIDATION_DAYS,
            'volatility_contraction_threshold': Config.VOLATILITY_CONTRACTION_THRESHOLD,
            'volume_decline_threshold': Config.VOLUME_DECLINE_THRESHOLD,
            'breakout_percentage': Config.BREAKOUT_PERCENTAGE,
            'breakout_volume_multiplier': Config.BREAKOUT_VOLUME_MULTIPLIER,
            'bollinger_period': Config.BOLLINGER_BANDS_PERIOD,
            'bollinger_std': Config.BOLLINGER_BANDS_STD,
            'rsi_period': Config.RSI_PERIOD,
            'stop_loss_percentage': Config.STOP_LOSS_PERCENTAGE,
            'profit_target_multiplier': Config.PROFIT_TARGET_MULTIPLIER
        }
        
        print("✅ Testing data generation...")
        df = tuner.generate_ideal_vcp_data(default_params)
        print(f"   Generated {len(df)} data points")
        
        print("✅ Testing technical indicators...")
        df_with_indicators = tuner.calculate_technical_indicators(df, default_params)
        print(f"   Added technical indicators: {list(df_with_indicators.columns)}")
        
        print("✅ Testing chart generation...")
        vcp_fig = tuner.create_vcp_chart(default_params)
        volume_fig = tuner.create_volume_chart(default_params)
        volatility_fig = tuner.create_volatility_chart(default_params)
        print("   Generated all chart types successfully")
        
        print("✅ Testing metrics display...")
        metrics = tuner.create_metrics_display(default_params)
        print("   Generated metrics display successfully")
        
        # Test with different parameters
        print("✅ Testing parameter variations...")
        aggressive_params = default_params.copy()
        aggressive_params['min_consolidation_days'] = 10
        aggressive_params['volatility_contraction_threshold'] = 0.7
        aggressive_params['volume_decline_threshold'] = 0.2
        
        df_aggressive = tuner.generate_ideal_vcp_data(aggressive_params)
        print(f"   Generated aggressive pattern with {len(df_aggressive)} data points")
        
        conservative_params = default_params.copy()
        conservative_params['min_consolidation_days'] = 25
        conservative_params['volatility_contraction_threshold'] = 0.3
        conservative_params['volume_decline_threshold'] = 0.5
        
        df_conservative = tuner.generate_ideal_vcp_data(conservative_params)
        print(f"   Generated conservative pattern with {len(df_conservative)} data points")
        
        print("\n🎉 All tests passed! Parameter tuner is working correctly.")
        print("\n📊 You can now run the interactive dashboard:")
        print("   python run_parameter_tuner.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_parameter_tuner()
    sys.exit(0 if success else 1) 