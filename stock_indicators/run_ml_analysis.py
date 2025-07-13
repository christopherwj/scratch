import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import fetch_data
from src.ml_backtester import MLBacktester

def run_ml_vs_classical_analysis():
    """
    Runs a comprehensive analysis comparing ML strategy vs Classical strategy.
    """
    print("="*80)
    print("MACHINE LEARNING vs CLASSICAL TRADING STRATEGY ANALYSIS")
    print("="*80)
    print("Using RTX 3090 Ti GPU acceleration for ML training")
    print("="*80)
    
    # Load data
    print("\n1. Loading AAPL data...")
    data = fetch_data('AAPL', '2018-01-01', '2023-12-31')
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"   Data loaded: {len(data)} days from {data.index.min().date()} to {data.index.max().date()}")
    
    # Split data for training and testing
    split_date = '2022-01-01'
    train_data = data.loc[:split_date]
    test_data = data.loc[split_date:]
    
    print(f"\n2. Data split:")
    print(f"   Training period: {train_data.index.min().date()} to {train_data.index.max().date()} ({len(train_data)} days)")
    print(f"   Testing period: {test_data.index.min().date()} to {test_data.index.max().date()} ({len(test_data)} days)")
    
    # Initialize ML backtester
    print(f"\n3. Initializing ML trading system...")
    ml_backtester = MLBacktester(initial_cash=10000)
    
    # Train ML model
    print(f"\n4. Training ML models on historical data...")
    try:
        training_results = ml_backtester.train_ml_model(
            train_data, 
            lookahead_days=5,  # Predict 5 days ahead
            threshold=0.02     # 2% threshold for buy/sell signals
        )
        print(f"   ✓ ML models trained successfully")
        
        # Print training results
        print(f"\n   Training Results:")
        for model_name, accuracy in training_results.items():
            print(f"   - {model_name.capitalize()}: {accuracy:.1%} accuracy")
            
    except Exception as e:
        print(f"   ✗ ML training failed: {e}")
        return
    
    # Backtest ML strategy
    print(f"\n5. Backtesting ML strategy on test data...")
    try:
        portfolio_value, trades = ml_backtester.backtest_ml_strategy(test_data)
        print(f"   ✓ ML backtest completed")
        print(f"   - Total trades executed: {len(trades)}")
        
    except Exception as e:
        print(f"   ✗ ML backtesting failed: {e}")
        return
    
    # Compare with classical strategy
    print(f"\n6. Comparing ML vs Classical strategy...")
    try:
        ml_metrics, classical_metrics = ml_backtester.compare_with_classical_strategy(test_data)
        
        # Calculate buy and hold performance
        buy_hold_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0] - 1) * 100
        
        print(f"\n" + "="*80)
        print("FINAL PERFORMANCE COMPARISON")
        print("="*80)
        
        print(f"\n📊 RETURNS:")
        print(f"   ML Strategy:      {ml_metrics['Total Return (%)']:+8.2f}%")
        print(f"   Classical MACD:   {classical_metrics['Total Return (%)']:+8.2f}%") 
        print(f"   Buy & Hold:       {buy_hold_return:+8.2f}%")
        
        print(f"\n📈 RISK-ADJUSTED PERFORMANCE:")
        print(f"   ML Sharpe Ratio:       {ml_metrics['Sharpe Ratio']:8.3f}")
        print(f"   Classical Sharpe:      {classical_metrics['Sharpe Ratio']:8.3f}")
        
        print(f"\n💰 FINAL PORTFOLIO VALUES:")
        print(f"   ML Strategy:      ${ml_metrics['Final Portfolio Value ($)']:10,.2f}")
        print(f"   Classical MACD:   ${classical_metrics['Final Portfolio Value ($)']:10,.2f}")
        print(f"   Buy & Hold:       ${10000 * (1 + buy_hold_return/100):10,.2f}")
        
        print(f"\n🎯 TRADING ACTIVITY:")
        print(f"   ML Trades:        {ml_metrics['Total Trades']:8.0f}")
        print(f"   Classical Trades: {classical_metrics['Total Trades']:8.0f}")
        
        # Determine winner
        ml_return = ml_metrics['Total Return (%)']
        classical_return = classical_metrics['Total Return (%)']
        
        print(f"\n🏆 WINNER:")
        if ml_return > classical_return and ml_return > buy_hold_return:
            improvement_vs_classical = ((ml_return - classical_return) / abs(classical_return)) * 100
            improvement_vs_buyhold = ((ml_return - buy_hold_return) / abs(buy_hold_return)) * 100
            print(f"   🥇 ML STRATEGY WINS!")
            print(f"   - {improvement_vs_classical:+.1f}% better than Classical")
            print(f"   - {improvement_vs_buyhold:+.1f}% better than Buy & Hold")
        elif classical_return > buy_hold_return:
            print(f"   🥈 Classical strategy wins")
        else:
            print(f"   🥉 Buy & Hold wins")
            
        # Feature importance
        print(f"\n🔍 TOP ML FEATURES:")
        top_features = ml_backtester.ml_trader.get_feature_importance(5)
        if top_features:
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"   {i}. {feature}: {importance:.4f}")
        
        print(f"\n" + "="*80)
        
    except Exception as e:
        print(f"   ✗ Comparison failed: {e}")
        return
    
    # Generate plots
    print(f"\n7. Generating performance plots...")
    try:
        ml_backtester.plot_results(test_data, 'ml_vs_classical_results.png')
        print(f"   ✓ Performance plots saved")
        
    except Exception as e:
        print(f"   ⚠ Plotting failed: {e}")
    
    print(f"\n✅ Analysis completed successfully!")
    print(f"📁 Check 'ml_vs_classical_results.png' for visual results")

if __name__ == '__main__':
    run_ml_vs_classical_analysis() 