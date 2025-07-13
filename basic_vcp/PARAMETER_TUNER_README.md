# VCP Parameter Tuner - Interactive GUI

## Overview

The VCP Parameter Tuner is an interactive web-based dashboard that allows you to adjust VCP (Volatility Contraction Pattern) detection parameters in real-time and see how these changes affect the ideal VCP pattern that the model will look for when scanning stocks.

## Features

### 🎛️ Interactive Parameter Controls
- **Consolidation Settings**: Adjust minimum and maximum consolidation days
- **Contraction Thresholds**: Fine-tune volatility and volume decline thresholds
- **Breakout Settings**: Configure breakout percentage and volume multiplier
- **Technical Indicators**: Modify Bollinger Bands and RSI parameters
- **Risk Management**: Set stop loss and profit target ratios

### 📊 Real-Time Chart Updates
- **Ideal VCP Pattern Chart**: Shows the perfect VCP pattern based on current parameters
- **Volume Pattern**: Displays volume behavior during consolidation and breakout
- **Volatility Contraction**: Visualizes Bollinger Band width changes
- **Parameter Summary**: Real-time display of all current settings

### 💾 Export & Import System
- **Export Parameters**: Save optimized settings to JSON files
- **Import Parameters**: Apply exported settings back to config.py
- **Backup System**: Automatic backup of original configuration

## Quick Start

### 1. Launch the Parameter Tuner

```bash
python run_parameter_tuner.py
```

The dashboard will be available at: http://localhost:8051

### 2. Adjust Parameters

Use the sliders in the left panel to adjust parameters:
- **Min Consolidation Days**: 5-50 days (default: 5)
- **Max Consolidation Days**: 20-100 days (default: 30)
- **Volatility Contraction**: 10%-90% (default: 50%)
- **Volume Decline**: 10%-90% (default: 30%)
- **Breakout Percentage**: 1%-5% (default: 2%)
- **Volume Multiplier**: 1.0x-3.0x (default: 1.5x)

### 3. Observe Chart Changes

The charts update automatically as you adjust parameters:
- **Price Chart**: Shows how the ideal VCP pattern changes
- **Volume Chart**: Displays volume behavior patterns
- **Volatility Chart**: Shows Bollinger Band width contraction

### 4. Export Optimized Parameters

Click the "Export Parameters" button to save your settings to a JSON file.

### 5. Apply to Main System

Use the configuration updater to apply your optimized parameters:

```bash
python config_updater.py
```

## Understanding the Parameters

### Consolidation Settings

**Min/Max Consolidation Days**
- Controls how long the VCP pattern should consolidate
- Shorter periods = more aggressive patterns
- Longer periods = more conservative patterns

**Example**: 
- Min: 5 days, Max: 30 days = Quick, tight patterns
- Min: 20 days, Max: 60 days = Longer, more established patterns

### Contraction Thresholds

**Volatility Contraction Threshold**
- How much the Bollinger Bands should contract during consolidation
- Lower values = tighter, more precise patterns
- Higher values = looser, more flexible patterns

**Volume Decline Threshold**
- How much volume should decrease during consolidation
- Lower values = require more volume decline
- Higher values = allow higher volume during consolidation

### Breakout Settings

**Breakout Percentage**
- How far above resistance the price should move for a breakout
- Lower values = more sensitive breakouts
- Higher values = require stronger breakouts

**Volume Multiplier**
- How much volume should increase during breakout
- Higher values = require stronger volume confirmation

## Parameter Relationships

### Conservative vs Aggressive Settings

**Conservative Approach** (Fewer false signals):
- Min Consolidation: 15-20 days
- Volatility Contraction: 30-40%
- Volume Decline: 40-50%
- Breakout Percentage: 3-4%

**Aggressive Approach** (More signals):
- Min Consolidation: 5-10 days
- Volatility Contraction: 60-70%
- Volume Decline: 20-30%
- Breakout Percentage: 1-2%

### Market Conditions

**Bull Market Settings**:
- Lower volatility contraction (40-50%)
- Lower volume decline (20-30%)
- Higher breakout percentage (3-4%)

**Bear Market Settings**:
- Higher volatility contraction (60-70%)
- Higher volume decline (40-50%)
- Lower breakout percentage (1-2%)

## Best Practices

### 1. Start with Defaults
Begin with the default parameters and make small adjustments.

### 2. Test Different Scenarios
- Try conservative settings for live trading
- Use aggressive settings for backtesting
- Adjust based on market conditions

### 3. Monitor Chart Changes
Watch how the ideal pattern changes as you adjust parameters:
- Tighter consolidation = more precise entries
- Looser consolidation = more flexible detection

### 4. Export Multiple Configurations
Save different parameter sets for:
- Different market conditions
- Different timeframes
- Different risk tolerances

### 5. Backtest Your Changes
After applying new parameters, run backtests to validate performance.

## File Structure

```
basic_vcp/
├── dashboard/
│   └── vcp_parameter_tuner.py    # Main parameter tuner
├── run_parameter_tuner.py        # Launcher script
├── config_updater.py             # Configuration updater
├── config.py                     # Main configuration file
└── vcp_parameters_*.json         # Exported parameter files
```

## Troubleshooting

### Dashboard Won't Start
- Check if port 8051 is available
- Ensure all dependencies are installed
- Try a different port in the launcher

### Charts Not Updating
- Refresh the browser page
- Check browser console for errors
- Ensure JavaScript is enabled

### Export Not Working
- Check file permissions in the current directory
- Ensure sufficient disk space
- Try a different filename

### Config Update Fails
- Check if config.py is writable
- Verify the exported JSON file is valid
- Check the backup file for comparison

## Advanced Usage

### Custom Parameter Ranges
You can modify the parameter ranges in `vcp_parameter_tuner.py`:

```python
# In setup_layout() method, modify slider ranges:
dcc.Slider(
    id='min-consolidation-slider',
    min=1, max=100, step=1, value=Config.VCP_MIN_CONSOLIDATION_DAYS,
    # ... other settings
)
```

### Adding New Parameters
To add new parameters:

1. Add the parameter to the Config class in `config.py`
2. Add a slider in the `setup_layout()` method
3. Update the callback in `setup_callbacks()`
4. Add the parameter to the `param_mapping` in `config_updater.py`

### Integration with Main Dashboard
The parameter tuner runs independently but can be integrated with the main dashboard by:

1. Adding a link to the parameter tuner in the main dashboard
2. Sharing parameter state between dashboards
3. Creating a unified configuration management system

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the console output for error messages
3. Verify all dependencies are installed correctly
4. Ensure the config.py file is properly formatted

## Dependencies

Make sure you have all required packages installed:

```bash
pip install -r requirements.txt
```

Key dependencies:
- dash
- dash-bootstrap-components
- plotly
- pandas
- numpy
- matplotlib
- seaborn 