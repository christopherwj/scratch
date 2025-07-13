"""
VCP Parameter Tuner - Interactive GUI for adjusting VCP detection parameters
Allows real-time parameter adjustment with live chart updates showing ideal VCP patterns
"""

import dash
from dash import dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class VCPParameterTuner:
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "VCP Parameter Tuner"
        self.setup_layout()
        self.setup_callbacks()
    
    def generate_ideal_vcp_data(self, params):
        """
        Generate ideal VCP pattern data based on given parameters
        """
        # Extract parameters
        min_consolidation = params['min_consolidation_days']
        max_consolidation = params['max_consolidation_days']
        volatility_threshold = params['volatility_contraction_threshold']
        volume_threshold = params['volume_decline_threshold']
        breakout_percentage = params['breakout_percentage']
        volume_multiplier = params['breakout_volume_multiplier']
        
        # Total days for the pattern
        total_days = 252
        dates = pd.date_range(start='2023-01-01', periods=total_days, freq='D')
        
        # Initialize arrays
        prices = np.zeros(total_days)
        volumes = np.zeros(total_days)
        
        # Phase 1: Uptrend (first 80 days)
        uptrend_days = 80
        base_price = 100
        uptrend_slope = 0.005
        uptrend_volatility = 0.02
        
        for i in range(uptrend_days):
            trend_component = base_price * (1 + uptrend_slope) ** i
            noise = np.random.normal(0, uptrend_volatility * trend_component)
            prices[i] = trend_component + noise
            volumes[i] = np.random.uniform(1000000, 2000000)
        
        # Phase 2: First consolidation (days 80-110)
        consolidation_start = uptrend_days
        consolidation_end = consolidation_start + 30
        resistance_level = prices[consolidation_start - 1] * 1.05
        support_level = prices[consolidation_start - 1] * 0.95
        
        for i in range(consolidation_start, consolidation_end):
            cycle = np.sin((i - consolidation_start) * 2 * np.pi / 15) * 0.02
            prices[i] = (resistance_level + support_level) / 2 + cycle * (resistance_level - support_level) / 2
            volumes[i] = np.random.uniform(800000, 1200000)
        
        # Phase 3: Second uptrend (days 110-140)
        second_uptrend_start = consolidation_end
        second_uptrend_end = second_uptrend_start + 30
        second_base = prices[second_uptrend_start - 1]
        
        for i in range(second_uptrend_start, second_uptrend_end):
            trend_component = second_base * (1 + uptrend_slope) ** (i - second_uptrend_start)
            noise = np.random.normal(0, uptrend_volatility * trend_component)
            prices[i] = trend_component + noise
            volumes[i] = np.random.uniform(1000000, 2000000)
        
        # Phase 4: VCP Consolidation (days 140-170) - Volatility Contraction
        vcp_start = second_uptrend_end
        vcp_end = vcp_start + min_consolidation  # Use min_consolidation_days
        
        # Calculate volatility contraction based on threshold
        contraction_factor = 1 - volatility_threshold
        vcp_resistance = prices[vcp_start - 1] * (1 + 0.03 * contraction_factor)
        vcp_support = prices[vcp_start - 1] * (1 - 0.03 * contraction_factor)
        
        for i in range(vcp_start, vcp_end):
            # Tighter oscillation based on volatility threshold
            cycle = np.sin((i - vcp_start) * 2 * np.pi / 20) * (0.01 * contraction_factor)
            prices[i] = (vcp_resistance + vcp_support) / 2 + cycle * (vcp_resistance - vcp_support) / 2
            
            # Volume decline based on threshold
            volume_decline_factor = 1 - volume_threshold
            volumes[i] = np.random.uniform(600000 * volume_decline_factor, 900000 * volume_decline_factor)
        
        # Phase 5: Breakout (days 170-252)
        breakout_start = vcp_end
        breakout_resistance = vcp_resistance
        
        for i in range(breakout_start, total_days):
            # Strong breakout above resistance
            breakout_strength = breakout_percentage  # Use breakout_percentage
            days_since_breakout = i - breakout_start
            prices[i] = breakout_resistance * (1 + breakout_strength) ** days_since_breakout
            volumes[i] = np.random.uniform(2000000 * volume_multiplier, 4000000 * volume_multiplier)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Volume': volumes
        })
        
        return df
    
    def calculate_technical_indicators(self, df, params):
        """Calculate technical indicators for the chart"""
        # Bollinger Bands
        period = params.get('bollinger_period', 20)
        std_dev = params.get('bollinger_std', 2)
        
        df['SMA'] = df['Close'].rolling(window=period).mean()
        df['BB_Upper'] = df['SMA'] + (df['Close'].rolling(window=period).std() * std_dev)
        df['BB_Lower'] = df['SMA'] - (df['Close'].rolling(window=period).std() * std_dev)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA']
        
        # RSI
        rsi_period = params.get('rsi_period', 14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    def create_vcp_chart(self, params, show_volume_overlay=False):
        """Create the VCP pattern chart with current parameters"""
        df = self.generate_ideal_vcp_data(params)
        df = self.calculate_technical_indicators(df, params)
        
        # Create subplots
        fig = go.Figure()
        
        # Add price candlesticks (simplified as line for demo)
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
        ))
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['BB_Upper'],
            mode='lines',
            name='Bollinger Upper',
            line=dict(color='#ff7f0e', width=1, dash='dash'),
            opacity=0.7
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['BB_Lower'],
            mode='lines',
            name='Bollinger Lower',
            line=dict(color='#ff7f0e', width=1, dash='dash'),
            opacity=0.7,
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['SMA'],
            mode='lines',
            name='20 SMA',
            line=dict(color='#2ca02c', width=1, dash='dash'),
            opacity=0.7
        ))
        
        # Add volume overlay if requested
        if show_volume_overlay:
            # Normalize volume to fit on the price chart
            max_price = float(df['Close'].max())
            min_price = float(df['Close'].min())
            price_range = max_price - min_price
            max_volume = float(df['Volume'].max())
            
            # Scale volume to 20% of price range
            volume_scale = (price_range * 0.2) / max_volume
            scaled_volume = df['Volume'] * volume_scale
            volume_baseline = min_price - (price_range * 0.1)  # Position below price
            
            fig.add_trace(go.Bar(
                x=df['Date'],
                y=scaled_volume,
                name='Volume',
                marker_color='rgba(52, 73, 94, 0.3)',
                yaxis='y',
                base=volume_baseline,
                hovertemplate='<b>Date:</b> %{x}<br><b>Volume:</b> %{text:,.0f}<extra></extra>',
                text=df['Volume']
            ))
        
        # Highlight VCP phases
        phases = [
            (0, 80, 'Phase 1: Uptrend', '#2ecc71'),
            (80, 110, 'Phase 2: First Consolidation', '#f39c12'),
            (110, 140, 'Phase 3: Second Uptrend', '#2ecc71'),
            (140, 140 + params['min_consolidation_days'], 'Phase 4: VCP Consolidation', '#e74c3c'),
            (140 + params['min_consolidation_days'], 252, 'Phase 5: Breakout', '#3498db')
        ]
        
        for start, end, label, color in phases:
            fig.add_vrect(
                x0=df['Date'].iloc[start],
                x1=df['Date'].iloc[end-1],
                fillcolor=color,
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text=label,
                annotation_position="top left"
            )
        
        # Update layout
        fig.update_layout(
            title=f"Ideal VCP Pattern - Parameters: Min Days={params['min_consolidation_days']}, "
                  f"Volatility Contraction={params['volatility_contraction_threshold']:.1%}, "
                  f"Volume Decline={params['volume_decline_threshold']:.1%}",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_volume_chart(self, params):
        """Create volume chart"""
        df = self.generate_ideal_vcp_data(params)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name='Volume',
            marker_color='#34495e',
            opacity=0.7,
            hovertemplate='<b>Date:</b> %{x}<br><b>Volume:</b> %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Volume Pattern - Declining During Consolidation, Spike on Breakout",
            xaxis_title="Date",
            yaxis_title="Volume",
            height=300,
            showlegend=False
        )
        
        return fig
    
    def create_volatility_chart(self, params):
        """Create Bollinger Band width chart showing volatility contraction"""
        df = self.generate_ideal_vcp_data(params)
        df = self.calculate_technical_indicators(df, params)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['BB_Width'],
            mode='lines',
            name='BB Width (Volatility)',
            line=dict(color='red', width=2)
        ))
        
        # Add threshold line
        fig.add_hline(
            y=params['volatility_contraction_threshold'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Target: {params['volatility_contraction_threshold']:.1%}",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title="Bollinger Band Width - Volatility Contraction",
            xaxis_title="Date",
            yaxis_title="BB Width",
            height=300,
            showlegend=False
        )
        
        return fig
    
    def create_metrics_display(self, params):
        """Create metrics display showing current parameter values"""
        metrics_text = f"""
        **Current VCP Parameters:**
        
        **Consolidation Settings:**
        • Min Consolidation Days: {params['min_consolidation_days']}
        • Max Consolidation Days: {params['max_consolidation_days']}
        
        **Contraction Thresholds:**
        • Volatility Contraction: {params['volatility_contraction_threshold']:.1%}
        • Volume Decline: {params['volume_decline_threshold']:.1%}
        
        **Breakout Settings:**
        • Breakout Percentage: {params['breakout_percentage']:.1%}
        • Volume Multiplier: {params['breakout_volume_multiplier']}x
        
        **Swing Point Settings:**
        • Swing Lookback Days: {params['swing_lookback_days']}
        • Swing Threshold: {params['swing_threshold']:.1%}
        
        **Technical Indicators:**
        • Bollinger Period: {params['bollinger_period']}
        • Bollinger Std Dev: {params['bollinger_std']}
        • RSI Period: {params['rsi_period']}
        
        **Risk Management:**
        • Stop Loss: {params['stop_loss_percentage']:.1%}
        • Profit Target: {params['profit_target_multiplier']:.1f}:1 R:R
        """
        
        return dbc.Card([
            dbc.CardBody([
                html.H5("Parameter Summary", className="card-title"),
                dcc.Markdown(metrics_text)
            ])
        ], className="mb-3")
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("VCP Parameter Tuner", className="text-center mb-4"),
                    html.P("Adjust VCP detection parameters and see how the ideal pattern changes in real-time", 
                           className="text-center text-muted")
                ])
            ]),
            
            # Information Panel
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.H5("💡 How to Use This Tool", className="alert-heading"),
                        html.P([
                            "• Hover over parameter labels for detailed explanations",
                            html.Br(),
                            "• Adjust sliders to see real-time chart updates",
                            html.Br(),
                            "• Use 'Reset to Defaults' to return to original settings",
                            html.Br(),
                            "• Export parameters when you find optimal settings",
                            html.Br(),
                            "• The charts show the 'perfect' VCP pattern your model will look for"
                        ])
                    ], color="info", className="mb-3")
                ])
            ]),
            
            dbc.Row([
                # Parameter Controls
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("VCP Detection Parameters"),
                        dbc.CardBody([
                            html.H6("Consolidation Settings"),
                            
                            # Min Consolidation Days with tooltip
                            dbc.Tooltip(
                                "Minimum number of days the stock must consolidate before qualifying as a VCP pattern. Lower values = more aggressive detection, higher values = more conservative. Typical range: 5-20 days.",
                                target="min-consolidation-label",
                                placement="top"
                            ),
                            html.Label("Min Consolidation Days:", id="min-consolidation-label", className="fw-bold"),
                            dcc.Slider(
                                id='min-consolidation-slider',
                                min=5, max=50, step=1, value=Config.VCP_MIN_CONSOLIDATION_DAYS,
                                marks={i: str(i) for i in range(5, 51, 5)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            
                            # Max Consolidation Days with tooltip
                            dbc.Tooltip(
                                "Maximum number of days the consolidation can last. Patterns longer than this are ignored. Higher values allow for longer, more established patterns. Typical range: 30-60 days.",
                                target="max-consolidation-label",
                                placement="top"
                            ),
                            html.Label("Max Consolidation Days:", id="max-consolidation-label", className="fw-bold"),
                            dcc.Slider(
                                id='max-consolidation-slider',
                                min=20, max=100, step=5, value=Config.VCP_MAX_CONSOLIDATION_DAYS,
                                marks={i: str(i) for i in range(20, 101, 20)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            
                            html.Hr(),
                            html.H6("Contraction Thresholds"),
                            
                            # Volatility Contraction Threshold with tooltip
                            dbc.Tooltip(
                                "How much the Bollinger Bands should contract during consolidation (as a percentage). Lower values = tighter, more precise patterns. Higher values = looser, more flexible patterns. 0.3 = 30% contraction required.",
                                target="volatility-threshold-label",
                                placement="top"
                            ),
                            html.Label("Volatility Contraction Threshold:", id="volatility-threshold-label", className="fw-bold"),
                            dcc.Slider(
                                id='volatility-threshold-slider',
                                min=0.1, max=0.9, step=0.05, value=Config.VOLATILITY_CONTRACTION_THRESHOLD,
                                marks={i/10: f"{i/10:.1f}" for i in range(1, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            
                            # Volume Decline Threshold with tooltip
                            dbc.Tooltip(
                                "How much volume should decrease during consolidation (as a percentage). Lower values = require more volume decline. Higher values = allow higher volume during consolidation. 0.3 = 30% volume decline required.",
                                target="volume-threshold-label",
                                placement="top"
                            ),
                            html.Label("Volume Decline Threshold:", id="volume-threshold-label", className="fw-bold"),
                            dcc.Slider(
                                id='volume-threshold-slider',
                                min=0.1, max=0.9, step=0.05, value=Config.VOLUME_DECLINE_THRESHOLD,
                                marks={i/10: f"{i/10:.1f}" for i in range(1, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            
                            html.Hr(),
                            html.H6("Breakout Settings"),
                            
                            # Breakout Percentage with tooltip
                            dbc.Tooltip(
                                "How far above resistance the price should move to confirm a breakout (as a percentage). Lower values = more sensitive breakouts, higher values = require stronger breakouts. 0.02 = 2% above resistance required.",
                                target="breakout-percentage-label",
                                placement="top"
                            ),
                            html.Label("Breakout Percentage:", id="breakout-percentage-label", className="fw-bold"),
                            dcc.Slider(
                                id='breakout-percentage-slider',
                                min=0.01, max=0.05, step=0.005, value=Config.BREAKOUT_PERCENTAGE,
                                marks={i/100: f"{i/100:.1%}" for i in range(1, 6)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            
                            # Volume Multiplier with tooltip
                            dbc.Tooltip(
                                "How much volume should increase during breakout compared to average volume. Higher values = require stronger volume confirmation. 1.5x = 50% more volume than average required.",
                                target="volume-multiplier-label",
                                placement="top"
                            ),
                            html.Label("Volume Multiplier:", id="volume-multiplier-label", className="fw-bold"),
                            dcc.Slider(
                                id='volume-multiplier-slider',
                                min=1.0, max=3.0, step=0.1, value=Config.BREAKOUT_VOLUME_MULTIPLIER,
                                marks={i/10: f"{i/10:.1f}" for i in range(10, 31, 5)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            
                            html.Hr(),
                            html.H6("Technical Indicators"),
                            
                            # Bollinger Period with tooltip
                            dbc.Tooltip(
                                "Number of periods used to calculate Bollinger Bands. Higher values = smoother bands, lower values = more responsive to recent price changes. Standard is 20 periods.",
                                target="bollinger-period-label",
                                placement="top"
                            ),
                            html.Label("Bollinger Period:", id="bollinger-period-label", className="fw-bold"),
                            dcc.Slider(
                                id='bollinger-period-slider',
                                min=10, max=50, step=5, value=Config.BOLLINGER_BANDS_PERIOD,
                                marks={i: str(i) for i in range(10, 51, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            
                            # Bollinger Std Dev with tooltip
                            dbc.Tooltip(
                                "Number of standard deviations for Bollinger Bands. Higher values = wider bands, lower values = tighter bands. Standard is 2.0. Affects volatility contraction measurement.",
                                target="bollinger-std-label",
                                placement="top"
                            ),
                            html.Label("Bollinger Std Dev:", id="bollinger-std-label", className="fw-bold"),
                            dcc.Slider(
                                id='bollinger-std-slider',
                                min=1.0, max=3.0, step=0.1, value=Config.BOLLINGER_BANDS_STD,
                                marks={i/10: f"{i/10:.1f}" for i in range(10, 31, 5)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            
                            # RSI Period with tooltip
                            dbc.Tooltip(
                                "Number of periods used to calculate RSI (Relative Strength Index). Higher values = smoother RSI, lower values = more responsive. Standard is 14 periods. Used for momentum confirmation.",
                                target="rsi-period-label",
                                placement="top"
                            ),
                            html.Label("RSI Period:", id="rsi-period-label", className="fw-bold"),
                            dcc.Slider(
                                id='rsi-period-slider',
                                min=7, max=21, step=1, value=Config.RSI_PERIOD,
                                marks={i: str(i) for i in range(7, 22, 7)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            
                            # RSI Overbought Level with tooltip
                            dbc.Tooltip(
                                "RSI level considered overbought (sell signal). Lower values = more sensitive to overbought conditions, higher values = require stronger overbought signals. Standard is 70. Used for momentum confirmation.",
                                target="rsi-overbought-label",
                                placement="top"
                            ),
                            html.Label("RSI Overbought Level:", id="rsi-overbought-label", className="fw-bold"),
                            dcc.Slider(
                                id='rsi-overbought-slider',
                                min=60, max=85, step=5, value=Config.RSI_OVERBOUGHT,
                                marks={i: str(i) for i in range(60, 86, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            
                            # RSI Oversold Level with tooltip
                            dbc.Tooltip(
                                "RSI level considered oversold (buy signal). Higher values = more sensitive to oversold conditions, lower values = require stronger oversold signals. Standard is 30. Used for momentum confirmation.",
                                target="rsi-oversold-label",
                                placement="top"
                            ),
                            html.Label("RSI Oversold Level:", id="rsi-oversold-label", className="fw-bold"),
                            dcc.Slider(
                                id='rsi-oversold-slider',
                                min=15, max=40, step=5, value=Config.RSI_OVERSOLD,
                                marks={i: str(i) for i in range(15, 41, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            
                            html.Hr(),
                            html.H6("Swing Point Settings"),
                            
                            # Swing Point Lookback with tooltip
                            dbc.Tooltip(
                                "Number of days to look back when identifying swing high and low points. Higher values = more significant swing points, lower values = more sensitive detection. Standard is 10 days.",
                                target="swing-lookback-label",
                                placement="top"
                            ),
                            html.Label("Swing Point Lookback:", id="swing-lookback-label", className="fw-bold"),
                            dcc.Slider(
                                id='swing-lookback-slider',
                                min=5, max=20, step=1, value=Config.SWING_POINT_LOOKBACK,
                                marks={i: str(i) for i in range(5, 21, 5)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            
                            # Swing Point Threshold with tooltip
                            dbc.Tooltip(
                                "Minimum percentage change required to qualify as a swing point. Higher values = more significant swings, lower values = more sensitive detection. 0.02 = 2% minimum swing.",
                                target="swing-threshold-label",
                                placement="top"
                            ),
                            html.Label("Swing Point Threshold:", id="swing-threshold-label", className="fw-bold"),
                            dcc.Slider(
                                id='swing-threshold-slider',
                                min=0.01, max=0.05, step=0.005, value=Config.SWING_POINT_THRESHOLD,
                                marks={i/100: f"{i/100:.1%}" for i in range(1, 6)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            
                            html.Hr(),
                            html.H6("Risk Management"),
                            
                            # Stop Loss Percentage with tooltip
                            dbc.Tooltip(
                                "Stop loss percentage below entry price. Lower values = tighter stops, higher values = more room for price movement. 0.05 = 5% stop loss. Used for position sizing and risk calculation.",
                                target="stop-loss-label",
                                placement="top"
                            ),
                            html.Label("Stop Loss Percentage:", id="stop-loss-label", className="fw-bold"),
                            dcc.Slider(
                                id='stop-loss-slider',
                                min=0.02, max=0.10, step=0.01, value=Config.STOP_LOSS_PERCENTAGE,
                                marks={i/100: f"{i/100:.1%}" for i in range(2, 11, 2)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            
                            # Profit Target Multiplier with tooltip
                            dbc.Tooltip(
                                "Risk-to-reward ratio for profit targets. Higher values = higher profit targets relative to stop loss. 2.0 = 2:1 risk-reward ratio. Used for position sizing and trade management.",
                                target="profit-target-label",
                                placement="top"
                            ),
                            html.Label("Profit Target Multiplier:", id="profit-target-label", className="fw-bold"),
                            dcc.Slider(
                                id='profit-target-slider',
                                min=1.0, max=4.0, step=0.1, value=Config.PROFIT_TARGET_MULTIPLIER,
                                marks={i/10: f"{i/10:.1f}" for i in range(10, 41, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            
                            html.Hr(),
                            
                            # Reset button with tooltip
                            dbc.Tooltip(
                                "Reset all parameters to their default values from config.py. This will update the charts to show the default ideal VCP pattern.",
                                target="reset-btn",
                                placement="top"
                            ),
                            dbc.Button("Reset to Defaults", id="reset-btn", color="secondary", className="me-2"),
                            
                            # Export button with tooltip
                            dbc.Tooltip(
                                "Export current parameters to a JSON file. The file will be saved with a timestamp and can be imported later using the config updater.",
                                target="export-btn",
                                placement="top"
                            ),
                            dbc.Button("Export Parameters", id="export-btn", color="success")
                        ])
                    ], className="mb-3"),
                    
                    # Chart Options
                    dbc.Card([
                        dbc.CardHeader("Chart Options"),
                        dbc.CardBody([
                            dbc.Checklist(
                                id='volume-overlay-checkbox',
                                options=[{"label": "Show Volume Overlay", "value": "show_volume"}],
                                value=[],
                                inline=True,
                                className="mb-2"
                            ),
                            dbc.Tooltip(
                                "Toggle volume bars overlay on the main VCP pattern chart. This helps visualize the relationship between price action and volume during consolidation and breakout phases.",
                                target="volume-overlay-checkbox",
                                placement="top"
                            )
                        ])
                    ], className="mb-3")
                ], width=3),
                
                # Charts
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='vcp-chart', style={'height': '600px'})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='volume-chart', style={'height': '300px'})
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='volatility-chart', style={'height': '300px'})
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='metrics-display')
                        ])
                    ])
                ], width=9)
            ]),
            
            # Hidden div to store current parameters
            dcc.Store(id='current-params-store')
            
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup the dashboard callbacks"""
        
        @self.app.callback(
            Output('current-params-store', 'data'),
            [Input('min-consolidation-slider', 'value'),
             Input('max-consolidation-slider', 'value'),
             Input('volatility-threshold-slider', 'value'),
             Input('volume-threshold-slider', 'value'),
             Input('breakout-percentage-slider', 'value'),
             Input('volume-multiplier-slider', 'value'),
             Input('bollinger-period-slider', 'value'),
             Input('bollinger-std-slider', 'value'),
             Input('rsi-period-slider', 'value'),
             Input('rsi-overbought-slider', 'value'),
             Input('rsi-oversold-slider', 'value'),
             Input('swing-lookback-slider', 'value'),
             Input('swing-threshold-slider', 'value'),
             Input('stop-loss-slider', 'value'),
             Input('profit-target-slider', 'value')]
        )
        def update_params_store(*values):
            return {
                'min_consolidation_days': values[0],
                'max_consolidation_days': values[1],
                'volatility_contraction_threshold': values[2],
                'volume_decline_threshold': values[3],
                'breakout_percentage': values[4],
                'breakout_volume_multiplier': values[5],
                'bollinger_period': values[6],
                'bollinger_std': values[7],
                'rsi_period': values[8],
                'rsi_overbought': values[9],
                'rsi_oversold': values[10],
                'swing_lookback_days': values[11],
                'swing_threshold': values[12],
                'stop_loss_percentage': values[13],
                'profit_target_multiplier': values[14]
            }
        
        @self.app.callback(
            [Output('vcp-chart', 'figure'),
             Output('volume-chart', 'figure'),
             Output('volatility-chart', 'figure'),
             Output('metrics-display', 'children')],
            [Input('current-params-store', 'data'),
             Input('volume-overlay-checkbox', 'value')]
        )
        def update_charts(params, volume_overlay):
            if not params:
                # Return default charts
                params = {
                    'min_consolidation_days': Config.VCP_MIN_CONSOLIDATION_DAYS,
                    'max_consolidation_days': Config.VCP_MAX_CONSOLIDATION_DAYS,
                    'volatility_contraction_threshold': Config.VOLATILITY_CONTRACTION_THRESHOLD,
                    'volume_decline_threshold': Config.VOLUME_DECLINE_THRESHOLD,
                    'breakout_percentage': Config.BREAKOUT_PERCENTAGE,
                    'breakout_volume_multiplier': Config.BREAKOUT_VOLUME_MULTIPLIER,
                    'bollinger_period': Config.BOLLINGER_BANDS_PERIOD,
                    'bollinger_std': Config.BOLLINGER_BANDS_STD,
                    'rsi_period': Config.RSI_PERIOD,
                    'rsi_overbought': Config.RSI_OVERBOUGHT,
                    'rsi_oversold': Config.RSI_OVERSOLD,
                    'swing_lookback_days': Config.SWING_POINT_LOOKBACK,
                    'swing_threshold': Config.SWING_POINT_THRESHOLD,
                    'stop_loss_percentage': Config.STOP_LOSS_PERCENTAGE,
                    'profit_target_multiplier': Config.PROFIT_TARGET_MULTIPLIER
                }
            
            # Check if volume overlay is enabled
            show_volume_overlay = 'show_volume' in (volume_overlay or [])
            
            vcp_fig = self.create_vcp_chart(params, show_volume_overlay)
            volume_fig = self.create_volume_chart(params)
            volatility_fig = self.create_volatility_chart(params)
            metrics_display = self.create_metrics_display(params)
            
            return vcp_fig, volume_fig, volatility_fig, metrics_display
        
        @self.app.callback(
            [Output('min-consolidation-slider', 'value'),
             Output('max-consolidation-slider', 'value'),
             Output('volatility-threshold-slider', 'value'),
             Output('volume-threshold-slider', 'value'),
             Output('breakout-percentage-slider', 'value'),
             Output('volume-multiplier-slider', 'value'),
             Output('bollinger-period-slider', 'value'),
             Output('bollinger-std-slider', 'value'),
             Output('rsi-period-slider', 'value'),
             Output('rsi-overbought-slider', 'value'),
             Output('rsi-oversold-slider', 'value'),
             Output('swing-lookback-slider', 'value'),
             Output('swing-threshold-slider', 'value'),
             Output('stop-loss-slider', 'value'),
             Output('profit-target-slider', 'value')],
            [Input('reset-btn', 'n_clicks')],
            prevent_initial_call=True
        )
        def reset_to_defaults(n_clicks):
            return (
                Config.VCP_MIN_CONSOLIDATION_DAYS,
                Config.VCP_MAX_CONSOLIDATION_DAYS,
                Config.VOLATILITY_CONTRACTION_THRESHOLD,
                Config.VOLUME_DECLINE_THRESHOLD,
                Config.BREAKOUT_PERCENTAGE,
                Config.BREAKOUT_VOLUME_MULTIPLIER,
                Config.BOLLINGER_BANDS_PERIOD,
                Config.BOLLINGER_BANDS_STD,
                Config.RSI_PERIOD,
                Config.RSI_OVERBOUGHT,
                Config.RSI_OVERSOLD,
                Config.SWING_POINT_LOOKBACK,
                Config.SWING_POINT_THRESHOLD,
                Config.STOP_LOSS_PERCENTAGE,
                Config.PROFIT_TARGET_MULTIPLIER
            )
        
        @self.app.callback(
            Output('export-btn', 'children'),
            [Input('export-btn', 'n_clicks'),
             Input('current-params-store', 'data')],
            prevent_initial_call=True
        )
        def export_parameters(n_clicks, params):
            if n_clicks and params:
                # Create a timestamp for the filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"vcp_parameters_{timestamp}.json"
                
                # Save parameters to file
                with open(filename, 'w') as f:
                    json.dump(params, f, indent=2)
                
                return f"Exported! ({filename})"
            
            return "Export Parameters"
    
    def run(self, debug=True, port=8051):
        """Run the parameter tuner dashboard"""
        self.app.run(debug=debug, port=port)

if __name__ == "__main__":
    tuner = VCPParameterTuner()
    tuner.run() 