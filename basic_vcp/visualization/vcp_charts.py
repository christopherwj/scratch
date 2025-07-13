"""
VCP Pattern Visualization
Creates charts showing VCP patterns, swing points, and breakout signals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from analysis.vcp_detector import VCPPattern, SwingPoint, BreakoutSignal

class VCPVisualizer:
    def __init__(self):
        """Initialize the VCP visualizer"""
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_vcp_pattern(self, df: pd.DataFrame, vcp_pattern: VCPPattern, 
                        signals: List[BreakoutSignal] = None, 
                        save_path: str = None) -> go.Figure:
        """
        Create an interactive plot showing a VCP pattern
        
        Args:
            df: DataFrame with price data
            vcp_pattern: VCP pattern to visualize
            signals: List of breakout signals
            save_path: Path to save the plot (optional)
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & VCP Pattern', 'Volume', 'Bollinger Bands Width'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Get pattern date range
        pattern_start = vcp_pattern.start_date
        pattern_end = vcp_pattern.end_date
        
        # Filter data for pattern period and some context
        start_idx = max(0, df.index.get_loc(pattern_start) - 50)
        end_idx = min(len(df), df.index.get_loc(pattern_end) + 50)
        plot_data = df.iloc[start_idx:end_idx]
        
        # Plot candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=plot_data.index,
                open=plot_data['open'],
                high=plot_data['high'],
                low=plot_data['low'],
                close=plot_data['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data['sma_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data['sma_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
        
        # Add Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data['bb_upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data['bb_lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # Highlight VCP pattern area
        pattern_data = df.loc[pattern_start:pattern_end]
        fig.add_trace(
            go.Scatter(
                x=pattern_data.index,
                y=pattern_data['high'],
                mode='lines',
                name='VCP Pattern',
                line=dict(color='purple', width=3),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add support and resistance levels
        fig.add_hline(
            y=vcp_pattern.resistance_level,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Resistance: ${vcp_pattern.resistance_level:.2f}",
            row=1, col=1
        )
        
        fig.add_hline(
            y=vcp_pattern.support_level,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Support: ${vcp_pattern.support_level:.2f}",
            row=1, col=1
        )
        
        # Add breakout level
        fig.add_hline(
            y=vcp_pattern.breakout_level,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"Breakout: ${vcp_pattern.breakout_level:.2f}",
            row=1, col=1
        )
        
        # Add swing points
        for swing in vcp_pattern.swing_points:
            color = 'red' if swing.point_type == 'high' else 'green'
            fig.add_trace(
                go.Scatter(
                    x=[swing.date],
                    y=[swing.price],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=10,
                        color=color,
                        line=dict(width=2, color='black')
                    ),
                    name=f'Swing {swing.point_type}',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Add breakout signals
        if signals:
            for signal in signals:
                if signal.signal_type == 'breakout':
                    color = 'green'
                    symbol = 'triangle-up'
                else:
                    color = 'red'
                    symbol = 'triangle-down'
                
                fig.add_trace(
                    go.Scatter(
                        x=[signal.date],
                        y=[signal.price],
                        mode='markers',
                        marker=dict(
                            symbol=symbol,
                            size=15,
                            color=color,
                            line=dict(width=2, color='black')
                        ),
                        name=f'{signal.signal_type.title()} Signal',
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # Plot volume
        if 'volume' in plot_data.columns:
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(plot_data['close'], plot_data['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=plot_data.index,
                    y=plot_data['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Add volume moving average
            if 'volume_sma' in plot_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=plot_data['volume_sma'],
                        mode='lines',
                        name='Volume SMA',
                        line=dict(color='black', width=1)
                    ),
                    row=2, col=1
                )
        
        # Plot Bollinger Band width
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data['bb_width'],
                mode='lines',
                name='BB Width',
                line=dict(color='purple', width=2)
            ),
            row=3, col=1
        )
        
        # Highlight VCP period in BB width
        pattern_bb = plot_data.loc[pattern_start:pattern_end, 'bb_width']
        fig.add_trace(
            go.Scatter(
                x=pattern_bb.index,
                y=pattern_bb,
                mode='lines',
                name='VCP BB Width',
                line=dict(color='red', width=3),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'VCP Pattern Analysis - {vcp_pattern.ticker}',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="BB Width", row=3, col=1)
        
        # Add pattern information
        fig.add_annotation(
            x=0.02, y=0.98,
            xref='paper', yref='paper',
            text=f"Pattern Strength: {vcp_pattern.pattern_strength:.3f}<br>"
                 f"Consolidation Days: {vcp_pattern.consolidation_days}<br>"
                 f"Volatility Contraction: {vcp_pattern.volatility_contraction:.3f}<br>"
                 f"Volume Decline: {vcp_pattern.volume_decline:.3f}",
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_multiple_patterns(self, patterns: List[VCPPattern], 
                              save_path: str = None) -> go.Figure:
        """
        Create a summary plot showing multiple VCP patterns
        
        Args:
            patterns: List of VCP patterns
            save_path: Path to save the plot (optional)
            
        Returns:
            Plotly figure object
        """
        # Create summary data
        tickers = [p.ticker for p in patterns]
        strengths = [p.pattern_strength for p in patterns]
        days = [p.consolidation_days for p in patterns]
        contractions = [p.volatility_contraction for p in patterns]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Pattern Strength by Ticker', 'Consolidation Days',
                          'Volatility Contraction', 'Pattern Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # Pattern strength bar chart
        fig.add_trace(
            go.Bar(
                x=tickers,
                y=strengths,
                name='Pattern Strength',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Consolidation days bar chart
        fig.add_trace(
            go.Bar(
                x=tickers,
                y=days,
                name='Consolidation Days',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        # Volatility contraction scatter
        fig.add_trace(
            go.Scatter(
                x=strengths,
                y=contractions,
                mode='markers',
                text=tickers,
                name='Volatility vs Strength',
                marker=dict(
                    size=10,
                    color=strengths,
                    colorscale='Viridis',
                    showscale=True
                )
            ),
            row=2, col=1
        )
        
        # Pattern distribution pie chart
        strength_ranges = {
            'Strong (0.8-1.0)': len([s for s in strengths if s >= 0.8]),
            'Good (0.6-0.8)': len([s for s in strengths if 0.6 <= s < 0.8]),
            'Fair (0.4-0.6)': len([s for s in strengths if 0.4 <= s < 0.6]),
            'Weak (0.0-0.4)': len([s for s in strengths if s < 0.4])
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(strength_ranges.keys()),
                values=list(strength_ranges.values()),
                name='Pattern Strength Distribution'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='VCP Patterns Summary',
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_breakout_signals(self, signals: List[BreakoutSignal], 
                             save_path: str = None) -> go.Figure:
        """
        Create a plot showing breakout signals
        
        Args:
            signals: List of breakout signals
            save_path: Path to save the plot (optional)
            
        Returns:
            Plotly figure object
        """
        # Create summary data
        tickers = [s.ticker for s in signals]
        confidences = [s.confidence for s in signals]
        risk_rewards = [s.risk_reward_ratio for s in signals]
        signal_types = [s.signal_type for s in signals]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Signal Confidence by Ticker', 'Risk/Reward Ratios',
                          'Signal Types Distribution', 'Confidence vs Risk/Reward'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # Signal confidence bar chart
        fig.add_trace(
            go.Bar(
                x=tickers,
                y=confidences,
                name='Signal Confidence',
                marker_color='lightcoral'
            ),
            row=1, col=1
        )
        
        # Risk/reward ratios bar chart
        fig.add_trace(
            go.Bar(
                x=tickers,
                y=risk_rewards,
                name='Risk/Reward Ratio',
                marker_color='lightyellow'
            ),
            row=1, col=2
        )
        
        # Confidence vs Risk/Reward scatter
        fig.add_trace(
            go.Scatter(
                x=confidences,
                y=risk_rewards,
                mode='markers',
                text=tickers,
                name='Confidence vs R/R',
                marker=dict(
                    size=10,
                    color=confidences,
                    colorscale='Plasma',
                    showscale=True
                )
            ),
            row=2, col=1
        )
        
        # Signal types distribution
        type_counts = {}
        for signal_type in signal_types:
            type_counts[signal_type] = type_counts.get(signal_type, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(type_counts.keys()),
                values=list(type_counts.values()),
                name='Signal Types'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Breakout Signals Summary',
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_dashboard_data(self, scanner_results: dict) -> dict:
        """
        Create data for dashboard visualization
        
        Args:
            scanner_results: Results from VCP scanner
            
        Returns:
            Dictionary with dashboard data
        """
        dashboard_data = {
            'summary_stats': {
                'total_stocks': scanner_results['tickers_scanned'],
                'patterns_found': scanner_results['vcp_patterns_found'],
                'signals_found': scanner_results['breakout_signals'],
                'stocks_with_patterns': scanner_results['summary']['stocks_with_patterns'],
                'stocks_with_signals': scanner_results['summary']['stocks_with_signals']
            },
            'top_patterns': scanner_results['summary']['top_patterns'],
            'top_signals': scanner_results['summary']['top_signals'],
            'pattern_distribution': scanner_results['summary']['pattern_strength_distribution'],
            'signal_distribution': scanner_results['summary']['signal_confidence_distribution']
        }
        
        return dashboard_data 