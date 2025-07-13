"""
VCP Detection System Dashboard
Web-based interface for viewing VCP patterns and signals
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import glob

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import VCPScanner
from visualization.vcp_charts import VCPVisualizer
from data.stock_data import SQLiteStockDataLoader

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "VCP Detection System Dashboard"

# Initialize components
scanner = VCPScanner()
scanner.data_loader = SQLiteStockDataLoader()
visualizer = VCPVisualizer()

def load_latest_results():
    """Load the most recent scan results"""
    result_files = glob.glob("vcp_scan_results_*.json")
    if not result_files:
        return None
    
    # Get the most recent file
    latest_file = max(result_files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def create_summary_cards(results):
    """Create summary cards for the dashboard"""
    if not results:
        return []
    
    summary = results.get('summary', {})
    
    cards = [
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{results.get('tickers_scanned', 0)}", className="card-title"),
                html.P("Stocks Scanned", className="card-text")
            ])
        ], className="text-center mb-3"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{results.get('vcp_patterns_found', 0)}", className="card-title text-primary"),
                html.P("VCP Patterns Found", className="card-text")
            ])
        ], className="text-center mb-3"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{results.get('breakout_signals', 0)}", className="card-title text-success"),
                html.P("Breakout Signals", className="card-text")
            ])
        ], className="text-center mb-3"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{summary.get('stocks_with_patterns', 0)}", className="card-title text-warning"),
                html.P("Stocks with Patterns", className="card-text")
            ])
        ], className="text-center mb-3")
    ]
    
    return cards

def create_pattern_table(results):
    """Create a table of VCP patterns"""
    if not results or not results.get('summary', {}).get('top_patterns'):
        return html.P("No VCP patterns found in the latest scan.")
    
    patterns = results['summary']['top_patterns']
    
    table_header = [
        html.Thead(html.Tr([
            html.Th("Rank"),
            html.Th("Ticker"),
            html.Th("Strength"),
            html.Th("Days"),
            html.Th("Start Date"),
            html.Th("End Date")
        ]))
    ]
    
    table_rows = []
    for i, pattern in enumerate(patterns[:10], 1):
        table_rows.append(html.Tr([
            html.Td(i),
            html.Td(pattern['ticker']),
            html.Td(f"{pattern['strength']:.3f}"),
            html.Td(pattern['consolidation_days']),
            html.Td(pattern['start_date']),
            html.Td(pattern['end_date'])
        ]))
    
    table_body = [html.Tbody(table_rows)]
    
    return dbc.Table(table_header + table_body, bordered=True, hover=True)

def create_signal_table(results):
    """Create a table of breakout signals"""
    if not results or not results.get('summary', {}).get('top_signals'):
        return html.P("No breakout signals found in the latest scan.")
    
    signals = results['summary']['top_signals']
    
    table_header = [
        html.Thead(html.Tr([
            html.Th("Rank"),
            html.Th("Ticker"),
            html.Th("Type"),
            html.Th("Confidence"),
            html.Th("Date"),
            html.Th("Price"),
            html.Th("Risk/Reward")
        ]))
    ]
    
    table_rows = []
    for i, signal in enumerate(signals[:10], 1):
        table_rows.append(html.Tr([
            html.Td(i),
            html.Td(signal['ticker']),
            html.Td(signal['signal_type'].upper()),
            html.Td(f"{signal['confidence']:.3f}"),
            html.Td(signal['date']),
            html.Td(f"${signal['price']:.2f}"),
            html.Td(f"{signal['risk_reward_ratio']:.2f}")
        ]))
    
    table_body = [html.Tbody(table_rows)]
    
    return dbc.Table(table_header + table_body, bordered=True, hover=True)

def format_volume(volume):
    """Safely format volume with comma separators"""
    try:
        if volume and volume != 'N/A' and volume != '':
            return f"{int(float(volume)):,}"
        else:
            return "N/A"
    except (ValueError, TypeError):
        return "N/A"

def create_detailed_signals_table(results):
    """Create a detailed table of all signals with confidence > 0.5"""
    if not results or not results.get('signals_by_ticker'):
        return html.P("No detailed signal data found in the latest scan.")
    
    # Collect all signals with confidence > 0.5
    high_confidence_signals = []
    
    for ticker, signals in results['signals_by_ticker'].items():
        for signal in signals:
            if signal.get('confidence', 0) > 0.5:
                high_confidence_signals.append({
                    'ticker': ticker,
                    'signal_type': signal.get('signal_type', ''),
                    'confidence': signal.get('confidence', 0),
                    'date': signal.get('date', ''),
                    'price': signal.get('price', 0),
                    'stop_loss': signal.get('stop_loss', 0),
                    'target': signal.get('profit_target', 0),
                    'risk_reward': signal.get('risk_reward_ratio', 0),
                    'volume': signal.get('volume', 0)
                })
    
    if not high_confidence_signals:
        return html.P("No signals found with confidence > 0.5.")
    
    # Sort by confidence (highest first)
    high_confidence_signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    table_header = [
        html.Thead(html.Tr([
            html.Th("Ticker"),
            html.Th("Signal Type"),
            html.Th("Confidence"),
            html.Th("Entry Date"),
            html.Th("Entry Price"),
            html.Th("Stop Loss"),
            html.Th("Target"),
            html.Th("R/R Ratio"),
            html.Th("Volume"),
            html.Th("Potential Profit %")
        ]))
    ]
    
    table_rows = []
    for signal in high_confidence_signals:
        # Calculate potential profit percentage
        if signal['signal_type'] == 'breakout':
            profit_pct = ((signal['target'] - signal['price']) / signal['price']) * 100
        else:
            profit_pct = ((signal['price'] - signal['target']) / signal['price']) * 100
        
        # Color code the row based on signal type
        row_color = "table-success" if signal['signal_type'] == 'breakout' else "table-danger"
        
        table_rows.append(html.Tr([
            html.Td(signal['ticker'], className="fw-bold"),
            html.Td(signal['signal_type'].upper(), 
                   className="text-success" if signal['signal_type'] == 'breakout' else "text-danger"),
            html.Td(f"{signal['confidence']:.3f}", 
                   className="fw-bold" if signal['confidence'] > 0.8 else ""),
            html.Td(signal['date'][:10]),  # Just show date part
            html.Td(f"${signal['price']:.2f}", className="fw-bold"),
            html.Td(f"${signal['stop_loss']:.2f}", className="text-danger"),
            html.Td(f"${signal['target']:.2f}", className="text-success"),
            html.Td(f"{signal['risk_reward']:.2f}", 
                   className="text-success" if signal['risk_reward'] > 2 else "text-warning"),
            html.Td(format_volume(signal['volume'])),
            html.Td(f"{profit_pct:.1f}%", 
                   className="text-success" if profit_pct > 20 else "text-warning")
        ], className=row_color))
    
    table_body = [html.Tbody(table_rows)]
    
    return dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True)

def create_signals_summary_stats(results):
    """Create summary statistics for high confidence signals"""
    if not results or not results.get('signals_by_ticker'):
        return []
    
    # Collect all signals with confidence > 0.5
    high_confidence_signals = []
    
    for ticker, signals in results['signals_by_ticker'].items():
        for signal in signals:
            if signal.get('confidence', 0) > 0.5:
                high_confidence_signals.append(signal)
    
    if not high_confidence_signals:
        return []
    
    # Calculate statistics
    total_signals = len(high_confidence_signals)
    breakout_signals = len([s for s in high_confidence_signals if s.get('signal_type') == 'breakout'])
    breakdown_signals = len([s for s in high_confidence_signals if s.get('signal_type') == 'breakdown'])
    avg_confidence = sum(s.get('confidence', 0) for s in high_confidence_signals) / total_signals
    avg_risk_reward = sum(s.get('risk_reward_ratio', 0) for s in high_confidence_signals) / total_signals
    
    # Calculate average potential profit
    total_profit_pct = 0
    for signal in high_confidence_signals:
        if signal['signal_type'] == 'breakout':
            profit_pct = ((signal['profit_target'] - signal['price']) / signal['price']) * 100
        else:
            profit_pct = ((signal['price'] - signal['profit_target']) / signal['price']) * 100
        total_profit_pct += profit_pct
    
    avg_profit_pct = total_profit_pct / total_signals
    
    cards = [
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{total_signals}", className="card-title"),
                html.P("High Confidence Signals", className="card-text")
            ])
        ], className="text-center mb-3"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{breakout_signals}", className="card-title text-success"),
                html.P("Breakout Signals", className="card-text")
            ])
        ], className="text-center mb-3"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{breakdown_signals}", className="card-title text-danger"),
                html.P("Breakdown Signals", className="card-text")
            ])
        ], className="text-center mb-3"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{avg_confidence:.3f}", className="card-title text-primary"),
                html.P("Avg Confidence", className="card-text")
            ])
        ], className="text-center mb-3"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{avg_risk_reward:.2f}", className="card-title text-warning"),
                html.P("Avg Risk/Reward", className="card-text")
            ])
        ], className="text-center mb-3"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{avg_profit_pct:.1f}%", className="card-title text-info"),
                html.P("Avg Potential Profit", className="card-text")
            ])
        ], className="text-center mb-3")
    ]
    
    return cards

def create_confidence_distribution_chart(results):
    """Create a histogram showing confidence distribution"""
    if not results or not results.get('signals_by_ticker'):
        return go.Figure().add_annotation(
            text="No signal data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Collect all signals with confidence > 0.5
    high_confidence_signals = []
    
    for ticker, signals in results['signals_by_ticker'].items():
        for signal in signals:
            if signal.get('confidence', 0) > 0.5:
                high_confidence_signals.append(signal)
    
    if not high_confidence_signals:
        return go.Figure().add_annotation(
            text="No high confidence signals found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    confidences = [s['confidence'] for s in high_confidence_signals]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=confidences,
        nbinsx=20,
        name='Signal Confidence',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Distribution of Signal Confidence (>0.5)",
        xaxis_title="Confidence Score",
        yaxis_title="Number of Signals",
        height=300,
        showlegend=False
    )
    
    return fig

def create_stock_price_chart_with_signals(results, selected_ticker=None):
    """Create a stock price chart with entry/exit points for a specific ticker"""
    if not results or not results.get('signals_by_ticker'):
        return go.Figure().add_annotation(
            text="No signal data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # If no ticker selected, use the first one with signals
    if not selected_ticker:
        for ticker, signals in results['signals_by_ticker'].items():
            if signals and any(s.get('confidence', 0) > 0.5 for s in signals):
                selected_ticker = ticker
                break
    
    if not selected_ticker:
        return go.Figure().add_annotation(
            text="No high confidence signals found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Get signals for the selected ticker
    signals = results['signals_by_ticker'].get(selected_ticker, [])
    high_confidence_signals = [s for s in signals if s.get('confidence', 0) > 0.5]
    
    if not high_confidence_signals:
        return go.Figure().add_annotation(
            text=f"No high confidence signals for {selected_ticker}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Load stock data
    try:
        df = scanner.data_loader.load_stock_data(selected_ticker)
        if df is None or df.empty:
            return go.Figure().add_annotation(
                text=f"No price data available for {selected_ticker}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    except:
        return go.Figure().add_annotation(
            text=f"Error loading data for {selected_ticker}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Create candlestick chart
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    # Add moving averages if available
    if 'sma_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['sma_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='blue', width=1)
        ))
    
    if 'sma_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['sma_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='orange', width=1)
        ))
    
    # Add entry points (buy signals)
    breakout_signals = [s for s in high_confidence_signals if s['signal_type'] == 'breakout']
    if breakout_signals:
        fig.add_trace(go.Scatter(
            x=[s['date'] for s in breakout_signals],
            y=[s['price'] for s in breakout_signals],
            mode='markers',
            name='Buy Signal',
            marker=dict(
                symbol='triangle-up',
                size=15,
                color='green',
                line=dict(width=2, color='darkgreen')
            ),
            text=[f"Buy: ${s['price']:.2f}<br>Target: ${s['profit_target']:.2f}<br>Stop: ${s['stop_loss']:.2f}<br>Conf: {s['confidence']:.3f}" for s in breakout_signals],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
    
    # Add entry points (sell signals)
    breakdown_signals = [s for s in high_confidence_signals if s['signal_type'] == 'breakdown']
    if breakdown_signals:
        fig.add_trace(go.Scatter(
            x=[s['date'] for s in breakdown_signals],
            y=[s['price'] for s in breakdown_signals],
            mode='markers',
            name='Sell Signal',
            marker=dict(
                symbol='triangle-down',
                size=15,
                color='red',
                line=dict(width=2, color='darkred')
            ),
            text=[f"Sell: ${s['price']:.2f}<br>Target: ${s['profit_target']:.2f}<br>Stop: ${s['stop_loss']:.2f}<br>Conf: {s['confidence']:.3f}" for s in breakdown_signals],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
    
    # Add target and stop loss lines for the most recent signal
    if high_confidence_signals:
        latest_signal = max(high_confidence_signals, key=lambda x: x['date'])
        
        # Add target line
        fig.add_hline(
            y=latest_signal['profit_target'],
            line_dash="dot",
            line_color="green" if latest_signal['signal_type'] == 'breakout' else "red",
            annotation_text=f"Target: ${latest_signal['profit_target']:.2f}",
            annotation_position="top right"
        )
        
        # Add stop loss line
        fig.add_hline(
            y=latest_signal['stop_loss'],
            line_dash="dot",
            line_color="red" if latest_signal['signal_type'] == 'breakout' else "green",
            annotation_text=f"Stop Loss: ${latest_signal['stop_loss']:.2f}",
            annotation_position="bottom right"
        )
    
    fig.update_layout(
        title=f"{selected_ticker} - Price Chart with Entry/Exit Points",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        showlegend=True
    )
    
    return fig

def create_signal_type_pie_chart(results):
    """Create a pie chart showing breakout vs breakdown signals"""
    if not results or not results.get('signals_by_ticker'):
        return go.Figure().add_annotation(
            text="No signal data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Collect all signals with confidence > 0.5
    high_confidence_signals = []
    
    for ticker, signals in results['signals_by_ticker'].items():
        for signal in signals:
            if signal.get('confidence', 0) > 0.5:
                high_confidence_signals.append(signal)
    
    if not high_confidence_signals:
        return go.Figure().add_annotation(
            text="No high confidence signals found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    breakout_count = len([s for s in high_confidence_signals if s['signal_type'] == 'breakout'])
    breakdown_count = len([s for s in high_confidence_signals if s['signal_type'] == 'breakdown'])
    
    fig = go.Figure(data=[go.Pie(
        labels=['Breakout', 'Breakdown'],
        values=[breakout_count, breakdown_count],
        marker_colors=['green', 'red'],
        hole=0.3
    )])
    
    fig.update_layout(
        title="Signal Type Distribution",
        height=300,
        showlegend=True
    )
    
    return fig

def create_profit_potential_chart(results):
    """Create a scatter plot showing confidence vs potential profit"""
    if not results or not results.get('signals_by_ticker'):
        return go.Figure().add_annotation(
            text="No signal data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Collect all signals with confidence > 0.5
    high_confidence_signals = []
    
    for ticker, signals in results['signals_by_ticker'].items():
        for signal in signals:
            if signal.get('confidence', 0) > 0.5:
                # Calculate potential profit percentage
                if signal['signal_type'] == 'breakout':
                    profit_pct = ((signal['profit_target'] - signal['price']) / signal['price']) * 100
                else:
                    profit_pct = ((signal['price'] - signal['profit_target']) / signal['price']) * 100
                
                high_confidence_signals.append({
                    'confidence': signal['confidence'],
                    'profit_pct': profit_pct,
                    'signal_type': signal['signal_type'],
                    'ticker': ticker
                })
    
    if not high_confidence_signals:
        return go.Figure().add_annotation(
            text="No high confidence signals found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Separate breakout and breakdown signals
    breakout_data = [s for s in high_confidence_signals if s['signal_type'] == 'breakout']
    breakdown_data = [s for s in high_confidence_signals if s['signal_type'] == 'breakdown']
    
    fig = go.Figure()
    
    if breakout_data:
        fig.add_trace(go.Scatter(
            x=[s['confidence'] for s in breakout_data],
            y=[s['profit_pct'] for s in breakout_data],
            mode='markers',
            name='Breakout',
            marker=dict(color='green', size=8),
            text=[s['ticker'] for s in breakout_data],
            hovertemplate='<b>%{text}</b><br>Confidence: %{x:.3f}<br>Profit: %{y:.1f}%<extra></extra>'
        ))
    
    if breakdown_data:
        fig.add_trace(go.Scatter(
            x=[s['confidence'] for s in breakdown_data],
            y=[s['profit_pct'] for s in breakdown_data],
            mode='markers',
            name='Breakdown',
            marker=dict(color='red', size=8),
            text=[s['ticker'] for s in breakdown_data],
            hovertemplate='<b>%{text}</b><br>Confidence: %{x:.3f}<br>Profit: %{y:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title="Confidence vs Potential Profit",
        xaxis_title="Signal Confidence",
        yaxis_title="Potential Profit (%)",
        height=400,
        showlegend=True
    )
    
    return fig

def create_pattern_strength_chart(results):
    """Create a chart showing pattern strength distribution"""
    if not results or not results.get('summary', {}).get('pattern_strength_distribution'):
        return go.Figure().add_annotation(
            text="No pattern data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    dist = results['summary']['pattern_strength_distribution']
    
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=dist.get('mean', 0),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Average Pattern Strength"},
        delta={'reference': 0.5},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_signal_confidence_chart(results):
    """Create a chart showing signal confidence distribution"""
    if not results or not results.get('summary', {}).get('signal_confidence_distribution'):
        return go.Figure().add_annotation(
            text="No signal data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    dist = results['summary']['signal_confidence_distribution']
    
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=dist.get('mean', 0),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Average Signal Confidence"},
        delta={'reference': 0.5},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

# Load initial data
initial_results = load_latest_results()

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("VCP Detection System Dashboard", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Control Panel", className="card-title"),
                    dbc.Button("Run New Scan", id="run-scan-btn", color="primary", className="me-2"),
                    dbc.Button("Refresh Data", id="refresh-btn", color="secondary", className="me-2"),
                    html.Div(id="scan-status", className="mt-2")
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Tabs
    dbc.Tabs([
        # Overview Tab
        dbc.Tab([
            # Summary Cards
            dbc.Row([
                dbc.Col(card, width=3) for card in create_summary_cards(initial_results)
            ], className="mb-4"),
            
            # Charts Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Pattern Strength Distribution", className="card-title"),
                            dcc.Graph(id="pattern-strength-chart")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Signal Confidence Distribution", className="card-title"),
                            dcc.Graph(id="signal-confidence-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Tables Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Top VCP Patterns", className="card-title"),
                            html.Div(id="pattern-table")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Top Breakout Signals", className="card-title"),
                            html.Div(id="signal-table")
                        ])
                    ])
                ], width=6)
            ], className="mb-4")
        ], label="Overview", tab_id="overview"),
        
        # High Confidence Signals Tab
        dbc.Tab([
            # High Confidence Signals Summary Cards
            dbc.Row([
                dbc.Col(card, width=2) for card in create_signals_summary_stats(initial_results)
            ], className="mb-4", id="signals-summary-cards"),
            

            
            # Stock Selection and Price Chart
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Stock Price Chart with Entry/Exit Points", className="card-title"),
                            html.P("Select a stock to view its price chart with buy/sell signals", className="text-muted mb-3"),
                            dcc.Dropdown(
                                id="stock-selector",
                                placeholder="Select a stock...",
                                style={"marginBottom": "10px"}
                            ),
                            dcc.Graph(id="stock-price-chart")
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Charts Row 2
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Confidence vs Potential Profit", className="card-title"),
                            dcc.Graph(id="profit-potential-chart")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Signal Confidence Distribution", className="card-title"),
                            dcc.Graph(id="confidence-distribution-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Charts Row 3
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Signal Type Distribution", className="card-title"),
                            dcc.Graph(id="signal-type-pie-chart")
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Detailed Signals Table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("High Confidence Signals (>0.5) - Entry & Exit Points", className="card-title"),
                            html.P("All trading signals with confidence greater than 0.5, showing entry prices, stop losses, and targets.", 
                                   className="text-muted mb-3"),
                            html.Div(id="detailed-signals-table")
                        ])
                    ])
                ])
            ], className="mb-4")
        ], label="High Confidence Signals", tab_id="signals")
    ], id="tabs", active_tab="overview"),
    
    # Hidden div for storing scan results
    html.Div(id="scan-results-store", style={"display": "none"}),
    
    # Interval component for auto-refresh
    dcc.Interval(
        id='interval-component',
        interval=30000,  # 30 seconds
        n_intervals=0
    )
], fluid=True)

# Callbacks
@app.callback(
    [Output("scan-status", "children"),
     Output("scan-results-store", "children")],
    [Input("run-scan-btn", "n_clicks"),
     Input("refresh-btn", "n_clicks")],
    prevent_initial_call=True
)
def run_scan(run_clicks, refresh_clicks):
    """Run a new scan or refresh data"""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return "", ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "run-scan-btn":
        try:
            # Run scan on first 100 stocks
            available_tickers = scanner.data_loader.get_available_tickers()
            results = scanner.scan_stocks(available_tickers[:100])
            
            # Save results
            scanner.save_results()
            
            return f"Scan completed! Found {results['vcp_patterns_found']} patterns and {results['breakout_signals']} signals.", json.dumps(results)
        except Exception as e:
            return f"Error running scan: {str(e)}", ""
    
    elif button_id == "refresh-btn":
        try:
            results = load_latest_results()
            if results:
                return "Data refreshed successfully!", json.dumps(results)
            else:
                return "No scan results found.", ""
        except Exception as e:
            return f"Error refreshing data: {str(e)}", ""
    
    return "", ""

@app.callback(
    [Output("pattern-strength-chart", "figure"),
     Output("signal-confidence-chart", "figure"),
     Output("pattern-table", "children"),
     Output("signal-table", "children")],
    [Input("scan-results-store", "children"),
     Input("interval-component", "n_intervals")],
    prevent_initial_call=True
)
def update_overview_charts_and_tables(results_json, n_intervals):
    """Update overview charts and tables with latest data"""
    if results_json:
        results = json.loads(results_json)
    else:
        results = load_latest_results()
    
    if not results:
        # Return empty charts and tables
        empty_fig = go.Figure().add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return empty_fig, empty_fig, "No data available", "No data available"
    
    pattern_chart = create_pattern_strength_chart(results)
    signal_chart = create_signal_confidence_chart(results)
    pattern_table = create_pattern_table(results)
    signal_table = create_signal_table(results)
    
    return pattern_chart, signal_chart, pattern_table, signal_table

@app.callback(
    [Output("detailed-signals-table", "children"),
     Output("confidence-distribution-chart", "figure"),
     Output("signal-type-pie-chart", "figure"),
     Output("profit-potential-chart", "figure")],
    [Input("scan-results-store", "children"),
     Input("interval-component", "n_intervals")],
    prevent_initial_call=True
)
def update_signals_charts_and_tables(results_json, n_intervals):
    """Update signals charts and tables with latest data"""
    if results_json:
        results = json.loads(results_json)
    else:
        results = load_latest_results()
    
    if not results:
        # Return empty charts and tables
        empty_fig = go.Figure().add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return "No data available", empty_fig, empty_fig, empty_fig
    
    detailed_signals_table = create_detailed_signals_table(results)
    confidence_dist_chart = create_confidence_distribution_chart(results)
    signal_type_chart = create_signal_type_pie_chart(results)
    profit_potential_chart = create_profit_potential_chart(results)
    
    return detailed_signals_table, confidence_dist_chart, signal_type_chart, profit_potential_chart

@app.callback(
    [Output("stock-selector", "options"),
     Output("stock-selector", "value")],
    [Input("scan-results-store", "children"),
     Input("interval-component", "n_intervals")],
    prevent_initial_call=True
)
def update_stock_selector(results_json, n_intervals):
    """Update stock selector dropdown with available stocks"""
    if results_json:
        results = json.loads(results_json)
    else:
        results = load_latest_results()
    
    if not results or not results.get('signals_by_ticker'):
        return [], None
    
    # Get stocks with high confidence signals
    stocks_with_signals = []
    for ticker, signals in results['signals_by_ticker'].items():
        if signals and any(s.get('confidence', 0) > 0.5 for s in signals):
            stocks_with_signals.append(ticker)
    
    options = [{"label": ticker, "value": ticker} for ticker in sorted(stocks_with_signals)]
    
    # Set default value to first stock
    default_value = stocks_with_signals[0] if stocks_with_signals else None
    
    return options, default_value

@app.callback(
    Output("stock-price-chart", "figure"),
    [Input("stock-selector", "value"),
     Input("scan-results-store", "children"),
     Input("interval-component", "n_intervals")],
    prevent_initial_call=True
)
def update_stock_price_chart(selected_ticker, results_json, n_intervals):
    """Update stock price chart with selected ticker"""
    if results_json:
        results = json.loads(results_json)
    else:
        results = load_latest_results()
    
    if not results or not selected_ticker:
        return go.Figure().add_annotation(
            text="Select a stock to view price chart",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    return create_stock_price_chart_with_signals(results, selected_ticker)

@app.callback(
    [Output("signals-summary-cards", "children")],
    [Input("scan-results-store", "children"),
     Input("interval-component", "n_intervals")],
    prevent_initial_call=True
)
def update_signals_summary_cards(results_json, n_intervals):
    """Update high confidence signals summary cards"""
    if results_json:
        results = json.loads(results_json)
    else:
        results = load_latest_results()
    
    if not results:
        return [[]]
    
    summary_cards = create_signals_summary_stats(results)
    return [summary_cards]

if __name__ == '__main__':
    print("Starting VCP Dashboard...")
    print("Access the dashboard at: http://localhost:8050")
    app.run(debug=True, host='0.0.0.0', port=8050) 