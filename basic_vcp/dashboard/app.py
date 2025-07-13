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
    
    # Summary Cards
    dbc.Row([
        dbc.Col(card, width=3) for card in create_summary_cards(initial_results)
    ], className="mb-4"),
    
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
    ], className="mb-4"),
    
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
     Input("interval-component", "n_intervals")]
)
def update_charts_and_tables(results_json, n_intervals):
    """Update charts and tables with latest data"""
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

if __name__ == '__main__':
    print("Starting VCP Dashboard...")
    print("Access the dashboard at: http://localhost:8050")
    app.run(debug=True, host='0.0.0.0', port=8050) 