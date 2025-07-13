"""
Main VCP Scanner - Orchestrates the entire VCP detection system
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import List, Dict, Optional
from pathlib import Path
import json
from datetime import datetime, timedelta
import concurrent.futures
import multiprocessing

from data.stock_data import SQLiteStockDataLoader
from analysis.vcp_detector import VCPDetector, VCPPattern, BreakoutSignal
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vcp_scanner.log'),
        logging.StreamHandler()
    ]
)

class VCPScanner:
    def __init__(self, config: Config = None):
        """
        Initialize the VCP Scanner
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_loader = SQLiteStockDataLoader()
        self.vcp_detector = VCPDetector(self.config.get_vcp_parameters())
        
        # Results storage
        self.vcp_patterns = []
        self.breakout_signals = []
        self.scan_results = {}
        self.max_workers = multiprocessing.cpu_count()  # Use all available cores
        
    def scan_stocks(self, tickers: List[str] = None, 
                   start_date: str = None, end_date: str = None) -> Dict:
        """
        Scan multiple stocks for VCP patterns in parallel
        """
        if tickers is None:
            tickers = self.config.DEFAULT_STOCKS
        
        self.logger.info(f"Starting VCP scan for {len(tickers)} stocks (parallel, {self.max_workers} workers)")
        
        results = {
            'scan_date': datetime.now().isoformat(),
            'tickers_scanned': len(tickers),
            'vcp_patterns_found': 0,
            'breakout_signals': 0,
            'patterns_by_ticker': {},
            'signals_by_ticker': {},
            'summary': {}
        }
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {executor.submit(self._scan_ticker_worker, ticker, start_date, end_date): ticker for ticker in tickers}
            for i, future in enumerate(concurrent.futures.as_completed(future_to_ticker), 1):
                ticker, ticker_results = future.result()
                results['patterns_by_ticker'][ticker] = ticker_results['patterns']
                results['signals_by_ticker'][ticker] = ticker_results['signals']
                if ticker_results['patterns']:
                    results['vcp_patterns_found'] += len(ticker_results['patterns'])
                if ticker_results['signals']:
                    results['breakout_signals'] += len(ticker_results['signals'])
                self.logger.info(f"Scanned {ticker} ({i}/{len(tickers)})")
        
        # Generate summary statistics
        results['summary'] = self._generate_summary(results)
        self.scan_results = results
        return results
    
    def scan_single_stock(self, ticker: str, start_date: str = None, 
                         end_date: str = None) -> Dict:
        """
        Scan a single stock for VCP patterns
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with patterns and signals for this stock
        """
        # Load stock data
        df = self.data_loader.load_stock_data(ticker, start_date, end_date)
        
        if df is None:
            self.logger.warning(f"No data available for {ticker}")
            return {'patterns': [], 'signals': []}
        
        # Validate data quality
        if not self.data_loader.validate_data_quality(df):
            self.logger.warning(f"Data quality insufficient for {ticker}")
            return {'patterns': [], 'signals': []}
        
        # Detect VCP patterns
        patterns = self.vcp_detector.detect_vcp_patterns(df, ticker)
        
        # Detect breakout signals for each pattern
        all_signals = []
        for pattern in patterns:
            signals = self.vcp_detector.detect_breakout_signals(df, pattern)
            all_signals.extend(signals)
        
        return {
            'patterns': patterns,
            'signals': all_signals
        }
    
    def _generate_summary(self, results: Dict) -> Dict:
        """
        Generate summary statistics from scan results
        
        Args:
            results: Scan results dictionary
            
        Returns:
            Summary statistics
        """
        summary = {
            'total_patterns': results['vcp_patterns_found'],
            'total_signals': results['breakout_signals'],
            'stocks_with_patterns': 0,
            'stocks_with_signals': 0,
            'pattern_strength_distribution': {},
            'signal_confidence_distribution': {},
            'top_patterns': [],
            'top_signals': []
        }
        
        # Count stocks with patterns and signals
        for ticker in results['patterns_by_ticker']:
            if results['patterns_by_ticker'][ticker]:
                summary['stocks_with_patterns'] += 1
            if results['signals_by_ticker'][ticker]:
                summary['stocks_with_signals'] += 1
        
        # Collect all patterns and signals for analysis
        all_patterns = []
        all_signals = []
        
        for ticker in results['patterns_by_ticker']:
            all_patterns.extend(results['patterns_by_ticker'][ticker])
            all_signals.extend(results['signals_by_ticker'][ticker])
        
        # Analyze pattern strength distribution
        if all_patterns:
            strengths = [p.pattern_strength for p in all_patterns]
            summary['pattern_strength_distribution'] = {
                'mean': np.mean(strengths),
                'median': np.median(strengths),
                'std': np.std(strengths),
                'min': np.min(strengths),
                'max': np.max(strengths)
            }
            
            # Top patterns by strength
            sorted_patterns = sorted(all_patterns, key=lambda x: x.pattern_strength, reverse=True)
            summary['top_patterns'] = [
                {
                    'ticker': p.ticker,
                    'strength': p.pattern_strength,
                    'consolidation_days': p.consolidation_days,
                    'start_date': p.start_date.strftime('%Y-%m-%d'),
                    'end_date': p.end_date.strftime('%Y-%m-%d')
                }
                for p in sorted_patterns[:10]
            ]
        
        # Analyze signal confidence distribution
        if all_signals:
            confidences = [s.confidence for s in all_signals]
            summary['signal_confidence_distribution'] = {
                'mean': np.mean(confidences),
                'median': np.median(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
            
            # Top signals by confidence
            sorted_signals = sorted(all_signals, key=lambda x: x.confidence, reverse=True)
            summary['top_signals'] = [
                {
                    'ticker': s.ticker,
                    'confidence': s.confidence,
                    'signal_type': s.signal_type,
                    'date': s.date.strftime('%Y-%m-%d'),
                    'price': s.price,
                    'risk_reward_ratio': s.risk_reward_ratio
                }
                for s in sorted_signals[:10]
            ]
        
        return summary
    
    def _scan_ticker_worker(self, ticker: str, start_date: str = None, end_date: str = None):
        """Worker method for parallel scanning"""
        try:
            return ticker, self.scan_single_stock(ticker, start_date, end_date)
        except Exception as e:
            return ticker, {'patterns': [], 'signals': []}
    
    def save_results(self, filename: str = None) -> str:
        """
        Save scan results to a JSON file
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"vcp_scan_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = self._make_serializable(self.scan_results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filename}")
        return filename
    
    def _make_serializable(self, obj):
        """
        Convert objects to serializable format for JSON
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Handle dataclasses and other objects with __dict__
            serializable_dict = {}
            for k, v in obj.__dict__.items():
                if isinstance(v, (pd.Timestamp, datetime)):
                    serializable_dict[k] = v.isoformat()
                else:
                    serializable_dict[k] = self._make_serializable(v)
            return serializable_dict
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif obj is None:
            # Convert None to appropriate default values based on context
            return 0.0  # Default for numeric fields like volume
        else:
            return obj
    
    def get_active_patterns(self, min_strength: float = 0.7) -> List[VCPPattern]:
        """
        Get currently active VCP patterns above minimum strength
        
        Args:
            min_strength: Minimum pattern strength (0-1)
            
        Returns:
            List of active VCP patterns
        """
        active_patterns = []
        
        for ticker in self.scan_results.get('patterns_by_ticker', {}):
            patterns = self.scan_results['patterns_by_ticker'][ticker]
            for pattern in patterns:
                if pattern.pattern_strength >= min_strength and pattern.status == 'forming':
                    active_patterns.append(pattern)
        
        return sorted(active_patterns, key=lambda x: x.pattern_strength, reverse=True)
    
    def get_recent_signals(self, days: int = 30, min_confidence: float = 0.7) -> List[BreakoutSignal]:
        """
        Get recent breakout signals above minimum confidence
        
        Args:
            days: Number of days to look back
            min_confidence: Minimum signal confidence (0-1)
            
        Returns:
            List of recent breakout signals
        """
        recent_signals = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for ticker in self.scan_results.get('signals_by_ticker', {}):
            signals = self.scan_results['signals_by_ticker'][ticker]
            for signal in signals:
                if (signal.confidence >= min_confidence and 
                    signal.date >= cutoff_date):
                    recent_signals.append(signal)
        
        return sorted(recent_signals, key=lambda x: x.confidence, reverse=True)

def main():
    """Main function to run the VCP scanner"""
    print("VCP Pattern Detection System")
    print("=" * 50)
    
    # Initialize scanner
    scanner = VCPScanner()
    
    # Get available tickers
    available_tickers = scanner.data_loader.get_available_tickers()
    print(f"Found {len(available_tickers)} stock data files")
    
    # Scan stocks (use first 50 for demo)
    demo_tickers = available_tickers[:10]
    print(f"Scanning {len(demo_tickers)} stocks for VCP patterns...")
    
    # Run scan
    results = scanner.scan_stocks(demo_tickers)
    
    # Print results
    print("\nScan Results:")
    print(f"Stocks scanned: {results['tickers_scanned']}")
    print(f"VCP patterns found: {results['vcp_patterns_found']}")
    print(f"Breakout signals: {results['breakout_signals']}")
    
    # Print top patterns
    if results['summary']['top_patterns']:
        print("\nTop VCP Patterns:")
        for i, pattern in enumerate(results['summary']['top_patterns'][:5]):
            print(f"{i+1}. {pattern['ticker']} - Strength: {pattern['strength']:.3f} "
                  f"({pattern['consolidation_days']} days)")
    
    # Print top signals
    if results['summary']['top_signals']:
        print("\nTop Breakout Signals:")
        for i, signal in enumerate(results['summary']['top_signals'][:5]):
            print(f"{i+1}. {signal['ticker']} - {signal['signal_type'].upper()} "
                  f"Confidence: {signal['confidence']:.3f} "
                  f"R/R: {signal['risk_reward_ratio']:.2f}")
    
    # Save results
    filename = scanner.save_results()
    print(f"\nResults saved to: {filename}")
    
    return results

if __name__ == "__main__":
    main() 