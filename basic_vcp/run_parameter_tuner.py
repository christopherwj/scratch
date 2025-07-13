#!/usr/bin/env python3
"""
VCP Parameter Tuner Launcher
Interactive GUI for adjusting VCP detection parameters with real-time chart updates
"""

import sys
import os

# Add the current directory to Python path to ensure proper imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dashboard.vcp_parameter_tuner import VCPParameterTuner

def main():
    """Launch the VCP Parameter Tuner dashboard"""
    print("🚀 Launching VCP Parameter Tuner...")
    print("=" * 50)
    print("This interactive dashboard allows you to:")
    print("• Adjust VCP detection parameters in real-time")
    print("• See how parameter changes affect the ideal VCP pattern")
    print("• Export optimized parameters for use in the main system")
    print("• Understand the relationship between parameters and pattern detection")
    print("=" * 50)
    
    try:
        # Create and run the parameter tuner
        tuner = VCPParameterTuner()
        
        print(f"📊 Dashboard will be available at: http://localhost:8051")
        print("🔄 Charts will update automatically as you adjust parameters")
        print("💡 Use the sliders to experiment with different settings")
        print("💾 Click 'Export Parameters' to save your optimized settings")
        print("\nPress Ctrl+C to stop the server")
        
        # Run the dashboard
        tuner.run(debug=True, port=8051)
        
    except KeyboardInterrupt:
        print("\n👋 Parameter tuner stopped by user")
    except Exception as e:
        print(f"❌ Error launching parameter tuner: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 