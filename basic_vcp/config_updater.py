"""
Configuration Updater
Updates config.py with exported parameters from the VCP Parameter Tuner
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, Any

def load_exported_parameters(filename: str) -> Dict[str, Any]:
    """
    Load parameters from an exported JSON file
    
    Args:
        filename: Path to the exported parameters JSON file
        
    Returns:
        Dictionary of parameters
    """
    try:
        with open(filename, 'r') as f:
            params = json.load(f)
        return params
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return {}
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON file: {filename}")
        return {}

def update_config_file(params: Dict[str, Any], backup: bool = True) -> bool:
    """
    Update config.py with new parameters
    
    Args:
        params: Dictionary of parameters to update
        backup: Whether to create a backup of the original config
        
    Returns:
        True if successful, False otherwise
    """
    config_file = "config.py"
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return False
    
    # Create backup if requested
    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"config_backup_{timestamp}.py"
        try:
            with open(config_file, 'r') as f:
                content = f.read()
            with open(backup_file, 'w') as f:
                f.write(content)
            print(f"✅ Backup created: {backup_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not create backup: {e}")
    
    # Read current config
    try:
        with open(config_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading config file: {e}")
        return False
    
    # Parameter mapping from tuner to config
    param_mapping = {
        'min_consolidation_days': 'VCP_MIN_CONSOLIDATION_DAYS',
        'max_consolidation_days': 'VCP_MAX_CONSOLIDATION_DAYS',
        'volatility_contraction_threshold': 'VOLATILITY_CONTRACTION_THRESHOLD',
        'volume_decline_threshold': 'VOLUME_DECLINE_THRESHOLD',
        'breakout_percentage': 'BREAKOUT_PERCENTAGE',
        'breakout_volume_multiplier': 'BREAKOUT_VOLUME_MULTIPLIER',
        'bollinger_period': 'BOLLINGER_BANDS_PERIOD',
        'bollinger_std': 'BOLLINGER_BANDS_STD',
        'rsi_period': 'RSI_PERIOD',
        'swing_lookback_days': 'SWING_POINT_LOOKBACK',
        'swing_threshold': 'SWING_POINT_THRESHOLD',
        'stop_loss_percentage': 'STOP_LOSS_PERCENTAGE',
        'profit_target_multiplier': 'PROFIT_TARGET_MULTIPLIER'
    }
    
    # Update each parameter
    updated_content = content
    updated_count = 0
    
    for tuner_param, config_param in param_mapping.items():
        if tuner_param in params:
            # Find the line with the config parameter
            pattern = rf"(\s*{config_param}\s*=\s*)[^#\n]+"
            replacement = rf"\g<1>{params[tuner_param]}"
            
            if re.search(pattern, updated_content):
                updated_content = re.sub(pattern, replacement, updated_content)
                updated_count += 1
                print(f"✅ Updated {config_param} = {params[tuner_param]}")
            else:
                print(f"⚠️  Warning: Could not find {config_param} in config file")
    
    # Write updated config
    try:
        with open(config_file, 'w') as f:
            f.write(updated_content)
        print(f"\n✅ Successfully updated {updated_count} parameters in {config_file}")
        return True
    except Exception as e:
        print(f"❌ Error writing config file: {e}")
        return False

def list_exported_files() -> list:
    """List all exported parameter files"""
    files = []
    for file in os.listdir('.'):
        if file.startswith('vcp_parameters_') and file.endswith('.json'):
            files.append(file)
    return sorted(files, reverse=True)  # Most recent first

def main():
    """Main function for the config updater"""
    print("🔧 VCP Configuration Updater")
    print("=" * 40)
    
    # List available exported files
    exported_files = list_exported_files()
    
    if not exported_files:
        print("❌ No exported parameter files found.")
        print("💡 Export parameters from the VCP Parameter Tuner first.")
        return
    
    print("📁 Available exported parameter files:")
    for i, file in enumerate(exported_files, 1):
        timestamp = file.replace('vcp_parameters_', '').replace('.json', '')
        try:
            dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = timestamp
        
        print(f"  {i}. {file} ({formatted_time})")
    
    # Get user selection
    try:
        selection = input(f"\nSelect file to apply (1-{len(exported_files)}) or 'q' to quit: ").strip()
        
        if selection.lower() == 'q':
            print("👋 Exiting...")
            return
        
        file_index = int(selection) - 1
        if file_index < 0 or file_index >= len(exported_files):
            print("❌ Invalid selection")
            return
        
        selected_file = exported_files[file_index]
        
    except (ValueError, KeyboardInterrupt):
        print("👋 Exiting...")
        return
    
    # Load and display parameters
    print(f"\n📖 Loading parameters from {selected_file}...")
    params = load_exported_parameters(selected_file)
    
    if not params:
        print("❌ Failed to load parameters")
        return
    
    print("\n📊 Parameters to be applied:")
    for key, value in params.items():
        print(f"  • {key}: {value}")
    
    # Confirm update
    try:
        confirm = input("\n🤔 Apply these parameters to config.py? (y/N): ").strip().lower()
        if confirm != 'y':
            print("👋 Update cancelled")
            return
    except KeyboardInterrupt:
        print("👋 Update cancelled")
        return
    
    # Update config
    print("\n🔧 Updating config.py...")
    success = update_config_file(params, backup=True)
    
    if success:
        print("\n🎉 Configuration updated successfully!")
        print("💡 You can now run the VCP scanner with the new parameters.")
        print("💡 The original config has been backed up.")
    else:
        print("\n❌ Failed to update configuration")

if __name__ == "__main__":
    main() 