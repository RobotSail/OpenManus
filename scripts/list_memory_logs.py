#!/usr/bin/env python3
"""
Memory Log Listing and Quick Viewer

A convenience script to list available memory logs and provide quick viewing options.

Usage:
    python list_memory_logs.py [options]

Examples:
    python list_memory_logs.py                      # List all logs
    python list_memory_logs.py --latest             # Show latest log summary
    python list_memory_logs.py --agent browser      # Show logs for browser agent
    python list_memory_logs.py --latest --full      # Show full latest log
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add OpenManus to path if running from scripts directory
current_dir = Path(__file__).parent
openmanus_dir = current_dir.parent
sys.path.insert(0, str(openmanus_dir))

from scripts.format_memory_log import format_messages, format_summary, load_memory_log


def find_memory_logs(logs_dir: str = "logs/memory") -> List[Dict]:
    """Find all memory log files and return metadata."""
    logs = []

    if not os.path.exists(logs_dir):
        return logs

    for filename in os.listdir(logs_dir):
        if filename.endswith('_memory.json'):
            filepath = os.path.join(logs_dir, filename)
            try:
                # Parse filename for metadata
                parts = filename.replace('_memory.json', '').split('_')
                if len(parts) >= 3:
                    agent_name = parts[0]
                    timestamp = '_'.join(parts[1:])

                    # Get file stats
                    stat = os.stat(filepath)
                    file_size = stat.st_size

                    # Load basic info from file
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        message_count = len(data.get('messages', []))
                        final_state = data.get('final_state', 'unknown')
                        llm_model = data.get('llm_info', {}).get('model', 'unknown')
                    except:
                        message_count = 0
                        final_state = 'unknown'
                        llm_model = 'unknown'

                    logs.append({
                        'filename': filename,
                        'filepath': filepath,
                        'agent_name': agent_name,
                        'timestamp': timestamp,
                        'file_size': file_size,
                        'message_count': message_count,
                        'final_state': final_state,
                        'llm_model': llm_model
                    })
            except Exception as e:
                print(f"Warning: Could not parse {filename}: {e}")

    # Sort by timestamp (newest first)
    logs.sort(key=lambda x: x['timestamp'], reverse=True)
    return logs


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S_%f")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return timestamp


def list_logs(logs: List[Dict], agent_filter: Optional[str] = None) -> None:
    """List all memory logs in a table format."""
    if agent_filter:
        logs = [log for log in logs if log['agent_name'].lower() == agent_filter.lower()]

    if not logs:
        print("No memory logs found.")
        return

    print("Memory Logs")
    print("=" * 80)
    print(f"{'Agent':<15} {'Timestamp':<20} {'Messages':<8} {'State':<10} {'Size':<8} {'Model':<15}")
    print("-" * 80)

    for log in logs:
        agent = log['agent_name'][:14]
        timestamp = format_timestamp(log['timestamp'])[:19]
        messages = str(log['message_count'])
        state = log['final_state'][:9]
        size = format_file_size(log['file_size'])
        model = log['llm_model'][:14]

        print(f"{agent:<15} {timestamp:<20} {messages:<8} {state:<10} {size:<8} {model:<15}")

    print(f"\nTotal: {len(logs)} log files")


def show_latest_log(logs: List[Dict], agent_filter: Optional[str] = None, full: bool = False) -> None:
    """Show the latest memory log."""
    if agent_filter:
        logs = [log for log in logs if log['agent_name'].lower() == agent_filter.lower()]

    if not logs:
        print("No memory logs found.")
        return

    latest = logs[0]  # Already sorted newest first
    print(f"Latest log: {latest['filename']}")
    print("=" * 50)

    try:
        memory_data = load_memory_log(latest['filepath'])

        # Always show summary
        print(format_summary(memory_data, "text"))

        if full:
            print(format_messages(memory_data, "text"))
        else:
            print("Use --full to see all messages")

    except Exception as e:
        print(f"Error loading log: {e}")


def main():
    parser = argparse.ArgumentParser(description="List and view OpenManus agent memory logs")
    parser.add_argument("--agent", help="Filter by agent name")
    parser.add_argument("--latest", action="store_true", help="Show latest log summary")
    parser.add_argument("--full", action="store_true", help="Show full log (with --latest)")
    parser.add_argument("--logs-dir", default="logs/memory", help="Memory logs directory")

    args = parser.parse_args()

    try:
        logs = find_memory_logs(args.logs_dir)

        if args.latest:
            show_latest_log(logs, args.agent, args.full)
        else:
            list_logs(logs, args.agent)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
