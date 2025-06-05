#!/usr/bin/env python3
"""
Utility script to view and analyze agent thoughts from the thought logs.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.config import PROJECT_ROOT


def list_thought_files() -> List[Path]:
    """List all available thought log files"""
    thoughts_dir = PROJECT_ROOT / "logs/thoughts"
    if not thoughts_dir.exists():
        return []

    return sorted(thoughts_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)


def load_thoughts(file_path: Path) -> List[Dict[str, Any]]:
    """Load thoughts from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading thoughts from {file_path}: {e}")
        return []


def format_thought(thought: Dict[str, Any], show_context: bool = False) -> str:
    """Format a single thought for display"""
    timestamp = thought.get('timestamp', 'Unknown')
    step = thought.get('step_number', 'N/A')
    thought_type = thought.get('thought_type', 'unknown')
    content = thought.get('content', '')
    tool_calls = thought.get('tool_calls', [])
    confidence = thought.get('confidence')

    # Format timestamp
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        time_str = dt.strftime('%H:%M:%S')
    except:
        time_str = timestamp

    # Choose icon based on thought type
    icons = {
        'reasoning': '🤔',
        'planning': '📋',
        'decision': '⚡',
        'reflection': '🪞',
        'observation': '👁️',
        'error': '❌'
    }
    icon = icons.get(thought_type, '💭')

    lines = [f"{icon} [{time_str}] Step {step} ({thought_type}):"]
    lines.append(f"   {content}")

    if tool_calls:
        lines.append(f"   🛠️ Tools: {', '.join(tool_calls)}")

    if confidence is not None:
        lines.append(f"   📊 Confidence: {confidence:.2f}")

    if show_context and thought.get('context'):
        lines.append(f"   📝 Context: {thought['context']}")

    return '\n'.join(lines)


def analyze_thoughts(thoughts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze thoughts and provide statistics"""
    if not thoughts:
        return {"error": "No thoughts to analyze"}

    analysis = {
        "total_thoughts": len(thoughts),
        "thought_types": {},
        "tools_used": {},
        "timeline": {
            "first_thought": thoughts[0].get('timestamp'),
            "last_thought": thoughts[-1].get('timestamp'),
        },
        "confidence_stats": {
            "thoughts_with_confidence": 0,
            "avg_confidence": 0,
            "min_confidence": None,
            "max_confidence": None
        }
    }

    confidences = []

    for thought in thoughts:
        # Count thought types
        thought_type = thought.get('thought_type', 'unknown')
        analysis["thought_types"][thought_type] = analysis["thought_types"].get(thought_type, 0) + 1

        # Count tools used
        tool_calls = thought.get('tool_calls', []) or []
        for tool in tool_calls:
            analysis["tools_used"][tool] = analysis["tools_used"].get(tool, 0) + 1

        # Collect confidence scores
        confidence = thought.get('confidence')
        if confidence is not None:
            confidences.append(confidence)

    # Calculate confidence statistics
    if confidences:
        analysis["confidence_stats"]["thoughts_with_confidence"] = len(confidences)
        analysis["confidence_stats"]["avg_confidence"] = sum(confidences) / len(confidences)
        analysis["confidence_stats"]["min_confidence"] = min(confidences)
        analysis["confidence_stats"]["max_confidence"] = max(confidences)

    return analysis


def filter_thoughts(thoughts: List[Dict[str, Any]],
                   thought_type: str = None,
                   tool_name: str = None,
                   min_step: int = None,
                   max_step: int = None) -> List[Dict[str, Any]]:
    """Filter thoughts based on criteria"""
    filtered = thoughts

    if thought_type:
        filtered = [t for t in filtered if t.get('thought_type') == thought_type]

    if tool_name:
        filtered = [t for t in filtered if tool_name in (t.get('tool_calls') or [])]

    if min_step is not None:
        filtered = [t for t in filtered if t.get('step_number', 0) >= min_step]

    if max_step is not None:
        filtered = [t for t in filtered if t.get('step_number', float('inf')) <= max_step]

    return filtered


def main():
    parser = argparse.ArgumentParser(description="View and analyze agent thoughts")
    parser.add_argument("--list", "-l", action="store_true", help="List available thought files")
    parser.add_argument("--file", "-f", help="Specific thought file to analyze")
    parser.add_argument("--latest", action="store_true", help="Use the latest thought file")
    parser.add_argument("--type", "-t", help="Filter by thought type")
    parser.add_argument("--tool", help="Filter by tool name")
    parser.add_argument("--min-step", type=int, help="Minimum step number")
    parser.add_argument("--max-step", type=int, help="Maximum step number")
    parser.add_argument("--analyze", "-a", action="store_true", help="Show analysis/statistics")
    parser.add_argument("--context", "-c", action="store_true", help="Show context information")
    parser.add_argument("--export", help="Export filtered thoughts to file (json or txt)")

    args = parser.parse_args()

    # List available files
    thought_files = list_thought_files()

    if args.list:
        print("Available thought files:")
        for i, file_path in enumerate(thought_files):
            size = file_path.stat().st_size
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            print(f"  {i+1}. {file_path.name} ({size} bytes, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        return

    if not thought_files:
        print("No thought files found. Make sure to run the agent first to generate thought logs.")
        return

    # Select file to analyze
    if args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = PROJECT_ROOT / "logs/thoughts" / file_path
    elif args.latest:
        file_path = thought_files[0]
    else:
        # Interactive selection
        print("Available thought files:")
        for i, fp in enumerate(thought_files[:10]):  # Show only last 10
            mtime = datetime.fromtimestamp(fp.stat().st_mtime)
            print(f"  {i+1}. {fp.name} ({mtime.strftime('%Y-%m-%d %H:%M:%S')})")

        try:
            choice = int(input("Select a file (number): ")) - 1
            file_path = thought_files[choice]
        except (ValueError, IndexError):
            print("Invalid selection")
            return

    # Load thoughts
    thoughts = load_thoughts(file_path)
    if not thoughts:
        print(f"No thoughts found in {file_path}")
        return

    print(f"\nLoaded {len(thoughts)} thoughts from {file_path.name}")

    # Apply filters
    filtered_thoughts = filter_thoughts(
        thoughts,
        thought_type=args.type,
        tool_name=args.tool,
        min_step=args.min_step,
        max_step=args.max_step
    )

    if len(filtered_thoughts) != len(thoughts):
        print(f"Filtered to {len(filtered_thoughts)} thoughts")

    # Show analysis
    if args.analyze:
        analysis = analyze_thoughts(filtered_thoughts)
        print("\n" + "="*50)
        print("ANALYSIS")
        print("="*50)
        print(f"Total thoughts: {analysis['total_thoughts']}")
        print(f"Thought types: {analysis['thought_types']}")
        print(f"Tools used: {analysis['tools_used']}")
        print(f"Timeline: {analysis['timeline']['first_thought']} to {analysis['timeline']['last_thought']}")

        conf_stats = analysis['confidence_stats']
        if conf_stats['thoughts_with_confidence'] > 0:
            print(f"Confidence: {conf_stats['avg_confidence']:.2f} avg, "
                  f"{conf_stats['min_confidence']:.2f} min, "
                  f"{conf_stats['max_confidence']:.2f} max "
                  f"({conf_stats['thoughts_with_confidence']} thoughts)")

    # Export if requested
    if args.export:
        export_path = Path(args.export)
        if export_path.suffix.lower() == '.json':
            with open(export_path, 'w') as f:
                json.dump(filtered_thoughts, f, indent=2)
        else:
            with open(export_path, 'w') as f:
                for thought in filtered_thoughts:
                    f.write(format_thought(thought, show_context=args.context) + '\n\n')
        print(f"Exported {len(filtered_thoughts)} thoughts to {export_path}")
        return

    # Display thoughts
    print("\n" + "="*50)
    print("THOUGHTS")
    print("="*50)

    for thought in filtered_thoughts:
        print(format_thought(thought, show_context=args.context))
        print()


if __name__ == "__main__":
    main()
