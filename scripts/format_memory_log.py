#!/usr/bin/env python3
"""
Memory Log Formatter

A script to format and display OpenManus agent memory logs in various formats.
Supports both JSON and Markdown input/output formats.

Usage:
    python format_memory_log.py <log_file> [options]

Examples:
    python format_memory_log.py logs/memory/browser_20250106_memory.json
    python format_memory_log.py logs/memory/browser_20250106_memory.json --format markdown
    python format_memory_log.py logs/memory/browser_20250106_memory.json --messages-only
    python format_memory_log.py logs/memory/browser_20250106_memory.json --filter-role user
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

# Add OpenManus to path if running from scripts directory
current_dir = Path(__file__).parent
openmanus_dir = current_dir.parent
sys.path.insert(0, str(openmanus_dir))


def load_memory_log(filepath: str) -> Dict:
    """Load memory log from JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Log file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        if filepath.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError("Only JSON log files are supported for input")


def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display."""
    try:
        # Try parsing the timestamp format used in memory logs
        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S_%f")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return timestamp  # Return as-is if parsing fails


def format_summary(memory_data: Dict, format_type: str = "text") -> str:
    """Format the execution summary."""
    if format_type == "markdown":
        summary = f"# Agent Memory: {memory_data['agent_name']}\n\n"
        summary += "## 📊 Execution Summary\n\n"
        summary += f"- **Agent**: {memory_data['agent_name']}\n"
        if memory_data.get('agent_description'):
            summary += f"- **Description**: {memory_data['agent_description']}\n"
        summary += f"- **Timestamp**: {format_timestamp(memory_data['timestamp'])}\n"
        summary += f"- **Total Steps**: {memory_data['total_steps']}/{memory_data['max_steps']}\n"
        summary += f"- **Final State**: {memory_data['final_state']}\n"
        summary += f"- **LLM Model**: {memory_data['llm_info']['model']}\n"
        summary += f"- **Is Reasoning Model**: {memory_data['llm_info']['is_reasoning']}\n"
        summary += f"- **Is Multimodal Model**: {memory_data['llm_info']['is_multimodal']}\n"
        summary += f"- **Images Included**: {memory_data['llm_info']['images_included']}\n"
        summary += f"- **Total Messages**: {len(memory_data['messages'])}\n\n"
    else:
        summary = f"Agent Memory: {memory_data['agent_name']}\n"
        summary += "=" * 50 + "\n"
        summary += f"Agent: {memory_data['agent_name']}\n"
        if memory_data.get('agent_description'):
            summary += f"Description: {memory_data['agent_description']}\n"
        summary += f"Timestamp: {format_timestamp(memory_data['timestamp'])}\n"
        summary += f"Total Steps: {memory_data['total_steps']}/{memory_data['max_steps']}\n"
        summary += f"Final State: {memory_data['final_state']}\n"
        summary += f"LLM Model: {memory_data['llm_info']['model']}\n"
        summary += f"Is Reasoning Model: {memory_data['llm_info']['is_reasoning']}\n"
        summary += f"Is Multimodal Model: {memory_data['llm_info']['is_multimodal']}\n"
        summary += f"Images Included: {memory_data['llm_info']['images_included']}\n"
        summary += f"Total Messages: {len(memory_data['messages'])}\n\n"

    return summary


def format_message(message: Dict, msg_num: int, format_type: str = "text") -> str:
    """Format a single message."""
    role = message.get('role', 'unknown').upper()
    content = message.get('content', '')

    if format_type == "markdown":
        # Role header with emoji
        role_emoji = {
            'USER': '👤',
            'ASSISTANT': '🤖',
            'SYSTEM': '⚙️',
            'TOOL': '🔧'
        }.get(role, '❓')

        msg_str = f"### {role_emoji} Message {msg_num}: {role}\n\n"

        # Handle tool messages with additional metadata
        if role == 'TOOL':
            tool_name = message.get('name', 'unknown')
            tool_call_id = message.get('tool_call_id', 'unknown')
            msg_str += f"**Tool**: {tool_name} (ID: {tool_call_id})\n\n"

        # Handle tool calls in assistant messages
        if 'tool_calls' in message and message['tool_calls']:
            msg_str += "**Tool Calls:**\n"
            for tc in message['tool_calls']:
                func_name = tc.get('function', {}).get('name', 'unknown')
                func_args = tc.get('function', {}).get('arguments', '{}')
                msg_str += f"- `{func_name}`: {func_args}\n"
            msg_str += "\n"

        # Handle image filtering note
        if message.get('image_filtered'):
            msg_str += "🖼️ *[Image content was filtered out]*\n\n"

        # Content
        if content:
            if content.startswith('{') and content.endswith('}'):
                # JSON content
                msg_str += "```json\n"
                try:
                    formatted_json = json.dumps(json.loads(content), indent=2)
                    msg_str += formatted_json
                except json.JSONDecodeError:
                    msg_str += content
                msg_str += "\n```\n\n"
            else:
                msg_str += f"{content}\n\n"
        else:
            msg_str += "*[No content]*\n\n"

        msg_str += "---\n\n"
    else:
        # Text format
        msg_str = f"Message {msg_num}: {role}\n"
        msg_str += "-" * 30 + "\n"

        if role == 'TOOL':
            tool_name = message.get('name', 'unknown')
            tool_call_id = message.get('tool_call_id', 'unknown')
            msg_str += f"Tool: {tool_name} (ID: {tool_call_id})\n"

        if 'tool_calls' in message and message['tool_calls']:
            msg_str += "Tool Calls:\n"
            for tc in message['tool_calls']:
                func_name = tc.get('function', {}).get('name', 'unknown')
                func_args = tc.get('function', {}).get('arguments', '{}')
                msg_str += f"  - {func_name}: {func_args}\n"

        if message.get('image_filtered'):
            msg_str += "[Image content was filtered out]\n"

        if content:
            msg_str += f"Content:\n{content}\n"
        else:
            msg_str += "Content: [No content]\n"

        msg_str += "\n"

    return msg_str


def format_messages(memory_data: Dict, format_type: str = "text", role_filter: Optional[str] = None) -> str:
    """Format all messages."""
    messages = memory_data['messages']

    if role_filter:
        messages = [msg for msg in messages if msg.get('role', '').lower() == role_filter.lower()]

    if format_type == "markdown":
        result = "## 💬 Conversation History\n\n"
    else:
        result = "Conversation History\n"
        result += "=" * 50 + "\n\n"

    for i, message in enumerate(messages, 1):
        result += format_message(message, i, format_type)

    return result


def save_formatted_log(content: str, output_path: str) -> None:
    """Save formatted content to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Formatted log saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Format OpenManus agent memory logs")
    parser.add_argument("log_file", help="Path to the memory log file (JSON)")
    parser.add_argument("--format", choices=["text", "markdown"], default="text",
                       help="Output format (default: text)")
    parser.add_argument("--output", "-o", help="Output file path (default: print to stdout)")
    parser.add_argument("--messages-only", action="store_true",
                       help="Show only messages, skip summary")
    parser.add_argument("--summary-only", action="store_true",
                       help="Show only summary, skip messages")
    parser.add_argument("--filter-role", choices=["user", "assistant", "system", "tool"],
                       help="Filter messages by role")

    args = parser.parse_args()

    try:
        # Load the memory log
        memory_data = load_memory_log(args.log_file)

        # Generate formatted content
        content = ""

        if not args.messages_only:
            content += format_summary(memory_data, args.format)

        if not args.summary_only:
            content += format_messages(memory_data, args.format, args.filter_role)

        # Output the result
        if args.output:
            save_formatted_log(content, args.output)
        else:
            print(content)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
