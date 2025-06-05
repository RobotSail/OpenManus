# Agent Thought Logging

OpenManus now includes comprehensive thought logging capabilities that capture and store the agent's reasoning process, making it easier to understand how the agent thinks and makes decisions.

## Overview

The thought logging system captures different types of agent thoughts:

- **🤔 Reasoning**: General reasoning and analysis
- **📋 Planning**: Strategic planning and approach decisions
- **⚡ Decision**: Specific decision-making moments
- **🪞 Reflection**: Self-reflection and evaluation
- **👁️ Observation**: Observations about tool results and environment
- **❌ Error**: Error-related thinking and problem-solving

## Features

- **Real-time logging**: Thoughts are logged as they happen during agent execution
- **Structured storage**: Thoughts are stored in JSON format with metadata
- **Console display**: Thoughts are also displayed in the console with emoji indicators
- **Analysis tools**: Built-in tools to analyze thought patterns and statistics
- **Filtering**: Filter thoughts by type, tool usage, step number, etc.
- **Export**: Export thoughts to JSON or text format

## How It Works

The thought logging is automatically integrated into all agent classes. When an agent runs:

1. **User requests** are logged as observations
2. **LLM responses** are analyzed and categorized by thought type
3. **Tool selections** are logged as decisions
4. **Tool executions** are logged as planning and observations
5. **Results** are logged as observations and reflections
6. **Errors** are logged with error thinking

## Viewing Agent Thoughts

### Using the Command Line Tool

The `view_agent_thoughts.py` script provides a comprehensive way to view and analyze thoughts:

```bash
# List available thought files
python view_agent_thoughts.py --list

# View the latest thought file
python view_agent_thoughts.py --latest

# View a specific file with analysis
python view_agent_thoughts.py --file manus_20250604_183246.json --analyze

# Filter thoughts by type
python view_agent_thoughts.py --latest --type reasoning

# Filter thoughts by tool usage
python view_agent_thoughts.py --latest --tool browser_use

# Filter by step range
python view_agent_thoughts.py --latest --min-step 5 --max-step 10

# Show context information
python view_agent_thoughts.py --latest --context

# Export filtered thoughts
python view_agent_thoughts.py --latest --type decision --export decisions.txt
```

### Command Line Options

- `--list, -l`: List available thought files
- `--file, -f`: Specify a thought file to analyze
- `--latest`: Use the most recent thought file
- `--type, -t`: Filter by thought type (reasoning, planning, decision, reflection, observation, error)
- `--tool`: Filter by tool name
- `--min-step`: Minimum step number to include
- `--max-step`: Maximum step number to include
- `--analyze, -a`: Show analysis and statistics
- `--context, -c`: Show context information
- `--export`: Export to file (JSON or text format)

## File Locations

Thought logs are stored in:
```
logs/thoughts/
├── manus_20250604_183246.json
├── browser_20250604_184512.json
└── toolcall_20250604_185023.json
```

Each file contains a JSON array of thought objects with the following structure:

```json
{
  "timestamp": "2025-06-04T18:32:46.123456",
  "agent_name": "manus",
  "step_number": 1,
  "thought_type": "reasoning",
  "content": "I need to analyze the user's request and determine the best approach...",
  "context": {
    "tool_choice_mode": "auto",
    "available_tools": 5,
    "response_has_tool_calls": true
  },
  "tool_calls": ["browser_use"],
  "confidence": 0.85
}
```

## Integration in Your Code

The thought logging is automatically integrated, but you can also manually log thoughts:

```python
from app.agent_thought_logger import get_thought_logger, ThoughtType

# Get the thought logger for your agent
thought_logger = get_thought_logger("my_agent")

# Log different types of thoughts
thought_logger.log_reasoning("I'm analyzing the problem...")
thought_logger.log_planning("My strategy will be to...")
thought_logger.log_decision("I've decided to use the browser tool")
thought_logger.log_observation("The tool returned successful results")
thought_logger.log_reflection("This approach worked well")
thought_logger.log_error_thinking("I encountered an error and need to retry")

# Log with additional context
thought_logger.log_thought(
    "Complex reasoning about the problem",
    thought_type=ThoughtType.REASONING,
    tool_calls=["tool1", "tool2"],
    confidence=0.9,
    context={"custom_data": "value"}
)
```

## Analysis Examples

### Basic Analysis
```bash
python view_agent_thoughts.py --latest --analyze
```

Output:
```
ANALYSIS
==================================================
Total thoughts: 25
Thought types: {'reasoning': 8, 'planning': 5, 'decision': 4, 'observation': 6, 'reflection': 2}
Tools used: {'browser_use': 12, 'python_execute': 3, 'terminate': 1}
Timeline: 2025-06-04T18:32:46.123456 to 2025-06-04T18:45:23.789012
```

### Filtering Examples
```bash
# See only decision-making thoughts
python view_agent_thoughts.py --latest --type decision

# See thoughts related to browser usage
python view_agent_thoughts.py --latest --tool browser_use

# See thoughts from steps 5-10
python view_agent_thoughts.py --latest --min-step 5 --max-step 10
```

## Benefits

1. **Debugging**: Understand why the agent made certain decisions
2. **Optimization**: Identify patterns in agent reasoning
3. **Transparency**: See the complete thought process
4. **Learning**: Understand how different prompts affect agent thinking
5. **Troubleshooting**: Quickly identify where things went wrong

## Console Output

During agent execution, you'll see enhanced console output like:

```
🤔 manus (reasoning) Step 1: I need to analyze the user's request for creating a web scraper...
   🛠️ Planning to use tools: browser_use
   📊 Confidence: 0.85

⚡ manus (decision) Step 1: Decided to use 1 tool(s): browser_use

📋 manus (planning) Step 1: About to execute 1 tool(s): browser_use

👁️ manus (observation) Step 1: Executing tool 1/1: browser_use

👁️ manus (observation) Step 1: Tool 'browser_use' executed successfully. Result: Navigated to https://example.com

🪞 manus (reflection) Step 1: Completed execution of all 1 tool(s). Ready to analyze results and plan next steps.
```

## Configuration

The thought logging system works out of the box with no configuration needed. However, you can customize:

- Log file locations by modifying `PROJECT_ROOT` in `app/config.py`
- Thought categorization logic in `app/agent/toolcall.py`
- Console output formatting in `app/agent_thought_logger.py`

## Cleanup

Thought loggers are automatically cleaned up when the agent session ends. You can also manually clean up:

```python
from app.agent_thought_logger import cleanup_thought_loggers
cleanup_thought_loggers()
```

This will log final summaries and clear the logger registry.
