import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import PROJECT_ROOT
from app.logger import logger


class ThoughtType(Enum):
    REASONING = "reasoning"
    PLANNING = "planning"
    DECISION = "decision"
    REFLECTION = "reflection"
    OBSERVATION = "observation"
    ERROR = "error"


@dataclass
class AgentThought:
    """Represents a single thought/reasoning step from the agent"""
    timestamp: str
    agent_name: str
    step_number: int
    thought_type: ThoughtType
    content: str
    context: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[str]] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
            "step_number": self.step_number,
            "thought_type": self.thought_type.value,
            "content": self.content,
            "context": self.context,
            "tool_calls": self.tool_calls,
            "confidence": self.confidence
        }


class AgentThoughtLogger:
    """Logger specifically for capturing and storing agent reasoning/thinking processes"""

    def __init__(self, agent_name: str = "unknown"):
        self.agent_name = agent_name
        self.thoughts: List[AgentThought] = []
        self.current_step = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = PROJECT_ROOT / f"logs/thoughts/{agent_name}_{self.session_id}.json"

        # Ensure thoughts directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize log file
        self._write_to_file([])

    def log_thought(
        self,
        content: str,
        thought_type: ThoughtType = ThoughtType.REASONING,
        context: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[str]] = None,
        confidence: Optional[float] = None
    ) -> None:
        """Log a single agent thought/reasoning step"""

        if not content or not content.strip():
            return  # Skip empty thoughts

        self.current_step += 1

        thought = AgentThought(
            timestamp=datetime.now().isoformat(),
            agent_name=self.agent_name,
            step_number=self.current_step,
            thought_type=thought_type,
            content=content.strip(),
            context=context or {},
            tool_calls=tool_calls,
            confidence=confidence
        )

        self.thoughts.append(thought)

        # Log to console with nice formatting
        self._log_to_console(thought)

        # Append to file
        self._append_to_file(thought)

    def log_reasoning(self, content: str, **kwargs) -> None:
        """Log reasoning thoughts"""
        self.log_thought(content, ThoughtType.REASONING, **kwargs)

    def log_planning(self, content: str, **kwargs) -> None:
        """Log planning thoughts"""
        self.log_thought(content, ThoughtType.PLANNING, **kwargs)

    def log_decision(self, content: str, **kwargs) -> None:
        """Log decision-making thoughts"""
        self.log_thought(content, ThoughtType.DECISION, **kwargs)

    def log_reflection(self, content: str, **kwargs) -> None:
        """Log reflection thoughts"""
        self.log_thought(content, ThoughtType.REFLECTION, **kwargs)

    def log_observation(self, content: str, **kwargs) -> None:
        """Log observation thoughts"""
        self.log_thought(content, ThoughtType.OBSERVATION, **kwargs)

    def log_error_thinking(self, content: str, **kwargs) -> None:
        """Log error-related thoughts"""
        self.log_thought(content, ThoughtType.ERROR, **kwargs)

    def get_thoughts_summary(self) -> Dict[str, Any]:
        """Get a summary of all logged thoughts"""
        if not self.thoughts:
            return {"total_thoughts": 0, "summary": "No thoughts logged yet"}

        summary = {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "total_thoughts": len(self.thoughts),
            "total_steps": self.current_step,
            "thought_types": {},
            "latest_thought": self.thoughts[-1].to_dict() if self.thoughts else None,
            "log_file": str(self.log_file)
        }

        # Count thought types
        for thought in self.thoughts:
            thought_type = thought.thought_type.value
            summary["thought_types"][thought_type] = summary["thought_types"].get(thought_type, 0) + 1

        return summary

    def _log_to_console(self, thought: AgentThought) -> None:
        """Log thought to console with nice formatting"""
        icon_map = {
            ThoughtType.REASONING: "🤔",
            ThoughtType.PLANNING: "📋",
            ThoughtType.DECISION: "⚡",
            ThoughtType.REFLECTION: "🪞",
            ThoughtType.OBSERVATION: "👁️",
            ThoughtType.ERROR: "❌"
        }

        icon = icon_map.get(thought.thought_type, "💭")
        logger.info(f"{icon} {thought.agent_name} ({thought.thought_type.value}) Step {thought.step_number}: {thought.content}")

        if thought.tool_calls:
            logger.info(f"   🛠️ Planning to use tools: {', '.join(thought.tool_calls)}")

        if thought.confidence is not None:
            logger.info(f"   📊 Confidence: {thought.confidence:.2f}")

    def _write_to_file(self, thoughts: List[AgentThought]) -> None:
        """Write thoughts to JSON file (overwrites)"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump([thought.to_dict() for thought in thoughts], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write thoughts to file: {e}")

    def _append_to_file(self, thought: AgentThought) -> None:
        """Append a single thought to the JSON file"""
        try:
            # Read existing thoughts
            existing_thoughts = []
            if self.log_file.exists():
                try:
                    with open(self.log_file, 'r') as f:
                        existing_thoughts = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_thoughts = []

            # Append new thought
            existing_thoughts.append(thought.to_dict())

            # Write back to file
            with open(self.log_file, 'w') as f:
                json.dump(existing_thoughts, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to append thought to file: {e}")

    def export_thoughts(self, format: str = "json") -> str:
        """Export thoughts in different formats"""
        if format.lower() == "json":
            return json.dumps([thought.to_dict() for thought in self.thoughts], indent=2)
        elif format.lower() == "txt":
            lines = []
            for thought in self.thoughts:
                lines.append(f"[{thought.timestamp}] Step {thought.step_number} ({thought.thought_type.value}):")
                lines.append(f"  {thought.content}")
                if thought.tool_calls:
                    lines.append(f"  Tools: {', '.join(thought.tool_calls)}")
                lines.append("")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Global thought logger registry
_thought_loggers: Dict[str, AgentThoughtLogger] = {}


def get_thought_logger(agent_name: str) -> AgentThoughtLogger:
    """Get or create a thought logger for a specific agent"""
    if agent_name not in _thought_loggers:
        _thought_loggers[agent_name] = AgentThoughtLogger(agent_name)
    return _thought_loggers[agent_name]


def cleanup_thought_loggers() -> None:
    """Clean up all thought loggers"""
    global _thought_loggers
    for logger_name, thought_logger in _thought_loggers.items():
        logger.info(f"Cleaning up thought logger for {logger_name}")
        summary = thought_logger.get_thoughts_summary()
        logger.info(f"Final summary: {summary}")
    _thought_loggers.clear()
