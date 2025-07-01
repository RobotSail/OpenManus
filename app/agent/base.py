import json
import os
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.agent_thought_logger import ThoughtType, get_thought_logger
from app.llm import LLM
from app.logger import logger
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import ROLE_TYPE, AgentState, Memory, Message


class BaseAgent(BaseModel, ABC):
    """Abstract base class for managing agent state and execution.

    Provides foundational functionality for state transitions, memory management,
    and a step-based execution loop. Subclasses must implement the `step` method.
    """

    # Core attributes
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")

    # Prompts
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )

    # Dependencies
    llm: LLM = Field(default_factory=LLM, description="Language model instance")
    memory: Memory = Field(default_factory=Memory, description="Agent's memory store")
    state: AgentState = Field(
        default=AgentState.IDLE, description="Current agent state"
    )

    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")

    duplicate_threshold: int = 2

    # Thought logging
    _thought_logger = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for flexibility in subclasses

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with default settings if not provided."""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        # Initialize thought logger
        self._thought_logger = get_thought_logger(self.name)
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """Context manager for safe agent state transitions.

        Args:
            new_state: The state to transition to during the context.

        Yields:
            None: Allows execution within the new state.

        Raises:
            ValueError: If the new_state is invalid.
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # Transition to ERROR on failure
            raise e
        finally:
            self.state = previous_state  # Revert to previous state

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Add a message to the agent's memory.

        Args:
            role: The role of the message sender (user, system, assistant, tool).
            content: The message content.
            base64_image: Optional base64 encoded image.
            **kwargs: Additional arguments (e.g., tool_call_id for tool messages).

        Raises:
            ValueError: If the role is unsupported.
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        # Create message with appropriate parameters based on role
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))

    def save_memory(self) -> None:
        """Save the agent's memory to disk with image filtering based on LLM capabilities.

        Saves both JSON and Markdown formats for different use cases.
        For now, filters out all images as requested. In the future, images will be
        saved separately and referenced by path in the logs.
        """
        try:
            # Create memory logs directory
            memory_logs_dir = os.path.join("logs", "memory")
            os.makedirs(memory_logs_dir, exist_ok=True)

            # Generate timestamp and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            base_filename = f"{self.name}_{timestamp}_memory"
            json_filepath = os.path.join(memory_logs_dir, f"{base_filename}.json")
            md_filepath = os.path.join(memory_logs_dir, f"{base_filename}.md")

            # Check LLM capabilities for future use (currently filtering all images)
            should_include_images = False  # As requested, filter all images for now
            # In the future, this would be based on:
            # should_include_images = self.llm.is_multimodal and not self.llm.is_reasoning

            # Prepare memory data for serialization
            memory_data = {
                "agent_name": self.name,
                "agent_description": self.description,
                "timestamp": timestamp,
                "total_steps": self.current_step,
                "max_steps": self.max_steps,
                "final_state": self.state.value if hasattr(self.state, 'value') else str(self.state),
                "llm_info": {
                    "model": self.llm.model,
                    "is_reasoning": self.llm.is_reasoning,
                    "is_multimodal": self.llm.is_multimodal,
                    "images_included": should_include_images
                },
                "messages": []
            }

            # Process messages and filter images
            for message in self.memory.messages:
                msg_dict = message.to_dict() if hasattr(message, 'to_dict') else message.__dict__.copy()

                # Filter out base64_image data as requested
                if "base64_image" in msg_dict and msg_dict["base64_image"]:
                    if should_include_images:
                        # Future implementation: save image to disk and reference path
                        msg_dict["image_path"] = f"[PLACEHOLDER: {len(msg_dict['base64_image'])} chars of image data]"
                    else:
                        # For now, just add a note that an image was present
                        msg_dict["image_filtered"] = True
                    # Remove the actual base64 data
                    del msg_dict["base64_image"]

                memory_data["messages"].append(msg_dict)

            # Write JSON memory to file
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False, default=str)

            # Write Markdown memory to file
            self._save_memory_as_markdown(memory_data, md_filepath)

            logger.info(f"💾 Agent '{self.name}' memory saved to {json_filepath}")
            logger.info(f"📝 Agent '{self.name}' memory saved to {md_filepath}")
            logger.info(f"📊 Memory stats: {len(memory_data['messages'])} messages, images_filtered={not should_include_images}")

            # Log to thought logger if available
            if self._thought_logger:
                self._thought_logger.log_reflection(
                    f"Memory saved to disk with {len(memory_data['messages'])} messages",
                    context={
                        "memory_json_file": json_filepath,
                        "memory_md_file": md_filepath,
                        "images_filtered": not should_include_images,
                        "total_steps": self.current_step,
                        "final_state": memory_data["final_state"]
                    }
                )

        except Exception as e:
            error_msg = f"Failed to save memory for agent '{self.name}': {str(e)}"
            logger.error(f"💥 {error_msg}")
            if self._thought_logger:
                self._thought_logger.log_error_thinking(error_msg)

    def _save_memory_as_markdown(self, memory_data: dict, filepath: str) -> None:
        """Save memory data as formatted markdown."""
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# Agent Memory: {memory_data['agent_name']}\n\n")

            # Metadata
            f.write("## 📊 Execution Summary\n\n")
            f.write(f"- **Agent**: {memory_data['agent_name']}\n")
            if memory_data['agent_description']:
                f.write(f"- **Description**: {memory_data['agent_description']}\n")
            f.write(f"- **Timestamp**: {memory_data['timestamp']}\n")
            f.write(f"- **Total Steps**: {memory_data['total_steps']}/{memory_data['max_steps']}\n")
            f.write(f"- **Final State**: {memory_data['final_state']}\n")
            f.write(f"- **LLM Model**: {memory_data['llm_info']['model']}\n")
            f.write(f"- **Is Reasoning Model**: {memory_data['llm_info']['is_reasoning']}\n")
            f.write(f"- **Is Multimodal Model**: {memory_data['llm_info']['is_multimodal']}\n")
            f.write(f"- **Images Included**: {memory_data['llm_info']['images_included']}\n")
            f.write(f"- **Total Messages**: {len(memory_data['messages'])}\n\n")

            # Messages
            f.write("## 💬 Conversation History\n\n")

            for i, message in enumerate(memory_data['messages'], 1):
                role = message.get('role', 'unknown').upper()
                content = message.get('content', '')

                # Role header with emoji
                role_emoji = {
                    'USER': '👤',
                    'ASSISTANT': '🤖',
                    'SYSTEM': '⚙️',
                    'TOOL': '🔧'
                }.get(role, '❓')

                f.write(f"### {role_emoji} Message {i}: {role}\n\n")

                # Handle tool messages with additional metadata
                if role == 'TOOL':
                    tool_name = message.get('name', 'unknown')
                    tool_call_id = message.get('tool_call_id', 'unknown')
                    f.write(f"**Tool**: {tool_name} (ID: {tool_call_id})\n\n")

                # Handle tool calls in assistant messages
                if 'tool_calls' in message and message['tool_calls']:
                    f.write("**Tool Calls:**\n")
                    for tc in message['tool_calls']:
                        func_name = tc.get('function', {}).get('name', 'unknown')
                        func_args = tc.get('function', {}).get('arguments', '{}')
                        f.write(f"- `{func_name}`: {func_args}\n")
                    f.write("\n")

                # Handle image filtering note
                if message.get('image_filtered'):
                    f.write("🖼️ *[Image content was filtered out]*\n\n")

                # Content
                if content:
                    # Format content nicely
                    if content.startswith('{') and content.endswith('}'):
                        # JSON content - format with code blocks
                        f.write("```json\n")
                        try:
                            formatted_json = json.dumps(json.loads(content), indent=2)
                            f.write(formatted_json)
                        except json.JSONDecodeError:
                            f.write(content)
                        f.write("\n```\n\n")
                    elif '\n' in content:
                        # Multi-line content
                        f.write(f"{content}\n\n")
                    else:
                        # Single line content
                        f.write(f"{content}\n\n")
                else:
                    f.write("*[No content]*\n\n")

                # Separator between messages
                f.write("---\n\n")

    async def additional_cleanup(self) -> None:
        """Additional cleanup hook for subclasses to override.

        This method is called after the main execution loop but before
        memory is saved, allowing subclasses to perform their own cleanup.
        """
        pass

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously.

        Args:
            request: Optional initial user request to process.

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory("user", request)
            # Log the initial user request
            if self._thought_logger:
                self._thought_logger.log_observation(f"User request received: {request}")

        results: List[str] = []
        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(f"Executing step {self.current_step}/{self.max_steps}")

                # Log start of step
                if self._thought_logger:
                    self._thought_logger.log_planning(f"Starting execution step {self.current_step}/{self.max_steps}")

                step_result = await self.step()

                # Check for stuck state
                if self.is_stuck():
                    self.handle_stuck_state()

                results.append(f"Step {self.current_step}: {step_result}")

            if self.current_step >= self.max_steps:
                if self._thought_logger:
                    self._thought_logger.log_reflection(f"Reached maximum steps ({self.max_steps}). Task may be incomplete.")
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(f"Terminated: Reached max steps ({self.max_steps})")

        # Log completion
        if self._thought_logger:
            summary = self._thought_logger.get_thoughts_summary()
            logger.info(f"Agent thoughts summary: {summary}")

        # Clean up in proper order: sandbox, additional cleanup, then save memory
        await SANDBOX_CLIENT.cleanup()
        await self.additional_cleanup()
        self.save_memory()

        return "\n".join(results) if results else "No steps executed"

    @abstractmethod
    async def step(self) -> str:
        """Execute a single step in the agent's workflow.

        Must be implemented by subclasses to define specific behavior.
        """

    def handle_stuck_state(self):
        """Handle stuck state by adding a prompt to change strategy"""
        stuck_prompt = "\
        Observed duplicate responses. Consider new strategies and avoid repeating ineffective paths already attempted."
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"Agent detected stuck state. Added prompt: {stuck_prompt}")

        # Log the stuck state thinking
        if self._thought_logger:
            self._thought_logger.log_reflection(
                "Detected repetitive behavior pattern. Need to try a different approach to avoid getting stuck.",
                context={"stuck_prompt_added": stuck_prompt}
            )

    def is_stuck(self) -> bool:
        """Check if the agent is stuck in a loop by detecting duplicate content"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # Count identical content occurrences
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""
        self.memory.messages = value
