import json
from typing import TYPE_CHECKING, Optional

from pydantic import Field, model_validator

from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.prompt.browser import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import Function, Message, ToolCall, ToolChoice
from app.tool import BrowserUseTool, Terminate, ToolCollection
from app.tool.base import BaseTool

# Avoid circular import if BrowserAgent needs BrowserContextHelper
if TYPE_CHECKING:
    from app.agent.base import BaseAgent  # Or wherever memory is defined




# A tool for introspecting the current state and assessing what we should do next.
_INTROSPECT_DESCRIPTION = """A tool for evaluating the current browser state and planning next actions.

This tool helps track progress and plan next steps by:
1. Evaluating if previous actions were successful
2. Maintaining memory of what has been done
3. Setting the next immediate goal

The response must include:
- previous_goal_evaluation: Analysis of whether previous actions succeeded as intended
- previous_goal_status: Success/failed/unknown status of previous goal
- memory: Detailed tracking of progress, including counts of completed vs remaining items
- next_goal: Clear statement of the next immediate action needed

Use this tool to:
- Track progress on multi-step tasks
- Maintain counts for repetitive tasks (e.g. "0 out of 10 websites analyzed")
- Evaluate success/failure of previous actions
- Plan next steps based on current state

Important:
When present, this tool must ALWAYS be called, and must be the first tool in the call order.
"""


class IntrospectCurrentStateTool(BaseTool):
    name: str = "introspect_current_state"
    description: str = _INTROSPECT_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "previous_goal_evaluation": {
                "type": "string",
                "description": "Analyze the current elements and the image to check if the previous goals/actions are successful like intended by the task. Mention if something unexpected happened. Shortly state why/why not",
            },
            "previous_goal_status": {
                "type": "string",
                "description": "Indicate the status of the previous goal, should be one of: success|failed|unknown",
                "enum": ["success", "failed", "unknown"]
            },
            "memory": {
                "type": "string",
                "description": "Description of what has been done and what you need to remember. Be very specific. Count here ALWAYS how many times you have done something and how many remain. E.g. 0 out of 10 websites analyzed. Continue with abc and xyz"
            },
            "next_goal": {
                "description": "What needs to be done with the next immediate action"
            },
        },
        "required": ["previous_goal_evaluation", "previous_goal_status", "memory", "next_goal"]
    }

    async def execute(self, previous_goal_evaluation: str, previous_goal_status: str, memory: str, next_goal: str) -> str:
        """Finish the current execution"""
        # this message just serves as extra conditioning for the actual model, we don't have to do much of anything here
        print(f"{previous_goal_evaluation=}")
        print(f"{previous_goal_status=}")
        print(f"{memory=}")
        print(f"{next_goal=}")




class BrowserContextHelper:
    def __init__(self, agent: "BaseAgent"):
        self.agent = agent
        self._current_base64_image: Optional[str] = None

    async def get_browser_state(self) -> Optional[dict]:
        browser_tool: BrowserUseTool = self.agent.available_tools.get_tool(BrowserUseTool().name)
        if not browser_tool or not hasattr(browser_tool, "get_current_state"):
            logger.warning("BrowserUseTool not found or doesn't have get_current_state")
            return None
        try:
            result = await browser_tool.get_current_state()
            if result.error:
                logger.debug(f"Browser state error: {result.error}")
                return None
            if hasattr(result, "base64_image") and result.base64_image:
                self._current_base64_image = result.base64_image
            else:
                self._current_base64_image = None
            return json.loads(result.output)
        except Exception as e:
            logger.debug(f"Failed to get browser state: {str(e)}")
            return None

    async def format_next_step_prompt(self) -> str:
        """Gets browser state and formats the browser prompt."""
        browser_state = await self.get_browser_state()
        url_info, tabs_info, content_above_info, content_below_info = "", "", "", ""
        results_info = ""  # Or get from agent if needed elsewhere

        if browser_state and not browser_state.get("error"):
            url_info = f"\n   URL: {browser_state.get('url', 'N/A')}\n   Title: {browser_state.get('title', 'N/A')}"
            tabs = browser_state.get("tabs", [])
            if tabs:
                tabs_info = f"\n   {len(tabs)} tab(s) available"
            pixels_above = browser_state.get("pixels_above", 0)
            pixels_below = browser_state.get("pixels_below", 0)
            if pixels_above > 0:
                content_above_info = f" ({pixels_above} pixels)"
            if pixels_below > 0:
                content_below_info = f" ({pixels_below} pixels)"

            if self._current_base64_image:
                image_message = Message.user_message(
                    content="Current browser screenshot:",
                    base64_image=self._current_base64_image,
                )
                self.agent.memory.add_message(image_message)
                self._current_base64_image = None  # Consume the image after adding

        pixels_info = "\n"
        if not pixels_above and not pixels_below:
            pixels_info += "    No pixels above or below viewport."
        else:
            pixels_above = f"    " + ("Above: {pixels_above} pixels" if pixels_above else "No pixels above")
            pixels_below = f"    " + ("Below: {pixels_below} pixels" if pixels_below else "No pixels below")
            pixels_info += "\n".join([pixels_above, pixels_below])



        next_step_prompt = f"""[Current State]
Step: {self.agent.current_step}/{self.agent.max_steps}
Current URL and page title:{url_info}
Available tabs:{tabs_info}
Interactive elements:
{browser_state["interactive_elements"]}
Area outside of viewport:{pixels_info}

What should I do next to accomplish my goal?
"""
        return next_step_prompt
        # return NEXT_STEP_PROMPT.format(
        #     url_placeholder=url_info,
        #     tabs_placeholder=tabs_info,
        #     content_above_placeholder=content_above_info,
        #     content_below_placeholder=content_below_info,
        #     results_placeholder=results_info,
        # )

    async def cleanup_browser(self):
        browser_tool = self.agent.available_tools.get_tool(BrowserUseTool().name)
        if browser_tool and hasattr(browser_tool, "cleanup"):
            await browser_tool.cleanup()

def print_messages(messages: list[Message]):
    content = ""
    for msg in messages:
        content += f"{msg.role.upper()}:\n{msg.content}"
        if msg.base64_image:
            content += "\n\n{[IMAGE]}"


        content += "\n\n"
    print(content)

class BrowserAgent(ToolCallAgent):
    """
    A browser agent that uses the browser_use library to control a browser.

    This agent can navigate web pages, interact with elements, fill forms,
    extract content, and perform other browser-based actions to accomplish tasks.
    """

    name: str = "browser"
    description: str = "A browser agent that can control a browser to accomplish tasks"

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # Configure the available tools
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(IntrospectCurrentStateTool(), BrowserUseTool(), Terminate())
    )

    # Use Auto for tool choice to allow both tool usage and free-form responses
    tool_choices: ToolChoice = ToolChoice.AUTO
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    browser_context_helper: Optional[BrowserContextHelper] = None

    @model_validator(mode="after")
    def initialize_helper(self) -> "BrowserAgent":
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    async def think(self) -> bool:
        """Process current state and decide next actions using tools, with browser state info added"""
        # Get the browser state prompt
        browser_state_prompt = await self.browser_context_helper.format_next_step_prompt()

        # If there's an existing next_step_prompt (user request), append the browser state
        self.next_step_prompt = browser_state_prompt
        # if self.next_step_prompt:
        #     self.next_step_prompt = f"{self.next_step_prompt}\n\n[Current state starts here]\n{browser_state_prompt}"
        # else:
        #     self.next_step_prompt = browser_state_prompt

        try:
            # here is what we're about to send to the LLM

            # Add user message to conversation
            if self.next_step_prompt:
                user_msg = Message.user_message(self.next_step_prompt)
                self.messages += [user_msg]

            system_msgs=(
                [Message.system_message(self.system_prompt)]
                if self.system_prompt
                else None
            )
            print_messages(system_msgs + self.messages)
            # Use ask_tool to get proper tool responses
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except Exception as e:
            error_msg = f"LLM call failed: {str(e)}"
            logger.error(f"🚨 {error_msg}")
            if self._thought_logger:
                self._thought_logger.log_error_thinking(error_msg, context={"browser_context": True})
            self.memory.add_message(
                Message.assistant_message(f"Error encountered while processing: {str(e)}")
            )
            return False

        # Extract tool calls and content from response
        tool_calls = response.tool_calls if response and response.tool_calls else []
        content = response.content if response and response.content else ""

        # If no tool calls but we have content, try to parse JSON from content
        if not tool_calls and content:
            content = content.strip().lstrip('```json').lstrip('```').rstrip('```')
            try:
                if content and content.strip():
                    # Extract JSON from the response
                    json_data = json.loads(content)
                    actions = json_data.get("action", [])

                    # Convert JSON actions to ToolCall objects
                    for i, action in enumerate(actions):
                        if "browser_use" in action:
                            browser_action = action["browser_use"]
                            tool_call = ToolCall(
                                id=f"call_{i+1}",
                                function=Function(
                                    name="browser_use",
                                    arguments=json.dumps(browser_action)
                                )
                            )
                            tool_calls.append(tool_call)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to parse JSON tool calls from response: {e}")
                # If parsing fails, keep tool_calls as empty list

        self.tool_calls = tool_calls

        # Log the agent's reasoning/thoughts to the thought logger
        if self._thought_logger:
            # Browser agents often get responses with no content but tool calls
            # Let's capture both scenarios
            if content and content.strip():
                # Determine the type of thinking based on content
                from app.agent_thought_logger import ThoughtType
                thought_type = ThoughtType.REASONING
                if "plan" in content.lower() or "strategy" in content.lower() or "approach" in content.lower():
                    thought_type = ThoughtType.PLANNING
                elif "decide" in content.lower() or "choose" in content.lower() or "select" in content.lower():
                    thought_type = ThoughtType.DECISION
                elif "observe" in content.lower() or "see" in content.lower() or "notice" in content.lower():
                    thought_type = ThoughtType.OBSERVATION

                tool_names = [call.function.name for call in tool_calls] if tool_calls else None

                self._thought_logger.log_thought(
                    content,
                    thought_type=thought_type,
                    tool_calls=tool_names,
                    context={
                        "tool_choice_mode": self.tool_choices.value if hasattr(self.tool_choices, 'value') else str(self.tool_choices),
                        "available_tools": len(self.available_tools.tools) if self.available_tools else 0,
                        "response_has_tool_calls": bool(tool_calls),
                        "browser_context": True
                    }
                )
            elif tool_calls:
                # No content but there are tool calls - this is common for browser agents
                # Log what tools are being used and why (inferred)
                tool_names = [call.function.name for call in tool_calls]
                if len(tool_calls) == 1 and tool_calls[0].function.name == "browser_use":
                    # Parse the browser action to understand what the agent is thinking
                    try:
                        args = json.loads(tool_calls[0].function.arguments)
                        action = args.get('action', 'unknown')
                        reasoning_content = f"Need to perform browser action: {action}"
                        if action == "go_to_url":
                            reasoning_content = f"Navigating to URL: {args.get('url', 'unknown')}"
                        elif action == "click_element":
                            reasoning_content = f"Clicking on element at index {args.get('index', 'unknown')}"
                        elif action == "input_text":
                            reasoning_content = f"Inputting text into element at index {args.get('index', 'unknown')}"
                        elif action == "scroll_down":
                            reasoning_content = f"Scrolling down by {args.get('scroll_amount', 'unknown')} pixels"
                        elif action == "extract_content":
                            reasoning_content = f"Extracting content with goal: {args.get('goal', 'unknown')}"

                        self._thought_logger.log_decision(
                            reasoning_content,
                            tool_calls=tool_names,
                            context={
                                "browser_action": action,
                                "tool_arguments": args,
                                "inferred_reasoning": True
                            }
                        )
                    except Exception as e:
                        # Fallback if parsing fails
                        self._thought_logger.log_decision(
                            f"Decided to use browser tool (content not provided by model)",
                            tool_calls=tool_names,
                            context={"content_empty": True, "parsing_error": str(e)}
                        )

        # Log response info (keeping original logging for compatibility)
        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(f"🛠️ {self.name} selected {len(tool_calls)} tools to use")
        if tool_calls:
            logger.info(f"🧰 Tools being prepared: {[call.function.name for call in tool_calls]}")
            logger.info(f"🔧 Tool arguments: {tool_calls[0].function.arguments}")

            # Log decision-making about tool selection to thought logger
            if self._thought_logger:
                self._thought_logger.log_decision(
                    f"Decided to use {len(tool_calls)} tool(s): {', '.join([call.function.name for call in tool_calls])}",
                    tool_calls=[call.function.name for call in tool_calls],
                    context={
                        "tool_arguments": {call.function.name: call.function.arguments for call in tool_calls},
                        "browser_context": True
                    }
                )

        # Here, this only works for R1-style reasoning models
        si = content.find("<think>")
        ei = content.find("</think>", si)
        ei = ei + len("</think>") if ei > 0 else -1

        if 0 <= si < ei:
            assert "<think>" in content and "</think>" in content
            content = content[ei:]
            assert "<think>" not in content and "</think>" not in content

        # also let's just quickly check our memory
        for m in self.memory.messages:
            assert "<think>" not in m.content and "</think>" not in m.content


        # Create and add assistant message
        assistant_msg = (
            Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
            if self.tool_calls
            else Message.assistant_message(content)
        )
        self.memory.add_message(assistant_msg)

        # Handle different tool_choices modes
        if self.tool_choices == ToolChoice.NONE:
            if tool_calls:
                warning_msg = f"Attempted to use tools when they weren't available!"
                logger.warning(f"🤔 Hmm, {self.name} {warning_msg}")
                if self._thought_logger:
                    self._thought_logger.log_error_thinking(warning_msg, context={"browser_context": True})
            if content:
                return True
            return False

        if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
            if self._thought_logger:
                self._thought_logger.log_reflection("Tool calls were required but none were provided. Will need to try again.", context={"browser_context": True})
            return True  # Will be handled in act()

        # For 'auto' mode, continue with content if no commands but content exists
        if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
            if self._thought_logger and content:
                self._thought_logger.log_reflection("No tools selected, but provided reasoning. Continuing with text response.", context={"browser_context": True})
            return bool(content)

        return bool(self.tool_calls)

    async def additional_cleanup(self):
        """Clean up browser agent resources by calling browser cleanup and parent cleanup."""
        await self.browser_context_helper.cleanup_browser()
        await super().additional_cleanup()

    async def cleanup(self):
        """Clean up browser agent resources by calling parent cleanup."""
        await self.browser_context_helper.cleanup_browser()
