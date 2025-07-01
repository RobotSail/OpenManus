import asyncio
import base64
import json
from typing import Generic, Optional, TypeVar

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService
from pydantic import Field, field_validator, model_validator
from pydantic_core.core_schema import ValidationInfo

from app.config import config
from app.llm import LLM
from app.logger import logger
from app.tool.base import BaseTool, ToolResult
from app.tool.web_search import WebSearch

_BROWSER_DESCRIPTION = """\
A powerful browser automation tool that allows interaction with web pages through various actions.
* This tool provides commands for controlling a browser session, navigating web pages, and extracting information
* It maintains state across calls, keeping the browser session alive until explicitly closed
* Use this when you need to browse websites, fill forms, click buttons, extract content, or perform web searches
* Each action requires specific parameters as defined in the tool's dependencies

Key capabilities include:
* Navigation: Go to specific URLs, go back, search the web, or refresh pages
* Interaction: Click elements, input text, select from dropdowns, send keyboard commands
* Scrolling: Scroll up/down by pixel amount or scroll to specific text
* Content extraction: Extract and analyze content from web pages based on specific goals
* Tab management: Switch between tabs, open new tabs, or close tabs

Note: When using element indices, refer to the numbered elements shown in the current browser state.
"""

Context = TypeVar("Context")


class BrowserUseTool(BaseTool, Generic[Context]):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "go_to_url",
                    "click_element",
                    "input_text",
                    "scroll_down",
                    "scroll_up",
                    "scroll_to_text",
                    "send_keys",
                    "get_dropdown_options",
                    "select_dropdown_option",
                    "go_back",
                    "web_search",
                    "wait",
                    "extract_content",
                    "switch_tab",
                    "open_tab",
                    "close_tab",
                ],
                "description": "The browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL for 'go_to_url' or 'open_tab' actions",
            },
            "index": {
                "type": "integer",
                "description": "Element index for 'click_element', 'input_text', 'get_dropdown_options', or 'select_dropdown_option' actions",
            },
            "text": {
                "type": "string",
                "description": "Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Pixels to scroll (positive for down, negative for up) for 'scroll_down' or 'scroll_up' actions",
            },
            "tab_id": {
                "type": "integer",
                "description": "Tab ID for 'switch_tab' action",
            },
            "query": {
                "type": "string",
                "description": "Search query for 'web_search' action",
            },
            "goal": {
                "type": "string",
                "description": "Extraction goal for 'extract_content' action",
            },
            "keys": {
                "type": "string",
                "description": "Keys to send for 'send_keys' action",
            },
            "seconds": {
                "type": "integer",
                "description": "Seconds to wait for 'wait' action",
            },
        },
        "required": ["action"],
        "dependencies": {
            "go_to_url": ["url"],
            "click_element": ["index"],
            "input_text": ["index", "text"],
            "switch_tab": ["tab_id"],
            "open_tab": ["url"],
            "scroll_down": ["scroll_amount"],
            "scroll_up": ["scroll_amount"],
            "scroll_to_text": ["text"],
            "send_keys": ["keys"],
            "get_dropdown_options": ["index"],
            "select_dropdown_option": ["index", "text"],
            "go_back": [],
            "web_search": ["query"],
            "wait": ["seconds"],
            "extract_content": ["goal"],
        },
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    context: Optional[BrowserContext] = Field(default=None, exclude=True)
    dom_service: Optional[DomService] = Field(default=None, exclude=True)
    web_search_tool: WebSearch = Field(default_factory=WebSearch, exclude=True)

    # Context for generic functionality
    tool_context: Optional[Context] = Field(default=None, exclude=True)

    # Initialize LLM with vision configuration
    llm: Optional[LLM] = Field(default_factory=lambda: LLM("vision"), exclude=True)

    @model_validator(mode="after")
    def initialize_llm(self) -> "BrowserUseTool":
        """Initialize LLM with vision configuration if not already set."""
        if not self.llm:
            self.llm = LLM("vision")
        return self

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, v: dict, info: ValidationInfo) -> dict:
        if not v:
            raise ValueError("Parameters cannot be empty")
        return v

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """Ensure browser and context are initialized."""
        if self.browser is None:
            browser_config_kwargs = {"headless": False, "disable_security": True}

            if config.browser_config:
                from browser_use.browser.browser import ProxySettings

                # handle proxy settings.
                if config.browser_config.proxy and config.browser_config.proxy.server:
                    browser_config_kwargs["proxy"] = ProxySettings(
                        server=config.browser_config.proxy.server,
                        username=config.browser_config.proxy.username,
                        password=config.browser_config.proxy.password,
                    )

                browser_attrs = [
                    "headless",
                    "disable_security",
                    "extra_chromium_args",
                    "chrome_instance_path",
                    "wss_url",
                    "cdp_url",
                ]

                for attr in browser_attrs:
                    value = getattr(config.browser_config, attr, None)
                    if value is not None:
                        if not isinstance(value, list) or value:
                            browser_config_kwargs[attr] = value

            self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))

        if self.context is None:
            context_config = BrowserContextConfig()

            # if there is context config in the config, use it.
            if (
                config.browser_config
                and hasattr(config.browser_config, "new_context_config")
                and config.browser_config.new_context_config
            ):
                context_config = config.browser_config.new_context_config

            self.context = await self.browser.new_context(context_config)
            self.dom_service = DomService(await self.context.get_current_page())

        return self.context

    async def execute(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        scroll_amount: Optional[int] = None,
        tab_id: Optional[int] = None,
        query: Optional[str] = None,
        goal: Optional[str] = None,
        keys: Optional[str] = None,
        seconds: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a specified browser action.

        Args:
            action: The browser action to perform
            url: URL for navigation or new tab
            index: Element index for click or input actions
            text: Text for input action or search query
            scroll_amount: Pixels to scroll for scroll action
            tab_id: Tab ID for switch_tab action
            query: Search query for Google search
            goal: Extraction goal for content extraction
            keys: Keys to send for keyboard actions
            seconds: Seconds to wait
            **kwargs: Additional arguments

        Returns:
            ToolResult with the action's output or error
        """
        async with self.lock:
            try:
                context = await self._ensure_browser_initialized()

                # Get max content length from config
                max_content_length = getattr(
                    config.browser_config, "max_content_length", 2000
                )

                # Navigation actions
                if action == "go_to_url":
                    if not url:
                        return ToolResult(
                            error="URL is required for 'go_to_url' action"
                        )
                    page = await context.get_current_page()
                    await page.goto(url)
                    await page.wait_for_load_state("domcontentloaded")

                    # For SPAs like Reddit, wait for network to be idle
                    try:
                        await page.wait_for_load_state("networkidle", timeout=5000)
                    except:
                        # If networkidle fails, just wait a bit longer
                        await asyncio.sleep(3)

                    # Give extra time for dynamic content and scripts to load
                    await asyncio.sleep(2)

                    # Try a small scroll to trigger any lazy loading
                    try:
                        await page.evaluate("window.scrollTo(0, 100); window.scrollTo(0, 0);")
                        await asyncio.sleep(0.5)
                    except:
                        pass

                    # Refresh DOM service for the new page
                    if self.dom_service:
                        self.dom_service = DomService(page)

                    return ToolResult(output=f"Navigated to {url}")

                elif action == "go_back":
                    await context.go_back()
                    return ToolResult(output="Navigated back")

                elif action == "refresh":
                    await context.refresh_page()
                    return ToolResult(output="Refreshed current page")

                elif action == "web_search":
                    if not query:
                        return ToolResult(
                            error="Query is required for 'web_search' action"
                        )
                    # Execute the web search and return results directly without browser navigation
                    search_response = await self.web_search_tool.execute(
                        query=query, fetch_content=True, num_results=1
                    )
                    # Navigate to the first search result
                    first_search_result = search_response.results[0]
                    url_to_navigate = first_search_result.url

                    page = await context.get_current_page()
                    await page.goto(url_to_navigate)
                    await page.wait_for_load_state()

                    return search_response

                # Element interaction actions
                elif action == "click_element":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'click_element' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")

                    # Try to locate the element first
                    element_handle = await context.get_locate_element(element)
                    if not element_handle:
                        # Fallback: try direct interaction using JavaScript and XPath
                        page = await context.get_current_page()
                        try:
                            # Use XPath to find and click the element
                            xpath = element.xpath
                            await page.evaluate(f"""
                                () => {{
                                    const element = document.evaluate('{xpath}', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                    if (element) {{
                                        element.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                                        setTimeout(() => element.click(), 100);
                                        return true;
                                    }}
                                    return false;
                                }}
                            """)
                            return ToolResult(output=f"Clicked element at index {index} using JavaScript fallback")
                        except Exception as e:
                            return ToolResult(error=f"Failed to click element at index {index}. Element location failed and JavaScript fallback failed: {str(e)}")

                    # Standard click if element was found
                    try:
                        download_path = await context._click_element_node(element)
                        output = f"Clicked element at index {index}"
                        if download_path:
                            output += f" - Downloaded file to {download_path}"
                        return ToolResult(output=output)
                    except Exception as e:
                        return ToolResult(error=f"Failed to click element at index {index}: {str(e)}")

                elif action == "input_text":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'input_text' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")

                    # Try to locate the element first
                    element_handle = await context.get_locate_element(element)
                    if not element_handle:
                        # Fallback: try direct interaction using JavaScript and XPath
                        page = await context.get_current_page()
                        try:
                            # Use XPath to find and fill the element
                            xpath = element.xpath
                            escaped_text = text.replace("'", "\\'")
                            await page.evaluate(f"""
                                () => {{
                                    const element = document.evaluate('{xpath}', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                    if (element) {{
                                        element.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                                        element.focus();

                                        // Clear existing content
                                        if (element.value !== undefined) {{
                                            element.value = '';
                                        }} else if (element.textContent !== undefined) {{
                                            element.textContent = '';
                                        }}

                                        // Input new text
                                        if (element.value !== undefined) {{
                                            element.value = '{escaped_text}';
                                            element.dispatchEvent(new Event('input', {{bubbles: true}}));
                                            element.dispatchEvent(new Event('change', {{bubbles: true}}));
                                        }} else {{
                                            element.textContent = '{escaped_text}';
                                            element.dispatchEvent(new Event('input', {{bubbles: true}}));
                                        }}

                                        return true;
                                    }}
                                    return false;
                                }}
                            """)
                            return ToolResult(output=f"Input '{text}' into element at index {index} using JavaScript fallback")
                        except Exception as e:
                            return ToolResult(error=f"Failed to input text into element at index {index}. Element location failed and JavaScript fallback failed: {str(e)}")

                    # Standard input if element was found
                    try:
                        await context._input_text_element_node(element, text)
                        return ToolResult(
                            output=f"Input '{text}' into element at index {index}"
                        )
                    except Exception as e:
                        return ToolResult(error=f"Failed to input text into element at index {index}: {str(e)}")

                elif action == "scroll_down" or action == "scroll_up":
                    direction = 1 if action == "scroll_down" else -1
                    amount = (
                        scroll_amount
                        if scroll_amount is not None
                        else context.config.browser_window_size["height"]
                    )
                    await context.execute_javascript(
                        f"window.scrollBy(0, {direction * amount});"
                    )
                    return ToolResult(
                        output=f"Scrolled {'down' if direction > 0 else 'up'} by {amount} pixels"
                    )

                elif action == "scroll_to_text":
                    if not text:
                        return ToolResult(
                            error="Text is required for 'scroll_to_text' action"
                        )
                    page = await context.get_current_page()
                    try:
                        locator = page.get_by_text(text, exact=False)
                        await locator.scroll_into_view_if_needed()
                        return ToolResult(output=f"Scrolled to text: '{text}'")
                    except Exception as e:
                        return ToolResult(error=f"Failed to scroll to text: {str(e)}")

                elif action == "send_keys":
                    if not keys:
                        return ToolResult(
                            error="Keys are required for 'send_keys' action"
                        )
                    page = await context.get_current_page()
                    await page.keyboard.press(keys)
                    return ToolResult(output=f"Sent keys: {keys}")

                elif action == "get_dropdown_options":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'get_dropdown_options' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    options = await page.evaluate(
                        """
                        (xpath) => {
                            const select = document.evaluate(xpath, document, null,
                                XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                            if (!select) return null;
                            return Array.from(select.options).map(opt => ({
                                text: opt.text,
                                value: opt.value,
                                index: opt.index
                            }));
                        }
                    """,
                        element.xpath,
                    )
                    return ToolResult(output=f"Dropdown options: {options}")

                elif action == "select_dropdown_option":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'select_dropdown_option' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    await page.select_option(element.xpath, label=text)
                    return ToolResult(
                        output=f"Selected option '{text}' from dropdown at index {index}"
                    )

                # Content extraction actions
                elif action == "extract_content":
                    if not goal:
                        return ToolResult(
                            error="Goal is required for 'extract_content' action"
                        )

                    page = await context.get_current_page()
                    import markdownify

                    content = markdownify.markdownify(await page.content())

                    # Take a screenshot for visual analysis - only capture visible viewport
                    screenshot = await page.screenshot(
                        full_page=False, animations="disabled", type="jpeg", quality=100
                    )
                    base64_screenshot = base64.b64encode(screenshot).decode("utf-8")

                    # Split the prompt into system and user messages
                    system_prompt = "You are an AI assistant that extracts and analyzes webpage content both visually and textually. You should extract information based on specific goals and format your response in JSON."

                    user_prompt = f"""\
Please extract content from this webpage based on the following goal. If the goal is vague, summarize the page.
Extraction goal: {goal}

Page content:
{content[:max_content_length]}
"""
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": user_prompt,
                        }
                    ]

                    # Use ask_with_images to leverage vision capabilities
                    try:
                        response = await self.llm.ask_with_images(
                            messages=messages,
                            images=[{
                                "url": f"data:image/jpeg;base64,{base64_screenshot}"
                            }],
                            # temperature=0.0
                        )

                        # Parse the response as JSON
                        try:
                            extracted_content = json.loads(response)
                        except json.JSONDecodeError:
                            # If response is not JSON, wrap it in a basic structure
                            extracted_content = {
                                "extracted_content": {
                                    "text": response,
                                    "visual_elements": {
                                        "layout": "Could not parse visual elements",
                                        "interactive_elements": []
                                    },
                                    "metadata": {
                                        "source": "webpage"
                                    }
                                }
                            }

                        return ToolResult(
                            output=f"Extracted from page:\n{extracted_content}\n"
                        )
                    except Exception as e:
                        return ToolResult(
                            error=f"Failed to extract content: {str(e)}"
                        )

                # Tab management actions
                elif action == "switch_tab":
                    if tab_id is None:
                        return ToolResult(
                            error="Tab ID is required for 'switch_tab' action"
                        )
                    await context.switch_to_tab(tab_id)
                    page = await context.get_current_page()
                    await page.wait_for_load_state()
                    return ToolResult(output=f"Switched to tab {tab_id}")

                elif action == "open_tab":
                    if not url:
                        return ToolResult(error="URL is required for 'open_tab' action")
                    await context.create_new_tab(url)
                    return ToolResult(output=f"Opened new tab with {url}")

                elif action == "close_tab":
                    await context.close_current_tab()
                    return ToolResult(output="Closed current tab")

                # Utility actions
                elif action == "wait":
                    seconds_to_wait = seconds if seconds is not None else 3
                    await asyncio.sleep(seconds_to_wait)
                    return ToolResult(output=f"Waited for {seconds_to_wait} seconds")

                else:
                    return ToolResult(error=f"Unknown action: {action}")

            except Exception as e:
                return ToolResult(error=f"Browser action '{action}' failed: {str(e)}")

    async def get_current_state(
        self, context: Optional[BrowserContext] = None
    ) -> ToolResult:
        """
        Get the current browser state as a ToolResult.
        If context is not provided, uses self.context.
        """
        try:
            # First ensure browser is initialized
            if not context:
                context = await self._ensure_browser_initialized()

            # Get the current page and ensure it's fully loaded
            page = await context.get_current_page()
            await page.bring_to_front()
            await page.wait_for_load_state("domcontentloaded")

            # Add a small delay to ensure dynamic content loads
            await asyncio.sleep(0.5)

            # Force refresh the element tree if it's None or empty
            state = await context.get_state()

            # Check if we have elements before trying to get them
            has_elements = False
            if state.element_tree:
                try:
                    test_elements = state.element_tree.clickable_elements_to_string()
                    has_elements = len(test_elements.strip()) > 0
                except:
                    has_elements = False

            # If element tree is None or has no clickable elements, try to force regeneration
            if not state.element_tree or not has_elements:
                try:
                    logger.debug(f"DEBUG: Forcing element tree regeneration (tree exists: {state.element_tree is not None}, has elements: {has_elements})")

                    # Special handling for Reddit and other SPAs with anti-bot detection
                    if 'reddit.com' in state.url.lower():
                        await self._setup_reddit_stealth_mode(page)

                    # Wait longer for dynamic content (especially for SPAs like Reddit)
                    await asyncio.sleep(2)

                    # Try to trigger any lazy loading by scrolling
                    try:
                        await page.evaluate("window.scrollTo(0, 100); window.scrollTo(0, 0);")
                        await asyncio.sleep(0.5)
                    except:
                        pass

                    # Wait for network to be idle (important for React/Vue apps)
                    try:
                        await page.wait_for_load_state("networkidle", timeout=3000)
                    except:
                        pass

                    # Force DOM service refresh
                    if self.dom_service:
                        self.dom_service = DomService(page)

                    # Get state again
                    state = await context.get_state()

                    # If still no elements, try one more time with longer wait
                    if state.element_tree:
                        test_elements = state.element_tree.clickable_elements_to_string()
                        if not test_elements.strip():
                            logger.debug("DEBUG: Still no elements, trying final regeneration...")
                            await asyncio.sleep(3)
                            state = await context.get_state()

                except Exception as dom_error:
                    logger.warning(f"DOM service refresh failed: {dom_error}")

            # Create a viewport_info dictionary if it doesn't exist
            viewport_height = 0
            if hasattr(state, "viewport_info") and state.viewport_info:
                viewport_height = state.viewport_info.height
            elif hasattr(context, "config") and hasattr(context.config, "browser_window_size"):
                viewport_height = context.config.browser_window_size.get("height", 0)

            # Only capture the visible viewport
            screenshot = await page.screenshot(
                full_page=False, animations="disabled", type="jpeg", quality=100
            )

            screenshot = base64.b64encode(screenshot).decode("utf-8")

            # Get interactive elements
            interactive_elements = ""
            if state.element_tree:
                try:
                    interactive_elements = state.element_tree.clickable_elements_to_string()
                except Exception as e:
                    logger.warning(f"Failed to get clickable elements: {e}")
                    interactive_elements = ""

            # Debug output to help troubleshoot
            logger.debug(f"DEBUG: Element tree exists: {state.element_tree is not None}")
            logger.debug(f"DEBUG: Interactive elements length: {len(interactive_elements)}")
            if interactive_elements:
                logger.debug(f"DEBUG: Interactive elements preview: {interactive_elements[:200]}...")

            # Build the state info with all required fields
            state_info = {
                "url": state.url,
                "title": state.title,
                "tabs": [tab.model_dump() for tab in state.tabs],
                "help": "[1], [2], [3], etc., represent clickable indices corresponding to the elements listed. Clicking on these indices will navigate to or interact with the respective content behind them.",
                "interactive_elements": interactive_elements,
                "scroll_info": {
                    "pixels_above": getattr(state, "pixels_above", 0),
                    "pixels_below": getattr(state, "pixels_below", 0),
                    "total_height": getattr(state, "pixels_above", 0)
                    + getattr(state, "pixels_below", 0)
                    + viewport_height,
                },
                "viewport_height": viewport_height,
            }

            return ToolResult(
                output=json.dumps(state_info, indent=4, ensure_ascii=False),
                base64_image=screenshot,
            )
        except Exception as e:
            return ToolResult(error=f"Failed to get browser state: {str(e)}")

    async def cleanup(self):
        """Clean up browser resources."""
        async with self.lock:
            if self.context is not None:
                await self.context.close()
                self.context = None
                self.dom_service = None
            if self.browser is not None:
                await self.browser.close()
                self.browser = None

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        if self.browser is not None or self.context is not None:
            try:
                asyncio.run(self.cleanup())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.cleanup())
                loop.close()

    @classmethod
    def create_with_context(cls, context: Context) -> "BrowserUseTool[Context]":
        """Factory method to create a BrowserUseTool with a specific context."""
        tool = cls()
        tool.tool_context = context
        return tool
