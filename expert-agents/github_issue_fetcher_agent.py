# expert-agents/github_issue_fetcher_agent.py
import re
import logging
import json
from pydantic import BaseModel, Field
from typing import Optional, Any

from google.adk.agents import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext
# CallbackContext no longer needed here if after_tool_callback is removed
# from google.adk.agents.callback_context import CallbackContext 
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool # Still needed for direct call
from google.adk.tools.base_tool import BaseTool # Not used by this agent's LLM
from google.adk.tools.tool_context import ToolContext # For direct call
from google.adk.agents.invocation_context import InvocationContext # For direct call
from google.genai import types as genai_types

from .browser_agent import browser_agent, BrowserAgentInput, BrowserAgentOutput 
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call
from .tools import get_gemini_api_key_from_secret_manager

logger = logging.getLogger(__name__)
get_gemini_api_key_from_secret_manager()

BOILERPLATE_STRINGS_TO_REMOVE = [
    "Is your feature request related to a problem? Please describe.",
    "Describe the solution you'd like",
    "Describe alternatives you've considered",
    "Describe the bug",
    "Minimal Reproduction",
    "Minimal steps to reproduce",
    "Desktop (please complete the following information):",
    "Please make sure you read the contribution guide and file the issues in the rigth place.",
    "To Reproduce",
    "Expected behavior",
    "Screenshots",
    "Additional context"
]

def clean_github_issue_text(text: str) -> str:
    if not text: return ""
    cleaned_text = text
    for boilerplate in BOILERPLATE_STRINGS_TO_REMOVE:
        cleaned_text = cleaned_text.replace(boilerplate, "")
    return re.sub(r'\n\s*\n', '\n\n', cleaned_text).strip()

class GitHubIssueFetcherToolInput(BaseModel):
    issue_number: str = Field(description="The GitHub issue number for 'google/adk-python'.")

class GitHubIssueFetcherOutput(BaseModel):
    cleaned_issue_details: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None

def get_text_from_content(content: genai_types.Content) -> str:
    if content and content.parts and content.parts[0].text:
        return content.parts[0].text
    return ""

# This function now needs to be async
async def github_issue_fetcher_instruction_provider(context: ReadonlyContext) -> str:
    invocation_ctx = context._invocation_context
    user_input_json_str = get_text_from_content(invocation_ctx.user_content) if invocation_ctx and invocation_ctx.user_content else ""
    
    issue_number_str = None
    final_output_data_for_llm = {}

    if not user_input_json_str:
        logger.error("FetcherAgent (instruction_provider): No input JSON provided.")
        final_output_data_for_llm = {"error": "Fetcher agent received no input."}
    else:
        try:
            input_data = GitHubIssueFetcherToolInput.model_validate_json(user_input_json_str)
            issue_number_str = input_data.issue_number
        except Exception as e:
            logger.error(f"FetcherAgent (instruction_provider): Error parsing input '{user_input_json_str}': {e}")
            final_output_data_for_llm = {"error": "Fetcher agent received invalid input for issue_number."}

    if issue_number_str: # Proceed with fetching if issue number is valid
        github_url = f"https://github.com/google/adk-python/issues/{issue_number_str}"
        logger.info(f"FetcherAgent (instruction_provider): Directly calling browser_agent for URL: {github_url}")

        browser_agent_as_tool = AgentTool(agent=browser_agent)
        # browser_agent (sub-agent) expects BrowserAgentInput
        browser_agent_args = BrowserAgentInput(url=github_url).model_dump()
        
        # Create a dummy ToolContext for the direct call
        # The InvocationContext for this agent is `invocation_ctx`
        dummy_tool_context_for_browser_call = ToolContext(invocation_ctx)

        browser_tool_response_dict = None
        try:
            # Directly await the AgentTool's run_async.
            # AgentTool will handle calling browser_agent, which in turn directly calls ExtractGitHubIssueDetailsTool
            # and then browser_agent's LLM formats the BrowserAgentOutput JSON.
            # AgentTool then parses that JSON into a dict because browser_agent has an output_schema.
            browser_tool_response_dict = await browser_agent_as_tool.run_async(
                args=browser_agent_args, 
                tool_context=dummy_tool_context_for_browser_call
            )
            logger.info(f"FetcherAgent: Received from direct browser_agent call: {str(browser_tool_response_dict)[:150]}")
            
            # Process the dictionary received from browser_agent
            if isinstance(browser_tool_response_dict, dict):
                browser_data = BrowserAgentOutput.model_validate(browser_tool_response_dict)
                if browser_data.extracted_details:
                    cleaned_details = clean_github_issue_text(browser_data.extracted_details)
                    if not cleaned_details:
                        final_output_data_for_llm = {"message": "GitHub issue content empty/template after cleaning."}
                    else:
                        final_output_data_for_llm = {"cleaned_issue_details": cleaned_details}
                elif browser_data.error:
                    final_output_data_for_llm = {"error": f"Browser sub-agent reported: {browser_data.error}"}
                elif browser_data.message:
                    final_output_data_for_llm = {"message": f"Browser sub-agent message: {browser_data.message}"}
                else:
                    final_output_data_for_llm = {"message": "Browser sub-agent returned no specific details, error, or message."}
            else:
                logger.error(f"FetcherAgent: Expected dict from browser_agent call, got {type(browser_tool_response_dict)}")
                final_output_data_for_llm = {"error": "Internal error: Unexpected data type from browser utility."}

        except Exception as e:
            logger.error(f"FetcherAgent (instruction_provider): Error during direct call to browser_agent: {e}", exc_info=True)
            final_output_data_for_llm = {"error": f"Failed to process GitHub issue via browser: {e}"}
    
    elif not final_output_data_for_llm.get("error"): # If no issue_number and no prior error
        logger.error("FetcherAgent (instruction_provider): No issue_number provided and no prior error recorded.")
        final_output_data_for_llm = {"error": "Fetcher agent not provided with issue_number."}

    # Instruct this agent's LLM to output the final JSON based on GitHubIssueFetcherOutput
    output_model_instance = GitHubIssueFetcherOutput(**final_output_data_for_llm)
    json_to_output_llm = output_model_instance.model_dump_json(exclude_none=True)
    
    return f"Your final response for this turn MUST be exactly: '{json_to_output_llm}'"


# after_tool_callback is no longer needed for AgentTool(browser_agent)
# as it's now called directly within the instruction_provider.
def github_issue_fetcher_after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext, 
    tool_response: Any 
) -> dict | None: 
    logger.warning(f"FetcherAgent (after_tool_callback): Unexpected tool call intercepted: {tool.name}. This agent should not be calling ADK tools via LLM.")
    return None


github_issue_fetcher_agent = ADKAgent(
    name="github_issue_fetcher_agent",
    description="Receives a GitHub issue number for 'google/adk-python', fetches its content using a browser utility, cleans it, and returns a JSON object conforming to GitHubIssueFetcherOutput.",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    input_schema=GitHubIssueFetcherToolInput,
    output_schema=GitHubIssueFetcherOutput, 
    instruction=github_issue_fetcher_instruction_provider,
    tools=[], # IMPORTANT: tools list is now empty
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=None, # No ADK tools for its LLM to call
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=20000, 
        top_p=0.6
    )
)