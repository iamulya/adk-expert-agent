# expert-agents/github_issue_fetcher_agent.py
import re
import logging
import json
from pydantic import BaseModel, Field
from typing import Optional, Any

from google.adk.agents import Agent as ADKAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext 
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
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

# NEW: Output schema for GitHubIssueFetcherAgent
class GitHubIssueFetcherOutput(BaseModel):
    cleaned_issue_details: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None

def get_text_from_content(content: genai_types.Content) -> str:
    if content and content.parts and content.parts[0].text:
        return content.parts[0].text
    return ""

def github_issue_fetcher_instruction_provider(context: ReadonlyContext) -> str:
    invocation_ctx = context._invocation_context
    
    # Check if data is ready in state to be outputted to root_agent via this agent's output_schema
    processed_data_for_output_schema = None
    if invocation_ctx and invocation_ctx.session:
        processed_data_for_output_schema = invocation_ctx.session.state.get("temp:fetcher_processed_data_for_output")

    if processed_data_for_output_schema:
        logger.info(f"FetcherAgent (instruction_provider): Found processed data in state. Instructing LLM to format as per output_schema: {str(processed_data_for_output_schema)[:100]}...")
        
        output_model_instance = GitHubIssueFetcherOutput(
            cleaned_issue_details=processed_data_for_output_schema.get("cleaned_issue_details"),
            error=processed_data_for_output_schema.get("error"),
            message=processed_data_for_output_schema.get("message")
        )
        json_to_output_llm = output_model_instance.model_dump_json(exclude_none=True)
        
        if invocation_ctx and invocation_ctx.session: # Consume state
            invocation_ctx.session.state.pop("temp:fetcher_processed_data_for_output", None)
            
        return f"Your final response for this turn MUST be exactly: '{json_to_output_llm}'"
    else:
        # Initial call from root_agent: instruct LLM to call browser_agent
        user_input_json_str = get_text_from_content(invocation_ctx.user_content) if invocation_ctx and invocation_ctx.user_content else ""
        issue_number_str = None
        if user_input_json_str:
            try:
                input_data = GitHubIssueFetcherToolInput.model_validate_json(user_input_json_str)
                issue_number_str = input_data.issue_number
            except Exception as e:
                logger.error(f"FetcherAgent (instruction_provider - initial): Error parsing input '{user_input_json_str}': {e}")
                error_output = GitHubIssueFetcherOutput(error="Fetcher agent received invalid input for issue_number.")
                return f"Your final response for this turn MUST be exactly: '{error_output.model_dump_json(exclude_none=True)}'"

        if not issue_number_str:
            logger.error("FetcherAgent (instruction_provider - initial): No issue_number provided.")
            error_output = GitHubIssueFetcherOutput(error="Fetcher agent not provided with issue_number.")
            return f"Your final response for this turn MUST be exactly: '{error_output.model_dump_json(exclude_none=True)}'"

        github_url = f"https://github.com/google/adk-python/issues/{issue_number_str}"
        browser_agent_input_for_tool = BrowserAgentInput(url=github_url)
        browser_agent_args_json_str = browser_agent_input_for_tool.model_dump_json()
        
        logger.info(f"FetcherAgent (instruction_provider - initial): Instructing LLM to call '{browser_agent.name}' with args: {browser_agent_args_json_str}")
        return f"Your ONLY task is to call the '{browser_agent.name}' tool with the following JSON arguments: '{browser_agent_args_json_str}'. Do not add any other text or commentary. A callback will process the tool's output."


def github_issue_fetcher_after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext, 
    tool_response: Any  # Expected: dict (parsed BrowserAgentOutput)
) -> dict | None: 
    if tool.name == browser_agent.name:
        logger.info(f"FetcherAgent (after_tool_callback for {tool.name}): Received tool_response from browser_agent. Type: {type(tool_response)}, Value: '{str(tool_response)[:150]}'.")

        processed_data_for_output = {}

        if isinstance(tool_response, dict):
            try:
                # browser_agent has output_schema, so tool_response should be a dict matching BrowserAgentOutput
                browser_data = BrowserAgentOutput.model_validate(tool_response) 
                
                if browser_data.extracted_details:
                    cleaned_details = clean_github_issue_text(browser_data.extracted_details)
                    if not cleaned_details:
                        processed_data_for_output = {"message": "GitHub issue content empty/template after cleaning."}
                    else:
                        processed_data_for_output = {"cleaned_issue_details": cleaned_details}
                elif browser_data.error:
                    processed_data_for_output = {"error": f"Browser agent reported: {browser_data.error}"}
                elif browser_data.message:
                    processed_data_for_output = {"message": f"Browser agent message: {browser_data.message}"}
                else: 
                    processed_data_for_output = {"message": "Browser agent returned no details, error, or message."}
            except Exception as e: 
                logger.warning(f"FetcherAgent: tool_response dict from browser_agent couldn't be validated/processed: {str(tool_response)[:150]}. Error: {e}")
                processed_data_for_output = {"error": "Browser agent returned data in an unexpected dictionary format."}
        
        elif isinstance(tool_response, str): # Should not happen if browser_agent output_schema works
            logger.warning(f"FetcherAgent: tool_response from browser_agent was a string (expected dict): {str(tool_response)[:150]}.")
            processed_data_for_output = {"error": f"Browser agent returned unparsable text: {tool_response[:100]}..."}
        
        else: 
            logger.error(f"FetcherAgent: tool_response from browser_agent was neither dict nor string. Type: {type(tool_response)}, Value: {str(tool_response)[:150]}")
            processed_data_for_output = {"error": "Received unexpected data type from browser sub-process."}
        
        # Store the processed data that this agent's LLM needs to format for its own output_schema
        tool_context.state["temp:fetcher_processed_data_for_output"] = processed_data_for_output
        logger.info(f"FetcherAgent (after_tool_callback): Stored processed data for own output in state 'temp:fetcher_processed_data_for_output': {str(processed_data_for_output)[:100]}...")
        
        tool_context.actions.skip_summarization = True 
        return {"status": "fetcher_processed_browser_data_ready_for_own_output_schema"} 
    
    logger.warning(f"FetcherAgent (after_tool_callback): Callback for unexpected tool: {tool.name}")
    return None


github_issue_fetcher_agent = ADKAgent(
    name="github_issue_fetcher_agent",
    description="Receives a GitHub issue number for 'google/adk-python', fetches its content using a browser utility, cleans it, and returns a JSON object conforming to GitHubIssueFetcherOutput.",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    input_schema=GitHubIssueFetcherToolInput,
    output_schema=GitHubIssueFetcherOutput, # NEW output schema for this agent
    instruction=github_issue_fetcher_instruction_provider,
    tools=[
        AgentTool(agent=browser_agent) 
    ],
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=github_issue_fetcher_after_tool_callback, 
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=20000, # For outputting its own JSON (GitHubIssueFetcherOutput)
        top_p=0.6
    )
)