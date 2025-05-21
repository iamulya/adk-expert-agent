# expert-agents/agent.py
import os
import logging
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, Literal, Optional, override

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from google.genai import types as genai_types
from pydantic import BaseModel, Field, ValidationError


from .context_loader import get_escaped_adk_context_for_llm
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call
from .sequential_issue_processor import github_issue_processing_agent, GitHubIssueProcessingInput, SequentialProcessorFinalOutput
from .document_generator_agent import document_generator_agent, DocumentGeneratorAgentToolInput

load_dotenv()
logger = logging.getLogger(__name__)

def get_text_from_content(content: genai_types.Content) -> str:
    if content and content.parts and content.parts[0].text:
        return content.parts[0].text
    return ""

# Pydantic model for the input to the 'prepare_document_content_tool'
# This is what the root_agent's LLM will be instructed to provide as arguments
class PrepareDocumentContentToolInput(BaseModel):
    markdown_content: str = Field(description="The generated Markdown content for the document.")
    document_type: Literal["pdf", "html", "pptx"] = Field(description="The type of document requested by the user (pdf, html, or pptx).")
    output_filename_base: str = Field(description="A base name for the output file, e.g., 'adk_report'. The document_generator_agent will append the correct extension.")
    original_user_query: str = Field(description="The original user query that requested the document.")


async def root_agent_after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: Any
) -> Optional[genai_types.Content]:

    if tool.name == github_issue_processing_agent.name:
        logger.info(f"RootAgent (after_tool_callback): Processing response from '{github_issue_processing_agent.name}'.")
        response_text = "Error: Could not process response from sequential agent."
        try:
            if isinstance(tool_response, str):
                 response_dict = json.loads(tool_response)
            elif isinstance(tool_response, dict):
                 response_dict = tool_response
            else:
                raise ValueError(f"Unexpected tool_response type: {type(tool_response)}")

            validated_output = SequentialProcessorFinalOutput.model_validate(response_dict)
            response_text = validated_output.guidance
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            logger.error(f"RootAgent: Error parsing/validating response from sequential agent: {e}. Response: {str(tool_response)[:200]}", exc_info=True)
            if isinstance(tool_response, str) and ("error" in tool_response.lower() or not tool_response.strip().startswith("{")):
                response_text = tool_response
            else:
                 response_text = f"Error: Sequential agent returned an unexpected structure: {str(tool_response)[:200]}"
        logger.info(f"RootAgent (after_tool_callback): Relaying guidance from sequential agent: {response_text[:200]}...")
        return genai_types.Content(parts=[genai_types.Part(text=response_text)])

    elif tool.name == document_generator_agent.name:
        logger.info(f"RootAgent (after_tool_callback): Received response from '{document_generator_agent.name}': {str(tool_response)[:200]}")
        return genai_types.Content(parts=[genai_types.Part(text=str(tool_response))])

    elif tool.name == "prepare_document_content_tool":
        # 'tool_response' here IS the dictionary of arguments that the LLM provided
        # when it CALLED 'prepare_document_content_tool'.
        logger.info(f"RootAgent (after_tool_callback): 'prepare_document_content_tool' was called by LLM. Output from tool: {str(tool_response)[:200]}")
        try:
            # Validate that the LLM provided the correct arguments to our internal tool
            validated_prepared_content = PrepareDocumentContentToolInput.model_validate(tool_response)

            # Prepare the input for the actual document_generator_agent (AgentTool)
            doc_gen_agent_actual_input = DocumentGeneratorAgentToolInput(
                markdown_content=validated_prepared_content.markdown_content,
                document_type=validated_prepared_content.document_type,
                output_filename=validated_prepared_content.output_filename_base # Specialist agent expects 'output_filename'
            )

            function_call_to_doc_gen_agent = genai_types.FunctionCall(
                name=document_generator_agent.name,
                args=doc_gen_agent_actual_input.model_dump()
            )
            logger.info(f"RootAgent (after_tool_callback): Preparing to call '{document_generator_agent.name}' (AgentTool) with args: {doc_gen_agent_actual_input.model_dump_json()[:200]}")
            
            tool_context.actions.skip_summarization = True
            return genai_types.Content(parts=[genai_types.Part(function_call=function_call_to_doc_gen_agent)])

        except (ValidationError, ValueError) as e:
            logger.error(f"RootAgent: Error validating/processing LLM's arguments from 'prepare_document_content_tool': {e}. LLM output (tool_response): {str(tool_response)}", exc_info=True)
            return genai_types.Content(parts=[genai_types.Part(text=f"Error: Could not process the prepared document content: {e}")])

    logger.warning(f"RootAgent (after_tool_callback): Callback for unhandled tool: {tool.name}")
    return None


def root_agent_instruction_provider(context: ReadonlyContext) -> str:
    adk_context_for_llm = get_escaped_adk_context_for_llm()
    invocation_ctx = context._invocation_context if hasattr(context, '_invocation_context') else None
    user_query_text = get_text_from_content(invocation_ctx.user_content) if invocation_ctx and invocation_ctx.user_content else ""

    patterns = [
        re.compile(r"(?:issue|ticket|bug|fix|feature|problem|error)\s*(?:number|num|report)?\s*(?:#)?\s*(\d+)(?:\s*(?:on|for|in|related to)\s*google/adk-python)?", re.IGNORECASE),
        re.compile(r"google/adk-python\s*(?:issue|ticket|bug|fix|feature|problem|error)\s*(?:number|num|report)?\s*(?:#)?\s*(\d+)", re.IGNORECASE),
        re.compile(r"(\d+)\s*(?:on|for|in|related to)\s*google/adk-python", re.IGNORECASE)
    ]
    extracted_issue_number = None
    for pattern in patterns:
        match = pattern.search(user_query_text)
        if match:
            for group_val in match.groups():
                if group_val and group_val.isdigit():
                    extracted_issue_number = group_val
                    break
        if extracted_issue_number:
            break
    
    is_github_keywords_present = "github" in user_query_text.lower() or \
                                 any(kw in user_query_text.lower() for kw in ["issue", "bug", "ticket", "feature"])

    doc_gen_keywords_pdf = ["pdf", "document", "report"]
    doc_gen_keywords_slides = ["slides", "presentation", "deck", "pptx", "powerpoint", "html slides"]
    
    requested_doc_type = None
    if any(kw in user_query_text.lower() for kw in doc_gen_keywords_pdf):
        requested_doc_type = "pdf"
    elif any(kw in user_query_text.lower() for kw in doc_gen_keywords_slides):
        if "pptx" in user_query_text.lower() or "powerpoint" in user_query_text.lower():
            requested_doc_type = "pptx"
        else:
            requested_doc_type = "html"

    if requested_doc_type:
        logger.info(f"RootAgent (instruction_provider): Detected document generation request for type '{requested_doc_type}'. Query: '{user_query_text}'")
        
        # Instruct the LLM to call 'prepare_document_content_tool'
        system_instruction = f"""
You are an expert on Google's Agent Development Kit (ADK) version 1.0.0 and a document content creator.
You have access to a tool called 'prepare_document_content_tool'.
This tool is used to gather the necessary Markdown content, the type of document, and a base filename before actual document generation.
You also have extensive ADK knowledge from the context below.

ADK Knowledge Context:
--- START OF ADK CONTEXT ---
{adk_context_for_llm}
--- END OF ADK CONTEXT ---

The user wants you to generate a document of type '{requested_doc_type}'.
Their request is: "{user_query_text}"

Your tasks are:
1.  Analyze the user's request: "{user_query_text}".
2.  Using the ADK Knowledge Context, generate comprehensive and well-structured Markdown content suitable for `marp-cli` that directly addresses the user's request.
3.  Determine a suitable base filename for this document (e.g., if the user asks for "a report on ADK tools", a good base filename could be "adk_tools_report"). This filename should NOT include an extension.
4.  You MUST call the tool named 'prepare_document_content_tool'.
    The arguments for this tool MUST be a JSON object with the following keys:
    -   "markdown_content": (string) Your generated Markdown text.
    -   "document_type": (string) The value MUST be "{requested_doc_type}".
    -   "output_filename_base": (string) Your determined base filename.
    -   "original_user_query": (string) The exact text of the user's request: "{user_query_text}".

Your response should ONLY be the function call to 'prepare_document_content_tool'. Do not include any other text, greetings, or explanations.
"""
    elif extracted_issue_number:
        logger.info(f"RootAgent (instruction_provider): Found issue number '{extracted_issue_number}'. Instructing to call GitHubIssueProcessingSequentialAgent.")
        sequential_agent_input_payload_str = GitHubIssueProcessingInput(issue_number=extracted_issue_number).model_dump_json()
        system_instruction = f"""
You are an expert orchestrator for Google's Agent Development Kit (ADK).
The user is asking about GitHub issue number {extracted_issue_number} for 'google/adk-python'.
Your task is to call the '{github_issue_processing_agent.name}' tool.
The tool expects a single argument named "request".
The value for the "request" argument MUST be the following JSON string:
{sequential_agent_input_payload_str}
Ensure the value you provide for the "request" key is precisely this JSON string.
This is your only action for this turn. Output only the tool call.
"""
    elif is_github_keywords_present:
        logger.info("RootAgent (instruction_provider): GitHub keywords present, but no issue number. Asking.")
        system_instruction = "Your final response for this turn MUST be exactly: 'It looks like you're asking about a GitHub issue for google/adk-python, but I couldn't find a specific issue number. Please provide the GitHub issue number.'"
    else: 
        logger.info(f"RootAgent (instruction_provider): General ADK query: '{user_query_text}'")
        system_instruction = f"""
You are an expert on Google's Agent Development Kit (ADK) version 1.0.0.
Your primary role is to answer general questions about ADK.
When a user starts a conversation, greet them by introducing yourself as an ADK 1.0.0 expert.
Use your ADK knowledge (from the context below) to answer the user's query directly. This is your final answer.

ADK Knowledge Context (for general ADK questions):
--- START OF ADK CONTEXT ---
{adk_context_for_llm}
--- END OF ADK CONTEXT ---
"""
    return system_instruction

# This tool is called BY THE ROOT AGENT'S LLM.
# Its `run_async` method returns the structured data (markdown, doc_type, etc.)
# which is then picked up by `root_agent_after_tool_callback` to make the *actual*
# call to the `document_generator_agent` (which is an AgentTool).
class PrepareDocumentContentTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="prepare_document_content_tool", # Renamed for clarity
            description="Gathers generated markdown content, document type, and filename base. This tool is called by the orchestrator agent; its output triggers the actual document generation agent."
        )

    @override
    def _get_declaration(self):
        # The schema for the arguments this tool expects from the LLM
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=PrepareDocumentContentToolInput.model_json_schema()
        )

    async def run_async(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
        logger.info(f"PrepareDocumentContentTool 'run_async' called with args: {str(args)[:200]}. Returning these args directly.")
        try:
            validated_args = PrepareDocumentContentToolInput.model_validate(args)
            return validated_args.model_dump() 
        except ValidationError as ve:
            logger.error(f"PrepareDocumentContentTool: LLM provided invalid arguments: {ve}. Args: {args}", exc_info=True)
            return {"error": f"Invalid arguments from LLM for content preparation: {str(ve)}", "original_args": args}


root_agent_tools = [
    AgentTool(agent=github_issue_processing_agent),
    AgentTool(agent=document_generator_agent), 
    PrepareDocumentContentTool(), # Changed from GenerateMarkdownForDocumentTool
]

root_agent = ADKAgent(
    name="adk_expert_orchestrator",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=root_agent_instruction_provider,
    tools=root_agent_tools,
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=root_agent_after_tool_callback,
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=60000, 
        top_p=0.6,
    )
)