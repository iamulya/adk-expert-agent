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
from .config import DEFAULT_MODEL_NAME, PRO_MODEL_NAME
from .callbacks import log_prompt_before_model_call

# Import agents from the new 'agents' subpackage
from .agents.github_issue_processing_agent import github_issue_processing_agent, GitHubIssueProcessingInput, SequentialProcessorFinalOutput
from .agents.document_generator_agent import document_generator_agent, DocumentGeneratorAgentToolInput
from .agents.diagram_generator_agent import diagram_generator_agent, DiagramGeneratorAgentToolInput

# Import tools from the new 'tools' subpackage
from .tools.prepare_document_content_tool import PrepareDocumentContentTool, PrepareDocumentContentToolInput
# Note: mermaid_gcs_tool_instance and GCS_LINK_STATE_KEY are used by diagram_generator_agent,
# which is now in its own file and imports them directly. So, no need to import them here for root_agent.

load_dotenv()
logger = logging.getLogger(__name__)

def get_text_from_content(content: genai_types.Content) -> str:
    if content and content.parts and content.parts[0].text:
        return content.parts[0].text
    return ""

# Pydantic model for PrepareDocumentContentTool's output when used by root_agent's LLM
# This is the same as PrepareDocumentContentToolInput, but named for clarity in root_agent context
class PreparedContentDataForDocGen(PrepareDocumentContentToolInput):
    pass

async def root_agent_after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: Any
) -> Optional[Any]:

    if tool.name == github_issue_processing_agent.name:
        logger.info(f"RootAgent (after_tool_callback): Processing response from '{github_issue_processing_agent.name}'.")
        tool_context.actions.skip_summarization = True
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
        if isinstance(tool_response, str) and tool_response.strip():
            tool_context.actions.skip_summarization = True
            return genai_types.Content(parts=[genai_types.Part(text=tool_response)])
        else:
            error_msg = f"Error: Document generation agent returned an unexpected or empty response: {str(tool_response)[:100]}"
            logger.error(f"RootAgent: {error_msg}")
            tool_context.actions.skip_summarization = True
            return genai_types.Content(parts=[genai_types.Part(text=error_msg)])

    elif tool.name == diagram_generator_agent.name:
        logger.info(f"RootAgent (after_tool_callback): Received response from '{diagram_generator_agent.name}': {str(tool_response)[:200]}")
        if isinstance(tool_response, str) and tool_response.strip():
            tool_context.actions.skip_summarization = True
            return genai_types.Content(parts=[genai_types.Part(text=tool_response)])
        else:
            error_msg = f"Error: Diagram generation agent returned an unexpected or empty response: {str(tool_response)[:100]}"
            logger.error(f"RootAgent: {error_msg}")
            tool_context.actions.skip_summarization = True
            return genai_types.Content(parts=[genai_types.Part(text=error_msg)])

    elif tool.name == "prepare_document_content_tool":
        logger.info(f"RootAgent (after_tool_callback): 'prepare_document_content_tool' completed. Output from tool: {str(tool_response)[:200]}")
        tool_context.actions.skip_summarization = False
        return tool_response

    logger.warning(f"RootAgent (after_tool_callback): Callback for unhandled tool: {tool.name}")
    return None


def root_agent_instruction_provider(context: ReadonlyContext) -> str:
    adk_context_for_llm = get_escaped_adk_context_for_llm()
    invocation_ctx = getattr(context, '_invocation_context', None)
    user_query_text = get_text_from_content(invocation_ctx.user_content) if invocation_ctx and invocation_ctx.user_content else ""
    
    if invocation_ctx and invocation_ctx.session and invocation_ctx.session.events:
        last_event = invocation_ctx.session.events[-1]
        if last_event.author == root_agent.name and            last_event.content and last_event.content.parts and            last_event.content.parts[0].function_response and            last_event.content.parts[0].function_response.name == "prepare_document_content_tool":
            
            logger.info("RootAgent (instruction_provider): Detected response from prepare_document_content_tool. Instructing to call document_generator_agent.")
            tool_output_data_container = last_event.content.parts[0].function_response.response
            
            prepared_content_data = tool_output_data_container.get("result", tool_output_data_container)

            try:
                # Validate against the Pydantic model for what prepare_document_content_tool returns
                validated_content_for_doc_gen = PreparedContentDataForDocGen.model_validate(prepared_content_data)
                # Map to DocumentGeneratorAgentToolInput
                doc_gen_agent_actual_input = DocumentGeneratorAgentToolInput(
                    markdown_content=validated_content_for_doc_gen.markdown_content,
                    document_type=validated_content_for_doc_gen.document_type,
                    output_filename=validated_content_for_doc_gen.output_filename_base # document_generator_agent expects 'output_filename'
                )
                system_instruction = f"""
You have received structured data from the 'prepare_document_content_tool'.
The data is: {json.dumps(prepared_content_data)}
Your task is to call the tool named '{document_generator_agent.name}'.
This tool expects arguments conforming to this schema:
{DocumentGeneratorAgentToolInput.model_json_schema()}
Based on the data received from 'prepare_document_content_tool', you MUST call the '{document_generator_agent.name}' tool with the correctly mapped arguments:
{doc_gen_agent_actual_input.model_dump_json()}
Your response should ONLY be the function call. Do not include any other text.
"""
                return system_instruction
            except (ValidationError, Exception) as e:
                logger.error(f"RootAgent (instruction_provider): Error processing data from prepare_document_content_tool for doc gen: {e}. Data: {str(prepared_content_data)[:200]}", exc_info=True)
                return "Error: Could not process the data from the content preparation step. Please try rephrasing your request."

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
    
    is_github_keywords_present = "github" in user_query_text.lower() or                                  any(kw in user_query_text.lower() for kw in ["issue", "bug", "ticket", "feature"])

    doc_gen_keywords_pdf = ["pdf", "document", "report"]
    doc_gen_keywords_slides = ["slides", "presentation", "deck", "pptx", "powerpoint", "html slides"]
    requested_doc_type = None
    if any(kw in user_query_text.lower() for kw in doc_gen_keywords_pdf):
        requested_doc_type = "pdf"
    elif any(kw in user_query_text.lower() for kw in doc_gen_keywords_slides):
        requested_doc_type = "pptx" if "pptx" in user_query_text.lower() or "powerpoint" in user_query_text.lower() else "html"

    diagram_keywords = ["diagram", "architecture", "visualize", "mermaid"]
    is_diagram_request = any(kw in user_query_text.lower() for kw in diagram_keywords)

    if is_diagram_request:
        logger.info(f"RootAgent (instruction_provider): Detected architecture diagram request: '{user_query_text}'")
        diagram_agent_input_payload = DiagramGeneratorAgentToolInput(diagram_query=user_query_text).model_dump_json()
        system_instruction = f"""
You are an expert orchestrator for Google's Agent Development Kit (ADK).
The user is asking for an architecture diagram. Their query is: "{user_query_text}"
Your task is to call the '{diagram_generator_agent.name}' tool.
The tool expects its input as a JSON string. The value for the "request" argument MUST be the following JSON string:
{diagram_agent_input_payload}
This is your only action for this turn. Output only the tool call.
"""
    elif requested_doc_type:
        logger.info(f"RootAgent (instruction_provider): Detected document generation request for type '{requested_doc_type}'. Query: '{user_query_text}'")
        system_instruction = f"""
You are an expert on Google's Agent Development Kit (ADK) version 1.0.0 and a document content creator.
You have access to a tool called '{PrepareDocumentContentTool().name}'.
ADK Knowledge Context:
--- START OF ADK CONTEXT ---
{adk_context_for_llm}
--- END OF ADK CONTEXT ---
The user wants you to generate a document of type '{requested_doc_type}'. Their request is: "{user_query_text}"
Your tasks are:
1.  Analyze the user's request: "{user_query_text}".
2.  Using the ADK Knowledge Context, generate comprehensive Markdown content for `marp-cli`.
3.  Determine a suitable base filename (e.g., "adk_overview").
4.  You MUST call '{PrepareDocumentContentTool().name}' with "markdown_content", "document_type" ("{requested_doc_type}"), "output_filename_base", and "original_user_query".
Your response should ONLY be the function call.
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
Output only the tool call.
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

root_agent_tools = [
    AgentTool(agent=github_issue_processing_agent),
    AgentTool(agent=document_generator_agent),
    AgentTool(agent=diagram_generator_agent),
    PrepareDocumentContentTool(), # This tool is defined in .tools.prepare_document_content_tool
]

root_agent = ADKAgent(
    name="adk_expert_orchestrator",
    model=Gemini(model=PRO_MODEL_NAME),
    instruction=root_agent_instruction_provider,
    tools=root_agent_tools,
    before_model_callback=log_prompt_before_model_call,
    after_tool_callback=root_agent_after_tool_callback,
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=60000, # Consider if this is still appropriate
        top_p=0.6,
    )
)
