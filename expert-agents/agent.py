"""
This module defines the root agent for the ADK Expert application.

The `adk_expert_orchestrator` (also named `root_agent`) acts as the main entry
point and orchestrator. It analyzes incoming user queries and delegates tasks
to specialized sub-agents for GitHub issue processing, document generation, or
diagram creation. It can also handle general ADK-related questions directly.
"""

import logging
import re
import json
from dotenv import load_dotenv
from typing import Any, Optional

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from google.genai import types as genai_types
from pydantic import ValidationError

from .context_loader import get_escaped_adk_context_for_llm
from .config import DEFAULT_MODEL_NAME
from .callbacks import log_prompt_before_model_call

# Import agents from the 'agents' subpackage that this root agent will orchestrate.
from .agents.github_issue_processing_agent import (
    github_issue_processing_agent,
    GitHubIssueProcessingInput,
    SequentialProcessorFinalOutput,
)
from .agents.document_generator_agent import (
    document_generator_agent,
    DocumentGeneratorAgentToolInput,
)
from .agents.mermaid_diagram_orchestrator_agent import (
    mermaid_diagram_orchestrator_agent,
    DiagramGeneratorAgentToolInput,
)  # Updated name

# Import tools from the 'tools' subpackage used directly by this root agent.
from .tools.prepare_document_content_tool import (
    PrepareDocumentContentTool,
    PrepareDocumentContentToolInput,
)

load_dotenv()
logger = logging.getLogger(__name__)


def get_text_from_content(content: genai_types.Content) -> str:
    """
    Safely extracts the text from a genai_types.Content object.

    Args:
        content: The Content object, which may or may not contain text.

    Returns:
        The extracted text as a string, or an empty string if not found.
    """
    if content and content.parts and content.parts[0].text:
        return content.parts[0].text
    return ""


class PreparedContentDataForDocGen(PrepareDocumentContentToolInput):
    """
    A Pydantic model used for validating the data received from the
    `prepare_document_content_tool` before passing it to the document generator.
    It inherits from the tool's input model to ensure structure consistency.
    """

    pass


async def root_agent_after_tool_callback(
    tool: BaseTool, args: dict, tool_context: ToolContext, tool_response: Any
) -> Optional[Any]:
    """
    Callback executed after any tool call made by the root agent.

    This function processes the output from the specialized sub-agents,
    formats their responses, and relays the final answer directly to the user
    by returning a Content object and setting `skip_summarization`. For the
    `prepare_document_content_tool`, it allows the response to be processed
    by the LLM in the next turn.

    Args:
        tool: The tool that was just executed.
        args: The arguments passed to the tool.
        tool_context: The context of the tool call.
        tool_response: The raw response from the tool.

    Returns:
        A `genai_types.Content` object to be sent as the agent's final response,
        or the original `tool_response` to be processed further by the LLM, or None.
    """

    # When a sub-agent (like the GitHub processor) finishes, its final output is the tool_response.
    if tool.name == github_issue_processing_agent.name:
        logger.info(
            f"RootAgent (after_tool_callback): Processing response from '{github_issue_processing_agent.name}'."
        )
        # We want to relay the agent's final output directly, without LLM summarization.
        tool_context.actions.skip_summarization = True
        response_text = "Error: Could not process response from sequential agent."
        try:
            # The sequential agent's output is a JSON string which needs to be parsed.
            if isinstance(tool_response, str):
                response_dict = json.loads(tool_response)
            elif isinstance(tool_response, dict):
                response_dict = tool_response
            else:
                raise ValueError(
                    f"Unexpected tool_response type: {type(tool_response)}"
                )

            validated_output = SequentialProcessorFinalOutput.model_validate(
                response_dict
            )
            response_text = validated_output.guidance
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            logger.error(
                f"RootAgent: Error parsing/validating response from sequential agent: {e}. Response: {str(tool_response)[:200]}",
                exc_info=True,
            )
            if isinstance(tool_response, str) and (
                "error" in tool_response.lower()
                or not tool_response.strip().startswith("{")
            ):
                response_text = tool_response
            else:
                response_text = f"Error: Sequential agent returned an unexpected structure: {str(tool_response)[:200]}"
        logger.info(
            f"RootAgent (after_tool_callback): Relaying guidance from sequential agent: {response_text[:200]}..."
        )
        # Return a Content object to make this the final response for the turn.
        return genai_types.Content(parts=[genai_types.Part(text=response_text)])

    # Handle the response from the document generation sub-agent.
    elif tool.name == document_generator_agent.name:
        logger.info(
            f"RootAgent (after_tool_callback): Received response from '{document_generator_agent.name}': {str(tool_response)[:200]}"
        )
        if isinstance(tool_response, str) and tool_response.strip():
            tool_context.actions.skip_summarization = True
            return genai_types.Content(parts=[genai_types.Part(text=tool_response)])
        else:
            error_msg = f"Error: Document generation agent returned an unexpected or empty response: {str(tool_response)[:100]}"
            logger.error(f"RootAgent: {error_msg}")
            tool_context.actions.skip_summarization = True
            return genai_types.Content(parts=[genai_types.Part(text=error_msg)])

    # Handle the response from the Mermaid diagram orchestration sub-agent.
    elif tool.name == mermaid_diagram_orchestrator_agent.name:  # Updated name
        logger.info(
            f"RootAgent (after_tool_callback): Received response from '{mermaid_diagram_orchestrator_agent.name}': {str(tool_response)[:200]}"
        )
        # The orchestrator agent's after_agent_cb now ensures the output is a string.
        if isinstance(tool_response, str) and tool_response.strip():
            tool_context.actions.skip_summarization = True
            return genai_types.Content(parts=[genai_types.Part(text=tool_response)])
        elif (
            isinstance(tool_response, dict)
            and "result" in tool_response
            and isinstance(tool_response["result"], str)
        ):  # ADK might wrap it
            tool_context.actions.skip_summarization = True
            return genai_types.Content(
                parts=[genai_types.Part(text=tool_response["result"])]
            )
        else:
            error_msg = f"Error: Diagram orchestrator agent returned an unexpected or empty response: {str(tool_response)[:100]}"
            logger.error(f"RootAgent: {error_msg}")
            tool_context.actions.skip_summarization = True
            return genai_types.Content(parts=[genai_types.Part(text=error_msg)])

    # This tool is the first step in a two-step document generation process.
    # Its output should be processed by the LLM in the next turn.
    elif tool.name == "prepare_document_content_tool":
        logger.info(
            f"RootAgent (after_tool_callback): 'prepare_document_content_tool' completed. Output from tool: {str(tool_response)[:200]}"
        )
        # Do not skip summarization, let the LLM see this tool's output.
        tool_context.actions.skip_summarization = False
        return tool_response

    logger.warning(
        f"RootAgent (after_tool_callback): Callback for unhandled tool: {tool.name}"
    )
    return None


def root_agent_instruction_provider(context: ReadonlyContext) -> str:
    """
    Dynamically generates the system instruction for the root agent.

    This is the core routing logic of the application. It analyzes the user's query
    and the conversation history to decide which tool or sub-agent to call, or
    whether to answer the query directly.

    Args:
        context: The read-only context of the agent invocation.

    Returns:
        A string containing the system instruction for the LLM.
    """
    adk_context_for_llm = get_escaped_adk_context_for_llm()
    invocation_ctx = getattr(context, "_invocation_context", None)
    user_query_text = (
        get_text_from_content(invocation_ctx.user_content)
        if invocation_ctx and invocation_ctx.user_content
        else ""
    )

    # Check if the last event was the output of `prepare_document_content_tool`.
    # This is the second step of the two-step document generation flow.
    if invocation_ctx and invocation_ctx.session and invocation_ctx.session.events:
        last_event = invocation_ctx.session.events[-1]
        if (
            last_event.author == root_agent.name
            and last_event.content
            and last_event.content.parts
            and last_event.content.parts[0].function_response
            and last_event.content.parts[0].function_response.name
            == "prepare_document_content_tool"
        ):
            logger.info(
                "RootAgent (instruction_provider): Detected response from prepare_document_content_tool. Instructing to call document_generator_agent."
            )
            tool_output_data_container = last_event.content.parts[
                0
            ].function_response.response

            # The tool output might be nested inside a 'result' key.
            prepared_content_data = tool_output_data_container.get(
                "result", tool_output_data_container
            )

            try:
                # Validate the data and create the exact input for the document generator agent.
                validated_content_for_doc_gen = (
                    PreparedContentDataForDocGen.model_validate(prepared_content_data)
                )
                doc_gen_agent_actual_input = DocumentGeneratorAgentToolInput(
                    markdown_content=validated_content_for_doc_gen.markdown_content,
                    document_type=validated_content_for_doc_gen.document_type,
                    output_filename=validated_content_for_doc_gen.output_filename_base,
                )
                # This highly specific instruction forces the LLM to call the next agent in the chain.
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
                logger.error(
                    f"RootAgent (instruction_provider): Error processing data from prepare_document_content_tool for doc gen: {e}. Data: {str(prepared_content_data)[:200]}",
                    exc_info=True,
                )
                return "Error: Could not process the data from the content preparation step. Please try rephrasing your request."

    # Define regex patterns to detect GitHub issue numbers in the user query.
    patterns = [
        re.compile(
            r"(?:issue|ticket|bug|fix|feature|problem|error)\s*(?:number|num|report)?\s*(?:#)?\s*(\d+)(?:\s*(?:on|for|in|related to)\s*google/adk-python)?",
            re.IGNORECASE,
        ),
        re.compile(
            r"google/adk-python\s*(?:issue|ticket|bug|fix|feature|problem|error)\s*(?:number|num|report)?\s*(?:#)?\s*(\d+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(\d+)\s*(?:on|for|in|related to)\s*google/adk-python", re.IGNORECASE
        ),
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

    is_github_keywords_present = "github" in user_query_text.lower() or any(
        kw in user_query_text.lower() for kw in ["issue", "bug", "ticket", "feature"]
    )

    # Define keywords to detect document generation requests.
    doc_gen_keywords_pdf = ["pdf", "document", "report"]
    doc_gen_keywords_slides = [
        "slides",
        "presentation",
        "deck",
        "pptx",
        "powerpoint",
        "html slides",
    ]
    requested_doc_type = None
    if any(kw in user_query_text.lower() for kw in doc_gen_keywords_pdf):
        requested_doc_type = "pdf"
    elif any(kw in user_query_text.lower() for kw in doc_gen_keywords_slides):
        requested_doc_type = (
            "pptx"
            if "pptx" in user_query_text.lower()
            or "powerpoint" in user_query_text.lower()
            else "html"
        )

    # Define keywords to detect diagram generation requests.
    diagram_keywords = ["diagram", "architecture", "visualize", "mermaid", "graph"]
    is_diagram_request = any(kw in user_query_text.lower() for kw in diagram_keywords)

    # --- Instruction Generation based on Query Analysis ---

    if is_diagram_request:
        logger.info(
            f"RootAgent (instruction_provider): Detected architecture diagram request: '{user_query_text}'"
        )
        diagram_agent_input_payload = DiagramGeneratorAgentToolInput(
            diagram_query=user_query_text
        ).model_dump_json()
        system_instruction = f"""
You are an expert orchestrator for Google's Agent Development Kit (ADK).
The user is asking for an architecture diagram. Their query is: "{user_query_text}"
Your task is to call the '{mermaid_diagram_orchestrator_agent.name}' tool.
The tool expects its input as a JSON string. The value for the "request" argument MUST be the following JSON string:
{diagram_agent_input_payload}
This is your only action for this turn. Output only the tool call.
"""
    elif requested_doc_type:
        # This is the first step of document generation: instruct the LLM to generate content
        # and call the `PrepareDocumentContentTool`.
        logger.info(
            f"RootAgent (instruction_provider): Detected document generation request for type '{requested_doc_type}'. Query: '{user_query_text}'"
        )
        system_instruction = f"""
You are an expert on Google's Agent Development Kit (ADK) version 1.2.0 and a document content creator.
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
        logger.info(
            f"RootAgent (instruction_provider): Found issue number '{extracted_issue_number}'. Instructing to call GitHubIssueProcessingSequentialAgent."
        )
        sequential_agent_input_payload_str = GitHubIssueProcessingInput(
            issue_number=extracted_issue_number
        ).model_dump_json()
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
        # If the user mentions GitHub/issues but doesn't provide a number, ask for it.
        logger.info(
            "RootAgent (instruction_provider): GitHub keywords present, but no issue number. Asking."
        )
        system_instruction = "Your final response for this turn MUST be exactly: 'It looks like you're asking about a GitHub issue for google/adk-python, but I couldn't find a specific issue number. Please provide the GitHub issue number.'"
    else:
        # Default case: Handle a general ADK question by answering directly.
        logger.info(
            f"RootAgent (instruction_provider): General ADK query: '{user_query_text}'"
        )
        system_instruction = f"""
You are an expert on Google's Agent Development Kit (ADK) version 1.2.0.
Your primary role is to answer general questions about ADK.
When a user starts a conversation, greet them by introducing yourself as an ADK 1.2.0 expert.
Use your ADK knowledge (from the context below) to answer the user's query directly. This is your final answer.

ADK Knowledge Context (for general ADK questions):
--- START OF ADK CONTEXT ---
{adk_context_for_llm}
--- END OF ADK CONTEXT ---
"""
    return system_instruction


# The set of tools available to the root agent. These are either sub-agents (wrapped in AgentTool) or preparatory tools.
root_agent_tools = [
    AgentTool(agent=github_issue_processing_agent),
    AgentTool(agent=document_generator_agent),
    AgentTool(agent=mermaid_diagram_orchestrator_agent),  # Updated name
    PrepareDocumentContentTool(),
]

# Definition of the root agent.
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
    ),
)
