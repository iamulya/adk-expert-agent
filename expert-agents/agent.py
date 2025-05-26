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
from google.adk.agents import Agent as ADKAgent # Renamed to avoid conflict
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from google.genai import types as genai_types
from pydantic import BaseModel, Field, ValidationError


from .context_loader import get_escaped_adk_context_for_llm
from .config import DEFAULT_MODEL_NAME, PRO_MODEL_NAME
from .callbacks import log_prompt_before_model_call
from .sequential_issue_processor import github_issue_processing_agent, GitHubIssueProcessingInput, SequentialProcessorFinalOutput
from .document_generator_agent import document_generator_agent, DocumentGeneratorAgentToolInput
from .mermaid_tool import GCS_LINK_STATE_KEY, mermaid_gcs_tool_instance # Import the new tool

load_dotenv()
logger = logging.getLogger(__name__)

def get_text_from_content(content: genai_types.Content) -> str:
    if content and content.parts and content.parts[0].text:
        return content.parts[0].text
    return ""

# --- Pydantic Models for Agent Tools ---
class DiagramGeneratorAgentToolInput(BaseModel):
    diagram_query: str = Field(description="The user's query describing the architecture diagram to be generated.")

class MermaidSyntaxVerifierAgentToolInput(BaseModel):
    mermaid_syntax: str = Field(description="The Mermaid syntax to verify and correct.")


# --- New Mermaid-related Agents ---
mermaid_syntax_verifier_agent = ADKAgent(
    name="mermaid_syntax_verifier_agent",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=(
        "You are an expert in Mermaid diagram syntax. "
        "Your sole responsibility is to receive Mermaid syntax, verify its correctness, "
        "and correct any syntax errors. "
        "If the provided syntax is already correct, return it as is. "
        "If there are errors, return the corrected Mermaid syntax. "
        "Your output MUST ONLY be the Mermaid code block itself (e.g., ```mermaid\\n...\\n```). "
        "Do not add any other explanations, greetings, or conversational text."
    ),
    description="Verifies and corrects Mermaid diagram syntax. Expects input as a JSON string with a 'mermaid_syntax' key.",
    input_schema=MermaidSyntaxVerifierAgentToolInput, # Schema for when this agent is called as a tool
    disallow_transfer_to_parent=True, # This agent is a specialist
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.0)
)

def diagram_generator_agent_instruction_provider(context: ReadonlyContext) -> str:
    invocation_ctx = getattr(context, '_invocation_context', None)
    user_diagram_query = ""
    if invocation_ctx and hasattr(invocation_ctx, 'user_content') and invocation_ctx.user_content:
        if invocation_ctx.user_content.parts and invocation_ctx.user_content.parts[0].text:
            try:
                # This agent expects its input (when called as a tool) to be a JSON string
                # conforming to DiagramGeneratorAgentToolInput.
                input_data = DiagramGeneratorAgentToolInput.model_validate_json(invocation_ctx.user_content.parts[0].text)
                user_diagram_query = input_data.diagram_query
            except (ValidationError, json.JSONDecodeError) as e:
                logger.error(f"DiagramGeneratorAgent: Error parsing input: {e}. Raw input: {invocation_ctx.user_content.parts[0].text[:100]}")
                return "Error: Could not understand the diagram request. Please provide a clear query for the diagram."
    
    if not user_diagram_query:
        return "Error: No diagram query provided. Please specify what kind of diagram you need."

    adk_context_for_llm = get_escaped_adk_context_for_llm()
    
    instruction = f"""
You are an AI assistant that generates architecture diagrams in Mermaid syntax. You have access to the ADK (Agent Development Kit) knowledge context.

ADK Knowledge Context:
--- START OF ADK CONTEXT ---
{adk_context_for_llm}
--- END OF ADK CONTEXT ---

The user's request for the diagram is: "{user_diagram_query}"

Follow these steps precisely:
1.  Based on the user's request ("{user_diagram_query}") and the ADK Knowledge Context, generate the Mermaid syntax for the diagram.
    The Mermaid syntax MUST be enclosed in a standard Mermaid code block, like so:
    ```mermaid
    graph TD
        A --> B
    ```
2.  After generating the initial Mermaid syntax, you MUST call the '{mermaid_syntax_verifier_agent.name}' tool.
    The input to this tool MUST be a JSON object with a single key "mermaid_syntax", where the value is the Mermaid syntax you just generated (including the ```mermaid ... ``` block).
    Example call: {{"mermaid_syntax": "```mermaid\\ngraph TD; A-->B;\\n```"}}
3.  The '{mermaid_syntax_verifier_agent.name}' tool will return the verified (and potentially corrected) Mermaid syntax.
    This response will also be a JSON object, and you need to extract the value of the "mermaid_syntax" key from its "result" field.
4.  Once you have the final, verified Mermaid syntax, you MUST call the '{mermaid_gcs_tool_instance.name}' tool.
    The input to this tool MUST be a JSON object with a single key "mermaid_syntax", where the value is the final verified Mermaid syntax (including the ```mermaid ... ``` block).
5.  Your final response for this turn MUST ONLY be the result from the '{mermaid_gcs_tool_instance.name}' tool (which will be a GCS signed URL or an error message).
    Do not add any conversational fluff, explanations, or greetings around this final URL.
"""
    return instruction

async def diagram_generator_after_agent_cb(callback_context: CallbackContext) -> genai_types.Content | None:
    """
    This callback runs after DiagramGeneratorAgent has finished its internal processing.
    It retrieves the GCS link stored by its tool and returns it as the agent's final output.
    """
    gcs_link = callback_context.state.get(GCS_LINK_STATE_KEY)
    if gcs_link:
        logger.info(f"DiagramGeneratorAgent (after_agent_callback): Returning GCS link: {gcs_link}")
        # This Content will be the final output of DiagramGeneratorAgent
        return genai_types.Content(parts=[genai_types.Part(text=str(gcs_link))])
    logger.warning("DiagramGeneratorAgent (after_agent_callback): GCS link not found in state.")
    return genai_types.Content(parts=[genai_types.Part(text="Error: Could not generate diagram link.")])

diagram_generator_agent = ADKAgent(
    name="mermaid_diagram_orchestrator_agent",
    model=Gemini(model=PRO_MODEL_NAME),
    instruction=diagram_generator_agent_instruction_provider,
    tools=[
        AgentTool(agent=mermaid_syntax_verifier_agent), # Output is raw mermaid string
        mermaid_gcs_tool_instance, # Output is URL string
    ],
    input_schema=DiagramGeneratorAgentToolInput, 
    disallow_transfer_to_parent=True, 
    disallow_transfer_to_peers=True,
    before_model_callback=log_prompt_before_model_call,
    after_agent_callback=diagram_generator_after_agent_cb,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.0) # More deterministic
)


# --- Existing Root Agent Logic (Modified) ---
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
        # tool_response should be the URL string (or error string) from diagram_generator_agent.
        if isinstance(tool_response, str) and tool_response.strip():
            tool_context.actions.skip_summarization = True # For root_agent's LLM
            return genai_types.Content(parts=[genai_types.Part(text=tool_response)])
        else:
            error_msg = f"Error: Document generation agent returned an unexpected or empty response: {str(tool_response)[:100]}"
            logger.error(f"RootAgent: {error_msg}")
            tool_context.actions.skip_summarization = True # Still skip, and return the error directly
            return genai_types.Content(parts=[genai_types.Part(text=error_msg)])

    elif tool.name == diagram_generator_agent.name: 
        logger.info(f"RootAgent (after_tool_callback): Received response from '{diagram_generator_agent.name}': {str(tool_response)[:200]}")
        # tool_response should be the URL string (or error string) from diagram_generator_agent.
        if isinstance(tool_response, str) and tool_response.strip():
            tool_context.actions.skip_summarization = True # For root_agent's LLM
            return genai_types.Content(parts=[genai_types.Part(text=tool_response)])
        else:
            error_msg = f"Error: Diagram generation agent returned an unexpected or empty response: {str(tool_response)[:100]}"
            logger.error(f"RootAgent: {error_msg}")
            tool_context.actions.skip_summarization = True # Still skip, and return the error directly
            return genai_types.Content(parts=[genai_types.Part(text=error_msg)])

    elif tool.name == "prepare_document_content_tool":
        logger.info(f"RootAgent (after_tool_callback): 'prepare_document_content_tool' completed. Output from tool: {str(tool_response)[:200]}")
        tool_context.actions.skip_summarization = False 
        return tool_response # Return the raw data for the instruction provider to process

    logger.warning(f"RootAgent (after_tool_callback): Callback for unhandled tool: {tool.name}")
    return None


def root_agent_instruction_provider(context: ReadonlyContext) -> str:
    adk_context_for_llm = get_escaped_adk_context_for_llm()
    invocation_ctx = getattr(context, '_invocation_context', None)
    user_query_text = get_text_from_content(invocation_ctx.user_content) if invocation_ctx and invocation_ctx.user_content else ""
    
    # Check for document generation from prepare_document_content_tool response
    if invocation_ctx and invocation_ctx.session and invocation_ctx.session.events:
        last_event = invocation_ctx.session.events[-1]
        if last_event.author == root_agent.name and \
           last_event.content and last_event.content.parts and \
           last_event.content.parts[0].function_response and \
           last_event.content.parts[0].function_response.name == "prepare_document_content_tool":
            
            logger.info("RootAgent (instruction_provider): Detected response from prepare_document_content_tool. Instructing to call document_generator_agent.")
            tool_output_data_container = last_event.content.parts[0].function_response.response
            
            if "result" in tool_output_data_container and isinstance(tool_output_data_container["result"], dict):
                prepared_content_data = tool_output_data_container["result"]
            else:
                prepared_content_data = tool_output_data_container
                logger.warning(f"RootAgent (instruction_provider): Assuming direct data from prepare_document_content_tool response, no 'result' wrapper found. Data: {str(prepared_content_data)[:100]}")

            try:
                validated_content_for_doc_gen = PrepareDocumentContentToolInput.model_validate(prepared_content_data)
                doc_gen_agent_actual_input = DocumentGeneratorAgentToolInput(
                    markdown_content=validated_content_for_doc_gen.markdown_content,
                    document_type=validated_content_for_doc_gen.document_type,
                    output_filename=validated_content_for_doc_gen.output_filename_base
                )
                system_instruction = f"""
You have received structured data from the 'prepare_document_content_tool'.
The data is: {json.dumps(prepared_content_data)}
Your task is to call the tool named '{document_generator_agent.name}'.
This tool expects arguments conforming to this schema:
{DocumentGeneratorAgentToolInput.model_json_schema()}
Based on the data received from 'prepare_document_content_tool', you MUST call the '{document_generator_agent.name}' tool with the correctly mapped arguments.
Your response should ONLY be the function call. Do not include any other text.
"""
                return system_instruction
            except (ValidationError, Exception) as e:
                logger.error(f"RootAgent (instruction_provider): Error processing data from prepare_document_content_tool for doc gen: {e}. Data: {str(prepared_content_data)[:200]}", exc_info=True)
                return "Error: Could not process the data from the content preparation step. Please try rephrasing your request."

    # GitHub Issue Detection
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

    # Document Generation Request Detection (Initial)
    doc_gen_keywords_pdf = ["pdf", "document", "report"]
    doc_gen_keywords_slides = ["slides", "presentation", "deck", "pptx", "powerpoint", "html slides"]
    requested_doc_type = None
    if any(kw in user_query_text.lower() for kw in doc_gen_keywords_pdf):
        requested_doc_type = "pdf"
    elif any(kw in user_query_text.lower() for kw in doc_gen_keywords_slides):
        requested_doc_type = "pptx" if "pptx" in user_query_text.lower() or "powerpoint" in user_query_text.lower() else "html"

    # Architecture Diagram Request Detection
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
You have access to a tool called 'prepare_document_content_tool'.
ADK Knowledge Context:
--- START OF ADK CONTEXT ---
{adk_context_for_llm}
--- END OF ADK CONTEXT ---
The user wants you to generate a document of type '{requested_doc_type}'. Their request is: "{user_query_text}"
Your tasks are:
1.  Analyze the user's request: "{user_query_text}".
2.  Using the ADK Knowledge Context, generate comprehensive Markdown content for `marp-cli`.
3.  Determine a suitable base filename (e.g., "adk_overview").
4.  You MUST call 'prepare_document_content_tool' with "markdown_content", "document_type" ("{requested_doc_type}"), "output_filename_base", and "original_user_query".
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

class PrepareDocumentContentTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="prepare_document_content_tool", 
            description="Gathers generated markdown content, document type, and filename base. This tool is called by the orchestrator agent; its output triggers the actual document generation agent."
        )

    @override
    def _get_declaration(self):
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
    AgentTool(agent=diagram_generator_agent), # Add the new diagram agent tool
    PrepareDocumentContentTool(),
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
        max_output_tokens=60000, 
        top_p=0.6,
    )
)