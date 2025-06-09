# expert-agents/tools/adk_guidance_tool.py
import logging
from typing import Any, Dict, override
from pydantic import BaseModel, Field

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import DEFAULT_MODEL_NAME, API_KEY
from ..context_loader import get_escaped_adk_context_for_llm

logger = logging.getLogger(__name__)

class ADKGuidanceToolInput(BaseModel):
    document_text: str = Field(description="The cleaned GitHub issue text or other document for ADK guidance.")

class ADKGuidanceToolOutput(BaseModel):
    guidance: str

class ADKGuidanceTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="adk_guidance_tool",
            description="Provides ADK guidance based on the provided document text and ADK context."
        )
        self.llm = ChatGoogleGenerativeAI(model=DEFAULT_MODEL_NAME, temperature=0.1, google_api_key=API_KEY)
        self.adk_context_for_llm = get_escaped_adk_context_for_llm()

    @override
    def _get_declaration(self):
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=ADKGuidanceToolInput.model_json_schema()
        )

    async def run_async(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
        tool_context.actions.skip_summarization = True
        try:
            input_data = ADKGuidanceToolInput.model_validate(args)
            document_text = input_data.document_text

            if document_text.startswith("Error fetching issue details:") or document_text.startswith("Message from issue fetcher:") or                document_text.startswith("Error: No content extracted") or                document_text == "GitHub issue content was empty after cleaning." or                document_text.startswith("Error retrieving GitHub issue:"):
                logger.warning(f"Tool: {self.name} received prior error/message: {document_text}")
                return ADKGuidanceToolOutput(guidance=document_text).model_dump()

            if not document_text.strip():
                guidance_message = "The provided document text was empty. Cannot provide ADK guidance."
                logger.warning(f"Tool: {self.name} - {guidance_message}")
                return ADKGuidanceToolOutput(guidance=guidance_message).model_dump()

            prompt = f"""
You are an expert on Google's Agent Development Kit (ADK) version 1.2.0.
Your task is to provide feedback based on your comprehensive ADK knowledge and the provided document text.

Your ADK Knowledge Context:
--- START OF ADK CONTEXT ---
{self.adk_context_for_llm}
--- END OF ADK CONTEXT ---

Provided Document Text:
--- START OF DOCUMENT TEXT ---
{document_text}
--- END OF DOCUMENT TEXT ---

Analyze the "Provided Document Text" in conjunction with "Your ADK Knowledge Context".
Formulate a helpful, detailed, and actionable response based SOLELY on these two pieces of information.
This is your final response to be presented to the user. Do not ask further questions or try to use tools.
"""
            logger.info(f"Tool: {self.name} - Prompting LLM with document (first 100 chars): {document_text[:100]}")
            response = await self.llm.ainvoke(prompt)
            guidance = response.content
            logger.info(f"Tool: {self.name} - Guidance received (first 100 chars): {guidance[:100]}")
            return ADKGuidanceToolOutput(guidance=guidance).model_dump()
        except Exception as e:
            logger.error(f"Tool: {self.name} - Error generating guidance: {e}. Args: {args}", exc_info=True)
            return ADKGuidanceToolOutput(guidance=f"Error generating ADK guidance: {e}").model_dump()
