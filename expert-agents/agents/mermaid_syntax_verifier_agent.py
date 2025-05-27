# expert-agents/agents/mermaid_syntax_verifier_agent.py
from google.adk.agents import Agent as ADKAgent
from google.adk.models import Gemini
from google.genai import types as genai_types
from pydantic import BaseModel, Field

from ..config import DEFAULT_MODEL_NAME # Relative import

class MermaidSyntaxVerifierAgentToolInput(BaseModel):
    mermaid_syntax: str = Field(description="The Mermaid syntax to verify and correct.")

mermaid_syntax_verifier_agent = ADKAgent(
    name="mermaid_syntax_verifier_agent",
    model=Gemini(model=DEFAULT_MODEL_NAME),
    instruction=(
        "You are an expert in Mermaid diagram syntax. "
        "Your sole responsibility is to receive Mermaid syntax, verify its correctness, "
        "and correct any syntax errors. "
        "If the provided syntax is already correct, return it as is. "
        "If there are errors, return the corrected Mermaid syntax. "
        "Your output MUST ONLY be the Mermaid code block itself (e.g., ```mermaid\n...\n```). "
        "Do not add any other explanations, greetings, or conversational text."
    ),
    description="Verifies and corrects Mermaid diagram syntax. Expects input as a JSON string with a 'mermaid_syntax' key.",
    input_schema=MermaidSyntaxVerifierAgentToolInput,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.0)
)
