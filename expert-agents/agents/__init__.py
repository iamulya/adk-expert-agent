"""
This __init__.py file makes the specialized agents and their Pydantic input
models directly importable from the 'expert_agents.agents' subpackage.
This provides a clean way to access and manage the different agent components.
"""

from .mermaid_diagram_orchestrator_agent import (
    mermaid_diagram_orchestrator_agent,
    DiagramGeneratorAgentToolInput,
)
from .document_generator_agent import (
    document_generator_agent,
    DocumentGeneratorAgentToolInput,
)
from .github_issue_processing_agent import (
    github_issue_processing_agent,
    GitHubIssueProcessingInput,
    SequentialProcessorFinalOutput,
)
from .mermaid_syntax_verifier_agent import (
    mermaid_syntax_verifier_agent,
    MermaidSyntaxVerifierAgentToolInput,
)
from .mermaid_syntax_generator_agent import (
    mermaid_syntax_generator_agent,
    MermaidSyntaxGeneratorAgentInput,
)

# The __all__ list defines the public API of this subpackage.
# It includes the agent instances and their corresponding Pydantic input/output models.
__all__ = [
    # Mermaid Diagram Agent and its input model
    "mermaid_diagram_orchestrator_agent",
    "DiagramGeneratorAgentToolInput",
    # Document Generator Agent and its input model
    "document_generator_agent",
    "DocumentGeneratorAgentToolInput",
    # GitHub Issue Agent and its input/output models
    "github_issue_processing_agent",
    "GitHubIssueProcessingInput",
    "SequentialProcessorFinalOutput",
    # Mermaid Syntax Verifier Agent (available but not in main flow) and its input model
    "mermaid_syntax_verifier_agent",
    "MermaidSyntaxVerifierAgentToolInput",  # Kept for now
    # Mermaid Syntax Generator Agent and its input model
    "mermaid_syntax_generator_agent",
    "MermaidSyntaxGeneratorAgentInput",
]
