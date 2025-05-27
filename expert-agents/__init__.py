# expert-agents/__init__.py
from .agent import root_agent
from .agents.github_issue_processing_agent import github_issue_processing_agent

__all__ = ["root_agent", "github_issue_processing_agent"]
