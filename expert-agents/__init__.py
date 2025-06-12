"""
This __init__.py file makes the primary agents of the application
(like the root_agent) directly importable from the 'expert-agents' package.
This simplifies the entry point for running the agent server.
"""

from .agent import root_agent
from .agents.github_issue_processing_agent import github_issue_processing_agent

# The __all__ list defines the public API of this package.
# When a user does `from expert-agents import *`, only these names will be imported.
__all__ = ["root_agent", "github_issue_processing_agent"]
