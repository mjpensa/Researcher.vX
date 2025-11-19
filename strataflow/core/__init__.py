"""Core foundational types, protocols, and base classes for StrataFlow."""

from strataflow.core.base import Agent, AgentResult
from strataflow.core.protocols import (
    KnowledgeGraphProtocol,
    ReasonerProtocol,
    VerifierProtocol,
)
from strataflow.core.state import ResearchState
from strataflow.core.types import *

__all__ = [
    "Agent",
    "AgentResult",
    "KnowledgeGraphProtocol",
    "ReasonerProtocol",
    "VerifierProtocol",
    "ResearchState",
]
