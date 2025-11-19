"""
StrataFlow v2.0: Neuro-Symbolic Research Engine

A paradigm shift from stochastic text generation to deterministic knowledge synthesis,
combining symbolic reasoning, causal inference, and neural language models to produce
audit-grade research with mathematical guarantees on logical consistency.
"""

__version__ = "2.0.0"
__author__ = "StrataFlow Team"

from strataflow.core.state import ResearchState
from strataflow.core.types import *
from strataflow.orchestrator.engine import StrataFlowEngine

__all__ = [
    "ResearchState",
    "StrataFlowEngine",
    "__version__",
]
