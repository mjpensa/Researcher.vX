"""
Advanced type definitions using Python 3.11+ features.

This module defines the core type system for StrataFlow using:
- Generics with TypeVar and ParamSpec
- Protocol classes for structural subtyping
- NewType for semantic type safety
- Literal types for compile-time constants
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    NewType,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)
from uuid import UUID, uuid4

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# Semantic Type Aliases for Domain Modeling
# ============================================================================

NodeID = NewType("NodeID", UUID)
EdgeID = NewType("EdgeID", UUID)
PropositionID = NewType("PropositionID", UUID)
SourceID = NewType("SourceID", UUID)
AgentID = NewType("AgentID", str)
JobID = NewType("JobID", UUID)

# Type aliases for clarity
ConfidenceScore: TypeAlias = float  # Range: [0.0, 1.0]
Timestamp: TypeAlias = datetime
MerkleHash: TypeAlias = str  # Hex-encoded hash

# NumPy and PyTorch type aliases
FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
Tensor: TypeAlias = torch.Tensor

# ============================================================================
# Enumerations for State and Classification
# ============================================================================


class ResearchPhase(str, Enum):
    """State machine phases for research workflow."""

    INITIALIZATION = "INITIALIZATION"
    CORPUS_DISCOVERY = "CORPUS_DISCOVERY"
    EPISTEMIC_MAPPING = "EPISTEMIC_MAPPING"
    CAUSAL_DISCOVERY = "CAUSAL_DISCOVERY"
    KNOWLEDGE_CONSTRUCTION = "KNOWLEDGE_CONSTRUCTION"
    DIALECTICAL_SYNTHESIS = "DIALECTICAL_SYNTHESIS"
    PROPOSITIONAL_ATOMIZATION = "PROPOSITIONAL_ATOMIZATION"
    ENTAILMENT_VALIDATION = "ENTAILMENT_VALIDATION"
    SOURCE_AUTHENTICATION = "SOURCE_AUTHENTICATION"
    NARRATIVE_CONSTRUCTION = "NARRATIVE_CONSTRUCTION"
    LINGUISTIC_RENDERING = "LINGUISTIC_RENDERING"
    QUALITY_ASSURANCE = "QUALITY_ASSURANCE"
    PUBLICATION = "PUBLICATION"
    REVISION = "REVISION"


class ClaimType(str, Enum):
    """Classification of claims for verification."""

    CITED_FACT = "CITED_FACT"  # Directly cited from source
    CONSERVATIVE_INFERENCE = "CONSERVATIVE_INFERENCE"  # High-confidence entailment
    SPECULATIVE_INFERENCE = "SPECULATIVE_INFERENCE"  # Moderate-confidence inference
    UNSUPPORTED = "UNSUPPORTED"  # No support found


class LogicType(str, Enum):
    """Types of logical systems used in reasoning."""

    PROPOSITIONAL = "PROPOSITIONAL"
    FIRST_ORDER = "FIRST_ORDER"
    TEMPORAL = "TEMPORAL"
    MODAL = "MODAL"
    EPISTEMIC = "EPISTEMIC"
    DEONTIC = "DEONTIC"


class OutputFormat(str, Enum):
    """Supported output formats."""

    TECHNICAL_PAPER = "technical_paper"
    EXECUTIVE_SUMMARY = "executive_summary"
    KNOWLEDGE_GRAPH_JSON = "knowledge_graph_json"
    AUDIT_TRAIL = "audit_trail"
    INTERACTIVE_DASHBOARD = "interactive_dashboard"


# ============================================================================
# Generic Type Variables
# ============================================================================

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

StateT = TypeVar("StateT", bound="ResearchState")
GraphT = TypeVar("GraphT", bound="KnowledgeGraph")
PropositionT = TypeVar("PropositionT", bound="Proposition")

# ============================================================================
# Pydantic Models for Data Validation
# ============================================================================


class StrataFlowModel(BaseModel):
    """Base model with strict configuration."""

    model_config = ConfigDict(
        strict=True,
        frozen=False,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=False,
    )


class Proposition(StrataFlowModel):
    """Atomic proposition with logical structure."""

    id: PropositionID = Field(default_factory=lambda: PropositionID(uuid4()))
    text: str = Field(min_length=1)
    logic_type: LogicType = LogicType.PROPOSITIONAL
    variables: dict[str, Any] = Field(default_factory=dict)
    quantifiers: list[tuple[str, str]] = Field(default_factory=list)  # [(type, var)]
    confidence: ConfidenceScore = Field(ge=0.0, le=1.0, default=1.0)
    created_at: Timestamp = Field(default_factory=datetime.utcnow)

    def __hash__(self) -> int:
        return hash(self.id)


class Citation(StrataFlowModel):
    """Source citation with provenance."""

    source_id: SourceID
    excerpt: str
    page: int | None = None
    url: str | None = Field(None, pattern=r"^https?://")
    doi: str | None = None
    reliability_score: ConfidenceScore = Field(ge=0.0, le=1.0, default=0.5)
    verified_at: Timestamp = Field(default_factory=datetime.utcnow)


class ProofTree(StrataFlowModel):
    """Proof tree for inference validation."""

    conclusion: PropositionID
    premises: list[PropositionID]
    inference_rule: str
    confidence: ConfidenceScore = Field(ge=0.0, le=1.0)
    children: list["ProofTree"] = Field(default_factory=list)
    depth: int = Field(ge=0, default=0)


class VerificationResult(StrataFlowModel):
    """Result of claim verification."""

    proposition_id: PropositionID
    claim_type: ClaimType
    confidence: ConfidenceScore = Field(ge=0.0, le=1.0)
    evidence: list[Citation] = Field(default_factory=list)
    proof_tree: ProofTree | None = None
    contradictions: list[PropositionID] = Field(default_factory=list)


class KnowledgeNode(StrataFlowModel):
    """Node in knowledge hypergraph."""

    id: NodeID = Field(default_factory=lambda: NodeID(uuid4()))
    concept: str
    propositions: list[PropositionID] = Field(default_factory=list)
    embedding: list[float] | None = None  # High-dimensional semantic embedding
    ontology_class: str | None = None
    created_at: Timestamp = Field(default_factory=datetime.utcnow)


class KnowledgeEdge(StrataFlowModel):
    """Hyperedge connecting multiple nodes with typed relations."""

    id: EdgeID = Field(default_factory=lambda: EdgeID(uuid4()))
    source_nodes: list[NodeID]
    target_nodes: list[NodeID]
    relation_type: str
    confidence: ConfidenceScore = Field(ge=0.0, le=1.0, default=1.0)
    is_causal: bool = False
    temporal_order: int | None = None


class ResearchRequest(StrataFlowModel):
    """Input request for research synthesis."""

    topic: str = Field(min_length=3)
    depth: Literal["brief", "standard", "comprehensive", "exhaustive"] = "standard"
    output_formats: list[OutputFormat] = Field(default_factory=lambda: [OutputFormat.TECHNICAL_PAPER])
    verification_level: Literal["basic", "standard", "audit_grade"] = "standard"
    max_sources: int = Field(ge=10, le=10000, default=500)
    language: str = Field(default="en", pattern=r"^[a-z]{2}$")
    deadline: datetime | None = None


class ResearchMetrics(StrataFlowModel):
    """Metrics collected during research execution."""

    # Accuracy metrics
    fact_precision: float = Field(ge=0.0, le=1.0, default=0.0)
    inference_validity: float = Field(ge=0.0, le=1.0, default=0.0)
    causal_accuracy: float = Field(ge=0.0, le=1.0, default=0.0)

    # Knowledge metrics
    total_facts: int = Field(ge=0, default=0)
    conservative_inferences: int = Field(ge=0, default=0)
    speculative_inferences: int = Field(ge=0, default=0)
    contradictions_resolved: int = Field(ge=0, default=0)

    # Graph metrics
    knowledge_graph_nodes: int = Field(ge=0, default=0)
    knowledge_graph_edges: int = Field(ge=0, default=0)
    causal_links: int = Field(ge=0, default=0)

    # Quality metrics
    semantic_coherence: float = Field(ge=0.0, le=1.0, default=0.0)
    logical_consistency: float = Field(ge=0.0, le=1.0, default=0.0)
    citation_coverage: float = Field(ge=0.0, le=1.0, default=0.0)


# ============================================================================
# Result Types with Monadic Error Handling
# ============================================================================


@dataclass(frozen=True)
class Success(Generic[T]):
    """Successful result wrapper."""

    value: T

    def is_success(self) -> bool:
        return True

    def is_failure(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value


@dataclass(frozen=True)
class Failure(Generic[T]):
    """Failed result wrapper with error information."""

    error: Exception
    context: dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        return False

    def is_failure(self) -> bool:
        return True

    def unwrap(self) -> T:
        raise self.error


Result: TypeAlias = Success[T] | Failure[T]


# ============================================================================
# Export all public types
# ============================================================================

__all__ = [
    # Type aliases
    "NodeID",
    "EdgeID",
    "PropositionID",
    "SourceID",
    "AgentID",
    "JobID",
    "ConfidenceScore",
    "Timestamp",
    "MerkleHash",
    "FloatArray",
    "IntArray",
    "Tensor",
    "Result",
    # Enums
    "ResearchPhase",
    "ClaimType",
    "LogicType",
    "OutputFormat",
    # Type vars
    "T",
    "T_co",
    "T_contra",
    "StateT",
    "GraphT",
    "PropositionT",
    # Models
    "StrataFlowModel",
    "Proposition",
    "Citation",
    "ProofTree",
    "VerificationResult",
    "KnowledgeNode",
    "KnowledgeEdge",
    "ResearchRequest",
    "ResearchMetrics",
    "Success",
    "Failure",
]
