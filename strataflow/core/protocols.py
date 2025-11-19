"""
Structural protocols for dependency injection and interface segregation.

Uses Python's Protocol class for structural subtyping, enabling dependency
injection without concrete inheritance - following Interface Segregation Principle.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, AsyncIterator, Protocol, runtime_checkable

from strataflow.core.types import (
    AgentID,
    EdgeID,
    KnowledgeEdge,
    KnowledgeNode,
    NodeID,
    Proposition,
    PropositionID,
    ProofTree,
    VerificationResult,
)

if TYPE_CHECKING:
    from strataflow.core.state import ResearchState


# ============================================================================
# Knowledge Graph Protocols
# ============================================================================


@runtime_checkable
class KnowledgeGraphProtocol(Protocol):
    """Protocol for knowledge graph implementations."""

    @abstractmethod
    async def add_node(self, node: KnowledgeNode) -> NodeID:
        """Add a node to the knowledge graph."""
        ...

    @abstractmethod
    async def add_edge(self, edge: KnowledgeEdge) -> EdgeID:
        """Add a hyperedge to the knowledge graph."""
        ...

    @abstractmethod
    async def get_node(self, node_id: NodeID) -> KnowledgeNode | None:
        """Retrieve a node by ID."""
        ...

    @abstractmethod
    async def query_subgraph(
        self, query: str, limit: int = 100
    ) -> list[tuple[KnowledgeNode, list[KnowledgeEdge]]]:
        """Query for a subgraph matching criteria."""
        ...

    @abstractmethod
    async def get_causal_paths(
        self, source: NodeID, target: NodeID
    ) -> list[list[EdgeID]]:
        """Find causal paths between nodes."""
        ...

    @abstractmethod
    def node_count(self) -> int:
        """Get total number of nodes."""
        ...

    @abstractmethod
    def edge_count(self) -> int:
        """Get total number of edges."""
        ...


# ============================================================================
# Reasoning Engine Protocols
# ============================================================================


@runtime_checkable
class ReasonerProtocol(Protocol):
    """Protocol for logical reasoning engines."""

    @abstractmethod
    async def prove(
        self,
        goal: Proposition,
        axioms: list[Proposition],
        max_depth: int = 5,
    ) -> ProofTree | None:
        """Attempt to prove a proposition from axioms."""
        ...

    @abstractmethod
    async def check_consistency(self, propositions: list[Proposition]) -> bool:
        """Check if a set of propositions is logically consistent."""
        ...

    @abstractmethod
    async def find_contradictions(
        self, propositions: list[Proposition]
    ) -> list[tuple[PropositionID, PropositionID]]:
        """Find pairs of contradicting propositions."""
        ...


@runtime_checkable
class CausalReasonerProtocol(Protocol):
    """Protocol for causal inference engines."""

    @abstractmethod
    async def discover_causal_structure(
        self, data: dict[str, list[float]]
    ) -> list[tuple[str, str, float]]:
        """Discover causal relationships from observational data.

        Returns: List of (cause, effect, confidence) tuples.
        """
        ...

    @abstractmethod
    async def compute_counterfactual(
        self,
        intervention: dict[str, float],
        outcome: str,
    ) -> float:
        """Compute counterfactual outcome under intervention."""
        ...

    @abstractmethod
    async def estimate_causal_effect(
        self, cause: str, effect: str, confounders: list[str]
    ) -> float:
        """Estimate causal effect adjusting for confounders."""
        ...


# ============================================================================
# Verification Protocols
# ============================================================================


@runtime_checkable
class VerifierProtocol(Protocol):
    """Protocol for claim verification systems."""

    @abstractmethod
    async def verify_claim(
        self, claim: Proposition, state: ResearchState
    ) -> VerificationResult:
        """Verify a claim against the current research state."""
        ...

    @abstractmethod
    async def batch_verify(
        self, claims: list[Proposition], state: ResearchState
    ) -> list[VerificationResult]:
        """Verify multiple claims in batch."""
        ...


@runtime_checkable
class EntailmentCheckerProtocol(Protocol):
    """Protocol for natural language inference / entailment checking."""

    @abstractmethod
    async def check_entailment(
        self, premise: str | list[str], hypothesis: str
    ) -> float:
        """Check if hypothesis is entailed by premise(s).

        Returns: Confidence score in [0, 1].
        """
        ...

    @abstractmethod
    async def check_batch_entailment(
        self, pairs: list[tuple[str, str]]
    ) -> list[float]:
        """Check entailment for multiple premise-hypothesis pairs."""
        ...


# ============================================================================
# Storage Protocols
# ============================================================================


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol for vector similarity search."""

    @abstractmethod
    async def add_embedding(
        self, id: str, embedding: list[float], metadata: dict
    ) -> None:
        """Add an embedding vector with metadata."""
        ...

    @abstractmethod
    async def search_similar(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float, dict]]:
        """Search for similar vectors.

        Returns: List of (id, similarity_score, metadata) tuples.
        """
        ...

    @abstractmethod
    async def delete_embedding(self, id: str) -> bool:
        """Delete an embedding by ID."""
        ...


@runtime_checkable
class ProvenanceStoreProtocol(Protocol):
    """Protocol for provenance tracking and audit trails."""

    @abstractmethod
    async def record_inference(
        self,
        conclusion: PropositionID,
        premises: list[PropositionID],
        method: str,
        confidence: float,
    ) -> str:
        """Record an inference step and return its hash."""
        ...

    @abstractmethod
    async def get_lineage(self, proposition_id: PropositionID) -> list[dict]:
        """Get the full lineage of a proposition."""
        ...

    @abstractmethod
    async def verify_integrity(self) -> bool:
        """Verify the cryptographic integrity of the provenance chain."""
        ...


# ============================================================================
# Semantic Processing Protocols
# ============================================================================


@runtime_checkable
class SemanticParserProtocol(Protocol):
    """Protocol for semantic parsing."""

    @abstractmethod
    async def parse_to_amr(self, text: str) -> dict:
        """Parse text to Abstract Meaning Representation."""
        ...

    @abstractmethod
    async def extract_semantic_roles(self, text: str) -> list[dict]:
        """Extract semantic roles (agent, patient, etc.)."""
        ...

    @abstractmethod
    async def extract_events(self, text: str) -> list[dict]:
        """Extract event structures from text."""
        ...


@runtime_checkable
class OntologyReasonerProtocol(Protocol):
    """Protocol for ontology reasoning."""

    @abstractmethod
    async def classify_concept(self, concept: str) -> list[str]:
        """Classify a concept within the ontology hierarchy."""
        ...

    @abstractmethod
    async def find_relations(
        self, concept1: str, concept2: str
    ) -> list[tuple[str, float]]:
        """Find semantic relations between concepts.

        Returns: List of (relation_type, confidence) tuples.
        """
        ...

    @abstractmethod
    async def expand_concept(self, concept: str, depth: int = 2) -> set[str]:
        """Expand concept to related concepts up to given depth."""
        ...


# ============================================================================
# LLM Provider Protocols
# ============================================================================


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """Protocol for LLM API providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> str:
        """Generate text completion."""
        ...

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: type,
        system_prompt: str | None = None,
    ) -> dict:
        """Generate structured output matching a schema."""
        ...

    @abstractmethod
    async def stream_generate(
        self, prompt: str, system_prompt: str | None = None
    ) -> AsyncIterator[str]:
        """Stream text generation token by token."""
        ...


# ============================================================================
# Export all protocols
# ============================================================================

__all__ = [
    "KnowledgeGraphProtocol",
    "ReasonerProtocol",
    "CausalReasonerProtocol",
    "VerifierProtocol",
    "EntailmentCheckerProtocol",
    "VectorStoreProtocol",
    "ProvenanceStoreProtocol",
    "SemanticParserProtocol",
    "OntologyReasonerProtocol",
    "LLMProviderProtocol",
]
