"""
Research state management with immutable state objects.

The ResearchState is the central data structure passed between agents,
implementing immutability for safe concurrent access and state tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

import numpy as np
import torch
from pydantic import Field

from strataflow.core.types import (
    Citation,
    ConfidenceScore,
    FloatArray,
    KnowledgeEdge,
    KnowledgeNode,
    NodeID,
    Proposition,
    PropositionID,
    ProofTree,
    ResearchMetrics,
    ResearchPhase,
    SourceID,
    StrataFlowModel,
    Timestamp,
)


# ============================================================================
# Knowledge Graph Components
# ============================================================================


class KnowledgeGraph(StrataFlowModel):
    """In-memory knowledge hypergraph representation."""

    nodes: dict[NodeID, KnowledgeNode] = Field(default_factory=dict)
    edges: dict[tuple[NodeID, ...], list[KnowledgeEdge]] = Field(default_factory=dict)
    causal_edges: list[KnowledgeEdge] = Field(default_factory=list)

    def add_node(self, node: KnowledgeNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node

    def add_edge(self, edge: KnowledgeEdge) -> None:
        """Add an edge to the graph."""
        key = tuple(sorted(edge.source_nodes + edge.target_nodes))
        if key not in self.edges:
            self.edges[key] = []
        self.edges[key].append(edge)

        if edge.is_causal:
            self.causal_edges.append(edge)

    def node_count(self) -> int:
        """Get total number of nodes."""
        return len(self.nodes)

    def edge_count(self) -> int:
        """Get total number of edges."""
        return sum(len(edges) for edges in self.edges.values())


class CausalDAG(StrataFlowModel):
    """Directed Acyclic Graph for causal relationships."""

    nodes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    edges: list[tuple[str, str, float]] = Field(default_factory=list)  # (cause, effect, strength)

    def add_causal_link(self, cause: str, effect: str, strength: float) -> None:
        """Add a causal edge."""
        self.edges.append((cause, effect, strength))


class EpistemicGraph(StrataFlowModel):
    """Graph representing epistemic dependencies (what depends on knowing what)."""

    concepts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    dependencies: dict[str, list[str]] = Field(default_factory=dict)
    uncertainty_scores: dict[str, float] = Field(default_factory=dict)


# ============================================================================
# Temporal and Modal Logic Structures
# ============================================================================


class LinearTemporalLogic(StrataFlowModel):
    """LTL constraints and formulas."""

    formulas: list[str] = Field(default_factory=list)
    temporal_relations: dict[PropositionID, list[tuple[str, PropositionID]]] = Field(
        default_factory=dict
    )  # relation_type -> related_proposition


class KripkeStructure(StrataFlowModel):
    """Modal logic possible worlds structure."""

    worlds: dict[str, dict[str, Any]] = Field(default_factory=dict)
    accessibility: dict[str, list[str]] = Field(default_factory=dict)
    valuations: dict[str, dict[PropositionID, bool]] = Field(default_factory=dict)


# ============================================================================
# Provenance Structures
# ============================================================================


class MerkleDAG(StrataFlowModel):
    """Merkle DAG for fact lineage tracking."""

    nodes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    edges: dict[str, list[str]] = Field(default_factory=dict)
    hashes: dict[str, str] = Field(default_factory=dict)
    root_hash: str | None = None

    def add_fact(self, fact_id: str, content: str, dependencies: list[str]) -> str:
        """Add a fact and compute its hash."""
        import hashlib

        # Simple hash computation (in production, use proper Merkle hashing)
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        dep_hashes = "".join(self.hashes.get(dep, "") for dep in dependencies)
        final_hash = hashlib.sha256((content_hash + dep_hashes).encode()).hexdigest()

        self.nodes[fact_id] = {"content": content, "dependencies": dependencies}
        self.edges[fact_id] = dependencies
        self.hashes[fact_id] = final_hash

        return final_hash


# ============================================================================
# Main Research State
# ============================================================================


class ResearchState(StrataFlowModel):
    """
    Immutable state object passed between agents.

    This is the central data structure in StrataFlow. Each agent receives
    a ResearchState and returns a new ResearchState with updates.

    Design principles:
    - Immutability: Never modify in place, always create new instances
    - Completeness: Contains all information needed by any agent
    - Typed: Strongly typed for compile-time safety
    - Serializable: Can be persisted and restored
    """

    # Metadata
    research_id: str
    current_phase: ResearchPhase = ResearchPhase.INITIALIZATION
    created_at: Timestamp = Field(default_factory=datetime.utcnow)
    updated_at: Timestamp = Field(default_factory=datetime.utcnow)

    # Topic and Configuration
    topic: str
    depth: str = "standard"
    verification_level: str = "standard"

    # Core Knowledge Structures
    knowledge_hypergraph: KnowledgeGraph = Field(default_factory=KnowledgeGraph)
    causal_dag: CausalDAG = Field(default_factory=CausalDAG)
    epistemic_graph: EpistemicGraph = Field(default_factory=EpistemicGraph)

    # Propositions and Proofs
    propositions: dict[PropositionID, Proposition] = Field(default_factory=dict)
    propositional_forest: list[Proposition] = Field(default_factory=list)
    proof_trees: dict[PropositionID, ProofTree] = Field(default_factory=dict)

    # Citations and Sources
    sources: dict[SourceID, Citation] = Field(default_factory=dict)
    cited_facts: set[PropositionID] = Field(default_factory=set)

    # Verification Structures
    entailment_scores: dict[tuple[PropositionID, PropositionID], float] = Field(
        default_factory=dict
    )
    source_reliability: dict[SourceID, float] = Field(default_factory=dict)

    # Temporal & Modal Logic
    temporal_constraints: LinearTemporalLogic = Field(default_factory=LinearTemporalLogic)
    modal_worlds: KripkeStructure = Field(default_factory=KripkeStructure)

    # Provenance
    fact_lineage: MerkleDAG = Field(default_factory=MerkleDAG)
    inference_chains: list[ProofTree] = Field(default_factory=list)

    # Metadata and Confidence
    confidence_intervals: dict[NodeID, tuple[float, float]] = Field(default_factory=dict)
    information_gain: float = 0.0
    epistemic_uncertainty: float = 1.0

    # Metrics
    metrics: ResearchMetrics = Field(default_factory=ResearchMetrics)

    # Agent Execution History
    agent_history: list[tuple[str, Timestamp]] = Field(default_factory=list)

    # Output Artifacts
    narrative_structure: dict[str, Any] = Field(default_factory=dict)
    rendered_outputs: dict[str, str] = Field(default_factory=dict)

    def transition_to(self, new_phase: ResearchPhase) -> ResearchState:
        """Transition to a new research phase."""
        return self.model_copy(
            update={
                "current_phase": new_phase,
                "updated_at": datetime.utcnow(),
                "agent_history": [
                    *self.agent_history,
                    (new_phase.value, datetime.utcnow()),
                ],
            }
        )

    def add_proposition(self, proposition: Proposition) -> ResearchState:
        """Add a proposition to the state."""
        new_propositions = {**self.propositions, proposition.id: proposition}
        return self.model_copy(
            update={"propositions": new_propositions, "updated_at": datetime.utcnow()}
        )

    def add_source(self, source_id: SourceID, citation: Citation) -> ResearchState:
        """Add a source citation."""
        new_sources = {**self.sources, source_id: citation}
        return self.model_copy(
            update={"sources": new_sources, "updated_at": datetime.utcnow()}
        )

    def mark_cited_fact(self, proposition_id: PropositionID) -> ResearchState:
        """Mark a proposition as a cited fact."""
        new_cited_facts = {*self.cited_facts, proposition_id}
        return self.model_copy(
            update={"cited_facts": new_cited_facts, "updated_at": datetime.utcnow()}
        )

    def update_metrics(self, **kwargs: Any) -> ResearchState:
        """Update metrics with new values."""
        new_metrics = self.metrics.model_copy(update=kwargs)
        return self.model_copy(update={"metrics": new_metrics, "updated_at": datetime.utcnow()})

    def record_agent_execution(self, agent_id: str) -> ResearchState:
        """Record that an agent executed."""
        new_history = [*self.agent_history, (agent_id, datetime.utcnow())]
        return self.model_copy(
            update={"agent_history": new_history, "updated_at": datetime.utcnow()}
        )

    def get_node_count(self) -> int:
        """Get total knowledge graph nodes."""
        return self.knowledge_hypergraph.node_count()

    def get_edge_count(self) -> int:
        """Get total knowledge graph edges."""
        return self.knowledge_hypergraph.edge_count()

    def get_causal_link_count(self) -> int:
        """Get total causal links."""
        return len(self.causal_dag.edges)

    def get_proposition_count(self) -> int:
        """Get total propositions."""
        return len(self.propositions)


# ============================================================================
# State Factory
# ============================================================================


def create_initial_state(
    research_id: str,
    topic: str,
    depth: str = "standard",
    verification_level: str = "standard",
) -> ResearchState:
    """Create an initial research state."""
    return ResearchState(
        research_id=research_id,
        topic=topic,
        depth=depth,
        verification_level=verification_level,
        current_phase=ResearchPhase.INITIALIZATION,
    )


__all__ = [
    "ResearchState",
    "KnowledgeGraph",
    "CausalDAG",
    "EpistemicGraph",
    "LinearTemporalLogic",
    "KripkeStructure",
    "MerkleDAG",
    "create_initial_state",
]
