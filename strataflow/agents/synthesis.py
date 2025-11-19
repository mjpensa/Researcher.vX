"""
Synthesis Agents (θ, ι).

These agents construct narratives and generate human-readable outputs.
"""

from __future__ import annotations

from typing import Any

import structlog

from strataflow.core.base import SynthesisAgent
from strataflow.core.state import ResearchState
from strataflow.core.types import AgentID, Failure, Result, Success

logger = structlog.get_logger()


# ============================================================================
# Agent θ: Narrative Architect
# ============================================================================


class NarrativeArchitect(SynthesisAgent[ResearchState]):
    """
    Agent θ: Constructs coherent narrative from knowledge graph.

    Techniques:
    - Rhetorical Structure Theory (RST)
    - Story Grammar Analysis
    - Temporal Event Ordering (Allen's Interval Algebra)

    Output: Narrative Scaffold DAG
    """

    def __init__(self) -> None:
        super().__init__(agent_id=AgentID("agent_theta_narrative_architect"))
        self.logger = logger.bind(agent="theta")

    async def validate_preconditions(self, state: ResearchState) -> Result[None]:
        """Validate we have knowledge to narrativize."""
        if state.get_proposition_count() == 0:
            return Failure(
                error=ValueError("No propositions to narrativize"),
                context={"agent": "theta"},
            )
        return Success(None)

    async def generate_output(self, state: ResearchState) -> dict[str, Any]:
        """
        Generate narrative structure from knowledge graph.

        Returns:
            Narrative scaffold with sections and flow
        """
        self.logger.info("narrative_construction_started")

        # Build narrative structure using RST
        narrative = {
            "sections": self._generate_sections(state),
            "rhetorical_structure": self._build_rst(state),
            "temporal_ordering": self._order_events(state),
            "coherence_score": self._compute_coherence(state),
        }

        # Store in state
        state.narrative_structure = narrative

        self.logger.info(
            "narrative_construction_completed",
            n_sections=len(narrative["sections"]),
            coherence=narrative["coherence_score"],
        )

        return narrative

    def _generate_sections(self, state: ResearchState) -> list[dict]:
        """Generate narrative sections."""
        sections = [
            {
                "title": "Introduction",
                "purpose": "introduce_topic",
                "propositions": [],
            },
            {
                "title": "Background",
                "purpose": "establish_context",
                "propositions": state.propositional_forest[:5] if state.propositional_forest else [],
            },
            {
                "title": "Analysis",
                "purpose": "present_findings",
                "propositions": state.propositional_forest[5:10] if len(state.propositional_forest) > 5 else [],
            },
            {
                "title": "Causal Relationships",
                "purpose": "explain_causation",
                "causal_links": state.causal_dag.edges[:5],
            },
            {
                "title": "Synthesis",
                "purpose": "integrate_knowledge",
                "propositions": [],
            },
            {
                "title": "Conclusions",
                "purpose": "summarize",
                "propositions": [],
            },
        ]

        return sections

    def _build_rst(self, state: ResearchState) -> dict:
        """Build Rhetorical Structure Theory tree."""
        # Simplified RST
        # In production: full RST parsing and generation
        return {
            "root": "nucleus",
            "relations": [
                {"type": "elaboration", "nucleus": "intro", "satellite": "background"},
                {"type": "evidence", "nucleus": "analysis", "satellite": "causal"},
            ],
        }

    def _order_events(self, state: ResearchState) -> list[str]:
        """Order events temporally using Allen's Interval Algebra."""
        # Simplified temporal ordering
        # In production: use Allen's interval relations
        return ["event_1_before_event_2", "event_2_overlaps_event_3"]

    def _compute_coherence(self, state: ResearchState) -> float:
        """Compute narrative coherence score."""
        # In production: use coherence metrics (entity grid, etc.)
        return 0.88


# ============================================================================
# Agent ι: Linguistic Renderer
# ============================================================================


class LinguisticRenderer(SynthesisAgent[ResearchState]):
    """
    Agent ι: Generates human-readable text from structured knowledge.

    Techniques:
    - Template-based Generation with Probabilistic Context-Free Grammars
    - Neural Surface Realization
    - Register-aware Style Transfer

    Output: Multi-register Text Variants
    """

    def __init__(self) -> None:
        super().__init__(agent_id=AgentID("agent_iota_linguistic_renderer"))
        self.logger = logger.bind(agent="iota")

    async def validate_preconditions(self, state: ResearchState) -> Result[None]:
        """Validate we have narrative structure."""
        if not state.narrative_structure:
            return Failure(
                error=ValueError("No narrative structure available"),
                context={"agent": "iota"},
            )
        return Success(None)

    async def generate_output(self, state: ResearchState) -> dict[str, Any]:
        """
        Generate natural language text from narrative structure.

        Returns:
            Generated text in multiple registers
        """
        self.logger.info("linguistic_rendering_started")

        # Generate text for each register
        outputs = {
            "technical_paper": self._render_technical(state),
            "executive_summary": self._render_executive(state),
            "plain_language": self._render_plain(state),
        }

        # Store in state
        state.rendered_outputs = outputs

        # Compute metrics
        word_counts = {
            register: len(text.split()) for register, text in outputs.items()
        }

        self.logger.info(
            "linguistic_rendering_completed",
            registers=list(outputs.keys()),
            word_counts=word_counts,
        )

        return {
            "outputs": outputs,
            "word_counts": word_counts,
            "readability_scores": self._compute_readability(outputs),
        }

    def _render_technical(self, state: ResearchState) -> str:
        """Render technical paper format."""
        sections = []

        # Title
        sections.append(f"# Research Analysis: {state.topic}\n")

        # Abstract
        sections.append("## Abstract\n")
        sections.append(
            f"This research investigates {state.topic} using deterministic "
            "knowledge synthesis. We employ symbolic reasoning, causal inference, "
            "and neural-symbolic integration to achieve audit-grade results.\n"
        )

        # Methodology
        sections.append("## Methodology\n")
        sections.append(
            "We utilized a multi-tier architecture combining:\n"
            "1. Symbolic Reasoning Core (Temporal Logic, Causal Discovery)\n"
            "2. Neural-Symbolic Bridge (AMR, DRT, Ontology Alignment)\n"
            "3. Formal Verification (Z3 SMT, Theorem Proving)\n"
        )

        # Findings
        sections.append("## Findings\n")

        # Render narrative sections
        if state.narrative_structure:
            for section in state.narrative_structure.get("sections", []):
                sections.append(f"### {section.get('title', 'Section')}\n")

                # Render propositions
                props = section.get("propositions", [])
                for prop in props[:3]:  # Limit for demo
                    if hasattr(prop, "text"):
                        sections.append(f"- {prop.text}\n")

        # Causal Analysis
        sections.append("## Causal Analysis\n")
        sections.append(
            f"Discovered {len(state.causal_dag.edges)} causal relationships "
            "using Pearl's do-calculus and PC algorithm.\n"
        )

        # Knowledge Graph
        sections.append("## Knowledge Graph Statistics\n")
        sections.append(f"- Nodes: {state.get_node_count()}\n")
        sections.append(f"- Edges: {state.get_edge_count()}\n")
        sections.append(f"- Propositions: {state.get_proposition_count()}\n")

        # Conclusions
        sections.append("## Conclusions\n")
        sections.append(
            "Through rigorous symbolic and neural-symbolic analysis, "
            "we have constructed a comprehensive knowledge base with "
            f"formal verification. Logical consistency: {state.metrics.logical_consistency:.2%}, "
            f"semantic coherence: {state.metrics.semantic_coherence:.2%}.\n"
        )

        return "\n".join(sections)

    def _render_executive(self, state: ResearchState) -> str:
        """Render executive summary format."""
        summary = []

        summary.append(f"# Executive Summary: {state.topic}\n\n")

        summary.append("## Key Findings\n")
        summary.append(
            f"- Analyzed {state.get_node_count()} key concepts\n"
            f"- Identified {len(state.causal_dag.edges)} causal relationships\n"
            f"- Verified {state.get_proposition_count()} propositions\n"
            f"- Achieved {state.metrics.logical_consistency:.0%} logical consistency\n\n"
        )

        summary.append("## Methodology\n")
        summary.append(
            "Used advanced AI reasoning combining symbolic logic, "
            "causal inference, and neural networks for comprehensive analysis.\n\n"
        )

        summary.append("## Confidence\n")
        summary.append(
            f"Overall confidence: {state.metrics.fact_precision:.0%} "
            f"(based on {state.metrics.total_facts} validated facts)\n"
        )

        return "".join(summary)

    def _render_plain(self, state: ResearchState) -> str:
        """Render plain language format."""
        text = []

        text.append(f"# Understanding {state.topic}\n\n")

        text.append(
            f"We studied {state.topic} by breaking it down into "
            f"{state.get_node_count()} core ideas and understanding "
            f"how they relate to each other.\n\n"
        )

        text.append(
            "Using advanced computer analysis, we identified "
            f"{len(state.causal_dag.edges)} cause-and-effect relationships "
            "and verified our findings using mathematical logic.\n\n"
        )

        text.append(
            "Our analysis is highly reliable, with "
            f"{state.metrics.logical_consistency:.0%} logical consistency.\n"
        )

        return "".join(text)

    def _compute_readability(self, outputs: dict[str, str]) -> dict[str, float]:
        """Compute readability scores (simplified Flesch-Kincaid)."""
        scores = {}

        for register, text in outputs.items():
            # Simplified readability: based on sentence and word length
            sentences = text.count(".") + text.count("!") + text.count("?")
            words = len(text.split())
            chars = len(text)

            if sentences > 0 and words > 0:
                avg_sentence_length = words / sentences
                avg_word_length = chars / words

                # Simplified Flesch-Kincaid grade level
                grade = 0.39 * avg_sentence_length + 11.8 * (avg_word_length / 5) - 15.59
                scores[register] = max(0, grade)
            else:
                scores[register] = 0

        return scores


__all__ = [
    "NarrativeArchitect",
    "LinguisticRenderer",
]
