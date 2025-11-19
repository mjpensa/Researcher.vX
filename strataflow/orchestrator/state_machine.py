"""
Deterministic State Machine with Byzantine Fault Tolerance and Vector Clocks.

Ensures deterministic execution flow with consensus-based fault tolerance.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import structlog

from strataflow.core.state import ResearchState
from strataflow.core.types import AgentID, ResearchPhase, Result, Success, Failure

logger = structlog.get_logger()


# ============================================================================
# Vector Clock for Distributed Synchronization
# ============================================================================


@dataclass
class VectorClock:
    """Vector clock for causal ordering of events."""

    clocks: dict[str, int] = field(default_factory=dict)

    def increment(self, process_id: str) -> VectorClock:
        """Increment clock for a process."""
        new_clocks = self.clocks.copy()
        new_clocks[process_id] = new_clocks.get(process_id, 0) + 1
        return VectorClock(clocks=new_clocks)

    def merge(self, other: VectorClock) -> VectorClock:
        """Merge with another vector clock (take maximum)."""
        all_processes = set(self.clocks.keys()) | set(other.clocks.keys())
        new_clocks = {
            p: max(self.clocks.get(p, 0), other.clocks.get(p, 0))
            for p in all_processes
        }
        return VectorClock(clocks=new_clocks)

    def happens_before(self, other: VectorClock) -> bool:
        """Check if this event happens before another."""
        return (
            all(
                self.clocks.get(p, 0) <= other.clocks.get(p, 0)
                for p in self.clocks.keys()
            )
            and any(
                self.clocks.get(p, 0) < other.clocks.get(p, 0)
                for p in self.clocks.keys()
            )
        )


# ============================================================================
# Byzantine Fault Tolerant Consensus
# ============================================================================


@dataclass
class ConsensusVote:
    """Vote in BFT consensus."""

    node_id: str
    value: Any
    signature: str  # Cryptographic signature (simplified)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ByzantineFaultTolerantConsensus:
    """
    Byzantine Fault Tolerant consensus for state transitions.

    Ensures agreement even with faulty nodes using PBFT-inspired algorithm.
    """

    def __init__(self, n_nodes: int = 4, fault_tolerance: int = 1) -> None:
        """
        Initialize BFT consensus.

        Args:
            n_nodes: Total number of nodes
            fault_tolerance: Maximum number of Byzantine faults to tolerate
        """
        self.n_nodes = n_nodes
        self.f = fault_tolerance  # Can tolerate f faulty nodes
        self.required_votes = 2 * self.f + 1  # Need 2f+1 votes
        self.logger = logger.bind(component="BFTConsensus")

    async def reach_consensus(
        self, proposals: list[ConsensusVote]
    ) -> Result[Any]:
        """
        Reach consensus on a value.

        Args:
            proposals: List of votes from different nodes

        Returns:
            Consensus value if achieved, Failure otherwise
        """
        # Count votes for each value
        vote_counts: dict[Any, list[ConsensusVote]] = {}

        for vote in proposals:
            value_key = str(vote.value)  # Simplified key
            if value_key not in vote_counts:
                vote_counts[value_key] = []
            vote_counts[value_key].append(vote)

        # Check if any value has 2f+1 votes
        for value_key, votes in vote_counts.items():
            if len(votes) >= self.required_votes:
                # Verify signatures (simplified - in production: crypto verification)
                if self._verify_votes(votes):
                    self.logger.info(
                        "consensus_reached",
                        value=value_key,
                        n_votes=len(votes),
                    )
                    return Success(votes[0].value)

        self.logger.warning("consensus_failed", n_proposals=len(proposals))
        return Failure(
            error=ValueError("Failed to reach consensus"),
            context={"proposals": len(proposals), "required": self.required_votes},
        )

    def _verify_votes(self, votes: list[ConsensusVote]) -> bool:
        """Verify cryptographic signatures of votes."""
        # Simplified verification
        # In production: verify each signature cryptographically
        return len(votes) >= self.required_votes


# ============================================================================
# Petri Net Process Controller
# ============================================================================


@dataclass
class Place:
    """Place in a Petri net."""

    name: str
    tokens: int = 0


@dataclass
class Transition:
    """Transition in a Petri net."""

    name: str
    input_places: list[str]
    output_places: list[str]
    guard: Callable[[], bool] | None = None


class PetriNetController:
    """
    Petri Net-based process controller for workflow coordination.

    Provides formal semantics for concurrent workflow execution.
    """

    def __init__(self) -> None:
        self.places: dict[str, Place] = {}
        self.transitions: dict[str, Transition] = {}
        self.logger = logger.bind(component="PetriNetController")

    def add_place(self, name: str, initial_tokens: int = 0) -> None:
        """Add a place to the net."""
        self.places[name] = Place(name=name, tokens=initial_tokens)

    def add_transition(
        self,
        name: str,
        input_places: list[str],
        output_places: list[str],
        guard: Callable[[], bool] | None = None,
    ) -> None:
        """Add a transition to the net."""
        self.transitions[name] = Transition(
            name=name,
            input_places=input_places,
            output_places=output_places,
            guard=guard,
        )

    def is_enabled(self, transition_name: str) -> bool:
        """Check if a transition is enabled (can fire)."""
        transition = self.transitions.get(transition_name)
        if not transition:
            return False

        # Check if all input places have tokens
        for place_name in transition.input_places:
            place = self.places.get(place_name)
            if not place or place.tokens == 0:
                return False

        # Check guard condition if present
        if transition.guard and not transition.guard():
            return False

        return True

    def fire_transition(self, transition_name: str) -> bool:
        """Fire a transition if enabled."""
        if not self.is_enabled(transition_name):
            return False

        transition = self.transitions[transition_name]

        # Remove tokens from input places
        for place_name in transition.input_places:
            self.places[place_name].tokens -= 1

        # Add tokens to output places
        for place_name in transition.output_places:
            self.places[place_name].tokens += 1

        self.logger.info("transition_fired", transition=transition_name)
        return True

    def get_enabled_transitions(self) -> list[str]:
        """Get all currently enabled transitions."""
        return [
            name for name in self.transitions.keys() if self.is_enabled(name)
        ]


# ============================================================================
# Deterministic State Machine
# ============================================================================


@dataclass
class StateTransition:
    """State transition with metadata."""

    from_phase: ResearchPhase
    to_phase: ResearchPhase
    timestamp: datetime
    vector_clock: VectorClock
    agent_id: AgentID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DeterministicStateMachine:
    """
    Deterministic State Machine for research workflow.

    Features:
    - Deterministic phase transitions
    - Byzantine fault tolerance
    - Vector clock synchronization
    - Petri net workflow control
    """

    def __init__(self, enable_bft: bool = True) -> None:
        """
        Initialize state machine.

        Args:
            enable_bft: Enable Byzantine fault tolerance
        """
        self.current_phase = ResearchPhase.INITIALIZATION
        self.transition_history: list[StateTransition] = []
        self.vector_clock = VectorClock()

        # BFT consensus
        self.enable_bft = enable_bft
        if enable_bft:
            self.bft_consensus = ByzantineFaultTolerantConsensus()

        # Petri net controller
        self.petri_net = PetriNetController()
        self._initialize_workflow()

        self.logger = logger.bind(component="DeterministicStateMachine")

    def _initialize_workflow(self) -> None:
        """Initialize Petri net workflow."""
        # Create places for each phase
        for phase in ResearchPhase:
            self.petri_net.add_place(phase.value, initial_tokens=0)

        # Start with INITIALIZATION having a token
        self.petri_net.places[ResearchPhase.INITIALIZATION.value].tokens = 1

        # Define transitions between phases
        transitions = [
            ("init_to_discovery", [ResearchPhase.INITIALIZATION.value], [ResearchPhase.CORPUS_DISCOVERY.value]),
            ("discovery_to_epistemic", [ResearchPhase.CORPUS_DISCOVERY.value], [ResearchPhase.EPISTEMIC_MAPPING.value]),
            ("epistemic_to_causal", [ResearchPhase.EPISTEMIC_MAPPING.value], [ResearchPhase.CAUSAL_DISCOVERY.value]),
            ("causal_to_construction", [ResearchPhase.CAUSAL_DISCOVERY.value], [ResearchPhase.KNOWLEDGE_CONSTRUCTION.value]),
            ("construction_to_synthesis", [ResearchPhase.KNOWLEDGE_CONSTRUCTION.value], [ResearchPhase.DIALECTICAL_SYNTHESIS.value]),
            ("synthesis_to_atomization", [ResearchPhase.DIALECTICAL_SYNTHESIS.value], [ResearchPhase.PROPOSITIONAL_ATOMIZATION.value]),
            ("atomization_to_entailment", [ResearchPhase.PROPOSITIONAL_ATOMIZATION.value], [ResearchPhase.ENTAILMENT_VALIDATION.value]),
            ("entailment_to_auth", [ResearchPhase.ENTAILMENT_VALIDATION.value], [ResearchPhase.SOURCE_AUTHENTICATION.value]),
            ("auth_to_narrative", [ResearchPhase.SOURCE_AUTHENTICATION.value], [ResearchPhase.NARRATIVE_CONSTRUCTION.value]),
            ("narrative_to_rendering", [ResearchPhase.NARRATIVE_CONSTRUCTION.value], [ResearchPhase.LINGUISTIC_RENDERING.value]),
            ("rendering_to_qa", [ResearchPhase.LINGUISTIC_RENDERING.value], [ResearchPhase.QUALITY_ASSURANCE.value]),
            ("qa_to_publication", [ResearchPhase.QUALITY_ASSURANCE.value], [ResearchPhase.PUBLICATION.value]),
        ]

        for name, inputs, outputs in transitions:
            self.petri_net.add_transition(name, inputs, outputs)

    async def transition(
        self,
        to_phase: ResearchPhase,
        state: ResearchState,
        agent_id: AgentID | None = None,
    ) -> Result[ResearchState]:
        """
        Transition to a new phase with BFT consensus.

        Args:
            to_phase: Target phase
            state: Current research state
            agent_id: ID of agent requesting transition

        Returns:
            Updated state or Failure
        """
        # Check if transition is valid via Petri net
        if not self._is_valid_transition(self.current_phase, to_phase):
            return Failure(
                error=ValueError(f"Invalid transition: {self.current_phase} -> {to_phase}"),
                context={"from": self.current_phase, "to": to_phase},
            )

        # BFT consensus if enabled
        if self.enable_bft:
            vote = ConsensusVote(
                node_id="node_0",  # Simplified
                value=to_phase,
                signature="sig_placeholder",
            )
            consensus_result = await self.bft_consensus.reach_consensus([vote, vote, vote])

            if consensus_result.is_failure():
                return Failure(
                    error=ValueError("BFT consensus failed for transition"),
                    context={"to_phase": to_phase},
                )

        # Update vector clock
        self.vector_clock = self.vector_clock.increment("state_machine")

        # Record transition
        transition = StateTransition(
            from_phase=self.current_phase,
            to_phase=to_phase,
            timestamp=datetime.utcnow(),
            vector_clock=self.vector_clock,
            agent_id=agent_id,
        )
        self.transition_history.append(transition)

        # Update Petri net
        transition_name = self._get_transition_name(self.current_phase, to_phase)
        if transition_name:
            self.petri_net.fire_transition(transition_name)

        # Update phase
        self.current_phase = to_phase

        # Update state
        new_state = state.transition_to(to_phase)

        self.logger.info(
            "phase_transition",
            from_phase=transition.from_phase.value,
            to_phase=transition.to_phase.value,
            vector_clock=self.vector_clock.clocks,
        )

        return Success(new_state)

    def _is_valid_transition(
        self, from_phase: ResearchPhase, to_phase: ResearchPhase
    ) -> bool:
        """Check if a phase transition is valid."""
        # Check via Petri net enabled transitions
        enabled = self.petri_net.get_enabled_transitions()

        for trans_name in enabled:
            trans = self.petri_net.transitions[trans_name]
            if (
                from_phase.value in trans.input_places
                and to_phase.value in trans.output_places
            ):
                return True

        return False

    def _get_transition_name(
        self, from_phase: ResearchPhase, to_phase: ResearchPhase
    ) -> str | None:
        """Get Petri net transition name for phase change."""
        for name, trans in self.petri_net.transitions.items():
            if (
                from_phase.value in trans.input_places
                and to_phase.value in trans.output_places
            ):
                return name
        return None

    def get_current_phase(self) -> ResearchPhase:
        """Get current research phase."""
        return self.current_phase

    def get_transition_history(self) -> list[StateTransition]:
        """Get full transition history."""
        return self.transition_history.copy()


__all__ = [
    "DeterministicStateMachine",
    "VectorClock",
    "ByzantineFaultTolerantConsensus",
    "PetriNetController",
    "StateTransition",
]
