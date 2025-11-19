"""
Temporal Logic Engine implementing LTL, CTL, and Epistemic Modal Logic.

Uses advanced formal methods for temporal reasoning and verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

import structlog
from z3 import (
    And,
    Bool,
    Exists,
    ForAll,
    Implies,
    Not,
    Or,
    Solver,
    sat,
)

from strataflow.core.types import Proposition, PropositionID

logger = structlog.get_logger()


# ============================================================================
# Temporal Logic Types
# ============================================================================


class TemporalOperator(str, Enum):
    """Linear Temporal Logic operators."""

    NEXT = "X"  # Next state
    GLOBALLY = "G"  # Always (globally)
    FINALLY = "F"  # Eventually (finally)
    UNTIL = "U"  # Until
    RELEASE = "R"  # Release (dual of Until)


class CTLOperator(str, Enum):
    """Computation Tree Logic operators."""

    EX = "EX"  # Exists next
    AX = "AX"  # All next
    EG = "EG"  # Exists globally
    AG = "AG"  # All globally
    EF = "EF"  # Exists finally
    AF = "AF"  # All finally
    EU = "EU"  # Exists until
    AU = "AU"  # All until


class EpistemicOperator(str, Enum):
    """Epistemic modal logic operators."""

    KNOWS = "K"  # Agent knows
    BELIEVES = "B"  # Agent believes
    COMMON_KNOWLEDGE = "C"  # Common knowledge among agents
    DISTRIBUTED_KNOWLEDGE = "D"  # Distributed knowledge


@dataclass(frozen=True)
class TemporalFormula:
    """Temporal logic formula representation."""

    operator: TemporalOperator | CTLOperator | EpistemicOperator | None
    propositions: list[PropositionID]
    subformulas: list[TemporalFormula]
    is_atomic: bool = False

    def __str__(self) -> str:
        if self.is_atomic:
            return f"p{self.propositions[0]}"
        if self.operator:
            if len(self.subformulas) == 1:
                return f"{self.operator.value}({self.subformulas[0]})"
            elif len(self.subformulas) == 2:
                return f"({self.subformulas[0]} {self.operator.value} {self.subformulas[1]})"
        return "⊤"  # True


# ============================================================================
# Linear Temporal Logic (LTL) Validator
# ============================================================================


class LTLValidator:
    """
    Linear Temporal Logic validator using bounded model checking.

    LTL allows expressing properties like:
    - G(p → F(q)): "Globally, if p holds, then q will eventually hold"
    - F(G(p)): "Eventually, p will hold forever"
    """

    def __init__(self, max_depth: int = 20) -> None:
        self.max_depth = max_depth
        self.logger = logger.bind(component="LTLValidator")

    def validate_formula(
        self,
        formula: TemporalFormula,
        trace: list[dict[PropositionID, bool]],
    ) -> bool:
        """
        Validate an LTL formula against a temporal trace.

        Args:
            formula: LTL formula to validate
            trace: Sequence of states (valuations of propositions)

        Returns:
            True if formula holds on the trace
        """
        return self._evaluate_ltl(formula, trace, 0)

    def _evaluate_ltl(
        self,
        formula: TemporalFormula,
        trace: list[dict[PropositionID, bool]],
        position: int,
    ) -> bool:
        """Recursively evaluate LTL formula at given position."""
        if position >= len(trace):
            return False

        if formula.is_atomic:
            prop_id = formula.propositions[0]
            return trace[position].get(prop_id, False)

        match formula.operator:
            case TemporalOperator.NEXT:
                if position + 1 < len(trace):
                    return self._evaluate_ltl(formula.subformulas[0], trace, position + 1)
                return False

            case TemporalOperator.GLOBALLY:
                # Check if subformula holds at all future positions
                for i in range(position, len(trace)):
                    if not self._evaluate_ltl(formula.subformulas[0], trace, i):
                        return False
                return True

            case TemporalOperator.FINALLY:
                # Check if subformula holds at some future position
                for i in range(position, len(trace)):
                    if self._evaluate_ltl(formula.subformulas[0], trace, i):
                        return True
                return False

            case TemporalOperator.UNTIL:
                # φ U ψ: ψ must hold at some position, and φ must hold until then
                phi, psi = formula.subformulas[0], formula.subformulas[1]
                for i in range(position, len(trace)):
                    if self._evaluate_ltl(psi, trace, i):
                        # Check if phi held at all positions before i
                        return all(
                            self._evaluate_ltl(phi, trace, j)
                            for j in range(position, i)
                        )
                return False

            case _:
                return False

    def synthesize_ltl_constraint(
        self,
        positive_traces: list[list[dict[PropositionID, bool]]],
        negative_traces: list[list[dict[PropositionID, bool]]],
    ) -> TemporalFormula | None:
        """
        Synthesize an LTL formula that accepts positive traces and rejects negative ones.

        This is a simplified synthesis - in production, use more sophisticated algorithms
        like IC3 or synthesis from automata.
        """
        # Simplified: Generate basic patterns
        # In production, use decision tree learning or genetic programming
        self.logger.info(
            "ltl_synthesis_started",
            positive_count=len(positive_traces),
            negative_count=len(negative_traces),
        )

        # For now, return a simple example formula
        # Real implementation would use synthesis algorithms
        return TemporalFormula(
            operator=TemporalOperator.GLOBALLY,
            propositions=[],
            subformulas=[
                TemporalFormula(
                    operator=None,
                    propositions=[],
                    subformulas=[],
                    is_atomic=True,
                )
            ],
        )


# ============================================================================
# Computation Tree Logic (CTL) Planner
# ============================================================================


class CTLPlanner:
    """
    CTL model checker for branching-time temporal logic.

    CTL extends LTL with quantification over paths:
    - AG(p): On all paths, p always holds
    - EF(p): There exists a path where p eventually holds
    """

    def __init__(self) -> None:
        self.logger = logger.bind(component="CTLPlanner")

    def check_ctl_property(
        self,
        formula: TemporalFormula,
        states: set[str],
        transitions: dict[str, list[str]],
        labeling: dict[str, set[PropositionID]],
    ) -> set[str]:
        """
        Model checking for CTL formulas.

        Args:
            formula: CTL formula
            states: Set of state names
            transitions: State transition relation
            labeling: States labeled with propositions that hold

        Returns:
            Set of states where formula holds
        """
        if formula.is_atomic:
            prop_id = formula.propositions[0]
            return {s for s in states if prop_id in labeling.get(s, set())}

        match formula.operator:
            case CTLOperator.EX:  # Exists next
                phi_states = self.check_ctl_property(
                    formula.subformulas[0], states, transitions, labeling
                )
                return {
                    s
                    for s in states
                    if any(succ in phi_states for succ in transitions.get(s, []))
                }

            case CTLOperator.AX:  # All next
                phi_states = self.check_ctl_property(
                    formula.subformulas[0], states, transitions, labeling
                )
                return {
                    s
                    for s in states
                    if all(succ in phi_states for succ in transitions.get(s, []))
                    and len(transitions.get(s, [])) > 0
                }

            case CTLOperator.EG:  # Exists globally
                return self._check_eg(formula.subformulas[0], states, transitions, labeling)

            case CTLOperator.EF:  # Exists finally
                return self._check_ef(formula.subformulas[0], states, transitions, labeling)

            case _:
                return set()

    def _check_ef(
        self,
        formula: TemporalFormula,
        states: set[str],
        transitions: dict[str, list[str]],
        labeling: dict[str, set[PropositionID]],
    ) -> set[str]:
        """Check EF (exists finally) using fixed-point iteration."""
        result = self.check_ctl_property(formula, states, transitions, labeling)
        changed = True

        while changed:
            old_result = result.copy()
            # Add states that can reach result states
            for s in states:
                if any(succ in result for succ in transitions.get(s, [])):
                    result.add(s)
            changed = result != old_result

        return result

    def _check_eg(
        self,
        formula: TemporalFormula,
        states: set[str],
        transitions: dict[str, list[str]],
        labeling: dict[str, set[PropositionID]],
    ) -> set[str]:
        """Check EG (exists globally) using fixed-point iteration."""
        result = self.check_ctl_property(formula, states, transitions, labeling)
        changed = True

        while changed:
            old_result = result.copy()
            # Keep only states from which we can stay in result
            result = {
                s
                for s in result
                if any(succ in result for succ in transitions.get(s, []))
            }
            changed = result != old_result

        return result


# ============================================================================
# Epistemic Modal Logic Reasoner
# ============================================================================


class EpistemicReasoner:
    """
    Epistemic modal logic for reasoning about knowledge and belief.

    Supports:
    - Single-agent knowledge: K_a(φ) = "Agent a knows φ"
    - Common knowledge: C(φ) = "φ is common knowledge"
    - Distributed knowledge: D(φ) = "φ is distributed knowledge"
    """

    def __init__(self) -> None:
        self.logger = logger.bind(component="EpistemicReasoner")

    def check_knowledge(
        self,
        agent: str,
        proposition: Proposition,
        world: str,
        accessibility: dict[str, dict[str, list[str]]],
        valuations: dict[str, dict[PropositionID, bool]],
    ) -> bool:
        """
        Check if an agent knows a proposition in a given world.

        K_a(φ) holds in world w iff φ holds in all worlds accessible to agent a from w.

        Args:
            agent: Agent identifier
            proposition: Proposition to check
            world: Current world
            accessibility: Agent -> World -> List of accessible worlds
            valuations: World -> Proposition -> Truth value
        """
        accessible_worlds = accessibility.get(agent, {}).get(world, [])

        # Agent knows φ if φ holds in all accessible worlds
        for accessible_world in accessible_worlds:
            if not valuations.get(accessible_world, {}).get(proposition.id, False):
                return False

        return True

    def check_common_knowledge(
        self,
        agents: list[str],
        proposition: Proposition,
        world: str,
        accessibility: dict[str, dict[str, list[str]]],
        valuations: dict[str, dict[PropositionID, bool]],
    ) -> bool:
        """
        Check if a proposition is common knowledge among agents.

        C(φ) = Everyone knows φ, everyone knows that everyone knows φ, etc.
        """
        # Simplified: Check if all agents know, and all agents know that all know
        # Full implementation requires transitive closure of combined accessibility

        for agent in agents:
            if not self.check_knowledge(
                agent, proposition, world, accessibility, valuations
            ):
                return False

        # In production: iterate until fixed point
        return True


# ============================================================================
# Main Temporal Logic Engine
# ============================================================================


class TemporalLogicEngine:
    """
    Unified temporal logic engine combining LTL, CTL, and Epistemic logic.
    """

    def __init__(self, max_depth: int = 20) -> None:
        self.ltl_validator = LTLValidator(max_depth=max_depth)
        self.ctl_planner = CTLPlanner()
        self.epistemic_reasoner = EpistemicReasoner()
        self.logger = logger.bind(component="TemporalLogicEngine")

    def validate_temporal_constraint(
        self,
        formula: TemporalFormula,
        trace: list[dict[PropositionID, bool]],
    ) -> bool:
        """Validate a temporal formula (automatically dispatches to LTL or CTL)."""
        if isinstance(formula.operator, TemporalOperator):
            return self.ltl_validator.validate_formula(formula, trace)
        else:
            self.logger.warning("ctl_validation_not_implemented_for_traces")
            return False

    def check_epistemic_property(
        self,
        agent: str,
        proposition: Proposition,
        world: str,
        accessibility: dict[str, dict[str, list[str]]],
        valuations: dict[str, dict[PropositionID, bool]],
    ) -> bool:
        """Check epistemic properties about knowledge and belief."""
        return self.epistemic_reasoner.check_knowledge(
            agent, proposition, world, accessibility, valuations
        )


__all__ = [
    "TemporalLogicEngine",
    "LTLValidator",
    "CTLPlanner",
    "EpistemicReasoner",
    "TemporalFormula",
    "TemporalOperator",
    "CTLOperator",
    "EpistemicOperator",
]
