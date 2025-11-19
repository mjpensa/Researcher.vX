"""
Base classes for agents and core abstractions.

Implements abstract base classes with dependency injection, async/await patterns,
and comprehensive error handling using Result monads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, final

import structlog

from strataflow.core.types import (
    AgentID,
    Failure,
    ResearchMetrics,
    Result,
    StateT,
    Success,
    Timestamp,
)

if TYPE_CHECKING:
    from strataflow.core.state import ResearchState

logger = structlog.get_logger()


# ============================================================================
# Agent Result Wrapper
# ============================================================================


@dataclass(frozen=True)
class AgentResult(Generic[StateT]):
    """Result of agent execution with updated state and metadata."""

    state: StateT
    agent_id: AgentID
    execution_time: float  # seconds
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    completed_at: Timestamp = field(default_factory=datetime.utcnow)

    def with_metadata(self, **kwargs: Any) -> AgentResult[StateT]:
        """Add metadata to the result."""
        new_metadata = {**self.metadata, **kwargs}
        return AgentResult(
            state=self.state,
            agent_id=self.agent_id,
            execution_time=self.execution_time,
            metadata=new_metadata,
            warnings=self.warnings,
            completed_at=self.completed_at,
        )

    def with_warning(self, warning: str) -> AgentResult[StateT]:
        """Add a warning to the result."""
        new_warnings = [*self.warnings, warning]
        return AgentResult(
            state=self.state,
            agent_id=self.agent_id,
            execution_time=self.execution_time,
            metadata=self.metadata,
            warnings=new_warnings,
            completed_at=self.completed_at,
        )


# ============================================================================
# Base Agent Abstract Class
# ============================================================================


class Agent(ABC, Generic[StateT]):
    """
    Abstract base class for all StrataFlow agents.

    Agents are stateless, async-first, and operate on immutable state objects.
    Each agent transforms a ResearchState into a new ResearchState.

    Design Principles:
    - Dependency Injection: All dependencies passed via constructor
    - Single Responsibility: Each agent has one clear purpose
    - Immutability: State is never mutated, only transformed
    - Composability: Agents can be chained and parallelized
    """

    def __init__(self, agent_id: AgentID) -> None:
        """Initialize the agent with a unique identifier."""
        self.agent_id = agent_id
        self.logger = logger.bind(agent_id=agent_id)

    @abstractmethod
    async def execute(self, state: StateT) -> Result[AgentResult[StateT]]:
        """
        Execute the agent's primary logic.

        Args:
            state: Current research state (immutable)

        Returns:
            Result containing either:
            - Success with new state and metadata
            - Failure with error information

        Note:
            This method should NEVER mutate the input state.
            Always create a new state object for the result.
        """
        ...

    @abstractmethod
    async def validate_preconditions(self, state: StateT) -> Result[None]:
        """
        Validate that preconditions for execution are met.

        Args:
            state: Current research state

        Returns:
            Success(None) if preconditions met, Failure otherwise
        """
        ...

    async def safe_execute(self, state: StateT) -> Result[AgentResult[StateT]]:
        """
        Execute with automatic precondition checking and error handling.

        This is the recommended entry point for agent execution.
        """
        # Validate preconditions
        validation = await self.validate_preconditions(state)
        if validation.is_failure():
            return Failure(
                error=validation.error,  # type: ignore
                context={"agent_id": self.agent_id, "phase": "precondition_check"},
            )

        # Execute main logic with timing
        start_time = datetime.utcnow()
        try:
            result = await self.execute(state)

            if result.is_success():
                agent_result = result.unwrap()
                # Calculate execution time if not already set
                if agent_result.execution_time == 0:
                    execution_time = (datetime.utcnow() - start_time).total_seconds()
                    agent_result = AgentResult(
                        state=agent_result.state,
                        agent_id=agent_result.agent_id,
                        execution_time=execution_time,
                        metadata=agent_result.metadata,
                        warnings=agent_result.warnings,
                        completed_at=agent_result.completed_at,
                    )
                return Success(agent_result)
            else:
                return result

        except Exception as e:
            self.logger.error("agent_execution_failed", error=str(e), exc_info=e)
            return Failure(
                error=e,
                context={
                    "agent_id": self.agent_id,
                    "phase": "execution",
                    "state_snapshot": state.model_dump() if hasattr(state, "model_dump") else str(state),
                },
            )

    @final
    def get_agent_id(self) -> AgentID:
        """Get the agent's unique identifier."""
        return self.agent_id

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agent_id={self.agent_id})"


# ============================================================================
# Specialized Agent Base Classes
# ============================================================================


class ResearchAgent(Agent[StateT], ABC):
    """Base class for primary research agents (α, β, γ, δ)."""

    @abstractmethod
    async def analyze(self, state: StateT) -> dict[str, Any]:
        """Perform agent-specific analysis."""
        ...

    @abstractmethod
    async def synthesize(self, state: StateT, analysis: dict[str, Any]) -> StateT:
        """Synthesize analysis results into new state."""
        ...

    async def execute(self, state: StateT) -> Result[AgentResult[StateT]]:
        """Execute research agent workflow: analyze then synthesize."""
        try:
            analysis = await self.analyze(state)
            new_state = await self.synthesize(state, analysis)

            return Success(
                AgentResult(
                    state=new_state,
                    agent_id=self.agent_id,
                    execution_time=0.0,  # Will be set by safe_execute
                    metadata={"analysis": analysis},
                )
            )
        except Exception as e:
            return Failure(error=e, context={"agent_id": self.agent_id})


class VerificationAgent(Agent[StateT], ABC):
    """Base class for verification agents (ε, ζ, η)."""

    @abstractmethod
    async def verify(self, state: StateT) -> tuple[bool, list[str]]:
        """
        Verify state integrity.

        Returns:
            (is_valid, list_of_issues)
        """
        ...

    async def execute(self, state: StateT) -> Result[AgentResult[StateT]]:
        """Execute verification workflow."""
        try:
            is_valid, issues = await self.verify(state)

            if not is_valid:
                return Failure(
                    error=ValueError(f"Verification failed: {'; '.join(issues)}"),
                    context={"agent_id": self.agent_id, "issues": issues},
                )

            return Success(
                AgentResult(
                    state=state,
                    agent_id=self.agent_id,
                    execution_time=0.0,
                    metadata={"verification": "passed", "issues_checked": len(issues)},
                )
            )
        except Exception as e:
            return Failure(error=e, context={"agent_id": self.agent_id})


class SynthesisAgent(Agent[StateT], ABC):
    """Base class for synthesis agents (θ, ι)."""

    @abstractmethod
    async def generate_output(self, state: StateT) -> dict[str, Any]:
        """Generate output artifacts."""
        ...

    async def execute(self, state: StateT) -> Result[AgentResult[StateT]]:
        """Execute synthesis workflow."""
        try:
            output = await self.generate_output(state)

            return Success(
                AgentResult(
                    state=state,
                    agent_id=self.agent_id,
                    execution_time=0.0,
                    metadata={"output": output},
                )
            )
        except Exception as e:
            return Failure(error=e, context={"agent_id": self.agent_id})


# ============================================================================
# Export all base classes
# ============================================================================

__all__ = [
    "Agent",
    "AgentResult",
    "ResearchAgent",
    "VerificationAgent",
    "SynthesisAgent",
]
