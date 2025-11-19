"""
Main StrataFlow orchestration engine.

Coordinates all components for deterministic knowledge synthesis.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import structlog

from strataflow.core.base import Agent, AgentResult
from strataflow.core.config import get_config
from strataflow.core.state import ResearchState, create_initial_state
from strataflow.core.types import (
    AgentID,
    JobID,
    ResearchPhase,
    ResearchRequest,
    Result,
    Success,
    Failure,
)
from strataflow.orchestrator.provenance import ProvenanceTracker
from strataflow.orchestrator.resource_allocator import (
    AdaptiveResourceAllocator,
    State as RLState,
)
from strataflow.orchestrator.state_machine import DeterministicStateMachine

logger = structlog.get_logger()


# ============================================================================
# Research Job
# ============================================================================


class ResearchJob:
    """Represents a running research job."""

    def __init__(self, job_id: JobID, request: ResearchRequest) -> None:
        self.job_id = job_id
        self.request = request
        self.state: ResearchState | None = None
        self.status: str = "initializing"
        self.progress: float = 0.0
        self.error: str | None = None

        # Create initial state
        self.state = create_initial_state(
            research_id=str(job_id),
            topic=request.topic,
            depth=request.depth,
            verification_level=request.verification_level,
        )


# ============================================================================
# Main Orchestration Engine
# ============================================================================


class StrataFlowEngine:
    """
    Main orchestration engine for StrataFlow v2.0.

    Coordinates:
    - Deterministic state machine
    - Agent execution
    - Resource allocation
    - Provenance tracking
    - Workflow progression
    """

    def __init__(self) -> None:
        """Initialize the StrataFlow engine."""
        self.config = get_config()
        self.logger = logger.bind(component="StrataFlowEngine")

        # Core components
        self.state_machine = DeterministicStateMachine(enable_bft=True)
        self.provenance_tracker = ProvenanceTracker(enable_blockchain=True)
        self.resource_allocator = AdaptiveResourceAllocator(
            total_cpu_cores=self.config.agents.max_parallel_agents * 2,
            total_memory_mb=8192,
            total_gpu=1.0,
        )

        # Agent registry
        self.agents: dict[ResearchPhase, list[Agent]] = {}
        self.registered_agents: dict[AgentID, Agent] = {}

        # Active jobs
        self.jobs: dict[JobID, ResearchJob] = {}

        self.logger.info("strataflow_engine_initialized")

    def register_agent(
        self, phase: ResearchPhase, agent: Agent
    ) -> None:
        """
        Register an agent for a specific phase.

        Args:
            phase: Research phase this agent handles
            agent: Agent instance
        """
        if phase not in self.agents:
            self.agents[phase] = []

        self.agents[phase].append(agent)
        self.registered_agents[agent.get_agent_id()] = agent

        # Register with resource allocator
        self.resource_allocator.register_agent(agent.get_agent_id())

        self.logger.info(
            "agent_registered",
            phase=phase.value,
            agent_id=agent.get_agent_id(),
        )

    async def start_research(
        self, request: ResearchRequest
    ) -> JobID:
        """
        Start a new research job.

        Args:
            request: Research request

        Returns:
            Job ID for tracking
        """
        job_id = JobID(uuid4())
        job = ResearchJob(job_id=job_id, request=request)
        self.jobs[job_id] = job

        self.logger.info(
            "research_started",
            job_id=str(job_id),
            topic=request.topic,
            depth=request.depth,
        )

        # Execute research asynchronously
        asyncio.create_task(self._execute_research(job))

        return job_id

    async def _execute_research(self, job: ResearchJob) -> None:
        """
        Execute research workflow.

        Args:
            job: Research job to execute
        """
        try:
            job.status = "running"
            state = job.state

            # Phase progression map
            phase_sequence = [
                ResearchPhase.INITIALIZATION,
                ResearchPhase.CORPUS_DISCOVERY,
                ResearchPhase.EPISTEMIC_MAPPING,
                ResearchPhase.CAUSAL_DISCOVERY,
                ResearchPhase.KNOWLEDGE_CONSTRUCTION,
                ResearchPhase.DIALECTICAL_SYNTHESIS,
                ResearchPhase.PROPOSITIONAL_ATOMIZATION,
                ResearchPhase.ENTAILMENT_VALIDATION,
                ResearchPhase.SOURCE_AUTHENTICATION,
                ResearchPhase.NARRATIVE_CONSTRUCTION,
                ResearchPhase.LINGUISTIC_RENDERING,
                ResearchPhase.QUALITY_ASSURANCE,
                ResearchPhase.PUBLICATION,
            ]

            total_phases = len(phase_sequence)

            for idx, phase in enumerate(phase_sequence):
                # Update progress
                job.progress = idx / total_phases

                # Transition to phase
                transition_result = await self.state_machine.transition(
                    to_phase=phase,
                    state=state,
                )

                if transition_result.is_failure():
                    raise transition_result.error  # type: ignore

                state = transition_result.unwrap()

                # Execute agents for this phase
                if phase in self.agents:
                    for agent in self.agents[phase]:
                        # Execute agent
                        result = await self._execute_agent(agent, state)

                        if result.is_success():
                            agent_result = result.unwrap()
                            state = agent_result.state
                        else:
                            self.logger.warning(
                                "agent_failed",
                                agent_id=agent.get_agent_id(),
                                phase=phase.value,
                            )

            # Update final state
            job.state = state
            job.status = "completed"
            job.progress = 1.0

            self.logger.info("research_completed", job_id=str(job.job_id))

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            self.logger.error(
                "research_failed",
                job_id=str(job.job_id),
                error=str(e),
                exc_info=e,
            )

    async def _execute_agent(
        self, agent: Agent, state: ResearchState
    ) -> Result[AgentResult[ResearchState]]:
        """
        Execute a single agent with resource allocation.

        Args:
            agent: Agent to execute
            state: Current state

        Returns:
            Agent result or failure
        """
        agent_id = agent.get_agent_id()

        # Convert state to RL state for resource allocator
        rl_state = RLState(
            current_phase=state.current_phase.value,
            knowledge_graph_size=state.get_node_count(),
            n_propositions=state.get_proposition_count(),
            epistemic_uncertainty=state.epistemic_uncertainty,
        )

        # Allocate resources
        allocation = self.resource_allocator.allocate_resources(
            agent_id=agent_id,
            agent_type="research",  # Simplified
        )

        if not allocation:
            return Failure(
                error=RuntimeError("Failed to allocate resources"),
                context={"agent_id": agent_id},
            )

        try:
            # Execute agent
            import time
            start_time = time.time()

            result = await agent.safe_execute(state)

            execution_time = time.time() - start_time

            # Update resource allocator performance
            next_rl_state = RLState(
                current_phase=state.current_phase.value,
                knowledge_graph_size=state.get_node_count() if result.is_success() else rl_state.knowledge_graph_size,
                n_propositions=state.get_proposition_count() if result.is_success() else rl_state.n_propositions,
                epistemic_uncertainty=state.epistemic_uncertainty,
            )

            self.resource_allocator.update_performance(
                agent_id=agent_id,
                state=rl_state,
                next_state=next_rl_state,
                execution_time=execution_time,
                success=result.is_success(),
            )

            return result

        finally:
            # Release resources
            self.resource_allocator.release_resources(agent_id)

    def get_job_status(self, job_id: JobID) -> dict[str, Any]:
        """
        Get status of a research job.

        Args:
            job_id: Job identifier

        Returns:
            Job status information
        """
        if job_id not in self.jobs:
            return {"error": "Job not found"}

        job = self.jobs[job_id]

        return {
            "job_id": str(job_id),
            "status": job.status,
            "progress": job.progress,
            "current_phase": job.state.current_phase.value if job.state else None,
            "error": job.error,
            "metrics": job.state.metrics.model_dump() if job.state else None,
        }

    async def execute_research(
        self, request: ResearchRequest
    ) -> dict[str, Any]:
        """
        Execute research synchronously and return results.

        Args:
            request: Research request

        Returns:
            Research results
        """
        job_id = await self.start_research(request)

        # Wait for completion
        while True:
            status = self.get_job_status(job_id)

            if status["status"] in ["completed", "failed"]:
                break

            await asyncio.sleep(1)

        job = self.jobs[job_id]

        if job.status == "failed":
            return {
                "success": False,
                "error": job.error,
            }

        # Return results
        return {
            "success": True,
            "job_id": str(job_id),
            "state": job.state.model_dump() if job.state else None,
            "metrics": job.state.metrics.model_dump() if job.state else None,
            "outputs": job.state.rendered_outputs if job.state else {},
        }


__all__ = [
    "StrataFlowEngine",
    "ResearchJob",
]
