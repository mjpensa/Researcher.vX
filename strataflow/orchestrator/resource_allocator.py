"""
Adaptive Resource Allocator using Reinforcement Learning and Multi-Armed Bandits.

Optimizes agent selection and resource allocation using learned policies.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray

from strataflow.core.types import AgentID

logger = structlog.get_logger()


# ============================================================================
# Multi-Armed Bandit for Agent Selection
# ============================================================================


@dataclass
class BanditArm:
    """Arm in multi-armed bandit (represents an agent)."""

    agent_id: AgentID
    total_reward: float = 0.0
    n_pulls: int = 0
    estimated_value: float = 0.0


class MultiArmedBandit:
    """
    Multi-Armed Bandit for adaptive agent selection.

    Uses UCB1 (Upper Confidence Bound) algorithm for exploration-exploitation.
    """

    def __init__(self, exploration_constant: float = 2.0) -> None:
        """
        Initialize MAB.

        Args:
            exploration_constant: Controls exploration vs exploitation trade-off
        """
        self.arms: dict[AgentID, BanditArm] = {}
        self.exploration_constant = exploration_constant
        self.total_pulls = 0
        self.logger = logger.bind(component="MultiArmedBandit")

    def add_arm(self, agent_id: AgentID) -> None:
        """Add an agent as a bandit arm."""
        if agent_id not in self.arms:
            self.arms[agent_id] = BanditArm(agent_id=agent_id)

    def select_arm(self) -> AgentID:
        """
        Select an arm using UCB1 algorithm.

        Returns:
            Selected agent ID
        """
        # Ensure all arms pulled at least once
        for agent_id, arm in self.arms.items():
            if arm.n_pulls == 0:
                return agent_id

        # UCB1: select arm with highest upper confidence bound
        best_agent = None
        best_ucb = float('-inf')

        for agent_id, arm in self.arms.items():
            # UCB1 formula: Q(a) + c * sqrt(ln(total_pulls) / pulls(a))
            exploitation = arm.estimated_value
            exploration = self.exploration_constant * math.sqrt(
                math.log(self.total_pulls) / arm.n_pulls
            )
            ucb = exploitation + exploration

            if ucb > best_ucb:
                best_ucb = ucb
                best_agent = agent_id

        return best_agent  # type: ignore

    def update(self, agent_id: AgentID, reward: float) -> None:
        """
        Update arm statistics after observing reward.

        Args:
            agent_id: Agent that was selected
            reward: Observed reward
        """
        if agent_id not in self.arms:
            self.add_arm(agent_id)

        arm = self.arms[agent_id]
        arm.n_pulls += 1
        arm.total_reward += reward

        # Update estimated value (running average)
        arm.estimated_value = arm.total_reward / arm.n_pulls

        self.total_pulls += 1

        self.logger.debug(
            "arm_updated",
            agent_id=agent_id,
            reward=reward,
            estimated_value=arm.estimated_value,
        )

    def get_best_arm(self) -> AgentID:
        """Get arm with highest estimated value."""
        best_agent = max(
            self.arms.items(),
            key=lambda x: x[1].estimated_value,
        )[0]
        return best_agent


# ============================================================================
# Reinforcement Learning Optimizer
# ============================================================================


@dataclass
class State:
    """RL state representation."""

    current_phase: str
    knowledge_graph_size: int
    n_propositions: int
    epistemic_uncertainty: float

    def to_vector(self) -> NDArray[np.float64]:
        """Convert to feature vector."""
        # Simple feature encoding
        return np.array([
            hash(self.current_phase) % 100,  # Simplified
            self.knowledge_graph_size,
            self.n_propositions,
            self.epistemic_uncertainty,
        ], dtype=np.float64)


@dataclass
class Action:
    """RL action representation."""

    agent_id: AgentID
    priority: float = 1.0


class QLearningOptimizer:
    """
    Q-Learning based optimizer for agent scheduling.

    Learns optimal agent selection policy based on state.
    """

    def __init__(
        self,
        state_dim: int = 4,
        n_agents: int = 10,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
    ) -> None:
        """
        Initialize Q-Learning optimizer.

        Args:
            state_dim: Dimensionality of state space
            n_agents: Number of agents (actions)
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate for epsilon-greedy
        """
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Q-table: state_hash -> agent_idx -> Q-value
        self.q_table: dict[int, NDArray[np.float64]] = defaultdict(
            lambda: np.zeros(n_agents)
        )

        self.agent_ids: list[AgentID] = []
        self.logger = logger.bind(component="QLearningOptimizer")

    def register_agent(self, agent_id: AgentID) -> None:
        """Register an agent."""
        if agent_id not in self.agent_ids:
            self.agent_ids.append(agent_id)

    def _state_to_hash(self, state: State) -> int:
        """Convert state to hash for Q-table lookup."""
        # Simple discretization
        state_vec = state.to_vector()
        # Discretize continuous values
        discretized = tuple(int(x / 10) for x in state_vec)
        return hash(discretized)

    def select_action(self, state: State) -> AgentID:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected agent ID
        """
        if not self.agent_ids:
            raise ValueError("No agents registered")

        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Explore: random agent
            return np.random.choice(self.agent_ids)

        # Exploit: best agent according to Q-values
        state_hash = self._state_to_hash(state)
        q_values = self.q_table[state_hash]

        if len(q_values) < len(self.agent_ids):
            # Ensure Q-table has values for all agents
            self.q_table[state_hash] = np.zeros(len(self.agent_ids))
            return self.agent_ids[0]

        best_agent_idx = np.argmax(q_values)
        return self.agent_ids[best_agent_idx]

    def update(
        self,
        state: State,
        action: AgentID,
        reward: float,
        next_state: State,
    ) -> None:
        """
        Update Q-values using Q-learning update rule.

        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

        Args:
            state: Current state
            action: Action taken
            reward: Observed reward
            next_state: Next state
        """
        if action not in self.agent_ids:
            self.register_agent(action)

        state_hash = self._state_to_hash(state)
        next_state_hash = self._state_to_hash(next_state)
        action_idx = self.agent_ids.index(action)

        # Q-learning update
        current_q = self.q_table[state_hash][action_idx]
        max_next_q = np.max(self.q_table[next_state_hash])

        new_q = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )

        self.q_table[state_hash][action_idx] = new_q

        self.logger.debug(
            "q_value_updated",
            agent_id=action,
            old_q=current_q,
            new_q=new_q,
        )


# ============================================================================
# Resource Allocation
# ============================================================================


@dataclass
class ResourceAllocation:
    """Resource allocation decision."""

    agent_id: AgentID
    cpu_cores: int
    memory_mb: int
    gpu_allocation: float  # Fraction of GPU (0.0 - 1.0)
    timeout_seconds: int
    priority: int


class ResourceAllocator:
    """
    Allocates computational resources to agents.

    Manages CPU, memory, and GPU allocation based on agent requirements
    and system constraints.
    """

    def __init__(
        self,
        total_cpu_cores: int = 8,
        total_memory_mb: int = 16384,
        total_gpu: float = 1.0,
    ) -> None:
        """
        Initialize resource allocator.

        Args:
            total_cpu_cores: Total available CPU cores
            total_memory_mb: Total available memory in MB
            total_gpu: Total GPU resources (1.0 = 1 full GPU)
        """
        self.total_cpu = total_cpu_cores
        self.total_memory = total_memory_mb
        self.total_gpu = total_gpu

        self.available_cpu = total_cpu_cores
        self.available_memory = total_memory_mb
        self.available_gpu = total_gpu

        self.allocations: dict[AgentID, ResourceAllocation] = {}
        self.logger = logger.bind(component="ResourceAllocator")

    def allocate(
        self,
        agent_id: AgentID,
        cpu_cores: int = 2,
        memory_mb: int = 2048,
        gpu_allocation: float = 0.0,
        timeout_seconds: int = 300,
        priority: int = 1,
    ) -> ResourceAllocation | None:
        """
        Allocate resources to an agent.

        Args:
            agent_id: Agent to allocate resources to
            cpu_cores: CPU cores requested
            memory_mb: Memory in MB requested
            gpu_allocation: GPU fraction requested
            timeout_seconds: Execution timeout
            priority: Priority level

        Returns:
            ResourceAllocation if successful, None if insufficient resources
        """
        # Check availability
        if (
            cpu_cores > self.available_cpu
            or memory_mb > self.available_memory
            or gpu_allocation > self.available_gpu
        ):
            self.logger.warning(
                "insufficient_resources",
                agent_id=agent_id,
                requested_cpu=cpu_cores,
                available_cpu=self.available_cpu,
            )
            return None

        # Create allocation
        allocation = ResourceAllocation(
            agent_id=agent_id,
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            gpu_allocation=gpu_allocation,
            timeout_seconds=timeout_seconds,
            priority=priority,
        )

        # Update availability
        self.available_cpu -= cpu_cores
        self.available_memory -= memory_mb
        self.available_gpu -= gpu_allocation

        self.allocations[agent_id] = allocation

        self.logger.info(
            "resources_allocated",
            agent_id=agent_id,
            cpu=cpu_cores,
            memory_mb=memory_mb,
            gpu=gpu_allocation,
        )

        return allocation

    def deallocate(self, agent_id: AgentID) -> None:
        """Release resources allocated to an agent."""
        if agent_id not in self.allocations:
            return

        allocation = self.allocations[agent_id]

        # Return resources to pool
        self.available_cpu += allocation.cpu_cores
        self.available_memory += allocation.memory_mb
        self.available_gpu += allocation.gpu_allocation

        del self.allocations[agent_id]

        self.logger.info("resources_deallocated", agent_id=agent_id)


# ============================================================================
# Adaptive Resource Allocator
# ============================================================================


class AdaptiveResourceAllocator:
    """
    Adaptive resource allocator combining MAB, Q-learning, and resource management.

    Learns optimal agent selection and resource allocation policies.
    """

    def __init__(
        self,
        total_cpu_cores: int = 8,
        total_memory_mb: int = 16384,
        total_gpu: float = 1.0,
    ) -> None:
        """Initialize adaptive allocator."""
        self.mab = MultiArmedBandit()
        self.q_learning = QLearningOptimizer()
        self.resource_allocator = ResourceAllocator(
            total_cpu_cores, total_memory_mb, total_gpu
        )
        self.logger = logger.bind(component="AdaptiveResourceAllocator")

        # Track performance history
        self.performance_history: list[dict[str, Any]] = []

    def register_agent(self, agent_id: AgentID) -> None:
        """Register an agent for selection."""
        self.mab.add_arm(agent_id)
        self.q_learning.register_agent(agent_id)

    def select_agent(self, state: State, strategy: str = "mab") -> AgentID:
        """
        Select next agent to execute.

        Args:
            state: Current system state
            strategy: Selection strategy ('mab', 'q_learning', 'greedy')

        Returns:
            Selected agent ID
        """
        if strategy == "mab":
            return self.mab.select_arm()
        elif strategy == "q_learning":
            return self.q_learning.select_action(state)
        elif strategy == "greedy":
            return self.mab.get_best_arm()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def allocate_resources(
        self, agent_id: AgentID, agent_type: str
    ) -> ResourceAllocation | None:
        """
        Allocate resources based on agent type.

        Args:
            agent_id: Agent to allocate to
            agent_type: Type of agent (determines resource requirements)

        Returns:
            Resource allocation or None
        """
        # Define resource profiles for different agent types
        profiles = {
            "research": {"cpu": 2, "memory": 2048, "gpu": 0.0},
            "verification": {"cpu": 1, "memory": 1024, "gpu": 0.0},
            "synthesis": {"cpu": 4, "memory": 4096, "gpu": 0.25},
            "neural": {"cpu": 2, "memory": 4096, "gpu": 0.5},
        }

        profile = profiles.get(agent_type, profiles["research"])

        return self.resource_allocator.allocate(
            agent_id=agent_id,
            cpu_cores=profile["cpu"],
            memory_mb=profile["memory"],
            gpu_allocation=profile["gpu"],
        )

    def update_performance(
        self,
        agent_id: AgentID,
        state: State,
        next_state: State,
        execution_time: float,
        success: bool,
    ) -> None:
        """
        Update learned models based on agent performance.

        Args:
            agent_id: Agent that executed
            state: State before execution
            next_state: State after execution
            execution_time: Time taken
            success: Whether execution succeeded
        """
        # Compute reward
        # Reward based on: success, information gain, time efficiency
        reward = 0.0

        if success:
            reward += 10.0

            # Reward for information gain
            info_gain = (
                next_state.n_propositions - state.n_propositions
            ) / max(state.n_propositions, 1)
            reward += info_gain * 5.0

            # Penalty for time (encourage efficiency)
            time_penalty = min(execution_time / 60.0, 5.0)  # Up to 5 points
            reward -= time_penalty

            # Reward for reducing uncertainty
            uncertainty_reduction = (
                state.epistemic_uncertainty - next_state.epistemic_uncertainty
            )
            reward += uncertainty_reduction * 10.0

        # Update MAB
        self.mab.update(agent_id, reward)

        # Update Q-learning
        self.q_learning.update(state, agent_id, reward, next_state)

        # Track history
        self.performance_history.append({
            "agent_id": agent_id,
            "reward": reward,
            "execution_time": execution_time,
            "success": success,
        })

        self.logger.info(
            "performance_updated",
            agent_id=agent_id,
            reward=reward,
            success=success,
        )

    def release_resources(self, agent_id: AgentID) -> None:
        """Release resources allocated to an agent."""
        self.resource_allocator.deallocate(agent_id)


__all__ = [
    "AdaptiveResourceAllocator",
    "MultiArmedBandit",
    "QLearningOptimizer",
    "ResourceAllocator",
    "State",
    "Action",
    "ResourceAllocation",
]
