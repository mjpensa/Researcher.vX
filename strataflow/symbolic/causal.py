"""
Causal Discovery Module implementing Pearl's causal inference framework.

Implements:
- PC Algorithm for causal structure learning
- Structural Equation Modeling (SEM)
- Do-calculus for interventional reasoning
- Counterfactual inference
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable

import networkx as nx
import numpy as np
import structlog
from numpy.typing import NDArray
from scipy import stats

logger = structlog.get_logger()


# ============================================================================
# Causal Graph Structures
# ============================================================================


@dataclass
class CausalEdge:
    """Directed causal edge from cause to effect."""

    cause: str
    effect: str
    strength: float  # Causal effect size
    confidence: float  # Statistical confidence
    mechanism: str | None = None  # Causal mechanism description


@dataclass
class StructuralEquation:
    """Structural equation: Y := f(Parents(Y), U_Y)."""

    variable: str
    parents: list[str]
    function: Callable[[dict[str, float]], float]
    noise_distribution: str = "normal"  # Type of exogenous noise
    noise_params: dict[str, float] = field(default_factory=dict)


class StructuralCausalModel:
    """
    Structural Causal Model (SCM) as defined by Judea Pearl.

    An SCM consists of:
    1. Endogenous variables V
    2. Exogenous variables U
    3. Structural equations F mapping (V, U) -> V
    """

    def __init__(self) -> None:
        self.equations: dict[str, StructuralEquation] = {}
        self.graph = nx.DiGraph()
        self.exogenous: dict[str, dict[str, float]] = {}
        self.logger = logger.bind(component="StructuralCausalModel")

    def add_equation(self, equation: StructuralEquation) -> None:
        """Add a structural equation to the model."""
        self.equations[equation.variable] = equation

        # Update causal graph
        for parent in equation.parents:
            self.graph.add_edge(parent, equation.variable)

    def intervene(self, interventions: dict[str, float]) -> StructuralCausalModel:
        """
        Perform Pearl's do-operator intervention.

        do(X=x) removes all incoming edges to X and fixes X=x.

        Returns:
            New SCM with interventions applied
        """
        intervened_model = StructuralCausalModel()

        for var, eq in self.equations.items():
            if var in interventions:
                # Replace equation with constant function
                new_eq = StructuralEquation(
                    variable=var,
                    parents=[],
                    function=lambda v, val=interventions[var]: val,
                )
                intervened_model.add_equation(new_eq)
            else:
                intervened_model.add_equation(eq)

        intervened_model.exogenous = self.exogenous.copy()
        return intervened_model

    def sample(self, n_samples: int = 1000) -> dict[str, NDArray[np.float64]]:
        """
        Sample from the SCM using ancestral sampling.

        Returns:
            Dictionary mapping variable names to arrays of samples
        """
        # Topological sort for ancestral sampling
        try:
            topo_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            self.logger.error("causal_graph_has_cycle")
            raise ValueError("Causal graph must be acyclic (DAG)")

        samples: dict[str, list[float]] = {var: [] for var in self.equations.keys()}

        for _ in range(n_samples):
            current_values: dict[str, float] = {}

            for var in topo_order:
                if var in self.equations:
                    eq = self.equations[var]
                    # Sample exogenous noise
                    if eq.noise_distribution == "normal":
                        noise = np.random.normal(
                            eq.noise_params.get("mean", 0),
                            eq.noise_params.get("std", 1),
                        )
                    else:
                        noise = 0

                    # Evaluate structural function
                    parent_values = {p: current_values.get(p, 0) for p in eq.parents}
                    value = eq.function(parent_values) + noise
                    current_values[var] = value
                    samples[var].append(value)

        return {var: np.array(vals) for var, vals in samples.items()}

    def compute_causal_effect(
        self, cause: str, effect: str, intervention_value: float, n_samples: int = 10000
    ) -> float:
        """
        Compute average causal effect: E[Y | do(X=x)].

        Args:
            cause: Intervention variable
            effect: Outcome variable
            intervention_value: Value to set cause to
            n_samples: Number of samples for Monte Carlo estimation

        Returns:
            Expected value of effect under intervention
        """
        intervened_model = self.intervene({cause: intervention_value})
        samples = intervened_model.sample(n_samples=n_samples)

        if effect not in samples:
            raise ValueError(f"Effect variable {effect} not found in model")

        return float(np.mean(samples[effect]))


# ============================================================================
# PC Algorithm for Causal Discovery
# ============================================================================


class PCAlgorithm:
    """
    PC Algorithm (Peter-Clark) for learning causal DAGs from observational data.

    The algorithm uses conditional independence tests to discover the causal structure.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        """
        Initialize PC algorithm.

        Args:
            alpha: Significance level for independence tests
        """
        self.alpha = alpha
        self.logger = logger.bind(component="PCAlgorithm")

    def learn_structure(
        self, data: dict[str, NDArray[np.float64]]
    ) -> tuple[nx.DiGraph, list[CausalEdge]]:
        """
        Learn causal structure from observational data.

        Args:
            data: Dictionary mapping variable names to observations

        Returns:
            Tuple of (DAG, list of causal edges with metadata)
        """
        variables = list(data.keys())
        n_vars = len(variables)

        # Phase 1: Build complete undirected graph
        graph = nx.Graph()
        graph.add_nodes_from(variables)
        for i, j in combinations(range(n_vars), 2):
            graph.add_edge(variables[i], variables[j])

        # Phase 2: Remove edges based on conditional independence
        graph = self._remove_edges(graph, data, variables)

        # Phase 3: Orient edges
        dag = self._orient_edges(graph, data, variables)

        # Compute edge strengths
        edges = self._compute_edge_strengths(dag, data)

        return dag, edges

    def _remove_edges(
        self,
        graph: nx.Graph,
        data: dict[str, NDArray[np.float64]],
        variables: list[str],
    ) -> nx.Graph:
        """Remove edges based on conditional independence tests."""
        # Test independence conditioning on sets of increasing size
        for cond_set_size in range(len(variables)):
            edges_to_remove = []

            for edge in list(graph.edges()):
                x, y = edge
                # Get potential conditioning sets (neighbors excluding x and y)
                neighbors = set(graph.neighbors(x)) | set(graph.neighbors(y))
                neighbors.discard(x)
                neighbors.discard(y)

                if len(neighbors) < cond_set_size:
                    continue

                # Test independence conditioning on all subsets of given size
                for cond_set in combinations(neighbors, cond_set_size):
                    if self._test_independence(x, y, list(cond_set), data):
                        edges_to_remove.append((x, y))
                        break

            for edge in edges_to_remove:
                if graph.has_edge(*edge):
                    graph.remove_edge(*edge)

        return graph

    def _test_independence(
        self,
        x: str,
        y: str,
        cond_set: list[str],
        data: dict[str, NDArray[np.float64]],
    ) -> bool:
        """
        Test conditional independence X ⊥ Y | Z using partial correlation.

        Returns:
            True if X and Y are independent given Z
        """
        if len(cond_set) == 0:
            # Unconditional independence - use simple correlation
            corr, p_value = stats.pearsonr(data[x], data[y])
            return p_value > self.alpha

        # Partial correlation using regression residuals
        x_data = data[x]
        y_data = data[y]
        z_data = np.column_stack([data[z] for z in cond_set])

        # Regress X on Z
        x_resid = self._compute_residuals(x_data, z_data)

        # Regress Y on Z
        y_resid = self._compute_residuals(y_data, z_data)

        # Test correlation of residuals
        if len(x_resid) > 2:
            corr, p_value = stats.pearsonr(x_resid, y_resid)
            return p_value > self.alpha

        return False

    def _compute_residuals(
        self, y: NDArray[np.float64], X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute regression residuals."""
        # Simple linear regression
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        # Solve normal equations
        try:
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            y_pred = X_with_intercept @ beta
            return y - y_pred
        except np.linalg.LinAlgError:
            return y

    def _orient_edges(
        self, graph: nx.Graph, data: dict[str, NDArray[np.float64]], variables: list[str]
    ) -> nx.DiGraph:
        """
        Orient edges to create a DAG using orientation rules.

        Simplified version - in production, use full PC orientation rules.
        """
        dag = nx.DiGraph()

        # For simplicity, use time-based ordering if available
        # Or use heuristics based on variable names
        # In production: implement Meek's rules for proper orientation

        for edge in graph.edges():
            # Heuristic: orient based on correlation strength and variance
            x, y = edge
            corr_xy = np.corrcoef(data[x], data[y])[0, 1]

            # Simple heuristic: higher variance -> more likely to be cause
            if np.var(data[x]) > np.var(data[y]):
                dag.add_edge(x, y)
            else:
                dag.add_edge(y, x)

        # Ensure DAG (remove cycles if any)
        if not nx.is_directed_acyclic_graph(dag):
            # Remove edges to break cycles
            while not nx.is_directed_acyclic_graph(dag):
                try:
                    cycle = nx.find_cycle(dag)
                    dag.remove_edge(cycle[0][0], cycle[0][1])
                except nx.NetworkXNoCycle:
                    break

        return dag

    def _compute_edge_strengths(
        self, dag: nx.DiGraph, data: dict[str, NDArray[np.float64]]
    ) -> list[CausalEdge]:
        """Compute strength of causal edges."""
        edges = []

        for cause, effect in dag.edges():
            # Simple linear regression to estimate effect size
            X = data[cause].reshape(-1, 1)
            y = data[effect]

            # Fit linear model
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

            strength = float(beta[1])  # Coefficient for cause
            confidence = 0.95  # Simplified - should compute from p-value

            edges.append(
                CausalEdge(
                    cause=cause,
                    effect=effect,
                    strength=strength,
                    confidence=confidence,
                )
            )

        return edges


# ============================================================================
# Counterfactual Inference Engine
# ============================================================================


class CounterfactualEngine:
    """
    Counterfactual reasoning using the three-step process:
    1. Abduction: Infer exogenous variables from observations
    2. Action: Modify model with intervention
    3. Prediction: Compute counterfactual outcome
    """

    def __init__(self, scm: StructuralCausalModel) -> None:
        self.scm = scm
        self.logger = logger.bind(component="CounterfactualEngine")

    def compute_counterfactual(
        self,
        observation: dict[str, float],
        intervention: dict[str, float],
        query: str,
    ) -> float:
        """
        Compute counterfactual: "What would Y be if we had done X=x, given that we observed E=e?"

        Args:
            observation: Observed values {var: value}
            intervention: Counterfactual intervention {var: value}
            query: Variable to query

        Returns:
            Counterfactual value of query variable
        """
        # Step 1: Abduction - Infer exogenous variables
        exogenous = self._abduction(observation)

        # Step 2: Action - Apply intervention
        intervened_scm = self.scm.intervene(intervention)
        intervened_scm.exogenous = exogenous

        # Step 3: Prediction - Sample from intervened model
        samples = intervened_scm.sample(n_samples=1)

        return float(samples[query][0]) if query in samples else 0.0

    def _abduction(self, observation: dict[str, float]) -> dict[str, dict[str, float]]:
        """
        Infer exogenous variables from observations.

        Simplified version - in production, solve the inverse problem properly.
        """
        # For now, return empty exogenous variables
        # Full implementation would solve for U given V
        return {}


# ============================================================================
# Main Causal Discovery Engine
# ============================================================================


class CausalDiscoveryEngine:
    """
    Unified causal discovery engine combining multiple algorithms.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.pc_algorithm = PCAlgorithm(alpha=alpha)
        self.logger = logger.bind(component="CausalDiscoveryEngine")

    def discover_causal_structure(
        self, data: dict[str, NDArray[np.float64]]
    ) -> tuple[StructuralCausalModel, list[CausalEdge]]:
        """
        Discover causal structure from observational data.

        Returns:
            Tuple of (learned SCM, causal edges)
        """
        self.logger.info("causal_discovery_started", n_variables=len(data))

        # Run PC algorithm
        dag, edges = self.pc_algorithm.learn_structure(data)

        # Build SCM from discovered structure
        scm = self._build_scm_from_dag(dag, data)

        self.logger.info(
            "causal_discovery_completed",
            n_edges=len(edges),
            n_nodes=len(dag.nodes),
        )

        return scm, edges

    def _build_scm_from_dag(
        self, dag: nx.DiGraph, data: dict[str, NDArray[np.float64]]
    ) -> StructuralCausalModel:
        """Build SCM from discovered DAG by fitting structural equations."""
        scm = StructuralCausalModel()

        for node in dag.nodes():
            parents = list(dag.predecessors(node))

            if not parents:
                # Root node - just noise
                mean_val = float(np.mean(data[node]))
                std_val = float(np.std(data[node]))

                eq = StructuralEquation(
                    variable=node,
                    parents=[],
                    function=lambda v: 0.0,
                    noise_distribution="normal",
                    noise_params={"mean": mean_val, "std": std_val},
                )
            else:
                # Fit linear model
                X = np.column_stack([data[p] for p in parents])
                y = data[node]
                X_with_intercept = np.column_stack([np.ones(len(X)), X])
                coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

                # Create function
                def make_function(intercept: float, betas: NDArray[np.float64], parent_names: list[str]) -> Callable:
                    def func(parent_values: dict[str, float]) -> float:
                        result = intercept
                        for i, p in enumerate(parent_names):
                            result += betas[i] * parent_values.get(p, 0.0)
                        return result
                    return func

                func = make_function(coeffs[0], coeffs[1:], parents)

                eq = StructuralEquation(
                    variable=node,
                    parents=parents,
                    function=func,
                    noise_distribution="normal",
                    noise_params={"mean": 0.0, "std": 1.0},
                )

            scm.add_equation(eq)

        return scm


__all__ = [
    "StructuralCausalModel",
    "CausalEdge",
    "StructuralEquation",
    "PCAlgorithm",
    "CounterfactualEngine",
    "CausalDiscoveryEngine",
]
