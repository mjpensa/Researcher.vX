"""
Verification Pipeline and Fact vs. Inference Loop.

Implements the core verification system as specified in the design plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from strataflow.core.state import ResearchState
from strataflow.core.types import (
    Citation,
    ClaimType,
    Proposition,
    PropositionID,
    ProofTree,
    SourceID,
    VerificationResult,
)
from strataflow.symbolic.verification import AutomatedTheoremProver

logger = structlog.get_logger()


# ============================================================================
# Fact-Inference Classification
# ============================================================================


@dataclass
class AtomVerification:
    """Verification result for an atomic proposition."""

    atom: Proposition
    classification: ClaimType
    evidence: list[Citation]
    confidence: float
    proof_tree: ProofTree | None = None


class FactInferenceVerifier:
    """
    Verifies claims and classifies them as facts vs. inferences.

    Implements the verification loop from the design plan:
    1. Decompose into atomic propositions
    2. Classify each atom (CITED_FACT, CONSERVATIVE_INFERENCE, etc.)
    3. Aggregate verification results
    """

    def __init__(
        self,
        conservative_threshold: float = 0.95,
        speculative_threshold: float = 0.75,
    ) -> None:
        """
        Initialize verifier.

        Args:
            conservative_threshold: Threshold for conservative inference
            speculative_threshold: Threshold for speculative inference
        """
        self.conservative_threshold = conservative_threshold
        self.speculative_threshold = speculative_threshold
        self.theorem_prover = AutomatedTheoremProver()
        self.logger = logger.bind(component="FactInferenceVerifier")

    async def verify_claim(
        self, claim: Proposition, state: ResearchState
    ) -> VerificationResult:
        """
        Verify a claim against the current research state.

        Implements the verification pipeline:
        1. Check for direct citation match
        2. Check entailment from cited facts
        3. Construct proof if possible
        4. Classify as CITED_FACT, CONSERVATIVE_INFERENCE, etc.

        Args:
            claim: Proposition to verify
            state: Current research state

        Returns:
            Verification result with classification
        """
        self.logger.info("claim_verification_started", claim=claim.text[:100])

        # Step 1: Check for direct citation match
        citation = self._find_exact_citation(claim, state)
        if citation:
            return VerificationResult(
                proposition_id=claim.id,
                claim_type=ClaimType.CITED_FACT,
                confidence=1.0,
                evidence=[citation],
                proof_tree=None,
                contradictions=[],
            )

        # Step 2: Check entailment from cited facts
        cited_facts = [
            state.propositions[pid]
            for pid in state.cited_facts
            if pid in state.propositions
        ]

        if cited_facts:
            entailment_score = await self._check_entailment(claim, cited_facts)

            # Step 3: Strong entailment - Conservative inference
            if entailment_score > self.conservative_threshold:
                proof_tree = await self._construct_proof(claim, cited_facts)

                return VerificationResult(
                    proposition_id=claim.id,
                    claim_type=ClaimType.CONSERVATIVE_INFERENCE,
                    confidence=entailment_score,
                    evidence=[],
                    proof_tree=proof_tree,
                    contradictions=[],
                )

            # Moderate entailment - Speculative inference
            elif entailment_score > self.speculative_threshold:
                return VerificationResult(
                    proposition_id=claim.id,
                    claim_type=ClaimType.SPECULATIVE_INFERENCE,
                    confidence=entailment_score,
                    evidence=[],
                    proof_tree=None,
                    contradictions=[],
                )

        # No support found - Unsupported
        return VerificationResult(
            proposition_id=claim.id,
            claim_type=ClaimType.UNSUPPORTED,
            confidence=0.0,
            evidence=[],
            proof_tree=None,
            contradictions=[],
        )

    def _find_exact_citation(
        self, claim: Proposition, state: ResearchState
    ) -> Citation | None:
        """Find exact citation match for claim."""
        # Simplified: text matching
        # In production: semantic similarity + exact span matching

        claim_text = claim.text.lower().strip()

        for source_id, citation in state.sources.items():
            if hasattr(citation, "excerpt"):
                if claim_text in citation.excerpt.lower():
                    return citation

        return None

    async def _check_entailment(
        self, hypothesis: Proposition, premises: list[Proposition]
    ) -> float:
        """
        Check if hypothesis is entailed by premises.

        Uses NLI model (simplified here).

        Returns:
            Entailment confidence score [0, 1]
        """
        # Simplified entailment checking
        # In production: use BERT-based NLI models

        hypothesis_words = set(hypothesis.text.lower().split())

        # Check overlap with premises
        max_overlap = 0.0

        for premise in premises:
            premise_words = set(premise.text.lower().split())
            overlap = len(hypothesis_words & premise_words) / max(
                len(hypothesis_words | premise_words), 1
            )
            max_overlap = max(max_overlap, overlap)

        return max_overlap

    async def _construct_proof(
        self, conclusion: Proposition, premises: list[Proposition]
    ) -> ProofTree | None:
        """
        Construct proof tree using automated theorem prover.

        Args:
            conclusion: Goal to prove
            premises: Available premises

        Returns:
            Proof tree if proof found
        """
        proof = self.theorem_prover.prove(
            goal=conclusion,
            axioms=premises,
            strategy="backward_chaining",
        )

        return proof

    async def batch_verify(
        self, claims: list[Proposition], state: ResearchState
    ) -> list[VerificationResult]:
        """
        Verify multiple claims in batch.

        Args:
            claims: List of claims to verify
            state: Research state

        Returns:
            List of verification results
        """
        results = []

        for claim in claims:
            result = await self.verify_claim(claim, state)
            results.append(result)

        # Log statistics
        classifications = [r.claim_type for r in results]
        stats = {
            "total": len(results),
            "cited_facts": classifications.count(ClaimType.CITED_FACT),
            "conservative": classifications.count(ClaimType.CONSERVATIVE_INFERENCE),
            "speculative": classifications.count(ClaimType.SPECULATIVE_INFERENCE),
            "unsupported": classifications.count(ClaimType.UNSUPPORTED),
        }

        self.logger.info("batch_verification_completed", **stats)

        return results


# ============================================================================
# Verification Pipeline
# ============================================================================


class VerificationPipeline:
    """
    Complete verification pipeline orchestrating all verification steps.

    Pipeline stages:
    1. Propositional atomization
    2. Fact vs. inference classification
    3. Entailment validation
    4. Source authentication
    5. Consistency checking
    """

    def __init__(self) -> None:
        """Initialize verification pipeline."""
        self.fact_verifier = FactInferenceVerifier()
        self.logger = logger.bind(component="VerificationPipeline")

    async def run_full_verification(
        self, state: ResearchState
    ) -> dict[str, Any]:
        """
        Run complete verification pipeline on research state.

        Args:
            state: Research state to verify

        Returns:
            Verification report with statistics
        """
        self.logger.info("full_verification_started")

        # Stage 1: Verify all propositions
        propositions = list(state.propositions.values())
        verification_results = await self.fact_verifier.batch_verify(
            propositions, state
        )

        # Stage 2: Check consistency
        consistency_result = await self._check_global_consistency(
            propositions, verification_results
        )

        # Stage 3: Validate source reliability
        source_reliability = self._validate_sources(state, verification_results)

        # Stage 4: Compute aggregate metrics
        metrics = self._compute_metrics(verification_results)

        # Build report
        report = {
            "verification_results": verification_results,
            "consistency": consistency_result,
            "source_reliability": source_reliability,
            "metrics": metrics,
            "passed": consistency_result["is_consistent"] and metrics["fact_precision"] > 0.7,
        }

        self.logger.info(
            "full_verification_completed",
            passed=report["passed"],
            fact_precision=metrics["fact_precision"],
        )

        return report

    async def _check_global_consistency(
        self,
        propositions: list[Proposition],
        verification_results: list[VerificationResult],
    ) -> dict[str, Any]:
        """Check global logical consistency."""
        from strataflow.symbolic.verification import SMTSolver

        smt_solver = SMTSolver()

        # Check satisfiability
        is_sat, model = smt_solver.check_satisfiability(propositions)

        # Find contradictions
        contradictions = []
        for result in verification_results:
            if result.contradictions:
                contradictions.extend(result.contradictions)

        return {
            "is_consistent": is_sat and len(contradictions) == 0,
            "contradictions": contradictions,
            "model": model,
        }

    def _validate_sources(
        self, state: ResearchState, verification_results: list[VerificationResult]
    ) -> dict[str, float]:
        """Validate source reliability across all verifications."""
        source_scores: dict[SourceID, list[float]] = {}

        # Aggregate confidence scores by source
        for result in verification_results:
            for evidence in result.evidence:
                if hasattr(evidence, "source_id"):
                    source_id = evidence.source_id
                    if source_id not in source_scores:
                        source_scores[source_id] = []
                    source_scores[source_id].append(result.confidence)

        # Compute average reliability
        avg_reliability = {
            source_id: sum(scores) / len(scores)
            for source_id, scores in source_scores.items()
        }

        return avg_reliability

    def _compute_metrics(
        self, verification_results: list[VerificationResult]
    ) -> dict[str, Any]:
        """Compute aggregate verification metrics."""
        total = len(verification_results)
        if total == 0:
            return {
                "fact_precision": 0.0,
                "conservative_ratio": 0.0,
                "speculative_ratio": 0.0,
                "unsupported_ratio": 0.0,
                "avg_confidence": 0.0,
            }

        classifications = [r.claim_type for r in verification_results]
        confidences = [r.confidence for r in verification_results]

        cited = classifications.count(ClaimType.CITED_FACT)
        conservative = classifications.count(ClaimType.CONSERVATIVE_INFERENCE)
        speculative = classifications.count(ClaimType.SPECULATIVE_INFERENCE)
        unsupported = classifications.count(ClaimType.UNSUPPORTED)

        # Fact precision: (cited + conservative) / total
        fact_precision = (cited + conservative) / total

        return {
            "total_claims": total,
            "cited_facts": cited,
            "conservative_inferences": conservative,
            "speculative_inferences": speculative,
            "unsupported_claims": unsupported,
            "fact_precision": fact_precision,
            "conservative_ratio": conservative / total,
            "speculative_ratio": speculative / total,
            "unsupported_ratio": unsupported / total,
            "avg_confidence": sum(confidences) / total,
        }


__all__ = [
    "FactInferenceVerifier",
    "VerificationPipeline",
    "AtomVerification",
]
