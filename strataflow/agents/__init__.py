"""Agent implementations for StrataFlow."""

from strataflow.agents.research import (
    CausalArchaeologist,
    DialecticalSynthesizer,
    EpistemologicalCartographer,
    SemanticWeaver,
)
from strataflow.agents.synthesis import LinguisticRenderer, NarrativeArchitect
from strataflow.agents.verification import (
    EntailmentValidator,
    PropositionalAtomizer,
    SourceAuthenticator,
)

__all__ = [
    # Research Agents
    "EpistemologicalCartographer",  # α
    "CausalArchaeologist",  # β
    "SemanticWeaver",  # γ
    "DialecticalSynthesizer",  # δ
    # Verification Agents
    "PropositionalAtomizer",  # ε
    "EntailmentValidator",  # ζ
    "SourceAuthenticator",  # η
    # Synthesis Agents
    "NarrativeArchitect",  # θ
    "LinguisticRenderer",  # ι
]
