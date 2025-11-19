"""
Basic example of using StrataFlow for research synthesis.

Demonstrates the complete workflow from request to results.
"""

import asyncio

from strataflow import StrataFlowEngine
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
from strataflow.core.types import ResearchPhase, ResearchRequest


async def main() -> None:
    """Run basic research example."""
    print("=== StrataFlow v2.0: Basic Research Example ===\n")

    # Create engine
    engine = StrataFlowEngine()

    # Register all agents
    print("Registering agents...")

    # Research agents
    engine.register_agent(ResearchPhase.EPISTEMIC_MAPPING, EpistemologicalCartographer())
    engine.register_agent(ResearchPhase.CAUSAL_DISCOVERY, CausalArchaeologist())
    engine.register_agent(ResearchPhase.KNOWLEDGE_CONSTRUCTION, SemanticWeaver())
    engine.register_agent(ResearchPhase.DIALECTICAL_SYNTHESIS, DialecticalSynthesizer())

    # Verification agents
    engine.register_agent(ResearchPhase.PROPOSITIONAL_ATOMIZATION, PropositionalAtomizer())
    engine.register_agent(ResearchPhase.ENTAILMENT_VALIDATION, EntailmentValidator())
    engine.register_agent(ResearchPhase.SOURCE_AUTHENTICATION, SourceAuthenticator())

    # Synthesis agents
    engine.register_agent(ResearchPhase.NARRATIVE_CONSTRUCTION, NarrativeArchitect())
    engine.register_agent(ResearchPhase.LINGUISTIC_RENDERING, LinguisticRenderer())

    print(f"Registered {len(engine.registered_agents)} agents\n")

    # Create research request
    request = ResearchRequest(
        topic="Impact of AI on Scientific Discovery",
        depth="comprehensive",
        output_formats=["technical_paper", "executive_summary"],
        verification_level="audit_grade",
    )

    print(f"Research Topic: {request.topic}")
    print(f"Depth: {request.depth}")
    print(f"Verification Level: {request.verification_level}\n")

    # Execute research
    print("Executing research pipeline...")
    print("(This is a demonstration - agents will execute with simplified logic)\n")

    result = await engine.execute_research(request)

    # Display results
    print("\n=== Research Complete ===\n")

    if result["success"]:
        print("✓ Research completed successfully\n")

        # Metrics
        metrics = result.get("metrics", {})
        print("Metrics:")
        print(f"  • Knowledge Graph Nodes: {metrics.get('knowledge_graph_nodes', 0)}")
        print(f"  • Knowledge Graph Edges: {metrics.get('knowledge_graph_edges', 0)}")
        print(f"  • Total Facts: {metrics.get('total_facts', 0)}")
        print(f"  • Conservative Inferences: {metrics.get('conservative_inferences', 0)}")
        print(f"  • Logical Consistency: {metrics.get('logical_consistency', 0):.1%}")
        print(f"  • Semantic Coherence: {metrics.get('semantic_coherence', 0):.1%}")
        print()

        # Outputs
        outputs = result.get("outputs", {})
        if outputs:
            print("Generated Outputs:")
            for format_name, content in outputs.items():
                word_count = len(content.split())
                print(f"  • {format_name}: {word_count} words")
                print(f"    Preview: {content[:200]}...")
                print()

    else:
        print(f"✗ Research failed: {result.get('error')}\n")

    print("=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
