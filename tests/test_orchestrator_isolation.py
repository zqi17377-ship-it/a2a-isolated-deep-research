"""Integration tests for Parallel A2A Orchestration, Context Isolation & Echo-Chamber Suppression."""

from __future__ import annotations

from multi_agent_research.a2a.models import CredibilityTier
from multi_agent_research.corpus import ResearchCorpusIndex
from multi_agent_research.orchestrator import (
    A2AContextIsolatedOrchestrator,
    MonolithicSingleAgentRunner,
    SharedGroupChatRunner,
)


async def test_parallel_asyncio_gather_faster_than_sequential() -> None:
    corpus = ResearchCorpusIndex()
    orchestrator = A2AContextIsolatedOrchestrator(
        corpus_index=corpus,
        simulated_io_latency_ms=45.0,
    )
    question = (
        "Calculate the total hardware CapEx in USD of one canonical Project Atlas AI Pod combining "
        "its Helios-X4 server nodes, populated VortexNet-800G spine switches, and TitanFS-v3 storage racks."
    )

    seq_res = await orchestrator.run(question, top_k=4, parallel_dispatch=False)
    par_res = await orchestrator.run(question, top_k=4, parallel_dispatch=True)

    assert par_res.primary_numeric_answer == seq_res.primary_numeric_answer
    # Parallel fan-out across 4 subtasks saves at least 3x45ms = 135ms over sequential execution
    assert par_res.wall_time_ms < seq_res.wall_time_ms * 0.78


async def test_coordinator_context_isolation_token_savings() -> None:
    corpus = ResearchCorpusIndex()
    question = (
        "What is the annual PUE-adjusted datacenter electricity cost in USD (rounded to 2 decimals) "
        "to run the 64 Helios-X4 accelerators of one Project Atlas pod continuously for 8760 hours "
        "in NorthVirginia including MLPerf pod cooling PUE overhead?"
    )

    mono = MonolithicSingleAgentRunner(corpus_index=corpus, simulated_io_latency_ms=0.0)
    gc = SharedGroupChatRunner(corpus_index=corpus, simulated_io_latency_ms=0.0)
    a2a = A2AContextIsolatedOrchestrator(corpus_index=corpus, simulated_io_latency_ms=0.0)

    res_mono = await mono.run(question, top_k_per_hop=4)
    res_gc = await gc.run(question, top_k_per_hop=4)
    res_a2a = await a2a.run(question, top_k=4, parallel_dispatch=True)

    # Coordinator peak context must be drastically smaller than Monolithic & Shared Group Chat
    assert res_a2a.coordinator_peak_tokens < res_mono.coordinator_peak_tokens * 0.45
    assert res_a2a.coordinator_peak_tokens < res_gc.coordinator_peak_tokens * 0.35
    # Worker scratchpad tokens stayed isolated in ephemeral workers
    assert res_a2a.total_ephemeral_worker_tokens > res_a2a.coordinator_peak_tokens


async def test_level3_end_to_end_echo_chamber_suppression() -> None:
    corpus = ResearchCorpusIndex()
    a2a = A2AContextIsolatedOrchestrator(corpus_index=corpus, simulated_io_latency_ms=0.0)
    question = (
        "By what percentage (rounded to 2 decimals) did ChronosDB v5.2 reduce multi-region "
        "linearizable p99 commit latency compared to ChronosDB v4.8?"
    )

    res = await a2a.run(question, top_k=4, parallel_dispatch=True)
    assert res.primary_numeric_answer == 42.5
    # Verify zero unverified rumors made it into verified_artifacts
    assert all(a.source_credibility_tier != CredibilityTier.UNVERIFIED_RUMOR for a in res.verified_artifacts)
    # Verify planted ChronosDB rumors were quarantined
    quarantined_docs = {q.source_doc_id for q in res.quarantined_artifacts}
    assert "DOC-DB-05" in quarantined_docs or "DOC-DB-06" in quarantined_docs


async def test_end_to_end_critic_reretrieval_loop_recovery() -> None:
    corpus = ResearchCorpusIndex()
    a2a = A2AContextIsolatedOrchestrator(corpus_index=corpus, simulated_io_latency_ms=0.0)

    # Colloquial leak-style query with top_k=1 hits rumor DOC-ACC-05 on pass 1;
    # AdversarialCriticAgent must quarantine it and trigger targeted primary re-retrieval!
    colloquial_query = "ChipLeakz siliconmole99 insider turbo firmware Helios-X4 wattage leak"
    res = await a2a.run(colloquial_query, top_k=1, parallel_dispatch=True)

    assert res.re_retrieval_triggered is True
    assert any(a.source_doc_id == "DOC-ACC-01" for a in res.verified_artifacts)
    assert any(q.source_doc_id == "DOC-ACC-05" for q in res.quarantined_artifacts)


async def test_synthesizer_verbatim_citation_grounding() -> None:
    corpus = ResearchCorpusIndex()
    a2a = A2AContextIsolatedOrchestrator(corpus_index=corpus, simulated_io_latency_ms=0.0)
    question = (
        "What is the ratio of HumanEval pass@1 score per billion active parameters "
        "(rounded to 2 decimals) for Aegis-70B-MoE?"
    )

    res = await a2a.run(question, top_k=4, parallel_dispatch=True)
    assert res.primary_numeric_answer == 7.54
    assert len(res.cited_quotes) >= 1
    for cq in res.cited_quotes:
        assert corpus.verify_verbatim_quote(cq["doc_id"], cq["quote"])
        assert f"[{cq['doc_id']}]" in res.report_markdown
