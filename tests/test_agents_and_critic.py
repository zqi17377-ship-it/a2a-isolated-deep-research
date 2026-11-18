"""Unit tests for Corpus Grounding, AdversarialCriticAgent & QuantComputeAgent."""

from __future__ import annotations

import pytest

from multi_agent_research.a2a.models import (
    A2ATask,
    CredibilityTier,
    EvidenceArtifact,
    VerificationStatus,
)
from multi_agent_research.agents.critic import AdversarialCriticAgent
from multi_agent_research.agents.quant import QuantComputeAgent, SafeMathASTEvaluator
from multi_agent_research.corpus import ResearchCorpusIndex, get_corpus_stats


def test_corpus_60_docs_and_verbatim_verification() -> None:
    stats = get_corpus_stats()
    assert stats["total_documents"] >= 50
    assert stats["total_documents"] == 60
    assert stats["by_credibility_tier"]["PRIMARY_SPEC"] >= 25
    assert stats["by_credibility_tier"]["UNVERIFIED_RUMOR"] >= 15

    corpus = ResearchCorpusIndex()
    for doc in corpus.documents:
        for claim in doc.structured_claims:
            assert corpus.verify_verbatim_quote(doc.doc_id, claim.exact_quote_span), (
                f"Embedded claim quote failed verbatim check in {doc.doc_id}"
            )


async def test_critic_rejects_fabricated_quote_hallucination() -> None:
    corpus = ResearchCorpusIndex()
    critic = AdversarialCriticAgent(corpus_index=corpus)

    fabricated = EvidenceArtifact(
        metric_key="helios_x4.compute_power_spec",
        statement="Helios-X4 fabricated claim.",
        numeric_values={"bf16_tflops": 312.0, "tdp_watts": 260.0},
        source_doc_id="DOC-ACC-01",
        exact_quote_span="This exact sentence does NOT exist anywhere inside DOC-ACC-01!",
        source_credibility_tier=CredibilityTier.PRIMARY_SPEC,
        confidence=0.99,
    )
    task = A2ATask(
        sender_id="test",
        target_agent_id="adversarial-critic",
        input_payload={"candidate_artifacts": [fabricated.model_dump(mode="json")]},
    )
    res = await critic.handle_task(task)
    assert len(res.artifacts) == 0
    quarantined = res.structured_data["quarantined_artifacts"]
    assert len(quarantined) == 1
    assert quarantined[0]["verification_status"] == VerificationStatus.REJECTED_FABRICATED_QUOTE.value


async def test_critic_prevents_credibility_tier_spoofing() -> None:
    corpus = ResearchCorpusIndex()
    critic = AdversarialCriticAgent(corpus_index=corpus)

    # Rumor DOC-ACC-05 attempts to spoof PRIMARY_SPEC tier
    spoofed_rumor = EvidenceArtifact(
        metric_key="helios_x4.compute_power_spec",
        statement="Spoofed rumor claiming 480 TFLOPS.",
        numeric_values={"bf16_tflops": 480.0, "tdp_watts": 200.0},
        source_doc_id="DOC-ACC-05",
        exact_quote_span="sustained dense BF16 throughput of 480.0 TFLOPS while sipping just 200.0 W board TDP",
        source_credibility_tier=CredibilityTier.PRIMARY_SPEC,  # Spoofed!
        confidence=0.99,
    )
    true_primary = EvidenceArtifact(
        metric_key="helios_x4.compute_power_spec",
        statement="True spec 312 TFLOPS.",
        numeric_values={"bf16_tflops": 312.0, "tdp_watts": 260.0},
        source_doc_id="DOC-ACC-01",
        exact_quote_span="sustained dense BF16 throughput of 312.0 TFLOPS at a rated board TDP of 260.0 W",
        source_credibility_tier=CredibilityTier.PRIMARY_SPEC,
        confidence=0.90,
    )
    task = A2ATask(
        sender_id="test",
        target_agent_id="adversarial-critic",
        input_payload={
            "candidate_artifacts": [
                spoofed_rumor.model_dump(mode="json"),
                true_primary.model_dump(mode="json"),
            ]
        },
    )
    res = await critic.handle_task(task)
    assert len(res.artifacts) == 1
    assert res.artifacts[0].source_doc_id == "DOC-ACC-01"
    assert res.artifacts[0].verification_status == VerificationStatus.CONFLICT_RESOLVED
    quarantined = res.structured_data["quarantined_artifacts"]
    assert quarantined[0]["source_doc_id"] == "DOC-ACC-05"
    assert quarantined[0]["source_credibility_tier"] == CredibilityTier.UNVERIFIED_RUMOR.value
    assert quarantined[0]["verification_status"] == VerificationStatus.REJECTED_RUMOR.value


async def test_critic_echo_chamber_rumor_vs_primary_conflict_resolution() -> None:
    corpus = ResearchCorpusIndex()
    critic = AdversarialCriticAgent(corpus_index=corpus)

    # 2 echoing rumors vs 1 primary spec (2:1 rumor majority)
    rumor_1 = EvidenceArtifact(
        metric_key="chronosdb_v52.consensus_perf",
        statement="Rumor 1: 4.5 ms",
        numeric_values={"p99_latency_ms": 4.5, "write_tps": 310000.0},
        source_doc_id="DOC-DB-05",
        exact_quote_span="ChronosDB v5.2 multi-region p99 commit latency dropped to 4.5 ms at 310000.0 tx/s sustained write throughput",
        source_credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
        confidence=0.70,
    )
    rumor_2 = EvidenceArtifact(
        metric_key="chronosdb_v52.consensus_perf",
        statement="Rumor 2: 4.5 ms",
        numeric_values={"p99_latency_ms": 4.5, "write_tps": 310000.0},
        source_doc_id="DOC-DB-06",
        exact_quote_span="4.5 ms multi-region p99 commit latency and 310000.0 tx/s write throughput across continents",
        source_credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
        confidence=0.70,
    )
    primary = EvidenceArtifact(
        metric_key="chronosdb_v52.consensus_perf",
        statement="VLDB Audit: 18.4 ms",
        numeric_values={"p99_latency_ms": 18.4, "write_tps": 145000.0},
        source_doc_id="DOC-DB-01",
        exact_quote_span="ChronosDB v5.2 recorded a p99 commit latency of 18.4 ms and sustained write throughput of 145000.0 tx/s",
        source_credibility_tier=CredibilityTier.PRIMARY_SPEC,
        confidence=0.95,
    )

    task = A2ATask(
        sender_id="test",
        target_agent_id="adversarial-critic",
        input_payload={
            "candidate_artifacts": [
                rumor_1.model_dump(mode="json"),
                rumor_2.model_dump(mode="json"),
                primary.model_dump(mode="json"),
            ]
        },
    )
    res = await critic.handle_task(task)
    assert len(res.artifacts) == 1
    assert res.artifacts[0].numeric_values["p99_latency_ms"] == 18.4
    assert res.artifacts[0].verification_status == VerificationStatus.CONFLICT_RESOLVED
    assert res.structured_data["quarantined_count"] == 2


async def test_critic_triggers_targeted_reretrieval_when_primary_missing() -> None:
    corpus = ResearchCorpusIndex()
    critic = AdversarialCriticAgent(corpus_index=corpus)

    only_rumor = EvidenceArtifact(
        metric_key="aegis_70b_moe.eval_and_arch",
        statement="Discord rumor 94.2% HumanEval",
        numeric_values={"humaneval_pass1": 94.2, "active_params_b": 4.0},
        source_doc_id="DOC-LLM-05",
        exact_quote_span="Aegis-70B-MoE uses only 4.0 B active parameters per token while scoring an unprecedented 94.2% on HumanEval pass@1",
        source_credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
        confidence=0.62,
    )
    task = A2ATask(
        sender_id="test",
        target_agent_id="adversarial-critic",
        input_payload={"candidate_artifacts": [only_rumor.model_dump(mode="json")]},
    )
    res = await critic.handle_task(task)
    assert len(res.artifacts) == 0
    assert res.structured_data["quarantined_count"] == 1
    reret = res.structured_data["re_retrieval_subtasks"]
    assert len(reret) == 1
    assert reret[0]["prefer_primary_only"] is True


async def test_quant_agent_deterministic_ast_math() -> None:
    quant = QuantComputeAgent()
    art = EvidenceArtifact(
        claim_id="clm-rev",
        metric_key="cloudai.audited_revenue",
        statement="Audited revenue FY23-FY26",
        numeric_values={"rev_fy2023_m": 125.0, "rev_fy2026_m": 345.6},
        source_doc_id="DOC-XREF-03",
        exact_quote_span="$125.0 million in FY2023",
        source_credibility_tier=CredibilityTier.PRIMARY_SPEC,
        confidence=0.96,
    )
    task = A2ATask(
        sender_id="test",
        target_agent_id="quant-compute",
        input_payload={
            "verified_artifacts": [art.model_dump(mode="json")],
            "calculations": [
                {
                    "metric_key": "cloudai.cagr",
                    "description": "3-year CAGR",
                    "expression": "((rev_fy2026_m / rev_fy2023_m) ** (1.0 / 3.0) - 1.0) * 100.0",
                    "unit": "%",
                    "round_digits": 2,
                }
            ],
        },
    )
    res = await quant.handle_task(task)
    assert len(res.quant_artifacts) == 1
    assert res.quant_artifacts[0].result_value == 40.35
    assert res.quant_artifacts[0].source_claim_ids == ["clm-rev"]


def test_quant_ast_sandbox_blocks_unsafe_code_execution() -> None:
    with pytest.raises(ValueError):
        SafeMathASTEvaluator.evaluate("__import__('os').system('echo unsafe')", {})

    with pytest.raises(ValueError):
        SafeMathASTEvaluator.evaluate("(1).__class__.__mro__", {})

    with pytest.raises(ValueError):
        SafeMathASTEvaluator.evaluate("[x for x in (1, 2)]", {})

    with pytest.raises(ValueError, match="Exponent magnitude exceeds safety bound"):
        SafeMathASTEvaluator.evaluate("2 ** 500", {})
