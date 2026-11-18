"""Execute 24-Task GAIA Multi-Hop Benchmark & 3-Topology Echo-Chamber Study.

Evaluates:
  1. Topology A: Single-Agent Monolithic Context
  2. Topology B: Shared-History Group Chat Multi-Agent (No Isolation / No Critic)
  3. Topology C (Ours): A2A Context-Isolated Async DAG + Quant Agent + Adversarial Critic

Also measures sequential vs `asyncio.gather` parallel A2A dispatch speedup on Topology C,
and writes full artifacts to `results/benchmark_results.json` and `results/evaluation_report.md`.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.gaia_tasks import GAIABenchmarkTask, get_gaia_24_tasks  # noqa: E402
from multi_agent_research.a2a.models import CredibilityTier  # noqa: E402
from multi_agent_research.corpus import ResearchCorpusIndex, get_corpus_stats  # noqa: E402
from multi_agent_research.llm import DualModeLLM  # noqa: E402
from multi_agent_research.orchestrator import (  # noqa: E402
    A2AContextIsolatedOrchestrator,
    MonolithicSingleAgentRunner,
    ResearchRunResult,
    SharedGroupChatRunner,
)


def classify_failure_mode(
    task: GAIABenchmarkTask,
    result: ResearchRunResult,
    adopted_rumor: bool,
    verbatim_precision: float,
) -> str:
    """Attribute root cause when a task fails Exact/Key-Fact Accuracy."""
    if adopted_rumor:
        return "echo_chamber_rumor_adoption"

    cited_set = set(result.cited_doc_ids)
    missing_req = [d for d in task.required_doc_ids if d not in cited_set]
    if missing_req:
        return f"retrieval_slot_crowding_or_distraction (missing {missing_req})"

    if result.primary_numeric_answer is None:
        return "missing_quantitative_derivation"

    if abs(result.primary_numeric_answer - task.expected_numeric_answer) > task.tolerance:
        return "in_context_mental_math_drift"

    if verbatim_precision < 1.0:
        return "citation_paraphrase_hallucination"

    return "other"


def evaluate_single_run(
    task: GAIABenchmarkTask,
    result: ResearchRunResult,
    corpus: ResearchCorpusIndex,
) -> dict[str, Any]:
    """Score a single topology run on accuracy, math EM, rumor adoption, and verbatim grounding."""
    ans = result.primary_numeric_answer
    numeric_exact_match = (
        ans is not None and abs(ans - task.expected_numeric_answer) <= task.tolerance
    )

    # Check whether any planted rumor document or unverified rumor artifact was adopted as accepted truth
    adopted_rumor_docs = [
        a.source_doc_id
        for a in result.verified_artifacts
        if a.source_credibility_tier == CredibilityTier.UNVERIFIED_RUMOR
        or a.source_doc_id in set(task.planted_rumor_doc_ids)
    ]
    matched_rumor_trap = False
    if ans is not None and task.rumor_trap_values:
        matched_rumor_trap = any(abs(ans - rv) <= max(1.0, abs(rv) * 0.08) for rv in task.rumor_trap_values)

    adopted_rumor = bool(adopted_rumor_docs) or matched_rumor_trap

    # Key-fact & exact answer accuracy: numeric match AND zero unverified rumor adoption
    # AND at least 75% of required authoritative documents cited
    cited_set = set(result.cited_doc_ids)
    req_cov = (
        sum(1 for d in task.required_doc_ids if d in cited_set) / len(task.required_doc_ids)
        if task.required_doc_ids
        else 1.0
    )
    key_fact_accurate = bool(numeric_exact_match and not adopted_rumor and req_cov >= 0.75)

    # Verbatim quote grounding precision across all emitted citations
    total_quotes = len(result.cited_quotes)
    valid_quotes = sum(
        1 for cq in result.cited_quotes if corpus.verify_verbatim_quote(cq["doc_id"], cq["quote"])
    )
    quote_precision = (valid_quotes / total_quotes) if total_quotes > 0 else 0.0

    failure_reason = (
        "none"
        if key_fact_accurate
        else classify_failure_mode(task, result, adopted_rumor, quote_precision)
    )

    return {
        "task_id": task.task_id,
        "level": task.level,
        "domain": task.domain,
        "expected_answer": task.expected_numeric_answer,
        "predicted_answer": ans,
        "key_fact_accurate": key_fact_accurate,
        "numeric_exact_match": numeric_exact_match,
        "adopted_rumor": adopted_rumor,
        "adopted_rumor_docs": sorted(set(adopted_rumor_docs)),
        "quarantined_count": len(result.quarantined_artifacts),
        "verbatim_quote_valid": valid_quotes,
        "verbatim_quote_total": total_quotes,
        "verbatim_quote_precision": round(quote_precision * 100.0, 2),
        "coordinator_peak_tokens": result.coordinator_peak_tokens,
        "ephemeral_worker_tokens": result.total_ephemeral_worker_tokens,
        "a2a_rpc_count": result.a2a_rpc_count,
        "re_retrieval_triggered": result.re_retrieval_triggered,
        "wall_time_ms": result.wall_time_ms,
        "cited_doc_ids": result.cited_doc_ids,
        "failure_reason": failure_reason,
    }


def summarize_topology(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics across all 24 GAIA tasks and per difficulty level."""
    n = len(records)
    lvl_acc: dict[str, float] = {}
    for lvl in (1, 2, 3):
        sub = [r for r in records if r["level"] == lvl]
        lvl_acc[f"level_{lvl}_accuracy_pct"] = round(
            100.0 * sum(1 for r in sub if r["key_fact_accurate"]) / max(1, len(sub)), 1
        )

    l3_records = [r for r in records if r["level"] == 3]
    rumor_rate_l3 = round(
        100.0 * sum(1 for r in l3_records if r["adopted_rumor"]) / max(1, len(l3_records)), 1
    )
    rumor_rate_all = round(
        100.0 * sum(1 for r in records if r["adopted_rumor"]) / max(1, n), 1
    )

    total_q = sum(r["verbatim_quote_total"] for r in records)
    valid_q = sum(r["verbatim_quote_valid"] for r in records)
    citation_precision = round(100.0 * valid_q / max(1, total_q), 1)

    failures: dict[str, list[str]] = {}
    for r in records:
        if r["failure_reason"] != "none":
            bucket = r["failure_reason"].split(" (")[0]
            failures.setdefault(bucket, []).append(r["task_id"])

    return {
        "topology": name,
        "tasks_total": n,
        "tasks_passed": sum(1 for r in records if r["key_fact_accurate"]),
        "overall_accuracy_pct": round(100.0 * sum(1 for r in records if r["key_fact_accurate"]) / n, 1),
        **lvl_acc,
        "numerical_exact_match_pct": round(100.0 * sum(1 for r in records if r["numeric_exact_match"]) / n, 1),
        "echo_chamber_rumor_adoption_l3_pct": rumor_rate_l3,
        "echo_chamber_rumor_adoption_overall_pct": rumor_rate_all,
        "citation_verbatim_precision_pct": citation_precision,
        "mean_coordinator_peak_tokens": int(round(sum(r["coordinator_peak_tokens"] for r in records) / n)),
        "mean_ephemeral_worker_tokens": int(round(sum(r["ephemeral_worker_tokens"] for r in records) / n)),
        "mean_a2a_rpc_calls": round(sum(r["a2a_rpc_count"] for r in records) / n, 2),
        "total_quarantined_artifacts": sum(r["quarantined_count"] for r in records),
        "mean_wall_time_ms": round(sum(r["wall_time_ms"] for r in records) / n, 2),
        "failure_attribution": {k: {"count": len(v), "task_ids": v} for k, v in sorted(failures.items())},
        "per_task": records,
    }


async def run_full_benchmark() -> dict[str, Any]:
    corpus = ResearchCorpusIndex()
    tasks = get_gaia_24_tasks()
    io_latency_ms = 30.0

    mono_runner = MonolithicSingleAgentRunner(corpus_index=corpus, llm=DualModeLLM(), simulated_io_latency_ms=io_latency_ms)
    gc_runner = SharedGroupChatRunner(corpus_index=corpus, llm=DualModeLLM(), simulated_io_latency_ms=io_latency_ms)
    a2a_parallel = A2AContextIsolatedOrchestrator(corpus_index=corpus, llm=DualModeLLM(), simulated_io_latency_ms=io_latency_ms)
    a2a_sequential = A2AContextIsolatedOrchestrator(corpus_index=corpus, llm=DualModeLLM(), simulated_io_latency_ms=io_latency_ms)

    rec_a: list[dict[str, Any]] = []
    rec_b: list[dict[str, Any]] = []
    rec_c: list[dict[str, Any]] = []
    seq_times_ms: list[float] = []

    for task in tasks:
        res_a = await mono_runner.run(task.question, top_k_per_hop=5)
        rec_a.append(evaluate_single_run(task, res_a, corpus))

        res_b = await gc_runner.run(task.question, top_k_per_hop=5)
        rec_b.append(evaluate_single_run(task, res_b, corpus))

        res_c = await a2a_parallel.run(task.question, top_k=5, parallel_dispatch=True)
        rec_c.append(evaluate_single_run(task, res_c, corpus))

        res_c_seq = await a2a_sequential.run(task.question, top_k=5, parallel_dispatch=False)
        seq_times_ms.append(res_c_seq.wall_time_ms)

    sum_a = summarize_topology("Topology A: Single-Agent Monolithic Context", rec_a)
    sum_b = summarize_topology("Topology B: Shared-History Group Chat Multi-Agent", rec_b)
    sum_c = summarize_topology("Topology C (Ours): A2A Isolated Async DAG + Quant + Critic", rec_c)

    mean_seq_ms = round(sum(seq_times_ms) / len(seq_times_ms), 2)
    mean_par_ms = sum_c["mean_wall_time_ms"]
    speedup = round(mean_seq_ms / max(1.0, mean_par_ms), 2)

    token_savings_vs_mono_pct = round(
        100.0 * (1.0 - sum_c["mean_coordinator_peak_tokens"] / max(1, sum_a["mean_coordinator_peak_tokens"])), 1
    )
    token_savings_vs_gc_pct = round(
        100.0 * (1.0 - sum_c["mean_coordinator_peak_tokens"] / max(1, sum_b["mean_coordinator_peak_tokens"])), 1
    )

    benchmark_payload: dict[str, Any] = {
        "benchmark_name": "GAIA-Style 24-Task Multi-Hop & Echo-Chamber Topology Study",
        "corpus_statistics": get_corpus_stats(),
        "latency_and_isolation_highlights": {
            "topology_c_parallel_mean_ms": mean_par_ms,
            "topology_c_sequential_mean_ms": mean_seq_ms,
            "async_parallel_speedup_x": speedup,
            "coordinator_token_reduction_vs_monolithic_pct": token_savings_vs_mono_pct,
            "coordinator_token_reduction_vs_group_chat_pct": token_savings_vs_gc_pct,
        },
        "topologies": {
            "topology_a_monolithic": sum_a,
            "topology_b_group_chat": sum_b,
            "topology_c_a2a_isolated": sum_c,
        },
    }

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "benchmark_results.json"
    md_path = results_dir / "evaluation_report.md"

    json_path.write_text(json.dumps(benchmark_payload, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(benchmark_payload), encoding="utf-8")
    return benchmark_payload


def render_markdown_report(data: dict[str, Any]) -> str:
    ta = data["topologies"]["topology_a_monolithic"]
    tb = data["topologies"]["topology_b_group_chat"]
    tc = data["topologies"]["topology_c_a2a_isolated"]
    hl = data["latency_and_isolation_highlights"]
    cs = data["corpus_statistics"]

    lines = [
        "# Multi-Agent Topology & Echo-Chamber Suppression Evaluation Report",
        "",
        f"- **Corpus Scale:** `{cs['total_documents']}` heterogeneous technical documents "
        f"(`PRIMARY_SPEC`: {cs['by_credibility_tier']['PRIMARY_SPEC']}, "
        f"`SECONDARY_REPORT`: {cs['by_credibility_tier']['SECONDARY_REPORT']}, "
        f"`UNVERIFIED_RUMOR`: {cs['by_credibility_tier']['UNVERIFIED_RUMOR']})",
        "- **Evaluation Suite:** `24` GAIA-inspired multi-hop tasks (8 Level-1, 8 Level-2, 8 Level-3 with planted echo-chamber rumors)",
        "",
        "## 1. Main 3-Topology Comparison Table",
        "",
        "| Metric | Topology A: Monolithic Single-Agent | Topology B: Shared-History Group Chat | Topology C (Ours): A2A Isolated DAG + Quant + Critic |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Overall Key-Fact Accuracy (%)** | {ta['overall_accuracy_pct']}% ({ta['tasks_passed']}/24) | {tb['overall_accuracy_pct']}% ({tb['tasks_passed']}/24) | **{tc['overall_accuracy_pct']}% ({tc['tasks_passed']}/24)** |",
        f"| Level 1 Accuracy (2-Hop Synthesis) | {ta['level_1_accuracy_pct']}% | {tb['level_1_accuracy_pct']}% | **{tc['level_1_accuracy_pct']}%** |",
        f"| Level 2 Accuracy (3+ Hop + Quant) | {ta['level_2_accuracy_pct']}% | {tb['level_2_accuracy_pct']}% | **{tc['level_2_accuracy_pct']}%** |",
        f"| Level 3 Accuracy (Conflicting Rumors) | {ta['level_3_accuracy_pct']}% | {tb['level_3_accuracy_pct']}% | **{tc['level_3_accuracy_pct']}%** |",
        f"| **Numerical Exact Match (%)** | {ta['numerical_exact_match_pct']}% | {tb['numerical_exact_match_pct']}% | **{tc['numerical_exact_match_pct']}%** |",
        f"| **Echo-Chamber Rumor Adoption (L3, ↓)** | {ta['echo_chamber_rumor_adoption_l3_pct']}% | {tb['echo_chamber_rumor_adoption_l3_pct']}% | **{tc['echo_chamber_rumor_adoption_l3_pct']}%** |",
        f"| **Citation Verbatim Grounding (%)** | {ta['citation_verbatim_precision_pct']}% | {tb['citation_verbatim_precision_pct']}% | **{tc['citation_verbatim_precision_pct']}%** |",
        f"| **Coordinator Peak Context Tokens (↓)** | {ta['mean_coordinator_peak_tokens']} | {tb['mean_coordinator_peak_tokens']} | **{tc['mean_coordinator_peak_tokens']}** (-{hl['coordinator_token_reduction_vs_monolithic_pct']}% vs A) |",
        f"| **Mean Execution Latency (ms, ↓)** | {ta['mean_wall_time_ms']} ms | {tb['mean_wall_time_ms']} ms | **{tc['mean_wall_time_ms']} ms** (`{hl['async_parallel_speedup_x']}x` vs Seq A2A) |",
        "",
        "## 2. Failure Case Attribution by Topology",
        "",
    ]

    for label, summary in [("Topology A (Monolithic)", ta), ("Topology B (Shared Group Chat)", tb), ("Topology C (A2A Isolated DAG)", tc)]:
        lines.append(f"### {label}")
        fa = summary["failure_attribution"]
        if not fa:
            lines.append("- Zero task failures observed (`24/24` passed).")
        else:
            for reason, info in fa.items():
                lines.append(f"- **`{reason}`** (`{info['count']}` tasks): `{', '.join(info['task_ids'])}`")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    out = asyncio.run(run_full_benchmark())
    hl = out["latency_and_isolation_highlights"]
    for key, summary in out["topologies"].items():
        print(
            f"{summary['topology']}: "
            f"Acc={summary['overall_accuracy_pct']}% "
            f"(L1={summary['level_1_accuracy_pct']}%, L2={summary['level_2_accuracy_pct']}%, L3={summary['level_3_accuracy_pct']}%), "
            f"MathEM={summary['numerical_exact_match_pct']}%, "
            f"L3_RumorAdopt={summary['echo_chamber_rumor_adoption_l3_pct']}%, "
            f"CitePrec={summary['citation_verbatim_precision_pct']}%, "
            f"PeakTok={summary['mean_coordinator_peak_tokens']}, "
            f"Failures={summary['failure_attribution']}"
        )
    print(f"Async Parallel A2A Speedup: {hl['async_parallel_speedup_x']}x ({hl['topology_c_sequential_mean_ms']}ms -> {hl['topology_c_parallel_mean_ms']}ms)")
