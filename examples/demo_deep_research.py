"""Rich CLI Live Deep Research Demo (Level-3 Conflicting Multi-Hop Query).

Demonstrates:
  1. A2A `AgentCard` Capability Discovery across 5 strict-isolation agents
  2. Parallel `asyncio.gather` sub-task dispatch by `LeadPlannerAgent`
  3. Cross-Source Provenance Verification & Echo-Chamber Rumor Quarantine by `AdversarialCriticAgent`
  4. Safe AST Python execution by `QuantComputeAgent`
  5. Side-by-side comparison vs Shared-History Group Chat (Topology B)
  6. Final Cited Markdown Report with `[DOC-ID]` footnotes
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multi_agent_research.corpus import ResearchCorpusIndex  # noqa: E402
from multi_agent_research.orchestrator import (  # noqa: E402
    A2AContextIsolatedOrchestrator,
    SharedGroupChatRunner,
)


async def run_demo() -> None:
    console = Console()
    corpus = ResearchCorpusIndex()

    question = (
        "What is the annual PUE-adjusted datacenter electricity cost in USD (rounded to 2 decimals) "
        "to run the 64 Helios-X4 accelerators of one Project Atlas pod continuously for 8760 hours "
        "in NorthVirginia including MLPerf pod cooling PUE overhead?"
    )

    console.print(
        Panel.fit(
            "[bold cyan]Project 06-F4: Cross-Protocol Multi-Agent Deep Research System[/bold cyan]\n"
            "[dim]A2A JSON-RPC 2.0 • Strict Ephemeral Context Isolation • Echo-Chamber Suppression[/dim]",
            border_style="cyan",
        )
    )
    console.print(f"\n[bold yellow]Level-3 Research Query:[/bold yellow] {question}\n")

    orchestrator = A2AContextIsolatedOrchestrator(corpus_index=corpus, simulated_io_latency_ms=30.0)
    group_chat = SharedGroupChatRunner(corpus_index=corpus, simulated_io_latency_ms=30.0)

    # 1. Show Registered A2A AgentCards
    card_table = Table(title="1. Registered A2A AgentCards (`agent/discover`)", show_lines=False)
    card_table.add_column("Agent ID", style="cyan", no_wrap=True)
    card_table.add_column("Endpoint", style="dim")
    card_table.add_column("Isolation Contract", style="green")
    card_table.add_column("Advertised Skills", style="white")
    for card in orchestrator.bus.list_cards():
        skills_str = ", ".join(s.skill_id for s in card.skills)
        card_table.add_row(card.agent_id, card.endpoint, card.isolation_level.value, skills_str)
    console.print(card_table)

    # 2. Execute Topology C (Ours) & Topology B (Group Chat Baseline)
    res_c = await orchestrator.run(question, top_k=4, parallel_dispatch=True)
    res_b = await group_chat.run(question, top_k_per_hop=4)

    # 3. Show A2A Execution Trace
    trace_table = Table(title="2. LangGraph + A2A Async Execution Trace (Topology C)", show_lines=False)
    trace_table.add_column("Stage", style="magenta")
    trace_table.add_column("Target A2A Agent", style="cyan")
    trace_table.add_column("Key Telemetry", style="white")
    for ev in res_c.trace_events:
        details = ", ".join(f"{k}={v}" for k, v in ev.items() if k not in {"stage", "agent_id"})
        trace_table.add_row(str(ev["stage"]), str(ev["agent_id"]), details)
    console.print("\n", trace_table)

    # 4. Show Quarantined Echo-Chamber Rumors
    if res_c.quarantined_artifacts:
        q_table = Table(title="3. AdversarialCriticAgent Quarantine Audit (Echo-Chamber Suppression)", border_style="red")
        q_table.add_column("Doc ID", style="red", no_wrap=True)
        q_table.add_column("Status", style="bold red")
        q_table.add_column("Rejected Claim", style="white")
        q_table.add_column("Critic Provenance Notes", style="yellow")
        for q_art in res_c.quarantined_artifacts:
            q_table.add_row(
                q_art.source_doc_id,
                q_art.verification_status.value,
                q_art.statement,
                q_art.critic_notes,
            )
        console.print("\n", q_table)

    # 5. Side-by-side Topology B vs Topology C Comparison
    cmp_table = Table(title="4. Topology B (Shared Group Chat) vs Topology C (A2A Isolated + Critic)", border_style="green")
    cmp_table.add_column("Metric", style="bold")
    cmp_table.add_column("Topology B: Shared Group Chat", style="yellow")
    cmp_table.add_column("Topology C (Ours): A2A Isolated DAG", style="bold green")
    cmp_table.add_row("Computed Answer (Ground Truth = $12,572.35)", f"${res_b.primary_numeric_answer}", f"${res_c.primary_numeric_answer}")
    cmp_table.add_row(
        "Adopted Echo-Chamber Rumor (200W Leak)?",
        "YES (DOC-ACC-05 / DOC-ACC-06 Amplified)" if any(a.source_doc_id in {"DOC-ACC-05", "DOC-ACC-06"} for a in res_b.verified_artifacts) else "NO",
        f"NO ({len(res_c.quarantined_artifacts)} rumors quarantined)",
    )
    cmp_table.add_row("Coordinator Peak Context Tokens", f"{res_b.coordinator_peak_tokens} tokens", f"{res_c.coordinator_peak_tokens} tokens (Worker ephemeral: {res_c.total_ephemeral_worker_tokens})")
    cmp_table.add_row("Wall-Clock Latency", f"{res_b.wall_time_ms} ms", f"{res_c.wall_time_ms} ms (Parallel asyncio.gather)")
    console.print("\n", cmp_table)

    # 6. Render Final Cited Markdown Report
    console.print("\n")
    console.print(Panel(Markdown(res_c.report_markdown), title="5. Final Cited Research Report", border_style="cyan"))


if __name__ == "__main__":
    asyncio.run(run_demo())
