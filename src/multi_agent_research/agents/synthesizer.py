"""ReportSynthesizer: Compiles Verified Evidence & Quant Artifacts into Cited Markdown."""

from __future__ import annotations

from typing import Any, Optional

from multi_agent_research.a2a.models import (
    A2ATask,
    AgentCard,
    AgentSkill,
    EvidenceArtifact,
    IsolationLevel,
    QuantArtifact,
)
from multi_agent_research.agents.base import BaseA2AAgent, EphemeralAgentContext
from multi_agent_research.llm import DualModeLLM


class ReportSynthesizer(BaseA2AAgent):
    """Synthesizes verified artifacts into a provenance-grounded Markdown report."""

    def __init__(self, llm: DualModeLLM) -> None:
        self._llm = llm
        self._card = AgentCard(
            agent_id="report-synthesizer",
            name="Provenance-Grounded Report Synthesizer",
            version="1.0.0",
            description="Compiles verified EvidenceArtifacts and QuantArtifacts into Markdown with inline [DOC-ID] citations.",
            endpoint="a2a://local/report-synthesizer",
            skills=[
                AgentSkill(
                    skill_id="cited_report_synthesis",
                    name="Traceable Markdown Synthesis",
                    description="Generates final research reports with verbatim provenance footnotes and rumor quarantine disclosures.",
                    tags=["synthesis", "report", "citations", "markdown"],
                )
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "verified_artifacts": {"type": "array"},
                    "quant_artifacts": {"type": "array"},
                    "quarantined_artifacts": {"type": "array"},
                },
            },
            output_schema={"type": "object", "properties": {"report_markdown": {"type": "string"}}},
            isolation_level=IsolationLevel.STRICT_EPHEMERAL,
        )

    @property
    def card(self) -> AgentCard:
        return self._card

    async def _execute_isolated(self, task: A2ATask, ctx: EphemeralAgentContext) -> A2ATask:
        question = str(task.input_payload.get("question", ""))
        verified = [EvidenceArtifact.model_validate(a) for a in task.input_payload.get("verified_artifacts", [])]
        quants = [QuantArtifact.model_validate(q) for q in task.input_payload.get("quant_artifacts", [])]
        quarantined = [EvidenceArtifact.model_validate(q) for q in task.input_payload.get("quarantined_artifacts", [])]

        ctx.append_thought(
            f"Synthesizing report from {len(verified)} verified claims, {len(quants)} quant results, "
            f"and {len(quarantined)} quarantined claims."
        )

        primary_answer: Optional[float] = quants[0].result_value if quants else None
        secondary_answers: list[float] = [q.result_value for q in quants[1:]]

        lines: list[str] = [
            "## Verified Deep Research Report",
            f"**Research Question:** {question}",
            "",
            "### 1. Executive Conclusion",
        ]

        if quants:
            for q_art in quants:
                unit_str = f" {q_art.unit}" if q_art.unit else ""
                lines.append(
                    f"- **{q_art.description}:** `{q_art.result_value}{unit_str}` "
                    f"(derived via deterministic formula `{q_art.formula_expression}`)"
                )
        elif verified:
            lines.append(f"- Synthesized from {len(verified)} primary/corroborated technical sources.")
        else:
            lines.append("- Insufficient verified primary evidence retrieved to establish a quantitative conclusion.")

        lines.extend(["", "### 2. Verified Evidence & Provenance Footnotes"])
        cited_quotes: list[dict[str, str]] = []
        cited_doc_ids: list[str] = []

        for art in verified:
            lines.append(
                f"- **[{art.source_doc_id}]** ({art.source_credibility_tier.value} | `{art.verification_status.value}`): "
                f"{art.statement}\n"
                f"  - *Verbatim Quote:* \"{art.exact_quote_span}\""
            )
            cited_quotes.append({"doc_id": art.source_doc_id, "quote": art.exact_quote_span})
            if art.source_doc_id not in cited_doc_ids:
                cited_doc_ids.append(art.source_doc_id)

        if quants:
            lines.extend(["", "### 3. Deterministic Quantitative Audit"])
            for q_art in quants:
                lines.append(
                    f"- `{q_art.metric_key}`: `{q_art.formula_expression}` = **{q_art.result_value} {q_art.unit}** "
                    f"| Inputs: `{q_art.input_variables}`"
                )

        if quarantined:
            lines.extend(["", "### 4. Echo-Chamber & Hallucination Quarantine Log"])
            for q_clm in quarantined:
                lines.append(
                    f"- **QUARANTINED [{q_clm.source_doc_id}]** (`{q_clm.verification_status.value}`): "
                    f"{q_clm.statement} — *{q_clm.critic_notes}*"
                )

        report_md = "\n".join(lines)
        fallback_out: dict[str, Any] = {
            "report_markdown": report_md,
            "primary_numeric_answer": primary_answer,
            "all_numeric_answers": [primary_answer, *secondary_answers] if primary_answer is not None else [],
            "cited_doc_ids": cited_doc_ids,
            "cited_quotes": cited_quotes,
        }

        compact_prompt = "\n".join(
            [f"[{a.source_doc_id}] {a.statement}" for a in verified]
            + [f"[CALC] {q.description} = {q.result_value} {q.unit}" for q in quants]
        )
        llm_res, _ = await self._llm.chat_json(
            system_prompt="Compile verified artifacts into a concise cited Markdown report JSON.",
            user_prompt=f"Question: {question}\n{compact_prompt}",
            fallback_payload=fallback_out,
        )

        task.structured_data = {
            "report_markdown": llm_res.get("report_markdown", report_md),
            "primary_numeric_answer": primary_answer,
            "all_numeric_answers": fallback_out["all_numeric_answers"],
            "cited_doc_ids": cited_doc_ids,
            "cited_quotes": cited_quotes,
        }
        return task
