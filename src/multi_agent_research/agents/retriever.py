"""SourceRetrieverAgent: Ephemeral-Context Corpus Search & Typed Claim Extractor."""

from __future__ import annotations

from typing import Any

from multi_agent_research.a2a.models import (
    A2ATask,
    AgentCard,
    AgentSkill,
    EvidenceArtifact,
    IsolationLevel,
    SubTaskSpec,
)
from multi_agent_research.agents.base import BaseA2AAgent, EphemeralAgentContext
from multi_agent_research.corpus.search import ResearchCorpusIndex
from multi_agent_research.llm import DualModeLLM


class SourceRetrieverAgent(BaseA2AAgent):
    """Executes isolated local repository search and emits typed `EvidenceArtifact` claims."""

    def __init__(self, corpus_index: ResearchCorpusIndex, llm: DualModeLLM) -> None:
        self._corpus = corpus_index
        self._llm = llm
        self._card = AgentCard(
            agent_id="source-retriever",
            name="Context-Isolated Source Retriever Agent",
            version="1.0.0",
            description="Searches multi-source technical corpus and emits provenance-tagged EvidenceArtifacts.",
            endpoint="a2a://local/source-retriever",
            skills=[
                AgentSkill(
                    skill_id="isolated_evidence_retrieval",
                    name="Isolated Corpus Search & Claim Extraction",
                    description="Retrieves specs, telemetry tables, and reports without leaking raw text scratchpads.",
                    tags=["retrieval", "evidence", "rag", "provenance"],
                )
            ],
            input_schema={
                "type": "object",
                "properties": {"subtask": {"type": "object"}, "top_k": {"type": "integer"}},
                "required": ["subtask"],
            },
            output_schema={"type": "object", "properties": {"artifacts": {"type": "array"}}},
            isolation_level=IsolationLevel.STRICT_EPHEMERAL,
        )

    @property
    def card(self) -> AgentCard:
        return self._card

    async def _execute_isolated(self, task: A2ATask, ctx: EphemeralAgentContext) -> A2ATask:
        raw_sub = task.input_payload.get("subtask", {})
        subtask = SubTaskSpec.model_validate(raw_sub)
        top_k = int(task.input_payload.get("top_k", 4))
        exclude_ids = set(task.input_payload.get("exclude_doc_ids", []))
        target_metric_filter: str | None = task.input_payload.get("target_metric_key")

        hits = self._corpus.search(
            query=subtask.search_query,
            top_k=top_k,
            prefer_primary_only=subtask.prefer_primary_only,
            exclude_doc_ids=exclude_ids,
        )

        ctx.append_thought(
            f"Executing search query='{subtask.search_query}' (prefer_primary={subtask.prefer_primary_only}, top_k={top_k})"
        )
        for doc, score in hits:
            ctx.append_thought(
                f"[RAW_DOC {doc.doc_id} score={score:.3f} tier={doc.credibility_tier.value}] "
                f"{doc.title} | {doc.source_org} ({doc.published_date})\n{doc.content}"
            )

        extracted: list[EvidenceArtifact] = []
        for doc, score in hits:
            for claim in doc.structured_claims:
                if target_metric_filter and claim.metric_key != target_metric_filter:
                    continue
                conf = min(0.98, round(0.55 + 0.35 * doc.credibility_tier.authority_weight + 0.05 * min(score, 1.0), 3))
                extracted.append(
                    EvidenceArtifact(
                        metric_key=claim.metric_key,
                        statement=claim.statement,
                        numeric_values=dict(claim.numeric_values),
                        source_doc_id=doc.doc_id,
                        exact_quote_span=claim.exact_quote_span,
                        source_credibility_tier=doc.credibility_tier,
                        confidence=conf,
                    )
                )

        fallback_payload: dict[str, Any] = {
            "artifacts": [a.model_dump(mode="json") for a in extracted]
        }
        raw_context_blob = "\n\n".join(ctx.scratchpad_entries)
        _, _ = await self._llm.chat_json(
            system_prompt=(
                "Extract structured EvidenceArtifact JSON records with exact verbatim quotes "
                "from the retrieved documents. Never paraphrase exact_quote_span."
            ),
            user_prompt=raw_context_blob,
            fallback_payload=fallback_payload,
        )

        task.artifacts = extracted
        task.structured_data = {
            "subtask_id": subtask.subtask_id,
            "retrieved_doc_ids": [d.doc_id for d, _ in hits],
        }
        return task
