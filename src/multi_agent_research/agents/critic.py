"""AdversarialCriticAgent: Cross-Source Provenance & Echo-Chamber Suppressor."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from multi_agent_research.a2a.models import (
    A2ATask,
    AgentCard,
    AgentSkill,
    CredibilityTier,
    EvidenceArtifact,
    IsolationLevel,
    SubTaskSpec,
    VerificationStatus,
)
from multi_agent_research.agents.base import BaseA2AAgent, EphemeralAgentContext
from multi_agent_research.corpus.search import ResearchCorpusIndex


class AdversarialCriticAgent(BaseA2AAgent):
    """Performs independent Cross-Source Provenance Verification & Echo-Chamber Quarantine.

    Verification rules:
      1. Verbatim Quote Grounding: Every `exact_quote_span` must appear verbatim inside
         `corpus.get_document(source_doc_id).content`; fabricated citations are rejected
         (`REJECTED_FABRICATED_QUOTE`).
      2. Credibility Tier Enforcement: Re-validates `source_credibility_tier` against the
         authoritative repository registry to prevent tier spoofing.
      3. Echo-Chamber Suppression: Multiple `UNVERIFIED_RUMOR` documents repeating identical
         fake numbers do NOT accumulate authority over a `PRIMARY_SPEC` or `SECONDARY_REPORT`.
         Conflicting rumors are quarantined (`REJECTED_RUMOR`), while the authoritative spec
         is marked `CONFLICT_RESOLVED`.
      4. Missing-Primary Re-Retrieval Trigger: If a metric only has `UNVERIFIED_RUMOR`
         evidence, quarantines the rumors and emits a `prefer_primary_only=True` subtask
         to pull authoritative specs before synthesis.
    """

    def __init__(self, corpus_index: ResearchCorpusIndex) -> None:
        self._corpus = corpus_index
        self._card = AgentCard(
            agent_id="adversarial-critic",
            name="Adversarial Provenance Critic & Echo-Chamber Suppressor",
            version="1.0.0",
            description="Verifies verbatim quote grounding, quarantines echo-chamber rumors, and triggers targeted re-retrieval.",
            endpoint="a2a://local/adversarial-critic",
            skills=[
                AgentSkill(
                    skill_id="cross_source_provenance_audit",
                    name="Verbatim Grounding & Echo-Chamber Suppression",
                    description="Quarantines fabricated citations and secondary rumor echo chambers.",
                    tags=["critic", "verification", "echo-chamber", "provenance"],
                )
            ],
            input_schema={"type": "object", "properties": {"candidate_artifacts": {"type": "array"}}},
            output_schema={
                "type": "object",
                "properties": {
                    "verified_artifacts": {"type": "array"},
                    "quarantined_artifacts": {"type": "array"},
                    "re_retrieval_subtasks": {"type": "array"},
                },
            },
            isolation_level=IsolationLevel.STRICT_EPHEMERAL,
        )

    @property
    def card(self) -> AgentCard:
        return self._card

    @staticmethod
    def _numbers_conflict(vals_a: dict[str, float], vals_b: dict[str, float]) -> bool:
        shared_keys = set(vals_a.keys()).intersection(vals_b.keys())
        for k in shared_keys:
            a, b = float(vals_a[k]), float(vals_b[k])
            tol = max(1e-5, 1e-3 * max(abs(a), abs(b)))
            if abs(a - b) > tol:
                return True
        return False

    async def _execute_isolated(self, task: A2ATask, ctx: EphemeralAgentContext) -> A2ATask:
        raw_candidates = task.input_payload.get("candidate_artifacts", [])
        candidates = [EvidenceArtifact.model_validate(item) for item in raw_candidates]
        ctx.append_thought(f"Auditing {len(candidates)} candidate EvidenceArtifacts across sources.")

        quote_valid_by_metric: dict[str, list[EvidenceArtifact]] = defaultdict(list)
        quarantined: list[EvidenceArtifact] = []

        # Stage 1: Verbatim Quote & Registry Tier Verification
        for art in candidates:
            doc = self._corpus.get_document(art.source_doc_id)
            if doc is None:
                art.verification_status = VerificationStatus.REJECTED_FABRICATED_QUOTE
                art.critic_notes = f"Source document '{art.source_doc_id}' does not exist in repository."
                quarantined.append(art)
                ctx.append_thought(f"[REJECT_DOC] {art.source_doc_id}: unknown doc_id")
                continue

            # Enforce true registry tier
            art.source_credibility_tier = doc.credibility_tier

            if not self._corpus.verify_verbatim_quote(art.source_doc_id, art.exact_quote_span):
                art.verification_status = VerificationStatus.REJECTED_FABRICATED_QUOTE
                art.critic_notes = (
                    f"Exact quote span failed verbatim substring check in {art.source_doc_id}."
                )
                quarantined.append(art)
                ctx.append_thought(f"[REJECT_QUOTE] {art.claim_id} from {art.source_doc_id}: fabricated span")
                continue

            quote_valid_by_metric[art.metric_key].append(art)

        # Stage 2: Cross-Source Conflict Resolution & Echo-Chamber Quarantine
        verified: list[EvidenceArtifact] = []
        re_retrieval_subtasks: list[SubTaskSpec] = []

        for metric_key, group in sorted(quote_valid_by_metric.items()):
            authoritative = [
                a for a in group if a.source_credibility_tier in {CredibilityTier.PRIMARY_SPEC, CredibilityTier.SECONDARY_REPORT}
            ]
            rumors = [a for a in group if a.source_credibility_tier == CredibilityTier.UNVERIFIED_RUMOR]

            if not authoritative and rumors:
                # Only rumor sources were retrieved! Quarantine and request targeted primary re-retrieval
                rumor_docs = sorted({r.source_doc_id for r in rumors})
                for r in rumors:
                    r.verification_status = VerificationStatus.REJECTED_RUMOR
                    r.critic_notes = (
                        f"Uncorroborated echo-chamber rumor ({len(rumor_docs)} low-tier source(s): "
                        f"{', '.join(rumor_docs)}) with zero PRIMARY_SPEC evidence; triggering primary re-retrieval."
                    )
                    quarantined.append(r)

                entity_hint = metric_key.replace(".", " ").replace("_", " ")
                re_retrieval_subtasks.append(
                    SubTaskSpec(
                        subtask_id=f"reret-{metric_key.replace('.', '-')}",
                        search_query=f"{entity_hint} official specification audited benchmark",
                        target_entity=metric_key.split(".")[0],
                        target_metrics=[metric_key],
                        prefer_primary_only=True,
                    )
                )
                ctx.append_thought(
                    f"[TRIGGER_RERETRIEVAL] {metric_key}: quarantined {len(rumors)} unbacked rumor(s)"
                )
                continue

            # Sort authoritative claims: PRIMARY_SPEC before SECONDARY_REPORT, then confidence
            authoritative.sort(
                key=lambda a: (-a.source_credibility_tier.authority_weight, -a.confidence, a.source_doc_id)
            )
            winner = authoritative[0]

            # Check if any rumor conflicts with the authoritative winner
            had_conflict = False
            rumor_doc_ids = sorted({r.source_doc_id for r in rumors})
            for r in rumors:
                if self._numbers_conflict(winner.numeric_values, r.numeric_values):
                    had_conflict = True
                    r.verification_status = VerificationStatus.REJECTED_RUMOR
                    r.critic_notes = (
                        f"Echo-chamber rumor ({len(rumor_doc_ids)} rumor doc(s): {', '.join(rumor_doc_ids)}) "
                        f"conflicts with {winner.source_credibility_tier.value} {winner.source_doc_id}: "
                        f"rumor={r.numeric_values} vs verified={winner.numeric_values}."
                    )
                else:
                    r.verification_status = VerificationStatus.REJECTED_RUMOR
                    r.critic_notes = f"Redundant low-credibility tier superseded by {winner.source_doc_id}."
                quarantined.append(r)

            distinct_auth_docs = {a.source_doc_id for a in authoritative}
            if had_conflict:
                winner.verification_status = VerificationStatus.CONFLICT_RESOLVED
                winner.critic_notes = (
                    f"Verified via {winner.source_credibility_tier.value} ({winner.source_doc_id}); "
                    f"overruled {len(rumor_doc_ids)} conflicting rumor source(s) ({', '.join(rumor_doc_ids)})."
                )
                ctx.append_thought(
                    f"[CONFLICT_RESOLVED] {metric_key}: kept {winner.source_doc_id}, rejected {rumor_doc_ids}"
                )
            elif len(distinct_auth_docs) > 1:
                winner.verification_status = VerificationStatus.VERIFIED_CORROBORATED
                winner.critic_notes = f"Corroborated across {len(distinct_auth_docs)} authoritative sources."
            else:
                winner.verification_status = VerificationStatus.VERIFIED_PRIMARY
                winner.critic_notes = f"Grounded verbatim in {winner.source_credibility_tier.value} {winner.source_doc_id}."

            verified.append(winner)

        task.artifacts = verified
        summary_data: dict[str, Any] = {
            "verified_count": len(verified),
            "quarantined_count": len(quarantined),
            "quarantined_artifacts": [q.model_dump(mode="json") for q in quarantined],
            "re_retrieval_subtasks": [s.model_dump(mode="json") for s in re_retrieval_subtasks],
        }
        task.structured_data = summary_data
        return task
