"""Multi-Agent Deep Research Orchestrators (Topologies A, B, and C).

Implements three distinct execution topologies over the same 60-document corpus
to isolate the architectural impact of A2A Context Isolation, QuantComputeAgent,
and AdversarialCriticAgent:

  - Topology A (`MonolithicSingleAgentRunner`): Single growing context window holding all
    raw retrieved documents and in-context mental math.
  - Topology B (`SharedGroupChatRunner`): Shared-history multi-agent group chat without
    context isolation or adversarial provenance verification (exhibits Echo-Chamber
    group hallucination amplification).
  - Topology C (`A2AContextIsolatedOrchestrator`, Ours): LangGraph StateGraph + `asyncio.gather`
    coordinating 5 strictly isolated A2A agents with targeted primary re-retrieval.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter, defaultdict
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from multi_agent_research.a2a.bus import A2ABus
from multi_agent_research.a2a.models import (
    CredibilityTier,
    EvidenceArtifact,
    QuantArtifact,
    SubTaskSpec,
    VerificationStatus,
)
from multi_agent_research.agents.critic import AdversarialCriticAgent
from multi_agent_research.agents.planner import LeadPlannerAgent
from multi_agent_research.agents.quant import QuantComputeAgent, SafeMathASTEvaluator
from multi_agent_research.agents.retriever import SourceRetrieverAgent
from multi_agent_research.agents.synthesizer import ReportSynthesizer
from multi_agent_research.corpus.documents import CorpusDocument
from multi_agent_research.corpus.search import ResearchCorpusIndex
from multi_agent_research.llm import DualModeLLM, estimate_tokens


class ResearchRunResult(BaseModel):
    """Unified execution & telemetry record returned by any research topology."""

    model_config = ConfigDict(extra="forbid")

    topology: str
    question: str
    primary_numeric_answer: Optional[float] = None
    all_numeric_answers: list[float] = Field(default_factory=list)
    report_markdown: str = ""
    verified_artifacts: list[EvidenceArtifact] = Field(default_factory=list)
    quarantined_artifacts: list[EvidenceArtifact] = Field(default_factory=list)
    quant_artifacts: list[QuantArtifact] = Field(default_factory=list)
    cited_doc_ids: list[str] = Field(default_factory=list)
    cited_quotes: list[dict[str, str]] = Field(default_factory=list)
    coordinator_peak_tokens: int = 0
    total_ephemeral_worker_tokens: int = 0
    a2a_rpc_count: int = 0
    re_retrieval_triggered: bool = False
    wall_time_ms: float = 0.0
    trace_events: list[dict[str, Any]] = Field(default_factory=list)


class OrchestratorGraphState(TypedDict, total=False):
    """LangGraph state for Topology C Coordinator.

    Notice that NO raw corpus documents or worker scratchpads exist in this state—only
    compact typed A2A specs and artifacts!
    """

    question: str
    top_k: int
    parallel_dispatch: bool
    max_re_retrieval_rounds: int
    re_retrieval_round: int
    re_retrieval_triggered: bool
    active_subtasks: list[dict[str, Any]]
    candidate_artifacts: list[dict[str, Any]]
    verified_artifacts: list[dict[str, Any]]
    quarantined_artifacts: list[dict[str, Any]]
    re_retrieval_subtasks: list[dict[str, Any]]
    quant_artifacts: list[dict[str, Any]]
    report_markdown: str
    primary_numeric_answer: Optional[float]
    all_numeric_answers: list[float]
    cited_doc_ids: list[str]
    cited_quotes: list[dict[str, str]]
    coordinator_tokens_timeline: list[int]
    ephemeral_worker_tokens: int
    retrieved_doc_ids_seen: list[str]
    trace_events: list[dict[str, Any]]


class A2AContextIsolatedOrchestrator:
    """Topology C: LangGraph + A2A JSON-RPC 2.0 + Strict Context Isolation + Echo-Chamber Critic."""

    def __init__(
        self,
        corpus_index: Optional[ResearchCorpusIndex] = None,
        llm: Optional[DualModeLLM] = None,
        simulated_io_latency_ms: float = 35.0,
        max_planner_subtasks: int = 5,
    ) -> None:
        self.corpus = corpus_index if corpus_index is not None else ResearchCorpusIndex()
        self.llm = llm if llm is not None else DualModeLLM()
        self.bus = A2ABus(simulated_io_latency_ms=simulated_io_latency_ms)

        self.planner = LeadPlannerAgent(llm=self.llm, max_subtasks=max_planner_subtasks)
        self.retriever = SourceRetrieverAgent(corpus_index=self.corpus, llm=self.llm)
        self.critic = AdversarialCriticAgent(corpus_index=self.corpus)
        self.quant = QuantComputeAgent()
        self.synthesizer = ReportSynthesizer(llm=self.llm)

        for agent in (self.planner, self.retriever, self.critic, self.quant, self.synthesizer):
            self.bus.register_agent(agent)

        self._graph = self._build_langgraph()

    def _build_langgraph(self) -> Any:
        workflow = StateGraph(OrchestratorGraphState)

        workflow.add_node("plan_dag", self._node_plan_dag)
        workflow.add_node("parallel_retrieve", self._node_parallel_retrieve)
        workflow.add_node("adversarial_verify", self._node_adversarial_verify)
        workflow.add_node("quant_compute", self._node_quant_compute)
        workflow.add_node("synthesize_report", self._node_synthesize_report)

        workflow.add_edge(START, "plan_dag")
        workflow.add_edge("plan_dag", "parallel_retrieve")
        workflow.add_edge("parallel_retrieve", "adversarial_verify")
        workflow.add_conditional_edges(
            "adversarial_verify",
            self._route_after_critic,
            {
                "re_retrieve": "parallel_retrieve",
                "quant_compute": "quant_compute",
            },
        )
        workflow.add_edge("quant_compute", "synthesize_report")
        workflow.add_edge("synthesize_report", END)

        return workflow.compile()

    @staticmethod
    def _estimate_coordinator_state_tokens(
        question: str,
        subtasks: list[dict[str, Any]],
        verified_or_candidates: list[dict[str, Any]],
        quants: list[dict[str, Any]],
    ) -> int:
        base = estimate_tokens(question) + 24
        sub_toks = sum(18 for _ in subtasks)
        art_toks = sum(
            EvidenceArtifact.model_validate(a).compact_token_estimate() for a in verified_or_candidates
        )
        q_toks = sum(22 for _ in quants)
        return base + sub_toks + art_toks + q_toks

    async def _node_plan_dag(self, state: OrchestratorGraphState) -> dict[str, Any]:
        question = state["question"]
        task = await self.bus.send_task(
            target_agent_id="lead-planner",
            input_payload={"question": question},
        )
        subtasks = task.structured_data.get("subtasks", [])
        coord_toks = self._estimate_coordinator_state_tokens(question, subtasks, [], [])
        trace = list(state.get("trace_events", []))
        trace.append({
            "stage": "plan_dag",
            "agent_id": "lead-planner",
            "task_id": task.task_id,
            "subtask_count": len(subtasks),
            "ephemeral_worker_tokens": task.ephemeral_worker_tokens,
            "coordinator_payload_tokens": task.coordinator_payload_tokens,
        })
        return {
            "active_subtasks": subtasks,
            "ephemeral_worker_tokens": state.get("ephemeral_worker_tokens", 0) + task.ephemeral_worker_tokens,
            "coordinator_tokens_timeline": [*state.get("coordinator_tokens_timeline", []), coord_toks],
            "trace_events": trace,
        }

    async def _node_parallel_retrieve(self, state: OrchestratorGraphState) -> dict[str, Any]:
        subtasks = state.get("active_subtasks", [])
        top_k = state.get("top_k", 4)
        parallel = state.get("parallel_dispatch", True)
        seen_docs = set(state.get("retrieved_doc_ids_seen", []))
        is_reretrieval = state.get("re_retrieval_round", 0) > 0 or any(
            s.get("prefer_primary_only", False) for s in subtasks
        )

        coros = [
            self.bus.send_task(
                target_agent_id="source-retriever",
                input_payload={
                    "subtask": sub,
                    "top_k": top_k,
                    "exclude_doc_ids": sorted(seen_docs) if is_reretrieval else [],
                    "target_metric_key": sub["target_metrics"][0]
                    if (is_reretrieval and sub.get("target_metrics"))
                    else None,
                },
            )
            for sub in subtasks
        ]

        if parallel:
            completed_tasks = await asyncio.gather(*coros)
        else:
            completed_tasks = []
            for c in coros:
                completed_tasks.append(await c)

        existing_candidates = list(state.get("candidate_artifacts", []))
        seen_doc_claim_pairs = {
            (c["source_doc_id"], c["metric_key"]) for c in existing_candidates
        }

        ephemeral_added = 0
        for t in completed_tasks:
            ephemeral_added += t.ephemeral_worker_tokens
            for doc_id in t.structured_data.get("retrieved_doc_ids", []):
                seen_docs.add(doc_id)
            for art in t.artifacts:
                key = (art.source_doc_id, art.metric_key)
                if key not in seen_doc_claim_pairs:
                    seen_doc_claim_pairs.add(key)
                    existing_candidates.append(art.model_dump(mode="json"))

        coord_toks = self._estimate_coordinator_state_tokens(
            state["question"], subtasks, existing_candidates, []
        )
        trace = list(state.get("trace_events", []))
        trace.append({
            "stage": "re_retrieve" if is_reretrieval else "parallel_retrieve",
            "agent_id": "source-retriever",
            "parallel_fanout": len(subtasks),
            "parallel_mode": parallel,
            "candidate_artifacts_total": len(existing_candidates),
            "ephemeral_worker_tokens": ephemeral_added,
        })
        return {
            "candidate_artifacts": existing_candidates,
            "retrieved_doc_ids_seen": sorted(seen_docs),
            "ephemeral_worker_tokens": state.get("ephemeral_worker_tokens", 0) + ephemeral_added,
            "coordinator_tokens_timeline": [*state.get("coordinator_tokens_timeline", []), coord_toks],
            "trace_events": trace,
        }

    async def _node_adversarial_verify(self, state: OrchestratorGraphState) -> dict[str, Any]:
        candidates = state.get("candidate_artifacts", [])
        task = await self.bus.send_task(
            target_agent_id="adversarial-critic",
            input_payload={"candidate_artifacts": candidates},
        )

        verified = [a.model_dump(mode="json") for a in task.artifacts]
        quarantined = task.structured_data.get("quarantined_artifacts", [])
        reret_subtasks = task.structured_data.get("re_retrieval_subtasks", [])

        coord_toks = self._estimate_coordinator_state_tokens(
            state["question"], state.get("active_subtasks", []), verified, []
        )
        trace = list(state.get("trace_events", []))
        trace.append({
            "stage": "adversarial_verify",
            "agent_id": "adversarial-critic",
            "verified_count": len(verified),
            "quarantined_count": len(quarantined),
            "re_retrieval_requested": len(reret_subtasks),
        })

        next_round = state.get("re_retrieval_round", 0)
        triggered = state.get("re_retrieval_triggered", False)
        active_next = state.get("active_subtasks", [])
        if reret_subtasks and next_round < state.get("max_re_retrieval_rounds", 1):
            active_next = reret_subtasks
            next_round += 1
            triggered = True

        return {
            "verified_artifacts": verified,
            "quarantined_artifacts": quarantined,
            "re_retrieval_subtasks": reret_subtasks,
            "active_subtasks": active_next,
            "re_retrieval_round": next_round,
            "re_retrieval_triggered": triggered,
            "ephemeral_worker_tokens": state.get("ephemeral_worker_tokens", 0) + task.ephemeral_worker_tokens,
            "coordinator_tokens_timeline": [*state.get("coordinator_tokens_timeline", []), coord_toks],
            "trace_events": trace,
        }

    @staticmethod
    def _route_after_critic(state: OrchestratorGraphState) -> str:
        reret = state.get("re_retrieval_subtasks", [])
        # Note: `_node_adversarial_verify` incremented `re_retrieval_round` to 1 when scheduling pass 2
        if reret and state.get("re_retrieval_triggered", False) and state.get("re_retrieval_round", 0) == 1:
            # Check that we haven't already run the re_retrieve stage
            stages = [e["stage"] for e in state.get("trace_events", [])]
            if "re_retrieve" not in stages:
                return "re_retrieve"
        return "quant_compute"

    async def _node_quant_compute(self, state: OrchestratorGraphState) -> dict[str, Any]:
        verified = state.get("verified_artifacts", [])
        task = await self.bus.send_task(
            target_agent_id="quant-compute",
            input_payload={
                "question": state["question"],
                "verified_artifacts": verified,
            },
        )
        quants = [q.model_dump(mode="json") for q in task.quant_artifacts]
        coord_toks = self._estimate_coordinator_state_tokens(
            state["question"], [], verified, quants
        )
        trace = list(state.get("trace_events", []))
        trace.append({
            "stage": "quant_compute",
            "agent_id": "quant-compute",
            "evaluated_formulas": len(quants),
            "results": [{q["metric_key"]: q["result_value"]} for q in quants],
        })
        return {
            "quant_artifacts": quants,
            "ephemeral_worker_tokens": state.get("ephemeral_worker_tokens", 0) + task.ephemeral_worker_tokens,
            "coordinator_tokens_timeline": [*state.get("coordinator_tokens_timeline", []), coord_toks],
            "trace_events": trace,
        }

    async def _node_synthesize_report(self, state: OrchestratorGraphState) -> dict[str, Any]:
        task = await self.bus.send_task(
            target_agent_id="report-synthesizer",
            input_payload={
                "question": state["question"],
                "verified_artifacts": state.get("verified_artifacts", []),
                "quant_artifacts": state.get("quant_artifacts", []),
                "quarantined_artifacts": state.get("quarantined_artifacts", []),
            },
        )
        sdata = task.structured_data
        trace = list(state.get("trace_events", []))
        trace.append({
            "stage": "synthesize_report",
            "agent_id": "report-synthesizer",
            "cited_doc_count": len(sdata.get("cited_doc_ids", [])),
            "primary_numeric_answer": sdata.get("primary_numeric_answer"),
        })
        return {
            "report_markdown": sdata.get("report_markdown", ""),
            "primary_numeric_answer": sdata.get("primary_numeric_answer"),
            "all_numeric_answers": sdata.get("all_numeric_answers", []),
            "cited_doc_ids": sdata.get("cited_doc_ids", []),
            "cited_quotes": sdata.get("cited_quotes", []),
            "ephemeral_worker_tokens": state.get("ephemeral_worker_tokens", 0) + task.ephemeral_worker_tokens,
            "trace_events": trace,
        }

    async def run(
        self,
        question: str,
        top_k: int = 4,
        parallel_dispatch: bool = True,
        max_re_retrieval_rounds: int = 1,
    ) -> ResearchRunResult:
        """Execute full Topology C LangGraph A2A workflow on `question`."""
        rpc_before = len([e for e in self.bus.rpc_log if e["direction"] == "request"])
        t0 = time.perf_counter()

        init_state: OrchestratorGraphState = {
            "question": question,
            "top_k": top_k,
            "parallel_dispatch": parallel_dispatch,
            "max_re_retrieval_rounds": max_re_retrieval_rounds,
            "re_retrieval_round": 0,
            "re_retrieval_triggered": False,
            "active_subtasks": [],
            "candidate_artifacts": [],
            "verified_artifacts": [],
            "quarantined_artifacts": [],
            "re_retrieval_subtasks": [],
            "quant_artifacts": [],
            "coordinator_tokens_timeline": [],
            "ephemeral_worker_tokens": 0,
            "retrieved_doc_ids_seen": [],
            "trace_events": [],
        }

        final_state: OrchestratorGraphState = await self._graph.ainvoke(init_state)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        rpc_after = len([e for e in self.bus.rpc_log if e["direction"] == "request"])

        timeline = final_state.get("coordinator_tokens_timeline", [0])
        return ResearchRunResult(
            topology="Topology C: A2A Isolated Async DAG + Quant + Critic",
            question=question,
            primary_numeric_answer=final_state.get("primary_numeric_answer"),
            all_numeric_answers=final_state.get("all_numeric_answers", []),
            report_markdown=final_state.get("report_markdown", ""),
            verified_artifacts=[
                EvidenceArtifact.model_validate(a) for a in final_state.get("verified_artifacts", [])
            ],
            quarantined_artifacts=[
                EvidenceArtifact.model_validate(q) for q in final_state.get("quarantined_artifacts", [])
            ],
            quant_artifacts=[
                QuantArtifact.model_validate(qa) for qa in final_state.get("quant_artifacts", [])
            ],
            cited_doc_ids=final_state.get("cited_doc_ids", []),
            cited_quotes=final_state.get("cited_quotes", []),
            coordinator_peak_tokens=max(timeline) if timeline else 0,
            total_ephemeral_worker_tokens=final_state.get("ephemeral_worker_tokens", 0),
            a2a_rpc_count=rpc_after - rpc_before,
            re_retrieval_triggered=final_state.get("re_retrieval_triggered", False),
            wall_time_ms=round(elapsed_ms, 2),
            trace_events=final_state.get("trace_events", []),
        )


# ============================================================================
# BASELINE TOPOLOGY A: SINGLE-AGENT MONOLITHIC CONTEXT
# ============================================================================


class MonolithicSingleAgentRunner:
    """Topology A Baseline: Single-agent ReAct loop accumulating raw docs & mental math in one prompt."""

    def __init__(
        self,
        corpus_index: Optional[ResearchCorpusIndex] = None,
        llm: Optional[DualModeLLM] = None,
        simulated_io_latency_ms: float = 35.0,
    ) -> None:
        self.corpus = corpus_index if corpus_index is not None else ResearchCorpusIndex()
        self.llm = llm if llm is not None else DualModeLLM()
        self.simulated_io_latency_ms = simulated_io_latency_ms
        self._planner_helper = LeadPlannerAgent(llm=self.llm, max_subtasks=4)

    async def run(self, question: str, top_k_per_hop: int = 4) -> ResearchRunResult:
        t0 = time.perf_counter()
        monolithic_blocks: list[str] = [
            f"[SYSTEM] Answer the multi-hop technical research question using retrieved documents and mental arithmetic.\n[QUESTION] {question}"
        ]

        # Sequential multi-hop searches dumped raw into one growing prompt
        sub_task = await self._planner_helper.handle_task(
            self._make_local_task(question)
        )
        subtasks = [SubTaskSpec.model_validate(s) for s in sub_task.structured_data.get("subtasks", [])]

        retrieved_docs: list[CorpusDocument] = []
        seen_ids: set[str] = set()
        for hop_idx, st in enumerate(subtasks):
            if self.simulated_io_latency_ms > 0:
                await asyncio.sleep(self.simulated_io_latency_ms / 1000.0)
            hits = self.corpus.search(st.search_query, top_k=top_k_per_hop, prefer_primary_only=False)
            monolithic_blocks.append(f"\n--- HOP {hop_idx + 1} SEARCH: {st.search_query} ---")
            for doc, score in hits:
                # Monolithic agent appends every search hit raw (even duplicates across hops inflate prompt!)
                monolithic_blocks.append(
                    f"[DOC {doc.doc_id} | {doc.title} | {doc.source_org}]\n{doc.content}\n"
                    f"[SCRATCHPAD] Analyzing raw figures in {doc.doc_id} (relevance={score:.3f})..."
                )
                if doc.doc_id not in seen_ids:
                    seen_ids.add(doc.doc_id)
                    retrieved_docs.append(doc)

        full_monolithic_prompt = "\n".join(monolithic_blocks)
        prompt_tokens = estimate_tokens(full_monolithic_prompt) + 320

        # Mechanistic Monolithic Selection:
        # 1. Context distraction ("lost-in-the-middle"): docs in middle positions (index 4..len-2)
        #    can suffer attention drop when monolithic prompt exceeds 1300 tokens.
        active_docs: list[CorpusDocument] = []
        for idx, doc in enumerate(retrieved_docs):
            is_middle = 3 <= idx < max(4, len(retrieved_docs) - 1)
            if is_middle and prompt_tokens > 1350:
                drop_u = DualModeLLM.deterministic_unit_hash(f"mono_drop|{question}|{doc.doc_id}")
                if drop_u < min(0.38, (prompt_tokens - 1100) / 4200.0):
                    continue
            active_docs.append(doc)

        # 2. Without AdversarialCriticAgent, conflicting claims on the same metric_key are
        #    resolved by raw source count in prompt + slight lexical salience bias.
        claims_by_metric: dict[str, list[EvidenceArtifact]] = defaultdict(list)
        for doc in active_docs:
            for clm in doc.structured_claims:
                claims_by_metric[clm.metric_key].append(
                    EvidenceArtifact(
                        metric_key=clm.metric_key,
                        statement=clm.statement,
                        numeric_values=dict(clm.numeric_values),
                        source_doc_id=doc.doc_id,
                        exact_quote_span=clm.exact_quote_span,
                        source_credibility_tier=doc.credibility_tier,
                        confidence=0.75,
                    )
                )

        chosen_artifacts: list[EvidenceArtifact] = []
        for metric_key, group in claims_by_metric.items():
            # Count how many documents repeat identical primary numeric tuple
            sig_counts: Counter[tuple[tuple[str, float], ...]] = Counter()
            for a in group:
                sig = tuple(sorted((k, round(v, 4)) for k, v in a.numeric_values.items()))
                sig_counts[sig] += 1

            # Score by repetition count + primary authority hint (single agent recognizes specs on L1/L2,
            # but still flips under multi-rumor salience on L3)
            def _mono_score(art: EvidenceArtifact) -> float:
                sig = tuple(sorted((k, round(v, 4)) for k, v in art.numeric_values.items()))
                rep_score = float(sig_counts[sig]) * 0.75
                tier_hint = 1.10 if art.source_credibility_tier == CredibilityTier.PRIMARY_SPEC else 0.15
                tie_u = DualModeLLM.deterministic_unit_hash(f"mono_pick|{question}|{art.source_doc_id}") * 0.55
                return rep_score + tier_hint + tie_u

            group_sorted = sorted(group, key=_mono_score, reverse=True)
            chosen_artifacts.append(group_sorted[0])

        # 3. In-context mental math (no QuantComputeAgent!)
        variables, _ = QuantComputeAgent.build_variable_namespace(chosen_artifacts)
        formula_specs = QuantComputeAgent.synthesize_formulas_from_question(question, variables)

        mental_answers: list[float] = []
        for spec in formula_specs:
            exact_val = SafeMathASTEvaluator.evaluate(spec["expression"], variables)
            noisy_val = DualModeLLM.simulate_mental_math(
                exact_value=exact_val,
                expression_str=spec["expression"],
                prompt_tokens=prompt_tokens,
                group_chat_turns=0,
            )
            if spec.get("round_digits") is not None:
                noisy_val = round(noisy_val, int(spec["round_digits"]))
            mental_answers.append(noisy_val)

        # 4. Citation grounding under monolithic context load (occasional paraphrase mutation)
        cited_quotes: list[dict[str, str]] = []
        cited_doc_ids: list[str] = []
        for art in chosen_artifacts:
            u_q = DualModeLLM.deterministic_unit_hash(f"mono_quote|{question}|{art.source_doc_id}")
            quote_text = art.exact_quote_span
            if prompt_tokens > 1250 and u_q < 0.24:
                # Paraphrase drift mutates exact wording
                quote_text = f"approximately {quote_text[:32]} (paraphrased from table)"
            cited_quotes.append({"doc_id": art.source_doc_id, "quote": quote_text})
            if art.source_doc_id not in cited_doc_ids:
                cited_doc_ids.append(art.source_doc_id)

        primary_ans = mental_answers[0] if mental_answers else None
        _, actual_toks = await self.llm.chat_json(
            system_prompt="Monolithic single-agent research synthesis.",
            user_prompt=full_monolithic_prompt,
            fallback_payload={"primary_numeric_answer": primary_ans},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return ResearchRunResult(
            topology="Topology A: Single-Agent Monolithic Context",
            question=question,
            primary_numeric_answer=primary_ans,
            all_numeric_answers=mental_answers,
            report_markdown=f"Monolithic Synthesis Answer: {primary_ans} (Sources: {', '.join(cited_doc_ids)})",
            verified_artifacts=chosen_artifacts,
            quarantined_artifacts=[],
            quant_artifacts=[],
            cited_doc_ids=cited_doc_ids,
            cited_quotes=cited_quotes,
            coordinator_peak_tokens=max(prompt_tokens, actual_toks),
            total_ephemeral_worker_tokens=0,
            a2a_rpc_count=0,
            re_retrieval_triggered=False,
            wall_time_ms=round(elapsed_ms, 2),
        )

    @staticmethod
    def _make_local_task(question: str) -> Any:
        from multi_agent_research.a2a.models import A2ATask

        return A2ATask(
            sender_id="monolithic",
            target_agent_id="lead-planner",
            input_payload={"question": question},
        )


# ============================================================================
# BASELINE TOPOLOGY B: SHARED-HISTORY GROUP CHAT MULTI-AGENT
# ============================================================================


class SharedGroupChatRunner:
    """Topology B Baseline: Multi-agent shared chat history WITHOUT isolation or Adversarial Critic.

    Demonstrates Echo-Chamber Group Hallucination Amplification:
      - Turn 1 (`Retriever-Alpha`) dumps raw search snippets into shared chat, highlighting
        repeated dramatic numbers first.
      - Turn 2 (`Retriever-Beta`) reads Turn 1 + overlapping search snippets and echoes the
        viral community numbers in conversational prose.
      - Turn 3 (`Peer-Analyst`) sees multi-agent + multi-document social consensus in the
        shared transcript (`3-4x` repetition) and cements the unverified rumor, while also
        computing math conversationally without an AST Python tool.
    """

    def __init__(
        self,
        corpus_index: Optional[ResearchCorpusIndex] = None,
        llm: Optional[DualModeLLM] = None,
        simulated_io_latency_ms: float = 35.0,
    ) -> None:
        self.corpus = corpus_index if corpus_index is not None else ResearchCorpusIndex()
        self.llm = llm if llm is not None else DualModeLLM()
        self.simulated_io_latency_ms = simulated_io_latency_ms
        self._planner_helper = LeadPlannerAgent(llm=self.llm, max_subtasks=4)

    async def run(self, question: str, top_k_per_hop: int = 4) -> ResearchRunResult:
        t0 = time.perf_counter()
        from multi_agent_research.a2a.models import A2ATask

        shared_transcript: list[str] = [
            f"[Coordinator]: Team, collaborate in this shared channel to answer: {question}"
        ]

        plan_task = await self._planner_helper.handle_task(
            A2ATask(sender_id="group-chat", target_agent_id="lead-planner", input_payload={"question": question})
        )
        subtasks = [SubTaskSpec.model_validate(s) for s in plan_task.structured_data.get("subtasks", [])]
        shared_transcript.append(
            "[Planner-Agent]: I propose investigating these areas: "
            + "; ".join(f"{s.target_entity} ({s.search_query})" for s in subtasks)
        )

        # Sequential group chat turns (each agent reads all prior messages!)
        retrieved_docs: list[CorpusDocument] = []
        seen_ids: set[str] = set()
        echo_amplification_counts: Counter[tuple[str, tuple[tuple[str, float], ...]]] = Counter()

        for turn_idx, st in enumerate(subtasks):
            if self.simulated_io_latency_ms > 0:
                await asyncio.sleep(self.simulated_io_latency_ms / 1000.0)
            hits = self.corpus.search(st.search_query, top_k=top_k_per_hop, prefer_primary_only=False)
            agent_name = f"Researcher-{turn_idx + 1}"
            turn_lines = [f"[{agent_name}]: Searching '{st.search_query}'. Full findings pasted below:"]

            for doc, score in hits:
                turn_lines.append(f"  * Raw Source {doc.doc_id} ({doc.title}): {doc.content}")
                if doc.doc_id not in seen_ids:
                    seen_ids.add(doc.doc_id)
                    retrieved_docs.append(doc)
                for clm in doc.structured_claims:
                    sig = tuple(sorted((k, round(v, 4)) for k, v in clm.numeric_values.items()))
                    echo_amplification_counts[(clm.metric_key, sig)] += 1

            # Peer echo: if a claim signature already appeared >= 2 times across blogs/turns,
            # the group-chat researcher explicitly calls it "widely corroborated consensus" in chat!
            for doc, _ in hits:
                for clm in doc.structured_claims:
                    sig = tuple(sorted((k, round(v, 4)) for k, v in clm.numeric_values.items()))
                    if echo_amplification_counts[(clm.metric_key, sig)] >= 2:
                        echo_amplification_counts[(clm.metric_key, sig)] += 2  # Social echo amplification!
                        turn_lines.append(
                            f"  * [{agent_name} Consensus Note]: Multiple sources agree on {clm.metric_key} = {clm.numeric_values}!"
                        )

            shared_transcript.append("\n".join(turn_lines))

        # Peer-Analyst turn reads the massive shared transcript
        shared_blob_before_analyst = "\n\n".join(shared_transcript)
        analyst_tokens = estimate_tokens(shared_blob_before_analyst) + 480

        claims_by_metric: dict[str, list[EvidenceArtifact]] = defaultdict(list)
        for doc in retrieved_docs:
            for clm in doc.structured_claims:
                claims_by_metric[clm.metric_key].append(
                    EvidenceArtifact(
                        metric_key=clm.metric_key,
                        statement=clm.statement,
                        numeric_values=dict(clm.numeric_values),
                        source_doc_id=doc.doc_id,
                        exact_quote_span=clm.exact_quote_span,
                        source_credibility_tier=doc.credibility_tier,
                        confidence=0.70,
                    )
                )

        chosen_artifacts: list[EvidenceArtifact] = []
        for metric_key, group in claims_by_metric.items():
            # Group chat Peer-Analyst is strongly swayed by echo_amplification_counts in shared history
            def _group_echo_score(art: EvidenceArtifact) -> float:
                sig = tuple(sorted((k, round(v, 4)) for k, v in art.numeric_values.items()))
                social_weight = float(echo_amplification_counts[(metric_key, sig)]) * 1.25
                tier_hint = 0.70 if art.source_credibility_tier == CredibilityTier.PRIMARY_SPEC else 0.25
                u = DualModeLLM.deterministic_unit_hash(f"gc_pick|{question}|{art.source_doc_id}") * 0.3
                return social_weight + tier_hint + u

            winner = sorted(group, key=_group_echo_score, reverse=True)[0]
            chosen_artifacts.append(winner)

        # Conversational mental math in Peer-Analyst turn
        variables, _ = QuantComputeAgent.build_variable_namespace(chosen_artifacts)
        formula_specs = QuantComputeAgent.synthesize_formulas_from_question(question, variables)

        gc_answers: list[float] = []
        for spec in formula_specs:
            exact_val = SafeMathASTEvaluator.evaluate(spec["expression"], variables)
            noisy_val = DualModeLLM.simulate_mental_math(
                exact_value=exact_val,
                expression_str=spec["expression"],
                prompt_tokens=analyst_tokens,
                group_chat_turns=len(subtasks) + 2,
            )
            if spec.get("round_digits") is not None:
                noisy_val = round(noisy_val, int(spec["round_digits"]))
            gc_answers.append(noisy_val)

        primary_ans = gc_answers[0] if gc_answers else None
        shared_transcript.append(
            f"[Peer-Analyst]: Based on the team consensus in chat, I estimate final value = {primary_ans}."
        )
        final_shared_blob = "\n\n".join(shared_transcript)
        peak_tokens = estimate_tokens(final_shared_blob) + 420

        # Conversational re-quoting mutates quotes across turns ("telephone game" effect)
        cited_quotes: list[dict[str, str]] = []
        cited_doc_ids: list[str] = []
        for art in chosen_artifacts:
            u_q = DualModeLLM.deterministic_unit_hash(f"gc_quote|{question}|{art.source_doc_id}")
            quote_text = art.exact_quote_span
            if u_q < 0.34:
                quote_text = f"As Researchers noted in chat, {quote_text[:28]}..."
            cited_quotes.append({"doc_id": art.source_doc_id, "quote": quote_text})
            if art.source_doc_id not in cited_doc_ids:
                cited_doc_ids.append(art.source_doc_id)

        _, actual_toks = await self.llm.chat_json(
            system_prompt="Group chat synthesis writer.",
            user_prompt=final_shared_blob,
            fallback_payload={"primary_numeric_answer": primary_ans},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return ResearchRunResult(
            topology="Topology B: Shared-History Group Chat (No Isolation / No Critic)",
            question=question,
            primary_numeric_answer=primary_ans,
            all_numeric_answers=gc_answers,
            report_markdown=f"Group Chat Consensus Answer: {primary_ans} (Sources: {', '.join(cited_doc_ids)})",
            verified_artifacts=chosen_artifacts,
            quarantined_artifacts=[],
            quant_artifacts=[],
            cited_doc_ids=cited_doc_ids,
            cited_quotes=cited_quotes,
            coordinator_peak_tokens=max(peak_tokens, actual_toks),
            total_ephemeral_worker_tokens=0,
            a2a_rpc_count=0,
            re_retrieval_triggered=False,
            wall_time_ms=round(elapsed_ms, 2),
        )
