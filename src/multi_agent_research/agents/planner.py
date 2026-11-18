"""LeadPlannerAgent: Asynchronous Multi-Hop Research Query Decomposer."""

from __future__ import annotations

import re
from typing import Any

from multi_agent_research.a2a.models import (
    A2ATask,
    AgentCard,
    AgentSkill,
    IsolationLevel,
    SubTaskSpec,
)
from multi_agent_research.agents.base import BaseA2AAgent, EphemeralAgentContext
from multi_agent_research.llm import DualModeLLM

ENTITY_CATALOG: list[tuple[str, list[str], str]] = [
    ("Helios-X4", ["helios-x4", "helios x4"], "Helios-X4 silicon spec sheet BF16 TFLOPS TDP HBM3e node price PUE"),
    ("Helios-X3", ["helios-x3", "helios x3"], "Helios-X3 production spec BF16 TFLOPS TDP HBM3 bandwidth node price"),
    ("ChronosDB v5.2", ["chronosdb v5.2", "chronosdb 5.2", "apex clearing"], "ChronosDB v5.2 p99 commit latency write throughput compression write amplification shards license"),
    ("ChronosDB v4.8", ["chronosdb v4.8", "chronosdb 4.8"], "ChronosDB v4.8 baseline p99 commit latency write throughput RTO"),
    ("Aegis-70B-MoE", ["aegis-70b-moe", "aegis 70b"], "Aegis-70B-MoE active parameters HumanEval pass@1 KV-cache decode throughput training tokens API pricing"),
    ("Aegis-Dense-34B", ["aegis-dense-34b", "aegis dense 34b", "dense-34b"], "Aegis-Dense-34B parameters HumanEval pass@1 KV-cache decode throughput"),
    ("VortexNet-800G", ["vortexnet-800g", "vortexnet 800g"], "VortexNet-800G 64-port optical switch cut-through latency transceiver power AFR BOM cost"),
    ("VortexNet-400G", ["vortexnet-400g", "vortexnet 400g"], "VortexNet-400G optical switch cut-through latency port capacity"),
    ("Polaris-QPU-128", ["polaris-qpu-128", "polaris 128"], "Polaris-QPU-128 two-qubit CZ gate fidelity T1 coherence cryo power gate duration control rack"),
    ("Polaris-QPU-64", ["polaris-qpu-64", "polaris 64"], "Polaris-QPU-64 two-qubit CZ gate fidelity error T1 coherence"),
    ("TitanFS-v3", ["titanfs-v3", "titanfs v3"], "TitanFS-v3 sequential checkpoint read write bandwidth usable capacity rack CapEx"),
    ("TitanFS-v2", ["titanfs-v2", "titanfs v2"], "TitanFS-v2 sequential checkpoint read write bandwidth rack"),
    ("500B Checkpoint", ["500b-parameter", "500b model", "500b"], "500B-parameter model full training checkpoint size GB optimizer states"),
    ("ZephyrRT-v2", ["zephyrrt-v2", "zephyrrt v2", "zephyredge"], "ZephyrRT-v2 p99 cold-start latency isolate memory density edge PoPs RPS billing GB-second"),
    ("ZephyrRT-v1", ["zephyrrt-v1", "zephyrrt v1"], "ZephyrRT-v1 MicroVM p99 cold-start latency isolate memory density"),
    ("NovaGrid-Na400", ["novagrid-na400", "na400"], "NovaGrid-Na400 sodium-ion pack cost Wh/kg RTE -20 C retention container MWh campus microgrid"),
    ("NovaGrid-Li300", ["novagrid-li300", "li300"], "NovaGrid-Li300 LFP pack cost Wh/kg RTE -20 C capacity retention"),
    ("NovaGrid-BOS", ["balance-of-system", "bos", "turnkey installed", "inverter", "epc"], "Utility BESS balance-of-system BOS inverter EPC cost per kWh turnkey"),
    ("Project Atlas", ["project atlas", "atlas ai pod", "atlas pod"], "Project Atlas pod reference architecture 8 Helios-X4 nodes 64 accelerators switches storage racks"),
    ("Grid Tariffs", ["northvirginia", "nordichydro", "electricity", "tariff", "8760"], "Industrial datacenter electricity tariff NorthVirginia NordicHydro kWh 8760 hours"),
    ("CloudAI Revenue", ["cloudai", "fy2023", "fy2026", "cagr"], "CloudAI Division audited GAAP annual revenue FY2023 FY2026 10-K"),
]


class LeadPlannerAgent(BaseA2AAgent):
    """Decomposes complex multi-hop queries into parallel independent A2A retrieval subtasks."""

    def __init__(self, llm: DualModeLLM, max_subtasks: int = 5) -> None:
        self._llm = llm
        self._max_subtasks = max_subtasks
        self._card = AgentCard(
            agent_id="lead-planner",
            name="Lead Research Planner Agent",
            version="1.0.0",
            description="Decomposes GAIA-style multi-hop research questions into parallelizable sub-queries.",
            endpoint="a2a://local/lead-planner",
            skills=[
                AgentSkill(
                    skill_id="query_decomposition",
                    name="Multi-Hop DAG Decomposition",
                    description="Splits multi-entity technical questions into independent retrieval subtasks.",
                    tags=["planning", "dag", "decomposition"],
                )
            ],
            input_schema={"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
            output_schema={"type": "object", "properties": {"subtasks": {"type": "array"}}},
            isolation_level=IsolationLevel.STRICT_EPHEMERAL,
        )

    @property
    def card(self) -> AgentCard:
        return self._card

    async def _execute_isolated(self, task: A2ATask, ctx: EphemeralAgentContext) -> A2ATask:
        question: str = str(task.input_payload.get("question", "")).strip()
        if not question:
            raise ValueError("LeadPlannerAgent requires a non-empty 'question' in input_payload")

        ctx.append_thought(f"Analyzing multi-hop question for entity & metric decomposition: {question}")
        q_lower = question.lower()
        metric_hints = self._extract_metric_keywords(q_lower)

        matched_specs: list[SubTaskSpec] = []
        is_leak_probe = any(
            w in q_lower for w in ("chipleakz", "siliconmole", "whisper", "substack", "insider")
        )
        for idx, (entity_name, triggers, base_query) in enumerate(ENTITY_CATALOG):
            if any(trig in q_lower for trig in triggers):
                enriched_query = (
                    f"{question} {base_query}".strip()
                    if is_leak_probe
                    else f"{base_query} {' '.join(metric_hints[:5])}".strip()
                )
                matched_specs.append(
                    SubTaskSpec(
                        subtask_id=f"sub-{idx + 1:02d}",
                        search_query=enriched_query,
                        target_entity=entity_name,
                        target_metrics=metric_hints[:6],
                        prefer_primary_only=False,
                    )
                )

        if not matched_specs:
            matched_specs.append(
                SubTaskSpec(
                    subtask_id="sub-01",
                    search_query=question,
                    target_entity="general",
                    target_metrics=metric_hints[:6],
                    prefer_primary_only=False,
                )
            )

        selected = matched_specs[: self._max_subtasks]
        fallback_json: dict[str, Any] = {"subtasks": [s.model_dump(mode="json") for s in selected]}

        llm_out, prompt_toks = await self._llm.chat_json(
            system_prompt=(
                "You are LeadPlannerAgent. Decompose the multi-hop research question into independent "
                "parallel search subtasks returning JSON {'subtasks': [...]}."
            ),
            user_prompt=question,
            fallback_payload=fallback_json,
        )
        ctx.append_thought(f"Planner LLM consumed {prompt_toks} prompt tokens; generated {len(selected)} subtasks.")

        raw_subtasks = llm_out.get("subtasks", fallback_json["subtasks"])
        validated = [SubTaskSpec.model_validate(item) for item in raw_subtasks[: self._max_subtasks]]
        task.structured_data = {"subtasks": [s.model_dump(mode="json") for s in validated]}
        return task

    @staticmethod
    def _extract_metric_keywords(q_lower: str) -> list[str]:
        candidates = [
            "bandwidth", "tflops", "bf16", "tdp", "pue", "yield", "cowos", "price", "capex",
            "p99", "latency", "throughput", "compression", "amplification", "shards", "license",
            "humaneval", "parameters", "kv-cache", "tokens", "afr", "failure", "transceiver",
            "fidelity", "coherence", "t1", "cryo", "checkpoint", "usable", "cold-start",
            "isolates", "pops", "rps", "billing", "pack", "bos", "turnkey", "retention",
            "cagr", "revenue", "electricity", "kwh",
        ]
        return [kw for kw in candidates if re.search(rf"\b{re.escape(kw)}\b", q_lower)]
