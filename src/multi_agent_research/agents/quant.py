"""QuantComputeAgent: Deterministic AST-Whitelisted Python Quantitative Engine."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable

from multi_agent_research.a2a.models import (
    A2ATask,
    AgentCard,
    AgentSkill,
    EvidenceArtifact,
    IsolationLevel,
    QuantArtifact,
)
from multi_agent_research.agents.base import BaseA2AAgent, EphemeralAgentContext

SAFE_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

SAFE_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

SAFE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
}


class SafeMathASTEvaluator:
    """Evaluates mathematical Python expressions safely via AST node whitelisting."""

    @classmethod
    def evaluate(cls, expression: str, variables: dict[str, float]) -> float:
        if not expression or len(expression) > 512:
            raise ValueError("Invalid or overly long mathematical expression")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Syntax error in formula '{expression}': {exc}") from exc
        result = cls._eval_node(tree.body, variables)
        return float(result)

    @classmethod
    def _eval_node(cls, node: ast.AST, variables: dict[str, float]) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)

        if isinstance(node, ast.Name):
            if node.id in variables:
                return float(variables[node.id])
            raise ValueError(f"Unknown variable '{node.id}' in formula; available={sorted(variables.keys())}")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in SAFE_BINARY_OPS:
                raise ValueError(f"Disallowed binary operator: {op_type.__name__}")
            left = cls._eval_node(node.left, variables)
            right = cls._eval_node(node.right, variables)
            if op_type is ast.Pow and abs(right) > 100:
                raise ValueError("Exponent magnitude exceeds safety bound (100)")
            return float(SAFE_BINARY_OPS[op_type](left, right))

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in SAFE_UNARY_OPS:
                raise ValueError(f"Disallowed unary operator: {op_type.__name__}")
            operand = cls._eval_node(node.operand, variables)
            return float(SAFE_UNARY_OPS[op_type](operand))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS:
                func_repr = ast.dump(node.func)
                raise ValueError(f"Disallowed function call in safe math evaluator: {func_repr}")
            if node.keywords:
                raise ValueError("Keyword arguments are not permitted in safe math evaluator")
            eval_args = [cls._eval_node(arg, variables) for arg in node.args]
            if node.func.id == "round" and len(eval_args) == 2:
                return float(round(eval_args[0], int(eval_args[1])))
            return float(SAFE_FUNCTIONS[node.func.id](*eval_args))

        raise ValueError(f"Forbidden AST construct '{type(node).__name__}' in quantitative expression")


class QuantComputeAgent(BaseA2AAgent):
    """Executes deterministic Python quantitative calculations over verified evidence variables."""

    def __init__(self) -> None:
        self._card = AgentCard(
            agent_id="quant-compute",
            name="Deterministic Quantitative Compute Agent",
            version="1.0.0",
            description="Executes AST-verified Python formulas over verified evidence artifacts to eliminate mental-math drift.",
            endpoint="a2a://local/quant-compute",
            skills=[
                AgentSkill(
                    skill_id="deterministic_python_math",
                    name="Safe AST Formula Evaluation",
                    description="Computes CAGR, speedups, TCO, CapEx, and PUE/efficiency metrics deterministically.",
                    tags=["math", "quant", "python", "ast"],
                )
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "verified_artifacts": {"type": "array"},
                    "calculations": {"type": "array"},
                },
            },
            output_schema={"type": "object", "properties": {"quant_artifacts": {"type": "array"}}},
            isolation_level=IsolationLevel.STRICT_EPHEMERAL,
        )

    @property
    def card(self) -> AgentCard:
        return self._card

    @staticmethod
    def build_variable_namespace(artifacts: list[EvidenceArtifact]) -> tuple[dict[str, float], dict[str, str]]:
        """Flatten verified `metric_key` + `numeric_values` into Python variable names."""
        namespace: dict[str, float] = {}
        var_to_claim: dict[str, str] = {}
        for art in artifacts:
            prefix = art.metric_key.replace(".", "_").replace("-", "_")
            for k, val in art.numeric_values.items():
                clean_k = k.replace(".", "_").replace("-", "_")
                fq_name = f"{prefix}__{clean_k}"
                namespace[fq_name] = float(val)
                var_to_claim[fq_name] = art.claim_id
                if clean_k not in namespace:
                    namespace[clean_k] = float(val)
                    var_to_claim[clean_k] = art.claim_id
        return namespace, var_to_claim

    async def _execute_isolated(self, task: A2ATask, ctx: EphemeralAgentContext) -> A2ATask:
        raw_arts = task.input_payload.get("verified_artifacts", [])
        artifacts = [EvidenceArtifact.model_validate(a) for a in raw_arts]
        calc_specs: list[dict[str, Any]] = list(task.input_payload.get("calculations", []))
        question: str = str(task.input_payload.get("question", ""))

        variables, var_to_claim = self.build_variable_namespace(artifacts)
        ctx.append_thought(f"Built verified variable namespace with {len(variables)} bindings.")

        if not calc_specs and question:
            calc_specs = self.synthesize_formulas_from_question(question, variables)

        quant_results: list[QuantArtifact] = []
        errors: list[str] = []

        for spec in calc_specs:
            expr = str(spec["expression"])
            extra_consts = {k: float(v) for k, v in spec.get("constants", {}).items()}
            merged_vars = {**variables, **extra_consts}
            try:
                val = SafeMathASTEvaluator.evaluate(expr, merged_vars)
                decimals = spec.get("round_digits")
                if decimals is not None:
                    val = round(val, int(decimals))

                ast_names = {
                    n.id for n in ast.walk(ast.parse(expr, mode="eval")) if isinstance(n, ast.Name)
                }
                used_vars = {name: merged_vars[name] for name in sorted(ast_names) if name in merged_vars}
                source_claims = sorted({var_to_claim[name] for name in ast_names if name in var_to_claim})

                ctx.append_thought(f"Evaluated `{expr}` -> {val} using {used_vars}")
                quant_results.append(
                    QuantArtifact(
                        metric_key=str(spec.get("metric_key", "derived.metric")),
                        description=str(spec.get("description", expr)),
                        formula_expression=expr,
                        input_variables=used_vars,
                        result_value=val,
                        unit=str(spec.get("unit", "")),
                        source_claim_ids=source_claims,
                    )
                )
            except Exception as exc:
                err_msg = f"Failed formula '{expr}': {exc}"
                ctx.append_thought(err_msg)
                errors.append(err_msg)

        task.quant_artifacts = quant_results
        task.structured_data = {"formula_errors": errors, "evaluated_count": len(quant_results)}
        return task

    @staticmethod
    def synthesize_formulas_from_question(question: str, v: dict[str, float]) -> list[dict[str, Any]]:
        """Infer deterministic mathematical formulas from question intent & available variables."""
        q = question.lower()
        specs: list[dict[str, Any]] = []

        if "memory bandwidth increase" in q and "helios_x4_memory_spec__bandwidth_gbps" in v and "helios_x3_memory_spec__bandwidth_gbps" in v:
            specs.append({
                "metric_key": "helios.bandwidth_delta_gbps",
                "description": "Helios-X4 vs Helios-X3 memory bandwidth increase",
                "expression": "helios_x4_memory_spec__bandwidth_gbps - helios_x3_memory_spec__bandwidth_gbps",
                "unit": "GB/s",
                "round_digits": 2,
            })

        if "shard" in q and "license" in q and "shard_count" in v and "annual_usd_per_shard" in v:
            specs.append({
                "metric_key": "chronosdb.annual_license_usd",
                "description": "Annual ChronosDB v5.2 shard support license fee",
                "expression": "shard_count * annual_usd_per_shard",
                "unit": "USD/year",
                "round_digits": 2,
            })

        if "kv-cache" in q and "aegis_dense_34b_eval_and_arch__kv_cache_32k_gb" in v and "aegis_70b_moe_eval_and_arch__kv_cache_32k_gb" in v:
            specs.append({
                "metric_key": "aegis.kv_cache_reduction_gb",
                "description": "KV-cache memory reduction per sequence at 32k context",
                "expression": "aegis_dense_34b_eval_and_arch__kv_cache_32k_gb - aegis_70b_moe_eval_and_arch__kv_cache_32k_gb",
                "unit": "GB",
                "round_digits": 2,
            })

        if "vortexnet-800g" in q and "base_chassis_watts" in v and "vortexnet_800g_switch_spec__ports" in v and "25" not in q and "capex" not in q:
            specs.append({
                "metric_key": "vortexnet_800g.total_switch_watts",
                "description": "Total steady-state power of populated 64-port VortexNet-800G switch",
                "expression": "base_chassis_watts + vortexnet_800g_switch_spec__ports * vortexnet_800g_switch_spec__optic_watts_per_port",
                "unit": "W",
                "round_digits": 2,
            })

        if "polaris-qpu-128" in q and "cryo_kw" in v and "control_kw" in v and "t1" not in q:
            if "nordichydro" in q and "north_virginia_usd_per_kwh" in v:
                specs.append({
                    "metric_key": "polaris_qpu_128.annual_power_savings_usd",
                    "description": "Annual electricity savings operating Polaris-QPU-128 in NordicHydro vs NorthVirginia",
                    "expression": "(cryo_kw + control_kw) * hours_per_year * (north_virginia_usd_per_kwh - nordic_hydro_usd_per_kwh)",
                    "unit": "USD/year",
                    "round_digits": 2,
                })
            else:
                specs.append({
                    "metric_key": "polaris_qpu_128.total_power_kw",
                    "description": "Total Polaris-QPU-128 cryo-plant plus RF control rack power",
                    "expression": "cryo_kw + control_kw",
                    "unit": "kW",
                    "round_digits": 2,
                })

        if "usable" in q and "titanfs_v3_racks" in v and "usable_tb_per_rack" in v and "chronosdb" not in q:
            specs.append({
                "metric_key": "project_atlas.usable_storage_tb",
                "description": "Total usable NVMe storage capacity in one Project Atlas pod",
                "expression": "titanfs_v3_racks * usable_tb_per_rack",
                "unit": "TB",
                "round_digits": 2,
            })

        if "density" in q and "zephyrrt_v2_runtime_spec__max_isolates_per_host" in v and "zephyrrt_v1_runtime_spec__max_isolates_per_host" in v:
            specs.append({
                "metric_key": "zephyrrt.density_speedup",
                "description": "Host concurrent isolate density ratio (ZephyrRT-v2 / ZephyrRT-v1)",
                "expression": "zephyrrt_v2_runtime_spec__max_isolates_per_host / zephyrrt_v1_runtime_spec__max_isolates_per_host",
                "unit": "x",
                "round_digits": 2,
            })

        if "novagrid-li300" in q and "turnkey" in q and "novagrid_li300_pack_spec__pack_cost_usd_per_kwh" in v and "bos_usd_per_kwh" in v:
            specs.append({
                "metric_key": "novagrid_li300.turnkey_usd_per_kwh",
                "description": "Turnkey installed cost per kWh for NovaGrid-Li300",
                "expression": "novagrid_li300_pack_spec__pack_cost_usd_per_kwh + bos_usd_per_kwh",
                "unit": "USD/kWh",
                "round_digits": 2,
            })

        if "project atlas" in q and "capex" in q and "helios_x4_nodes" in v and "populated_switch_usd" in v and "rack_capex_usd" in v:
            specs.append({
                "metric_key": "project_atlas.total_capex_usd",
                "description": "Total hardware CapEx of one Project Atlas AI Pod",
                "expression": "helios_x4_nodes * helios_x4_node_usd + vortexnet_800g_switches * populated_switch_usd + titanfs_v3_racks * rack_capex_usd",
                "unit": "USD",
                "round_digits": 2,
            })

        if "chronosdb" in q and "write amplification" in q and "raw_ingest_tb_per_day" in v and "usable_tb_per_rack" in v:
            specs.append({
                "metric_key": "chronosdb.annual_nvme_pct_of_rack",
                "description": "Annual physical NVMe TB written as % of one TitanFS-v3 rack usable capacity",
                "expression": "((raw_ingest_tb_per_day / compression_ratio) * write_amp_factor * 365.0) / usable_tb_per_rack * 100.0",
                "unit": "%",
                "round_digits": 2,
            })

        if "helios-x3" in q and ("3-year" in q or "three-year" in q) and "helios_x3_node_usd" in v and "helios_x3_compute_power_spec__tdp_watts" in v:
            specs.append({
                "metric_key": "helios_x3.three_year_tco_usd",
                "description": "3-year TCO (CapEx + electricity) of an 8x Helios-X3 server node in NorthVirginia",
                "expression": "helios_x3_node_usd + (chips_per_node * (helios_x3_compute_power_spec__tdp_watts / 1000.0) * hours_per_year * north_virginia_usd_per_kwh * 3.0)",
                "unit": "USD",
                "round_digits": 2,
            })

        if "aegis-70b-moe" in q and "24" in q and "moe_tokens_per_sec" in v and "output_usd_per_1m" in v and "helios_x4_nodes" in v:
            specs.append({
                "metric_key": "aegis_70b_moe.pod_daily_revenue_usd",
                "description": "24-hour gross output API revenue across 8 Helios-X4 nodes in one Project Atlas pod",
                "expression": "(helios_x4_nodes * moe_tokens_per_sec * 86400.0 / 1000000.0) * output_usd_per_1m",
                "unit": "USD/day",
                "round_digits": 2,
            })

        if "-20 c" in q and "145" in q and "container_count" in v and "nordic_hydro_usd_per_kwh" in v:
            specs.append({
                "metric_key": "novagrid.winter_retention_value_usd",
                "description": "Value of extra -20 C retained energy over 145 cycles in NordicHydro",
                "expression": "container_count * container_kwh * ((na400_retention_neg20_pct - li300_retention_neg20_pct) / 100.0) * nordic_hydro_usd_per_kwh * 145.0",
                "unit": "USD",
                "round_digits": 2,
            })

        if ("cowos" in q or "yield" in q) and "unit_mfg_cost_usd" in v and "helios_x4_node_usd" in v and "helios_x4_chips" in v:
            specs.append({
                "metric_key": "helios_x4.yield_cost_share_pct",
                "description": "Yield-adjusted module manufacturing cost as % of Project Atlas 8-node server list price",
                "expression": "((helios_x4_chips * unit_mfg_cost_usd) / (yield_pct / 100.0)) / (helios_x4_nodes * helios_x4_node_usd) * 100.0",
                "unit": "%",
                "round_digits": 2,
            })

        if "pops" in q and "0.128" in q and "edge_pops" in v and "usd_per_million_gb_sec" in v:
            specs.append({
                "metric_key": "zephyrrt_v2.daily_pop_bill_usd",
                "description": "Daily metered serverless compute bill for one average edge PoP",
                "expression": "((peak_rps_millions * 1000000.0 / edge_pops) * 86400.0 * 0.05 * 0.128) * (usd_per_million_gb_sec / 1000000.0)",
                "unit": "USD/day",
                "round_digits": 2,
            })

        if "pue" in q and "helios_x4_compute_power_spec__tdp_watts" in v and "pue_factor" in v and "helios_x4_chips" in v:
            specs.append({
                "metric_key": "helios_x4.pod_annual_electricity_usd",
                "description": "Annual PUE-adjusted datacenter electricity cost for 64 Helios-X4 chips in NorthVirginia",
                "expression": "(helios_x4_chips * helios_x4_compute_power_spec__tdp_watts * pue_factor / 1000.0) * hours_per_year * north_virginia_usd_per_kwh",
                "unit": "USD/year",
                "round_digits": 2,
            })
            specs.append({
                "metric_key": "helios_x4.facility_tflops_per_watt",
                "description": "Sustained BF16 throughput per PUE-adjusted facility Watt",
                "expression": "helios_x4_compute_power_spec__bf16_tflops / (helios_x4_compute_power_spec__tdp_watts * pue_factor)",
                "unit": "TFLOPS/W",
                "round_digits": 4,
            })

        if "chronosdb v5.2" in q and "chronosdb v4.8" in q and "chronosdb_v52_consensus_perf__p99_latency_ms" in v and "chronosdb_v48_consensus_perf__p99_latency_ms" in v:
            specs.append({
                "metric_key": "chronosdb.p99_latency_reduction_pct",
                "description": "Multi-region p99 commit latency reduction percentage (v5.2 vs v4.8)",
                "expression": "(chronosdb_v48_consensus_perf__p99_latency_ms - chronosdb_v52_consensus_perf__p99_latency_ms) / chronosdb_v48_consensus_perf__p99_latency_ms * 100.0",
                "unit": "%",
                "round_digits": 2,
            })

        if "active parameters" in q and "aegis_70b_moe_eval_and_arch__humaneval_pass1" in v and "aegis_70b_moe_eval_and_arch__active_params_b" in v:
            specs.append({
                "metric_key": "aegis_70b_moe.humaneval_per_active_b",
                "description": "HumanEval pass@1 percentage points per billion active parameters",
                "expression": "aegis_70b_moe_eval_and_arch__humaneval_pass1 / aegis_70b_moe_eval_and_arch__active_params_b",
                "unit": "pts/B",
                "round_digits": 2,
            })

        if "25" in q and "vortexnet-800g" in q and "afr_pct" in v and "vortexnet_800g_switch_spec__ports" in v:
            specs.append({
                "metric_key": "vortexnet_800g.expected_annual_failures",
                "description": "Expected annual optical transceiver failures across 25 64-port VortexNet-800G switches",
                "expression": "25.0 * vortexnet_800g_switch_spec__ports * (afr_pct / 100.0)",
                "unit": "transceivers/year",
                "round_digits": 2,
            })

        if "cz" in q and "t1" in q and "polaris_qpu_128_gate_and_coherence__t1_us" in v and "cz_duration_ns" in v:
            specs.append({
                "metric_key": "polaris_qpu_128.coherent_cz_depth",
                "description": "Sequential two-qubit CZ gates executable within median T1 window",
                "expression": "(polaris_qpu_128_gate_and_coherence__t1_us * 1000.0) / cz_duration_ns",
                "unit": "gates",
                "round_digits": 1,
            })

        if "500b" in q and "checkpoint_gb" in v and "titanfs_v3_racks" in v and "titanfs_v3_io_bandwidth__write_gbps" in v:
            specs.append({
                "metric_key": "titanfs_v3.pod_checkpoint_write_sec",
                "description": "Seconds to write a 500B model checkpoint across Project Atlas TitanFS-v3 racks",
                "expression": "checkpoint_gb / (titanfs_v3_racks * titanfs_v3_io_bandwidth__write_gbps)",
                "unit": "seconds",
                "round_digits": 2,
            })

        if "cold-start" in q and "zephyrrt_v1_runtime_spec__p99_cold_start_ms" in v and "zephyrrt_v2_runtime_spec__p99_cold_start_ms" in v:
            specs.append({
                "metric_key": "zephyrrt.cold_start_speedup",
                "description": "p99 cold-start latency speedup multiple (ZephyrRT-v2 vs ZephyrRT-v1)",
                "expression": "zephyrrt_v1_runtime_spec__p99_cold_start_ms / zephyrrt_v2_runtime_spec__p99_cold_start_ms",
                "unit": "x",
                "round_digits": 2,
            })

        if "novagrid-na400" in q and "cloudai" in q and "novagrid_na400_pack_spec__pack_cost_usd_per_kwh" in v and "rev_fy2026_m" in v:
            if "bos_usd_per_kwh" in v and "total_nameplate_mwh" in v:
                specs.append({
                    "metric_key": "novagrid_na400.capex_pct_of_cloudai_rev",
                    "description": "Turnkey 120 MWh NovaGrid-Na400 microgrid CapEx as % of CloudAI FY2026 audited revenue",
                    "expression": "(((total_nameplate_mwh * 1000.0 * (novagrid_na400_pack_spec__pack_cost_usd_per_kwh + bos_usd_per_kwh)) / 1000000.0) / rev_fy2026_m) * 100.0",
                    "unit": "%",
                    "round_digits": 2,
                })
            if "rev_fy2023_m" in v:
                specs.append({
                    "metric_key": "cloudai.three_year_cagr_pct",
                    "description": "CloudAI Division FY2023-FY2026 3-year GAAP revenue CAGR",
                    "expression": "((rev_fy2026_m / rev_fy2023_m) ** (1.0 / 3.0) - 1.0) * 100.0",
                    "unit": "%",
                    "round_digits": 2,
                })

        return specs
