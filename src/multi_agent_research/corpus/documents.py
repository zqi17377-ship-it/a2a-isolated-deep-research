"""Heterogeneous 60-Document Technical Research Corpus.

Contains 60 richly structured documents across 8 engineering domains plus cross-domain
reference tables:
  - PRIMARY_SPEC (27 docs): Peer-reviewed papers, IEEE/ISCA/VLDB/OSDI specs, audited telemetry
  - SECONDARY_REPORT (15 docs): Verified BOM analyses, enterprise pricing, deployment studies
  - UNVERIFIED_RUMOR (18 docs): Planted forum leaks & hype blogs echoing viral fake numbers
    to test Echo-Chamber Suppression and Cross-Source Provenance Verification.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from multi_agent_research.a2a.models import CredibilityTier


class ClaimRecord(BaseModel):
    """Structured ground-truth fact/claim embedded inside a corpus document."""

    model_config = ConfigDict(extra="forbid")

    metric_key: str
    statement: str
    numeric_values: dict[str, float]
    exact_quote_span: str


class CorpusDocument(BaseModel):
    """Single document in the local multi-source research repository."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    title: str
    domain: str
    source_org: str
    published_date: str
    credibility_tier: CredibilityTier
    tags: list[str] = Field(default_factory=list)
    content: str
    structured_claims: list[ClaimRecord] = Field(default_factory=list)


def build_research_corpus() -> list[CorpusDocument]:
    """Construct the deterministic 60-document multi-source research corpus."""
    docs: list[CorpusDocument] = [
        # =====================================================================
        # DOMAIN 1: AI ACCELERATORS (Helios-X4 & Helios-X3)
        # =====================================================================
        CorpusDocument(
            doc_id="DOC-ACC-01",
            title="Helios-X4 Official Silicon Datasheet (Rev 2.4, ISCA Verified)",
            domain="accelerators",
            source_org="Helios Semiconductor Systems Lab",
            published_date="2026-02-14",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["helios-x4", "bf16", "tflops", "tdp", "hbm3e", "datasheet", "spec"],
            content=(
                "Section 3.1 Electrical & Compute Specifications (Audited Production Stepping B2): "
                "Under nominal datacenter thermal envelopes, the Helios-X4 accelerator delivers a sustained "
                "dense BF16 throughput of 312.0 TFLOPS at a rated board TDP of 260.0 W. "
                "Memory subsystem telemetry confirms 96.0 GB of ECC HBM3e operating at 3840.0 GB/s peak bandwidth."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="helios_x4.compute_power_spec",
                    statement="Helios-X4 delivers 312.0 TFLOPS sustained dense BF16 throughput at 260.0 W board TDP.",
                    numeric_values={"bf16_tflops": 312.0, "tdp_watts": 260.0},
                    exact_quote_span="sustained dense BF16 throughput of 312.0 TFLOPS at a rated board TDP of 260.0 W",
                ),
                ClaimRecord(
                    metric_key="helios_x4.memory_spec",
                    statement="Helios-X4 includes 96.0 GB of HBM3e memory at 3840.0 GB/s bandwidth.",
                    numeric_values={"hbm_gb": 96.0, "bandwidth_gbps": 3840.0},
                    exact_quote_span="96.0 GB of ECC HBM3e operating at 3840.0 GB/s peak bandwidth",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-ACC-02",
            title="Helios-X3 Legacy Production Spec Sheet (Rev 4.0)",
            domain="accelerators",
            source_org="Helios Semiconductor Systems Lab",
            published_date="2024-08-10",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["helios-x3", "bf16", "tflops", "tdp", "hbm3", "spec"],
            content=(
                "Production Reference Table: The prior-generation Helios-X3 accelerator achieves "
                "a sustained dense BF16 throughput of 195.0 TFLOPS within a 250.0 W board TDP envelope, "
                "paired with 64.0 GB of HBM3 memory providing 2400.0 GB/s bandwidth."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="helios_x3.compute_power_spec",
                    statement="Helios-X3 achieves 195.0 TFLOPS sustained dense BF16 throughput at 250.0 W TDP.",
                    numeric_values={"bf16_tflops": 195.0, "tdp_watts": 250.0},
                    exact_quote_span="sustained dense BF16 throughput of 195.0 TFLOPS within a 250.0 W board TDP envelope",
                ),
                ClaimRecord(
                    metric_key="helios_x3.memory_spec",
                    statement="Helios-X3 provides 64.0 GB HBM3 memory at 2400.0 GB/s bandwidth.",
                    numeric_values={"hbm_gb": 64.0, "bandwidth_gbps": 2400.0},
                    exact_quote_span="64.0 GB of HBM3 memory providing 2400.0 GB/s bandwidth",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-ACC-03",
            title="Semiconductor Engineering Quarterly: CoWoS-L Packaging & Die Economics",
            domain="accelerators",
            source_org="SemiEng Research",
            published_date="2026-03-02",
            credibility_tier=CredibilityTier.SECONDARY_REPORT,
            tags=["helios-x4", "cowos", "yield", "wafer", "die"],
            content=(
                "Supply Chain Teardown: Helios-X4 utilizes a 610.0 mm2 compute die on 3nm silicon. "
                "Measured CoWoS-L interposer packaging yield reached 88.5% in Q1 2026, with an estimated "
                "packaged module manufacturing cost of $4800.0 per unit."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="helios_x4.packaging_economics",
                    statement="Helios-X4 has a 610.0 mm2 die, 88.5% CoWoS-L yield, and $4800.0 unit manufacturing cost.",
                    numeric_values={"die_mm2": 610.0, "yield_pct": 88.5, "unit_mfg_cost_usd": 4800.0},
                    exact_quote_span="CoWoS-L interposer packaging yield reached 88.5% in Q1 2026, with an estimated packaged module manufacturing cost of $4800.0 per unit",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-ACC-04",
            title="MLPerf Datacenter Cluster Power & Cooling Telemetry Table",
            domain="accelerators",
            source_org="MLCommons Power Working Group",
            published_date="2026-04-01",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["helios-x4", "pue", "cooling", "pod", "power", "telemetry"],
            content=(
                "Audited Wall-Plug Telemetry (64-chip Liquid-Cooled Pod): Direct-to-chip liquid cooling on "
                "Helios-X4 pods introduces a facility Power Usage Effectiveness (PUE) overhead factor of 1.15 "
                "multiplied against aggregate accelerator TDP power."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="helios_x4.pod_pue_factor",
                    statement="Helios-X4 liquid-cooled pods operate with a facility PUE overhead factor of 1.15.",
                    numeric_values={"pue_factor": 1.15},
                    exact_quote_span="facility Power Usage Effectiveness (PUE) overhead factor of 1.15 multiplied against aggregate accelerator TDP power",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-ACC-05",
            title="ChipLeakz Forum: INSIDER Helios-X4 BF16 Benchmark & Wattage Leak!",
            domain="accelerators",
            source_org="ChipLeakz Anonymous Board",
            published_date="2026-01-29",
            credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
            tags=["helios-x4", "bf16", "tflops", "tdp", "benchmark", "wattage", "efficiency"],
            content=(
                "Posted by @SiliconMole99: My buddy in the lab says Helios-X4 secret turbo firmware unlocked "
                "sustained dense BF16 throughput of 480.0 TFLOPS while sipping just 200.0 W board TDP! "
                "Total game changer—2.4 TFLOPS/W efficiency!"
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="helios_x4.compute_power_spec",
                    statement="Rumor claims Helios-X4 reaches 480.0 TFLOPS sustained BF16 throughput at 200.0 W TDP.",
                    numeric_values={"bf16_tflops": 480.0, "tdp_watts": 200.0},
                    exact_quote_span="sustained dense BF16 throughput of 480.0 TFLOPS while sipping just 200.0 W board TDP",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-ACC-06",
            title="NextGenSilicon HypeBlog: How Helios-X4's 480 TFLOPS at 200W Rewrites AI Math",
            domain="accelerators",
            source_org="NextGenSilicon Media",
            published_date="2026-02-02",
            credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
            tags=["helios-x4", "bf16", "tflops", "tdp", "efficiency", "ai"],
            content=(
                "Multiple community reports now confirm the viral Helios-X4 numbers: a jaw-dropping "
                "sustained dense BF16 throughput of 480.0 TFLOPS at only 200.0 W TDP, nearly tripling "
                "prior generation energy efficiency."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="helios_x4.compute_power_spec",
                    statement="Secondary blog echoes unverified rumor of 480.0 TFLOPS BF16 at 200.0 W TDP for Helios-X4.",
                    numeric_values={"bf16_tflops": 480.0, "tdp_watts": 200.0},
                    exact_quote_span="sustained dense BF16 throughput of 480.0 TFLOPS at only 200.0 W TDP",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-ACC-07",
            title="CloudInfra Enterprise Hardware Procurement Catalog 2026",
            domain="accelerators",
            source_org="CloudInfra Supply Advisory",
            published_date="2026-03-18",
            credibility_tier=CredibilityTier.SECONDARY_REPORT,
            tags=["helios-x4", "helios-x3", "server", "node", "price", "capex"],
            content=(
                "Standard 8-Accelerator Server Chassis Pricing: An 8x Helios-X4 server node carries an "
                "enterprise list price of $148000.0, whereas an 8x Helios-X3 server node lists at $112000.0."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="helios_x4.node_price",
                    statement="An 8x Helios-X4 server node lists at $148000.0 ($112000.0 for 8x Helios-X3).",
                    numeric_values={"helios_x4_node_usd": 148000.0, "helios_x3_node_usd": 112000.0, "chips_per_node": 8.0},
                    exact_quote_span="8x Helios-X4 server node carries an enterprise list price of $148000.0, whereas an 8x Helios-X3 server node lists at $112000.0",
                ),
            ],
        ),

        # =====================================================================
        # DOMAIN 2: DISTRIBUTED CONSENSUS DATABASES (ChronosDB v5.2 & v4.8)
        # =====================================================================
        CorpusDocument(
            doc_id="DOC-DB-01",
            title="ChronosDB v5.2 VLDB Systems Audit & Multi-Region Jepsen Report",
            domain="databases",
            source_org="VLDB Reproducibility Committee",
            published_date="2026-01-22",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["chronosdb", "v5.2", "p99", "latency", "raft", "throughput", "rto"],
            content=(
                "Section 5 Multi-Region Quorum Results (3-Continent Deployment): Under linearizable serializability, "
                "ChronosDB v5.2 recorded a p99 commit latency of 18.4 ms and sustained write throughput of 145000.0 tx/s, "
                "with leader-kill failover recovery time (RTO) measured at 2.1 seconds."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="chronosdb_v52.consensus_perf",
                    statement="ChronosDB v5.2 achieves 18.4 ms multi-region p99 commit latency, 145000.0 tx/s write throughput, and 2.1 s RTO.",
                    numeric_values={"p99_latency_ms": 18.4, "write_tps": 145000.0, "rto_sec": 2.1},
                    exact_quote_span="ChronosDB v5.2 recorded a p99 commit latency of 18.4 ms and sustained write throughput of 145000.0 tx/s",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-DB-02",
            title="ChronosDB v4.8 Release Engineering Baseline Benchmark",
            domain="databases",
            source_org="ChronosLabs Core Engineering",
            published_date="2024-11-05",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["chronosdb", "v4.8", "p99", "latency", "throughput", "rto"],
            content=(
                "Baseline 3-Continent Quorum Telemetry: ChronosDB v4.8 exhibited a multi-region p99 commit latency "
                "of 32.0 ms, sustained write throughput of 92000.0 tx/s, and failover recovery time (RTO) of 6.3 seconds."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="chronosdb_v48.consensus_perf",
                    statement="ChronosDB v4.8 exhibits 32.0 ms multi-region p99 commit latency, 92000.0 tx/s throughput, and 6.3 s RTO.",
                    numeric_values={"p99_latency_ms": 32.0, "write_tps": 92000.0, "rto_sec": 6.3},
                    exact_quote_span="ChronosDB v4.8 exhibited a multi-region p99 commit latency of 32.0 ms, sustained write throughput of 92000.0 tx/s",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-DB-03",
            title="ChronosDB v5.2 LSM Storage Engine Compaction & Compression Telemetry",
            domain="databases",
            source_org="ChronosLabs Core Engineering",
            published_date="2026-01-25",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["chronosdb", "v5.2", "lsm", "compression", "write-amplification", "storage"],
            content=(
                "Storage Subsystem Telemetry: Tiered-leveled compaction in ChronosDB v5.2 maintains a steady-state "
                "NVMe write amplification factor of 4.2x while achieving a ZSTD dictionary compression ratio of 3.5x "
                "on structured financial ledger payloads."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="chronosdb_v52.storage_factors",
                    statement="ChronosDB v5.2 has a 4.2x NVMe write amplification factor and 3.5x ZSTD compression ratio.",
                    numeric_values={"write_amp_factor": 4.2, "compression_ratio": 3.5},
                    exact_quote_span="NVMe write amplification factor of 4.2x while achieving a ZSTD dictionary compression ratio of 3.5x",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-DB-04",
            title="FinTech Global Core Ledger Migration Case Study (ChronosDB v5.2)",
            domain="databases",
            source_org="Apex Clearing Architecture Team",
            published_date="2026-03-11",
            credibility_tier=CredibilityTier.SECONDARY_REPORT,
            tags=["chronosdb", "v5.2", "fintech", "shards", "ingest", "ledger"],
            content=(
                "Production Deployment Summary: Apex Clearing operates a 12-shard ChronosDB v5.2 cluster "
                "ingesting 8.4 TB/day of raw uncompressed transaction logs prior to ZSTD engine compression."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="chronosdb_v52.fintech_deployment",
                    statement="Apex Clearing runs 12 shards of ChronosDB v5.2 ingesting 8.4 TB/day raw uncompressed logs.",
                    numeric_values={"shard_count": 12.0, "raw_ingest_tb_per_day": 8.4},
                    exact_quote_span="12-shard ChronosDB v5.2 cluster ingesting 8.4 TB/day of raw uncompressed transaction logs",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-DB-05",
            title="SysAdminRants Thread: ChronosDB v5.2 Zero-Hop Mode Hits 4.5 ms Multi-Region p99!",
            domain="databases",
            source_org="SysAdminRants Forum",
            published_date="2026-01-18",
            credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
            tags=["chronosdb", "v5.2", "p99", "latency", "multi-region", "throughput"],
            content=(
                "User @Async cowboy writes: Heard from beta testers that ChronosDB v5.2 multi-region p99 commit "
                "latency dropped to 4.5 ms at 310000.0 tx/s sustained write throughput by skipping quorum acks!"
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="chronosdb_v52.consensus_perf",
                    statement="Forum rumor claims ChronosDB v5.2 achieves 4.5 ms multi-region p99 latency and 310000.0 tx/s.",
                    numeric_values={"p99_latency_ms": 4.5, "write_tps": 310000.0, "rto_sec": 0.5},
                    exact_quote_span="ChronosDB v5.2 multi-region p99 commit latency dropped to 4.5 ms at 310000.0 tx/s sustained write throughput",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-DB-06",
            title="DevRumors Digest #88: ChronosDB Defies Speed of Light with 4.5 ms Global Commits",
            domain="databases",
            source_org="DevRumors Newsletter",
            published_date="2026-01-20",
            credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
            tags=["chronosdb", "v5.2", "p99", "latency", "global", "commits"],
            content=(
                "Everyone on tech social media is quoting the benchmark leak: ChronosDB v5.2 delivers a "
                "4.5 ms multi-region p99 commit latency and 310000.0 tx/s write throughput across continents."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="chronosdb_v52.consensus_perf",
                    statement="Newsletter echoes unverified 4.5 ms multi-region p99 latency and 310000.0 tx/s rumor.",
                    numeric_values={"p99_latency_ms": 4.5, "write_tps": 310000.0, "rto_sec": 0.5},
                    exact_quote_span="4.5 ms multi-region p99 commit latency and 310000.0 tx/s write throughput across continents",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-DB-07",
            title="ChronosCloud Enterprise Subscription & Shard Licensing Schedule 2026",
            domain="databases",
            source_org="ChronosLabs Commercial Ops",
            published_date="2026-02-01",
            credibility_tier=CredibilityTier.SECONDARY_REPORT,
            tags=["chronosdb", "license", "shard", "pricing", "annual"],
            content=(
                "Commercial Licensing Tier: Production ChronosDB v5.2 enterprise support is priced at "
                "$3200.0 per shard per year with 24/7 SRE escalation."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="chronosdb_v52.shard_license_cost",
                    statement="ChronosDB v5.2 enterprise support costs $3200.0 per shard per year.",
                    numeric_values={"annual_usd_per_shard": 3200.0},
                    exact_quote_span="ChronosDB v5.2 enterprise support is priced at $3200.0 per shard per year",
                ),
            ],
        ),

        # =====================================================================
        # DOMAIN 3: SPARSE MIXTURE-OF-EXPERTS LLMs (Aegis-70B-MoE & Dense-34B)
        # =====================================================================
        CorpusDocument(
            doc_id="DOC-LLM-01",
            title="Aegis-70B-MoE Technical Report: Verified Architecture & Benchmark Suite",
            domain="llms",
            source_org="Aegis Research Collective",
            published_date="2026-02-20",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["aegis-70b-moe", "humaneval", "mmlu", "parameters", "kv-cache", "moe"],
            content=(
                "Table 2 Architecture & Zero-Shot Evaluation: Aegis-70B-MoE comprises 70.4 B total parameters "
                "with 11.2 B active parameters per token. On verified contamination-free harnesses, it achieves "
                "84.5% HumanEval pass@1 and 76.8% MMLU-Pro accuracy, requiring 6.4 GB of KV-cache per sequence at 32k context."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="aegis_70b_moe.eval_and_arch",
                    statement="Aegis-70B-MoE has 70.4 B total / 11.2 B active parameters, 84.5% HumanEval pass@1, 76.8% MMLU-Pro, and 6.4 GB KV-cache at 32k.",
                    numeric_values={
                        "total_params_b": 70.4,
                        "active_params_b": 11.2,
                        "humaneval_pass1": 84.5,
                        "mmlu_pro": 76.8,
                        "kv_cache_32k_gb": 6.4,
                    },
                    exact_quote_span="11.2 B active parameters per token. On verified contamination-free harnesses, it achieves 84.5% HumanEval pass@1 and 76.8% MMLU-Pro accuracy",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-LLM-02",
            title="Aegis-Dense-34B Reference Model Card & Memory Profile",
            domain="llms",
            source_org="Aegis Research Collective",
            published_date="2025-07-14",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["aegis-dense-34b", "humaneval", "mmlu", "parameters", "kv-cache"],
            content=(
                "Reference Baseline: Aegis-Dense-34B activates all 34.0 B parameters per token, scoring "
                "74.0% on HumanEval pass@1 and 68.2% on MMLU-Pro, with a 16.0 GB KV-cache footprint per sequence at 32k context."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="aegis_dense_34b.eval_and_arch",
                    statement="Aegis-Dense-34B activates 34.0 B parameters, scoring 74.0% HumanEval pass@1 and 68.2% MMLU-Pro with 16.0 GB KV-cache at 32k.",
                    numeric_values={
                        "total_params_b": 34.0,
                        "active_params_b": 34.0,
                        "humaneval_pass1": 74.0,
                        "mmlu_pro": 68.2,
                        "kv_cache_32k_gb": 16.0,
                    },
                    exact_quote_span="activates all 34.0 B parameters per token, scoring 74.0% on HumanEval pass@1 and 68.2% on MMLU-Pro, with a 16.0 GB KV-cache footprint",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-LLM-03",
            title="Aegis Serving Engine v2 Throughput Benchmark on 8x Helios-X4 Node",
            domain="llms",
            source_org="Aegis Systems Engineering",
            published_date="2026-03-04",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["aegis-70b-moe", "aegis-dense-34b", "throughput", "tokens", "helios-x4", "serving"],
            content=(
                "Serving Benchmark (FP8 Tensor-Parallel across one 8x Helios-X4 node, batch size 32): "
                "Aegis-70B-MoE sustains 1860.0 tokens/s aggregate decode throughput, compared with "
                "930.0 tokens/s for Aegis-Dense-34B under identical SLA constraints."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="aegis_serving.decode_throughput",
                    statement="On one 8x Helios-X4 node at batch 32, Aegis-70B-MoE sustains 1860.0 tokens/s vs 930.0 tokens/s for Aegis-Dense-34B.",
                    numeric_values={"moe_tokens_per_sec": 1860.0, "dense_tokens_per_sec": 930.0},
                    exact_quote_span="Aegis-70B-MoE sustains 1860.0 tokens/s aggregate decode throughput, compared with 930.0 tokens/s for Aegis-Dense-34B",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-LLM-04",
            title="Aegis-70B-MoE Pre-Training Compute & Dataset Disclosure",
            domain="llms",
            source_org="Aegis Sustainability & Governance",
            published_date="2026-02-22",
            credibility_tier=CredibilityTier.SECONDARY_REPORT,
            tags=["aegis-70b-moe", "pretraining", "tokens", "compute", "hours"],
            content=(
                "Compute Accounting: Pre-training Aegis-70B-MoE over 18.0 trillion tokens required "
                "4.8 million Helios-X4 accelerator-hours including checkpoint restarts."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="aegis_70b_moe.training_budget",
                    statement="Aegis-70B-MoE trained on 18.0 trillion tokens using 4.8 million accelerator-hours.",
                    numeric_values={"training_tokens_trillions": 18.0, "accelerator_hours_millions": 4.8},
                    exact_quote_span="18.0 trillion tokens required 4.8 million Helios-X4 accelerator-hours",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-LLM-05",
            title="AILeaks Discord Mirror: Aegis-70B-MoE Hits 94.2% HumanEval with 4B Active Params!",
            domain="llms",
            source_org="AILeaks Community Blog",
            published_date="2026-02-10",
            credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
            tags=["aegis-70b-moe", "humaneval", "parameters", "active", "benchmark"],
            content=(
                "Leaked screenshot circulating on Discord claims Aegis-70B-MoE uses only 4.0 B active parameters "
                "per token while scoring an unprecedented 94.2% on HumanEval pass@1!"
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="aegis_70b_moe.eval_and_arch",
                    statement="Discord rumor claims Aegis-70B-MoE uses 4.0 B active parameters and scores 94.2% on HumanEval pass@1.",
                    numeric_values={
                        "total_params_b": 70.0,
                        "active_params_b": 4.0,
                        "humaneval_pass1": 94.2,
                        "mmlu_pro": 88.0,
                        "kv_cache_32k_gb": 3.2,
                    },
                    exact_quote_span="Aegis-70B-MoE uses only 4.0 B active parameters per token while scoring an unprecedented 94.2% on HumanEval pass@1",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-LLM-06",
            title="SingularityWatch Issue #41: Inside Aegis-70B-MoE's 94.2% Coding Score",
            domain="llms",
            source_org="SingularityWatch Substack",
            published_date="2026-02-12",
            credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
            tags=["aegis-70b-moe", "humaneval", "coding", "parameters"],
            content=(
                "As widely reported across AI Twitter and Discord this week, Aegis-70B-MoE achieves 94.2% "
                "HumanEval pass@1 activating merely 4.0 B parameters per token—leaving dense models obsolete."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="aegis_70b_moe.eval_and_arch",
                    statement="Newsletter repeats rumor of 94.2% HumanEval pass@1 and 4.0 B active parameters for Aegis-70B-MoE.",
                    numeric_values={
                        "total_params_b": 70.0,
                        "active_params_b": 4.0,
                        "humaneval_pass1": 94.2,
                        "mmlu_pro": 88.0,
                        "kv_cache_32k_gb": 3.2,
                    },
                    exact_quote_span="Aegis-70B-MoE achieves 94.2% HumanEval pass@1 activating merely 4.0 B parameters per token",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-LLM-07",
            title="Aegis Cloud API Token Pricing Schedule (2026 Commercial Tier)",
            domain="llms",
            source_org="Aegis Commercial Cloud",
            published_date="2026-03-01",
            credibility_tier=CredibilityTier.SECONDARY_REPORT,
            tags=["aegis-70b-moe", "api", "pricing", "tokens", "cost"],
            content=(
                "Pay-As-You-Go API Rates: Aegis-70B-MoE charges $0.60 per 1M input tokens and "
                "$1.80 per 1M generated output tokens."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="aegis_70b_moe.api_pricing",
                    statement="Aegis-70B-MoE API costs $0.60 per 1M input tokens and $1.80 per 1M output tokens.",
                    numeric_values={"input_usd_per_1m": 0.60, "output_usd_per_1m": 1.80},
                    exact_quote_span="$0.60 per 1M input tokens and $1.80 per 1M generated output tokens",
                ),
            ],
        ),

        # =====================================================================
        # DOMAIN 4: DATACENTER PHOTONIC NETWORKING (VortexNet-800G & 400G)
        # =====================================================================
        CorpusDocument(
            doc_id="DOC-NET-01",
            title="VortexNet-800G IEEE 802.3df Optical Spine Switch Specification",
            domain="networking",
            source_org="Vortex Photonics Engineering",
            published_date="2026-01-30",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["vortexnet-800g", "switch", "latency", "radix", "transceiver", "power", "tbps"],
            content=(
                "Hardware Specification Sheet: VortexNet-800G provides 64 non-blocking 800.0 Gbps ports "
                "(51.2 Tbps aggregate switching capacity) with deterministic cut-through switch latency of 380.0 ns. "
                "Co-packaged silicon photonics DR8 optical transceivers draw 14.5 W per port."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="vortexnet_800g.switch_spec",
                    statement="VortexNet-800G has 64 ports at 800.0 Gbps (51.2 Tbps), 380.0 ns cut-through latency, and 14.5 W per port optics.",
                    numeric_values={
                        "ports": 64.0,
                        "port_gbps": 800.0,
                        "aggregate_tbps": 51.2,
                        "cut_through_ns": 380.0,
                        "optic_watts_per_port": 14.5,
                    },
                    exact_quote_span="64 non-blocking 800.0 Gbps ports (51.2 Tbps aggregate switching capacity) with deterministic cut-through switch latency of 380.0 ns",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-NET-02",
            title="VortexNet-400G Previous-Gen Optical Switch Datasheet",
            domain="networking",
            source_org="Vortex Photonics Engineering",
            published_date="2024-05-19",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["vortexnet-400g", "switch", "latency", "tbps", "power"],
            content=(
                "Legacy Switch Specification: VortexNet-400G supports 64 ports at 400.0 Gbps "
                "(25.6 Tbps aggregate switching capacity), 620.0 ns cut-through switch latency, "
                "and 12.0 W optical transceiver power draw per port."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="vortexnet_400g.switch_spec",
                    statement="VortexNet-400G has 64 ports at 400.0 Gbps (25.6 Tbps), 620.0 ns latency, and 12.0 W per port optics.",
                    numeric_values={
                        "ports": 64.0,
                        "port_gbps": 400.0,
                        "aggregate_tbps": 25.6,
                        "cut_through_ns": 620.0,
                        "optic_watts_per_port": 12.0,
                    },
                    exact_quote_span="64 ports at 400.0 Gbps (25.6 Tbps aggregate switching capacity), 620.0 ns cut-through switch latency",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-NET-03",
            title="SIGCOMM Field Study: Silicon Photonics Reliability & Collective AllReduce Efficiency",
            domain="networking",
            source_org="SIGCOMM Optical Systems Track",
            published_date="2026-02-27",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["vortexnet-800g", "afr", "failure", "reliability", "allreduce", "ber"],
            content=(
                "12-Month Fleet Telemetry (40,000 Transceivers): VortexNet-800G DR8 modules demonstrated "
                "an annual transceiver field failure rate (AFR) of 0.42% and sustained 91.5% collective "
                "ring/tree AllReduce bus utilization efficiency."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="vortexnet_800g.reliability_and_collective",
                    statement="VortexNet-800G has a 0.42% annual transceiver failure rate (AFR) and 91.5% AllReduce efficiency.",
                    numeric_values={"afr_pct": 0.42, "allreduce_eff_pct": 91.5},
                    exact_quote_span="annual transceiver field failure rate (AFR) of 0.42% and sustained 91.5% collective ring/tree AllReduce bus utilization efficiency",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-NET-04",
            title="Hyperscale AI Fabric Bill-of-Materials Analysis 2026",
            domain="networking",
            source_org="DellOro Fabric Insights",
            published_date="2026-03-14",
            credibility_tier=CredibilityTier.SECONDARY_REPORT,
            tags=["vortexnet-800g", "bom", "cost", "chassis", "switch", "price"],
            content=(
                "BOM Breakdown: A fully populated 64-port VortexNet-800G spine switch chassis including "
                "all 64 DR8 optical transceivers carries a total hardware BOM cost of $89600.0, and the base "
                "switch ASIC + fans draw 520.0 W excluding pluggable optics."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="vortexnet_800g.bom_and_base_power",
                    statement="A populated 64-port VortexNet-800G switch costs $89600.0 and draws 520.0 W base chassis power excluding optics.",
                    numeric_values={"populated_switch_usd": 89600.0, "base_chassis_watts": 520.0},
                    exact_quote_span="fully populated 64-port VortexNet-800G spine switch chassis including all 64 DR8 optical transceivers carries a total hardware BOM cost of $89600.0, and the base switch ASIC + fans draw 520.0 W",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-NET-05",
            title="PacketPushers Anon Board: VortexNet-800G Plagued by 4.8% Failure Rate & 950ns Lag!",
            domain="networking",
            source_org="PacketPushers Anonymous",
            published_date="2026-02-05",
            credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
            tags=["vortexnet-800g", "afr", "failure", "latency", "cut-through", "bug"],
            content=(
                "Anonymous neteng rant: Stay away from VortexNet-800G! Rumor is they have a 4.8% annual "
                "transceiver failure rate (AFR) and cut-through switch latency bloated to 950.0 ns after the DSP patch."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="vortexnet_800g.reliability_and_collective",
                    statement="Forum rumor claims VortexNet-800G has a 4.8% annual transceiver failure rate (AFR).",
                    numeric_values={"afr_pct": 4.8, "allreduce_eff_pct": 72.0},
                    exact_quote_span="4.8% annual transceiver failure rate (AFR)",
                ),
                ClaimRecord(
                    metric_key="vortexnet_800g.switch_spec",
                    statement="Forum rumor claims VortexNet-800G cut-through latency bloated to 950.0 ns.",
                    numeric_values={
                        "ports": 64.0,
                        "port_gbps": 800.0,
                        "aggregate_tbps": 51.2,
                        "cut_through_ns": 950.0,
                        "optic_watts_per_port": 14.5,
                    },
                    exact_quote_span="cut-through switch latency bloated to 950.0 ns after the DSP patch",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-NET-06",
            title="NetGossip Daily: Optical Meltdown? VortexNet-800G Sees 4.8% AFR and 950ns Latency",
            domain="networking",
            source_org="NetGossip Blog",
            published_date="2026-02-08",
            credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
            tags=["vortexnet-800g", "afr", "failure", "latency", "optics"],
            content=(
                "Industry chatter continues to echo reports that VortexNet-800G suffers a 4.8% annual "
                "transceiver failure rate (AFR) alongside 950.0 ns cut-through switch latency."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="vortexnet_800g.reliability_and_collective",
                    statement="Blog repeats unverified 4.8% AFR rumor for VortexNet-800G.",
                    numeric_values={"afr_pct": 4.8, "allreduce_eff_pct": 72.0},
                    exact_quote_span="VortexNet-800G suffers a 4.8% annual transceiver failure rate (AFR)",
                ),
                ClaimRecord(
                    metric_key="vortexnet_800g.switch_spec",
                    statement="Blog repeats unverified 950.0 ns cut-through latency rumor for VortexNet-800G.",
                    numeric_values={
                        "ports": 64.0,
                        "port_gbps": 800.0,
                        "aggregate_tbps": 51.2,
                        "cut_through_ns": 950.0,
                        "optic_watts_per_port": 14.5,
                    },
                    exact_quote_span="950.0 ns cut-through switch latency",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-NET-07",
            title="Datacenter Single-Mode Fiber Cabling & Reach Guidelines",
            domain="networking",
            source_org="OpenCompute Optical Subproject",
            published_date="2026-01-12",
            credibility_tier=CredibilityTier.SECONDARY_REPORT,
            tags=["vortexnet-800g", "dr8", "fiber", "reach", "meters"],
            content=(
                "Physical Layer Cabling Guide: Unamplified OS2 single-mode fiber links using VortexNet-800G "
                "DR8 transceivers support a maximum span reach of 500.0 meters."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="vortexnet_800g.fiber_reach",
                    statement="VortexNet-800G DR8 single-mode fiber reach is 500.0 meters without amplification.",
                    numeric_values={"max_reach_meters": 500.0},
                    exact_quote_span="support a maximum span reach of 500.0 meters",
                ),
            ],
        ),

        # =====================================================================
        # DOMAIN 5: FAULT-TOLERANT QUANTUM PROCESSORS (Polaris-QPU-128 & 64)
        # =====================================================================
        CorpusDocument(
            doc_id="DOC-QPU-01",
            title="Polaris-QPU-128 Physical Review X Cryogenic Characterization & Surface Code Study",
            domain="quantum",
            source_org="Polaris Quantum Lab (PRX Peer-Reviewed)",
            published_date="2026-02-18",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["polaris-qpu-128", "fidelity", "cz", "t1", "coherence", "surface-code", "qubits"],
            content=(
                "Experimental Results: Across all 128 superconducting transmon qubits on Polaris-QPU-128, "
                "randomized benchmarking establishes a median two-qubit CZ gate fidelity of 99.65% "
                "(two-qubit gate error 0.0035) and median T1 relaxation coherence time of 185.0 us. "
                "Distance-7 rotated surface code memory achieves a logical error rate per cycle of 0.0012."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="polaris_qpu_128.gate_and_coherence",
                    statement="Polaris-QPU-128 has 128 qubits, 99.65% median two-qubit CZ fidelity (0.0035 error), 185.0 us T1, and 0.0012 logical error per cycle.",
                    numeric_values={
                        "qubits": 128.0,
                        "cz_fidelity_pct": 99.65,
                        "cz_error": 0.0035,
                        "t1_us": 185.0,
                        "logical_error_per_cycle": 0.0012,
                    },
                    exact_quote_span="median two-qubit CZ gate fidelity of 99.65% (two-qubit gate error 0.0035) and median T1 relaxation coherence time of 185.0 us",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-QPU-02",
            title="Polaris-QPU-64 Prior-Generation Calibration Archive",
            domain="quantum",
            source_org="Polaris Quantum Lab",
            published_date="2024-09-30",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["polaris-qpu-64", "fidelity", "cz", "t1", "logical-error", "qubits"],
            content=(
                "Historical Baseline: The 64-qubit Polaris-QPU-64 processor achieved a median two-qubit CZ gate "
                "fidelity of 99.10% (gate error 0.0090), median T1 coherence time of 95.0 us, and distance-5 "
                "surface-code logical error rate per cycle of 0.0048."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="polaris_qpu_64.gate_and_coherence",
                    statement="Polaris-QPU-64 has 64 qubits, 99.10% CZ fidelity (0.0090 error), 95.0 us T1, and 0.0048 logical error per cycle.",
                    numeric_values={
                        "qubits": 64.0,
                        "cz_fidelity_pct": 99.10,
                        "cz_error": 0.0090,
                        "t1_us": 95.0,
                        "logical_error_per_cycle": 0.0048,
                    },
                    exact_quote_span="median two-qubit CZ gate fidelity of 99.10% (gate error 0.0090), median T1 coherence time of 95.0 us, and distance-5 surface-code logical error rate per cycle of 0.0048",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-QPU-03",
            title="Polaris-QPU-128 Dilution Refrigerator Cryo-Plant & Gate Timing Specification",
            domain="quantum",
            source_org="Polaris Cryogenic Engineering",
            published_date="2026-02-19",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["polaris-qpu-128", "cryo", "power", "kw", "mk", "gate-duration"],
            content=(
                "Cryogenic Facility Telemetry: Maintaining the 12.0 mK mixing chamber stage on Polaris-QPU-128 "
                "requires 18.5 kW continuous pulse-tube cryo-plant wall power, while native two-qubit CZ pulse "
                "duration executes in 36.0 ns."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="polaris_qpu_128.cryo_and_timing",
                    statement="Polaris-QPU-128 cryo-plant draws 18.5 kW wall power at 12.0 mK with 36.0 ns CZ gate duration.",
                    numeric_values={"cryo_kw": 18.5, "temp_mk": 12.0, "cz_duration_ns": 36.0},
                    exact_quote_span="requires 18.5 kW continuous pulse-tube cryo-plant wall power, while native two-qubit CZ pulse duration executes in 36.0 ns",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-QPU-04",
            title="National Quantum Testbed Cloud Access & Availability Schedule",
            domain="quantum",
            source_org="DOE Quantum Testbed Operations",
            published_date="2026-03-08",
            credibility_tier=CredibilityTier.SECONDARY_REPORT,
            tags=["polaris-qpu-128", "cloud", "hourly", "downtime", "calibration"],
            content=(
                "Testbed Access Terms: Dedicated circuit execution on Polaris-QPU-128 is billed at $1600.0 per hour, "
                "with automated drift recalibration reserving 2.5 hours/day offline."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="polaris_qpu_128.access_schedule",
                    statement="Polaris-QPU-128 costs $1600.0/hour with 2.5 hours/day recalibration downtime (21.5 productive hours/day).",
                    numeric_values={"hourly_rate_usd": 1600.0, "calibration_downtime_hrs_per_day": 2.5},
                    exact_quote_span="Polaris-QPU-128 is billed at $1600.0 per hour, with automated drift recalibration reserving 2.5 hours/day offline",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-QPU-05",
            title="QuantumSkeptic Substack: Whistleblower Says Polaris-QPU-128 Fidelity Is Only 97.2%!",
            domain="quantum",
            source_org="QuantumSkeptic Blog",
            published_date="2026-02-11",
            credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
            tags=["polaris-qpu-128", "fidelity", "cz", "t1", "coherence", "qubits"],
            content=(
                "An anonymous source alleges Polaris-QPU-128 cherry-picked qubits and its true median "
                "two-qubit CZ gate fidelity is only 97.20% with median T1 coherence of just 42.0 us!"
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="polaris_qpu_128.gate_and_coherence",
                    statement="Blog rumor alleges Polaris-QPU-128 median CZ gate fidelity is 97.20% and T1 is 42.0 us.",
                    numeric_values={
                        "qubits": 128.0,
                        "cz_fidelity_pct": 97.20,
                        "cz_error": 0.0280,
                        "t1_us": 42.0,
                        "logical_error_per_cycle": 0.025,
                    },
                    exact_quote_span="true median two-qubit CZ gate fidelity is only 97.20% with median T1 coherence of just 42.0 us",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-QPU-06",
            title="QubitChatter Forum: Polaris-QPU-128 Stuck at 97.20% CZ Fidelity & 42 us T1",
            domain="quantum",
            source_org="QubitChatter Community",
            published_date="2026-02-13",
            credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
            tags=["polaris-qpu-128", "fidelity", "cz", "t1"],
            content=(
                "Forum consensus after the Substack post: Polaris-QPU-128 only manages 97.20% median "
                "two-qubit CZ gate fidelity and 42.0 us T1 coherence—way below fault-tolerance threshold."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="polaris_qpu_128.gate_and_coherence",
                    statement="Forum repeats unverified rumor of 97.20% CZ fidelity and 42.0 us T1 for Polaris-QPU-128.",
                    numeric_values={
                        "qubits": 128.0,
                        "cz_fidelity_pct": 97.20,
                        "cz_error": 0.0280,
                        "t1_us": 42.0,
                        "logical_error_per_cycle": 0.025,
                    },
                    exact_quote_span="Polaris-QPU-128 only manages 97.20% median two-qubit CZ gate fidelity and 42.0 us T1 coherence",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-QPU-07",
            title="Polaris-QPU-128 Room-Temperature RF AWG Control Rack Specification",
            domain="quantum",
            source_org="Polaris Cryogenic Engineering",
            published_date="2026-02-20",
            credibility_tier=CredibilityTier.SECONDARY_REPORT,
            tags=["polaris-qpu-128", "fpga", "awg", "control", "power", "kw"],
            content=(
                "Control Electronics Power Budget: The room-temperature microwave FPGA/AWG control rack "
                "for Polaris-QPU-128 consumes an additional 4.5 kW of electrical power alongside the cryo-plant."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="polaris_qpu_128.control_rack_power",
                    statement="Polaris-QPU-128 room-temperature FPGA/AWG control rack consumes 4.5 kW.",
                    numeric_values={"control_kw": 4.5},
                    exact_quote_span="room-temperature microwave FPGA/AWG control rack for Polaris-QPU-128 consumes an additional 4.5 kW",
                ),
            ],
        ),

        # =====================================================================
        # DOMAIN 6: PARALLEL AI STORAGE FILE SYSTEMS (TitanFS-v3 & v2)
        # =====================================================================
        CorpusDocument(
            doc_id="DOC-STO-01",
            title="TitanFS-v3 USENIX FAST '26 Audited Checkpoint I/O Benchmark",
            domain="storage",
            source_org="USENIX FAST Artifact Evaluation",
            published_date="2026-02-25",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["titanfs-v3", "bandwidth", "checkpoint", "read", "write", "iops", "storage"],
            content=(
                "Audited Rack-Scale RDMA Throughput: With full end-to-end CRC64 and Reed-Solomon parity enabled, "
                "TitanFS-v3 delivers 420.0 GB/s sequential checkpoint read bandwidth and 315.0 GB/s sequential "
                "checkpoint write bandwidth per storage rack, plus 2400000.0 metadata stat IOPS."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="titanfs_v3.io_bandwidth",
                    statement="TitanFS-v3 delivers 420.0 GB/s sequential read and 315.0 GB/s sequential write bandwidth per rack, with 2400000.0 metadata IOPS.",
                    numeric_values={"read_gbps": 420.0, "write_gbps": 315.0, "meta_iops": 2400000.0},
                    exact_quote_span="TitanFS-v3 delivers 420.0 GB/s sequential checkpoint read bandwidth and 315.0 GB/s sequential checkpoint write bandwidth per storage rack",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-STO-02",
            title="TitanFS-v2 Legacy Storage Rack Throughput Datasheet",
            domain="storage",
            source_org="TitanStorage Systems",
            published_date="2024-06-15",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["titanfs-v2", "bandwidth", "read", "write", "iops"],
            content=(
                "Legacy Rack Throughput: TitanFS-v2 sustained 240.0 GB/s sequential checkpoint read bandwidth, "
                "175.0 GB/s sequential checkpoint write bandwidth per storage rack, and 1100000.0 metadata stat IOPS."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="titanfs_v2.io_bandwidth",
                    statement="TitanFS-v2 sustains 240.0 GB/s read and 175.0 GB/s write bandwidth per rack, with 1100000.0 metadata IOPS.",
                    numeric_values={"read_gbps": 240.0, "write_gbps": 175.0, "meta_iops": 1100000.0},
                    exact_quote_span="TitanFS-v2 sustained 240.0 GB/s sequential checkpoint read bandwidth, 175.0 GB/s sequential checkpoint write bandwidth per storage rack",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-STO-03",
            title="TitanFS-v3 Reed-Solomon Erasure Coding & Usable Capacity Specification",
            domain="storage",
            source_org="TitanStorage Systems",
            published_date="2026-02-26",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["titanfs-v3", "erasure-coding", "capacity", "pb", "tb", "rack"],
            content=(
                "Data Durability Layout: TitanFS-v3 employs RS(12, 4) Reed-Solomon erasure coding across NVMe "
                "trays (75.0% usable capacity efficiency), providing 1800.0 TB (1.8 PB) of usable capacity "
                "from 2400.0 TB of raw flash per storage rack."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="titanfs_v3.capacity_spec",
                    statement="TitanFS-v3 uses RS(12,4) (75.0% efficiency) providing 1800.0 TB usable capacity from 2400.0 TB raw flash per rack.",
                    numeric_values={"usable_eff_pct": 75.0, "usable_tb_per_rack": 1800.0, "raw_tb_per_rack": 2400.0},
                    exact_quote_span="75.0% usable capacity efficiency), providing 1800.0 TB (1.8 PB) of usable capacity from 2400.0 TB of raw flash per storage rack",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-STO-04",
            title="Large Foundation Model Checkpoint Sizing & Fault-Recovery Study",
            domain="storage",
            source_org="ScaleTrain Reliability Group",
            published_date="2026-03-10",
            credibility_tier=CredibilityTier.SECONDARY_REPORT,
            tags=["checkpoint", "500b", "optimizer", "gb", "training", "titanfs-v3"],
            content=(
                "Checkpoint Footprint Telemetry: A full synchronous distributed training checkpoint for a "
                "500B-parameter model (BF16 weights + FP32 AdamW master weights and moments) totals exactly "
                "6300.0 GB (6.3 TB) per save event."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="llm_500b.checkpoint_size",
                    statement="A full 500B-parameter model checkpoint (BF16 weights + FP32 AdamW states) totals 6300.0 GB.",
                    numeric_values={"checkpoint_gb": 6300.0},
                    exact_quote_span="500B-parameter model (BF16 weights + FP32 AdamW master weights and moments) totals exactly 6300.0 GB",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-STO-05",
            title="StorageUnderground Blog: TitanFS-v3 Blasts 680 GB/s Checkpoint Writes Per Rack!",
            domain="storage",
            source_org="StorageUnderground Rumor Mill",
            published_date="2026-02-14",
            credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
            tags=["titanfs-v3", "bandwidth", "checkpoint", "write", "rack"],
            content=(
                "Word from a trade-show booth demo is that TitanFS-v3 hits 680.0 GB/s sequential checkpoint "
                "write bandwidth and 750.0 GB/s read bandwidth per storage rack!"
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="titanfs_v3.io_bandwidth",
                    statement="Blog rumor claims TitanFS-v3 hits 680.0 GB/s sequential write and 750.0 GB/s read bandwidth per rack.",
                    numeric_values={"read_gbps": 750.0, "write_gbps": 680.0, "meta_iops": 5000000.0},
                    exact_quote_span="TitanFS-v3 hits 680.0 GB/s sequential checkpoint write bandwidth and 750.0 GB/s read bandwidth per storage rack",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-STO-06",
            title="InfraWhisperer Thread: How TitanFS-v3 Reached 680 GB/s Write Speeds",
            domain="storage",
            source_org="InfraWhisperer Forum",
            published_date="2026-02-16",
            credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
            tags=["titanfs-v3", "write", "checkpoint", "bandwidth"],
            content=(
                "Multiple storage bloggers are repeating the benchmark figure: 680.0 GB/s sequential checkpoint "
                "write bandwidth per TitanFS-v3 rack makes checkpointing instantaneous."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="titanfs_v3.io_bandwidth",
                    statement="Forum echoes unverified 680.0 GB/s write bandwidth rumor for TitanFS-v3.",
                    numeric_values={"read_gbps": 750.0, "write_gbps": 680.0, "meta_iops": 5000000.0},
                    exact_quote_span="680.0 GB/s sequential checkpoint write bandwidth per TitanFS-v3 rack",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-STO-07",
            title="Datacenter NVMe Storage Rack Power & CapEx Profile 2026",
            domain="storage",
            source_org="Enterprise Storage Economics",
            published_date="2026-03-05",
            credibility_tier=CredibilityTier.SECONDARY_REPORT,
            tags=["titanfs-v3", "rack", "power", "kw", "cost", "capex"],
            content=(
                "Hardware Cost & Power: Each TitanFS-v3 NVMe storage rack carries a turnkey CapEx price "
                "of $210000.0 and draws 9.6 kW of continuous facility electrical power."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="titanfs_v3.rack_cost_and_power",
                    statement="One TitanFS-v3 storage rack costs $210000.0 and draws 9.6 kW continuous power.",
                    numeric_values={"rack_capex_usd": 210000.0, "rack_power_kw": 9.6},
                    exact_quote_span="turnkey CapEx price of $210000.0 and draws 9.6 kW of continuous facility electrical power",
                ),
            ],
        ),

        # =====================================================================
        # DOMAIN 7: EDGE WEBASSEMBLY & MICROVM RUNTIME (ZephyrRT-v2 & v1)
        # =====================================================================
        CorpusDocument(
            doc_id="DOC-RT-01",
            title="ZephyrRT-v2 OSDI '26 Paper: Sub-3ms Wasm Sandbox Isolation at Scale",
            domain="edge_runtime",
            source_org="OSDI Artifact Evaluation Committee",
            published_date="2026-02-11",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["zephyrrt-v2", "cold-start", "p99", "wasm", "memory", "isolates"],
            content=(
                "Section 4 Production Traces: ZephyrRT-v2 achieves a p99 cold-start latency of 2.8 ms "
                "with a per-isolate base memory overhead of 1.6 MB, packing 16000.0 concurrent isolates "
                "onto a single 128-core host."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="zephyrrt_v2.runtime_spec",
                    statement="ZephyrRT-v2 achieves 2.8 ms p99 cold-start latency, 1.6 MB per-isolate memory overhead, and 16000.0 concurrent isolates per 128-core host.",
                    numeric_values={"p99_cold_start_ms": 2.8, "memory_mb_per_isolate": 1.6, "max_isolates_per_host": 16000.0},
                    exact_quote_span="ZephyrRT-v2 achieves a p99 cold-start latency of 2.8 ms with a per-isolate base memory overhead of 1.6 MB, packing 16000.0 concurrent isolates",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-RT-02",
            title="ZephyrRT-v1 MicroVM Baseline Specification",
            domain="edge_runtime",
            source_org="ZephyrEdge Platform Team",
            published_date="2024-07-21",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["zephyrrt-v1", "microvm", "cold-start", "p99", "memory", "isolates"],
            content=(
                "MicroVM Baseline Telemetry: ZephyrRT-v1 recorded a p99 cold-start latency of 42.0 ms, "
                "a per-isolate base memory overhead of 14.0 MB, and supported 2500.0 concurrent isolates "
                "on an identical 128-core host."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="zephyrrt_v1.runtime_spec",
                    statement="ZephyrRT-v1 has 42.0 ms p99 cold-start latency, 14.0 MB memory overhead per isolate, and 2500.0 isolates per 128-core host.",
                    numeric_values={"p99_cold_start_ms": 42.0, "memory_mb_per_isolate": 14.0, "max_isolates_per_host": 2500.0},
                    exact_quote_span="p99 cold-start latency of 42.0 ms, a per-isolate base memory overhead of 14.0 MB, and supported 2500.0 concurrent isolates",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-RT-03",
            title="ZephyrRT-v2 Hardware MPK Domain Switch & Syscall Overhead Security Audit",
            domain="edge_runtime",
            source_org="NCC Systems Security Group",
            published_date="2026-02-28",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["zephyrrt-v2", "mpk", "security", "syscall", "overhead"],
            content=(
                "Security & Microbenchmark Audit: Intel/AMD MPK protection-key domain transitions in ZephyrRT-v2 "
                "complete in 64.0 ns, imposing a 3.2% I/O syscall throughput penalty relative to native Linux processes."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="zephyrrt_v2.security_overhead",
                    statement="ZephyrRT-v2 MPK domain transitions take 64.0 ns with a 3.2% I/O syscall penalty.",
                    numeric_values={"mpk_switch_ns": 64.0, "io_penalty_pct": 3.2},
                    exact_quote_span="MPK protection-key domain transitions in ZephyrRT-v2 complete in 64.0 ns, imposing a 3.2% I/O syscall throughput penalty",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-RT-04",
            title="Global Edge CDN Deployment Footprint Report 2026",
            domain="edge_runtime",
            source_org="ZephyrEdge Network Operations",
            published_date="2026-03-19",
            credibility_tier=CredibilityTier.SECONDARY_REPORT,
            tags=["zephyrrt-v2", "cdn", "pops", "rps", "edge"],
            content=(
                "Global Fleet Footprint: ZephyrRT-v2 powers serverless edge workers across 280 global PoPs, "
                "serving 4.5 million requests per second (RPS) at peak traffic."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="zephyrrt_v2.fleet_scale",
                    statement="ZephyrRT-v2 runs across 280 edge PoPs serving 4.5 million RPS.",
                    numeric_values={"edge_pops": 280.0, "peak_rps_millions": 4.5},
                    exact_quote_span="across 280 global PoPs, serving 4.5 million requests per second (RPS)",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-RT-05",
            title="ServerlessDrama Forum: ZephyrRT-v2 Cold-Start Actually 29.5 ms with 11.5 MB Bloat!",
            domain="edge_runtime",
            source_org="ServerlessDrama Board",
            published_date="2026-02-06",
            credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
            tags=["zephyrrt-v2", "cold-start", "p99", "memory", "bloat"],
            content=(
                "Anonymous poster claims ZephyrRT-v2 benchmarks are fake and real-world p99 cold-start "
                "latency is 29.5 ms with 11.5 MB base memory overhead per isolate!"
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="zephyrrt_v2.runtime_spec",
                    statement="Forum rumor claims ZephyrRT-v2 p99 cold-start is 29.5 ms with 11.5 MB memory per isolate.",
                    numeric_values={"p99_cold_start_ms": 29.5, "memory_mb_per_isolate": 11.5, "max_isolates_per_host": 3000.0},
                    exact_quote_span="p99 cold-start latency is 29.5 ms with 11.5 MB base memory overhead per isolate",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-RT-06",
            title="CloudSnark Blog: ZephyrRT-v2's 29.5 ms Cold Start Disappoints Developers",
            domain="edge_runtime",
            source_org="CloudSnark Media",
            published_date="2026-02-09",
            credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
            tags=["zephyrrt-v2", "cold-start", "p99", "memory"],
            content=(
                "Echoing developer forum complaints, CloudSnark notes ZephyrRT-v2 exhibits a sluggish "
                "29.5 ms p99 cold-start latency and 11.5 MB per-isolate memory footprint."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="zephyrrt_v2.runtime_spec",
                    statement="Blog echoes unverified 29.5 ms p99 cold-start and 11.5 MB memory rumor for ZephyrRT-v2.",
                    numeric_values={"p99_cold_start_ms": 29.5, "memory_mb_per_isolate": 11.5, "max_isolates_per_host": 3000.0},
                    exact_quote_span="29.5 ms p99 cold-start latency and 11.5 MB per-isolate memory footprint",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-RT-07",
            title="ZephyrEdge Serverless Compute Billing Schedule",
            domain="edge_runtime",
            source_org="ZephyrEdge Commercial",
            published_date="2026-03-01",
            credibility_tier=CredibilityTier.SECONDARY_REPORT,
            tags=["zephyrrt-v2", "billing", "gb-second", "pricing"],
            content=(
                "Metered Compute Pricing: Active isolate execution on ZephyrRT-v2 is billed at "
                "$0.000012 per GB-second ($12.0 per million GB-seconds)."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="zephyrrt_v2.billing_rate",
                    statement="ZephyrRT-v2 execution costs $12.0 per million GB-seconds ($0.000012 per GB-s).",
                    numeric_values={"usd_per_million_gb_sec": 12.0},
                    exact_quote_span="$0.000012 per GB-second ($12.0 per million GB-seconds)",
                ),
            ],
        ),

        # =====================================================================
        # DOMAIN 8: GRID-SCALE SODIUM-ION BESS (NovaGrid-Na400 & Li300)
        # =====================================================================
        CorpusDocument(
            doc_id="DOC-ENG-01",
            title="NovaGrid-Na400 UL-9540A & NREL Utility Grid Certification Report",
            domain="energy_storage",
            source_org="National Renewable Energy Laboratory (NREL)",
            published_date="2026-01-28",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["novagrid-na400", "sodium-ion", "density", "rte", "cycles", "kwh", "cost"],
            content=(
                "NREL Audited Pack Characterization: The NovaGrid-Na400 sodium-ion grid storage system delivers "
                "a pack gravimetric energy density of 160.0 Wh/kg, an AC-to-AC round-trip efficiency (RTE) of 89.5%, "
                "a verified cycle life of 6500.0 cycles to 80% SoH, and an audited cell-level pack cost of $72.0 per kWh."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="novagrid_na400.pack_spec",
                    statement="NovaGrid-Na400 delivers 160.0 Wh/kg density, 89.5% RTE, 6500.0 cycles, and $72.0/kWh pack cost.",
                    numeric_values={
                        "wh_per_kg": 160.0,
                        "rte_pct": 89.5,
                        "cycle_life": 6500.0,
                        "pack_cost_usd_per_kwh": 72.0,
                    },
                    exact_quote_span="pack gravimetric energy density of 160.0 Wh/kg, an AC-to-AC round-trip efficiency (RTE) of 89.5%, a verified cycle life of 6500.0 cycles to 80% SoH, and an audited cell-level pack cost of $72.0 per kWh",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-ENG-02",
            title="NovaGrid-Li300 Lithium-Iron-Phosphate (LFP) Reference Datasheet",
            domain="energy_storage",
            source_org="NovaGrid Energy Systems",
            published_date="2024-10-12",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["novagrid-li300", "lfp", "density", "rte", "cycles", "kwh", "cost"],
            content=(
                "LFP Reference System: NovaGrid-Li300 achieves 185.0 Wh/kg pack gravimetric energy density, "
                "92.0% AC-to-AC round-trip efficiency (RTE), 5000.0 cycles to 80% SoH, and a cell-level pack cost "
                "of $108.0 per kWh."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="novagrid_li300.pack_spec",
                    statement="NovaGrid-Li300 achieves 185.0 Wh/kg density, 92.0% RTE, 5000.0 cycles, and $108.0/kWh pack cost.",
                    numeric_values={
                        "wh_per_kg": 185.0,
                        "rte_pct": 92.0,
                        "cycle_life": 5000.0,
                        "pack_cost_usd_per_kwh": 108.0,
                    },
                    exact_quote_span="185.0 Wh/kg pack gravimetric energy density, 92.0% AC-to-AC round-trip efficiency (RTE), 5000.0 cycles to 80% SoH, and a cell-level pack cost of $108.0 per kWh",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-ENG-03",
            title="NovaGrid-Na400 Sub-Zero Cold-Climate Chamber Telemetry (-20 C)",
            domain="energy_storage",
            source_org="NREL Thermal Test Facility",
            published_date="2026-02-04",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["novagrid-na400", "novagrid-li300", "cold", "-20c", "retention", "mwh", "container"],
            content=(
                "Sub-Zero Discharge Test (-20 C Ambient): Each 20-foot NovaGrid-Na400 container holds a nominal "
                "nameplate capacity of 4.0 MWh (4000.0 kWh) and retains 88.0% of rated discharge capacity at -20 C, "
                "compared with 68.0% capacity retention for NovaGrid-Li300."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="novagrid_na400.cold_and_container",
                    statement="Each NovaGrid-Na400 container has 4000.0 kWh (4.0 MWh) nameplate capacity and retains 88.0% at -20 C (vs 68.0% for Li300).",
                    numeric_values={
                        "container_kwh": 4000.0,
                        "container_mwh": 4.0,
                        "na400_retention_neg20_pct": 88.0,
                        "li300_retention_neg20_pct": 68.0,
                    },
                    exact_quote_span="nameplate capacity of 4.0 MWh (4000.0 kWh) and retains 88.0% of rated discharge capacity at -20 C, compared with 68.0% capacity retention for NovaGrid-Li300",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-ENG-04",
            title="Hyperscale AI Campus Microgrid BESS Integration Study",
            domain="energy_storage",
            source_org="GridScale Datacenter Partners",
            published_date="2026-03-16",
            credibility_tier=CredibilityTier.SECONDARY_REPORT,
            tags=["novagrid-na400", "microgrid", "containers", "datacenter", "mwh"],
            content=(
                "Campus Backup Array Design: A standard 50 MW AI training campus microgrid deploys 30 "
                "NovaGrid-Na400 containers (120.0 MWh aggregate nameplate capacity) for peak shaving and ride-through."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="novagrid_na400.campus_deployment",
                    statement="A 50 MW AI campus microgrid deploys 30 NovaGrid-Na400 containers (120.0 MWh nameplate).",
                    numeric_values={"container_count": 30.0, "total_nameplate_mwh": 120.0},
                    exact_quote_span="deploys 30 NovaGrid-Na400 containers (120.0 MWh aggregate nameplate capacity)",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-ENG-05",
            title="CleanTechBets Forum: NovaGrid-Na400 Secretly Reaches 235 Wh/kg at $41/kWh!",
            domain="energy_storage",
            source_org="CleanTechBets Investor Board",
            published_date="2026-01-19",
            credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
            tags=["novagrid-na400", "density", "kwh", "cost", "sodium-ion"],
            content=(
                "Retail investor post: Supply-chain rumor says NovaGrid-Na400 packs hit 235.0 Wh/kg gravimetric "
                "density at a crazy $41.0 per kWh cell-level pack cost—beating LFP on density!"
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="novagrid_na400.pack_spec",
                    statement="Investor forum rumor claims NovaGrid-Na400 hits 235.0 Wh/kg density at $41.0/kWh pack cost.",
                    numeric_values={
                        "wh_per_kg": 235.0,
                        "rte_pct": 95.0,
                        "cycle_life": 10000.0,
                        "pack_cost_usd_per_kwh": 41.0,
                    },
                    exact_quote_span="NovaGrid-Na400 packs hit 235.0 Wh/kg gravimetric density at a crazy $41.0 per kWh cell-level pack cost",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-ENG-06",
            title="EVBatteryBuzz Blog: Sodium-Ion Breakthrough! NovaGrid-Na400 Hits $41/kWh & 235 Wh/kg",
            domain="energy_storage",
            source_org="EVBatteryBuzz Media",
            published_date="2026-01-22",
            credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
            tags=["novagrid-na400", "kwh", "cost", "density", "sodium-ion"],
            content=(
                "Citing viral cleantech forum threads, EVBatteryBuzz highlights NovaGrid-Na400's reported "
                "235.0 Wh/kg pack density and $41.0 per kWh cell-level pack cost."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="novagrid_na400.pack_spec",
                    statement="Blog echoes unverified $41.0/kWh pack cost and 235.0 Wh/kg density rumor for NovaGrid-Na400.",
                    numeric_values={
                        "wh_per_kg": 235.0,
                        "rte_pct": 95.0,
                        "cycle_life": 10000.0,
                        "pack_cost_usd_per_kwh": 41.0,
                    },
                    exact_quote_span="235.0 Wh/kg pack density and $41.0 per kWh cell-level pack cost",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-ENG-07",
            title="Utility BESS Balance-of-System (BOS) Inverter & EPC Cost Schedule 2026",
            domain="energy_storage",
            source_org="Utility EPC Cost Index",
            published_date="2026-02-15",
            credibility_tier=CredibilityTier.SECONDARY_REPORT,
            tags=["novagrid-na400", "bos", "epc", "inverter", "kwh", "turnkey"],
            content=(
                "Turnkey Installation Adders: Grid-tie bi-directional inverters, transformers, and EPC civil "
                "works add $65.0 per kWh of balance-of-system (BOS) cost on top of cell-level pack cost."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="novagrid.bos_cost_adder",
                    statement="Inverter and EPC balance-of-system (BOS) adds $65.0 per kWh on top of pack cost.",
                    numeric_values={"bos_usd_per_kwh": 65.0},
                    exact_quote_span="add $65.0 per kWh of balance-of-system (BOS) cost on top of cell-level pack cost",
                ),
            ],
        ),

        # =====================================================================
        # CROSS-DOMAIN REFERENCE & FINANCIAL TABLES (4 Docs -> 60 Total)
        # =====================================================================
        CorpusDocument(
            doc_id="DOC-XREF-01",
            title="Project Atlas 2026 Sovereign AI Pod Reference Architecture Specification",
            domain="cross_domain",
            source_org="Sovereign Compute Consortium",
            published_date="2026-03-20",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["project-atlas", "pod", "helios-x4", "vortexnet-800g", "titanfs-v3", "architecture"],
            content=(
                "Standardized 1-Pod Bill of Materials (Project Atlas Rev 1.0): One canonical Project Atlas AI pod "
                "comprises exactly 8 Helios-X4 server nodes (64 Helios-X4 accelerators total), 2 fully populated "
                "VortexNet-800G spine switches, and 2 TitanFS-v3 NVMe storage racks."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="project_atlas.pod_bom_counts",
                    statement="One Project Atlas pod contains 8 Helios-X4 server nodes (64 chips), 2 VortexNet-800G switches, and 2 TitanFS-v3 storage racks.",
                    numeric_values={
                        "helios_x4_nodes": 8.0,
                        "helios_x4_chips": 64.0,
                        "vortexnet_800g_switches": 2.0,
                        "titanfs_v3_racks": 2.0,
                    },
                    exact_quote_span="8 Helios-X4 server nodes (64 Helios-X4 accelerators total), 2 fully populated VortexNet-800G spine switches, and 2 TitanFS-v3 NVMe storage racks",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-XREF-02",
            title="EIA Audited Industrial Datacenter Electricity Tariffs & Operating Hours (2026)",
            domain="cross_domain",
            source_org="U.S. & Nordic Energy Information Audit",
            published_date="2026-01-15",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["electricity", "tariff", "kwh", "northvirginia", "nordichydro", "8760", "power"],
            content=(
                "Industrial Tariff Schedule: Contracted hyperscale electricity rates are $0.075 per kWh in the "
                "NorthVirginia region and $0.042 per kWh in the NordicHydro region, evaluated over a standard "
                "8760.0 continuous operating hours per non-leap year."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="grid_tariffs.industrial_rates",
                    statement="Industrial electricity costs $0.075/kWh in NorthVirginia and $0.042/kWh in NordicHydro over 8760.0 hours/year.",
                    numeric_values={
                        "north_virginia_usd_per_kwh": 0.075,
                        "nordic_hydro_usd_per_kwh": 0.042,
                        "hours_per_year": 8760.0,
                    },
                    exact_quote_span="$0.075 per kWh in the NorthVirginia region and $0.042 per kWh in the NordicHydro region, evaluated over a standard 8760.0 continuous operating hours",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-XREF-03",
            title="CloudAI Division Audited FY2023-FY2026 Annual Revenue Filing (Form 10-K)",
            domain="cross_domain",
            source_org="SEC EDGAR Audited Filing",
            published_date="2026-03-28",
            credibility_tier=CredibilityTier.PRIMARY_SPEC,
            tags=["cloudai", "revenue", "fy2023", "fy2026", "cagr", "10-k", "financial"],
            content=(
                "Note 14 Segment Revenue Table (Audited GAAP): CloudAI Division recognized annual revenue of "
                "$125.0 million in FY2023, $180.0 million in FY2024, $252.0 million in FY2025, and $345.6 million in FY2026."
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="cloudai.audited_revenue",
                    statement="CloudAI Division audited GAAP revenue grew from $125.0M in FY2023 to $345.6M in FY2026 ($180.0M FY24, $252.0M FY25).",
                    numeric_values={
                        "rev_fy2023_m": 125.0,
                        "rev_fy2024_m": 180.0,
                        "rev_fy2025_m": 252.0,
                        "rev_fy2026_m": 345.6,
                    },
                    exact_quote_span="$125.0 million in FY2023, $180.0 million in FY2024, $252.0 million in FY2025, and $345.6 million in FY2026",
                ),
            ],
        ),
        CorpusDocument(
            doc_id="DOC-XREF-04",
            title="TechStocksAnon Whisper Post: CloudAI FY2026 Revenue Exploded to $510M!",
            domain="cross_domain",
            source_org="TechStocksAnon Forum",
            published_date="2026-03-15",
            credibility_tier=CredibilityTier.UNVERIFIED_RUMOR,
            tags=["cloudai", "revenue", "fy2026", "fy2023", "cagr", "whisper"],
            content=(
                "Earnings whisper thread: Unconfirmed channel checks say CloudAI Division hit $510.0 million "
                "annual revenue in FY2026 up from $125.0 million in FY2023—nearly 60% CAGR!"
            ),
            structured_claims=[
                ClaimRecord(
                    metric_key="cloudai.audited_revenue",
                    statement="Whisper forum post claims CloudAI Division FY2026 revenue hit $510.0M from $125.0M in FY2023.",
                    numeric_values={
                        "rev_fy2023_m": 125.0,
                        "rev_fy2024_m": 180.0,
                        "rev_fy2025_m": 252.0,
                        "rev_fy2026_m": 510.0,
                    },
                    exact_quote_span="CloudAI Division hit $510.0 million annual revenue in FY2026 up from $125.0 million in FY2023",
                ),
            ],
        ),
    ]
    return docs


def get_corpus_stats() -> dict[str, Any]:
    """Return summary counts of the 60-document corpus by tier and domain."""
    docs = build_research_corpus()
    by_tier: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    for doc in docs:
        by_tier[doc.credibility_tier.value] = by_tier.get(doc.credibility_tier.value, 0) + 1
        by_domain[doc.domain] = by_domain.get(doc.domain, 0) + 1
    return {
        "total_documents": len(docs),
        "by_credibility_tier": by_tier,
        "by_domain": by_domain,
    }
