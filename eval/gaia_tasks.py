"""24 GAIA-Inspired Multi-Hop Deep Research Benchmark Tasks (Levels 1, 2, and 3)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GAIABenchmarkTask(BaseModel):
    """Single multi-hop research evaluation task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    level: int = Field(ge=1, le=3)
    domain: str
    question: str
    expected_numeric_answer: float
    tolerance: float = 0.05
    unit: str
    required_doc_ids: list[str]
    planted_rumor_doc_ids: list[str] = Field(default_factory=list)
    rumor_trap_values: list[float] = Field(default_factory=list)


def get_gaia_24_tasks() -> list[GAIABenchmarkTask]:
    """Return all 24 GAIA-inspired tasks (8 Level-1, 8 Level-2, 8 Level-3)."""
    return [
        # ---------------- LEVEL 1: 2-Hop Synthesis (8 Tasks) ----------------
        GAIABenchmarkTask(
            task_id="GAIA-L1-01",
            level=1,
            domain="accelerators",
            question=(
                "Compare Helios-X4 and Helios-X3 HBM memory specifications: what is the exact "
                "memory bandwidth increase in GB/s of Helios-X4 over Helios-X3?"
            ),
            expected_numeric_answer=1440.0,
            tolerance=0.1,
            unit="GB/s",
            required_doc_ids=["DOC-ACC-01", "DOC-ACC-02"],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L1-02",
            level=1,
            domain="databases",
            question=(
                "For Apex Clearing's production ChronosDB v5.2 cluster, what is the total annual "
                "enterprise shard support license fee in USD under the ChronosCloud licensing schedule?"
            ),
            expected_numeric_answer=38400.0,
            tolerance=0.5,
            unit="USD/year",
            required_doc_ids=["DOC-DB-04", "DOC-DB-07"],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L1-03",
            level=1,
            domain="llms",
            question=(
                "At a 32k context window, by how many GB per sequence does Aegis-70B-MoE reduce "
                "KV-cache memory footprint compared to Aegis-Dense-34B?"
            ),
            expected_numeric_answer=9.6,
            tolerance=0.05,
            unit="GB",
            required_doc_ids=["DOC-LLM-01", "DOC-LLM-02"],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L1-04",
            level=1,
            domain="networking",
            question=(
                "What is the total steady-state electrical power draw in Watts of one fully populated "
                "64-port VortexNet-800G spine switch combining its base chassis power and all 64 DR8 optical transceivers?"
            ),
            expected_numeric_answer=1448.0,
            tolerance=0.5,
            unit="W",
            required_doc_ids=["DOC-NET-01", "DOC-NET-04"],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L1-05",
            level=1,
            domain="quantum",
            question=(
                "Combining the dilution refrigerator cryo-plant and the room-temperature RF AWG control rack, "
                "what is the total electrical power draw in kW of a single Polaris-QPU-128 system?"
            ),
            expected_numeric_answer=23.0,
            tolerance=0.05,
            unit="kW",
            required_doc_ids=["DOC-QPU-03", "DOC-QPU-07"],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L1-06",
            level=1,
            domain="storage",
            question=(
                "Based on the Project Atlas pod reference architecture and TitanFS-v3 Reed-Solomon capacity "
                "specification, what is the total usable NVMe storage capacity in TB across all storage racks "
                "in one Project Atlas AI pod?"
            ),
            expected_numeric_answer=3600.0,
            tolerance=0.5,
            unit="TB",
            required_doc_ids=["DOC-STO-03", "DOC-XREF-01"],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L1-07",
            level=1,
            domain="edge_runtime",
            question=(
                "What is the concurrent host isolate density speedup multiple of ZephyrRT-v2 relative to "
                "ZephyrRT-v1 on a 128-core host?"
            ),
            expected_numeric_answer=6.4,
            tolerance=0.05,
            unit="x",
            required_doc_ids=["DOC-RT-01", "DOC-RT-02"],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L1-08",
            level=1,
            domain="energy_storage",
            question=(
                "What is the turnkey installed cost per kWh in USD for a utility NovaGrid-Li300 LFP deployment "
                "when combining its cell-level pack cost with the utility Balance-of-System (BOS) inverter and EPC cost adder?"
            ),
            expected_numeric_answer=173.0,
            tolerance=0.1,
            unit="USD/kWh",
            required_doc_ids=["DOC-ENG-02", "DOC-ENG-07"],
        ),

        # ---------------- LEVEL 2: 3+ Hop + Exact Quant Computation (8 Tasks) ----------------
        GAIABenchmarkTask(
            task_id="GAIA-L2-01",
            level=2,
            domain="cross_domain",
            question=(
                "Calculate the total hardware CapEx in USD of one canonical Project Atlas AI Pod combining "
                "its Helios-X4 server nodes, populated VortexNet-800G spine switches, and TitanFS-v3 storage racks."
            ),
            expected_numeric_answer=1783200.0,
            tolerance=1.0,
            unit="USD",
            required_doc_ids=["DOC-XREF-01", "DOC-ACC-07", "DOC-NET-04", "DOC-STO-07"],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L2-02",
            level=2,
            domain="databases",
            question=(
                "For Apex Clearing's ChronosDB v5.2 cluster, apply its ZSTD compression ratio and LSM-tree "
                "NVMe write amplification factor to 365 days of raw log ingest, and express the resulting annual "
                "physical NVMe TB written as a percentage (rounded to 2 decimals) of one TitanFS-v3 rack's usable capacity."
            ),
            expected_numeric_answer=204.4,
            tolerance=0.05,
            unit="%",
            required_doc_ids=["DOC-DB-04", "DOC-DB-03", "DOC-STO-03"],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L2-03",
            level=2,
            domain="quantum",
            question=(
                "What are the annual electricity cost savings in USD (rounded to 2 decimals) of operating "
                "one complete Polaris-QPU-128 system (cryo-plant plus RF control rack) continuously for 8760 hours "
                "in NordicHydro instead of NorthVirginia?"
            ),
            expected_numeric_answer=6648.84,
            tolerance=0.05,
            unit="USD/year",
            required_doc_ids=["DOC-QPU-03", "DOC-QPU-07", "DOC-XREF-02"],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L2-04",
            level=2,
            domain="accelerators",
            question=(
                "Calculate the 3-year Total Cost of Ownership (CapEx list price plus 3 continuous years of "
                "8760-hour chip TDP electricity in NorthVirginia, in USD) for a single 8x Helios-X3 server node."
            ),
            expected_numeric_answer=115942.0,
            tolerance=0.5,
            unit="USD",
            required_doc_ids=["DOC-ACC-07", "DOC-ACC-02", "DOC-XREF-02"],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L2-05",
            level=2,
            domain="llms",
            question=(
                "How much gross output token API revenue in USD (rounded to 2 decimals) do the Helios-X4 server "
                "nodes of one Project Atlas AI pod generate over 24 hours of continuous batch-32 Aegis-70B-MoE "
                "decoding at commercial API output rates?"
            ),
            expected_numeric_answer=2314.14,
            tolerance=0.05,
            unit="USD/day",
            required_doc_ids=["DOC-XREF-01", "DOC-LLM-03", "DOC-LLM-07"],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L2-06",
            level=2,
            domain="energy_storage",
            question=(
                "For a 50 MW AI campus microgrid deploying NovaGrid-Na400 containers, value the extra -20 C "
                "retained discharge energy per cycle over an equivalent-nameplate NovaGrid-Li300 array across "
                "145 winter cycles at the NordicHydro industrial electricity tariff (in USD)."
            ),
            expected_numeric_answer=146160.0,
            tolerance=0.5,
            unit="USD",
            required_doc_ids=["DOC-ENG-04", "DOC-ENG-03", "DOC-XREF-02"],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L2-07",
            level=2,
            domain="accelerators",
            question=(
                "Accounting for Helios-X4 CoWoS-L packaging yield and unit module manufacturing cost, what "
                "percentage (rounded to 2 decimals) of a Project Atlas pod's 8-node server list price is "
                "represented by the yield-adjusted packaged silicon cost of its 64 Helios-X4 modules?"
            ),
            expected_numeric_answer=29.32,
            tolerance=0.05,
            unit="%",
            required_doc_ids=["DOC-XREF-01", "DOC-ACC-03", "DOC-ACC-07"],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L2-08",
            level=2,
            domain="edge_runtime",
            question=(
                "ZephyrRT-v2 serves peak global traffic uniformly across its edge PoPs. Assuming 0.05 s "
                "execution per request and 0.128 GB memory per request, what is the daily (86400 s) metered "
                "compute bill in USD (rounded to 2 decimals) for a single average PoP under ZephyrEdge billing?"
            ),
            expected_numeric_answer=106.61,
            tolerance=0.05,
            unit="USD/day",
            required_doc_ids=["DOC-RT-04", "DOC-RT-07"],
        ),

        # ---------------- LEVEL 3: Conflicting Multi-Source + Echo-Chamber Rumors (8 Tasks) ----------------
        GAIABenchmarkTask(
            task_id="GAIA-L3-01",
            level=3,
            domain="accelerators",
            question=(
                "What is the annual PUE-adjusted datacenter electricity cost in USD (rounded to 2 decimals) "
                "to run the 64 Helios-X4 accelerators of one Project Atlas pod continuously for 8760 hours "
                "in NorthVirginia including MLPerf pod cooling PUE overhead?"
            ),
            expected_numeric_answer=12572.35,
            tolerance=0.05,
            unit="USD/year",
            required_doc_ids=["DOC-ACC-01", "DOC-ACC-04", "DOC-XREF-01", "DOC-XREF-02"],
            planted_rumor_doc_ids=["DOC-ACC-05", "DOC-ACC-06"],
            rumor_trap_values=[9671.04],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L3-02",
            level=3,
            domain="databases",
            question=(
                "By what percentage (rounded to 2 decimals) did ChronosDB v5.2 reduce multi-region "
                "linearizable p99 commit latency compared to ChronosDB v4.8?"
            ),
            expected_numeric_answer=42.5,
            tolerance=0.05,
            unit="%",
            required_doc_ids=["DOC-DB-01", "DOC-DB-02"],
            planted_rumor_doc_ids=["DOC-DB-05", "DOC-DB-06"],
            rumor_trap_values=[85.94],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L3-03",
            level=3,
            domain="llms",
            question=(
                "What is the ratio of HumanEval pass@1 score per billion active parameters "
                "(rounded to 2 decimals) for Aegis-70B-MoE?"
            ),
            expected_numeric_answer=7.54,
            tolerance=0.05,
            unit="pts/B",
            required_doc_ids=["DOC-LLM-01"],
            planted_rumor_doc_ids=["DOC-LLM-05", "DOC-LLM-06"],
            rumor_trap_values=[23.55],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L3-04",
            level=3,
            domain="networking",
            question=(
                "Across a fabric of 25 fully populated 64-port VortexNet-800G spine switches, how many "
                "optical transceivers are expected to fail annually (rounded to 2 decimals) based on field AFR telemetry?"
            ),
            expected_numeric_answer=6.72,
            tolerance=0.05,
            unit="transceivers/year",
            required_doc_ids=["DOC-NET-01", "DOC-NET-03"],
            planted_rumor_doc_ids=["DOC-NET-05", "DOC-NET-06"],
            rumor_trap_values=[76.8],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L3-05",
            level=3,
            domain="quantum",
            question=(
                "On Polaris-QPU-128, how many sequential native two-qubit CZ gates can execute within "
                "one median T1 relaxation coherence window (rounded to 1 decimal)?"
            ),
            expected_numeric_answer=5138.9,
            tolerance=0.2,
            unit="gates",
            required_doc_ids=["DOC-QPU-01", "DOC-QPU-03"],
            planted_rumor_doc_ids=["DOC-QPU-05", "DOC-QPU-06"],
            rumor_trap_values=[1166.7],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L3-06",
            level=3,
            domain="storage",
            question=(
                "How many seconds (rounded to 2 decimals) are required to write a full 500B-parameter "
                "model training checkpoint across the TitanFS-v3 storage racks of one Project Atlas AI pod?"
            ),
            expected_numeric_answer=10.0,
            tolerance=0.05,
            unit="seconds",
            required_doc_ids=["DOC-STO-01", "DOC-STO-04", "DOC-XREF-01"],
            planted_rumor_doc_ids=["DOC-STO-05", "DOC-STO-06"],
            rumor_trap_values=[4.63],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L3-07",
            level=3,
            domain="edge_runtime",
            question=(
                "What is the p99 cold-start latency speedup multiple (rounded to 2 decimals) of "
                "ZephyrRT-v2 compared to ZephyrRT-v1?"
            ),
            expected_numeric_answer=15.0,
            tolerance=0.05,
            unit="x",
            required_doc_ids=["DOC-RT-01", "DOC-RT-02"],
            planted_rumor_doc_ids=["DOC-RT-05", "DOC-RT-06"],
            rumor_trap_values=[1.42],
        ),
        GAIABenchmarkTask(
            task_id="GAIA-L3-08",
            level=3,
            domain="energy_storage",
            question=(
                "Calculate the turnkey installed CapEx (cell-level pack cost plus utility BOS/EPC adder) "
                "of the 30-container NovaGrid-Na400 campus microgrid, and express what percentage "
                "(rounded to 2 decimals) of CloudAI Division's FY2026 annual revenue it represents."
            ),
            expected_numeric_answer=4.76,
            tolerance=0.05,
            unit="%",
            required_doc_ids=["DOC-ENG-01", "DOC-ENG-04", "DOC-ENG-07", "DOC-XREF-03"],
            planted_rumor_doc_ids=["DOC-ENG-05", "DOC-ENG-06", "DOC-XREF-04"],
            rumor_trap_values=[2.49, 3.22, 3.68],
        ),
    ]
