# Multi-Agent Topology & Echo-Chamber Suppression Evaluation Report

- **Corpus Scale:** `60` heterogeneous technical documents (`PRIMARY_SPEC`: 27, `SECONDARY_REPORT`: 16, `UNVERIFIED_RUMOR`: 17)
- **Evaluation Suite:** `24` GAIA-inspired multi-hop tasks (8 Level-1, 8 Level-2, 8 Level-3 with planted echo-chamber rumors)

## 1. Main 3-Topology Comparison Table

| Metric | Topology A: Monolithic Single-Agent | Topology B: Shared-History Group Chat | Topology C (Ours): A2A Isolated DAG + Quant + Critic |
| :--- | :---: | :---: | :---: |
| **Overall Key-Fact Accuracy (%)** | 16.7% (4/24) | 0.0% (0/24) | **87.5% (21/24)** |
| Level 1 Accuracy (2-Hop Synthesis) | 50.0% | 0.0% | **87.5%** |
| Level 2 Accuracy (3+ Hop + Quant) | 0.0% | 0.0% | **75.0%** |
| Level 3 Accuracy (Conflicting Rumors) | 0.0% | 0.0% | **100.0%** |
| **Numerical Exact Match (%)** | 29.2% | 20.8% | **87.5%** |
| **Echo-Chamber Rumor Adoption (L3, ↓)** | 87.5% | 100.0% | **0.0%** |
| **Citation Verbatim Grounding (%)** | 92.0% | 60.7% | **100.0%** |
| **Coordinator Peak Context Tokens (↓)** | 1139 | 1290 | **376** (-67.0% vs A) |
| **Mean Execution Latency (ms, ↓)** | 81.56 ms | 82.12 ms | **237.76 ms** (`1.17x` vs Seq A2A) |

## 2. Failure Case Attribution by Topology

### Topology A (Monolithic)
- **`echo_chamber_rumor_adoption`** (`16` tasks): `GAIA-L1-02, GAIA-L1-04, GAIA-L1-06, GAIA-L1-07, GAIA-L2-01, GAIA-L2-03, GAIA-L2-04, GAIA-L2-06, GAIA-L2-08, GAIA-L3-01, GAIA-L3-03, GAIA-L3-04, GAIA-L3-05, GAIA-L3-06, GAIA-L3-07, GAIA-L3-08`
- **`in_context_mental_math_drift`** (`2` tasks): `GAIA-L2-07, GAIA-L3-02`
- **`retrieval_slot_crowding_or_distraction`** (`2` tasks): `GAIA-L2-02, GAIA-L2-05`

### Topology B (Shared Group Chat)
- **`echo_chamber_rumor_adoption`** (`24` tasks): `GAIA-L1-01, GAIA-L1-02, GAIA-L1-03, GAIA-L1-04, GAIA-L1-05, GAIA-L1-06, GAIA-L1-07, GAIA-L1-08, GAIA-L2-01, GAIA-L2-02, GAIA-L2-03, GAIA-L2-04, GAIA-L2-05, GAIA-L2-06, GAIA-L2-07, GAIA-L2-08, GAIA-L3-01, GAIA-L3-02, GAIA-L3-03, GAIA-L3-04, GAIA-L3-05, GAIA-L3-06, GAIA-L3-07, GAIA-L3-08`

### Topology C (A2A Isolated DAG)
- **`retrieval_slot_crowding_or_distraction`** (`3` tasks): `GAIA-L1-02, GAIA-L2-02, GAIA-L2-05`
