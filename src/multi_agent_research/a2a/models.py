"""Agent2Agent (A2A) Protocol Models & Typed Evidence Primitives.

Implements capability advertisement (`AgentCard`), task lifecycle state machine
(`A2ATask` & `A2ATaskState`), strict-isolation `EvidenceArtifact` & `QuantArtifact`
payloads, and JSON-RPC 2.0 request/response envelopes.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class IsolationLevel(str, Enum):
    """Context window isolation contract advertised by an A2A agent."""

    STRICT_EPHEMERAL = "strict_ephemeral"
    SHARED_SESSION = "shared_session"


class CredibilityTier(str, Enum):
    """Source authority tier used for cross-source conflict resolution."""

    PRIMARY_SPEC = "PRIMARY_SPEC"          # Official specs, audited telemetry, peer-reviewed papers
    SECONDARY_REPORT = "SECONDARY_REPORT"  # Technical trade press, third-party benchmark summaries
    UNVERIFIED_RUMOR = "UNVERIFIED_RUMOR"  # Anonymous forums, social rumor blogs, unverified leaks

    @property
    def authority_weight(self) -> float:
        weights = {
            CredibilityTier.PRIMARY_SPEC: 1.0,
            CredibilityTier.SECONDARY_REPORT: 0.65,
            CredibilityTier.UNVERIFIED_RUMOR: 0.15,
        }
        return weights[self]


class VerificationStatus(str, Enum):
    """Verification outcome assigned by AdversarialCriticAgent."""

    UNVERIFIED = "UNVERIFIED"
    VERIFIED_PRIMARY = "VERIFIED_PRIMARY"
    VERIFIED_CORROBORATED = "VERIFIED_CORROBORATED"
    CONFLICT_RESOLVED = "CONFLICT_RESOLVED"
    REJECTED_RUMOR = "REJECTED_RUMOR"
    REJECTED_FABRICATED_QUOTE = "REJECTED_FABRICATED_QUOTE"


class A2ATaskState(str, Enum):
    """Standard A2A Task lifecycle states."""

    SUBMITTED = "SUBMITTED"
    WORKING = "WORKING"
    INPUT_REQUIRED = "INPUT_REQUIRED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


VALID_TASK_TRANSITIONS: dict[A2ATaskState, set[A2ATaskState]] = {
    A2ATaskState.SUBMITTED: {
        A2ATaskState.WORKING,
        A2ATaskState.CANCELED,
        A2ATaskState.FAILED,
    },
    A2ATaskState.WORKING: {
        A2ATaskState.INPUT_REQUIRED,
        A2ATaskState.COMPLETED,
        A2ATaskState.FAILED,
        A2ATaskState.CANCELED,
    },
    A2ATaskState.INPUT_REQUIRED: {
        A2ATaskState.WORKING,
        A2ATaskState.CANCELED,
        A2ATaskState.FAILED,
    },
    A2ATaskState.COMPLETED: set(),
    A2ATaskState.FAILED: set(),
    A2ATaskState.CANCELED: set(),
}


class AgentSkill(BaseModel):
    """Single capability skill advertised inside an AgentCard."""

    model_config = ConfigDict(extra="forbid")

    skill_id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)


class AgentCard(BaseModel):
    """Standardized A2A capability advertisement card."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    name: str
    version: str = "1.0.0"
    description: str
    endpoint: str
    skills: list[AgentSkill]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    isolation_level: IsolationLevel = IsolationLevel.STRICT_EPHEMERAL


class EvidenceArtifact(BaseModel):
    """Typed, provenance-grounded claim returned across the A2A boundary.

    Sub-agents emit structured `EvidenceArtifact` objects instead of raw
    conversational scratchpads, preventing context bloat and rumor contagion
    in the Coordinator's context window.
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(default_factory=lambda: f"clm-{uuid.uuid4().hex[:8]}")
    metric_key: str = Field(
        description="Canonical entity.metric identifier used to align corroborating or conflicting claims"
    )
    statement: str
    numeric_values: dict[str, float] = Field(default_factory=dict)
    source_doc_id: str
    exact_quote_span: str
    source_credibility_tier: CredibilityTier
    confidence: float = Field(ge=0.0, le=1.0)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    critic_notes: str = ""

    def compact_token_estimate(self) -> int:
        """Approximate token footprint of this structured artifact in Coordinator state."""
        text = (
            f"{self.claim_id} {self.metric_key} {self.statement} "
            f"{self.numeric_values} {self.source_doc_id} {self.source_credibility_tier.value}"
        )
        return max(12, len(text.split()) + 8)


class QuantArtifact(BaseModel):
    """Deterministic Python computation result produced by QuantComputeAgent."""

    model_config = ConfigDict(extra="forbid")

    calc_id: str = Field(default_factory=lambda: f"calc-{uuid.uuid4().hex[:8]}")
    metric_key: str
    description: str
    formula_expression: str
    input_variables: dict[str, float]
    result_value: float
    unit: str = ""
    source_claim_ids: list[str] = Field(default_factory=list)


class SubTaskSpec(BaseModel):
    """Independent retrieval sub-task emitted by LeadPlannerAgent."""

    model_config = ConfigDict(extra="forbid")

    subtask_id: str
    search_query: str
    target_entity: str
    target_metrics: list[str] = Field(default_factory=list)
    prefer_primary_only: bool = False


class StateTransitionRecord(BaseModel):
    """Timestamped state transition entry for A2A auditability."""

    model_config = ConfigDict(extra="forbid")

    state: A2ATaskState
    timestamp_monotonic: float


class A2ATask(BaseModel):
    """Stateful A2A Task object managed by the A2A JSON-RPC 2.0 Router."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex[:10]}")
    sender_id: str
    target_agent_id: str
    state: A2ATaskState = A2ATaskState.SUBMITTED
    input_payload: dict[str, Any]
    required_input_prompt: Optional[str] = None
    artifacts: list[EvidenceArtifact] = Field(default_factory=list)
    quant_artifacts: list[QuantArtifact] = Field(default_factory=list)
    structured_data: dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    ephemeral_worker_tokens: int = Field(
        default=0,
        description="Tokens consumed inside the worker's isolated context (never copied to Coordinator)",
    )
    coordinator_payload_tokens: int = Field(
        default=0,
        description="Compact tokens serialized back across the A2A boundary to Coordinator",
    )
    history: list[StateTransitionRecord] = Field(
        default_factory=lambda: [
            StateTransitionRecord(
                state=A2ATaskState.SUBMITTED,
                timestamp_monotonic=time.perf_counter(),
            )
        ]
    )

    def transition_to(self, next_state: A2ATaskState, reason: Optional[str] = None) -> None:
        """Validate and apply an A2A lifecycle state transition."""
        allowed = VALID_TASK_TRANSITIONS.get(self.state, set())
        if next_state not in allowed:
            raise ValueError(
                f"Invalid A2A task state transition for {self.task_id}: "
                f"{self.state.value} -> {next_state.value}. Allowed: {[s.value for s in allowed]}"
            )
        self.state = next_state
        if next_state == A2ATaskState.INPUT_REQUIRED and reason:
            self.required_input_prompt = reason
        if next_state == A2ATaskState.FAILED and reason:
            self.error_message = reason
        self.history.append(
            StateTransitionRecord(
                state=next_state,
                timestamp_monotonic=time.perf_counter(),
            )
        )


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 Error object."""

    model_config = ConfigDict(extra="forbid")

    code: int
    message: str
    data: Optional[ dict[str, Any] ] = None


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 Request envelope for A2A methods."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: str = Field(default_factory=lambda: f"rpc-{uuid.uuid4().hex[:8]}")
    method: Literal[
        "agent/card",
        "agent/discover",
        "tasks/send",
        "tasks/get",
        "tasks/cancel",
        "tasks/provide_input",
    ]
    params: dict[str, Any] = Field(default_factory=dict)


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 Response envelope."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: str
    result: Optional[dict[str, Any]] = None
    error: Optional[JSONRPCError] = None
