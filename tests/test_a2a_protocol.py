"""Unit tests for Agent2Agent (A2A) JSON-RPC 2.0 Protocol, Task Lifecycle & Boundary Isolation."""

from __future__ import annotations

import httpx
import pytest

from multi_agent_research.a2a.bus import A2ABus
from multi_agent_research.a2a.models import (
    A2ATask,
    A2ATaskState,
    AgentCard,
    AgentSkill,
    CredibilityTier,
    EvidenceArtifact,
    IsolationLevel,
    JSONRPCRequest,
)
from multi_agent_research.agents.base import BaseA2AAgent, EphemeralAgentContext


class DummyEchoAgent(BaseA2AAgent):
    def __init__(self, agent_id: str = "dummy-echo", leak_scratchpad: bool = False) -> None:
        self._leak = leak_scratchpad
        self._card = AgentCard(
            agent_id=agent_id,
            name="Dummy Echo Worker",
            version="1.0.0",
            description="Test A2A worker agent",
            endpoint=f"a2a://local/{agent_id}",
            skills=[
                AgentSkill(
                    skill_id="echo_skill",
                    name="Echo Skill",
                    description="Echoes typed evidence artifact",
                    tags=["test", "echo"],
                )
            ],
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            isolation_level=IsolationLevel.STRICT_EPHEMERAL,
        )

    @property
    def card(self) -> AgentCard:
        return self._card

    async def _execute_isolated(self, task: A2ATask, ctx: EphemeralAgentContext) -> A2ATask:
        ctx.append_thought("Private internal scratchpad thought that must stay ephemeral " * 15)
        if task.input_payload.get("require_clarification") and "clarified_unit" not in task.input_payload:
            task.transition_to(A2ATaskState.INPUT_REQUIRED, reason="Please specify unit (MW or kW)")
            return task

        task.artifacts = [
            EvidenceArtifact(
                metric_key="test.metric",
                statement="Verified test metric is 42.0.",
                numeric_values={"val": 42.0},
                source_doc_id="DOC-ACC-01",
                exact_quote_span="312.0 TFLOPS",
                source_credibility_tier=CredibilityTier.PRIMARY_SPEC,
                confidence=0.95,
            )
        ]
        if self._leak:
            task.structured_data = {"raw_scratchpad": "LEAKED INTERNAL PROMPT"}
        else:
            task.structured_data = {"unit_used": task.input_payload.get("clarified_unit", "default")}
        return task


async def test_agent_card_discovery_and_skill_filtering() -> None:
    bus = A2ABus()
    bus.register_agent(DummyEchoAgent("worker-a"))
    bus.register_agent(DummyEchoAgent("worker-b"))

    resp_card = await bus.dispatch_rpc(
        JSONRPCRequest(method="agent/card", params={"agent_id": "worker-a"})
    )
    assert resp_card.error is None
    assert resp_card.result is not None
    assert resp_card.result["card"]["agent_id"] == "worker-a"
    assert resp_card.result["card"]["isolation_level"] == "strict_ephemeral"

    resp_disc = await bus.dispatch_rpc(
        JSONRPCRequest(method="agent/discover", params={"skill_tag": "echo"})
    )
    assert resp_disc.result is not None
    assert len(resp_disc.result["cards"]) == 2

    resp_missing = await bus.dispatch_rpc(
        JSONRPCRequest(method="agent/card", params={"agent_id": "non-existent"})
    )
    assert resp_missing.error is not None
    assert resp_missing.error.code == -32001


def test_a2a_task_lifecycle_state_machine() -> None:
    task = A2ATask(sender_id="coord", target_agent_id="worker-a", input_payload={})
    assert task.state == A2ATaskState.SUBMITTED

    # Direct SUBMITTED -> COMPLETED must be rejected (must pass through WORKING)
    with pytest.raises(ValueError, match="Invalid A2A task state transition"):
        task.transition_to(A2ATaskState.COMPLETED)

    task.transition_to(A2ATaskState.WORKING)
    task.transition_to(A2ATaskState.INPUT_REQUIRED, reason="Need parameter")
    assert task.required_input_prompt == "Need parameter"

    task.transition_to(A2ATaskState.WORKING)
    task.transition_to(A2ATaskState.COMPLETED)
    assert [h.state for h in task.history] == [
        A2ATaskState.SUBMITTED,
        A2ATaskState.WORKING,
        A2ATaskState.INPUT_REQUIRED,
        A2ATaskState.WORKING,
        A2ATaskState.COMPLETED,
    ]

    # Terminal state cannot transition again
    with pytest.raises(ValueError, match="Invalid A2A task state transition"):
        task.transition_to(A2ATaskState.WORKING)


async def test_a2a_input_required_and_provide_input_resume() -> None:
    bus = A2ABus()
    bus.register_agent(DummyEchoAgent("interactive-worker"))

    send_resp = await bus.dispatch_rpc(
        JSONRPCRequest(
            method="tasks/send",
            params={
                "target_agent_id": "interactive-worker",
                "input_payload": {"require_clarification": True},
            },
        )
    )
    assert send_resp.result is not None
    paused_task = A2ATask.model_validate(send_resp.result["task"])
    assert paused_task.state == A2ATaskState.INPUT_REQUIRED

    resume_resp = await bus.dispatch_rpc(
        JSONRPCRequest(
            method="tasks/provide_input",
            params={"task_id": paused_task.task_id, "input": {"clarified_unit": "MW"}},
        )
    )
    assert resume_resp.error is None
    assert resume_resp.result is not None
    done_task = A2ATask.model_validate(resume_resp.result["task"])
    assert done_task.state == A2ATaskState.COMPLETED
    assert done_task.structured_data["unit_used"] == "MW"
    assert len(done_task.artifacts) == 1


async def test_a2a_task_cancel_rpc() -> None:
    bus = A2ABus()
    bus.register_agent(DummyEchoAgent("cancel-worker"))

    defer_resp = await bus.dispatch_rpc(
        JSONRPCRequest(
            method="tasks/send",
            params={
                "target_agent_id": "cancel-worker",
                "input_payload": {},
                "defer_execution": True,
            },
        )
    )
    assert defer_resp.result is not None
    tid = defer_resp.result["task"]["task_id"]
    assert defer_resp.result["task"]["state"] == "SUBMITTED"

    cancel_resp = await bus.dispatch_rpc(
        JSONRPCRequest(method="tasks/cancel", params={"task_id": tid})
    )
    assert cancel_resp.error is None
    assert cancel_resp.result is not None
    assert cancel_resp.result["task"]["state"] == "CANCELED"

    double_cancel = await bus.dispatch_rpc(
        JSONRPCRequest(method="tasks/cancel", params={"task_id": tid})
    )
    assert double_cancel.error is not None
    assert double_cancel.error.code == -32005


async def test_a2a_httpx_http_jsonrpc_transport() -> None:
    bus = A2ABus()
    bus.register_agent(DummyEchoAgent("http-worker"))

    async with httpx.AsyncClient(transport=bus.as_httpx_transport(), base_url="http://a2a.local") as client:
        well_known = await client.get("/.well-known/agent.json")
        assert well_known.status_code == 200
        assert well_known.json()["agents"][0]["agent_id"] == "http-worker"

        rpc_req = JSONRPCRequest(
            method="tasks/send",
            params={"target_agent_id": "http-worker", "input_payload": {}},
        )
        http_resp = await client.post("/a2a/rpc", json=rpc_req.model_dump(mode="json"))
        assert http_resp.status_code == 200
        body = http_resp.json()
        assert body["error"] is None
        assert body["result"]["task"]["state"] == "COMPLETED"
        assert body["result"]["task"]["artifacts"][0]["numeric_values"]["val"] == 42.0


async def test_a2a_context_isolation_blocks_scratchpad_leak() -> None:
    bus = A2ABus()
    bus.register_agent(DummyEchoAgent("leaky-worker", leak_scratchpad=True))

    resp = await bus.dispatch_rpc(
        JSONRPCRequest(
            method="tasks/send",
            params={"target_agent_id": "leaky-worker", "input_payload": {}},
        )
    )
    assert resp.error is not None
    assert resp.error.code == -32002
    assert "Context Isolation Violation" in resp.error.message
    failed_tid = resp.error.data["task_id"]

    get_resp = await bus.dispatch_rpc(JSONRPCRequest(method="tasks/get", params={"task_id": failed_tid}))
    assert get_resp.result is not None
    assert get_resp.result["task"]["state"] == "FAILED"
