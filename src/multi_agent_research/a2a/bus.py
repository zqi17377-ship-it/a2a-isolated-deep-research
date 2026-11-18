"""Asynchronous A2A JSON-RPC 2.0 Router & HTTP-Compatible Message Bus.

Routes standardized JSON-RPC 2.0 envelopes (`agent/card`, `agent/discover`,
`tasks/send`, `tasks/get`, `tasks/cancel`, `tasks/provide_input`) across
registered A2A agents while enforcing strict scratchpad stripping and
measuring async parallel vs sequential latency.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable, Optional, Protocol

import httpx

from multi_agent_research.a2a.models import (
    A2ATask,
    A2ATaskState,
    AgentCard,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
)

FORBIDDEN_LEAK_KEYS = {"raw_scratchpad", "internal_messages", "raw_corpus_dump", "chain_of_thought"}


class A2AAgentHandler(Protocol):
    """Protocol implemented by every A2A-compliant specialist agent."""

    @property
    def card(self) -> AgentCard:
        ...

    async def handle_task(self, task: A2ATask) -> A2ATask:
        ...


class A2ABus:
    """In-process and HTTP-compatible JSON-RPC 2.0 router for A2A agents."""

    def __init__(self, simulated_io_latency_ms: float = 0.0) -> None:
        self._agents: dict[str, A2AAgentHandler] = {}
        self._tasks: dict[str, A2ATask] = {}
        self._resume_callbacks: dict[str, Callable[[A2ATask, dict[str, Any]], Awaitable[A2ATask]]] = {}
        self._simulated_io_latency_ms = max(0.0, simulated_io_latency_ms)
        self._rpc_log: list[dict[str, Any]] = []

    @property
    def rpc_log(self) -> list[dict[str, Any]]:
        """Audit log of JSON-RPC envelopes exchanged over this bus."""
        return list(self._rpc_log)

    def register_agent(
        self,
        agent: A2AAgentHandler,
        resume_callback: Optional[Callable[[A2ATask, dict[str, Any]], Awaitable[A2ATask]]] = None,
    ) -> None:
        """Register an A2A agent and advertise its AgentCard on the bus."""
        agent_id = agent.card.agent_id
        self._agents[agent_id] = agent
        if resume_callback is not None:
            self._resume_callbacks[agent_id] = resume_callback

    def list_cards(self, skill_tag: Optional[str] = None) -> list[AgentCard]:
        """Return registered AgentCards, optionally filtered by skill tag."""
        cards = [a.card for a in self._agents.values()]
        if skill_tag is None:
            return cards
        return [
            card
            for card in cards
            if any(skill_tag in s.tags or skill_tag == s.skill_id for s in card.skills)
        ]

    @staticmethod
    def _enforce_context_boundary(task: A2ATask) -> None:
        """Ensure no raw worker scratchpad or internal chat history leaks into task output."""
        leaked = FORBIDDEN_LEAK_KEYS.intersection(task.structured_data.keys())
        if leaked:
            raise RuntimeError(
                f"A2A Context Isolation Violation in task {task.task_id}: "
                f"forbidden scratchpad keys {sorted(leaked)} attempted to cross boundary."
            )
        # Compute compact coordinator-visible payload token estimate
        artifact_tokens = sum(a.compact_token_estimate() for a in task.artifacts)
        quant_tokens = sum(22 for _ in task.quant_artifacts)
        struct_tokens = len(json.dumps(task.structured_data, default=str).split()) if task.structured_data else 0
        task.coordinator_payload_tokens = artifact_tokens + quant_tokens + struct_tokens + 6

    async def dispatch_rpc(self, request: JSONRPCRequest | dict[str, Any]) -> JSONRPCResponse:
        """Process a JSON-RPC 2.0 request envelope and return a JSON-RPC 2.0 response."""
        try:
            req = request if isinstance(request, JSONRPCRequest) else JSONRPCRequest.model_validate(request)
        except Exception as exc:
            return JSONRPCResponse(
                id="invalid",
                error=JSONRPCError(code=-32600, message=f"Invalid Request: {exc}"),
            )

        self._rpc_log.append({"direction": "request", "id": req.id, "method": req.method})

        try:
            if req.method == "agent/card":
                agent_id = req.params.get("agent_id", "")
                if agent_id not in self._agents:
                    return self._error(req.id, -32001, f"Unknown agent_id '{agent_id}'")
                card = self._agents[agent_id].card
                return self._ok(req.id, {"card": card.model_dump(mode="json")})

            if req.method == "agent/discover":
                skill_tag = req.params.get("skill_tag")
                cards = self.list_cards(skill_tag=skill_tag)
                return self._ok(req.id, {"cards": [c.model_dump(mode="json") for c in cards]})

            if req.method == "tasks/send":
                target_id = req.params.get("target_agent_id", "")
                sender_id = req.params.get("sender_id", "coordinator")
                payload = req.params.get("input_payload", {})
                task_id = req.params.get("task_id")
                defer_execution = bool(req.params.get("defer_execution", False))

                if target_id not in self._agents:
                    return self._error(req.id, -32001, f"Target agent '{target_id}' not registered")

                task_kwargs: dict[str, Any] = {
                    "sender_id": sender_id,
                    "target_agent_id": target_id,
                    "input_payload": payload,
                }
                if task_id:
                    task_kwargs["task_id"] = task_id

                task = A2ATask(**task_kwargs)
                self._tasks[task.task_id] = task

                if defer_execution:
                    return self._ok(req.id, {"task": task.model_dump(mode="json")})

                task.transition_to(A2ATaskState.WORKING)
                if self._simulated_io_latency_ms > 0:
                    await asyncio.sleep(self._simulated_io_latency_ms / 1000.0)

                agent = self._agents[target_id]
                try:
                    updated_task = await agent.handle_task(task)
                    self._enforce_context_boundary(updated_task)
                    if updated_task.state == A2ATaskState.WORKING:
                        updated_task.transition_to(A2ATaskState.COMPLETED)
                    self._tasks[updated_task.task_id] = updated_task
                except Exception as exc:
                    if task.state in {A2ATaskState.SUBMITTED, A2ATaskState.WORKING, A2ATaskState.INPUT_REQUIRED}:
                        task.transition_to(A2ATaskState.FAILED, reason=str(exc))
                    return self._error(req.id, -32002, f"Agent execution failed: {exc}", {"task_id": task.task_id})

                return self._ok(req.id, {"task": updated_task.model_dump(mode="json")})

            if req.method == "tasks/get":
                task_id = req.params.get("task_id", "")
                if task_id not in self._tasks:
                    return self._error(req.id, -32004, f"Task '{task_id}' not found")
                return self._ok(req.id, {"task": self._tasks[task_id].model_dump(mode="json")})

            if req.method == "tasks/cancel":
                task_id = req.params.get("task_id", "")
                if task_id not in self._tasks:
                    return self._error(req.id, -32004, f"Task '{task_id}' not found")
                task = self._tasks[task_id]
                try:
                    task.transition_to(A2ATaskState.CANCELED)
                except ValueError as exc:
                    return self._error(req.id, -32005, str(exc))
                return self._ok(req.id, {"task": task.model_dump(mode="json")})

            if req.method == "tasks/provide_input":
                task_id = req.params.get("task_id", "")
                additional_input = req.params.get("input", {})
                if task_id not in self._tasks:
                    return self._error(req.id, -32004, f"Task '{task_id}' not found")
                task = self._tasks[task_id]
                if task.state != A2ATaskState.INPUT_REQUIRED:
                    return self._error(
                        req.id,
                        -32006,
                        f"Task '{task_id}' is in state {task.state.value}, expected INPUT_REQUIRED",
                    )
                task.transition_to(A2ATaskState.WORKING)
                task.input_payload = {**task.input_payload, **additional_input}

                resume_fn = self._resume_callbacks.get(task.target_agent_id)
                if resume_fn is not None:
                    task = await resume_fn(task, additional_input)
                else:
                    task = await self._agents[task.target_agent_id].handle_task(task)

                self._enforce_context_boundary(task)
                if task.state == A2ATaskState.WORKING:
                    task.transition_to(A2ATaskState.COMPLETED)
                self._tasks[task.task_id] = task
                return self._ok(req.id, {"task": task.model_dump(mode="json")})

            return self._error(req.id, -32601, f"Method '{req.method}' not found")
        except Exception as exc:
            return self._error(req.id, -32603, f"Internal JSON-RPC error: {exc}")

    async def send_task(
        self,
        target_agent_id: str,
        input_payload: dict[str, Any],
        sender_id: str = "coordinator",
    ) -> A2ATask:
        """Convenience typed wrapper over JSON-RPC 2.0 `tasks/send`."""
        req = JSONRPCRequest(
            method="tasks/send",
            params={
                "sender_id": sender_id,
                "target_agent_id": target_agent_id,
                "input_payload": input_payload,
            },
        )
        resp = await self.dispatch_rpc(req)
        if resp.error is not None or resp.result is None:
            err_msg = resp.error.message if resp.error else "Empty RPC result"
            raise RuntimeError(f"A2A tasks/send failed: {err_msg}")
        return A2ATask.model_validate(resp.result["task"])

    def as_httpx_transport(self) -> httpx.AsyncBaseTransport:
        """Expose this A2A bus as an HTTP JSON-RPC 2.0 endpoint (`POST /a2a/rpc`) for `httpx.AsyncClient`."""
        bus = self

        async def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/.well-known/agent.json":
                cards = [c.model_dump(mode="json") for c in bus.list_cards()]
                return httpx.Response(200, json={"agents": cards})
            if request.method != "POST" or request.url.path != "/a2a/rpc":
                return httpx.Response(404, json={"error": "Not Found"})
            payload = json.loads(request.content.decode("utf-8"))
            rpc_resp = await bus.dispatch_rpc(payload)
            return httpx.Response(200, json=rpc_resp.model_dump(mode="json"))

        return httpx.MockTransport(_handler)

    def _ok(self, req_id: str, result: dict[str, Any]) -> JSONRPCResponse:
        self._rpc_log.append({"direction": "response", "id": req_id, "ok": True})
        return JSONRPCResponse(id=req_id, result=result)

    def _error(
        self,
        req_id: str,
        code: int,
        message: str,
        data: Optional[dict[str, Any]] = None,
    ) -> JSONRPCResponse:
        self._rpc_log.append({"direction": "response", "id": req_id, "ok": False, "code": code})
        return JSONRPCResponse(id=req_id, error=JSONRPCError(code=code, message=message, data=data))
