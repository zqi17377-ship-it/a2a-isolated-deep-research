"""Base A2A Agent with Strict Ephemeral Context Window Isolation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from multi_agent_research.a2a.models import A2ATask, AgentCard
from multi_agent_research.llm import estimate_tokens


@dataclass
class EphemeralAgentContext:
    """Isolated per-invocation scratchpad destroyed upon A2A task completion."""

    agent_id: str
    task_id: str
    scratchpad_entries: list[str] = field(default_factory=list)

    def append_thought(self, text: str) -> None:
        self.scratchpad_entries.append(text)

    def total_ephemeral_tokens(self) -> int:
        return sum(estimate_tokens(entry) for entry in self.scratchpad_entries)


class BaseA2AAgent(ABC):
    """Abstract base class for context-isolated A2A specialist agents."""

    @property
    @abstractmethod
    def card(self) -> AgentCard:
        """Return standardized A2A AgentCard capability advertisement."""

    async def handle_task(self, task: A2ATask) -> A2ATask:
        """Create a fresh ephemeral context, execute worker logic, and discard scratchpad."""
        ctx = EphemeralAgentContext(agent_id=self.card.agent_id, task_id=task.task_id)
        updated_task = await self._execute_isolated(task, ctx)
        updated_task.ephemeral_worker_tokens = ctx.total_ephemeral_tokens()
        return updated_task

    @abstractmethod
    async def _execute_isolated(self, task: A2ATask, ctx: EphemeralAgentContext) -> A2ATask:
        """Execute task inside isolated `ctx` and populate typed A2ATask artifacts."""
