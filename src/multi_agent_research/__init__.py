"""Cross-Protocol Multi-Agent Deep Research System (Project 06-F4).

Implements Google A2A-inspired JSON-RPC 2.0 agent interoperability, strict ephemeral
context isolation, deterministic AST-verified quantitative computation, and adversarial
cross-source echo-chamber suppression on a 24-task GAIA-style multi-hop benchmark.
"""

from multi_agent_research.async_compat import ensure_event_loop_policy

ensure_event_loop_policy()

__all__ = ["__version__", "ensure_event_loop_policy"]
__version__ = "0.1.0"
