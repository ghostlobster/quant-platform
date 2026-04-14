"""
agents/base.py — AgentProvider protocol and AgentSignal result type.

All specialist agents implement AgentProvider.run(context) and return
an AgentSignal with a direction, confidence, and optional reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["AgentSignal", "AgentProvider"]


@dataclass
class AgentSignal:
    """Structured output from a single specialist agent."""

    agent_name: str
    signal: str                  # 'bullish' | 'bearish' | 'neutral'
    confidence: float            # [0.0, 1.0]
    reasoning: str = ""
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class AgentProvider(Protocol):
    """Duck-typed interface for specialist trading agents."""

    name: str

    def run(self, context: dict) -> AgentSignal:
        """
        Analyse *context* and return a signal.

        Parameters
        ----------
        context : dict with optional keys:
            ticker    : str   — target symbol
            regime    : str   — current market regime
            portfolio : dict  — portfolio snapshot
            prices    : dict  — {ticker: price}

        Returns
        -------
        AgentSignal
        """
        ...
