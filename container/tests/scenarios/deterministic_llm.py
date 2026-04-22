"""Deterministic LLM stub used by real-scenario tests.

The stub patches ``app.llm.client.complete`` and routes calls by
``agent_id`` to a pre-canned reply queue derived from the scenario's
``llm:`` block. Per-agent replies are consumed FIFO so prompt-text
drift in the production agents does not break scenarios.

A miss raises ``LLMStubMissError`` with enough context to add a fixture.
"""

from __future__ import annotations

from collections import defaultdict, deque


class LLMStubMissError(AssertionError):
    """Raised when the LLM stub has no canned reply for a request."""

    def __init__(self, agent_id: str, prompt_excerpt: str, scenario_id: str | None = None) -> None:
        self.agent_id = agent_id
        self.prompt_excerpt = prompt_excerpt
        self.scenario_id = scenario_id
        msg = (
            f"DeterministicLlmStub miss: agent_id={agent_id!r}\n"
            f"Scenario: {scenario_id}\n"
            f"Prompt excerpt: {prompt_excerpt[:400]!r}\n"
            f"Add a reply under llm.agents.{agent_id} (or llm.classify) in the YAML."
        )
        super().__init__(msg)


class DeterministicLlmStub:
    """FIFO-per-agent_id LLM stub.

    Construct with the YAML ``llm:`` payload via :meth:`feed_from_scenario`
    and install with :meth:`install` to patch ``app.llm.client.complete``.
    """

    def __init__(self) -> None:
        self._queues: dict[str, deque[str]] = defaultdict(deque)
        self._calls: list[tuple[str, str]] = []
        self._scenario_id: str | None = None
        self._default_reply: str | None = None

    def feed(self, agent_id: str, replies: list[str] | str) -> None:
        if isinstance(replies, str):
            replies = [replies]
        for r in replies:
            self._queues[agent_id].append(r)

    def feed_from_scenario(self, scenario) -> None:
        from .types import Scenario

        if not isinstance(scenario, Scenario):  # pragma: no cover - defensive
            raise TypeError("feed_from_scenario requires a Scenario instance")
        self._scenario_id = scenario.id
        if scenario.llm.classify:
            self.feed("orchestrator", scenario.llm.classify)
        for agent_id, replies in scenario.llm.agents.items():
            self.feed(agent_id, replies)
        for turn in scenario.follow_up:
            if turn.llm.classify:
                self.feed("orchestrator", turn.llm.classify)
            for agent_id, replies in turn.llm.agents.items():
                self.feed(agent_id, replies)

    def set_default_reply(self, reply: str | None) -> None:
        self._default_reply = reply

    @property
    def calls(self) -> list[tuple[str, str]]:
        return list(self._calls)

    async def complete(self, agent_id: str, messages, **_kwargs) -> str:
        try:
            last_msg = messages[-1] if messages else {}
            prompt = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
        except Exception:
            prompt = ""
        queue = self._queues.get(agent_id)
        if queue:
            reply = queue.popleft()
            self._calls.append((agent_id, prompt[:200]))
            return reply
        if self._default_reply is not None:
            self._calls.append((agent_id, prompt[:200]))
            return self._default_reply
        raise LLMStubMissError(agent_id, prompt, self._scenario_id)
