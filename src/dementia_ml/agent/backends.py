"""LLM backends. Responses are normalised to {"content": str, "tool_calls": [{name, arguments}]}."""

from __future__ import annotations

import json
from typing import Protocol

import httpx


class LLMBackend(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> dict: ...


class ScriptedBackend:
    """Replays a fixed list of responses (tests and demos without a model)."""

    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.seen: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        self.seen.append(list(messages))
        if not self.script:
            return {"content": "", "tool_calls": []}
        return self.script.pop(0)


class OllamaBackend:
    """Local model served by Ollama; data never leave the machine."""

    def __init__(self, model: str = "qwen2.5:7b-instruct", host: str = "http://localhost:11434"):
        self.model, self.host = model, host.rstrip("/")

    def available(self) -> bool:
        try:
            return httpx.get(f"{self.host}/api/tags", timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        r = httpx.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=600,
        )
        r.raise_for_status()
        msg = r.json()["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            args = c["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            calls.append({"name": c["function"]["name"], "arguments": args})
        return {"content": msg.get("content", ""), "tool_calls": calls}
