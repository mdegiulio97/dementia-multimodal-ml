"""Tool-using agent: the model chooses tools, the tools compute every number."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field

from dementia_ml.agent.backends import LLMBackend
from dementia_ml.agent.tools import Tool, ToolResult

SYSTEM_PROMPT = """You are an analysis assistant for a dementia research project.
You can only answer with numbers returned by the tools. Never estimate, recall or
invent a number. Call a tool for every quantitative claim, and report its
uncertainty and provenance. If no tool covers the question, say that the
project's data cannot answer it. Never ask for or discuss individual participants."""

ABSTAIN_TEXT = (
    "I can't answer that from this project's data and models. I can describe the data, "
    "summarise the MCI cohort, predict conversion risk for a profile, explain a "
    "prediction, or report stored results (conversion models, trajectories, Mendelian "
    "randomisation, EEG)."
)


UNVERIFIED_TEXT = (
    "The model's answer contained numbers that no tool returned, so it was withheld. Tool results: "
)
NUMBER = re.compile(r"\d+(?:\.\d+)?")
ALWAYS_ALLOWED = {95.0}  # "95% CI"


def _numbers(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, int | float):
        return [float(obj)] if math.isfinite(obj) else []
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _numbers(v)]
    if isinstance(obj, list | tuple):
        return [x for v in obj for x in _numbers(v)]
    return []


def unverified_numbers(text: str, payloads: list, question: str) -> list[str]:
    """Numbers in ``text`` that match no tool value (as is or as a percentage, at the
    precision written) and do not come from the question."""
    allowed = ALWAYS_ALLOWED | {float(n) for n in NUMBER.findall(question)}
    values = [abs(v) for v in _numbers(payloads)]
    bad = []
    for token in NUMBER.findall(text):
        n = float(token)
        if n in allowed:
            continue
        decimals = len(token.split(".")[1]) if "." in token else 0
        tol = 0.5 * 10**-decimals + 1e-9
        if not any(abs(v - n) <= tol or abs(100 * v - n) <= tol for v in values):
            bad.append(token)
    return bad


@dataclass
class Answer:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    abstained: bool = False


class Agent:
    def __init__(self, backend: LLMBackend, registry: dict[str, Tool], max_steps: int = 4):
        self.backend, self.registry, self.max_steps = backend, registry, max_steps
        self.schemas = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in registry.values()
        ]

    def _execute(self, name: str, arguments: dict) -> dict:
        record = {
            "name": name,
            "arguments": arguments,
            "refused": None,
            "error": None,
            "result": None,
        }
        tool = self.registry.get(name)
        if tool is None:
            record["error"] = f"unknown tool {name!r}"
            return record
        try:
            res: ToolResult = tool.fn(**arguments)
        except (TypeError, ValueError, KeyError, FileNotFoundError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            return record
        record["refused"] = res.refused
        record["result"] = asdict(res)
        return record

    def ask(self, question: str) -> Answer:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        calls: list[dict] = []
        final = ""
        for _ in range(self.max_steps):
            resp = self.backend.chat(messages, self.schemas)
            if not resp.get("tool_calls"):
                final = resp.get("content", "")
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": resp.get("content", ""),
                    "tool_calls": [
                        {"function": {"name": c["name"], "arguments": c.get("arguments", {})}}
                        for c in resp["tool_calls"]
                    ],
                }
            )
            for c in resp["tool_calls"]:
                rec = self._execute(c["name"], c.get("arguments", {}))
                calls.append(rec)
                payload = (
                    rec["result"]
                    if rec["result"]
                    else {"error": rec["error"], "refused": rec["refused"]}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": c["name"],
                        "content": json.dumps(payload, default=str),
                    }
                )
        ok = [c for c in calls if c["result"] and not c["refused"]]
        if not ok:
            refusals = [c["refused"] for c in calls if c["refused"]]
            text = f"Request refused: {refusals[0]}." if refusals else ABSTAIN_TEXT
            return Answer(text=text, tool_calls=calls, abstained=True)
        citations = [{"tool": c["name"], "provenance": c["result"]["provenance"]} for c in ok]
        payloads = [{"value": c["result"]["value"], "u": c["result"]["uncertainty"]} for c in ok]
        if not final.strip() or unverified_numbers(final, payloads, question):
            # Every number must come from a tool: otherwise show the tool results instead.
            text = UNVERIFIED_TEXT + json.dumps(payloads, default=str)[:4000]
            return Answer(text=text, tool_calls=calls, citations=citations, abstained=True)
        return Answer(text=final, tool_calls=calls, citations=citations)
