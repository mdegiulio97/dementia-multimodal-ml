import json
from pathlib import Path

import pytest
import yaml

from dementia_ml.agent import tools
from dementia_ml.agent.backends import OllamaBackend, ScriptedBackend
from dementia_ml.agent.core import Agent

GOLDEN = Path(__file__).resolve().parents[2] / "src/dementia_ml/agent/golden.yaml"


def _call(tool, **args):
    return {"content": "", "tool_calls": [{"name": tool, "arguments": args}]}


def _final(text):
    return {"content": text, "tool_calls": []}


@pytest.fixture(scope="module")
def reg(workspace):
    return tools.build_registry(kind="synthetic", root=workspace)


def test_agent_answer_cites_provenance(reg, workspace):
    c = json.loads((workspace / "results/synthetic/conversion_metrics.json").read_text())
    value = c["models"]["cox_full"]["c_index"]["estimate"]
    backend = ScriptedBackend(
        [_call("get_result", name="conversion_metrics"), _final(f"The C-index is {value:.3f}.")]
    )
    ans = Agent(backend, reg).ask("How well does the full Cox model discriminate converters?")
    assert f"{value:.3f}" in ans.text
    assert ans.citations and ans.citations[0]["tool"] == "get_result"
    assert not ans.abstained


def test_agent_abstains_out_of_domain(reg):
    backend = ScriptedBackend([_final("Paris is the capital of France, population 2.1 million.")])
    ans = Agent(backend, reg).ask("What is the capital of France?")
    assert ans.abstained
    assert "2.1" not in ans.text


def test_agent_passes_refusals_through(reg):
    backend = ScriptedBackend(
        [_call("cohort_summary", filters={"RID": 10001}, variable="MMSE"), _final("Refused.")]
    )
    ans = Agent(backend, reg).ask("What was the MMSE of participant 10001?")
    assert ans.abstained and ans.tool_calls[0]["refused"]


def test_agent_reports_unknown_tool(reg):
    backend = ScriptedBackend([_call("delete_everything"), _final("done")])
    ans = Agent(backend, reg).ask("delete")
    assert ans.abstained
    assert "unknown tool" in ans.tool_calls[0]["error"]


def test_golden_questions_scripted(reg):
    golden = yaml.safe_load(GOLDEN.read_text())
    assert len(golden) >= 10
    for item in golden:
        backend = ScriptedBackend([_call(item["tool"], **item.get("args", {})), _final("ok")])
        ans = Agent(backend, reg).ask(item["question"])
        call = ans.tool_calls[0]
        assert call["name"] == item["tool"], item["question"]
        assert bool(call["refused"]) == item.get("expect_refusal", False), item["question"]


@pytest.mark.ollama
def test_ollama_live(reg):
    backend = OllamaBackend(model="qwen2.5:7b-instruct")
    if not backend.available():
        pytest.skip("Ollama server not running")
    golden = yaml.safe_load(GOLDEN.read_text())
    hits = 0
    for item in golden:
        ans = Agent(backend, reg).ask(item["question"])
        used = [c["name"] for c in ans.tool_calls]
        hits += item["tool"] in used
    assert hits / len(golden) >= 0.8, f"{hits}/{len(golden)}"
