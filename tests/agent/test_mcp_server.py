import asyncio
import json

import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from dementia_ml.agent import mcp_server, tools  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def test_mcp_lists_all_registry_tools(workspace):
    server = mcp_server.build_server(kind="synthetic", root=workspace)

    async def go():
        async with Client(server) as client:
            return await client.list_tools()

    listed = {t.name for t in _run(go()).tools}
    assert listed == set(tools.build_registry("synthetic", workspace))


def test_mcp_call_refuses_small_cells(workspace):
    server = mcp_server.build_server(kind="synthetic", root=workspace)

    async def go():
        async with Client(server) as client:
            return await client.call_tool(
                "cohort_summary", {"filters": {"AGE": [95, 200]}, "variable": "MMSE"}
            )

    res = _run(go())
    payload = json.loads(res.content[0].text)
    assert payload["refused"] and payload["value"] is None


def test_mcp_call_returns_value_with_provenance(workspace):
    server = mcp_server.build_server(kind="synthetic", root=workspace)

    async def go():
        async with Client(server) as client:
            return await client.call_tool("get_result", {"name": "conversion_metrics"})

    payload = json.loads(_run(go()).content[0].text)
    assert payload["value"]["models"]["cox_full"]["c_index"]["estimate"] > 0.5
    assert payload["provenance"]["git_commit"]
