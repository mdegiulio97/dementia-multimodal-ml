"""Expose the analysis tools over the Model Context Protocol (stdio).

    uv run --extra mcp dementia-mcp            # synthetic results
    uv run --extra mcp dementia-mcp --kind adni --allow-adni-to-client   # local clients only

Any MCP client (a desktop assistant, an IDE, another agent) then gets the same
tools as the local agent, with the same refusals and provenance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from dementia_ml.agent.tools import build_registry


def build_server(
    kind: str = "synthetic", root: Path | None = None, allow_adni: bool = False
) -> Server:
    if kind == "adni" and not allow_adni:
        raise ValueError(
            "ADNI mode sends ADNI-derived aggregates to whatever model drives the MCP client, "
            "which is often a cloud service. The ADNI DUA restricts this; pass allow_adni=True "
            "(--allow-adni-to-client) only for a client that runs locally."
        )
    registry = build_registry(kind=kind, root=root)

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(name=t.name, description=t.description, input_schema=t.parameters)
                for t in registry.values()
            ]
        )

    async def call_tool(ctx, params) -> types.CallToolResult:
        tool = registry.get(params.name)
        if tool is None:
            text, error = json.dumps({"error": f"unknown tool {params.name!r}"}), True
        else:
            try:
                text, error = (
                    json.dumps(asdict(tool.fn(**(params.arguments or {}))), default=str),
                    False,
                )
            except (TypeError, ValueError, KeyError, FileNotFoundError) as exc:
                text, error = json.dumps({"error": f"{type(exc).__name__}: {exc}"}), True
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)], is_error=error
        )

    return Server("dementia-ml", on_list_tools=list_tools, on_call_tool=call_tool)


async def _serve(kind: str, allow_adni: bool) -> None:
    server = build_server(kind, allow_adni=allow_adni)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    ap = argparse.ArgumentParser(description="MCP server for the dementia-ml analysis tools")
    ap.add_argument("--kind", choices=["synthetic", "adni"], default="synthetic")
    ap.add_argument(
        "--allow-adni-to-client",
        action="store_true",
        help="serve ADNI-derived aggregates; only for an MCP client backed by a local model",
    )
    args = ap.parse_args()
    if args.allow_adni_to_client:
        print("warning: ADNI aggregates will reach the MCP client's model", file=sys.stderr)
    asyncio.run(_serve(args.kind, args.allow_adni_to_client))


if __name__ == "__main__":
    main()
