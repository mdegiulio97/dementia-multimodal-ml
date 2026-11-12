"""Ask the local analysis agent a question.

uv run dementia-agent "What is the 3-year risk for a 74-year-old with MMSE 27 and one APOE4 allele?"
"""

from __future__ import annotations

import argparse
import json
import sys

from dementia_ml.agent.backends import OllamaBackend
from dementia_ml.agent.core import Agent
from dementia_ml.agent.tools import build_registry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Local tool-using agent over this project's results.")
    ap.add_argument("question")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--kind", choices=["synthetic", "adni"], default="synthetic")
    ap.add_argument("--show-calls", action="store_true", help="print tool calls and provenance")
    args = ap.parse_args(argv)
    backend = OllamaBackend(model=args.model)
    if not backend.available():
        print(
            "Ollama is not running. Install it from https://ollama.com and run "
            f"`ollama pull {args.model}`.",
            file=sys.stderr,
        )
        return 1
    ans = Agent(backend, build_registry(kind=args.kind)).ask(args.question)
    print(ans.text)
    if args.show_calls:
        print(json.dumps(ans.tool_calls, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
