---
license: cc-by-nc-4.0
language:
  - en
tags:
  - opentelemetry
  - otel
  - agentic
  - multi-turn
  - gen_ai
  - observability
  - llm-tracing
  - swebench
  - gaia
  - tool-use
task_categories:
  - text-generation
pretty_name: "Agentic LLM Traces – OTel Format"
size_categories:
  - 10K<n<100K
---

# Agentic LLM Traces – OTel Format

## Overview

Real-world agentic LLM sessions converted to OpenTelemetry (OTel) trace format, derived from [sammshen/lmcache-agentic-traces](https://huggingface.co/datasets/sammshen/lmcache-agentic-traces). Each session is a multi-turn agent interaction involving tool calls (bash commands, file edits, web search, etc.), spanning 5–50 turns and totalling 24,880 spans.

Traces come from three agentic benchmarks: SWE-bench, GAIA, and WildClaw. They are formatted as OTel spans following `gen_ai.*` semantic conventions, suitable for testing and developing LLM observability and replay tools.

## Source Data

| Benchmark | Sessions | Description |
|-----------|----------|-------------|
| SWE-bench | 665 | Software engineering tasks: bug fixes and feature additions in real GitHub repos |
| GAIA | 94 | General-purpose AI assistant tasks requiring multi-step reasoning and tool use |
| WildClaw | 8 | Open-ended agentic tasks in complex environments |

Original dataset: [sammshen/lmcache-agentic-traces](https://huggingface.co/datasets/sammshen/lmcache-agentic-traces)

## License

Released under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Dataset Structure

### File Format

The dataset is provided as a single JSONL file (`lmcache-agentic-traces_otel.jsonl`) where each line is one complete session (all turns of one agent interaction):

```
lmcache-agentic-traces_otel.jsonl   # 767 lines, one JSON object per session
```

Each line contains a trace with all its spans:

```json
{
  "spans": [ ... ]
}
```

### Span Schema

Each span represents one LLM call — one agent turn with its full message history input and the model's response:

```json
{
  "span_id": "swebench__pydata__xarray-4094__minimax__run2-span-0000",
  "trace_id": "swebench__pydata__xarray-4094__minimax__run2",
  "start_time": "2024-01-01T12:00:00+00:00",
  "end_time": "2024-01-01T12:00:02.660000+00:00",
  "attributes": {
    "gen_ai.operation.name": "chat",
    "gen_ai.request.model": "minimax",
    "gen_ai.input.messages": [ ... ],
    "gen_ai.output.messages": [ ... ],
    "gen_ai.usage.prompt_tokens": 9474,
    "gen_ai.usage.completion_tokens": 133,
    "gen_ai.tool.definitions": [ ... ]
  }
}
```

### Span Attributes

| Attribute | Description |
|-----------|-------------|
| `gen_ai.operation.name` | Operation type — always `"chat"` for these LLM spans |
| `gen_ai.request.model` | Model/agent name extracted from the session ID (e.g., `"minimax"`, `"deepseek"`, `"claude"`) |
| `gen_ai.input.messages` | Full message history up to this turn (system + user + assistant + tool results) |
| `gen_ai.output.messages` | Assistant response for this turn, including any tool calls |
| `gen_ai.usage.prompt_tokens` | Input token count (estimated as chars / 4) |
| `gen_ai.usage.completion_tokens` | Output token count from source data |
| `gen_ai.tool.definitions` | Tool/function definitions available to the agent in this session |

### Message Formats

Input messages follow OpenAI format. Assistant messages with tool calls use OTel `parts` format (role `"assistant"`, with a `parts` array containing `"text"` and `"tool_call"` entries). Tool result messages use `role: "tool"` with a plain string `content`.

### Session ID Format

The `trace_id` field in each span encodes the benchmark, task, agent, and run:

```
swebench__<repo>__<issue>__<agent>[__<run>]
gaia__<level>_<task_id>__<agent>
wildclaw__<task_id>__<agent>
```

## Dataset Statistics

| | |
|---|---|
| Total sessions | 767 |
| Total spans | 24,880 |
| Min spans per session | 5 |
| Max spans per session | 50 |
| Average spans per session | 32.4 |
| Benchmarks | SWE-bench, GAIA, WildClaw |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="lmcache-agentic-traces_otel.jsonl", split="train")

# Each row is one session
session = ds[0]
print(f"{len(session['spans'])} turns")

# Iterate turns
for span in session["spans"]:
    attrs = span["attributes"]
    print(f"{span['span_id']} — model: {attrs['gen_ai.request.model']}, "
          f"output tokens: {attrs['gen_ai.usage.completion_tokens']}")
    for msg in attrs.get("gen_ai.output.messages", []):
        for part in msg.get("parts", []):
            if part["type"] == "tool_call":
                print(f"  Tool call: {part['name']}")
```
