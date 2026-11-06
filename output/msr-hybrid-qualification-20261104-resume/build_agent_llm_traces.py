"""Build Exgentic/agent-llm-traces-v2 directly from Exgentic/traces-v2.

Single-script pipeline — this is the *only* derivation step between the two
published repos and lives bundled with `agent-llm-traces-v2` so the derivation
is fully reproducible.

Input  : Exgentic/traces-v2 shards (full corpus, already typed + normalized)
Output : Exgentic/agent-llm-traces-v2 shards (chat-only, v1 minimal schema)

What this script does (in order, per session):

  1. Drop scaffolding chat spans:
       - `exgentic.actor.role == "user_simulator"`  (tau2 customer role-play)
       - `exgentic.actor.role == "judge"`           (BrowseCompPlus correctness judge)

  2. Collapse failed-retry groups (matches v1's
     `filter_failed_retries_and_adjust_timestamps`): a run of consecutive
     failed chat spans (status.code == 2) sharing identical
     `gen_ai.input.messages`, terminated by a successful chat span with the
     same input. Failed spans are dropped; subsequent spans' timestamps
     shift back by the wasted retry duration.

  3. Keep only chat spans with non-empty `gen_ai.input.messages`. This
     drops invoke_agent / execute_tool spans and the residual stragglers.

  4. Drop sessions where no spans survive.

  5. Project each span's `attributes` struct to v1's 10 semantic-convention
     keys + 8 populated gen_ai.* extras (drops all `exgentic.*` provenance):

       v1 keys (matching the published Exgentic/agent-llm-traces schema):
         gen_ai.operation.name           string
         gen_ai.request.model            string  (canonical, no routing prefix)
         gen_ai.response.model           string  (canonical, no routing prefix)
         gen_ai.usage.input_tokens       int64
         gen_ai.usage.output_tokens      int64
         gen_ai.response.id              string
         gen_ai.response.finish_reasons  list<string>
         gen_ai.input.messages           string  (JSON)
         gen_ai.output.messages          string  (JSON)
         gen_ai.tool.definitions         string  (JSON)

       Lossless extras (replay-relevant gen_ai params, not in v1):
         gen_ai.request.temperature      float64
         gen_ai.request.max_tokens       int64
         gen_ai.request.stop_sequences   list<string>
         gen_ai.system_instructions      string
         gen_ai.output.type              string
         gen_ai.provider.name            string
         gen_ai.conversation.id          string
         error.type                      string

  6. Recompute per-session `models` / `total_tokens` / `max_tokens` from
     surviving spans.

The input is already typed + model-normalized (fix_traces_v2 took care of
that). Values arrive natively — this script never re-parses or casts.

Usage:
    python build_agent_llm_traces.py <traces-v2-dir> <output-dir>
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SIMULATOR_ROLE = "user_simulator"
JUDGE_ROLE = "judge"

V1_KEYS = [
    "gen_ai.operation.name",
    "gen_ai.request.model",
    "gen_ai.response.model",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.response.id",
    "gen_ai.response.finish_reasons",
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.tool.definitions",
]
GEN_AI_EXTRAS = [
    "gen_ai.request.temperature",
    "gen_ai.request.max_tokens",
    "gen_ai.request.stop_sequences",
    "gen_ai.system_instructions",
    "gen_ai.output.type",
    "gen_ai.provider.name",
    "gen_ai.conversation.id",
    "error.type",
]
KEEP = V1_KEYS + GEN_AI_EXTRAS
KEEP_SET = set(KEEP)


# ───────────────────────────── filter helpers ─────────────────────────────

def _attrs(span: dict) -> dict:
    a = span.get("attributes")
    return a if isinstance(a, dict) else {}


def _is_dropped_actor(span: dict) -> bool:
    a = _attrs(span)
    if a.get("gen_ai.operation.name") != "chat": return False
    return a.get("exgentic.actor.role") in (SIMULATOR_ROLE, JUDGE_ROLE)


def _is_chat(span: dict) -> bool:
    return _attrs(span).get("gen_ai.operation.name") == "chat"


def _is_failed(span: dict) -> bool:
    status = span.get("status")
    return isinstance(status, dict) and status.get("code") == 2


def _parse_ts(s: str | None) -> datetime | None:
    if not s: return None
    try: return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError): return None


def filter_failed_retries(spans: list[dict]) -> list[dict]:
    """Drop consecutive failed chat spans with identical input followed by
    a successful one; subtract the wasted duration from every subsequent
    span (cumulative across multiple groups). Mirrors v1's algorithm."""
    groups = []
    i = 0
    while i < len(spans):
        s = spans[i]
        if _is_chat(s) and _is_failed(s):
            input_msg = _attrs(s).get("gen_ai.input.messages") or ""
            group_start = i
            j = i + 1
            while j < len(spans):
                nxt = spans[j]
                if not _is_chat(nxt): break
                if (_attrs(nxt).get("gen_ai.input.messages") or "") != input_msg: break
                if _is_failed(nxt):
                    j += 1; continue
                if j > group_start:
                    groups.append({"start": group_start, "end": j - 1, "success": j})
                break
            i = j if j > i else i + 1
        else:
            i += 1
    if not groups: return spans

    drop = set()
    offset = {}
    for g in groups:
        for k in range(g["start"], g["end"] + 1): drop.add(k)
        first = _parse_ts(spans[g["start"]].get("start_time"))
        succ = _parse_ts(spans[g["success"]].get("start_time"))
        if first is None or succ is None: continue
        secs = (succ - first).total_seconds()
        for k in range(g["success"], len(spans)):
            if k in drop: continue
            offset[k] = offset.get(k, 0.0) + secs

    out = []
    for i, s in enumerate(spans):
        if i in drop: continue
        if i in offset:
            new = dict(s)
            st, en = _parse_ts(s.get("start_time")), _parse_ts(s.get("end_time"))
            if st: new["start_time"] = (st - timedelta(seconds=offset[i])).isoformat()
            if en: new["end_time"]   = (en - timedelta(seconds=offset[i])).isoformat()
            out.append(new)
        else:
            out.append(s)
    return out


def filter_spans(spans: list[dict]) -> list[dict]:
    spans = [s for s in spans if not _is_dropped_actor(s)]
    spans = filter_failed_retries(spans)
    spans = [s for s in spans
             if _attrs(s).get("gen_ai.operation.name") == "chat"
             and _attrs(s).get("gen_ai.input.messages")]
    return spans


# ────────────────────────── projection helpers ──────────────────────────

def _project_attrs(attrs: dict) -> dict:
    """Keep only the 18 chosen keys. Values are already natively typed by
    fix_traces_v2 — no cast needed."""
    return {k: attrs.get(k) for k in KEEP}


def _project_row(spans: list[dict]) -> list[dict]:
    return [{**s, "attributes": _project_attrs(_attrs(s))} for s in spans]


def _build_projected_span_type(orig_struct: pa.StructType) -> pa.StructType:
    orig_attr = orig_struct.field("attributes").type
    new_attr_fields = [f for f in orig_attr if f.name in KEEP_SET]
    new_fields = [
        pa.field("attributes", pa.struct(new_attr_fields)) if f.name == "attributes" else f
        for f in orig_struct
    ]
    return pa.struct(new_fields)


# ────────────────────────── aggregates ──────────────────────────

def _recompute_aggregates(spans: list[dict]) -> tuple[list[str], int, int]:
    models = set()
    total = 0
    mx = 0
    for s in spans:
        a = _attrs(s)
        m = a.get("gen_ai.request.model") or a.get("gen_ai.response.model")
        if m: models.add(m)
        it = a.get("gen_ai.usage.input_tokens") or 0
        ot = a.get("gen_ai.usage.output_tokens") or 0
        if isinstance(it, str):
            try: it = int(it)
            except ValueError: it = 0
        if isinstance(ot, str):
            try: ot = int(ot)
            except ValueError: ot = 0
        total += it + ot
        if it + ot > mx: mx = it + ot
    return sorted(models), total, mx


# ────────────────────────── main per-shard work ──────────────────────────

def process_shard(src: Path, dst: Path):
    t = pq.read_table(src)
    schema = t.schema
    orig_struct = schema.field("spans").type.value_type
    new_span_struct = _build_projected_span_type(orig_struct)
    new_spans_type = pa.list_(new_span_struct)

    cols = {n: t.column(n).to_pylist() for n in schema.names}
    n_rows = t.num_rows

    keep_idx = []
    n_spans_in = n_spans_out = 0
    for i in range(n_rows):
        spans = cols["spans"][i] or []
        n_spans_in += len(spans)
        filtered = filter_spans(spans)
        if not filtered: continue
        projected = _project_row(filtered)
        n_spans_out += len(projected)
        cols["spans"][i] = projected
        models, total, mx = _recompute_aggregates(projected)
        cols["models"][i] = models
        cols["total_tokens"][i] = total
        cols["max_tokens"][i] = mx
        keep_idx.append(i)

    new_fields = [
        pa.field("spans", new_spans_type) if f.name == "spans" else f for f in schema
    ]
    new_schema = pa.schema(new_fields)

    out_table = pa.table({
        n: pa.array([cols[n][i] for i in keep_idx], type=new_schema.field(n).type)
        for n in schema.names
    })
    pq.write_table(
        out_table, dst,
        compression="zstd", compression_level=12,
        row_group_size=10, data_page_size=4 * 1024 * 1024,
    )
    return {
        "rows_in": n_rows,
        "rows_out": len(keep_idx),
        "spans_in": n_spans_in,
        "spans_out": n_spans_out,
    }


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    src_dir = Path(sys.argv[1])
    dst_dir = Path(sys.argv[2])
    dst_dir.mkdir(parents=True, exist_ok=True)
    totals = {"rows_in": 0, "rows_out": 0, "spans_in": 0, "spans_out": 0}
    for shard in sorted(src_dir.glob("*.parquet")):
        stats = process_shard(shard, dst_dir / shard.name)
        print(f"  {shard.name}: rows {stats['rows_in']}→{stats['rows_out']}  "
              f"spans {stats['spans_in']}→{stats['spans_out']}", flush=True)
        for k in totals: totals[k] += stats[k]
    print(f"\nTOTAL: rows {totals['rows_in']}→{totals['rows_out']}  "
          f"spans {totals['spans_in']}→{totals['spans_out']}")


if __name__ == "__main__":
    main()
