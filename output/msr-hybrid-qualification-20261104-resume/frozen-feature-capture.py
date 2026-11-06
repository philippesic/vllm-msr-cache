"""Causal recorded-completion feature pilot, separate from serving qualification.

The local-model re-encoding and the returning-only label population are explicit
limitations. A positive result here alone cannot authorize a live hybrid.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time


def clean_messages(messages):
    return [{k: v for k, v in m.items() if v is not None and v != []} for m in messages]


def task_group(session):
    # Keep different model/run variants of the same benchmark task together.
    fields = session.split("__")
    if fields[0] == "swebench":
        return "__".join(fields[:3])
    return "__".join(fields[:2])


def split_for(session):
    value = int(hashlib.sha256(("hybrid-pilot-v1:" + task_group(session)).encode()).hexdigest()[:8], 16) % 10
    return "train" if value < 6 else "validation" if value < 8 else "test"


def completed_example(previous, following, turn):
    """Future container supplies only the previous already-completed response.

    The next user/tool input and its length never become predictor features.
    """
    before, after = clean_messages(previous["input"]), clean_messages(following["input"])
    if previous["session_id"] != following["session_id"] or previous["model"] != following["model"]:
        return None
    if after[:len(before)] != before:
        return None
    output = []
    for message in after[len(before):]:
        if message["role"] != "assistant":
            break
        output.append(message)
    if not output:
        return None
    gap = following["pre_gap"]
    if not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap < 0:
        return None
    prefix = before + output
    tool_calls = sum(len(m.get("tool_calls", [])) for m in output)
    return dict(
        session_id=previous["session_id"], group=task_group(previous["session_id"]),
        split=split_for(previous["session_id"]), source_model=previous["model"],
        messages=prefix,
        cheap=[turn, len(before), len(json.dumps(before)), len(json.dumps(output)),
               previous["output_length"], previous["pre_gap"], tool_calls,
               sum(m["role"] == "tool" for m in before)],
        next_gap_s=gap, target=int(gap <= 1.0),
    )


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    counts = Counter()
    previous = None
    turn = 0
    chosen = set()
    excluded = set()
    args.output.mkdir(exist_ok=False)
    with (args.output / "examples.jsonl").open("x") as stream:
        for path in args.sources:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=1, use_threads=False):
                row = batch.to_pylist()[0]
                counts["rows"] += 1
                sid = row["session_id"]
                if previous is None or previous["session_id"] != sid:
                    turn = 0
                    if row["pre_gap"] != 0:
                        excluded.add(sid)  # shard begins mid-session
                else:
                    turn += 1
                    if sid not in chosen and sid not in excluded and turn >= 3:
                        example = completed_example(previous, row, turn)
                        if example is None:
                            counts["invalid_continuation"] += 1
                        else:
                            # No truncation. Freeze geometry and choice without
                            # using next_gap or the target to select examples.
                            tokens = tokenizer.apply_chat_template(example.pop("messages"), tokenize=True,
                                                                   return_dict=False, add_generation_prompt=False)
                            if len(tokens) > args.max_tokens:
                                counts["oversize"] += 1
                                excluded.add(sid)
                            else:
                                example["tokens"] = tokens
                                example["cheap"].append(len(tokens))
                                stream.write(json.dumps(example) + "\n")
                                chosen.add(sid)
                                counts["selected"] += 1
                previous = row
    manifest = dict(
        schema="hybrid-feature-pilot-v1", model=str(args.model), max_tokens=args.max_tokens,
        selection="One first valid completed-prefix decision at/after turn 3 per complete-start session; no truncation; label-blind token fit",
        primary_target="Observed next request within 1.0 second after response completion",
        split="SHA256 of benchmark task, 60/20/20; model/run variants grouped",
        hypotheses=["local_last_query_mean_key_scalar", "local_block_score_distribution", "local_query_key_representation"],
        counts=dict(counts),
        limitations=[
            "Only recorded continuing sessions are labelled. Terminal decisions are censored, not abandonment negatives.",
            "Local-model teacher-forced re-encoding differs from native generating-model scoring and measures no vLLM serving overhead.",
            "External tool schemas absent from the recorded input cannot be reconstructed; this is an input-message representation hypothesis, not an exact native score reproduction.",
            "No absolute inter-session arrivals, shared-cache competition or victim costs are supplied by pre_gap.",
            "A predictive improvement alone is not a cache-value qualification or permission to build the live hybrid.",
        ],
    )
    raw = (args.output / "examples.jsonl").read_bytes()
    manifest["examples_sha256"] = hashlib.sha256(raw).hexdigest()
    manifest["capture_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["sources"] = []
    for path in args.sources:
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        manifest["sources"].append(dict(path=str(path), sha256=digest))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(counts=counts, examples_sha256=manifest["examples_sha256"])))


def capture(args):
    import subprocess
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    from harness.gpu_lock import GPULock

    manifest = json.loads((args.inputs / "manifest.json").read_text())
    raw = (args.inputs / "examples.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["examples_sha256"]:
        raise ValueError("Prepared inputs changed")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != manifest["capture_source_sha256"]:
        raise ValueError("Capture source changed after preparation")
    examples = [json.loads(line) for line in raw.splitlines()]
    args.output.mkdir(exist_ok=False)
    with GPULock():
        clients = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if clients:
            raise RuntimeError("GPU has existing compute clients")
        model = AutoModelForCausalLM.from_pretrained(manifest["model"], local_files_only=True,
                                                    dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
        layer = model.model.layers[len(model.model.layers) // 2].self_attn
        state = {}

        def pre_hook(module, positional, kwargs):
            state["position_embeddings"] = kwargs["position_embeddings"]

        def query_hook(module, positional, result):
            state["q"] = result

        def key_hook(module, positional, result):
            q = state.pop("q")
            shape = (1, q.shape[1], -1, layer.head_dim)
            q = q.view(shape).transpose(1, 2)
            k = result.view(1, result.shape[1], -1, layer.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *state.pop("position_embeddings"))
            state["last_q"] = q[0, :, -1, :].float().mean(0).cpu()
            # This local hypothesis collapses query/key heads separately. It
            # must not be relabelled the cross-request native MSR policy.
            state["key_chunks"].append(k[0].float().mean(0).cpu())

        handles = [layer.register_forward_pre_hook(pre_hook, with_kwargs=True),
                   layer.q_proj.register_forward_hook(query_hook), layer.k_proj.register_forward_hook(key_hook)]
        generator = torch.Generator().manual_seed(63104)
        projection = torch.randn(layer.head_dim, 16, generator=generator) / math.sqrt(layer.head_dim)
        try:
            with (args.output / "features.jsonl").open("x", buffering=1) as stream, torch.inference_mode():
                for number, example in enumerate(examples):
                    state["key_chunks"] = []
                    tokens = torch.tensor([example["tokens"]], device="cuda")
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    cache = None
                    for offset in range(0, tokens.shape[1], 512):
                        result = model.model(input_ids=tokens[:, offset:offset + 512], past_key_values=cache, use_cache=True)
                        cache = result.past_key_values
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    keys = torch.cat(state.pop("key_chunks"))
                    query = state.pop("last_q")
                    complete = keys[:len(keys) // 16 * 16]
                    blocks = complete.reshape(-1, 16, layer.head_dim).mean(1)
                    scores = blocks @ query
                    quantiles = torch.quantile(scores, torch.tensor([0., .25, .5, .75, 1.]))
                    mean_key = keys.mean(0)
                    features = dict(
                        session_id=example["session_id"], split=example["split"],
                        scalar=[float(mean_key @ query)],
                        distribution=[float(x) for x in quantiles] + [float(scores.std()), float(scores.mean())],
                        representation=(query @ projection).tolist() + (mean_key @ projection).tolist(),
                        token_count=len(example["tokens"]), capture_s=elapsed,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    )
                    stream.write(json.dumps(features) + "\n")
                    print(f"captured {number + 1}/{len(examples)} tokens={len(example['tokens'])} seconds={elapsed:.2f}", flush=True)
                    del result, cache, tokens, keys, complete, blocks, scores
        finally:
            for handle in handles:
                handle.remove()
            del model
            torch.cuda.empty_cache()
    (args.output / "capture.json").write_text(json.dumps(dict(
        manifest=manifest, status="complete", count=len(examples),
        metric="Teacher-forced request-local final query against own key-prefix summaries, middle Qwen2 layer, post-RoPE; FP32 reductions",
        overhead="Capture time includes re-encoding, CPU transfers and scoring. Not incremental native vLLM capture overhead.",
    ), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("sources", nargs="+", type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("capture")
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        capture(args)


if __name__ == "__main__":
    main()
