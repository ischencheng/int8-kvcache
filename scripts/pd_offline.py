"""Extension 3.5(d): prefill/decode disaggregation with two separate vLLM engines.

  --role producer : prefill-only (max_tokens=1) engine, kv_producer; writes each request's KV pages
                    to shared storage (vLLM ExampleConnector).
  --role consumer : decode engine, kv_consumer; finds the stored KV, skips the prompt prefill and
                    decodes. Per-request TTFT (includes loading the KV) and TPOT come from the engine
                    timestamps; all requests are submitted at once (high concurrency).
  --role colocated: one engine does both (PD mixed), same metrics.
Run one role per process (fresh engine each time).
"""
import argparse
import json
import os
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--role", required=True, choices=["producer", "consumer", "colocated"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--store", default="/tmp/kvstore")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--llm", default="{}")
    args = ap.parse_args()
    for kv in args.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    from vllm import LLM, SamplingParams

    from kvq import data, evaluate

    kwargs = dict(max_model_len=4096, gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=False,
                  disable_log_stats=False)
    if args.role != "colocated":
        role = "kv_producer" if args.role == "producer" else "kv_consumer"
        kwargs["kv_transfer_config"] = {"kv_connector": "ExampleConnector", "kv_role": role,
                                        "kv_connector_extra_config": {"shared_storage_path": args.store}}
        # ExampleConnector writes one file per (prompt, layer) from the step that computed it; a prompt split
        # over several prefill chunks keeps only the last chunk's pages. Whole prompts per step avoid that.
        kwargs.update(enable_chunked_prefill=False, max_num_batched_tokens=16384)
    kwargs.update(json.loads(args.llm))
    llm = LLM(args.model, **kwargs)
    rows = data.gsm8k(args.n)
    sp = SamplingParams(temperature=0, max_tokens=1 if args.role == "producer" else 1024)
    t0 = time.perf_counter()
    outs = llm.chat([data.math_messages(r["question"]) for r in rows], sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    res = {"label": args.label, "role": args.role, "n": len(rows), "wall_s": wall}
    if args.role == "producer":
        du = subprocess.run(["du", "-sb", args.store], capture_output=True, text=True).stdout.split()[0]
        ptoks = sum(len(o.prompt_token_ids) for o in outs)
        res.update(store_bytes=int(du), prompt_tokens=ptoks, bytes_per_token=int(du) / ptoks)
    else:
        ttft, tpot = [], []
        for o in outs:
            m = o.metrics
            if m is None:
                continue
            ttft.append(m.first_token_ts - m.queued_ts)
            if m.num_generation_tokens > 1:
                tpot.append((m.last_token_ts - m.first_token_ts) / (m.num_generation_tokens - 1))
        res.update(accuracy=sum(evaluate.math_correct(o.outputs[0].text, r["answer"]) for o, r in zip(outs, rows))
                   / len(rows),
                   mean_ttft_ms=1e3 * sum(ttft) / len(ttft), mean_tpot_ms=1e3 * sum(tpot) / len(tpot),
                   output_tok_per_s=sum(len(o.outputs[0].token_ids) for o in outs) / wall)
    print(json.dumps(res))
    with open(f"{os.environ['OUT']}/pd_{args.label}_{args.role}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
