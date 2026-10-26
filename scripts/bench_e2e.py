"""Stage 8: end-to-end decode speed (TPOT) and memory-bound throughput of one configuration.

TPOT: B requests of ctx random tokens each (distinct, prefix caching off), timed twice
with ignore_eos: max_tokens=1 (prefill only) and max_tokens=out_len; the difference over
out_len - 1 steps is the decode time per output token. A point is flagged kv_limited when
B * (ctx + out_len) exceeds the KV capacity (vLLM would queue requests, so TPOT is not
comparable there).

--throughput N: N long requests at once, the regime where KV capacity caps concurrency.
"""
import argparse
import statistics
import json
import os
import random
import time


def _kv_blocks(worker):
    cc = worker.vllm_config.cache_config
    return cc.num_gpu_blocks, cc.block_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--label", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--llm", default="{}")
    ap.add_argument("--ctx", default="1024,4096,16384")
    ap.add_argument("--batch", default="1,16,64,128")
    ap.add_argument("--points", default=None, help='"ctx:b1,b2;ctx2:b3" overrides --ctx/--batch')
    ap.add_argument("--out-len", type=int, default=128)
    ap.add_argument("--throughput", type=int, default=0, help="N requests for the capacity-bound run")
    ap.add_argument("--pure", action="store_true",
                    help="pure-decode TPOT = (T(2L) - T(L)) / L: the extra L steps run after every prefill finished")
    ap.add_argument("--repeat", type=int, default=1, help="pure mode: (T(L), T(2L)) pairs per point, median reported")
    ap.add_argument("--tp-ctx", type=int, default=6144)
    ap.add_argument("--tp-out", type=int, default=1024)
    args = ap.parse_args()
    for kv in args.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kwargs = dict(max_model_len=32768, gpu_memory_utilization=0.9, seed=0, enable_prefix_caching=False,
                  max_num_seqs=256, disable_log_stats=False)  # stats on: per-request token timestamps
    kwargs.update(json.loads(args.llm))
    llm = LLM(args.model, **kwargs)
    # The engine core (another process) owns the final block count; ask the worker.
    blocks, block_size = llm.collective_rpc(_kv_blocks)[0]
    capacity = blocks * block_size
    rng = random.Random(0)

    def prompts(n, ctx):
        return [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 100000) for _ in range(ctx)]) for _ in range(n)]

    def timed(ps, max_tokens):
        t0 = time.perf_counter()
        outs = llm.generate(ps, SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=0),
                            use_tqdm=False)
        return time.perf_counter() - t0, outs

    llm.generate(prompts(4, 256), SamplingParams(max_tokens=8), use_tqdm=False)  # warm up
    res = {"label": args.label, "env": args.env, "llm": kwargs, "kv_capacity_tokens": capacity, "tpot": []}
    if args.throughput:
        ps = prompts(args.throughput, args.tp_ctx)
        wall, outs = timed(ps, args.tp_out)
        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        res["throughput"] = {"requests": args.throughput, "ctx": args.tp_ctx, "out": args.tp_out, "wall_s": wall,
                             "output_tok_per_s": gen / wall,
                             "max_concurrent_by_kv": capacity // (args.tp_ctx + args.tp_out)}
        print(json.dumps(res["throughput"]))
    else:
        if args.points:
            grid = [(int(c), [int(b) for b in bs.split(",")]) for c, bs in
                    (part.split(":") for part in args.points.split(";"))]
        else:
            grid = [(int(c), [int(b) for b in args.batch.split(",")]) for c in args.ctx.split(",")]
        for ctx, batches in grid:
            for b in batches:
                ps = prompts(b, ctx)
                if args.pure:
                    if args.repeat > 1:
                        timed(ps, 2)  # first call at this shape: JIT compiles land here, not in T(L)
                    pairs = [(timed(ps, args.out_len)[0], timed(ps, 2 * args.out_len)[0]) for _ in range(args.repeat)]
                    samples = [1e3 * (t2 - t1) / args.out_len for t1, t2 in pairs]
                    row = {"ctx": ctx, "batch": b, "t_L_s": [p[0] for p in pairs], "t_2L_s": [p[1] for p in pairs],
                           "tpot_pure_ms": statistics.median(samples), "samples_ms": samples,
                           "kv_limited": b * (ctx + 2 * args.out_len) > capacity}
                    res["tpot"].append(row)
                    print(json.dumps(row))
                    continue
                t_pre, _ = timed(ps, 1)
                t_all, outs = timed(ps, args.out_len)
                # Per-request TPOT from engine timestamps (what serving benchmarks report); the
                # (total - prefill) difference also counts steps where prefill chunks share the batch.
                per_req = [(o.metrics.last_token_ts - o.metrics.first_token_ts) / (o.metrics.num_generation_tokens - 1)
                           for o in outs if o.metrics is not None and o.metrics.num_generation_tokens > 1]
                row = {"ctx": ctx, "batch": b, "prefill_s": t_pre, "total_s": t_all,
                       "tpot_ms": 1e3 * (t_all - t_pre) / (args.out_len - 1),
                       "tpot_req_ms": 1e3 * sum(per_req) / len(per_req) if per_req else None,
                       "kv_limited": b * (ctx + args.out_len) > capacity}
                res["tpot"].append(row)
                print(json.dumps(row))
    suffix = "_tput" if args.throughput else ("_pure" if args.pure else "")
    with open(f"{os.environ['OUT']}/e2e_{args.label}{suffix}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
