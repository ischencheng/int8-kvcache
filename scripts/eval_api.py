"""GSM8K accuracy through an OpenAI-compatible endpoint (a vLLM server or the PD proxy)."""
import argparse
import asyncio
import json
import os

import aiohttp

from kvq import data, evaluate


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    rows = data.gsm8k(args.n)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(session, r):
        body = {"model": args.model, "messages": data.math_messages(r["question"]), "temperature": 0,
                "max_tokens": 1024}
        async with sem, session.post(f"{args.url}/v1/chat/completions", json=body) as resp:
            text = (await resp.json())["choices"][0]["message"]["content"]
        return {"id": r["id"], "answer": r["answer"], "correct": evaluate.math_correct(text, r["answer"]), "text": text}

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200)) as s:
        items = await asyncio.gather(*[one(s, r) for r in rows])
    res = {"label": args.label, "n": len(items), "accuracy": sum(i["correct"] for i in items) / len(items)}
    print(json.dumps(res))
    out = os.environ["OUT"]
    with open(f"{out}/eval_api_{args.label}.json", "w") as f:
        json.dump(res, f, indent=1)
    with open(f"{out}/eval_api_{args.label}.jsonl", "w") as f:
        for i in items:
            f.write(json.dumps(i) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
