"""Minimal prefill/decode proxy (extension 3.5(d)).

Each request goes to the prefill server first with max_tokens=1, then the original request goes
to the decode server, whose (streamed) answer is returned.
  --mode nixl     NixlConnector: the prefill response carries kv_transfer_params (block ids to pull),
                  forwarded to the decode server.
  --mode storage  ExampleConnector: the prefill server writes the KV pages to shared storage keyed by
                  the prompt, the decode server finds them there; no kv_transfer_params needed.
"""
import argparse
import copy
import itertools

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
ARGS = None
_ids = itertools.count()


async def _forward(path, req_json, request_id):
    session = app.state.session
    pre = copy.deepcopy(req_json)
    pre.update(max_tokens=1, stream=False)
    if ARGS.mode == "nixl":
        pre["kv_transfer_params"] = {"do_remote_decode": True, "do_remote_prefill": False, "remote_engine_id": None,
                                     "remote_block_ids": None, "remote_host": None, "remote_port": None}
    pre.pop("max_completion_tokens", None)
    pre.pop("stream_options", None)
    headers = {"X-Request-Id": request_id}
    async with session.post(f"{ARGS.prefill}{path}", json=pre, headers=headers) as r:
        r.raise_for_status()
        kv = (await r.json()).get("kv_transfer_params")
    dec = dict(req_json)
    if ARGS.mode == "nixl":
        dec["kv_transfer_params"] = kv
    return session.post(f"{ARGS.decode}{path}", json=dec, headers=headers)


async def handle(request: Request, path: str):
    req_json = await request.json()
    rid = f"pd-{next(_ids)}"
    resp_ctx = await _forward(path, req_json, rid)
    if req_json.get("stream"):
        async def gen():
            async with resp_ctx as r:
                async for chunk in r.content.iter_any():
                    yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")
    async with resp_ctx as r:
        return JSONResponse(await r.json(), status_code=r.status)


@app.post("/v1/completions")
async def completions(request: Request):
    return await handle(request, "/v1/completions")


@app.post("/v1/chat/completions")
async def chat(request: Request):
    return await handle(request, "/v1/chat/completions")


@app.get("/v1/models")
async def models():
    async with app.state.session.get(f"{ARGS.decode}/v1/models") as r:
        return JSONResponse(await r.json())


@app.on_event("startup")
async def startup():
    app.state.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600),
                                              connector=aiohttp.TCPConnector(limit=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", default="http://localhost:8100")
    ap.add_argument("--decode", default="http://localhost:8200")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mode", choices=["nixl", "storage"], default="storage")
    ARGS = ap.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=ARGS.port, log_level="warning")
