#!/bin/bash
# Bring up an ExampleConnector P/D pair and send one request by hand (debugging the PD setup).
M=Qwen/Qwen2.5-7B-Instruct
STORE=/tmp/kvstore/dbg; mkdir -p $STORE
COMMON="--max-model-len 8192 --gpu-memory-utilization 0.85 --no-enable-prefix-caching"
kvt() { echo "{\"kv_connector\":\"ExampleConnector\",\"kv_role\":\"$1\",\"kv_connector_extra_config\":{\"shared_storage_path\":\"$STORE\"}}"; }
CUDA_VISIBLE_DEVICES=0 vllm serve $M --port 8100 $COMMON --kv-transfer-config "$(kvt kv_producer)" > $OUT/log_p.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 vllm serve $M --port 8200 $COMMON --kv-transfer-config "$(kvt kv_consumer)" > $OUT/log_d.txt 2>&1 &
for p in 8100 8200; do for i in $(seq 1 200); do curl -sf localhost:$p/v1/models >/dev/null && break; sleep 3; done; done
BODY='{"model":"'$M'","messages":[{"role":"user","content":"What is 12*13? Answer briefly."}],"max_tokens":32,"temperature":0}'
echo "== P (max_tokens=1)"; curl -s localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d "${BODY/\"max_tokens\":32/\"max_tokens\":1}"; echo
ls -la $STORE | head; find $STORE | head -5
echo "== D"; curl -s localhost:8200/v1/chat/completions -H 'Content-Type: application/json' -d "$BODY"; echo
python3 scripts/pd_proxy.py --mode storage --port 8000 > $OUT/log_proxy.txt 2>&1 &
sleep 5
BODY2='{"model":"'$M'","messages":[{"role":"user","content":"What is 7*8? Answer briefly."}],"max_tokens":32,"temperature":0}'
echo "== via proxy"; curl -s -i localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d "$BODY2"; echo
echo "== via proxy, stream"; curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d "${BODY2/\"temperature\":0/\"temperature\":0,\"stream\":true}" | head -5; echo
echo "== proxy log"; tail -n 40 $OUT/log_proxy.txt
pkill -f "vllm serve"
