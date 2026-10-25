#!/bin/bash
# Give the mounted /root/repo/kvq package a vLLM plugin entry point without
# reinstalling it after every local edit (see modal_app.py).
set -e
SITE=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo /root/repo > "$SITE/kvq_repo.pth"
D="$SITE/kvq-0.0.0.dist-info"
mkdir -p "$D"
printf 'Metadata-Version: 2.1\nName: kvq\nVersion: 0.0.0\n' > "$D/METADATA"
printf '[vllm.general_plugins]\nkvq = kvq.vllm_plugin:register\n' > "$D/entry_points.txt"
touch "$D/RECORD"
