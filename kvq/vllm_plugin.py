"""vLLM general plugin entry point (see modal_app.py for how it is installed).

vLLM calls register() in every process it starts, before the model is built,
so this is where we patch or register things that must exist in the workers.
"""
import os


def register():
    if os.environ.get("KVQ_CAPTURE") == "1":
        from kvq import capture

        capture.install()
    if os.environ.get("KVQ_ATTN"):
        from kvq import override

        override.install()
