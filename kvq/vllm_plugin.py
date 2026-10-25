"""vLLM general plugin entry point (see modal_app.py for how it is installed).

vLLM calls register() in every process it starts, before the model is built,
so this is where we patch or register things that must exist in the workers.
"""
import os


def register():
    # Registering by class path does not import the backend module here.
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    register_backend(AttentionBackendEnum.CUSTOM, "kvq.vllm_int8.backend.Int8StaticBackend")
    if os.environ.get("KVQ_CAPTURE") == "1":
        from kvq import capture

        capture.install()
    if os.environ.get("KVQ_ATTN"):
        from kvq import override

        override.install()
    if os.environ.get("KVQ_MLA"):
        from kvq import mla

        mla.install()
