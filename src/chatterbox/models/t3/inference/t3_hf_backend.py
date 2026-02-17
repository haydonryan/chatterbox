"""Python T3 HF backend is disabled; Rust implementation is used instead."""


class T3HuggingfaceBackend:
    def __init__(self, *_, **__):
        raise RuntimeError("Python T3 HF backend is disabled; use the Rust backend.")


class T3ForwardOutput:
    def __init__(self, *_, **__):
        raise RuntimeError("Python T3 HF backend is disabled; use the Rust backend.")
