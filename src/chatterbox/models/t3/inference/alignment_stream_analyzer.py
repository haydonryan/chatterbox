"""Python T3 alignment analyzer is disabled; Rust implementation is used instead."""

LLAMA_ALIGNED_HEADS = []


class AlignmentStreamAnalyzer:
    def __init__(self, *_, **__):
        raise RuntimeError("Python T3 alignment analyzer is disabled; use the Rust backend.")
