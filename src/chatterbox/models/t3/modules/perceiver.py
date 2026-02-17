"""Python T3 perceiver is disabled; Rust implementation is used instead."""


class Perceiver:
    def __init__(self, *_, **__):
        raise RuntimeError("Python T3 perceiver is disabled; use the Rust backend.")

    def forward(self, *_, **__):
        raise RuntimeError("Python T3 perceiver is disabled; use the Rust backend.")
