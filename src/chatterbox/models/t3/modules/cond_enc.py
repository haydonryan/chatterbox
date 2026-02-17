"""Python T3 conditioning encoder is disabled; Rust implementation is used instead."""


class T3Cond:
    def __init__(self, *_, **__):
        raise RuntimeError("Python T3 conditionals are disabled; use the Rust backend.")


class T3CondEnc:
    def __init__(self, *_, **__):
        raise RuntimeError("Python T3 conditioning encoder is disabled; use the Rust backend.")

    def forward(self, *_, **__):
        raise RuntimeError("Python T3 conditioning encoder is disabled; use the Rust backend.")
