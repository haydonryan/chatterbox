"""Python T3 config is disabled; Rust implementation is used instead."""


class T3Config:
    def __init__(self, *_, **__):
        raise RuntimeError("Python T3 config is disabled; use the Rust backend.")

    @staticmethod
    def english_only(*_, **__):
        raise RuntimeError("Python T3 config is disabled; use the Rust backend.")
