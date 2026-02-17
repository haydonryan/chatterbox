"""Python T3 is disabled; Rust implementation is used instead."""


class T3:
    def __init__(self, *_, **__):
        raise RuntimeError("Python T3 is disabled; use the Rust T3 backend.")

    def forward(self, *_, **__):
        raise RuntimeError("Python T3 is disabled; use the Rust T3 backend.")

    def inference(self, *_, **__):
        raise RuntimeError("Python T3 is disabled; use the Rust T3 backend.")


def _ensure_BOT_EOT(*_, **__):
    raise RuntimeError("Python T3 is disabled; use the Rust T3 backend.")
