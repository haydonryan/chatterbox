"""Python T3 learned position embeddings are disabled; Rust implementation is used instead."""


class LearnedPositionEmbeddings:
    def __init__(self, *_, **__):
        raise RuntimeError("Python T3 learned position embeddings are disabled.")

    def forward(self, *_, **__):
        raise RuntimeError("Python T3 learned position embeddings are disabled.")
