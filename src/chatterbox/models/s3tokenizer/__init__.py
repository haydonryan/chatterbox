### S3TOKENIZER_PYTHON_DISABLED
# Original Python implementation commented out to avoid Python runtime usage.
# Rust replacement lives in `src/models/s3gen/tokenizer.rs`.

S3_SR = 16_000
S3_HOP = 160  # 100 frames/sec
S3_TOKEN_HOP = 640  # 25 tokens/sec
S3_TOKEN_RATE = 25
SPEECH_VOCAB_SIZE = 6561

SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1


class S3Tokenizer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Python S3Tokenizer disabled; use Rust implementation.")


def drop_invalid_tokens(_x):
    raise RuntimeError("Python S3Tokenizer disabled; use Rust implementation.")
