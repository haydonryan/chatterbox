# Chatterbox TTS Application Report

## Overview

This is a text-to-speech (TTS) application that converts text into natural-sounding speech audio. The system uses a two-stage generation pipeline:
1. **Text → Speech Tokens** (T3 model)
2. **Speech Tokens → Audio Waveform** (S3Gen model)

---

## Entry Point: `example_tts.py`

The example script demonstrates basic usage:
- Detects the best available compute device (CUDA GPU, Apple MPS, or CPU)
- Loads the pretrained ChatterboxTTS model
- Generates speech from text and saves it as a WAV file

---

## Top-Level Modules

### `src/chatterbox/__init__.py`
**Purpose**: Package entry point that exports the public API:
- `ChatterboxTTS` - Main English TTS class
- `ChatterboxMultilingualTTS` - Multi-language TTS (23 languages)
- `ChatterboxVC` - Voice conversion class
- `SUPPORTED_LANGUAGES` - Dictionary of supported languages

---

### `src/chatterbox/tts.py`
**Purpose**: Core English TTS implementation

**Classes**:
- `Conditionals` - Dataclass holding conditioning information (speaker embeddings, speech tokens, emotion settings)
- `ChatterboxTTS` - Main TTS engine

**Key Methods**:
- `from_pretrained()` - Downloads and loads models from HuggingFace Hub
- `from_local()` - Loads models from a local checkpoint directory
- `prepare_conditionals()` - Extracts speaker embeddings and voice characteristics from a reference audio file
- `generate()` - Converts text to speech audio, with optional voice cloning via an audio prompt

**Workflow**: Tokenizes input text → generates speech tokens via T3 → converts tokens to waveform via S3Gen

---

### `src/chatterbox/tts_turbo.py`
**Purpose**: Faster TTS variant optimized for speed

**Class**: `ChatterboxTurboTTS`

**Differences from standard TTS**:
- Uses GPT2 backbone instead of Llama (smaller, faster)
- Uses `T3Config(text_tokens_dict_size=50276)` for larger vocabulary
- Includes loudness normalization on output
- Enables "meanflow" mode in S3Gen for faster inference

---

### `src/chatterbox/mtl_tts.py`
**Purpose**: Multilingual TTS supporting 23 languages

**Class**: `ChatterboxMultilingualTTS`

**Supported Languages**: Arabic, Danish, German, Greek, English, Spanish, Finnish, French, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch, Norwegian, Polish, Portuguese, Russian, Swedish, Swahili, Turkish, Chinese

**Key Method**: `generate(text, language_id=...)` - Generates speech in the specified language

Uses `MTLTokenizer` for language-specific text processing and `T3Config.multilingual()` configuration.

---

### `src/chatterbox/vc.py`
**Purpose**: Voice conversion (timbre transfer)

**Class**: `ChatterboxVC`

**Key Methods**:
- `set_target_voice()` - Sets the target speaker voice from a reference audio file
- `convert()` - Converts speech tokens to the target voice

**Use Case**: Change the voice characteristics of speech while preserving content

---

## Models Subpackage

### `src/chatterbox/models/utils.py`
**Purpose**: Utility classes

**Content**: `AttrDict` - A dictionary subclass that allows attribute-style access (`dict.key` instead of `dict['key']`)

---

## T3 Module (Text-to-Speech Token Generation)

**Location**: `src/chatterbox/models/t3/`

### `t3.py`
**Purpose**: Core T3 model - converts text tokens to speech tokens

**Class**: `T3(nn.Module)`

**Architecture**:
- Transformer backbone (Llama or GPT2)
- Custom text and speech token embeddings
- Learned positional embeddings
- Conditioning encoder for speaker/emotion information
- Separate output heads for predicting next text and speech tokens

**Key Methods**:
- `prepare_conditioning()` - Embeds speaker and emotion conditioning
- `inference()` - Autoregressively generates speech tokens from text
- `inference_turbo()` - Optimized inference for turbo variant

---

### `modules/t3_config.py`
**Purpose**: Model configuration

**Class**: `T3Config`

**Key Parameters**:
- `text_tokens_dict_size` - Text vocabulary size (704 English, 2454 multilingual)
- `speech_tokens_dict_size` - Speech vocabulary size (8194)
- `llama_config_name` - Which backbone to use ("Llama_520M", "GPT2_medium", etc.)
- `speech_cond_prompt_len` - Length of conditioning prompt
- `emotion_adv` - Whether emotion control is enabled

**Factory Methods**:
- `english_only()` - Creates English-only configuration
- `multilingual()` - Creates multilingual configuration

---

### `modules/cond_enc.py`
**Purpose**: Conditioning encoder

**Classes**:
- `T3Cond` - Dataclass holding conditioning tensors (speaker_emb, clap_emb, speech_prompt_tokens, emotion_adv)
- `T3CondEnc` - Neural network that projects conditioning information into the model's embedding space

---

### `modules/perceiver.py`
**Purpose**: Perceiver resampler for conditioning

**Classes**:
- `Perceiver` - Resampler module using cross-attention
- `RelativePositionBias` - Relative position encoding
- `AttentionQKV` - Multi-head attention implementation

**Use**: Compresses variable-length conditioning into fixed-length representation

---

### `modules/learned_pos_emb.py`
**Purpose**: Learned positional embeddings

**Class**: `LearnedPositionEmbeddings`

Provides position information to the transformer without using rotary embeddings.

---

### `inference/t3_hf_backend.py`
**Purpose**: HuggingFace-compatible wrapper

**Class**: `T3HuggingfaceBackend`

Wraps T3 to be compatible with HuggingFace's `generate()` API for flexible decoding strategies.

---

### `inference/alignment_stream_analyzer.py`
**Purpose**: Streaming alignment analysis

Analyzes token-to-duration alignment for streaming TTS inference.

---

### `llama_configs.py`
**Purpose**: Predefined transformer backbone configurations

**Configurations**:
- `LLAMA_520M_CONFIG_DICT` - 520M parameter Llama
- `GPT2_MEDIUM_CONFIG` - GPT2 medium (1024 hidden, 24 layers)
- Additional Llama variants (1B, 3B)

---

## S3Gen Module (Speech Token to Waveform)

**Location**: `src/chatterbox/models/s3gen/`

### `s3gen.py`
**Purpose**: Core waveform synthesis model

**Class**: `S3Token2Wav` (aliased as `S3Gen`)

**Pipeline**:
1. S3Tokenizer embeds speech tokens
2. CAMPPlus extracts speaker embedding from reference
3. UpsampleConformerEncoder upsamples token sequence
4. ConditionalDecoder estimates flow velocity
5. CausalConditionalCFM performs flow-matching diffusion
6. HiFiGAN vocoder converts mel-spectrogram to waveform

**Key Methods**:
- `embed_ref()` - Extracts speaker embedding from reference audio
- `inference()` - Converts speech tokens to audio waveform

---

### `const.py`
**Purpose**: Constants

- `S3GEN_SR = 24000` - Output sample rate (24kHz)
- `S3GEN_SIL = 4299` - Silence token ID

---

### `configs.py`
**Purpose**: Flow matching configuration

**Content**: `CFM_PARAMS` - Parameters for conditional flow matching (step sizes, ODE solver settings)

---

### `flow.py`
**Purpose**: Flow/diffusion model

**Class**: `CausalMaskedDiffWithXvec`

Implements diffusion-based mel-spectrogram generation with speaker conditioning.

---

### `flow_matching.py`
**Purpose**: Conditional Flow Matching (CFM)

**Class**: `CausalConditionalCFM`

Modern alternative to diffusion that learns to transform noise into mel-spectrograms through continuous normalizing flows.

---

### `decoder.py`
**Purpose**: U-Net style decoder

**Class**: `ConditionalDecoder`

Estimates the velocity field for flow matching, conditioned on speaker and token information.

---

### `xvector.py`
**Purpose**: Speaker embedding extraction

**Class**: `CAMPPlus`

Advanced speaker embedding model using multi-layer aggregation and channel attention. Extracts speaker identity from audio.

---

### `f0_predictor.py`
**Purpose**: Pitch (F0) prediction

**Class**: `ConvRNNF0Predictor`

Predicts fundamental frequency contour for prosody control using convolutional and recurrent layers.

---

### `hifigan.py`
**Purpose**: Neural vocoder

**Classes**:
- `Snake` - Periodic activation function
- `HiFTGenerator` / `HiFiGANGenerator` - Waveform generators

Converts mel-spectrograms to raw audio waveforms using upsampling and residual convolutions.

---

### Transformer Components (`transformer/`)

| File | Purpose |
|------|---------|
| `attention.py` | Multi-head self-attention |
| `activation.py` | Swish, Mish activation functions |
| `convolution.py` | 1D convolution blocks |
| `embedding.py` | Token embeddings with positional encoding |
| `encoder_layer.py` | Conformer encoder layer |
| `positionwise_feed_forward.py` | Feed-forward network |
| `subsampling.py` | Sequence length reduction |
| `upsample_encoder.py` | Upsampling Conformer encoder |

---

### Utilities (`utils/`)

| File | Purpose |
|------|---------|
| `mel.py` | Mel-spectrogram extraction |
| `mask.py` | Attention mask creation |
| `intmeanflow.py` | Integer mean flow utilities |
| `class_utils.py` | Class helper functions |

---

### Matcha Components (`matcha/`)

Alternative flow matching implementation (Matcha-TTS style):

| File | Purpose |
|------|---------|
| `decoder.py` | Matcha decoder |
| `flow_matching.py` | Matcha CFM implementation |
| `transformer.py` | Matcha transformer blocks |
| `text_encoder.py` | Text encoder with ConvReluNorm |

---

## S3Tokenizer Module

**Location**: `src/chatterbox/models/s3tokenizer/`

### `s3tokenizer.py`
**Purpose**: Speech tokenization wrapper

**Class**: `S3Tokenizer` (extends external S3TokenizerV2)

**Constants**:
- `S3_SR = 16000` - Input sample rate
- `S3_HOP = 160` - Frame hop (100 frames/sec)
- `S3_TOKEN_HOP = 640` - Token hop (25 tokens/sec)
- `SPEECH_VOCAB_SIZE = 6561` - Number of discrete speech tokens

**Key Methods**:
- `forward()` - Tokenizes audio into discrete tokens
- `decode()` - Reconstructs audio from tokens
- `pad()` - Pads audio to token boundaries

---

## Tokenizers Module

**Location**: `src/chatterbox/models/tokenizers/`

### `tokenizer.py`
**Purpose**: Text tokenization

**Classes**:
- `EnTokenizer` - English tokenizer using BPE
- `MTLTokenizer` - Multilingual tokenizer with language-specific preprocessing

**Special Tokens**: `[START]`, `[STOP]`, `[UNK]`, `[SPACE]`, `[PAD]`, `[SEP]`, `[CLS]`, `[MASK]`

**Language-Specific Processing**:
- Japanese: Kanji-to-hiragana conversion
- Russian: Stress marking
- Arabic: Diacritics addition
- Chinese: Cangjie romanization support

**Key Methods**:
- `text_to_tokens()` - Converts text string to token tensor
- `encode()` - Encodes with language-specific preprocessing
- `decode()` - Converts tokens back to text

---

## Voice Encoder Module

**Location**: `src/chatterbox/models/voice_encoder/`

### `voice_encoder.py`
**Purpose**: Speaker embedding extraction

**Class**: `VoiceEncoder`

LSTM-based model that processes mel-spectrogram frames to produce a 256-dimensional speaker embedding vector.

**Key Methods**:
- `embeds_from_wavs()` - Batch extract embeddings from audio files
- `embed_utterance()` - Get single speaker embedding

---

### `config.py`
**Purpose**: Voice encoder configuration

**Class**: `VoiceEncConfig`

**Parameters**:
- `num_mels = 40` - Mel bins
- `sample_rate = 16000` - Input sample rate
- `speaker_embed_size = 256` - Output embedding dimension

---

### `melspec.py`
**Purpose**: Mel-spectrogram computation

**Function**: `melspectrogram()` - Computes mel-spectrograms from audio waveforms

---

## Data Flow Summary

```
Text Input
    │
    ▼
EnTokenizer/MTLTokenizer → Text Tokens
    │
    ▼
T3 Model (with VoiceEncoder conditioning) → Speech Tokens
    │
    ▼
S3Gen Model (with CAMPPlus speaker embedding) → Mel-Spectrogram
    │
    ▼
HiFiGAN Vocoder → Audio Waveform
    │
    ▼
Output WAV File
```

---

## File Statistics

| Category | Count |
|----------|-------|
| Total Python files | 51 |
| Top-level modules | 4 |
| T3 submodules | 8 |
| S3Gen submodules | 21 |
| S3Tokenizer | 2 |
| Tokenizers | 2 |
| VoiceEncoder | 3 |
