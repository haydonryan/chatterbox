use tch::{Device, IndexOp, Kind, Tensor};

use crate::error::{ChatterboxError, Result};
use crate::models::s3gen::weights::Weights;
use crate::models::t3::T3Config;

pub struct T3CondRust {
    pub speaker_emb: Tensor,
    pub cond_prompt_speech_tokens: Option<Tensor>,
    pub cond_prompt_speech_emb: Option<Tensor>,
    pub emotion_adv: Option<Tensor>,
}

impl T3CondRust {
    fn shallow_clone(&self) -> Self {
        Self {
            speaker_emb: self.speaker_emb.shallow_clone(),
            cond_prompt_speech_tokens: self
                .cond_prompt_speech_tokens
                .as_ref()
                .map(|t| t.shallow_clone()),
            cond_prompt_speech_emb: self
                .cond_prompt_speech_emb
                .as_ref()
                .map(|t| t.shallow_clone()),
            emotion_adv: self.emotion_adv.as_ref().map(|t| t.shallow_clone()),
        }
    }
}

impl Clone for T3CondRust {
    fn clone(&self) -> Self {
        self.shallow_clone()
    }
}

pub struct T3Rust {
    config: T3Config,
    device: Device,
    dtype: Kind,
    text_emb: Embedding,
    speech_emb: Embedding,
    text_pos_emb: Option<PositionEmbedding>,
    speech_pos_emb: Option<PositionEmbedding>,
    cond_enc: T3CondEnc,
    tfmr: LlamaModel,
    speech_head: Linear,
    is_gpt: bool,
}

impl T3Rust {
    pub fn from_safetensors(weights_path: &std::path::Path, device: Device) -> Result<Self> {
        let weights = Weights::load(weights_path, device)?;

        let text_emb_w = weights.get("text_emb.weight")?;
        let speech_emb_w = weights.get("speech_emb.weight")?;
        let hidden = text_emb_w.size()[1];
        let text_emb = Embedding::new(text_emb_w.shallow_clone());
        let speech_emb = Embedding::new(speech_emb_w.shallow_clone());

        let mut config = T3Config::english_only();
        config.text_tokens_dict_size = text_emb_w.size()[0] as u32;
        config.speech_tokens_dict_size = speech_emb_w.size()[0] as u32;

        let is_gpt = false;
        let dtype = text_emb_w.kind();

        let text_pos_emb = weights.get("text_pos_emb.emb.weight").ok().map(PositionEmbedding::new);
        let speech_pos_emb = weights.get("speech_pos_emb.emb.weight").ok().map(PositionEmbedding::new);

        let cond_enc = T3CondEnc::from_weights(&weights, hidden)?;
        let tfmr = LlamaModel::from_weights(&weights, hidden, device)?;

        let speech_head = Linear::from_weights(&weights, "speech_head")?;

        Ok(Self {
            config,
            device,
            dtype,
            text_emb,
            speech_emb,
            text_pos_emb,
            speech_pos_emb,
            cond_enc,
            tfmr,
            speech_head,
            is_gpt,
        })
    }

    pub fn config(&self) -> &T3Config {
        &self.config
    }

    pub fn inference(
        &self,
        text_tokens: &Tensor,
        cond: &T3CondRust,
        options: &crate::models::t3::T3InferenceOptions,
    ) -> Result<Tensor> {
        let mut text_tokens = if text_tokens.dim() == 1 {
            text_tokens.unsqueeze(0)
        } else {
            text_tokens.shallow_clone()
        };
        text_tokens = text_tokens.to_device(self.device).to_kind(Kind::Int64);

        let batch = text_tokens.size()[0];
        let start_speech = self.config.start_speech_token as i64;
        let stop_speech = self.config.stop_speech_token as i64;
        let max_speech = self.config.max_speech_tokens as i64;
        let max_new = options.max_new_tokens.map(|v| v as i64).unwrap_or(max_speech);

        let mut t3_cond = cond.shallow_clone();
        t3_cond.speaker_emb = t3_cond.speaker_emb.to_device(self.device).to_kind(self.dtype);
        if let Some(tokens) = &t3_cond.cond_prompt_speech_tokens {
            t3_cond.cond_prompt_speech_tokens = Some(tokens.to_device(self.device).to_kind(Kind::Int64));
        }
        if let Some(emb) = &t3_cond.cond_prompt_speech_emb {
            t3_cond.cond_prompt_speech_emb = Some(emb.to_device(self.device).to_kind(self.dtype));
        }
        if let Some(emotion) = &t3_cond.emotion_adv {
            t3_cond.emotion_adv = Some(emotion.to_device(self.device).to_kind(self.dtype));
        }

        let initial_speech_tokens = Tensor::full([batch, 1], start_speech, (Kind::Int64, self.device));
        let (embeds, _len_cond) =
            self.prepare_input_embeds(&t3_cond, &text_tokens, &initial_speech_tokens, options.cfg_weight)?;

        let (mut hidden, mut past) = self.tfmr.forward(&embeds, None)?;
        let mut logits = self.speech_head.forward(&hidden);
        let mut logits_step = logits.i((.., -1, ..));

        let mut generated = Tensor::zeros([1, 0], (Kind::Int64, self.device));
        let mut input_ids = Tensor::full([1, 1], start_speech, (Kind::Int64, self.device));

        for step in 0..max_new {
            if options.cfg_weight > 0.0 && batch == 2 {
                let cond = logits_step.i((0..1, ..));
                let uncond = logits_step.i((1..2, ..));
                let cfg = options.cfg_weight as f64;
                logits_step = &cond + (&cond - &uncond) * cfg;
            }

            let mut scores = logits_step.shallow_clone();
            if options.repetition_penalty != 1.0 {
                scores = repetition_penalty(&scores, &input_ids, options.repetition_penalty as f64);
            }
            if (options.temperature - 1.0).abs() > f32::EPSILON && options.temperature > 0.0 {
                scores = &scores / (options.temperature as f64);
            }
            if options.min_p > 0.0 {
                scores = min_p_warp(&scores, options.min_p as f64, 1)?;
            }
            if options.top_p < 1.0 {
                scores = top_p_warp(&scores, options.top_p as f64, 1)?;
            }

            let probs = scores.softmax(-1, Kind::Float);
            let next = probs.multinomial(1, true);

            generated = Tensor::cat(&[generated, next.shallow_clone()], 1);
            input_ids = Tensor::cat(&[input_ids, next.shallow_clone()], 1);

            let next_id = next.int64_value(&[0, 0]);
            if next_id == stop_speech {
                break;
            }

            let mut next_embed = self.speech_emb.forward(&next.to_kind(Kind::Int64));
            if let Some(pos) = &self.speech_pos_emb {
                let fixed = pos.fixed_embedding(step + 1)?;
                next_embed = next_embed + fixed;
            }

            let next_embed = if options.cfg_weight > 0.0 && batch == 2 {
                Tensor::cat(&[next_embed.shallow_clone(), next_embed.shallow_clone()], 0)
            } else {
                next_embed
            };

            let (h, p) = self.tfmr.forward(&next_embed, Some(&mut past))?;
            hidden = h;
            past = p;
            logits = self.speech_head.forward(&hidden);
            logits_step = logits.i((.., -1, ..));
        }

        Ok(generated)
    }

    fn prepare_input_embeds(
        &self,
        cond: &T3CondRust,
        text_tokens: &Tensor,
        speech_tokens: &Tensor,
        cfg_weight: f32,
    ) -> Result<(Tensor, i64)> {
        let mut cond = cond.shallow_clone();
        if cond.cond_prompt_speech_tokens.is_some() && cond.cond_prompt_speech_emb.is_none() {
            let tokens = cond.cond_prompt_speech_tokens.as_ref().unwrap();
            let mut emb = self.speech_emb.forward(tokens);
            if !self.is_gpt {
                if let Some(pos) = &self.speech_pos_emb {
                    emb = emb + pos.forward_tokens(tokens)?;
                }
            }
            cond.cond_prompt_speech_emb = Some(emb);
        }

        let cond_emb = self.cond_enc.forward(&cond)?;
        let mut text_emb = self.text_emb.forward(text_tokens);
        if cfg_weight > 0.0 && !self.is_gpt && text_emb.size()[0] > 1 {
            let _ = text_emb.i(1).zero_();
        }
        if let Some(pos) = &self.text_pos_emb {
            text_emb = text_emb + pos.forward_tokens(text_tokens)?;
        }

        let mut speech_emb = self.speech_emb.forward(speech_tokens);
        if let Some(pos) = &self.speech_pos_emb {
            speech_emb = speech_emb + pos.forward_tokens(speech_tokens)?;
        }

        let len_cond = cond_emb.size()[1];
        let cond_emb = if cond_emb.size()[0] != text_emb.size()[0] {
            cond_emb.expand(&[text_emb.size()[0], -1, -1], true)
        } else {
            cond_emb
        };

        let embeds = Tensor::cat(&[cond_emb, text_emb, speech_emb], 1);
        Ok((embeds, len_cond))
    }
}

struct PositionEmbedding {
    weight: Tensor,
}

impl PositionEmbedding {
    fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    fn forward_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        let seq_len = tokens.size()[1];
        let idx = Tensor::arange(seq_len, (Kind::Int64, tokens.device()));
        let emb = self.weight.to_device(tokens.device()).index_select(0, &idx);
        Ok(emb.unsqueeze(0))
    }

    fn fixed_embedding(&self, idx: i64) -> Result<Tensor> {
        let idx = Tensor::from_slice(&[idx])
            .to_kind(Kind::Int64)
            .to_device(self.weight.device());
        let emb = self.weight.to_device(idx.device()).index_select(0, &idx);
        Ok(emb.unsqueeze(0))
    }
}

struct Embedding {
    weight: Tensor,
}

impl Embedding {
    fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    fn forward(&self, tokens: &Tensor) -> Tensor {
        let tokens = if tokens.dim() == 1 {
            tokens.unsqueeze(0)
        } else {
            tokens.shallow_clone()
        };
        let b = tokens.size()[0];
        let t = tokens.size()[1];
        let flat = tokens.reshape([-1]);
        let w = self.weight.to_device(tokens.device()).to_kind(self.weight.kind());
        let out = w.index_select(0, &flat);
        out.view([b, t, w.size()[1]])
    }
}

struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        let weight = weights.get(&format!("{prefix}.weight"))?;
        let bias = weights.get(&format!("{prefix}.bias")).ok();
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let w = self.weight.to_device(x.device()).to_kind(x.kind());
        let mut y = x.matmul(&w.transpose(0, 1));
        if let Some(b) = &self.bias {
            y = y + b.to_device(x.device()).to_kind(x.kind());
        }
        y
    }
}

struct T3CondEnc {
    spkr_enc: Linear,
    emotion_adv_fc: Option<Linear>,
    perceiver: Option<Perceiver>,
}

impl T3CondEnc {
    fn from_weights(weights: &Weights, hidden: i64) -> Result<Self> {
        let spkr_enc = Linear::from_weights(weights, "cond_enc.spkr_enc")?;
        let emotion_adv_fc = if weights.contains("cond_enc.emotion_adv_fc.weight") {
            Some(Linear::from_weights(weights, "cond_enc.emotion_adv_fc")?)
        } else {
            None
        };
        let perceiver = if weights.contains("cond_enc.perceiver.pre_attention_query") {
            Some(Perceiver::from_weights(weights, hidden)?)
        } else {
            None
        };
        Ok(Self {
            spkr_enc,
            emotion_adv_fc,
            perceiver,
        })
    }

    fn forward(&self, cond: &T3CondRust) -> Result<Tensor> {
        let spk = self
            .spkr_enc
            .forward(&cond.speaker_emb.view([-1, cond.speaker_emb.size()[1]]))
            .unsqueeze(1);
        let empty = Tensor::zeros([spk.size()[0], 0, spk.size()[2]], (spk.kind(), spk.device()));

        let mut cond_prompt = match &cond.cond_prompt_speech_emb {
            Some(x) => x.shallow_clone(),
            None => empty.shallow_clone(),
        };
        if let Some(perceiver) = &self.perceiver {
            if cond_prompt.size()[1] > 0 {
                cond_prompt = perceiver.forward(&cond_prompt)?;
            }
        }

        let mut cond_emotion = empty.shallow_clone();
        if let Some(fc) = &self.emotion_adv_fc {
            let emotion = cond.emotion_adv.as_ref().ok_or_else(|| {
                ChatterboxError::ShapeMismatch("missing emotion_adv".to_string())
            })?;
            cond_emotion = fc.forward(&emotion.view([-1, 1, 1]));
        }

        Ok(Tensor::cat(&[spk, empty, cond_prompt, cond_emotion], 1))
    }
}

struct Perceiver {
    pre_attention_query: Tensor,
    attn: AttentionBlock2,
}

impl Perceiver {
    fn from_weights(weights: &Weights, hidden: i64) -> Result<Self> {
        let pre_attention_query = weights.get("cond_enc.perceiver.pre_attention_query")?;
        let attn = AttentionBlock2::from_weights(weights, "cond_enc.perceiver.attn", hidden)?;
        Ok(Self {
            pre_attention_query,
            attn,
        })
    }

    fn forward(&self, h: &Tensor) -> Result<Tensor> {
        let mut query = self.pre_attention_query.to_device(h.device()).to_kind(h.kind());
        query = query.expand(&[h.size()[0], -1, -1], true);
        let pre_att = self.attn.forward(&query, h, None)?;
        self.attn.forward(&pre_att, &pre_att, None)
    }
}

struct AttentionBlock2 {
    norm_weight: Tensor,
    norm_bias: Tensor,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    proj_out: Linear,
    num_heads: i64,
    head_dim: i64,
}

impl AttentionBlock2 {
    fn from_weights(weights: &Weights, prefix: &str, hidden: i64) -> Result<Self> {
        let norm_weight = weights.get(&format!("{prefix}.norm.weight"))?;
        let norm_bias = weights.get(&format!("{prefix}.norm.bias"))?;
        let to_q = Linear::from_weights(weights, &format!("{prefix}.to_q"))?;
        let to_k = Linear::from_weights(weights, &format!("{prefix}.to_k"))?;
        let to_v = Linear::from_weights(weights, &format!("{prefix}.to_v"))?;
        let proj_out = Linear::from_weights(weights, &format!("{prefix}.proj_out"))?;
        let num_heads = 4;
        let head_dim = hidden / num_heads;
        Ok(Self {
            norm_weight,
            norm_bias,
            to_q,
            to_k,
            to_v,
            proj_out,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x1: &Tensor, x2: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let x1n = layer_norm(x1, &self.norm_weight, &self.norm_bias, 1e-5);
        let x2n = layer_norm(x2, &self.norm_weight, &self.norm_bias, 1e-5);

        let q = self.to_q.forward(&x1n);
        let k = self.to_k.forward(&x2n);
        let v = self.to_v.forward(&x2n);

        let q = split_heads(&q, self.num_heads);
        let k = split_heads(&k, self.num_heads);
        let v = split_heads(&v, self.num_heads);

        let scale = (self.head_dim as f64).powf(-0.5);
        let mut scores = q.matmul(&k.transpose(-2, -1)) * scale;
        if let Some(mask) = mask {
            scores = scores.masked_fill(mask, f64::NEG_INFINITY);
        }
        let attn = scores.softmax(-1, Kind::Float);
        let out = attn.matmul(&v);
        let out = combine_heads(&out);
        Ok(self.proj_out.forward(&out) + x1)
    }
}

struct LlamaModel {
    layers: Vec<LlamaLayer>,
    norm: RmsNorm,
    rotary: RotaryEmbedding,
}

impl LlamaModel {
    fn from_weights(weights: &Weights, hidden: i64, device: Device) -> Result<Self> {
        let num_layers = 30;
        let mut layers = Vec::with_capacity(num_layers);
        for idx in 0..num_layers {
            layers.push(LlamaLayer::from_weights(weights, idx, hidden)?);
        }
        let norm = RmsNorm::from_weights(weights, "tfmr.norm.weight")?;
        let rotary = RotaryEmbedding::new(hidden / 16, 500000.0, device);
        Ok(Self { layers, norm, rotary })
    }

    fn forward(&self, inputs: &Tensor, past: Option<&mut LlamaPast>) -> Result<(Tensor, LlamaPast)> {
        let mut hidden = inputs.shallow_clone();
        let mut past = past.map(|p| p.clone()).unwrap_or_else(|| LlamaPast::new(self.layers.len()));
        let past_len = past.layers.get(0).and_then(|p| p.as_ref()).map(|kv| kv.k.size()[2]).unwrap_or(0);
        let seq_len = hidden.size()[1];
        let positions = Tensor::arange(seq_len, (Kind::Int64, hidden.device())) + past_len;
        let positions = positions.unsqueeze(0).expand(&[hidden.size()[0], -1], true);
        let (cos, sin) = self.rotary.cos_sin(&positions, hidden.kind());

        for (idx, layer) in self.layers.iter().enumerate() {
            let cache = past.layers.get(idx).and_then(|v| v.as_ref());
            let (out, kv) = layer.forward(&hidden, cache, &cos, &sin)?;
            hidden = out;
            past.layers[idx] = Some(kv);
        }

        let hidden = self.norm.forward(&hidden);
        Ok((hidden, past))
    }
}

struct LlamaLayer {
    input_norm: RmsNorm,
    post_norm: RmsNorm,
    attn: LlamaAttention,
    mlp: LlamaMlp,
}

impl LlamaLayer {
    fn from_weights(weights: &Weights, idx: usize, hidden: i64) -> Result<Self> {
        let prefix = format!("tfmr.layers.{idx}");
        let input_norm = RmsNorm::from_weights(weights, &format!("{prefix}.input_layernorm.weight"))?;
        let post_norm = RmsNorm::from_weights(weights, &format!("{prefix}.post_attention_layernorm.weight"))?;
        let attn = LlamaAttention::from_weights(weights, &format!("{prefix}.self_attn"), hidden)?;
        let mlp = LlamaMlp::from_weights(weights, &format!("{prefix}.mlp"), hidden)?;
        Ok(Self {
            input_norm,
            post_norm,
            attn,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cache: Option<&KvCache>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, KvCache)> {
        let h = self.input_norm.forward(x);
        let (attn_out, kv) = self.attn.forward(&h, cache, cos, sin)?;
        let x = x + attn_out;
        let h = self.post_norm.forward(&x);
        let mlp_out = self.mlp.forward(&h);
        Ok((x + mlp_out, kv))
    }
}

struct LlamaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: i64,
    head_dim: i64,
}

impl LlamaAttention {
    fn from_weights(weights: &Weights, prefix: &str, hidden: i64) -> Result<Self> {
        let q_proj = Linear::from_weights(weights, &format!("{prefix}.q_proj"))?;
        let k_proj = Linear::from_weights(weights, &format!("{prefix}.k_proj"))?;
        let v_proj = Linear::from_weights(weights, &format!("{prefix}.v_proj"))?;
        let o_proj = Linear::from_weights(weights, &format!("{prefix}.o_proj"))?;
        let num_heads = 16;
        let head_dim = hidden / num_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cache: Option<&KvCache>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, KvCache)> {
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let mut q = split_heads(&q, self.num_heads);
        let mut k = split_heads(&k, self.num_heads);
        let mut v = split_heads(&v, self.num_heads);

        let cos = cos.unsqueeze(1);
        let sin = sin.unsqueeze(1);
        q = apply_rope(&q, &cos, &sin);
        k = apply_rope(&k, &cos, &sin);

        let (k, v) = if let Some(cache) = cache {
            (
                Tensor::cat(&[cache.k.shallow_clone(), k], 2),
                Tensor::cat(&[cache.v.shallow_clone(), v], 2),
            )
        } else {
            (k, v)
        };

        let scale = (self.head_dim as f64).powf(-0.5);
        let mut scores = q.matmul(&k.transpose(-2, -1)) * scale;
        if x.size()[1] > 1 {
            let t = scores.size()[2];
            let s = scores.size()[3];
            let mask = Tensor::ones([t, s], (Kind::Float, x.device())).triu(1);
            let mask = mask.gt(0.0).unsqueeze(0).unsqueeze(0);
            scores = scores.masked_fill(&mask, f64::NEG_INFINITY);
        }
        let attn = scores.softmax(-1, Kind::Float);
        let out = attn.matmul(&v);
        let out = combine_heads(&out);
        let out = self.o_proj.forward(&out);
        Ok((out, KvCache { k, v }))
    }
}

struct LlamaMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl LlamaMlp {
    fn from_weights(weights: &Weights, prefix: &str, _hidden: i64) -> Result<Self> {
        Ok(Self {
            gate_proj: Linear::from_weights(weights, &format!("{prefix}.gate_proj"))?,
            up_proj: Linear::from_weights(weights, &format!("{prefix}.up_proj"))?,
            down_proj: Linear::from_weights(weights, &format!("{prefix}.down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let gate = self.gate_proj.forward(x).silu();
        let up = self.up_proj.forward(x);
        self.down_proj.forward(&(gate * up))
    }
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn from_weights(weights: &Weights, key: &str) -> Result<Self> {
        Ok(Self {
            weight: weights.get(key)?,
            eps: 1e-5,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let dtype = x.kind();
        let x_f = x.to_kind(Kind::Float);
        let variance = x_f.pow_tensor_scalar(2.0).mean_dim(&[-1_i64][..], true, Kind::Float);
        let normed = x_f * (variance + self.eps).rsqrt();
        normed.to_kind(dtype) * self.weight.to_device(x.device()).to_kind(dtype)
    }
}

struct RotaryEmbedding {
    inv_freq: Tensor,
    attention_scaling: f64,
}

impl RotaryEmbedding {
    fn new(dim: i64, base: f64, device: Device) -> Self {
        let inv_freq = Tensor::arange(dim / 2, (Kind::Float, device)) * 2.0 / (dim as f64);
        let inv_freq = (inv_freq * base.ln()).exp().reciprocal();
        let inv_freq = apply_llama3_scaling(&inv_freq);
        Self {
            inv_freq,
            attention_scaling: 1.0,
        }
    }

    fn cos_sin(&self, positions: &Tensor, dtype: Kind) -> (Tensor, Tensor) {
        let inv = self.inv_freq.to_device(positions.device()).to_kind(Kind::Float);
        let inv = inv.unsqueeze(0).unsqueeze(-1);
        let pos = positions.to_kind(Kind::Float).unsqueeze(1);
        let freqs = inv.matmul(&pos).transpose(1, 2);
        let emb = Tensor::cat(&[freqs.shallow_clone(), freqs], -1);
        let cos = emb.cos() * self.attention_scaling;
        let sin = emb.sin() * self.attention_scaling;
        (cos.to_kind(dtype), sin.to_kind(dtype))
    }
}

struct KvCache {
    k: Tensor,
    v: Tensor,
}

impl Clone for KvCache {
    fn clone(&self) -> Self {
        Self {
            k: self.k.shallow_clone(),
            v: self.v.shallow_clone(),
        }
    }
}

struct LlamaPast {
    layers: Vec<Option<KvCache>>,
}

impl Clone for LlamaPast {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.iter().map(|kv| kv.as_ref().map(|v| v.clone())).collect(),
        }
    }
}

impl LlamaPast {
    fn new(n: usize) -> Self {
        Self {
            layers: vec![None; n],
        }
    }
}

fn split_heads(x: &Tensor, num_heads: i64) -> Tensor {
    let b = x.size()[0];
    let t = x.size()[1];
    let dim = x.size()[2] / num_heads;
    x.view([b, t, num_heads, dim]).permute(&[0, 2, 1, 3])
}

fn combine_heads(x: &Tensor) -> Tensor {
    let b = x.size()[0];
    let t = x.size()[2];
    let h = x.size()[1];
    let d = x.size()[3];
    x.permute(&[0, 2, 1, 3]).contiguous().view([b, t, h * d])
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let x1 = x.i((.., .., .., 0..x.size()[3] / 2));
    let x2 = x.i((.., .., .., x.size()[3] / 2..));
    let rot = Tensor::cat(&[&x2.neg(), &x1], -1);
    x * cos + rot * sin
}

fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Tensor {
    let mean = x.mean_dim(&[-1_i64][..], true, Kind::Float);
    let var = (x - &mean).pow_tensor_scalar(2.0).mean_dim(&[-1_i64][..], true, Kind::Float);
    let y = (x - mean) / (var + eps).sqrt();
    y * weight.to_device(x.device()).to_kind(x.kind()) + bias.to_device(x.device()).to_kind(x.kind())
}

fn repetition_penalty(scores: &Tensor, input_ids: &Tensor, penalty: f64) -> Tensor {
    let gathered = scores.gather(1, input_ids, false);
    let neg_mask = gathered.lt(0.0).to_kind(Kind::Bool);
    let pos_mask = gathered.ge(0.0).to_kind(Kind::Bool);
    let penalized = &gathered * penalty;
    let rewarded = &gathered / penalty;
    let adjusted = blend_mask(&neg_mask, &penalized, &gathered);
    let adjusted = blend_mask(&pos_mask, &rewarded, &adjusted);
    scores.scatter(1, input_ids, &adjusted)
}

fn min_p_warp(scores: &Tensor, min_p: f64, min_keep: i64) -> Result<Tensor> {
    let probs = scores.softmax(-1, Kind::Float);
    let (top_probs, _) = probs.max_dim(-1, true);
    let scaled = top_probs * min_p;
    let tokens_to_remove = probs.lt_tensor(&scaled);
    let (_, sorted_idx) = scores.sort(-1, true);
    let mut sorted_remove = tokens_to_remove.gather(1, &sorted_idx, false);
    if min_keep > 0 {
        let _ = sorted_remove.i((.., 0..min_keep)).fill_(0);
    }
    let indices_to_remove = sorted_remove.scatter(1, &sorted_idx, &sorted_remove);
    Ok(scores.masked_fill(&indices_to_remove, f64::NEG_INFINITY))
}

fn top_p_warp(scores: &Tensor, top_p: f64, min_keep: i64) -> Result<Tensor> {
    let (sorted_scores, sorted_idx) = scores.sort(-1, false);
    let cumulative = sorted_scores.softmax(-1, Kind::Float).cumsum(-1, Kind::Float);
    let mut remove = cumulative.le(1.0 - top_p);
    if min_keep > 0 {
        let keep_start = (remove.size()[1] - min_keep).max(0);
        let _ = remove.i((.., keep_start..)).fill_(0);
    }
    let indices_to_remove = remove.scatter(1, &sorted_idx, &remove);
    Ok(scores.masked_fill(&indices_to_remove, f64::NEG_INFINITY))
}

fn apply_llama3_scaling(inv_freq: &Tensor) -> Tensor {
    let factor = 8.0;
    let low_freq_factor = 1.0;
    let high_freq_factor = 4.0;
    let old_context_len = 8192.0;

    let low_freq_wavelen = old_context_len / low_freq_factor;
    let high_freq_wavelen = old_context_len / high_freq_factor;
    let wavelen = (2.0 * std::f64::consts::PI) * inv_freq.reciprocal();

    let low_mask = wavelen.gt(low_freq_wavelen).to_kind(Kind::Bool);
    let inv_freq_llama = blend_mask(&low_mask, &(inv_freq / factor), inv_freq);
    let smooth_factor = ((wavelen.reciprocal() * old_context_len) - low_freq_factor)
        / (high_freq_factor - low_freq_factor);
    let smoothed =
        (1.0 - &smooth_factor) * (&inv_freq_llama / factor) + &smooth_factor * &inv_freq_llama;
    let is_medium = wavelen
        .ge(high_freq_wavelen)
        .logical_and(&wavelen.le(low_freq_wavelen))
        .to_kind(Kind::Bool);
    blend_mask(&is_medium, &smoothed, &inv_freq_llama)
}

fn blend_mask(mask: &Tensor, a: &Tensor, b: &Tensor) -> Tensor {
    let m = mask.to_kind(Kind::Float);
    a * &m + b * (1.0 - m)
}
