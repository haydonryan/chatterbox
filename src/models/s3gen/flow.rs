use std::f64::consts::PI;

use tch::{Device, IndexOp, Kind, Tensor};

use crate::error::{ChatterboxError, Result};
use crate::models::s3gen::utils::mask::{add_optional_chunk_mask, make_pad_mask};
use crate::models::s3gen::weights::Weights;

const TOKEN_MEL_RATIO: i64 = 2;
const PRE_LOOKAHEAD_LEN: i64 = 3;
const OUTPUT_SIZE: i64 = 80;
const VOCAB_SIZE: i64 = 6561;

pub struct CausalMaskedDiffWithXvec {
    input_embedding: Tensor,
    spk_embed_affine: Linear,
    encoder: UpsampleConformerEncoder,
    encoder_proj: Linear,
    decoder: CausalConditionalCFM,
}

impl CausalMaskedDiffWithXvec {
    pub fn from_weights(weights: &Weights) -> Result<Self> {
        Ok(Self {
            input_embedding: weights.get("flow.input_embedding.weight")?,
            spk_embed_affine: Linear::from_weights(weights, "flow.spk_embed_affine_layer")?,
            encoder: UpsampleConformerEncoder::from_weights(weights)?,
            encoder_proj: Linear::from_weights(weights, "flow.encoder_proj")?,
            decoder: CausalConditionalCFM::from_weights(weights)?,
        })
    }

    pub fn inference(
        &self,
        token: &Tensor,
        token_len: &Tensor,
        prompt_token: &Tensor,
        prompt_token_len: &Tensor,
        prompt_feat: &Tensor,
        embedding: &Tensor,
        finalize: bool,
        n_timesteps: Option<i64>,
    ) -> Result<(Tensor, Tensor)> {
        let device = token.device();
        let embedding = l2_normalize(embedding, 1);
        let embedding = self.spk_embed_affine.forward(&embedding);

        let token_len1 = prompt_token.size()[1];
        let token_len2 = token.size()[1];
        let token = Tensor::cat(&[prompt_token, token], 1);
        let token_len = prompt_token_len + token_len;

        let mask = make_pad_mask(&token_len, 0)
            .logical_not()
            .unsqueeze(-1)
            .to_kind(embedding.kind());

        let token = token.clamp(0, VOCAB_SIZE - 1).to_kind(Kind::Int64);
        let token = Tensor::embedding(&self.input_embedding, &token, -1, false, false) * mask;

        let (mut h, _h_lengths) = self.encoder.forward(&token, &token_len)?;
        if !finalize {
            let trim = PRE_LOOKAHEAD_LEN * TOKEN_MEL_RATIO;
            let keep = h.size()[1].saturating_sub(trim);
            h = h.narrow(1, 0, keep);
        }

        let prompt_feat = prompt_feat.to_kind(h.kind());
        let mel_len1 = prompt_feat.size()[1];
        let mel_len2 = h.size()[1] - mel_len1;
        if mel_len2 <= 0 {
            return Err(ChatterboxError::ShapeMismatch(format!(
                "flow: mel_len2 <= 0 (mel_len1={}, h_len={})",
                mel_len1,
                h.size()[1]
            )));
        }
        let h = self.encoder_proj.forward(&h);

        let mut conds = Tensor::zeros([1, mel_len1 + mel_len2, OUTPUT_SIZE], (h.kind(), device));
        conds.i((.., 0..mel_len1, ..)).copy_(&prompt_feat);
        let conds = conds.transpose(1, 2);

        let mask = make_pad_mask(&Tensor::from_slice(&[mel_len1 + mel_len2]).to_device(device), 0)
            .logical_not()
            .to_kind(h.kind());
        let mask = mask.unsqueeze(1);

        let steps = n_timesteps.unwrap_or(10);
        let (feat, flow_cache) = self.decoder.forward(
            &h.transpose(1, 2).contiguous(),
            &mask,
            &embedding,
            &conds,
            steps,
        )?;
        let feat = feat.i((.., .., mel_len1..));
        Ok((feat, flow_cache))
    }
}

fn l2_normalize(x: &Tensor, dim: i64) -> Tensor {
    let norm = x
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist(&[dim][..], true, Kind::Float)
        .sqrt();
    x / norm.clamp_min(1e-12)
}

fn leaky_relu(x: &Tensor, slope: f64) -> Tensor {
    let relu = x.relu();
    &relu + (x - &relu) * slope
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
        let y = x.matmul(&self.weight.transpose(0, 1));
        match &self.bias {
            Some(bias) => y + bias,
            None => y,
        }
    }
}

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn from_weights(weights: &Weights, prefix: &str, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: weights.get(&format!("{prefix}.weight"))?,
            bias: weights.get(&format!("{prefix}.bias"))?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        x.layer_norm(&[self.weight.size()[0]], Some(&self.weight), Some(&self.bias), self.eps, false)
    }
}

struct EspnetRelPositionalEncoding {
    d_model: i64,
    xscale: f64,
}

impl EspnetRelPositionalEncoding {
    fn new(d_model: i64) -> Self {
        Self {
            d_model,
            xscale: (d_model as f64).sqrt(),
        }
    }

    fn position_encoding(&self, size: i64, device: Device, kind: Kind) -> Tensor {
        let half = self.d_model / 2;
        let position = Tensor::arange(size, (Kind::Float, device)).unsqueeze(1);
        let div_term = Tensor::arange_start_step(0, self.d_model, 2, (Kind::Float, device))
            * -((10000.0_f64).ln() / self.d_model as f64);
        let div_term = div_term.exp();
        let angles = position * div_term.unsqueeze(0);
        let pe_positive = Tensor::stack(&[angles.sin(), angles.cos()], -1)
            .view([size, self.d_model]);
        let pe_negative = Tensor::stack(&[(-&angles).sin(), (-&angles).cos()], -1)
            .view([size, self.d_model]);
        let pe_positive = pe_positive.flip(&[0]).unsqueeze(0);
        let pe_negative = if size > 1 {
            pe_negative.narrow(0, 1, size - 1)
        } else {
            Tensor::zeros([0, self.d_model], (Kind::Float, device))
        }
        .unsqueeze(0);
        let pe = Tensor::cat(&[pe_positive, pe_negative], 1);
        pe.to_kind(kind)
    }

    fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let x = x * self.xscale;
        let pos_emb = self.position_encoding(x.size()[1], x.device(), x.kind());
        (x, pos_emb)
    }
}

struct LinearNoSubsampling {
    linear: Linear,
    norm: LayerNorm,
    pos_enc: EspnetRelPositionalEncoding,
}

impl LinearNoSubsampling {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear: Linear::from_weights(weights, &format!("{prefix}.out.0"))?,
            norm: LayerNorm::from_weights(weights, &format!("{prefix}.out.1"), 1e-5)?,
            pos_enc: EspnetRelPositionalEncoding::new(512),
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> (Tensor, Tensor, Tensor) {
        let x = self.linear.forward(x);
        let x = self.norm.forward(&x);
        let (x, pos_emb) = self.pos_enc.forward(&x);
        (x, pos_emb, mask.shallow_clone())
    }
}

struct PositionwiseFeedForward {
    w1: Linear,
    w2: Linear,
}

impl PositionwiseFeedForward {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: Linear::from_weights(weights, &format!("{prefix}.w_1"))?,
            w2: Linear::from_weights(weights, &format!("{prefix}.w_2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.w1.forward(x).silu();
        self.w2.forward(&h)
    }
}

struct RelPositionMultiHeadedAttention {
    h: i64,
    d_k: i64,
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
}

impl RelPositionMultiHeadedAttention {
    fn from_weights(weights: &Weights, prefix: &str, n_head: i64, n_feat: i64) -> Result<Self> {
        Ok(Self {
            h: n_head,
            d_k: n_feat / n_head,
            linear_q: Linear::from_weights(weights, &format!("{prefix}.linear_q"))?,
            linear_k: Linear::from_weights(weights, &format!("{prefix}.linear_k"))?,
            linear_v: Linear::from_weights(weights, &format!("{prefix}.linear_v"))?,
            linear_out: Linear::from_weights(weights, &format!("{prefix}.linear_out"))?,
            linear_pos: Linear::from_weights(weights, &format!("{prefix}.linear_pos"))?,
            pos_bias_u: weights.get(&format!("{prefix}.pos_bias_u"))?,
            pos_bias_v: weights.get(&format!("{prefix}.pos_bias_v"))?,
        })
    }

    fn rel_shift(&self, x: &Tensor) -> Tensor {
        let b = x.size()[0];
        let h = x.size()[1];
        let t = x.size()[2];
        let zero_pad = Tensor::zeros([b, h, t, 1], (x.kind(), x.device()));
        let x_padded = Tensor::cat(&[zero_pad, x.shallow_clone()], -1);
        let x_padded = x_padded.view([b, h, x.size()[3] + 1, t]);
        let x = x_padded.i((.., .., 1.., ..)).view_as(x);
        x.i((.., .., .., 0..(x.size()[3] / 2 + 1)))
    }

    fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor, mask: &Tensor, pos_emb: &Tensor) -> Tensor {
        let b = query.size()[0];
        let t = query.size()[1];
        let q = self
            .linear_q
            .forward(query)
            .view([b, t, self.h, self.d_k]);
        let k = self
            .linear_k
            .forward(key)
            .view([b, -1, self.h, self.d_k])
            .transpose(1, 2);
        let v = self
            .linear_v
            .forward(value)
            .view([b, -1, self.h, self.d_k])
            .transpose(1, 2);

        let q = q.transpose(1, 2); // (B, H, T, d_k)
        let p = self
            .linear_pos
            .forward(pos_emb)
            .view([1, -1, self.h, self.d_k])
            .transpose(1, 2); // (1, H, 2T-1, d_k)

        let q_with_bias_u = (&q + self.pos_bias_u.unsqueeze(0).unsqueeze(2));
        let q_with_bias_v = (&q + self.pos_bias_v.unsqueeze(0).unsqueeze(2));

        let matrix_ac = q_with_bias_u.matmul(&k.transpose(-2, -1));
        let mut matrix_bd = q_with_bias_v.matmul(&p.transpose(-2, -1));
        if matrix_ac.size() != matrix_bd.size() {
            matrix_bd = self.rel_shift(&matrix_bd);
        }

        let mut scores = (matrix_ac + matrix_bd) / (self.d_k as f64).sqrt();
        let has_mask = mask.size()[2] > 0;
        let mut attn_mask = None;
        if has_mask {
            let mut m = mask.unsqueeze(1).eq(0);
            m = m.i((.., .., .., 0..scores.size()[3]));
            scores = scores.masked_fill(&m, f64::NEG_INFINITY);
            attn_mask = Some(m);
        }
        let mut attn = scores.softmax(-1, scores.kind());
        if let Some(m) = attn_mask {
            attn = attn.masked_fill(&m, 0.0);
        }
        let x = attn.matmul(&v);
        let x = x.transpose(1, 2).contiguous().view([b, t, self.h * self.d_k]);
        self.linear_out.forward(&x)
    }
}

struct ConformerEncoderLayer {
    norm_mha: LayerNorm,
    norm_ff: LayerNorm,
    self_attn: RelPositionMultiHeadedAttention,
    feed_forward: PositionwiseFeedForward,
}

impl ConformerEncoderLayer {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm_mha: LayerNorm::from_weights(weights, &format!("{prefix}.norm_mha"), 1e-12)?,
            norm_ff: LayerNorm::from_weights(weights, &format!("{prefix}.norm_ff"), 1e-12)?,
            self_attn: RelPositionMultiHeadedAttention::from_weights(weights, &format!("{prefix}.self_attn"), 8, 512)?,
            feed_forward: PositionwiseFeedForward::from_weights(weights, &format!("{prefix}.feed_forward"))?,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor, pos_emb: &Tensor) -> Tensor {
        let residual = x.shallow_clone();
        let x_norm = self.norm_mha.forward(x);
        let x_att = self.self_attn.forward(&x_norm, &x_norm, &x_norm, mask, pos_emb);
        let x = residual + x_att;

        let residual = x.shallow_clone();
        let x_norm = self.norm_ff.forward(&x);
        let x_ff = self.feed_forward.forward(&x_norm);
        residual + x_ff
    }
}

struct PreLookaheadLayer {
    conv1: Tensor,
    conv1_b: Tensor,
    conv2: Tensor,
    conv2_b: Tensor,
    pre_lookahead_len: i64,
}

impl PreLookaheadLayer {
    fn from_weights(weights: &Weights, prefix: &str, pre_lookahead_len: i64) -> Result<Self> {
        Ok(Self {
            conv1: weights.get(&format!("{prefix}.conv1.weight"))?,
            conv1_b: weights.get(&format!("{prefix}.conv1.bias"))?,
            conv2: weights.get(&format!("{prefix}.conv2.weight"))?,
            conv2_b: weights.get(&format!("{prefix}.conv2.bias"))?,
            pre_lookahead_len,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut y = x.transpose(1, 2).contiguous();
        y = y.pad(&[0, self.pre_lookahead_len], "constant", Some(0.0));
        y = leaky_relu(&y.conv1d(&self.conv1, Some(&self.conv1_b), 1, 0, 1, 1), 0.01);
        y = y.pad(&[2, 0], "constant", Some(0.0));
        y = y.conv1d(&self.conv2, Some(&self.conv2_b), 1, 0, 1, 1);
        y = y.transpose(1, 2).contiguous();
        y + x
    }
}

struct Upsample1D {
    weight: Tensor,
    bias: Tensor,
    stride: i64,
}

impl Upsample1D {
    fn from_weights(weights: &Weights, prefix: &str, stride: i64) -> Result<Self> {
        Ok(Self {
            weight: weights.get(&format!("{prefix}.conv.weight"))?,
            bias: weights.get(&format!("{prefix}.conv.bias"))?,
            stride,
        })
    }

    fn forward(&self, x: &Tensor, lengths: &Tensor) -> (Tensor, Tensor) {
        let mut y = x.upsample_nearest1d_vec(None::<&[i64]>, &[self.stride as f64]);
        y = y.pad(&[self.stride * 2, 0], "constant", Some(0.0));
        y = y.conv1d(&self.weight, Some(&self.bias), 1, 0, 1, 1);
        (y, lengths * self.stride)
    }
}

struct UpsampleConformerEncoder {
    embed: LinearNoSubsampling,
    up_embed: LinearNoSubsampling,
    pre_lookahead: PreLookaheadLayer,
    encoders: Vec<ConformerEncoderLayer>,
    up_encoders: Vec<ConformerEncoderLayer>,
    up_layer: Upsample1D,
    after_norm: LayerNorm,
}

impl UpsampleConformerEncoder {
    fn from_weights(weights: &Weights) -> Result<Self> {
        let mut encoders = Vec::with_capacity(6);
        for idx in 0..6 {
            encoders.push(ConformerEncoderLayer::from_weights(
                weights,
                &format!("flow.encoder.encoders.{idx}"),
            )?);
        }
        let mut up_encoders = Vec::with_capacity(4);
        for idx in 0..4 {
            up_encoders.push(ConformerEncoderLayer::from_weights(
                weights,
                &format!("flow.encoder.up_encoders.{idx}"),
            )?);
        }
        Ok(Self {
            embed: LinearNoSubsampling::from_weights(weights, "flow.encoder.embed")?,
            up_embed: LinearNoSubsampling::from_weights(weights, "flow.encoder.up_embed")?,
            pre_lookahead: PreLookaheadLayer::from_weights(weights, "flow.encoder.pre_lookahead_layer", PRE_LOOKAHEAD_LEN)?,
            encoders,
            up_encoders,
            up_layer: Upsample1D::from_weights(weights, "flow.encoder.up_layer", 2)?,
            after_norm: LayerNorm::from_weights(weights, "flow.encoder.after_norm", 1e-5)?,
        })
    }

    fn forward(&self, xs: &Tensor, xs_lens: &Tensor) -> Result<(Tensor, Tensor)> {
        let t = xs.size()[1];
        let masks = make_pad_mask(xs_lens, t).logical_not().unsqueeze(1);
        let (mut xs, pos_emb, _masks) = self.embed.forward(xs, &masks);
        let mask_pad = masks.shallow_clone();
        let chunk_masks = add_optional_chunk_mask(
            &xs,
            &masks,
            false,
            false,
            0,
            0,
            -1,
            true,
        );
        xs = self.pre_lookahead.forward(&xs);
        for layer in &self.encoders {
            xs = layer.forward(&xs, &chunk_masks, &pos_emb);
        }

        let mut xs_t = xs.transpose(1, 2).contiguous();
        let (mut xs_t, xs_lens) = self.up_layer.forward(&xs_t, xs_lens);
        let mut xs = xs_t.transpose(1, 2).contiguous();
        let t = xs.size()[1];
        let masks = make_pad_mask(&xs_lens, t).logical_not().unsqueeze(1);
        let (xs_out, pos_emb, _masks) = self.up_embed.forward(&xs, &masks);
        xs = xs_out;
        let mask_pad = masks.shallow_clone();
        let chunk_masks = add_optional_chunk_mask(
            &xs,
            &masks,
            false,
            false,
            0,
            0,
            -1,
            true,
        );
        for layer in &self.up_encoders {
            xs = layer.forward(&xs, &chunk_masks, &pos_emb);
        }
        let xs = self.after_norm.forward(&xs);
        Ok((xs, mask_pad))
    }
}

struct SinusoidalPosEmb {
    dim: i64,
}

impl SinusoidalPosEmb {
    fn new(dim: i64) -> Self {
        Self { dim }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let half = self.dim / 2;
        let emb = (10000.0_f64).ln() / ((half - 1) as f64);
        let emb = Tensor::arange(half, (Kind::Float, x.device())) * -emb;
        let emb = emb.exp();
        let x = x.unsqueeze(1) * emb.unsqueeze(0) * 1000.0;
        Tensor::cat(&[x.sin(), x.cos()], 1)
    }
}

struct TimestepEmbedding {
    linear1: Linear,
    linear2: Linear,
}

impl TimestepEmbedding {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear1: Linear::from_weights(weights, &format!("{prefix}.linear_1"))?,
            linear2: Linear::from_weights(weights, &format!("{prefix}.linear_2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.linear1.forward(x).silu();
        self.linear2.forward(&h)
    }
}

struct CausalConv1d {
    weight: Tensor,
    bias: Tensor,
    kernel: i64,
}

impl CausalConv1d {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        let weight = weights.get(&format!("{prefix}.weight"))?;
        let bias = weights.get(&format!("{prefix}.bias"))?;
        let kernel = weight.size()[2];
        Ok(Self { weight, bias, kernel })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let pad = self.kernel - 1;
        let x = x.pad(&[pad, 0], "constant", Some(0.0));
        x.conv1d(&self.weight, Some(&self.bias), 1, 0, 1, 1)
    }
}

struct CausalBlock1D {
    conv: CausalConv1d,
    norm: LayerNorm,
}

impl CausalBlock1D {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv: CausalConv1d::from_weights(weights, &format!("{prefix}.block.0"))?,
            norm: LayerNorm::from_weights(weights, &format!("{prefix}.block.2"), 1e-5)?,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Tensor {
        let x = x * mask;
        let mut y = self.conv.forward(&x);
        y = y.transpose(1, 2);
        y = self.norm.forward(&y);
        y = y.transpose(1, 2);
        y = y.mish();
        y * mask
    }
}

struct CausalResnetBlock1D {
    block1: CausalBlock1D,
    block2: CausalBlock1D,
    mlp: Linear,
    res_conv: LinearConv1d,
}

impl CausalResnetBlock1D {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            block1: CausalBlock1D::from_weights(weights, &format!("{prefix}.block1"))?,
            block2: CausalBlock1D::from_weights(weights, &format!("{prefix}.block2"))?,
            mlp: Linear::from_weights(weights, &format!("{prefix}.mlp.1"))?,
            res_conv: LinearConv1d::from_weights(weights, &format!("{prefix}.res_conv"))?,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor, time_emb: &Tensor) -> Tensor {
        let mut h = self.block1.forward(x, mask);
        // Match PyTorch: mlp = Mish -> Linear
        let temb = self.mlp.forward(&time_emb.mish()).unsqueeze(-1);
        h = h + temb;
        h = self.block2.forward(&h, mask);
        h + self.res_conv.forward(&(x * mask))
    }
}

struct LinearConv1d {
    weight: Tensor,
    bias: Tensor,
}

impl LinearConv1d {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: weights.get(&format!("{prefix}.weight"))?,
            bias: weights.get(&format!("{prefix}.bias"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        x.conv1d(&self.weight, Some(&self.bias), 1, 0, 1, 1)
    }
}

struct BasicTransformerBlock {
    norm1: LayerNorm,
    norm3: LayerNorm,
    attn: SelfAttention,
    ff: FeedForward,
}

impl BasicTransformerBlock {
    fn from_weights(weights: &Weights, prefix: &str, heads: i64, head_dim: i64) -> Result<Self> {
        Ok(Self {
            norm1: LayerNorm::from_weights(weights, &format!("{prefix}.norm1"), 1e-5)?,
            norm3: LayerNorm::from_weights(weights, &format!("{prefix}.norm3"), 1e-5)?,
            attn: SelfAttention::from_weights(weights, &format!("{prefix}.attn1"), heads, head_dim)?,
            ff: FeedForward::from_weights(weights, &format!("{prefix}.ff"))?,
        })
    }

    fn forward(&self, x: &Tensor, attn_bias: &Tensor, t: &Tensor) -> Tensor {
        let h = self.norm1.forward(x);
        let h = self.attn.forward(&h, attn_bias) + x;
        let h2 = self.norm3.forward(&h);
        self.ff.forward(&h2) + h
    }
}

struct SelfAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    heads: i64,
    head_dim: i64,
}

impl SelfAttention {
    fn from_weights(weights: &Weights, prefix: &str, heads: i64, head_dim: i64) -> Result<Self> {
        Ok(Self {
            to_q: Linear::from_weights(weights, &format!("{prefix}.to_q"))?,
            to_k: Linear::from_weights(weights, &format!("{prefix}.to_k"))?,
            to_v: Linear::from_weights(weights, &format!("{prefix}.to_v"))?,
            to_out: Linear::from_weights(weights, &format!("{prefix}.to_out.0"))?,
            heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor, attn_bias: &Tensor) -> Tensor {
        let b = x.size()[0];
        let t = x.size()[1];
        let inner = self.heads * self.head_dim;
        let q = self.to_q.forward(x).view([b, t, self.heads, self.head_dim]).transpose(1, 2);
        let k = self.to_k.forward(x).view([b, t, self.heads, self.head_dim]).transpose(1, 2);
        let v = self.to_v.forward(x).view([b, t, self.heads, self.head_dim]).transpose(1, 2);
        let scale = (self.head_dim as f64).sqrt();
        let mut scores = q.matmul(&k.transpose(-2, -1)) / scale;
        if attn_bias.numel() > 0 {
            let bias = if attn_bias.dim() == 3 {
                attn_bias.unsqueeze(1)
            } else {
                attn_bias.shallow_clone()
            };
            scores = scores + &bias;
        }
        let attn = scores.softmax(-1, scores.kind());
        let out = attn.matmul(&v);
        let out = out.transpose(1, 2).contiguous().view([b, t, inner]);
        self.to_out.forward(&out)
    }
}

struct FeedForward {
    proj: Linear,
    proj_out: Linear,
}

impl FeedForward {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            proj: Linear::from_weights(weights, &format!("{prefix}.net.0.proj"))?,
            proj_out: Linear::from_weights(weights, &format!("{prefix}.net.2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.proj.forward(x).gelu("none");
        self.proj_out.forward(&h)
    }
}

struct ConditionalDecoder {
    time_embed: SinusoidalPosEmb,
    time_mlp: TimestepEmbedding,
    down_blocks: Vec<DecoderDownBlock>,
    mid_blocks: Vec<DecoderMidBlock>,
    up_blocks: Vec<DecoderUpBlock>,
    final_block: CausalBlock1D,
    final_proj: LinearConv1d,
    static_chunk_size: i64,
}

impl ConditionalDecoder {
    fn from_weights(weights: &Weights) -> Result<Self> {
        let time_embed = SinusoidalPosEmb::new(320);
        let time_mlp = TimestepEmbedding::from_weights(weights, "flow.decoder.estimator.time_mlp")?;
        let mut down_blocks = Vec::new();
        down_blocks.push(DecoderDownBlock::from_weights(weights, "flow.decoder.estimator.down_blocks.0")?);
        let mut mid_blocks = Vec::with_capacity(12);
        for idx in 0..12 {
            mid_blocks.push(DecoderMidBlock::from_weights(weights, &format!("flow.decoder.estimator.mid_blocks.{idx}"))?);
        }
        let mut up_blocks = Vec::new();
        up_blocks.push(DecoderUpBlock::from_weights(weights, "flow.decoder.estimator.up_blocks.0")?);
        Ok(Self {
            time_embed,
            time_mlp,
            down_blocks,
            mid_blocks,
            up_blocks,
            final_block: CausalBlock1D::from_weights(weights, "flow.decoder.estimator.final_block")?,
            final_proj: LinearConv1d::from_weights(weights, "flow.decoder.estimator.final_proj")?,
            static_chunk_size: 0,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor, mu: &Tensor, t: &Tensor, spks: &Tensor, cond: &Tensor) -> Tensor {
        let t = self.time_embed.forward(t);
        let t = self.time_mlp.forward(&t);
        let mut x = Tensor::cat(&[x, mu], 1);
        let spks = spks.unsqueeze(-1).repeat(&[1, 1, x.size()[2]]);
        x = Tensor::cat(&[x, spks], 1);
        x = Tensor::cat(&[x, cond.shallow_clone()], 1);

        let mut hiddens: Vec<Tensor> = Vec::new();
        let mut masks: Vec<Tensor> = vec![mask.shallow_clone()];

        for block in &self.down_blocks {
            let mask_down = masks.last().unwrap().shallow_clone();
            let (x_out, mask_next, hidden) = block.forward(&x, &mask_down, &t, self.static_chunk_size);
            x = x_out;
            hiddens.push(hidden);
            masks.push(mask_next);
        }
        masks.pop();
        let mask_mid = masks.last().unwrap().shallow_clone();

        for block in &self.mid_blocks {
            x = block.forward(&x, &mask_mid, &t, self.static_chunk_size);
        }

        let mut mask_up_last = mask_mid.shallow_clone();
        for block in &self.up_blocks {
            let mask_up = masks.pop().unwrap();
            let skip = hiddens.pop().unwrap();
            let x_cat = Tensor::cat(&[x.i((.., .., 0..skip.size()[2])), skip], 1);
            x = block.forward(&x_cat, &mask_up, &t, self.static_chunk_size);
            mask_up_last = mask_up;
        }
        x = self.final_block.forward(&x, &mask_up_last);
        self.final_proj.forward(&(x * &mask_up_last)) * mask_up_last
    }
}

struct DecoderDownBlock {
    resnet: CausalResnetBlock1D,
    transformers: Vec<BasicTransformerBlock>,
    downsample: CausalConv1d,
}

impl DecoderDownBlock {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        let resnet = CausalResnetBlock1D::from_weights(weights, &format!("{prefix}.0"))?;
        let mut transformers = Vec::with_capacity(4);
        for idx in 0..4 {
            transformers.push(BasicTransformerBlock::from_weights(
                weights,
                &format!("{prefix}.1.{idx}"),
                8,
                64,
            )?);
        }
        let downsample = CausalConv1d::from_weights(weights, &format!("{prefix}.2"))?;
        Ok(Self {
            resnet,
            transformers,
            downsample,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor, t: &Tensor, static_chunk_size: i64) -> (Tensor, Tensor, Tensor) {
        let mut x = self.resnet.forward(x, mask, t);
        let mut x_t = x.transpose(1, 2).contiguous();
        let attn_mask = add_optional_chunk_mask(
            &x_t,
            &mask.to_kind(Kind::Bool),
            false,
            false,
            0,
            static_chunk_size,
            -1,
            true,
        );
        let attn_bias = mask_to_bias(&attn_mask, x_t.kind());
        for block in &self.transformers {
            x_t = block.forward(&x_t, &attn_bias, t);
        }
        x = x_t.transpose(1, 2).contiguous();
        let hidden = x.shallow_clone();
        x = self.downsample.forward(&(x * mask));
        let idx = Tensor::arange_start_step(0, mask.size()[2], 2, (Kind::Int64, mask.device()));
        let mask_next = mask.index_select(2, &idx);
        (x, mask_next, hidden)
    }
}

struct DecoderMidBlock {
    resnet: CausalResnetBlock1D,
    transformers: Vec<BasicTransformerBlock>,
}

impl DecoderMidBlock {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        let resnet = CausalResnetBlock1D::from_weights(weights, &format!("{prefix}.0"))?;
        let mut transformers = Vec::with_capacity(4);
        for idx in 0..4 {
            transformers.push(BasicTransformerBlock::from_weights(
                weights,
                &format!("{prefix}.1.{idx}"),
                8,
                64,
            )?);
        }
        Ok(Self { resnet, transformers })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor, t: &Tensor, static_chunk_size: i64) -> Tensor {
        let mut x = self.resnet.forward(x, mask, t);
        let mut x_t = x.transpose(1, 2).contiguous();
        let attn_mask = add_optional_chunk_mask(
            &x_t,
            &mask.to_kind(Kind::Bool),
            false,
            false,
            0,
            static_chunk_size,
            -1,
            true,
        );
        let attn_bias = mask_to_bias(&attn_mask, x_t.kind());
        for block in &self.transformers {
            x_t = block.forward(&x_t, &attn_bias, t);
        }
        x_t.transpose(1, 2).contiguous()
    }
}

struct DecoderUpBlock {
    resnet: CausalResnetBlock1D,
    transformers: Vec<BasicTransformerBlock>,
    upsample: CausalConv1d,
}

impl DecoderUpBlock {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        let resnet = CausalResnetBlock1D::from_weights(weights, &format!("{prefix}.0"))?;
        let mut transformers = Vec::with_capacity(4);
        for idx in 0..4 {
            transformers.push(BasicTransformerBlock::from_weights(
                weights,
                &format!("{prefix}.1.{idx}"),
                8,
                64,
            )?);
        }
        let upsample = CausalConv1d::from_weights(weights, &format!("{prefix}.2"))?;
        Ok(Self {
            resnet,
            transformers,
            upsample,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor, t: &Tensor, static_chunk_size: i64) -> Tensor {
        let mut x = self.resnet.forward(x, mask, t);
        let mut x_t = x.transpose(1, 2).contiguous();
        let attn_mask = add_optional_chunk_mask(
            &x_t,
            &mask.to_kind(Kind::Bool),
            false,
            false,
            0,
            static_chunk_size,
            -1,
            true,
        );
        let attn_bias = mask_to_bias(&attn_mask, x_t.kind());
        for block in &self.transformers {
            x_t = block.forward(&x_t, &attn_bias, t);
        }
        x = x_t.transpose(1, 2).contiguous();
        self.upsample.forward(&(x * mask))
    }
}

fn mask_to_bias(mask: &Tensor, kind: Kind) -> Tensor {
    let mask = mask.to_kind(kind);
    (Tensor::ones_like(&mask) - mask) * -1.0e10
}

struct CausalConditionalCFM {
    estimator: ConditionalDecoder,
    rand_noise: Tensor,
    inference_cfg_rate: f64,
    t_scheduler_cosine: bool,
}

impl CausalConditionalCFM {
    fn from_weights(weights: &Weights) -> Result<Self> {
        Ok(Self {
            estimator: ConditionalDecoder::from_weights(weights)?,
            rand_noise: Tensor::randn([1, 80, 50 * 300], (Kind::Float, Device::Cpu)),
            inference_cfg_rate: 0.7,
            t_scheduler_cosine: true,
        })
    }

    fn forward(
        &self,
        mu: &Tensor,
        mask: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
        n_timesteps: i64,
    ) -> Result<(Tensor, Tensor)> {
        let mut z = Tensor::randn_like(mu);
        let t_span = if self.t_scheduler_cosine {
            let t = Tensor::linspace(0.0, 1.0, n_timesteps + 1, (mu.kind(), mu.device()));
            Tensor::ones_like(&t) - (t * 0.5 * PI).cos()
        } else {
            Tensor::linspace(0.0, 1.0, n_timesteps + 1, (mu.kind(), mu.device()))
        };
        let out = self.solve_euler(&z, &t_span, mu, mask, spks, cond)?;
        Ok((out, Tensor::zeros([1, 80, 0, 2], (mu.kind(), mu.device()))))
    }

    fn solve_euler(
        &self,
        x: &Tensor,
        t_span: &Tensor,
        mu: &Tensor,
        mask: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
    ) -> Result<Tensor> {
        let mut x = x.shallow_clone();
        let mut t = t_span.i(0).unsqueeze(0);
        let mut dt = t_span.i(1) - t_span.i(0);
        let steps = t_span.size()[0];

        let mut x_in = Tensor::zeros([2, 80, x.size()[2]], (x.kind(), x.device()));
        let mut mask_in = Tensor::zeros([2, 1, x.size()[2]], (mask.kind(), mask.device()));
        let mut mu_in = Tensor::zeros([2, 80, x.size()[2]], (mu.kind(), mu.device()));
        let mut t_in = Tensor::zeros([2], (t.kind(), t.device()));
        let mut spks_in = Tensor::zeros([2, 80], (spks.kind(), spks.device()));
        let mut cond_in = Tensor::zeros([2, 80, x.size()[2]], (cond.kind(), cond.device()));

        for step in 1..steps {
            x_in.i((0..1, .., ..)).copy_(&x);
            x_in.i((1..2, .., ..)).copy_(&x);
            mask_in.i((0..1, .., ..)).copy_(mask);
            mask_in.i((1..2, .., ..)).copy_(mask);
            mu_in.i((0..1, .., ..)).copy_(mu);
            t_in.i(0..1).copy_(&t);
            t_in.i(1..2).copy_(&t);
            spks_in.i((0..1, ..)).copy_(spks);
            cond_in.i((0..1, .., ..)).copy_(cond);

            let dphi = self
                .estimator
                .forward(&x_in, &mask_in, &mu_in, &t_in, &spks_in, &cond_in);
            let dphi_dt = dphi.i(0).unsqueeze(0);
            let cfg_dphi_dt = dphi.i(1).unsqueeze(0);
            let dphi_dt = (1.0 + self.inference_cfg_rate) * &dphi_dt - self.inference_cfg_rate * cfg_dphi_dt;
            x = x + &dt * dphi_dt;
            t = t + &dt;
            if step < steps - 1 {
                dt = t_span.i(step + 1) - &t;
            }
        }
        Ok(x)
    }
}
