use tch::{Device, Kind, Tensor};

/// Create mask for subsequent steps (size, size) with chunk size.
///
/// NOTE: This matches the modified implementation in Python and ignores `num_left_chunks`.
pub fn subsequent_chunk_mask(size: i64, chunk_size: i64, device: Device) -> Tensor {
    let pos_idx = Tensor::arange(size, (Kind::Int64, device));
    let block_value = ((&pos_idx / chunk_size) + 1) * chunk_size;
    pos_idx.unsqueeze(0).lt_tensor(&block_value.unsqueeze(1))
}

/// Apply optional mask for encoder.
#[allow(clippy::too_many_arguments)]
pub fn add_optional_chunk_mask(
    xs: &Tensor,
    masks: &Tensor,
    use_dynamic_chunk: bool,
    use_dynamic_left_chunk: bool,
    decoding_chunk_size: i64,
    static_chunk_size: i64,
    num_decoding_left_chunks: i64,
    enable_full_context: bool,
) -> Tensor {
    let device = xs.device();
    if use_dynamic_chunk {
        let max_len = xs.size()[1];
        let (chunk_size, num_left_chunks) = if decoding_chunk_size < 0 {
            (max_len, -1)
        } else if decoding_chunk_size > 0 {
            (decoding_chunk_size, num_decoding_left_chunks)
        } else {
            let mut chunk_size = Tensor::randint(max_len, &[1], (Kind::Int64, device))
                .int64_value(&[0]);
            let mut num_left_chunks = -1;
            if chunk_size > max_len / 2 && enable_full_context {
                chunk_size = max_len;
            } else {
                chunk_size = (chunk_size % 25) + 1;
                if use_dynamic_left_chunk {
                    let max_left_chunks = (max_len - 1) / chunk_size;
                    num_left_chunks = Tensor::randint(
                        max_left_chunks + 1,
                        &[1],
                        (Kind::Int64, device),
                    )
                    .int64_value(&[0]);
                }
            }
            (chunk_size, num_left_chunks)
        };

        let chunk_masks = subsequent_chunk_mask(max_len, chunk_size, device);
        let chunk_masks = chunk_masks.unsqueeze(0);
        let chunk_masks = masks.logical_and(&chunk_masks);
        return enforce_nonempty_masks(chunk_masks);
    }

    if static_chunk_size > 0 {
        let max_len = xs.size()[1];
        let chunk_masks = subsequent_chunk_mask(max_len, static_chunk_size, device);
        let chunk_masks = chunk_masks.unsqueeze(0);
        let chunk_masks = masks.logical_and(&chunk_masks);
        return enforce_nonempty_masks(chunk_masks);
    }

    enforce_nonempty_masks(masks.shallow_clone())
}

fn enforce_nonempty_masks(mask: Tensor) -> Tensor {
    let mask_sum = mask.sum_dim_intlist([-1].as_slice(), false, Kind::Int64);
    let zeros = mask_sum.eq(0);
    if zeros.any().int64_value(&[]) != 0 {
        let mut mask = mask;
        mask = mask.where_self(
            &zeros.unsqueeze(-1).logical_not(),
            &Tensor::ones_like(&mask),
        );
        return mask;
    }
    mask
}

/// Make mask tensor containing indices of padded part.
pub fn make_pad_mask(lengths: &Tensor, max_len: i64) -> Tensor {
    let lengths = lengths.to_kind(Kind::Int64);
    let batch_size = lengths.size()[0];
    let max_len = if max_len > 0 {
        max_len
    } else {
        lengths.max().int64_value(&[])
    };
    let seq_range = Tensor::arange(max_len, (Kind::Int64, lengths.device()));
    let seq_range_expand = seq_range.unsqueeze(0).expand([batch_size, max_len], true);
    let seq_length_expand = lengths.unsqueeze(-1);
    seq_range_expand.ge_tensor(&seq_length_expand)
}
