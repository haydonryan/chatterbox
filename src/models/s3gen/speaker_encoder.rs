use tch::Tensor;

use crate::error::{ChatterboxError, Result};
use crate::models::s3gen::weights::Weights;

pub struct CampPlus {
    _marker: (),
}

impl CampPlus {
    pub fn from_weights(weights: &Weights) -> Result<Self> {
        let _ = weights;
        Ok(Self {
            _marker: (),
        })
    }

    pub fn inference(&self, _audio_list: &[Tensor]) -> Result<Tensor> {
        Err(ChatterboxError::NotImplemented(
            "CAMPPlus speaker encoder is not implemented yet",
        ))
    }
}
