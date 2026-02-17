use std::collections::HashMap;
use std::fs;
use std::path::Path;

use safetensors::{Dtype, SafeTensors};
use tch::{Device, Kind, Tensor};

use crate::error::{ChatterboxError, Result};

#[derive(Debug)]
pub struct Weights {
    tensors: HashMap<String, Tensor>,
}

impl Weights {
    pub fn load<P: AsRef<Path>>(path: P, device: Device) -> Result<Self> {
        let data = fs::read(path)?;
        let st = SafeTensors::deserialize(&data)
            .map_err(|e| ChatterboxError::Safetensors(e.to_string()))?;

        let mut tensors = HashMap::with_capacity(st.len());
        for name in st.names() {
            let view = st
                .tensor(name)
                .map_err(|e| ChatterboxError::Safetensors(e.to_string()))?;
            let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
            let kind = dtype_to_kind(view.dtype())?;

            let tensor = Tensor::from_data_size(view.data(), &shape, kind).to_device(device);
            tensors.insert(name.to_string(), tensor);
        }

        Ok(Self { tensors })
    }

    pub fn get(&self, key: &str) -> Result<Tensor> {
        self.tensors
            .get(key)
            .map(|t| t.shallow_clone())
            .ok_or_else(|| ChatterboxError::MissingWeight(key.to_string()))
    }

    pub fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }
}

fn dtype_to_kind(dtype: Dtype) -> Result<Kind> {
    match dtype {
        Dtype::F16 => Ok(Kind::Half),
        Dtype::BF16 => Ok(Kind::BFloat16),
        Dtype::F32 => Ok(Kind::Float),
        Dtype::F64 => Ok(Kind::Double),
        Dtype::I8 => Ok(Kind::Int8),
        Dtype::U8 => Ok(Kind::Uint8),
        Dtype::I16 => Ok(Kind::Int16),
        Dtype::U16 => Err(ChatterboxError::Safetensors("Unsupported dtype U16".to_string())),
        Dtype::I32 => Ok(Kind::Int),
        Dtype::U32 => Err(ChatterboxError::Safetensors("Unsupported dtype U32".to_string())),
        Dtype::I64 => Ok(Kind::Int64),
        Dtype::U64 => Err(ChatterboxError::Safetensors("Unsupported dtype U64".to_string())),
        _ => Err(ChatterboxError::Safetensors(format!(
            "Unsupported dtype: {:?}",
            dtype
        ))),
    }
}
