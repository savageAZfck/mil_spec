//! `safetensors` — zero-dependency reader for the `.safetensors` format.
//!
//! Layout: 8-byte little-endian header length, a JSON object mapping
//! tensor names to `{dtype, shape, data_offsets}`, then raw tensor bytes.
//! We index the header eagerly and read tensor data lazily — a 14 GB
//! checkpoint streams through `read_f16` one tensor at a time instead of
//! landing in memory.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Safetensors element types we can read. Anything else errors at
/// `tensor_f16` time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StDType {
    /// float16
    F16,
    /// bfloat16
    Bf16,
    /// float32
    F32,
    /// int64
    I64,
    /// int32
    I32,
    /// uint8
    U8,
}

/// One tensor's location in the file.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    /// Element dtype as stored.
    pub dtype: StDType,
    /// Logical shape.
    pub shape: Vec<i64>,
    /// Byte offsets into the data section.
    pub data_start: u64,
    /// Byte offsets into the data section.
    pub data_end: u64,
}

/// An open `.safetensors` file.
pub struct Safetensors {
    path: PathBuf,
    /// Absolute offset where tensor data begins (8 + header_len).
    data_base: u64,
    /// name → tensor record
    map: BTreeMap<String, TensorInfo>,
}

/// Open and index a `.safetensors` file. Header parse only — no tensor
/// bytes are read until [`Safetensors::tensor_bytes`] or
/// [`Safetensors::tensor_f16`].
pub fn open(path: &Path) -> std::io::Result<Safetensors> {
    let mut f = std::fs::File::open(path)?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header)?;
    let json: serde_json::Value = serde_json::from_slice(&header)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let obj = json.as_object().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "header is not an object")
    })?;
    let mut map = BTreeMap::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = match v["dtype"].as_str().unwrap_or("") {
            "F16" => StDType::F16,
            "BF16" => StDType::Bf16,
            "F32" => StDType::F32,
            "I64" => StDType::I64,
            "I32" => StDType::I32,
            "U8" => StDType::U8,
            other => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{name}: unsupported dtype {other}"),
                ))
            }
        };
        let shape: Vec<i64> = v["shape"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|d| d.as_i64().unwrap_or(0))
            .collect();
        let offs = v["data_offsets"].as_array().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{name}: missing data_offsets"),
            )
        })?;
        map.insert(
            name.clone(),
            TensorInfo {
                dtype,
                shape,
                data_start: offs[0].as_u64().unwrap_or(0),
                data_end: offs[1].as_u64().unwrap_or(0),
            },
        );
    }
    Ok(Safetensors {
        path: path.to_path_buf(),
        data_base: 8 + header_len as u64,
        map,
    })
}

impl Safetensors {
    /// All tensor names in the file.
    pub fn names(&self) -> Vec<&str> {
        self.map.keys().map(|s| s.as_str()).collect()
    }
    /// Tensor metadata by name.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.map.get(name)
    }
    /// Whether a tensor exists.
    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    /// Raw stored bytes for a tensor.
    pub fn tensor_bytes(&self, name: &str) -> std::io::Result<Vec<u8>> {
        let t = self.map.get(name).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("tensor {name} not in {}", self.path.display()),
            )
        })?;
        let mut f = std::fs::File::open(&self.path)?;
        f.seek(SeekFrom::Start(self.data_base + t.data_start))?;
        let mut buf = vec![0u8; (t.data_end - t.data_start) as usize];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Tensor bytes converted to fp16 little-endian.
    ///
    /// F16 passes through; BF16 and F32 are converted per-element.
    /// Returned shape is the stored shape.
    pub fn tensor_f16(&self, name: &str) -> std::io::Result<(Vec<i64>, Vec<u8>)> {
        let t = self.map.get(name).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("tensor {name} not in {}", self.path.display()),
            )
        })?;
        let raw = self.tensor_bytes(name)?;
        let out = match t.dtype {
            StDType::F16 => raw,
            StDType::Bf16 => raw
                .chunks_exact(2)
                .flat_map(|c| {
                    // bf16 = top 16 bits of f32
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    let f = f32::from_bits((bits as u32) << 16);
                    half::f16::from_f32(f).to_le_bytes()
                })
                .collect(),
            StDType::F32 => raw
                .chunks_exact(4)
                .flat_map(|c| {
                    let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    half::f16::from_f32(f).to_le_bytes()
                })
                .collect(),
            other => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{name}: dtype {other:?} cannot convert to f16"),
                ))
            }
        };
        Ok((t.shape.clone(), out))
    }

    /// Tensor contents as f32 elements.
    ///
    /// F32 passes through; F16 and BF16 are widened per-element.
    /// Returned shape is the stored shape.
    pub fn tensor_f32(&self, name: &str) -> std::io::Result<(Vec<i64>, Vec<f32>)> {
        let t = self.map.get(name).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("tensor {name} not in {}", self.path.display()),
            )
        })?;
        let raw = self.tensor_bytes(name)?;
        let out = match t.dtype {
            StDType::F32 => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            StDType::F16 => raw
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            StDType::Bf16 => raw
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect(),
            other => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{name}: dtype {other:?} cannot convert to f32"),
                ))
            }
        };
        Ok((t.shape.clone(), out))
    }
}

/// Open every `.safetensors` file in a directory (handles sharded
/// checkpoints). Returns readers in sorted order; `find` locates the
/// shard owning a name.
pub fn open_dir(dir: &Path) -> std::io::Result<Vec<Safetensors>> {
    let mut files: Vec<PathBuf> = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.extension().and_then(|x| x.to_str()) == Some("safetensors") {
            files.push(p);
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no .safetensors files under {}", dir.display()),
        ));
    }
    files.iter().map(|p| open(p)).collect()
}

/// Find the shard containing `name` across open readers.
pub fn find<'a>(readers: &'a [Safetensors], name: &str) -> Option<&'a Safetensors> {
    readers.iter().find(|r| r.has(name))
}
