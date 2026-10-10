//! Per-tensor quantization planner.
//!
//! `WeightEmitter` applies one [`Precision`] per conv-weight tensor,
//! looked up from a [`Plan`]. Plans are built deterministically from
//! three signals:
//!
//! - **uniform** — today's policy: every conv weight is int8
//!   per-channel (`constexpr_blockwise_shift_scale`).
//! - **error** — measure the int8 quantize→dequantize error in f32
//!   (max-abs / RMSE / cosine). Tensors that quantize cleanly take int8
//!   — or `native` when the source already stores an exactly-
//!   transcodable block format (ggml Q8_0/Q4_0), which adds *zero*
//!   error. Tensors whose cosine similarity falls below
//!   [`ERROR_COSINE_FLOOR`] stay fp16.
//! - **placement** — consult [`mil_lint::classify_op`] on the op that
//!   consumes each weight. ANE-friendly consumers take the smallest
//!   *lossless-for-the-source* encoding (native when available, else
//!   int8); everything else stays fp16 — GPU/CPU have no reason to pay
//!   dequant ops.
//!
//! `milc convert --plan` prints a plan (JSON on stdout when
//! `--plan-file -` is used, otherwise a stderr table) without emitting
//! a package; `--plan-file <json>` feeds a hand-edited plan back in.
//! The file maps tensor name → precision and carries the metrics as
//! documentation — only `default`, `tensors` entries are consumed.

use std::collections::BTreeMap;

use crate::builder::{quantize_int8, Quant};
use crate::{ModelConfig, WeightSource};

/// Relative-RMSE ceiling for the `error` policy — a tensor whose int8
/// round-trip loses more than this fraction of its RMS stays fp16.
/// Cosine is reported for context but is a weak trigger (int8 can't
/// anticorrelate, so cosine stays ≥ ~0.999 even on bad tensors);
/// rel-RMSE is what actually discriminates.
pub const ERROR_REL_RMSE_MAX: f64 = 0.02;

/// Per-tensor precision decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Raw fp16 blob.
    Fp16,
    /// Per-channel int8 + fp16 scale.
    Int8,
    /// Source-native blockwise format (ggml Q8_0/Q4_0), exact.
    Native,
}

impl Precision {
    fn as_str(self) -> &'static str {
        match self {
            Precision::Fp16 => "fp16",
            Precision::Int8 => "int8",
            Precision::Native => "native",
        }
    }
    fn parse(s: &str) -> Option<Self> {
        match s {
            "fp16" => Some(Precision::Fp16),
            "int8" => Some(Precision::Int8),
            "native" => Some(Precision::Native),
            // a hand-edited plan may say "leave" — treat as "no
            // decision", i.e. the plan default
            _ => None,
        }
    }
}

/// Which signal drives plan decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantPolicy {
    /// Every conv weight int8 — historical `mil_convert` behaviour.
    Uniform,
    /// Error-driven: native > int8-if-clean > fp16.
    Error,
    /// ANE-consumer-driven: ANE→native-or-int8, else fp16.
    Placement,
}

impl QuantPolicy {
    /// Parse a `--quant-policy` value.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "uniform" => Some(QuantPolicy::Uniform),
            "error" => Some(QuantPolicy::Error),
            "placement" => Some(QuantPolicy::Placement),
            _ => None,
        }
    }
}

/// Quantize→dequantize error of one tensor under the int8 scheme,
/// measured in f32. `None` for tensors the policy didn't measure.
#[derive(Clone, Copy, Debug)]
pub struct Metrics {
    /// max |w − ŵ|
    pub max_abs: f64,
    /// sqrt(mean((w − ŵ)²))
    pub rmse: f64,
    /// cos(w, ŵ) — 1.0 is exact
    pub cosine: f64,
    /// RMS of the source tensor — `rmse / rms_w` is the relative error.
    pub rms_w: f64,
}

/// One tensor's decision plus the evidence behind it.
#[derive(Clone, Copy, Debug)]
pub struct TensorPlan {
    /// The chosen precision.
    pub precision: Precision,
    /// int8 round-trip metrics, when measured.
    pub metrics: Option<Metrics>,
}

/// tensor name → precision decision, with a default for unlisted names.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Precision for tensors absent from `tensors`.
    pub default: Precision,
    /// Per-tensor decisions (sorted — deterministic output).
    pub tensors: BTreeMap<String, TensorPlan>,
}

impl Plan {
    /// The effective precision for `name`.
    pub fn precision(&self, name: &str) -> Precision {
        self.tensors
            .get(name)
            .map(|t| t.precision)
            .unwrap_or(self.default)
    }

    /// The plan matching `--quant` exactly — every conv weight uses the
    /// uniform quantization. Used when no `--quant-policy`/`--plan-file`
    /// is given so the default path is byte-identical to before.
    pub fn uniform(quant: Quant) -> Self {
        Plan {
            default: match quant {
                Quant::Fp16 => Precision::Fp16,
                Quant::Int8 => Precision::Int8,
            },
            tensors: BTreeMap::new(),
        }
    }

    /// Build a plan for `names` (the conv-weight tensor list) under
    /// `policy`, reading tensors from `src`. Deterministic: names are
    /// visited in sorted order and every decision is a pure function of
    /// tensor bytes.
    pub fn compute(
        policy: QuantPolicy,
        names: &[String],
        src: &dyn WeightSource,
    ) -> std::io::Result<Self> {
        let mut tensors = BTreeMap::new();
        let mut sorted: Vec<&String> = names.iter().collect();
        sorted.sort();
        for name in sorted {
            let native = src.native_qblocks(name)?.is_some();
            let tp = match policy {
                QuantPolicy::Uniform => TensorPlan {
                    precision: Precision::Int8,
                    metrics: None,
                },
                QuantPolicy::Error => {
                    if native {
                        // exact transcode — zero added error
                        TensorPlan {
                            precision: Precision::Native,
                            metrics: Some(Metrics {
                                max_abs: 0.0,
                                rmse: 0.0,
                                cosine: 1.0,
                                rms_w: 0.0,
                            }),
                        }
                    } else {
                        let m = measure_int8(name, src)?;
                        let rel = if m.rms_w > 0.0 { m.rmse / m.rms_w } else { 0.0 };
                        TensorPlan {
                            precision: if rel <= ERROR_REL_RMSE_MAX {
                                Precision::Int8
                            } else {
                                Precision::Fp16
                            },
                            metrics: Some(m),
                        }
                    }
                }
                QuantPolicy::Placement => {
                    let ane = consumer_is_ane(name);
                    let precision = if !ane {
                        Precision::Fp16
                    } else if native {
                        Precision::Native
                    } else {
                        Precision::Int8
                    };
                    TensorPlan {
                        precision,
                        metrics: None,
                    }
                }
            };
            tensors.insert(name.clone(), tp);
        }
        Ok(Plan {
            default: match policy {
                QuantPolicy::Uniform => Precision::Int8,
                // a tensor the planner never saw (e.g. a new projection
                // added later) gets the conservative choice
                _ => Precision::Fp16,
            },
            tensors,
        })
    }

    /// JSON form: `{"default": "int8", "tensors": {name: "fp16"},
    /// "metrics": {name: {…}}}`. `metrics` is documentation only —
    /// [`Plan::from_json`] ignores it, so a hand-edit can't fake the
    /// measurements it carries.
    pub fn to_json(&self) -> String {
        let mut s = String::from("{\n  \"default\": \"");
        s.push_str(self.default.as_str());
        s.push_str("\",\n  \"tensors\": {");
        let mut first = true;
        for (name, tp) in &self.tensors {
            if !first {
                s.push(',');
            }
            first = false;
            s.push_str("\n    \"");
            s.push_str(&json_escape(name));
            s.push_str("\": \"");
            s.push_str(tp.precision.as_str());
            s.push('"');
        }
        s.push_str(if self.tensors.is_empty() {
            "}"
        } else {
            "\n  }"
        });
        let measured: Vec<_> = self
            .tensors
            .iter()
            .filter(|(_, t)| t.metrics.is_some())
            .collect();
        if !measured.is_empty() {
            s.push_str(",\n  \"metrics\": {");
            let mut first = true;
            for (name, tp) in measured {
                if !first {
                    s.push(',');
                }
                first = false;
                let m = tp.metrics.unwrap();
                s.push_str(&format!(
                    "\n    \"{}\": {{\"max_abs\": {:.6e}, \"rmse\": {:.6e}, \"rel_rmse\": {:.6}, \"cosine\": {:.8}}}",
                    json_escape(name),
                    m.max_abs,
                    m.rmse,
                    if m.rms_w > 0.0 { m.rmse / m.rms_w } else { 0.0 },
                    m.cosine
                ));
            }
            s.push_str("\n  }");
        }
        s.push_str("\n}\n");
        s
    }

    /// Parse a plan file — `default` plus per-tensor precision strings.
    /// Unknown precisions are rejected; `metrics` blocks are ignored.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let mut default = None;
        let mut tensors = BTreeMap::new();
        for (key, val) in json_pairs(json)? {
            match key.as_str() {
                "default" => {
                    let v = val.trim_matches('"');
                    default = Some(
                        Precision::parse(v)
                            .ok_or_else(|| format!("bad default precision {v:?}"))?,
                    );
                }
                "tensors" => {
                    for (name, p) in json_pairs(&val)? {
                        let v = p.trim_matches('"');
                        let prec = Precision::parse(v)
                            .ok_or_else(|| format!("{name}: bad precision {v:?}"))?;
                        tensors.insert(
                            name,
                            TensorPlan {
                                precision: prec,
                                metrics: None,
                            },
                        );
                    }
                }
                // "metrics" and anything else is documentation
                _ => {}
            }
        }
        Ok(Plan {
            default: default.unwrap_or(Precision::Fp16),
            tensors,
        })
    }
}

/// The conv-weight tensor list `build()` will emit — the set the
/// planner decides over. Mirrors the `conv_weight` calls in builder.rs:
/// keep in sync.
pub fn conv_weight_names(cfg: &ModelConfig) -> Vec<String> {
    let mut names = Vec::new();
    for l in 0..cfg.num_layers {
        for p in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            names.push(format!("model.layers.{l}.{p}.weight"));
        }
    }
    names.push(if cfg.tie_word_embeddings {
        "model.embed_tokens.weight".into()
    } else {
        "lm_head.weight".into()
    });
    names
}

/// int8 quantize→dequantize in f32, then max-abs / RMSE / cosine.
fn measure_int8(name: &str, src: &dyn WeightSource) -> std::io::Result<Metrics> {
    let (shape, w) = src.tensor_f32(name)?;
    if shape.len() != 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{name}: expected 2D weight, got {shape:?}"),
        ));
    }
    // quantize_int8 works on f16 bytes — reproduce the emit path
    // exactly (f32→f16→int8) so the measurement matches what ships.
    let f16b: Vec<u8> = w
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();
    let (q, s) = quantize_int8(&f16b, shape[0], shape[1]);
    let in_f = shape[1] as usize;
    let (mut dot, mut na, mut nb, mut se, mut max_abs) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (r, row) in w.chunks_exact(in_f).enumerate() {
        let scale = half::f16::from_le_bytes([s[2 * r], s[2 * r + 1]]).to_f32() as f64;
        for (c, &v) in row.iter().enumerate() {
            let deq = q[r * in_f + c] as i8 as f64 * scale;
            let a = v as f64;
            let e = a - deq;
            se += e * e;
            if e.abs() > max_abs {
                max_abs = e.abs();
            }
            dot += a * deq;
            na += a * a;
            nb += deq * deq;
        }
    }
    let n = (shape[0] * shape[1]) as f64;
    Ok(Metrics {
        max_abs,
        rmse: (se / n.max(1.0)).sqrt(),
        cosine: if na == 0.0 || nb == 0.0 {
            1.0
        } else {
            dot / (na.sqrt() * nb.sqrt())
        },
        rms_w: (na / n.max(1.0)).sqrt(),
    })
}

/// Would the op consuming `name` place on the ANE? Every conv weight
/// today feeds a rank-4 `conv` → [`mil_lint::classify_op`]. Kept as a
/// function so a future consumer map (e.g. gather-fed embeddings) has
/// one place to live.
fn consumer_is_ane(name: &str) -> bool {
    let _ = name;
    matches!(
        mil_lint::classify_op("conv", Some(mil_spec::DType::Fp16), 4, false).0,
        mil_lint::Unit::Ane
    )
}

/// Minimal `"key": value` pair splitter for the flat plan JSON —
/// handles one level of nested `{}` values (the `tensors`/`metrics`
/// blocks). Not a general parser; the plan format is intentionally
/// flat.
fn json_pairs(json: &str) -> Result<BTreeMap<String, String>, String> {
    let t = json.trim();
    let inner = t
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or("expected a JSON object")?;
    let mut out = BTreeMap::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    let mut splits = Vec::new();
    for c in inner.chars() {
        match c {
            '{' | '[' => depth += 1,
            '}' | ']' => depth -= 1,
            ',' if depth == 0 => {
                splits.push(std::mem::take(&mut cur));
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    if !cur.trim().is_empty() {
        splits.push(cur);
    }
    for kv in splits {
        let colon = kv.find(':').ok_or("expected key: value")?;
        let key = kv[..colon]
            .trim()
            .trim_matches('"')
            .replace("\\\"", "\"")
            .replace("\\\\", "\\");
        out.insert(key, kv[colon + 1..].trim().to_string());
    }
    Ok(out)
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NativeQuant;

    /// In-memory WeightSource: f32 tensors plus an optional set of
    /// "natively quantized" names.
    struct Mem {
        t: BTreeMap<String, (Vec<i64>, Vec<f32>)>,
        native: BTreeMap<String, NativeQuant>,
    }

    impl WeightSource for Mem {
        fn has(&self, name: &str) -> bool {
            self.t.contains_key(name)
        }
        fn tensor_f16(&self, name: &str) -> std::io::Result<(Vec<i64>, Vec<u8>)> {
            let (s, v) = self.t.get(name).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, name.to_string())
            })?;
            Ok((
                s.clone(),
                v.iter()
                    .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
                    .collect(),
            ))
        }
        fn tensor_f32(&self, name: &str) -> std::io::Result<(Vec<i64>, Vec<f32>)> {
            self.t
                .get(name)
                .cloned()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, name.to_string()))
        }
        fn native_qblocks(&self, name: &str) -> std::io::Result<Option<NativeQuant>> {
            Ok(self.native.get(name).map(|n| NativeQuant {
                data: n.data.clone(),
                data_dtype: n.data_dtype,
                scales: n.scales.clone(),
                offset: n.offset.as_ref().map(|(b, d)| (b.clone(), *d)),
                block: n.block,
            }))
        }
    }

    fn src(entries: &[(&str, Vec<i64>, Vec<f32>)], native: &[&str]) -> Mem {
        let mut m = Mem {
            t: BTreeMap::new(),
            native: BTreeMap::new(),
        };
        for (n, s, v) in entries {
            m.t.insert(n.to_string(), (s.clone(), v.clone()));
        }
        for n in native {
            m.native.insert(
                n.to_string(),
                NativeQuant {
                    data: vec![0; 4],
                    data_dtype: mil_spec::DType::Int8,
                    scales: vec![0; 4],
                    offset: None,
                    block: 32,
                },
            );
        }
        m
    }

    #[test]
    fn uniform_is_all_int8() {
        let m = src(&[("w", vec![4, 4], vec![0.1; 16])], &[]);
        let names = vec!["w".to_string()];
        let p = Plan::compute(QuantPolicy::Uniform, &names, &m).unwrap();
        assert_eq!(p.precision("w"), Precision::Int8);
        assert_eq!(p.precision("unlisted"), Precision::Int8);
    }

    #[test]
    fn error_prefers_native_when_exact() {
        let m = src(&[("w", vec![4, 32], vec![0.1; 128])], &["w"]);
        let names = vec!["w".to_string()];
        let p = Plan::compute(QuantPolicy::Error, &names, &m).unwrap();
        assert_eq!(p.precision("w"), Precision::Native);
        // exact transcode reports zero added error
        let mt = p.tensors["w"].metrics.unwrap();
        assert_eq!((mt.max_abs, mt.rmse, mt.cosine), (0.0, 0.0, 1.0));
    }

    #[test]
    fn error_picks_int8_for_clean_weights() {
        // int8-friendly: modest dynamic range, cosine >> floor
        let vals: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let m = src(&[("w", vec![4, 128], vals)], &[]);
        let names = vec!["w".to_string()];
        let p = Plan::compute(QuantPolicy::Error, &names, &m).unwrap();
        assert_eq!(p.precision("w"), Precision::Int8);
        let mt = p.tensors["w"].metrics.unwrap();
        assert!(mt.rmse / mt.rms_w <= ERROR_REL_RMSE_MAX);
    }

    #[test]
    fn error_keeps_fp16_on_pathological_tensor() {
        // Per-row scale is set by a modest outlier (2.0 → scale
        // 0.0157); the ±0.008 body then dequantizes to ±0.0157 — an
        // overshoot near 2× on every body element. Cosine lands ~0.9990
        // — below the 0.999 floor but not catastrophically, since int8
        // can't anticorrelate.
        let mut vals = Vec::new();
        for r in 0..4 {
            vals.push(2.0f32);
            for c in 0..127 {
                vals.push(if (r + c) % 2 == 0 { 0.008 } else { -0.008 });
            }
        }
        let m = src(&[("w", vec![4, 128], vals)], &[]);
        let names = vec!["w".to_string()];
        let p = Plan::compute(QuantPolicy::Error, &names, &m).unwrap();
        let mt = p.tensors["w"].metrics.unwrap();
        assert!(
            mt.rmse / mt.rms_w > ERROR_REL_RMSE_MAX,
            "rel_rmse {mt:?} should exceed the ceiling"
        );
        assert_eq!(p.precision("w"), Precision::Fp16);
    }

    #[test]
    fn placement_uses_native_or_int8_on_ane() {
        // every conv weight feeds an ANE `conv` — placement quantizes
        // all of them, preferring a native payload when the source has
        // one (exact) over the int8 requant (lossy).
        let m = src(
            &[
                ("w_nat", vec![4, 32], vec![0.1; 128]),
                ("w_fl", vec![4, 32], vec![0.1; 128]),
            ],
            &["w_nat"],
        );
        let names = vec!["w_nat".to_string(), "w_fl".to_string()];
        let p = Plan::compute(QuantPolicy::Placement, &names, &m).unwrap();
        assert_eq!(p.precision("w_nat"), Precision::Native);
        assert_eq!(p.precision("w_fl"), Precision::Int8);
    }

    #[test]
    fn plan_json_round_trip() {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "a".to_string(),
            TensorPlan {
                precision: Precision::Fp16,
                metrics: Some(Metrics {
                    max_abs: 1e-3,
                    rmse: 1e-4,
                    cosine: 0.9999,
                    rms_w: 0.5,
                }),
            },
        );
        tensors.insert(
            "b".to_string(),
            TensorPlan {
                precision: Precision::Native,
                metrics: None,
            },
        );
        let p = Plan {
            default: Precision::Int8,
            tensors,
        };
        let json = p.to_json();
        let q = Plan::from_json(&json).unwrap();
        assert_eq!(q.default, Precision::Int8);
        assert_eq!(q.precision("a"), Precision::Fp16);
        assert_eq!(q.precision("b"), Precision::Native);
        assert_eq!(q.precision("zzz"), Precision::Int8); // default
    }

    #[test]
    fn plan_file_rejects_bad_precision() {
        assert!(Plan::from_json(r#"{"default":"int8","tensors":{"a":"q4"}}"#).is_err());
        assert!(Plan::from_json(r#"{"default":"banana"}"#).is_err());
        // missing default is fine — falls back to fp16
        let p = Plan::from_json(r#"{"tensors":{"a":"int8"}}"#).unwrap();
        assert_eq!(p.precision("a"), Precision::Int8);
        assert_eq!(p.precision("b"), Precision::Fp16);
    }

    #[test]
    fn deterministic_across_runs() {
        let vals: Vec<f32> = (0..256).map(|i| (i as f32 * 0.013).cos()).collect();
        let m = src(&[("w", vec![2, 128], vals)], &[]);
        let names = vec!["w".to_string()];
        let a = Plan::compute(QuantPolicy::Error, &names, &m).unwrap();
        let b = Plan::compute(QuantPolicy::Error, &names, &m).unwrap();
        assert_eq!(a.to_json(), b.to_json());
    }
}
