//! Parser tests: a minimal in-memory GGUF writer exercises the full
//! header grammar — all 13 metadata value types, arrays, alignment,
//! tensor bounds — plus corrupt/truncated variants which must error,
//! never panic.

use mil_convert::gguf::{Gguf, GgufType, Value};
use std::path::{Path, PathBuf};

// ---- tiny GGUF writer ----

#[derive(Default)]
struct W(Vec<u8>);
impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.0.extend(v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend(v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.0.extend(v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend(v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.0.extend(v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend(v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend(v.to_le_bytes());
    }
    fn s(&mut self, v: &str) {
        self.u64(v.len() as u64);
        self.0.extend(v.as_bytes());
    }
    fn pad_to(&mut self, align: u64, file_base: u64) {
        let pos = file_base + self.0.len() as u64;
        let pad = pos.div_ceil(align) * align - pos;
        self.0.resize(self.0.len() + pad as usize, 0);
    }
}

struct TensorSpec {
    name: &'static str,
    ne: Vec<i64>,
    ty: i32,
    data: Vec<u8>,
}

/// Assemble a complete v3 file. `offset_override` overrides the stored
/// tensor-offset field (to craft malformed files).
fn make_gguf(
    kvs: &[(&str, Value)],
    tensors: &[TensorSpec],
    alignment: u64,
    offset_override: Option<u64>,
) -> Vec<u8> {
    let mut w = W::default();
    w.0.extend(b"GGUF");
    w.u32(3);
    let wrote_align = kvs.iter().any(|(k, _)| *k == "general.alignment");
    w.u64(tensors.len() as u64);
    w.u64((kvs.len() + usize::from(!wrote_align)) as u64);
    for (k, v) in kvs {
        w.s(k);
        write_value(&mut w, v);
    }
    if !wrote_align {
        w.s("general.alignment");
        write_value(&mut w, &Value::U32(alignment as u32));
    }
    let mut data_off = 0u64;
    let mut datas: Vec<&[u8]> = Vec::new();
    for t in tensors {
        w.s(t.name);
        w.u32(t.ne.len() as u32);
        for &d in &t.ne {
            w.i64(d);
        }
        w.i32(t.ty);
        w.u64(offset_override.unwrap_or(data_off));
        datas.push(&t.data);
        data_off += t.data.len() as u64;
        data_off = data_off.div_ceil(alignment) * alignment;
    }
    w.pad_to(alignment, 0);
    let mut out = w.0;
    for (i, d) in datas.iter().enumerate() {
        out.extend_from_slice(d);
        if i + 1 < datas.len() {
            let pos = out.len() as u64;
            let pad = pos.div_ceil(alignment) * alignment - pos;
            out.resize(out.len() + pad as usize, 0);
        }
    }
    out
}

fn arr_ty(kind: &mil_convert::gguf::ArrKind) -> i32 {
    match kind {
        mil_convert::gguf::ArrKind::U8 => 0,
        mil_convert::gguf::ArrKind::I8 => 1,
        mil_convert::gguf::ArrKind::U16 => 2,
        mil_convert::gguf::ArrKind::I16 => 3,
        mil_convert::gguf::ArrKind::U32 => 4,
        mil_convert::gguf::ArrKind::I32 => 5,
        mil_convert::gguf::ArrKind::F32 => 6,
        mil_convert::gguf::ArrKind::Bool => 7,
        mil_convert::gguf::ArrKind::Str => 8,
        mil_convert::gguf::ArrKind::U64 => 10,
        mil_convert::gguf::ArrKind::I64 => 11,
        mil_convert::gguf::ArrKind::F64 => 12,
    }
}

/// Payload only — array elements carry no per-element type tag.
fn write_payload(w: &mut W, v: &Value) {
    match v {
        Value::U8(x) => w.u8(*x),
        Value::I8(x) => w.u8(*x as u8),
        Value::U16(x) => w.u16(*x),
        Value::I16(x) => w.u16(*x as u16),
        Value::U32(x) => w.u32(*x),
        Value::I32(x) => w.i32(*x),
        Value::F32(x) => w.f32(*x),
        Value::Bool(x) => w.u8(*x as u8),
        Value::Str(x) => w.s(x),
        Value::U64(x) => w.u64(*x),
        Value::I64(x) => w.i64(*x),
        Value::F64(x) => w.f64(*x),
        Value::Arr(_, _) => panic!("nested arrays are not valid GGUF"),
    }
}

fn write_value(w: &mut W, v: &Value) {
    match v {
        Value::U8(x) => {
            w.i32(0);
            w.u8(*x);
        }
        Value::I8(x) => {
            w.i32(1);
            w.u8(*x as u8);
        }
        Value::U16(x) => {
            w.i32(2);
            w.u16(*x);
        }
        Value::I16(x) => {
            w.i32(3);
            w.u16(*x as u16);
        }
        Value::U32(x) => {
            w.i32(4);
            w.u32(*x);
        }
        Value::I32(x) => {
            w.i32(5);
            w.i32(*x);
        }
        Value::F32(x) => {
            w.i32(6);
            w.f32(*x);
        }
        Value::Bool(x) => {
            w.i32(7);
            w.u8(*x as u8);
        }
        Value::Str(x) => {
            w.i32(8);
            w.s(x);
        }
        Value::U64(x) => {
            w.i32(10);
            w.u64(*x);
        }
        Value::I64(x) => {
            w.i32(11);
            w.i64(*x);
        }
        Value::F64(x) => {
            w.i32(12);
            w.f64(*x);
        }
        Value::Arr(kind, vals) => {
            let ty = arr_ty(kind);
            w.i32(9);
            w.i32(ty);
            w.u64(vals.len() as u64);
            for v in vals {
                write_payload(w, v);
            }
        }
    }
}

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("gguf_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write_file(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, bytes).unwrap();
    p
}

fn f16_tensor(name: &'static str, ne: Vec<i64>, fill: f32) -> TensorSpec {
    let n: usize = ne.iter().product::<i64>() as usize;
    let mut data = Vec::with_capacity(n * 2);
    for i in 0..n {
        let v = half::f16::from_f32(fill + (i % 7) as f32 * 0.25);
        data.extend(v.to_le_bytes());
    }
    TensorSpec {
        name,
        ne,
        ty: 1,
        data,
    }
}

#[test]
fn parses_v3_all_value_types() {
    let dir = tmpdir("v3");
    let kvs: Vec<(&str, Value)> = vec![
        ("u8", Value::U8(255)),
        ("i8", Value::I8(-5)),
        ("u16", Value::U16(65535)),
        ("i16", Value::I16(-300)),
        ("u32", Value::U32(4_000_000_000)),
        ("i32", Value::I32(-99)),
        ("f32", Value::F32(1.5)),
        ("bool", Value::Bool(true)),
        ("str", Value::Str("hello".into())),
        ("u64", Value::U64(u64::MAX)),
        ("i64", Value::I64(-1)),
        ("f64", Value::F64(2.5)),
        (
            "arr",
            Value::Arr(
                mil_convert::gguf::ArrKind::I32,
                vec![Value::I32(3), Value::I32(-7)],
            ),
        ),
        (
            "arrstr",
            Value::Arr(
                mil_convert::gguf::ArrKind::Str,
                vec![Value::Str("a".into()), Value::Str("bb".into())],
            ),
        ),
        ("general.architecture", Value::Str("qwen3".into())),
    ];
    let tensors = vec![f16_tensor("token_embd.weight", vec![4, 8], 1.0)];
    let bytes = make_gguf(&kvs, &tensors, 32, None);
    let p = write_file(&dir, "m.gguf", &bytes);
    let g = Gguf::open(&p).unwrap();
    assert_eq!(g.version, 3);
    assert_eq!(g.alignment, 32);
    assert_eq!(g.arch.as_deref(), Some("qwen3"));
    assert_eq!(g.meta("u8"), Some(&Value::U8(255)));
    assert_eq!(g.meta("i64"), Some(&Value::I64(-1)));
    assert_eq!(
        g.meta("arrstr").and_then(Value::as_arr).map(|a| a.len()),
        Some(2)
    );
    let t = g.tensor("token_embd.weight").unwrap();
    assert_eq!(t.ne, vec![4, 8]);
    assert_eq!(t.hf_shape(), vec![8, 4]);
    let (shape, vals) = g.tensor_f32("model.embed_tokens.weight").unwrap();
    assert_eq!(shape, vec![8, 4]);
    assert!((vals[0] - 1.0).abs() < 1e-3);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rejects_v1_and_bad_magic_and_endian() {
    let dir = tmpdir("badver");
    let good = make_gguf(&[], &[], 32, None);

    // v1
    let mut v1 = good.clone();
    v1[4..8].copy_from_slice(&1u32.to_le_bytes());
    let p = write_file(&dir, "v1.gguf", &v1);
    let e = Gguf::open(&p).unwrap_err().to_string();
    assert!(e.contains("v1"), "v1 err: {e}");

    // v4
    let mut v4 = good.clone();
    v4[4..8].copy_from_slice(&4u32.to_le_bytes());
    let p = write_file(&dir, "v4.gguf", &v4);
    assert!(Gguf::open(&p).unwrap_err().to_string().contains('4'));

    // byte-swapped
    let mut be = good.clone();
    be[4..8].copy_from_slice(&3u32.to_be_bytes());
    let p = write_file(&dir, "be.gguf", &be);
    let e = Gguf::open(&p).unwrap_err().to_string();
    assert!(e.contains("endian") || e.contains("swapped"), "be err: {e}");

    // bad magic
    let mut bm = good.clone();
    bm[0] = b'X';
    let p = write_file(&dir, "bm.gguf", &bm);
    assert!(Gguf::open(&p).unwrap_err().to_string().contains("magic"));

    // v2 header parses
    let mut v2 = good.clone();
    v2[4..8].copy_from_slice(&2u32.to_le_bytes());
    let p = write_file(&dir, "v2.gguf", &v2);
    assert_eq!(Gguf::open(&p).unwrap().version, 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn truncated_and_corrupt_never_panic() {
    let dir = tmpdir("fuzz");
    let kvs = vec![
        ("general.architecture", Value::Str("llama".into())),
        ("k", Value::Str("v".into())),
    ];
    let tensors = vec![
        f16_tensor("token_embd.weight", vec![8, 4], 0.5),
        f16_tensor("output_norm.weight", vec![4], 1.0),
    ];
    let good = make_gguf(&kvs, &tensors, 32, None);
    // sanity
    let p = write_file(&dir, "good.gguf", &good);
    assert_eq!(Gguf::open(&p).unwrap().tensors().count(), 2);

    // every truncation length must error, not panic
    for len in 0..good.len() {
        let p = write_file(&dir, "t.gguf", &good[..len]);
        assert!(Gguf::open(&p).is_err(), "len {len} should fail");
    }
    // flip every byte of the header region (one at a time)
    for i in 0..good.len().min(160) {
        let mut b = good.clone();
        b[i] ^= 0xFF;
        let p = write_file(&dir, "c.gguf", &b);
        let _ = Gguf::open(&p); // must not panic; result may be ok or err
    }
    // unaligned tensor offset
    let bad = make_gguf(&kvs, &tensors, 32, Some(24));
    let p = write_file(&dir, "unal.gguf", &bad);
    let e = Gguf::open(&p).unwrap_err().to_string();
    assert!(e.contains("aligned"), "unaligned err: {e}");
    // non-power-of-2 alignment
    let bad = make_gguf(
        &[("general.alignment", Value::U32(24))],
        &tensors,
        24,
        Some(0),
    );
    let p = write_file(&dir, "np2.gguf", &bad);
    assert!(Gguf::open(&p).unwrap_err().to_string().contains("power"));
    let _ = std::fs::remove_dir_all(&dir);
}

fn split_shard(no: u32, count: u32, tensors: Vec<TensorSpec>) -> Vec<u8> {
    make_gguf(
        &[
            ("split.no", Value::U16(no as u16)),
            ("split.count", Value::U16(count as u16)),
        ],
        &tensors,
        32,
        None,
    )
}

#[test]
fn split_files_unify() {
    let dir = tmpdir("split");
    let s1 = split_shard(0, 2, vec![f16_tensor("token_embd.weight", vec![4, 4], 1.0)]);
    let s2 = split_shard(1, 2, vec![f16_tensor("output.weight", vec![4, 4], 2.0)]);
    write_file(&dir, "m-00001-of-00002.gguf", &s1);
    write_file(&dir, "m-00002-of-00002.gguf", &s2);

    // opening the SECOND shard discovers all
    let g = Gguf::open(&dir.join("m-00002-of-00002.gguf")).unwrap();
    assert_eq!(g.tensors().count(), 2);
    assert!(g.tensor("token_embd.weight").is_some());
    assert!(g.tensor("output.weight").is_some());
    let (_s, vals) = g.tensor_f32("model.embed_tokens.weight").unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-3);

    // missing shard → error
    let dir2 = tmpdir("split2");
    write_file(&dir2, "m-00001-of-00002.gguf", &s1);
    let e = Gguf::open(&dir2.join("m-00001-of-00002.gguf"))
        .unwrap_err()
        .to_string();
    assert!(e.contains("missing split shard"), "err: {e}");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&dir2);
}

#[test]
fn type_ids_match_pinned_ggml() {
    // spot-check against ggml/include/ggml.h at the pinned commit
    assert_eq!(GgufType::from_id(0), GgufType::F32);
    assert_eq!(GgufType::from_id(2), GgufType::Q4_0);
    assert_eq!(GgufType::from_id(8), GgufType::Q8_0);
    assert_eq!(GgufType::from_id(10), GgufType::Q2K);
    assert_eq!(GgufType::from_id(20), GgufType::Iq4Nl);
    assert_eq!(GgufType::from_id(29), GgufType::Iq1M);
    assert_eq!(GgufType::from_id(30), GgufType::Bf16);
    assert_eq!(GgufType::from_id(34), GgufType::Tq1_0);
    assert_eq!(GgufType::from_id(39), GgufType::Mxfp4);
    assert_eq!(GgufType::from_id(40), GgufType::Nvfp4);
    assert_eq!(GgufType::from_id(41), GgufType::Q1_0);
    assert_eq!(GgufType::from_id(42), GgufType::Q2_0);
    assert!(matches!(GgufType::from_id(4), GgufType::Removed(4)));
    assert!(matches!(GgufType::from_id(31), GgufType::Removed(31)));
    assert!(matches!(GgufType::from_id(43), GgufType::Unknown(43)));
}
