//! cy4b_q8 — quantize CY4B safetensors F16 -> per-channel int8 + fp16
//! scales, safetensors out. 2-D weight matrices [N,K] quantize per
//! output row (scale = max|w|/127); 1-D tensors (norms) and the
//! embedding table stay F16. Format: `{name}` I8 [N,K] +
//! `{name}.qscale` F16 [N]; decoder dequantizes w[j,i]=q*s[j].
//!
//! Run: cargo run --release -p mil_infer --example cy4b_q8 -- \
//!        /Users/savag3/cy4b/cy4b_model/model.safetensors \
//!        /Users/savag3/cy4b/cy4b_q8.safetensors

use std::collections::HashMap;
use std::path::Path;

fn f32_of_f16(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let src = args.get(1).expect("usage: cy4b_q8 <in.safetensors> <out.safetensors>");
    let dst = args.get(2).expect("out path");

    let buf = std::fs::read(src).expect("read");
    let hlen = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&buf[8..8 + hlen]).expect("header");

    // pass 1: plan tensors — quantize 2-D F16 matrices with K >= 256,
    // pass through everything else verbatim.
    struct T {
        name: String,
        dtype: String,
        shape: Vec<usize>,
        bytes: Vec<u8>, // payload as stored in `buf`
    }
    let mut tensors = Vec::new();
    for (name, v) in header.as_object().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        let dt = v["dtype"].as_str().unwrap().to_string();
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let off = v["data_offsets"].as_array().unwrap();
        let (a, b) = (
            off[0].as_u64().unwrap() as usize + 8 + hlen,
            off[1].as_u64().unwrap() as usize + 8 + hlen,
        );
        tensors.push(T {
            name: name.clone(),
            dtype: dt,
            shape,
            bytes: buf[a..b].to_vec(),
        });
    }

    // build output: header JSON + concatenated payload
    let mut out_header: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    let mut payload: Vec<u8> = Vec::new();
    let mut cursor = 0u64;
    let mut put = |name: &str,
                   dtype: &str,
                   shape: &[usize],
                   data: &[u8],
                   out_header: &mut serde_json::Map<String, serde_json::Value>,
                   payload: &mut Vec<u8>,
                   cursor: &mut u64| {
        let a = *cursor;
        payload.extend_from_slice(data);
        *cursor += data.len() as u64;
        out_header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [a, *cursor],
            }),
        );
    };

    let mut n_q8 = 0usize;
    for t in &tensors {
        let is_mat = t.shape.len() == 2 && t.dtype == "F16" && t.shape[1] >= 256;
        if !is_mat {
            // pass through: rewrite dtype/shape verbatim
            let dt = match t.dtype.as_str() {
                "F16" => "F16",
                "F32" => "F32",
                "BF16" => "BF16",
                other => panic!("{}: unsupported dtype {other}", t.name),
            };
            put(&t.name, dt, &t.shape, &t.bytes, &mut out_header, &mut payload, &mut cursor);
            continue;
        }
        let (n, k) = (t.shape[0], t.shape[1]);
        // dequant source f16 -> f32 rows, quantize per row
        let mut q = vec![0i8; n * k];
        let mut scales = vec![0u8; n * 2];
        for j in 0..n {
            let row = &t.bytes[j * k * 2..(j + 1) * k * 2];
            let mut amax = 0f32;
            let mut vals = Vec::with_capacity(k);
            for i in 0..k {
                let v = f32_of_f16(u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]));
                vals.push(v);
                amax = amax.max(v.abs());
            }
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let inv = 1.0 / s;
            for (i, &v) in vals.iter().enumerate() {
                let c = (v * inv).round().clamp(-127.0, 127.0) as i8;
                q[j * k + i] = c;
            }
            scales[j * 2..j * 2 + 2]
                .copy_from_slice(&half::f16::from_f32(s).to_bits().to_le_bytes());
        }
        let qbytes: Vec<u8> = q.iter().map(|&c| c as u8).collect();
        put(&t.name, "I8", &t.shape, &qbytes, &mut out_header, &mut payload, &mut cursor);
        put(
            &format!("{}.qscale", t.name),
            "F16",
            &[n],
            &scales,
            &mut out_header,
            &mut payload,
            &mut cursor,
        );
        n_q8 += 1;
        if n_q8 % 40 == 0 {
            eprintln!("q8: {n_q8} tensors ({:.1} GB payload)", payload.len() as f64 / 1e9);
        }
    }

    let hjson = serde_json::to_vec(&serde_json::Value::Object(out_header)).unwrap();
    let mut out = Vec::with_capacity(8 + hjson.len() + payload.len());
    out.extend_from_slice(&(hjson.len() as u64).to_le_bytes());
    out.extend_from_slice(&hjson);
    out.extend_from_slice(&payload);
    std::fs::write(Path::new(dst), &out).expect("write");
    println!(
        "wrote {dst}: {} tensors, {n_q8} quantized, {:.2} GB",
        tensors.len(),
        out.len() as f64 / 1e9
    );
    let _ = HashMap::<String, String>::new();
}
