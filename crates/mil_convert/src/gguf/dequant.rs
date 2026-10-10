//! Scalar dequantization kernels — every ggml block type with a `to_float`
//! at the pinned commit, ported statement-for-statement so the output is
//! bit-exact with the C reference.
//!
//! Ported from ggml (MIT), llama.cpp commit 10a60cf303566e10d6a7a2774c17d2085503d87b
//! (ggml/src/ggml-quants.c, ggml/src/ggml-common.h, ggml/src/ggml-impl.h).
//! Golden-tested bit-exact — see `tests/gguf_golden/`.
//
// Copyright (c) 2023-2024 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

use super::tables::*;
use super::GgufType;

const QK_K: usize = 256;
const IQ1S_DELTA: f32 = 0.125;

/// Error produced when a quantized blob is malformed.
#[derive(Debug)]
pub struct DequantError(pub String);

impl std::fmt::Display for DequantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for DequantError {}

#[inline]
fn fp16(x: u16) -> f32 {
    half::f16::from_bits(x).to_f32()
}

/// ggml `GGML_FP16_TO_FP32` — reads an f16 little-endian scalar field.
#[inline]
fn f16_at(buf: &[u8], off: usize) -> f32 {
    fp16(u16::from_le_bytes([buf[off], buf[off + 1]]))
}

/// `ggml_e8m0_to_fp32_half` — E8M0 scale decode, ÷2 variant (ggml-impl.h).
#[inline]
fn e8m0_to_fp32_half(x: u8) -> f32 {
    let bits: u32 = if x < 2 {
        0x0020_0000u32 << x
    } else {
        (x as u32 - 1) << 23
    };
    f32::from_bits(bits)
}

/// `ggml_ue4m3_to_fp32` — unsigned E4M3 (bias 7), returns raw*0.5 (ggml-impl.h).
#[inline]
fn ue4m3_to_fp32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = (x >> 3) & 0xF;
    let man = x & 0x7;
    let raw = if exp == 0 {
        (man as f32) * (2f32).powi(-9)
    } else {
        (1.0 + man as f32 / 8.0) * (2f32).powi(exp as i32 - 7)
    };
    raw * 0.5
}

/// Elements per block for a type.
pub fn block_len(t: GgufType) -> usize {
    match t {
        GgufType::Q1_0 => 128,
        GgufType::Q2_0 => 64,
        GgufType::Mxfp4 | GgufType::Iq4Nl => 32,
        GgufType::Nvfp4 => 64,
        GgufType::Q4_0
        | GgufType::Q4_1
        | GgufType::Q5_0
        | GgufType::Q5_1
        | GgufType::Q8_0
        | GgufType::Q8_1 => 32,
        GgufType::Q2K
        | GgufType::Q3K
        | GgufType::Q4K
        | GgufType::Q5K
        | GgufType::Q6K
        | GgufType::Q8K
        | GgufType::Iq2Xxs
        | GgufType::Iq2Xs
        | GgufType::Iq2S
        | GgufType::Iq3Xxs
        | GgufType::Iq3S
        | GgufType::Iq1S
        | GgufType::Iq1M
        | GgufType::Iq4Xs
        | GgufType::Tq1_0
        | GgufType::Tq2_0 => QK_K,
        _ => 1,
    }
}

/// Bytes per block for a type (raw element size for scalar types).
pub fn block_size(t: GgufType) -> usize {
    match t {
        GgufType::F32 => 4,
        GgufType::F16 | GgufType::Bf16 | GgufType::I16 => 2,
        GgufType::I8 => 1,
        GgufType::I32 => 4,
        GgufType::I64 | GgufType::F64 => 8,
        GgufType::Q1_0 => 2 + 16,
        GgufType::Q2_0 => 2 + 16,
        GgufType::Q4_0 => 2 + 16,
        GgufType::Q4_1 => 4 + 16,
        GgufType::Q5_0 => 2 + 4 + 16,
        GgufType::Q5_1 => 4 + 4 + 16,
        GgufType::Q8_0 => 2 + 32,
        GgufType::Q8_1 => 4 + 32,
        GgufType::Mxfp4 => 1 + 16,
        GgufType::Nvfp4 => 4 + 32,
        GgufType::Tq1_0 => 48 + 4 + 2,
        GgufType::Tq2_0 => 64 + 2,
        GgufType::Q2K => 16 + 64 + 4,
        GgufType::Q3K => 32 + 64 + 12 + 2,
        GgufType::Q4K => 4 + 12 + 128,
        GgufType::Q5K => 4 + 12 + 32 + 128,
        GgufType::Q6K => 128 + 64 + 16 + 2,
        GgufType::Q8K => 4 + 256 + 32,
        GgufType::Iq2Xxs => 2 + 64,
        GgufType::Iq2Xs => 2 + 64 + 8,
        GgufType::Iq2S => 2 + 64 + 8 + 8,
        GgufType::Iq3Xxs => 2 + 96,
        GgufType::Iq3S => 2 + 64 + 8 + 32 + 4,
        GgufType::Iq1S => 2 + 32 + 16,
        GgufType::Iq1M => 32 + 16 + 8,
        GgufType::Iq4Nl => 2 + 16,
        GgufType::Iq4Xs => 2 + 2 + 4 + 128,
        GgufType::Removed(id) | GgufType::Unknown(id) => {
            debug_assert!(id > 0, "removed/unknown type has no size");
            0
        }
    }
}

/// ggml `dequantize_row_q8_0` — 8-bit delta blocks.
fn dequant_q8_0(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for j in 0..32 {
        y[j] = (b[2 + j] as i8) as f32 * d;
    }
}

fn dequant_q4_0(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for j in 0..16 {
        let x0 = (b[2 + j] & 0x0F) as i32 - 8;
        let x1 = (b[2 + j] >> 4) as i32 - 8;
        y[j] = x0 as f32 * d;
        y[j + 16] = x1 as f32 * d;
    }
}

fn dequant_q4_1(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let m = f16_at(b, 2);
    for j in 0..16 {
        let x0 = (b[4 + j] & 0x0F) as i32;
        let x1 = (b[4 + j] >> 4) as i32;
        y[j] = x0 as f32 * d + m;
        y[j + 16] = x1 as f32 * d + m;
    }
}

fn dequant_q5_0(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    for j in 0..16 {
        let xh_0 = ((qh >> j) << 4) & 0x10;
        let xh_1 = (qh >> (j + 12)) & 0x10;
        let x0 = ((b[6 + j] as u32 & 0x0F | xh_0) as i32 - 16) as f32;
        let x1 = ((b[6 + j] as u32 >> 4 | xh_1) as i32 - 16) as f32;
        y[j] = x0 * d;
        y[j + 16] = x1 * d;
    }
}

fn dequant_q5_1(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let m = f16_at(b, 2);
    let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    for j in 0..16 {
        let xh_0 = ((qh >> j) << 4) & 0x10;
        let xh_1 = (qh >> (j + 12)) & 0x10;
        let x0 = (b[8 + j] as u32 & 0x0F | xh_0) as i32;
        let x1 = (b[8 + j] as u32 >> 4 | xh_1) as i32;
        y[j] = x0 as f32 * d + m;
        y[j + 16] = x1 as f32 * d + m;
    }
}

fn dequant_q1_0(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let neg_d = -d;
    for j in 0..128 {
        let bit = (b[2 + j / 8] >> (j % 8)) & 1;
        y[j] = if bit != 0 { d } else { neg_d };
    }
}

fn dequant_q2_0(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for j in 0..64 {
        let q = (b[2 + j / 4] >> ((j % 4) * 2)) & 0x03;
        // 00=-1, 01=0, 10=+1, 11=+2
        y[j] = (q as i32 - 1) as f32 * d;
    }
}

fn dequant_mxfp4(b: &[u8], y: &mut [f32]) {
    let d = e8m0_to_fp32_half(b[0]);
    for j in 0..16 {
        let x0 = KVALUES_FP4[(b[1 + j] & 0x0F) as usize] as f32;
        let x1 = KVALUES_FP4[(b[1 + j] >> 4) as usize] as f32;
        y[j] = x0 * d;
        y[j + 16] = x1 * d;
    }
}

fn dequant_nvfp4(b: &[u8], y: &mut [f32]) {
    // block_nvfp4: d[4] sub-block UE4M3 scales, qs[32] packed E2M1
    for s in 0..4 {
        let d = ue4m3_to_fp32(b[s]);
        let yb = &mut y[s * 16..];
        for j in 0..8 {
            let v0 = KVALUES_FP4[(b[4 + s * 8 + j] & 0x0F) as usize] as f32;
            let v1 = KVALUES_FP4[(b[4 + s * 8 + j] >> 4) as usize] as f32;
            yb[j] = v0 * d;
            yb[j + 8] = v1 * d;
        }
    }
}

fn dequant_tq1_0(b: &[u8], y: &mut [f32]) {
    // block_tq1_0: qs[48], qh[4], d — 256 values
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    let d = f16_at(b, 52);
    let qs = &b[0..48];
    let qh = &b[48..52];
    let mut out = 0;
    for j in (0..48 - 48 % 32).step_by(32) {
        for &p3 in POW3.iter().take(5) {
            for m in 0..32 {
                let q = qs[j + m].wrapping_mul(p3);
                let xi = ((q as u16 * 3) >> 8) as i16;
                y[out] = (xi - 1) as f32 * d;
                out += 1;
            }
        }
    }
    for j in (48 - 48 % 32..48).step_by(16) {
        for &p3 in POW3.iter().take(5) {
            for m in 0..16 {
                let q = qs[j + m].wrapping_mul(p3);
                let xi = ((q as u16 * 3) >> 8) as i16;
                y[out] = (xi - 1) as f32 * d;
                out += 1;
            }
        }
    }
    for &p3 in POW3.iter().take(4) {
        for &qhb in qh.iter().take(4) {
            let q = qhb.wrapping_mul(p3);
            let xi = ((q as u16 * 3) >> 8) as i16;
            y[out] = (xi - 1) as f32 * d;
            out += 1;
        }
    }
}

fn dequant_tq2_0(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 64);
    let mut out = 0;
    for j in (0..64).step_by(32) {
        for l in 0..4 {
            for m in 0..32 {
                let q = (b[j + m] >> (l * 2)) & 3;
                y[out] = (q as i8 - 1) as f32 * d;
                out += 1;
            }
        }
    }
}

fn dequant_q2_k(b: &[u8], y: &mut [f32]) {
    // scales[16] qs[64] d dmin
    let scales = &b[0..16];
    let d = f16_at(b, 80);
    let min = f16_at(b, 82);
    let mut is = 0usize;
    let mut out = 0usize;
    for n in 0..2 {
        let q = &b[16 + 32 * n..16 + 32 * n + 32];
        let mut shift = 0u32;
        for _j in 0..4 {
            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0xF) as f32;
            let ml = min * (sc >> 4) as f32;
            for &qv in q.iter().take(16) {
                y[out] = dl * ((qv >> shift) & 3) as f32 - ml;
                out += 1;
            }
            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0xF) as f32;
            let ml = min * (sc >> 4) as f32;
            for l in 0..16 {
                y[out] = dl * ((q[l + 16] >> shift) & 3) as f32 - ml;
                out += 1;
            }
            shift += 2;
        }
    }
    debug_assert_eq!(out, 256);
}

fn dequant_q3_k(b: &[u8], y: &mut [f32]) {
    // hmask[32] qs[64] scales[12] d
    let hm = &b[0..32];
    let mut q_off = 32usize;
    let scales_raw = &b[96..108];
    let d_all = f16_at(b, 108);

    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;
    let mut aux = [0u32; 4];
    for i in 0..3 {
        aux[i] = u32::from_le_bytes([
            scales_raw[4 * i],
            scales_raw[4 * i + 1],
            scales_raw[4 * i + 2],
            scales_raw[4 * i + 3],
        ]);
    }
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let mut scales = [0i8; 16];
    for i in 0..16 {
        scales[i] = aux[i / 4].to_le_bytes()[i % 4] as i8;
    }

    let mut is = 0usize;
    let mut m = 1u8;
    let mut out = 0usize;
    for _n in 0..2 {
        let mut shift = 0u32;
        for _j in 0..4 {
            let q = &b[q_off..q_off + 32];
            let dl = d_all * (scales[is] as i32 - 32) as f32;
            is += 1;
            for l in 0..16 {
                let hi = if hm[l] & m != 0 { 0i32 } else { 4 };
                y[out] = dl * (((q[l] >> shift) & 3) as i32 - hi) as f32;
                out += 1;
            }
            let dl = d_all * (scales[is] as i32 - 32) as f32;
            is += 1;
            for l in 0..16 {
                let hi = if hm[l + 16] & m != 0 { 0i32 } else { 4 };
                y[out] = dl * (((q[l + 16] >> shift) & 3) as i32 - hi) as f32;
                out += 1;
            }
            shift += 2;
            m <<= 1;
        }
        q_off += 32;
    }
    debug_assert_eq!(out, 256);
}

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn dequant_q4_k(b: &[u8], y: &mut [f32]) {
    // d dmin scales[12] qs[128]
    let d = f16_at(b, 0);
    let min = f16_at(b, 2);
    let scales = &b[4..16];
    let mut q = 16usize;
    let mut is = 0usize;
    let mut out = 0usize;
    for _j in 0..4 {
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = min * m as f32;
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = min * m as f32;
        for l in 0..32 {
            y[out] = d1 * (b[q + l] & 0xF) as f32 - m1;
            out += 1;
        }
        for l in 0..32 {
            y[out] = d2 * (b[q + l] >> 4) as f32 - m2;
            out += 1;
        }
        q += 32;
        is += 2;
    }
    debug_assert_eq!(out, 256);
}

fn dequant_q5_k(b: &[u8], y: &mut [f32]) {
    // d dmin scales[12] qh[32] qs[128]
    let d = f16_at(b, 0);
    let min = f16_at(b, 2);
    let scales = &b[4..16];
    let qh = &b[16..48];
    let mut ql = 48usize;
    let mut is = 0usize;
    let mut u1 = 1u8;
    let mut u2 = 2u8;
    let mut out = 0usize;
    for _j in 0..4 {
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = min * m as f32;
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = min * m as f32;
        for l in 0..32 {
            let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
            y[out] = d1 * ((b[ql + l] & 0xF) + hi) as f32 - m1;
            out += 1;
        }
        for l in 0..32 {
            let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
            y[out] = d2 * ((b[ql + l] >> 4) + hi) as f32 - m2;
            out += 1;
        }
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
    debug_assert_eq!(out, 256);
}

fn dequant_q6_k(b: &[u8], y: &mut [f32]) {
    // ql[128] qh[64] scales[16](i8) d
    let d = f16_at(b, 208);
    let mut ql = 0usize;
    let mut qh = 128usize;
    let mut sc = 192usize;
    let mut out = 0usize;
    for _n in 0..2 {
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((b[ql + l] & 0xF) | ((b[qh + l] & 3) << 4)) as i8 as i32 - 32;
            let q2 = ((b[ql + l + 32] & 0xF) | (((b[qh + l] >> 2) & 3) << 4)) as i8 as i32 - 32;
            let q3 = ((b[ql + l] >> 4) | (((b[qh + l] >> 4) & 3) << 4)) as i8 as i32 - 32;
            let q4 = ((b[ql + l + 32] >> 4) | (((b[qh + l] >> 6) & 3) << 4)) as i8 as i32 - 32;
            y[l + out] = d * (b[sc + is] as i8) as f32 * q1 as f32;
            y[l + 32 + out] = d * (b[sc + is + 2] as i8) as f32 * q2 as f32;
            y[l + 64 + out] = d * (b[sc + is + 4] as i8) as f32 * q3 as f32;
            y[l + 96 + out] = d * (b[sc + is + 6] as i8) as f32 * q4 as f32;
        }
        out += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

fn dequant_q8_k(b: &[u8], y: &mut [f32]) {
    let d = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    for j in 0..256 {
        y[j] = d * (b[4 + j] as i8) as f32;
    }
}

fn dequant_iq2_xxs(b: &[u8], y: &mut [f32]) {
    // d qs[64] (u16 × 32)
    let d = f16_at(b, 0);
    let mut out = 0usize;
    for ib32 in 0..8 {
        let base = 2 + 8 * ib32;
        let aux0 = u32::from_le_bytes([b[base], b[base + 1], b[base + 2], b[base + 3]]);
        let aux1 = u32::from_le_bytes([b[base + 4], b[base + 5], b[base + 6], b[base + 7]]);
        let aux8: [u8; 8] = [
            (aux0 & 0xFF) as u8,
            ((aux0 >> 8) & 0xFF) as u8,
            ((aux0 >> 16) & 0xFF) as u8,
            ((aux0 >> 24) & 0xFF) as u8,
            (aux1 & 0xFF) as u8,
            ((aux1 >> 8) & 0xFF) as u8,
            ((aux1 >> 16) & 0xFF) as u8,
            ((aux1 >> 24) & 0xFF) as u8,
        ];
        let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
        for l in 0..4 {
            let grid = IQ2XXS_GRID[aux8[l] as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                y[out] = db
                    * grid[j] as f32
                    * if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                out += 1;
            }
        }
    }
}

fn dequant_iq2_xs(b: &[u8], y: &mut [f32]) {
    // d qs[64](u16 × 32) scales[8]
    let d = f16_at(b, 0);
    let scales = &b[66..74];
    let mut out = 0usize;
    for ib32 in 0..8 {
        let db = [
            d * (0.5 + (scales[ib32] & 0xf) as f32) * 0.25,
            d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let idx = u16::from_le_bytes([b[2 + 8 * ib32 + 2 * l], b[2 + 8 * ib32 + 2 * l + 1]]);
            let grid = IQ2XS_GRID[(idx & 511) as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[(idx >> 9) as usize];
            for j in 0..8 {
                y[out] = db[l / 2]
                    * grid[j] as f32
                    * if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                out += 1;
            }
        }
    }
}

fn dequant_iq2_s(b: &[u8], y: &mut [f32]) {
    // d qs[64] qh[8] scales[8]; grid idxs in qs[0..32], signs in qs[32..64]
    let d = f16_at(b, 0);
    let scales = &b[74..82];
    let mut out = 0usize;
    for ib32 in 0..8 {
        let db = [
            d * (0.5 + (scales[ib32] & 0xf) as f32) * 0.25,
            d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let dl = db[l / 2];
            let idx =
                (b[2 + 4 * ib32 + l] as usize) | (((b[66 + ib32] as usize) << (8 - 2 * l)) & 0x300);
            let grid = IQ2S_GRID[idx].to_le_bytes();
            let signs = b[34 + 4 * ib32 + l];
            for j in 0..8 {
                y[out] = dl
                    * grid[j] as f32
                    * if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                out += 1;
            }
        }
    }
}

fn dequant_iq3_xxs(b: &[u8], y: &mut [f32]) {
    // d qs[96]: qs[0..64] grid bytes (2 per l), scales_and_signs qs[64..96]
    let d = f16_at(b, 0);
    let mut qs = 2usize;
    let sas = 66usize;
    let mut out = 0usize;
    for ib32 in 0..8 {
        let aux = u32::from_le_bytes([
            b[sas + 4 * ib32],
            b[sas + 4 * ib32 + 1],
            b[sas + 4 * ib32 + 2],
            b[sas + 4 * ib32 + 3],
        ]);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            let grid1 = IQ3XXS_GRID[b[qs + 2 * l] as usize].to_le_bytes();
            let grid2 = IQ3XXS_GRID[b[qs + 2 * l + 1] as usize].to_le_bytes();
            for j in 0..4 {
                y[out + j] = db
                    * grid1[j] as f32
                    * if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                y[out + j + 4] = db
                    * grid2[j] as f32
                    * if signs & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
            }
            out += 8;
        }
        qs += 8;
    }
}

fn dequant_iq3_s(b: &[u8], y: &mut [f32]) {
    // d qs[64] qh[8] signs[32] scales[4]
    let d = f16_at(b, 0);
    let mut qs = 2usize;
    let mut qh = 66usize;
    let mut signs = 74usize;
    let scales = &b[106..110];
    let mut out = 0usize;
    for ib32 in (0..8).step_by(2) {
        let db1 = d * (1 + 2 * ((scales[ib32 / 2] & 0xf) as i32)) as f32;
        let db2 = d * (1 + 2 * ((scales[ib32 / 2] >> 4) as i32)) as f32;
        for l in 0..4 {
            let g1 = IQ3S_GRID[(b[qs + 2 * l] as usize) | ((b[qh] as usize) << (8 - 2 * l) & 256)]
                .to_le_bytes();
            let g2 = IQ3S_GRID
                [(b[qs + 2 * l + 1] as usize) | ((b[qh] as usize) << (7 - 2 * l) & 256)]
                .to_le_bytes();
            for j in 0..4 {
                y[out + j] = db1
                    * g1[j] as f32
                    * if b[signs + l] & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                y[out + j + 4] = db1
                    * g2[j] as f32
                    * if b[signs + l] & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
            }
            out += 8;
        }
        qs += 8;
        signs += 4;
        for l in 0..4 {
            let g1 = IQ3S_GRID
                [(b[qs + 2 * l] as usize) | ((b[qh + 1] as usize) << (8 - 2 * l) & 256)]
                .to_le_bytes();
            let g2 = IQ3S_GRID
                [(b[qs + 2 * l + 1] as usize) | ((b[qh + 1] as usize) << (7 - 2 * l) & 256)]
                .to_le_bytes();
            for j in 0..4 {
                y[out + j] = db2
                    * g1[j] as f32
                    * if b[signs + l] & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                y[out + j + 4] = db2
                    * g2[j] as f32
                    * if b[signs + l] & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
            }
            out += 8;
        }
        qh += 2;
        qs += 8;
        signs += 4;
    }
}

fn dequant_iq1_s(b: &[u8], y: &mut [f32]) {
    // d qs[32] qh[16] (u16 × 8)
    let d = f16_at(b, 0);
    let mut qs = 2usize;
    let mut out = 0usize;
    for ib in 0..8 {
        let qhi = u16::from_le_bytes([b[34 + 2 * ib], b[34 + 2 * ib + 1]]);
        let dl = d * (2 * ((qhi >> 12) & 7) as i32 + 1) as f32;
        let delta = if qhi & 0x8000 != 0 {
            -IQ1S_DELTA
        } else {
            IQ1S_DELTA
        };
        for l in 0..4 {
            let idx = (b[qs + l] as usize) | ((((qhi >> (3 * l)) & 7) as usize) << 8);
            let grid = IQ1S_GRID[idx].to_le_bytes();
            for &gv in &grid {
                y[out] = dl * (gv as i8 as f32 + delta);
                out += 1;
            }
        }
        qs += 4;
    }
}

fn dequant_iq1_m(b: &[u8], y: &mut [f32]) {
    // qs[32] qh[16] scales[8]; sc = scales as u16[4]
    let scales = &b[48..56];
    let sc = [
        u16::from_le_bytes([scales[0], scales[1]]),
        u16::from_le_bytes([scales[2], scales[3]]),
        u16::from_le_bytes([scales[4], scales[5]]),
        u16::from_le_bytes([scales[6], scales[7]]),
    ];
    let scale_u16 =
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    let d = fp16(scale_u16);
    let mut qs = 0usize;
    let mut qh = 32usize;
    let mut out = 0usize;
    for ib in 0..8 {
        let dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2))) & 0x7) as i32 + 1) as f32;
        let dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) as i32 + 1) as f32;
        let idx = [
            b[qs] as usize | ((b[qh] as usize) << 8 & 0x700),
            b[qs + 1] as usize | ((b[qh] as usize) << 4 & 0x700),
            b[qs + 2] as usize | ((b[qh + 1] as usize) << 8 & 0x700),
            b[qs + 3] as usize | ((b[qh + 1] as usize) << 4 & 0x700),
        ];
        let delta = [
            if b[qh] & 0x08 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if b[qh] & 0x80 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if b[qh + 1] & 0x08 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if b[qh + 1] & 0x80 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
        ];
        for l in 0..2 {
            let grid = IQ1S_GRID[idx[l]].to_le_bytes();
            for &gv in &grid {
                y[out] = dl1 * (gv as i8 as f32 + delta[l]);
                out += 1;
            }
        }
        for l in 2..4 {
            let grid = IQ1S_GRID[idx[l]].to_le_bytes();
            for &gv in &grid {
                y[out] = dl2 * (gv as i8 as f32 + delta[l]);
                out += 1;
            }
        }
        qs += 4;
        qh += 2;
    }
}

fn dequant_iq4_nl(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for j in 0..16 {
        y[j] = d * KVALUES_IQ4NL[(b[2 + j] & 0xf) as usize] as f32;
        y[j + 16] = d * KVALUES_IQ4NL[(b[2 + j] >> 4) as usize] as f32;
    }
}

fn dequant_iq4_xs(b: &[u8], y: &mut [f32]) {
    // d scales_h(u16) scales_l[4] qs[128]
    let d = f16_at(b, 0);
    let scales_h = u16::from_le_bytes([b[2], b[3]]);
    let scales_l = &b[4..8];
    let mut qs = 8usize;
    let mut out = 0usize;
    for ib in 0..8 {
        let ls = (((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as u16
            | (((scales_h >> (2 * ib)) & 3) << 4)) as i32;
        let dl = d * (ls - 32) as f32;
        for j in 0..16 {
            y[out + j] = dl * KVALUES_IQ4NL[(b[qs + j] & 0xf) as usize] as f32;
            y[out + j + 16] = dl * KVALUES_IQ4NL[(b[qs + j] >> 4) as usize] as f32;
        }
        out += 32;
        qs += 16;
    }
}

/// Dequantize `n` elements of type `t` from `src` into `dst`.
///
/// `src` must be exactly the quantized row bytes (`n / block_len` blocks);
/// `dst` must hold `n` floats. Errors on a size mismatch — callers slice to
/// exact size first so this never reads out of bounds.
pub fn dequant(t: GgufType, src: &[u8], dst: &mut [f32]) -> Result<(), DequantError> {
    let n = dst.len();
    let qk = block_len(t);
    let bs = block_size(t);
    if qk == 0 || bs == 0 {
        return Err(DequantError(format!("{t} has no dequantizer")));
    }
    if n % qk != 0 {
        return Err(DequantError(format!(
            "{t}: element count {n} not a multiple of block {qk}"
        )));
    }
    let nb = n / qk;
    if src.len() != nb * bs {
        return Err(DequantError(format!(
            "{t}: byte count {} != {} blocks × {} bytes",
            src.len(),
            nb,
            bs
        )));
    }
    match t {
        GgufType::F32 => {
            for (o, c) in dst.iter_mut().zip(src.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        GgufType::F16 => {
            for (o, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                *o = fp16(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        GgufType::Bf16 => {
            for (o, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            }
        }
        GgufType::F64 => {
            for (o, c) in dst.iter_mut().zip(src.chunks_exact(8)) {
                *o = f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32;
            }
        }
        GgufType::I8 => {
            for (o, &v) in dst.iter_mut().zip(src.iter()) {
                *o = (v as i8) as f32;
            }
        }
        GgufType::I16 => {
            for (o, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                *o = i16::from_le_bytes([c[0], c[1]]) as f32;
            }
        }
        GgufType::I32 => {
            for (o, c) in dst.iter_mut().zip(src.chunks_exact(4)) {
                *o = i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32;
            }
        }
        GgufType::I64 => {
            for (o, c) in dst.iter_mut().zip(src.chunks_exact(8)) {
                *o = i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32;
            }
        }
        _ => {
            for (i, blk) in src.chunks_exact(bs).enumerate() {
                let y = &mut dst[i * qk..(i + 1) * qk];
                match t {
                    GgufType::Q1_0 => dequant_q1_0(blk, y),
                    GgufType::Q2_0 => dequant_q2_0(blk, y),
                    GgufType::Q4_0 => dequant_q4_0(blk, y),
                    GgufType::Q4_1 => dequant_q4_1(blk, y),
                    GgufType::Q5_0 => dequant_q5_0(blk, y),
                    GgufType::Q5_1 => dequant_q5_1(blk, y),
                    GgufType::Q8_0 => dequant_q8_0(blk, y),
                    GgufType::Mxfp4 => dequant_mxfp4(blk, y),
                    GgufType::Nvfp4 => dequant_nvfp4(blk, y),
                    GgufType::Tq1_0 => dequant_tq1_0(blk, y),
                    GgufType::Tq2_0 => dequant_tq2_0(blk, y),
                    GgufType::Q2K => dequant_q2_k(blk, y),
                    GgufType::Q3K => dequant_q3_k(blk, y),
                    GgufType::Q4K => dequant_q4_k(blk, y),
                    GgufType::Q5K => dequant_q5_k(blk, y),
                    GgufType::Q6K => dequant_q6_k(blk, y),
                    GgufType::Q8K => dequant_q8_k(blk, y),
                    GgufType::Iq2Xxs => dequant_iq2_xxs(blk, y),
                    GgufType::Iq2Xs => dequant_iq2_xs(blk, y),
                    GgufType::Iq2S => dequant_iq2_s(blk, y),
                    GgufType::Iq3Xxs => dequant_iq3_xxs(blk, y),
                    GgufType::Iq3S => dequant_iq3_s(blk, y),
                    GgufType::Iq1S => dequant_iq1_s(blk, y),
                    GgufType::Iq1M => dequant_iq1_m(blk, y),
                    GgufType::Iq4Nl => dequant_iq4_nl(blk, y),
                    GgufType::Iq4Xs => dequant_iq4_xs(blk, y),
                    GgufType::Q8_1 => {
                        return Err(DequantError(
                            "Q8_1 has no to_float in ggml — intermediate type".into(),
                        ))
                    }
                    GgufType::Removed(_) | GgufType::Unknown(_) => {
                        return Err(DequantError(format!("{t}: removed/unsupported type")))
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
    Ok(())
}
