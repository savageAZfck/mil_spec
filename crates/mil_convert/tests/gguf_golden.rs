//! Bit-exact dequantization proof against ggml's `to_float`.
//!
//! `tests/gguf_golden/vectors/<type>.{q,f32}` was produced by
//! `tests/gguf_golden/gen.c` linked against the pinned llama.cpp
//! commit — see `tests/gguf_golden/README.md`. Every type must match
//! bit-for-bit; there is no tolerance knob by design.

use mil_convert::gguf::{dequant, GgufType};
use std::path::PathBuf;

const NROWS: usize = 4;
const NPER: usize = 512;

fn vec_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/gguf_golden/vectors")
}

fn cases() -> Vec<(&'static str, GgufType)> {
    vec![
        ("f16", GgufType::F16),
        ("bf16", GgufType::Bf16),
        ("q1_0", GgufType::Q1_0),
        ("q2_0", GgufType::Q2_0),
        ("q4_0", GgufType::Q4_0),
        ("q4_1", GgufType::Q4_1),
        ("q5_0", GgufType::Q5_0),
        ("q5_1", GgufType::Q5_1),
        ("q8_0", GgufType::Q8_0),
        ("mxfp4", GgufType::Mxfp4),
        ("nvfp4", GgufType::Nvfp4),
        ("tq1_0", GgufType::Tq1_0),
        ("tq2_0", GgufType::Tq2_0),
        ("q2_K", GgufType::Q2K),
        ("q3_K", GgufType::Q3K),
        ("q4_K", GgufType::Q4K),
        ("q5_K", GgufType::Q5K),
        ("q6_K", GgufType::Q6K),
        ("iq1_s", GgufType::Iq1S),
        ("iq1_m", GgufType::Iq1M),
        ("iq2_xxs", GgufType::Iq2Xxs),
        ("iq2_xs", GgufType::Iq2Xs),
        ("iq2_s", GgufType::Iq2S),
        ("iq3_xxs", GgufType::Iq3Xxs),
        ("iq3_s", GgufType::Iq3S),
        ("iq4_nl", GgufType::Iq4Nl),
        ("iq4_xs", GgufType::Iq4Xs),
    ]
}

#[test]
fn golden_dequant_bit_exact() {
    let dir = vec_dir();
    let mut checked = 0;
    for (name, ty) in cases() {
        let q = std::fs::read(dir.join(format!("{name}.q")))
            .unwrap_or_else(|e| panic!("{name}: read .q — {e} (run gen.c, see README)"));
        let f = std::fs::read(dir.join(format!("{name}.f32")))
            .unwrap_or_else(|e| panic!("{name}: read .f32 — {e}"));
        let row_bytes = q.len() / NROWS;
        assert_eq!(
            q.len() % NROWS,
            0,
            "{name}: .q size {} not divisible by {NROWS} rows",
            q.len()
        );
        assert_eq!(
            f.len(),
            NROWS * NPER * 4,
            "{name}: .f32 size {} != {NROWS}×{NPER}×4",
            f.len()
        );
        let expect: Vec<f32> = f
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut got = vec![0f32; NROWS * NPER];
        for r in 0..NROWS {
            let src = &q[r * row_bytes..(r + 1) * row_bytes];
            let dst = &mut got[r * NPER..(r + 1) * NPER];
            dequant::dequant(ty, src, dst).unwrap_or_else(|e| panic!("{name} row {r}: {e}"));
        }

        let mut bad = usize::MAX;
        for (i, (a, b)) in got.iter().zip(expect.iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                bad = i;
                break;
            }
        }
        assert_eq!(
            bad,
            usize::MAX,
            "{name}: element {bad} differs — got {:#010x} expected {:#010x} \
             ({} vs {})",
            got[bad].to_bits(),
            expect[bad].to_bits(),
            got[bad],
            expect[bad]
        );
        checked += 1;
    }
    assert_eq!(checked, 27, "expected 27 golden type vectors");
}
