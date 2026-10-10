//! `milc convert --seq-range` / `milc reshape --seq-range` CLI behaviour
//! on a tiny synthetic Qwen3 checkpoint. Numerical equivalence of the
//! resulting packages is covered by `mil_convert`'s `enum_shapes_e2e`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct Tmp(PathBuf);
impl Tmp {
    fn new(tag: &str) -> Tmp {
        let p = std::env::temp_dir().join(format!("milc_sr_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Tmp(p)
    }
}
impl Drop for Tmp {
    fn drop(&mut self) {
        if std::env::var("MIL_KEEP_ARTIFACTS").as_deref() != Ok("1") {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn milc(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_milc"))
        .args(args)
        .output()
        .expect("spawn milc")
}
fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}
fn p(path: &Path) -> &str {
    path.to_str().unwrap()
}

fn tiny_qwen3(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type":"qwen3","hidden_size":16,"num_hidden_layers":1,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,
            "intermediate_size":32,"vocab_size":64,"rms_norm_eps":1e-6,
            "rope_theta":1000000.0,"max_position_embeddings":512,
            "tie_word_embeddings":false}"#,
    )
    .unwrap();
    let l = "model.layers.0";
    let specs: Vec<(String, Vec<i64>)> = vec![
        (format!("{l}.input_layernorm.weight"), vec![16]),
        (format!("{l}.post_attention_layernorm.weight"), vec![16]),
        (format!("{l}.self_attn.q_proj.weight"), vec![16, 16]),
        (format!("{l}.self_attn.k_proj.weight"), vec![8, 16]),
        (format!("{l}.self_attn.v_proj.weight"), vec![8, 16]),
        (format!("{l}.self_attn.o_proj.weight"), vec![16, 16]),
        (format!("{l}.self_attn.q_norm.weight"), vec![8]),
        (format!("{l}.self_attn.k_norm.weight"), vec![8]),
        (format!("{l}.mlp.gate_proj.weight"), vec![32, 16]),
        (format!("{l}.mlp.up_proj.weight"), vec![32, 16]),
        (format!("{l}.mlp.down_proj.weight"), vec![16, 32]),
        ("model.norm.weight".into(), vec![16]),
        ("lm_head.weight".into(), vec![64, 16]),
    ];
    let mut header = String::from("{");
    let mut data = Vec::new();
    let mut entries = Vec::new();
    for (name, shape) in &specs {
        let n: i64 = shape.iter().product();
        let off = data.len();
        for i in 0..n {
            let v = 0.02 + (i % 7) as f32 * 0.01;
            data.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        let sh: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        entries.push(format!(
            "\"{name}\":{{\"dtype\":\"F16\",\"shape\":[{}],\"data_offsets\":[{off},{}]}}",
            sh.join(","),
            data.len()
        ));
    }
    header.push_str(&entries.join(","));
    header.push('}');
    let mut f = std::fs::File::create(dir.join("model.safetensors")).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(&data).unwrap();
}

fn flex_summary(pkg: &Path) -> String {
    let o = milc(&["inspect", p(pkg)]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    text(&o.stdout)
}

#[test]
fn convert_and_reshape_seq_range() {
    let t = Tmp::new("cli");
    let model = t.0.join("model");
    tiny_qwen3(&model);
    let pkg = t.0.join("r.mlpackage");

    // bad ranges fail before writing anything
    for (range, needle) in [
        ("1..99", "--max-kv"),
        ("1..", "unbounded"),
        ("0..8", ">= 1"),
        ("9..8", "below"),
        ("nonsense", "LO..HI"),
    ] {
        let o = milc(&[
            "convert",
            p(&model),
            "-o",
            p(&pkg),
            "--max-kv",
            "16",
            "--seq-range",
            range,
        ]);
        assert!(!o.status.success(), "{range} should fail");
        let e = text(&o.stderr);
        assert!(e.contains(needle), "{range}: {e}");
        assert!(!pkg.exists());
    }
    let o = milc(&[
        "convert",
        p(&model),
        "-o",
        p(&pkg),
        "--max-kv",
        "16",
        "--seq-range",
        "1..8",
        "--seq-lens",
        "1,4",
    ]);
    assert!(!o.status.success());
    assert!(
        text(&o.stderr).contains("mutually exclusive"),
        "{}",
        text(&o.stderr)
    );

    // valid conversion
    let o = milc(&[
        "convert",
        p(&model),
        "-o",
        p(&pkg),
        "--max-kv",
        "16",
        "--seq-range",
        "1..8",
    ]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let err = text(&o.stderr);
    assert!(
        err.contains("--seq-lens") && err.contains("ANE"),
        "expected ANE note:\n{err}"
    );

    // reshape range → enumerated → range
    let o = milc(&[
        "reshape",
        p(&pkg),
        "--seq-lens",
        "2,6",
        "-o",
        p(&t.0.join("e.mlpackage")),
    ]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert!(
        text(&o.stdout).contains("1..8 → [2, 6]"),
        "{}",
        text(&o.stdout)
    );
    let o = milc(&[
        "reshape",
        p(&t.0.join("e.mlpackage")),
        "--seq-range",
        "1..12",
    ]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert!(
        text(&o.stdout).contains("[2, 6] → 1..12"),
        "{}",
        text(&o.stdout)
    );
    assert!(!flex_summary(&t.0.join("e.mlpackage")).is_empty());

    // fixed-shape packages are refused
    let fixed = t.0.join("f.mlpackage");
    let o = milc(&[
        "convert",
        p(&model),
        "-o",
        p(&fixed),
        "--max-kv",
        "16",
        "--seq",
        "4",
    ]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let o = milc(&["reshape", p(&fixed), "--seq-range", "1..8"]);
    assert!(!o.status.success());
    assert!(
        text(&o.stderr).contains("no sequence flexibility"),
        "{}",
        text(&o.stderr)
    );
}
