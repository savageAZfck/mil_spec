//! Does the masked KV update compute the same thing on the ANE as on
//! the CPU? Mini-model: inputs `pos` (int32) and new K rows
//! `(1,kvh,S,hd)`, one `(2,kvh,max_kv,hd)` state; output = the K half of
//! the state after the update. Compared across CpuOnly / ANE for a few
//! structural variants. Prints `KVPROBE` rows.

use mil_infer::{ComputeUnits, Input, Model};
use mil_spec::*;

fn f16s(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
        .collect()
}

struct V {
    name: &'static str,
    reread: bool,
    use_matmul: bool,
}

#[test]
#[ignore = "diagnostic: needs coremlc; prints a matrix"]
fn masked_kv_cpu_vs_ane() {
    if mil_compile::coremlc_path().is_none() {
        return;
    }
    let (kvh, max_kv, hd, s) = (3i64, 64i64, 64i64, 5i64);
    let root = std::env::temp_dir().join(format!("kvprobe_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for v in [
        V {
            name: "reread+matmul",
            reread: true,
            use_matmul: true,
        },
        V {
            name: "ssa+matmul",
            reread: false,
            use_matmul: true,
        },
        V {
            name: "reread, P only (no matmul: out=P-row-sum)",
            reread: true,
            use_matmul: false,
        },
    ] {
        let mut b = Block::new();
        let tt = |sh: &[i64]| ValueType::Tensor(TensorType::f16(sh));
        let posf = b.cast("pos", "fp16", &[1], "pos_f16", false);
        let posf = b.reshape(&posf, &[1, 1, 1, 1], "pos_f4");
        let jv: Vec<f32> = (0..max_kv).map(|j| j as f32).collect();
        let iota_j = b.op(
            "const",
            vec![],
            vec![("iota_j", tt(&[1, 1, max_kv, 1]))],
            vec![("val".into(), Value::f16s(&[1, 1, max_kv, 1], &jv))],
        )[0]
        .clone();
        let iv: Vec<f32> = (0..s).map(|i| i as f32).collect();
        let iota_i = b.op(
            "const",
            vec![],
            vec![("iota_i", tt(&[1, 1, 1, s]))],
            vec![("val".into(), Value::f16s(&[1, 1, 1, s], &iv))],
        )[0]
        .clone();
        let d1 = b.sub(&iota_j, &posf, &[1, 1, max_kv, 1], "d1");
        let d2 = b.sub(&d1, &iota_i, &[1, 1, max_kv, s], "d2");
        let ab = b.abs(&d2, &[1, 1, max_kv, s], "ab");
        let one = b.konst_f16("one", 1.0);
        let om = b.sub(&one, &ab, &[1, 1, max_kv, s], "om");
        let zero = b.konst_f16("zero", 0.0);
        let p = b.maximum(&om, &zero, &[1, 1, max_kv, s], "p");
        let rs = b.reduce_sum(&p, &[3], true, &[1, 1, max_kv, 1], "rs");
        let keep = b.sub(&one, &rs, &[1, 1, max_kv, 1], "keep");
        let pt = b.tile(&p, &[1, kvh as i32, 1, 1], &[1, kvh, max_kv, s], "pt");
        let kt = b.tile(
            &keep,
            &[1, kvh as i32, 1, hd as i32],
            &[1, kvh, max_kv, hd],
            "kt",
        );
        let kv2 = [2, kvh, max_kv, hd];
        let kv1 = [1, kvh, max_kv, hd];
        let old = b.read_state("st", &kv2, "old");
        let part = |b: &mut Block, x: &str, r: i32, nm: &str| {
            b.slice(
                x,
                &[r, 0, 0, 0],
                &[r + 1, kvh as i32, max_kv as i32, hd as i32],
                &kv1,
                nm,
            )
        };
        let kold = part(&mut b, &old, 0, "kold");
        let vold = part(&mut b, &old, 1, "vold");
        let (ksc, vsc) = if v.use_matmul {
            (
                b.matmul(&pt, "knew_in", false, &kv1, "ksc"),
                b.matmul(&pt, "knew_in", false, &kv1, "vsc"),
            )
        } else {
            let t = b.tile(&rs, &[1, kvh as i32, 1, hd as i32], &kv1, "rst");
            (t.clone(), t)
        };
        let kept = b.mul(&kold, &kt, &kv1, "kkeep");
        let knew = b.add(&kept, &ksc, &kv1, "knew");
        let vkept = b.mul(&vold, &kt, &kv1, "vkeep");
        let vnew = b.add(&vkept, &vsc, &kv1, "vnew");
        let both = b.concat(&[knew.clone(), vnew], 0, &kv2, "both");
        b.write_state("st", &both);
        let out = if v.reread {
            let upd = b.read_state("st", &kv2, "upd");
            part(&mut b, &upd, 0, "kfull")
        } else {
            knew
        };
        b.outputs = vec![out.clone()];
        let feat = |n: &str, sh: &[i64], dt| Feature {
            name: n.into(),
            shape: sh.to_vec(),
            dtype: dt,
            is_state: false,
        };
        let fin = vec![
            NVT {
                name: "pos".into(),
                ty: ValueType::Tensor(TensorType {
                    dtype: DType::Int32,
                    shape: vec![1],
                }),
            },
            NVT {
                name: "knew_in".into(),
                ty: tt(&[1, kvh, s, hd]),
            },
            NVT {
                name: "st".into(),
                ty: ValueType::State(TensorType::f16(&kv2)),
            },
        ];
        let state = Feature {
            name: "st".into(),
            shape: kv2.to_vec(),
            dtype: DType::Fp16,
            is_state: true,
        };
        let spec = encode_model(
            &[
                feat("pos", &[1], DType::Int32),
                feat("knew_in", &[1, kvh, s, hd], DType::Fp16),
            ],
            &[feat(&out, &kv1, DType::Fp16)],
            &[state],
            &b,
            &fin,
            &ModelMeta::new(10, "CoreML9"),
        );
        let pkg = root.join("p.mlpackage");
        write_mlpackage(&pkg, &spec, None).unwrap();
        let comp = match mil_compile::compile(&pkg, &root.join("c")) {
            Ok(c) => c,
            Err(e) => {
                println!(
                    "KVPROBE {:<44} coremlc rejects: {}",
                    v.name,
                    e.to_string().lines().next().unwrap_or("")
                );
                continue;
            }
        };
        let knew_data: Vec<f32> = (0..kvh * s * hd)
            .map(|i| 1.0 + (i % 13) as f32 * 0.01)
            .collect();
        let kb = f16s(&knew_data);
        let pos = 7i32.to_le_bytes();
        let sh_k = [1, kvh, s, hd];
        let sh_p = [1i64];
        let mut res = Vec::new();
        for (n, cu) in [
            ("cpu", ComputeUnits::CpuOnly),
            ("ane", ComputeUnits::CpuAndNeuralEngine),
        ] {
            let r = Model::load(&comp.path, cu)
                .map_err(|e| e.to_string())
                .and_then(|m| {
                    let st = m.new_state().map_err(|e| e.to_string())?;
                    let ins = [
                        Input {
                            name: "pos",
                            shape: &sh_p,
                            data: &pos,
                            dtype: DType::Int32,
                        },
                        Input {
                            name: "knew_in",
                            shape: &sh_k,
                            data: &kb,
                            dtype: DType::Fp16,
                        },
                    ];
                    m.predict_with_state(Some(&st), &ins)
                        .map(|p| p.outputs[0].values())
                        .map_err(|e| e.to_string())
                });
            res.push((n, r));
        }
        let summary = |r: &Result<Vec<f32>, String>| match r {
            Ok(v) => {
                let nz = v.iter().filter(|x| **x != 0.0).count();
                format!("nonzero {nz}/{}", v.len())
            }
            Err(e) => format!("ERR {}", e.chars().take(60).collect::<String>()),
        };
        let same = match (&res[0].1, &res[1].1) {
            (Ok(a), Ok(b)) => a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max),
            _ => f32::NAN,
        };
        println!(
            "KVPROBE {:<44} cpu[{}] ane[{}] max|cpu-ane|={same}",
            v.name,
            summary(&res[0].1),
            summary(&res[1].1)
        );
        let _ = std::fs::remove_dir_all(&pkg);
        let _ = std::fs::remove_dir_all(root.join("c"));
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// Numerics of a chain of `n` 1x1 convs (distinct blob weights) on the
/// ANE vs the CPU, at the shapes the decoder uses `(1, 576, 1, 5)`.
#[test]
#[ignore = "diagnostic: needs coremlc; prints a matrix"]
fn conv_chain_cpu_vs_ane() {
    if mil_compile::coremlc_path().is_none() {
        return;
    }
    let root = std::env::temp_dir().join(format!("convchain_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (c, s) = (576i64, 5i64);
    let shape = [1, c, 1, s];
    for n in [1usize, 2, 3, 4, 5, 8] {
        let mut b = Block::new();
        let mut wb = WeightBin::new();
        let mut cur = "x".to_string();
        for i in 0..n {
            let wdata: Vec<u8> = (0..c * c)
                .flat_map(|j| {
                    let v = (((j * 7 + i as i64 * 13) % 101) as f32 - 50.0) * 0.0004;
                    half::f16::from_f32(v).to_le_bytes()
                })
                .collect();
            let off = wb.put(&format!("w{i}"), DType::Fp16, &[c, c, 1, 1], &wdata);
            let w = b.konst_blob(
                &format!("w{i}"),
                "@model_path/weights/weight.bin",
                off,
                DType::Fp16,
                &[c, c, 1, 1],
            );
            let st = b.fresh("stride");
            let st = b.konst_i32(&st, &[1, 1]);
            let pt = b.fresh("padtype");
            let pt = b.konst_str(&pt, "valid");
            let pd = b.fresh("pad");
            let pd = b.konst_i32(&pd, &[0, 0, 0, 0]);
            let dl = b.fresh("dil");
            let dl = b.konst_i32(&dl, &[1, 1]);
            let gp = b.fresh("grp");
            let gp = b.konst_scalar_i32(&gp, 1);
            cur = b.o1(
                "conv",
                vec![
                    ("x".into(), bind(&cur).1),
                    ("weight".into(), bind(&w).1),
                    ("strides".into(), bind(&st).1),
                    ("pad_type".into(), bind(&pt).1),
                    ("pad".into(), bind(&pd).1),
                    ("dilations".into(), bind(&dl).1),
                    ("groups".into(), bind(&gp).1),
                ],
                &format!("cv{i}"),
                ValueType::Tensor(TensorType::f16(&shape)),
            );
        }
        b.outputs = vec![cur.clone()];
        let f = |name: &str| Feature {
            name: name.into(),
            shape: shape.to_vec(),
            dtype: DType::Fp16,
            is_state: false,
        };
        let fin = vec![NVT {
            name: "x".into(),
            ty: ValueType::Tensor(TensorType::f16(&shape)),
        }];
        let spec = encode_model(
            &[f("x")],
            &[f(&cur)],
            &[],
            &b,
            &fin,
            &ModelMeta::new(10, "CoreML9"),
        );
        let pkg = root.join("p.mlpackage");
        write_mlpackage(&pkg, &spec, Some(&wb.finish())).unwrap();
        let comp = mil_compile::compile(&pkg, &root.join("c")).unwrap();
        let x: Vec<f32> = (0..c * s).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let xb = f16s(&x);
        let mut outs = Vec::new();
        for cu in [ComputeUnits::CpuOnly, ComputeUnits::CpuAndNeuralEngine] {
            let m = Model::load(&comp.path, cu).unwrap();
            let p = m
                .predict(&[Input {
                    name: "x",
                    shape: &shape,
                    data: &xb,
                    dtype: DType::Fp16,
                }])
                .unwrap();
            outs.push(p.outputs[0].values());
        }
        let nonfinite = outs[1].iter().filter(|v| !v.is_finite()).count();
        let maxd = outs[0]
            .iter()
            .zip(&outs[1])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let cmax = outs[0].iter().fold(0f32, |m, v| m.max(v.abs()));
        println!(
            "CONVCHAIN n={n} cpu max|v|={cmax:.3} ane nonfinite={nonfinite} max|cpu-ane|={maxd:.4}"
        );
        let _ = std::fs::remove_dir_all(&pkg);
        let _ = std::fs::remove_dir_all(root.join("c"));
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// Attention-output → o_proj conv, in isolation: which way of getting
/// `(1, kvh, g*S, hd)` into the `(1, qh*hd, 1, S)` conv input is
/// numerically right on the ANE? Variants A.. below; S sweeps 4..9.
#[test]
#[ignore = "diagnostic: needs coremlc; prints a matrix"]
fn attn_out_to_oproj_cpu_vs_ane() {
    if mil_compile::coremlc_path().is_none() {
        return;
    }
    let root = std::env::temp_dir().join(format!("oproj_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (kvh, g, hd) = (3i64, 3i64, 64i64);
    let qh = kvh * g;
    let d = qh * hd;
    for variant in [
        "A reshape-transpose-reshape",
        "B transposes (old chain)",
        "C conv direct (control: x is (1,d,1,S))",
    ] {
        for s in [3i64, 4, 5, 8, 9, 16] {
            let mut b = Block::new();
            let mut wb = WeightBin::new();
            let wdata: Vec<u8> = (0..d * d)
                .flat_map(|j| {
                    let v = (((j * 7) % 101) as f32 - 50.0) * 0.0004;
                    half::f16::from_f32(v).to_le_bytes()
                })
                .collect();
            let off = wb.put("wo", DType::Fp16, &[d, d, 1, 1], &wdata);
            let w = b.konst_blob(
                "wo",
                "@model_path/weights/weight.bin",
                off,
                DType::Fp16,
                &[d, d, 1, 1],
            );
            let in_shape: Vec<i64> = if variant.starts_with('C') {
                vec![1, d, 1, s]
            } else {
                vec![1, kvh, g * s, hd]
            };
            let ac4 = match variant.chars().next().unwrap() {
                'A' => {
                    let ah = b.reshape("att", &[1, qh, s, hd], "ah");
                    let at = b.transpose(&ah, &[0, 1, 3, 2], &[1, qh, hd, s], "at");
                    b.reshape(&at, &[1, d, 1, s], "ac4")
                }
                'B' => {
                    let ah = b.reshape("att", &[1, qh, s, hd], "ah");
                    let at = b.transpose(&ah, &[0, 2, 1, 3], &[1, s, qh, hd], "at");
                    let ac = b.reshape(&at, &[1, s, d, 1], "ac");
                    b.transpose(&ac, &[0, 2, 3, 1], &[1, d, 1, s], "ac4")
                }
                _ => "att".to_string(),
            };
            let st = b.fresh("stride");
            let st = b.konst_i32(&st, &[1, 1]);
            let pt = b.fresh("padtype");
            let pt = b.konst_str(&pt, "valid");
            let pd = b.fresh("pad");
            let pd = b.konst_i32(&pd, &[0, 0, 0, 0]);
            let dl = b.fresh("dil");
            let dl = b.konst_i32(&dl, &[1, 1]);
            let gp = b.fresh("grp");
            let gp = b.konst_scalar_i32(&gp, 1);
            let out = b.o1(
                "conv",
                vec![
                    ("x".into(), bind(&ac4).1),
                    ("weight".into(), bind(&w).1),
                    ("strides".into(), bind(&st).1),
                    ("pad_type".into(), bind(&pt).1),
                    ("pad".into(), bind(&pd).1),
                    ("dilations".into(), bind(&dl).1),
                    ("groups".into(), bind(&gp).1),
                ],
                "o",
                ValueType::Tensor(TensorType::f16(&[1, d, 1, s])),
            );
            b.outputs = vec![out.clone()];
            let f = |name: &str, sh: &[i64]| Feature {
                name: name.into(),
                shape: sh.to_vec(),
                dtype: DType::Fp16,
                is_state: false,
            };
            let fin = vec![NVT {
                name: "att".into(),
                ty: ValueType::Tensor(TensorType::f16(&in_shape)),
            }];
            let spec = encode_model(
                &[f("att", &in_shape)],
                &[f(&out, &[1, d, 1, s])],
                &[],
                &b,
                &fin,
                &ModelMeta::new(10, "CoreML9"),
            );
            let pkg = root.join("p.mlpackage");
            write_mlpackage(&pkg, &spec, Some(&wb.finish())).unwrap();
            let comp = match mil_compile::compile(&pkg, &root.join("c")) {
                Ok(c) => c,
                Err(e) => {
                    println!(
                        "OPROJ {variant:<40} S={s}: coremlc rejects {}",
                        e.to_string().lines().next().unwrap_or("")
                    );
                    continue;
                }
            };
            let n: i64 = in_shape.iter().product();
            let x: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.03).collect();
            let xb = f16s(&x);
            let mut outs = Vec::new();
            for cu in [ComputeUnits::CpuOnly, ComputeUnits::CpuAndNeuralEngine] {
                let m = Model::load(&comp.path, cu).unwrap();
                let p = m
                    .predict(&[Input {
                        name: "att",
                        shape: &in_shape,
                        data: &xb,
                        dtype: DType::Fp16,
                    }])
                    .unwrap();
                outs.push(p.outputs[0].values());
            }
            let nonfinite = outs[1].iter().filter(|v| !v.is_finite()).count();
            let maxd = outs[0]
                .iter()
                .zip(&outs[1])
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            println!(
                "OPROJ {variant:<40} S={s:<2} ane nonfinite={nonfinite} max|cpu-ane|={maxd:.4}"
            );
            let _ = std::fs::remove_dir_all(&pkg);
            let _ = std::fs::remove_dir_all(root.join("c"));
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}
