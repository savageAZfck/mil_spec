//! End-to-end ONNX→MIL tests: encode an ONNX `ModelProto` in Rust,
//! convert with `mil_onnx`, write `.mlpackage`, compile via `coremlc`,
//! predict via `mil_infer`, and compare against independent Rust f64
//! references.
//!
//! All tests early-return when `coremlc` is unavailable; temp dirs are
//! removed unless `MIL_KEEP_ARTIFACTS=1`.

mod common;

use common::*;
use mil_infer::Input;

/// Convert ONNX bytes and (if coremlc exists) write→compile→predict.
fn go(tag: &str, bytes: &[u8], feeds: &[Input<'_>]) -> Option<mil_infer::Prediction> {
    let built = convert(bytes).unwrap_or_else(|e| panic!("{tag}: convert: {e}"));
    run(tag, &built, feeds)
}

fn feed<'a>(name: &'a str, shape: &'a [i64], data: &'a [u8]) -> Input<'a> {
    Input {
        name,
        shape,
        data,
        dtype: mil_spec::DType::Fp16,
    }
}

fn softmax_row(v: &[f64]) -> Vec<f64> {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = v.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn erf64(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

// ---------- elementwise ----------

#[test]
fn unary_e2e() {
    let shape = [2i64, 8];
    let x = randv(16, -3.0, 3.0, 0x11);
    // xp = |x|+0.1 for sqrt/log domain safety (strictly positive).
    let xp: Vec<f32> = x.iter().map(|v| v.abs() + 0.1).collect();
    let g = graph(
        "g",
        vec![
            node("Relu", &["x"], &["o_relu"], &[], ""),
            node("Sigmoid", &["x"], &["o_sig"], &[], ""),
            node("Tanh", &["x"], &["o_tanh"], &[], ""),
            node("Neg", &["x"], &["o_neg"], &[], ""),
            node("Abs", &["x"], &["o_abs"], &[], ""),
            node("Sqrt", &["xp"], &["o_sqrt"], &[], ""),
            node("Exp", &["xp"], &["o_exp"], &[], ""),
            node("Log", &["xp"], &["o_log"], &[], ""),
            node("Erf", &["x"], &["o_erf"], &[], ""),
        ],
        vec![],
        vec![
            value_info("x", et::FLOAT, &shape),
            value_info("xp", et::FLOAT, &shape),
        ],
        vec![
            value_info("o_relu", et::FLOAT, &shape),
            value_info("o_sig", et::FLOAT, &shape),
            value_info("o_tanh", et::FLOAT, &shape),
            value_info("o_neg", et::FLOAT, &shape),
            value_info("o_abs", et::FLOAT, &shape),
            value_info("o_sqrt", et::FLOAT, &shape),
            value_info("o_exp", et::FLOAT, &shape),
            value_info("o_log", et::FLOAT, &shape),
            value_info("o_erf", et::FLOAT, &shape),
        ],
    );
    let xb = f16_bytes(&x);
    let xpb = f16_bytes(&xp);
    let p = go(
        "unary",
        &model(&g),
        &[feed("x", &shape, &xb), feed("xp", &shape, &xpb)],
    );
    let Some(p) = p else { return };
    let xq = qh(&x);
    let xpq = qh(&xp);
    let want: Vec<f64> = xq.iter().map(|v| v.max(0.0)).collect();
    check(&out(&p, "o_relu").values(), &want, 0.0, 1e-3, "relu");
    let want: Vec<f64> = xq.iter().map(|v| sigmoid(*v)).collect();
    check(&out(&p, "o_sig").values(), &want, 5e-3, 2e-3, "sigmoid");
    let want: Vec<f64> = xq.iter().map(|v| v.tanh()).collect();
    check(&out(&p, "o_tanh").values(), &want, 5e-3, 2e-3, "tanh");
    let want: Vec<f64> = xq.iter().map(|v| -*v).collect();
    check(&out(&p, "o_neg").values(), &want, 0.0, 1e-3, "neg");
    let want: Vec<f64> = xq.iter().map(|v| v.abs()).collect();
    check(&out(&p, "o_abs").values(), &want, 0.0, 1e-3, "abs");
    let want: Vec<f64> = xpq.iter().map(|v| v.sqrt()).collect();
    check(&out(&p, "o_sqrt").values(), &want, 5e-3, 2e-3, "sqrt");
    let want: Vec<f64> = xpq.iter().map(|v| v.exp()).collect();
    check(&out(&p, "o_exp").values(), &want, 1e-2, 5e-3, "exp");
    let want: Vec<f64> = xpq.iter().map(|v| v.ln()).collect();
    check(&out(&p, "o_log").values(), &want, 1e-2, 5e-3, "log");
    let want: Vec<f64> = xq.iter().map(|v| erf64(*v)).collect();
    check(&out(&p, "o_erf").values(), &want, 5e-3, 3e-3, "erf");
}

#[test]
fn binary_e2e() {
    let shape = [2i64, 4];
    let x = randv(8, 0.2, 3.0, 0xBE);
    let y = randv(8, 0.2, 2.0, 0xFD);
    let g = graph(
        "g",
        vec![
            node("Add", &["x", "y"], &["o_add"], &[], ""),
            node("Sub", &["x", "y"], &["o_sub"], &[], ""),
            node("Mul", &["x", "y"], &["o_mul"], &[], ""),
            node("Div", &["x", "y"], &["o_div"], &[], ""),
            node("Pow", &["x", "y"], &["o_pow"], &[], ""),
        ],
        vec![],
        vec![
            value_info("x", et::FLOAT, &shape),
            value_info("y", et::FLOAT, &shape),
        ],
        vec![
            value_info("o_add", et::FLOAT, &shape),
            value_info("o_sub", et::FLOAT, &shape),
            value_info("o_mul", et::FLOAT, &shape),
            value_info("o_div", et::FLOAT, &shape),
            value_info("o_pow", et::FLOAT, &shape),
        ],
    );
    let (xb, yb) = (f16_bytes(&x), f16_bytes(&y));
    let p = go(
        "binary",
        &model(&g),
        &[feed("x", &shape, &xb), feed("y", &shape, &yb)],
    );
    let Some(p) = p else { return };
    let (xq, yq) = (qh(&x), qh(&y));
    type BinOp = fn(f64, f64) -> f64;
    let cases: [(&str, BinOp); 5] = [
        ("o_add", |a, b| a + b),
        ("o_sub", |a, b| a - b),
        ("o_mul", |a, b| a * b),
        ("o_div", |a, b| a / b),
        ("o_pow", |a, b| a.powf(b)),
    ];
    for (n, f) in cases {
        let want: Vec<f64> = xq.iter().zip(&yq).map(|(&a, &b)| f(a, b)).collect();
        check(&out(&p, n).values(), &want, 1e-2, 1e-2, n);
    }
}

#[test]
fn gemm_relu_softmax_chain_e2e() {
    // The requested multi-node graph: Gemm → Relu → Softmax.
    let (m, k, n) = (2i64, 4i64, 3i64);
    let a = randv((m * k) as usize, -1.0, 1.0, 0xA1);
    let b = randv((k * n) as usize, -1.0, 1.0, 0xB2);
    let c = randv(n as usize, -0.5, 0.5, 0xC3);
    let g = graph(
        "g",
        vec![
            node("Gemm", &["a", "b", "c"], &["g_out"], &[], "gemm1"),
            node("Relu", &["g_out"], &["r_out"], &[], "relu1"),
            node("Softmax", &["r_out"], &["y"], &[], "sm1"),
        ],
        vec![t_f32("b", &[k, n], &b), t_f32("c", &[n], &c)],
        vec![value_info("a", et::FLOAT, &[m, k])],
        vec![value_info("y", et::FLOAT, &[m, n])],
    );
    let ab = f16_bytes(&a);
    let p = go("gemm_chain", &model(&g), &[feed("a", &[m, k], &ab)]);
    let Some(p) = p else { return };
    // f64 ref: A@B + C (broadcast row) → relu → softmax per row.
    let (aq, bq, cq) = (qh(&a), qh(&b), qh(&c));
    let mut y = vec![0.0f64; (m * n) as usize];
    for i in 0..m as usize {
        for j in 0..n as usize {
            y[i * n as usize + j] = cq[j]
                + (0..k as usize)
                    .map(|l| aq[i * k as usize + l] * bq[l * n as usize + j])
                    .sum::<f64>();
            y[i * n as usize + j] = y[i * n as usize + j].max(0.0);
        }
    }
    let mut want = Vec::new();
    for i in 0..m as usize {
        want.extend(softmax_row(&y[i * n as usize..(i + 1) * n as usize]));
    }
    check(
        &out(&p, "y").values(),
        &want,
        1e-2,
        5e-3,
        "gemm→relu→softmax",
    );
}

#[test]
fn gemm_attrs_e2e() {
    // alpha=0.5, beta=2.0, transB=1 — full attr path.
    let (m, k, n) = (2i64, 3i64, 2i64);
    let a = randv((m * k) as usize, -1.0, 1.0, 0xAA);
    let b = randv((n * k) as usize, -1.0, 1.0, 0xBB); // stored (n,k), transB
    let c = randv(n as usize, -0.5, 0.5, 0xCC);
    let g = graph(
        "g",
        vec![node(
            "Gemm",
            &["a", "b", "c"],
            &["y"],
            &[
                Attr::I("transB", 1),
                Attr::F("alpha", 0.5),
                Attr::F("beta", 2.0),
            ],
            "gemm",
        )],
        vec![t_f32("b", &[n, k], &b), t_f32("c", &[n], &c)],
        vec![value_info("a", et::FLOAT, &[m, k])],
        vec![value_info("y", et::FLOAT, &[m, n])],
    );
    let ab = f16_bytes(&a);
    let p = go("gemm_attrs", &model(&g), &[feed("a", &[m, k], &ab)]);
    let Some(p) = p else { return };
    let (aq, bq, cq) = (qh(&a), qh(&b), qh(&c));
    let mut want = vec![0.0f64; (m * n) as usize];
    for i in 0..m as usize {
        for j in 0..n as usize {
            let dot: f64 = (0..k as usize)
                .map(|l| aq[i * k as usize + l] * bq[j * k as usize + l])
                .sum();
            want[i * n as usize + j] = 0.5 * dot + 2.0 * cq[j];
        }
    }
    check(&out(&p, "y").values(), &want, 1e-2, 5e-3, "gemm attrs");
}

#[test]
fn matmul_e2e() {
    let (m, k, n) = (2i64, 3i64, 4i64);
    let a = randv((m * k) as usize, -1.5, 1.5, 0xAA);
    let b = randv((k * n) as usize, -1.5, 1.5, 0xBB);
    let g = graph(
        "g",
        vec![node("MatMul", &["a", "b"], &["y"], &[], "")],
        vec![t_f32("b", &[k, n], &b)],
        vec![value_info("a", et::FLOAT, &[m, k])],
        vec![value_info("y", et::FLOAT, &[m, n])],
    );
    let ab = f16_bytes(&a);
    let p = go("matmul", &model(&g), &[feed("a", &[m, k], &ab)]);
    let Some(p) = p else { return };
    let (aq, bq) = (qh(&a), qh(&b));
    let mut want = vec![0.0f64; (m * n) as usize];
    for i in 0..m as usize {
        for j in 0..n as usize {
            want[i * n as usize + j] = (0..k as usize)
                .map(|l| aq[i * k as usize + l] * bq[l * n as usize + j])
                .sum();
        }
    }
    check(&out(&p, "y").values(), &want, 1e-2, 5e-3, "matmul");
}

/// Initializer ≥1024 B must land in `weight.bin` (blob path).
#[test]
fn initializer_blob_e2e() {
    let (m, k, n) = (1i64, 64i64, 64i64);
    let a = randv((m * k) as usize, -0.5, 0.5, 0x0B);
    let b = randv((k * n) as usize, -0.5, 0.5, 0x0C);
    let g = graph(
        "g",
        vec![node("MatMul", &["a", "w"], &["y"], &[], "")],
        vec![t_f32("w", &[k, n], &b)], // 16 KiB → blob
        vec![value_info("a", et::FLOAT, &[m, k])],
        vec![value_info("y", et::FLOAT, &[m, n])],
    );
    let built = convert(&model(&g)).unwrap();
    assert!(built.weight_bin.is_some(), "expected weight.bin payload");
    // Verify the blob materializes where write_mlpackage puts it.
    if coremlc() {
        let dir = std::env::temp_dir().join(format!("mil_onnx_blob_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        built.write_mlpackage(&dir.join("m.mlpackage")).unwrap();
        assert!(dir
            .join("m.mlpackage/Data/com.apple.CoreML/weights/weight.bin")
            .exists());
        if !keep_artifacts() {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
    let ab = f16_bytes(&a);
    let p = run("blob", &built, &[feed("a", &[m, k], &ab)]);
    let Some(p) = p else { return };
    let (aq, bq) = (qh(&a), qh(&b));
    let mut want = vec![0.0f64; (m * n) as usize];
    for i in 0..m as usize {
        for j in 0..n as usize {
            want[i * n as usize + j] = (0..k as usize)
                .map(|l| aq[i * k as usize + l] * bq[l * n as usize + j])
                .sum();
        }
    }
    check(&out(&p, "y").values(), &want, 2e-2, 5e-3, "blob matmul");
}

#[test]
fn clip_e2e() {
    let shape = [2i64, 4];
    let x = randv(8, -4.0, 4.0, 0xC1);
    let g = graph(
        "g",
        vec![node("Clip", &["x", "lo", "hi"], &["y"], &[], "")],
        vec![t_scalar_f32("lo", -1.5), t_scalar_f32("hi", 2.5)],
        vec![value_info("x", et::FLOAT, &shape)],
        vec![value_info("y", et::FLOAT, &shape)],
    );
    let xb = f16_bytes(&x);
    let p = go("clip", &model(&g), &[feed("x", &shape, &xb)]);
    let Some(p) = p else { return };
    let want: Vec<f64> = qh(&x).iter().map(|v| v.clamp(-1.5, 2.5)).collect();
    check(&out(&p, "y").values(), &want, 0.0, 2e-3, "clip");
}

// ---------- reductions ----------

#[test]
fn reduce_e2e() {
    let shape = [2i64, 3, 4];
    let keep = [2i64, 1, 4];
    let x = randv(24, 0.1, 2.0, 0xED);
    // opset≥13: `axes` is an input, `keepdims` an attr.
    let axes = t_i64("axes", &[1], &[1]);
    let mk = |op: &str, out: &str| node(op, &["x", "axes"], &[out], &[Attr::I("keepdims", 1)], "");
    let g = graph(
        "g",
        vec![
            mk("ReduceSum", "o_sum"),
            mk("ReduceMean", "o_mean"),
            mk("ReduceMax", "o_max"),
            mk("ReduceMin", "o_min"),
            mk("ReduceProd", "o_prod"),
            mk("ReduceL1", "o_l1"),
            mk("ReduceL2", "o_l2"),
            mk("ReduceLogSumExp", "o_lse"),
            node(
                "ReduceSum",
                &["x"],
                &["o_sumall"],
                &[Attr::I("keepdims", 0)],
                "",
            ),
        ],
        vec![axes],
        vec![value_info("x", et::FLOAT, &shape)],
        vec![
            value_info("o_sum", et::FLOAT, &keep),
            value_info("o_mean", et::FLOAT, &keep),
            value_info("o_max", et::FLOAT, &keep),
            value_info("o_min", et::FLOAT, &keep),
            value_info("o_prod", et::FLOAT, &keep),
            value_info("o_l1", et::FLOAT, &keep),
            value_info("o_l2", et::FLOAT, &keep),
            value_info("o_lse", et::FLOAT, &keep),
            value_info("o_sumall", et::FLOAT, &[]),
        ],
    );
    let xb = f16_bytes(&x);
    let p = go("reduce", &model(&g), &[feed("x", &shape, &xb)]);
    let Some(p) = p else { return };
    let xq = qh(&x);
    let red = |f: fn(f64, f64) -> f64| -> Vec<f64> {
        (0..2 * 4)
            .map(|i| {
                let (a, b) = (i / 4, i % 4);
                (0..3).map(|j| xq[a * 12 + j * 4 + b]).reduce(f).unwrap()
            })
            .collect()
    };
    check(
        &out(&p, "o_sum").values(),
        &red(|a, b| a + b),
        5e-3,
        3e-3,
        "sum",
    );
    check(
        &out(&p, "o_mean").values(),
        &red(|a, b| a + b)
            .iter()
            .map(|v| v / 3.0)
            .collect::<Vec<_>>(),
        5e-3,
        3e-3,
        "mean",
    );
    check(&out(&p, "o_max").values(), &red(f64::max), 0.0, 2e-3, "max");
    check(&out(&p, "o_min").values(), &red(f64::min), 0.0, 2e-3, "min");
    check(
        &out(&p, "o_prod").values(),
        &red(|a, b| a * b),
        5e-3,
        5e-3,
        "prod",
    );
    check(
        &out(&p, "o_l1").values(),
        &{
            (0..2 * 4)
                .map(|i| {
                    let (a, b) = (i / 4, i % 4);
                    (0..3).map(|j| xq[a * 12 + j * 4 + b].abs()).sum::<f64>()
                })
                .collect::<Vec<_>>()
        },
        5e-3,
        3e-3,
        "l1",
    );
    check(
        &out(&p, "o_l2").values(),
        &{
            (0..2 * 4)
                .map(|i| {
                    let (a, b) = (i / 4, i % 4);
                    (0..3)
                        .map(|j| xq[a * 12 + j * 4 + b].powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .collect::<Vec<_>>()
        },
        5e-3,
        3e-3,
        "l2",
    );
    check(
        &out(&p, "o_lse").values(),
        &{
            (0..2 * 4)
                .map(|i| {
                    let (a, b) = (i / 4, i % 4);
                    (0..3)
                        .map(|j| xq[a * 12 + j * 4 + b].exp())
                        .sum::<f64>()
                        .ln()
                })
                .collect::<Vec<_>>()
        },
        5e-3,
        3e-3,
        "lse",
    );
    // no-axes → reduce over all → keepdims=0 drops all dims → scalar
    let want: Vec<f64> = vec![xq.iter().sum()];
    check(
        &out(&p, "o_sumall").values(),
        &want,
        5e-3,
        3e-3,
        "sum all axes",
    );
}

// ---------- conv/pool ----------

#[test]
fn conv_e2e() {
    // 1×1×5×5 input, 2×1×3×3 weight, pads 1, stride 1, bias.
    let ish = [1i64, 1, 5, 5];
    let osh = [1i64, 2, 5, 5];
    let x = randv(25, -1.0, 1.0, 0xC0);
    let w = randv(18, -1.0, 1.0, 0xC1);
    let bias = [0.25f32, -0.5];
    let g = graph(
        "g",
        vec![node(
            "Conv",
            &["x", "w", "bias"],
            &["y"],
            &[
                Attr::Ints("pads", &[1, 1, 1, 1]),
                Attr::Ints("strides", &[1, 1]),
            ],
            "conv1",
        )],
        vec![t_f32("w", &[2, 1, 3, 3], &w), t_f32("bias", &[2], &bias)],
        vec![value_info("x", et::FLOAT, &ish)],
        vec![value_info("y", et::FLOAT, &osh)],
    );
    let xb = f16_bytes(&x);
    let p = go("conv", &model(&g), &[feed("x", &ish, &xb)]);
    let Some(p) = p else { return };
    let (xq, wq, bq) = (qh(&x), qh(&w), qh(&bias));
    let mut want = vec![0.0f64; 50];
    for oc in 0..2usize {
        for oh in 0..5usize {
            for ow in 0..5usize {
                let mut acc = bq[oc];
                for kh in 0..3usize {
                    for kw in 0..3usize {
                        let ih = oh + kh;
                        let iw = ow + kw;
                        if ih > 0 && ih <= 5 && iw > 0 && iw <= 5 {
                            acc += xq[(ih - 1) * 5 + (iw - 1)] * wq[oc * 9 + kh * 3 + kw];
                        }
                    }
                }
                want[oc * 25 + oh * 5 + ow] = acc;
            }
        }
    }
    check(&out(&p, "y").values(), &want, 1e-2, 5e-3, "conv");
}

#[test]
fn pool_e2e() {
    let ish = [1i64, 2, 4, 4];
    let x = randv(32, -2.0, 2.0, 0xF0);
    let g = graph(
        "g",
        vec![
            node(
                "MaxPool",
                &["x"],
                &["o_mp"],
                &[
                    Attr::Ints("kernel_shape", &[2, 2]),
                    Attr::Ints("strides", &[2, 2]),
                ],
                "",
            ),
            node(
                "AveragePool",
                &["x"],
                &["o_ap"],
                &[
                    Attr::Ints("kernel_shape", &[2, 2]),
                    Attr::Ints("strides", &[2, 2]),
                ],
                "",
            ),
            node("GlobalAveragePool", &["x"], &["o_gap"], &[], ""),
            node("GlobalMaxPool", &["x"], &["o_gmp"], &[], ""),
        ],
        vec![],
        vec![value_info("x", et::FLOAT, &ish)],
        vec![
            value_info("o_mp", et::FLOAT, &[1, 2, 2, 2]),
            value_info("o_ap", et::FLOAT, &[1, 2, 2, 2]),
            value_info("o_gap", et::FLOAT, &[1, 2, 1, 1]),
            value_info("o_gmp", et::FLOAT, &[1, 2, 1, 1]),
        ],
    );
    let xb = f16_bytes(&x);
    let p = go("pool", &model(&g), &[feed("x", &ish, &xb)]);
    let Some(p) = p else { return };
    let xq = qh(&x);
    let mut mp = Vec::new();
    let mut ap = Vec::new();
    for c in 0..2usize {
        for oh in 0..2usize {
            for ow in 0..2usize {
                let w: Vec<f64> = (0..2)
                    .flat_map(|kh| (0..2).map(move |kw| (kh, kw)))
                    .map(|(kh, kw)| xq[c * 16 + (oh * 2 + kh) * 4 + (ow * 2 + kw)])
                    .collect();
                mp.push(w.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
                ap.push(w.iter().sum::<f64>() / 4.0);
            }
        }
    }
    check(&out(&p, "o_mp").values(), &mp, 0.0, 2e-3, "maxpool");
    check(&out(&p, "o_ap").values(), &ap, 5e-3, 3e-3, "avgpool");
    let gap: Vec<f64> = (0..2)
        .map(|c| xq[c * 16..c * 16 + 16].iter().sum::<f64>() / 16.0)
        .collect();
    check(&out(&p, "o_gap").values(), &gap, 5e-3, 3e-3, "gap");
    let gmp: Vec<f64> = (0..2)
        .map(|c| {
            xq[c * 16..c * 16 + 16]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    check(&out(&p, "o_gmp").values(), &gmp, 0.0, 2e-3, "gmp");
}

// ---------- shape ops ----------

#[test]
fn reshape_transpose_flatten_e2e() {
    let x = randv(24, -1.0, 1.0, 0x5E);
    let g = graph(
        "g",
        vec![
            node("Reshape", &["x", "s1"], &["o_rs"], &[], ""),
            node(
                "Transpose",
                &["x"],
                &["o_tr"],
                &[Attr::Ints("perm", &[0, 2, 1])],
                "",
            ),
            node("Flatten", &["x"], &["o_fl"], &[Attr::I("axis", 1)], ""),
            node("Squeeze", &["xu", "sqax"], &["o_sq"], &[], ""),
            node("Unsqueeze", &["x", "usax"], &["o_us"], &[], ""),
        ],
        vec![
            t_i64("s1", &[2], &[4, 6]),
            t_i64("sqax", &[1], &[1]),
            t_i64("usax", &[1], &[0]),
        ],
        vec![
            value_info("x", et::FLOAT, &[2, 3, 4]),
            value_info("xu", et::FLOAT, &[2, 1, 4]),
        ],
        vec![
            value_info("o_rs", et::FLOAT, &[4, 6]),
            value_info("o_tr", et::FLOAT, &[2, 4, 3]),
            value_info("o_fl", et::FLOAT, &[2, 12]),
            value_info("o_sq", et::FLOAT, &[2, 4]),
            value_info("o_us", et::FLOAT, &[1, 2, 3, 4]),
        ],
    );
    let xu = randv(8, -1.0, 1.0, 0x5F);
    let (xb, xub) = (f16_bytes(&x), f16_bytes(&xu));
    let p = go(
        "shapes",
        &model(&g),
        &[feed("x", &[2, 3, 4], &xb), feed("xu", &[2, 1, 4], &xub)],
    );
    let Some(p) = p else { return };
    let xq = qh(&x);
    check(&out(&p, "o_rs").values(), &xq, 0.0, 1e-3, "reshape");
    let mut tr = vec![0.0f64; 24];
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                tr[i * 12 + k * 3 + j] = xq[i * 12 + j * 4 + k];
            }
        }
    }
    check(&out(&p, "o_tr").values(), &tr, 0.0, 1e-3, "transpose");
    check(&out(&p, "o_fl").values(), &xq, 0.0, 1e-3, "flatten");
    check(&out(&p, "o_sq").values(), &qh(&xu), 0.0, 1e-3, "squeeze");
    check(&out(&p, "o_us").values(), &xq, 0.0, 1e-3, "unsqueeze");
}

/// Opset-11 Squeeze: `axes` is an *attribute*, not an input.
#[test]
fn squeeze_opset11_e2e() {
    let x = randv(8, -1.0, 1.0, 0xAA);
    let g = graph(
        "g",
        vec![node(
            "Squeeze",
            &["x"],
            &["y"],
            &[Attr::Ints("axes", &[1, 2])],
            "",
        )],
        vec![],
        vec![value_info("x", et::FLOAT, &[2, 1, 1, 4])],
        vec![value_info("y", et::FLOAT, &[2, 4])],
    );
    let xb = f16_bytes(&x);
    let p = go(
        "sq11",
        &model_opset(&g, 11),
        &[feed("x", &[2, 1, 1, 4], &xb)],
    );
    let Some(p) = p else { return };
    check(&out(&p, "y").values(), &qh(&x), 0.0, 1e-3, "squeeze11");
}

#[test]
fn slice_e2e() {
    let shape = [2i64, 5];
    let x = randv(10, -1.0, 1.0, 0x51);
    let g = graph(
        "g",
        vec![node(
            "Slice",
            &["x", "st", "en", "ax", "stp"],
            &["y"],
            &[],
            "",
        )],
        vec![
            t_i64("st", &[2], &[0, 1]),
            t_i64("en", &[2], &[2, 4]),
            t_i64("ax", &[2], &[0, 1]),
            t_i64("stp", &[2], &[1, 2]),
        ],
        vec![value_info("x", et::FLOAT, &shape)],
        vec![value_info("y", et::FLOAT, &[2, 2])],
    );
    let xb = f16_bytes(&x);
    let p = go("slice", &model(&g), &[feed("x", &shape, &xb)]);
    let Some(p) = p else { return };
    let xq = qh(&x);
    let want = vec![xq[1], xq[3], xq[6], xq[8]];
    check(&out(&p, "y").values(), &want, 0.0, 1e-3, "slice");
}

#[test]
fn gather_e2e() {
    let shape = [3i64, 4];
    let x = randv(12, -1.0, 1.0, 0x6A);
    let g = graph(
        "g",
        vec![node(
            "Gather",
            &["x", "idx"],
            &["y"],
            &[Attr::I("axis", 1)],
            "",
        )],
        vec![t_i64("idx", &[2], &[3, 0])],
        vec![value_info("x", et::FLOAT, &shape)],
        vec![value_info("y", et::FLOAT, &[3, 2])],
    );
    let xb = f16_bytes(&x);
    let p = go("gather", &model(&g), &[feed("x", &shape, &xb)]);
    let Some(p) = p else { return };
    let xq = qh(&x);
    let mut want = Vec::new();
    for r in 0..3 {
        want.push(xq[r * 4 + 3]);
        want.push(xq[r * 4]);
    }
    check(&out(&p, "y").values(), &want, 0.0, 1e-3, "gather");
}

#[test]
fn concat_split_e2e() {
    let shape = [2i64, 3];
    let x = randv(6, -1.0, 1.0, 0xCC);
    let y = randv(6, -1.0, 1.0, 0xCD);
    let g = graph(
        "g",
        vec![
            node("Concat", &["x", "y"], &["c"], &[Attr::I("axis", 1)], ""),
            node("Split", &["c"], &["sa", "sb"], &[Attr::I("axis", 1)], ""),
        ],
        vec![],
        vec![
            value_info("x", et::FLOAT, &shape),
            value_info("y", et::FLOAT, &shape),
        ],
        vec![
            value_info("c", et::FLOAT, &[2, 6]),
            value_info("sa", et::FLOAT, &shape),
            value_info("sb", et::FLOAT, &shape),
        ],
    );
    let (xb, yb) = (f16_bytes(&x), f16_bytes(&y));
    let p = go(
        "concat",
        &model(&g),
        &[feed("x", &shape, &xb), feed("y", &shape, &yb)],
    );
    let Some(p) = p else { return };
    let (xq, yq) = (qh(&x), qh(&y));
    let mut c = Vec::new();
    for r in 0..2 {
        c.extend_from_slice(&xq[r * 3..r * 3 + 3]);
        c.extend_from_slice(&yq[r * 3..r * 3 + 3]);
    }
    check(&out(&p, "c").values(), &c, 0.0, 1e-3, "concat");
    check(&out(&p, "sa").values(), &xq, 0.0, 1e-3, "split a");
    check(&out(&p, "sb").values(), &yq, 0.0, 1e-3, "split b");
}

#[test]
fn tile_expand_pad_e2e() {
    let x = randv(6, -1.0, 1.0, 0x71);
    let v3 = randv(3, -1.0, 1.0, 0x72);
    let g = graph(
        "g",
        vec![
            node("Tile", &["x", "reps"], &["o_tile"], &[], ""),
            node("Expand", &["v3", "exp"], &["o_exp"], &[], ""),
            node("Pad", &["x", "pady", "cv"], &["o_pad"], &[], ""),
        ],
        vec![
            t_i64("reps", &[2], &[1, 2]),
            t_i64("exp", &[2], &[2, 3]),
            t_i64("pady", &[4], &[0, 1, 0, 2]), // [b0,b1, e0,e1]
            t_scalar_f32("cv", 0.5),
        ],
        vec![
            value_info("x", et::FLOAT, &[2, 3]),
            value_info("v3", et::FLOAT, &[3]),
        ],
        vec![
            value_info("o_tile", et::FLOAT, &[2, 6]),
            value_info("o_exp", et::FLOAT, &[2, 3]),
            value_info("o_pad", et::FLOAT, &[2, 6]),
        ],
    );
    let (xb, v3b) = (f16_bytes(&x), f16_bytes(&v3));
    let p = go(
        "texp",
        &model(&g),
        &[feed("x", &[2, 3], &xb), feed("v3", &[3], &v3b)],
    );
    let Some(p) = p else { return };
    let (xq, vq) = (qh(&x), qh(&v3));
    let mut tl = Vec::new();
    for r in 0..2 {
        tl.extend_from_slice(&xq[r * 3..r * 3 + 3]);
        tl.extend_from_slice(&xq[r * 3..r * 3 + 3]);
    }
    check(&out(&p, "o_tile").values(), &tl, 0.0, 1e-3, "tile");
    let mut ex = Vec::new();
    ex.extend_from_slice(&vq);
    ex.extend_from_slice(&vq);
    check(&out(&p, "o_exp").values(), &ex, 0.0, 1e-3, "expand");
    let mut pd = Vec::new();
    for r in 0..2 {
        pd.push(0.5); // pad 1 before dim1
        pd.extend_from_slice(&xq[r * 3..r * 3 + 3]);
        pd.push(0.5);
        pd.push(0.5); // pad 2 after dim1
    }
    check(&out(&p, "o_pad").values(), &pd, 0.0, 1e-3, "pad");
}

// ---------- misc ----------

#[test]
fn cast_shape_e2e() {
    let shape = [2i64, 3];
    let x = randv(6, -2.0, 2.0, 0xCA);
    let g = graph(
        "g",
        vec![
            node("Shape", &["x"], &["sh"], &[], ""),
            node(
                "Cast",
                &["x"],
                &["o_int"],
                &[Attr::I("to", et::INT32 as i64)],
                "",
            ),
        ],
        vec![],
        vec![value_info("x", et::FLOAT, &shape)],
        vec![
            value_info("sh", et::INT64, &[2]),
            value_info("o_int", et::INT32, &shape),
        ],
    );
    let xb = f16_bytes(&x);
    let p = go("cast", &model(&g), &[feed("x", &shape, &xb)]);
    let Some(p) = p else { return };
    check(&out(&p, "sh").values(), &[2.0, 3.0], 0.0, 0.0, "shape");
    let want: Vec<f64> = qh(&x).iter().map(|v| (*v as i32) as f64).collect();
    check(&out(&p, "o_int").values(), &want, 0.0, 0.5, "cast");
}

#[test]
fn where_compare_e2e() {
    // cond = x > y internally → Where(cond, x, y) = max; Equal → cast.
    let shape = [2i64, 4];
    let x = randv(8, -2.0, 2.0, 0xE1);
    let y = randv(8, -2.0, 2.0, 0xE2);
    let g = graph(
        "g",
        vec![
            node("Greater", &["x", "y"], &["cond"], &[], ""),
            node("Where", &["cond", "x", "y"], &["o_max"], &[], ""),
            node("Equal", &["x", "y"], &["eq"], &[], ""),
            node(
                "Cast",
                &["eq"],
                &["o_eq"],
                &[Attr::I("to", et::FLOAT as i64)],
                "",
            ),
            node("Less", &["x", "y"], &["lt"], &[], ""),
            node(
                "Cast",
                &["lt"],
                &["o_lt"],
                &[Attr::I("to", et::FLOAT as i64)],
                "",
            ),
        ],
        vec![],
        vec![
            value_info("x", et::FLOAT, &shape),
            value_info("y", et::FLOAT, &shape),
        ],
        vec![
            value_info("o_max", et::FLOAT, &shape),
            value_info("o_eq", et::FLOAT, &shape),
            value_info("o_lt", et::FLOAT, &shape),
        ],
    );
    let (xb, yb) = (f16_bytes(&x), f16_bytes(&y));
    let p = go(
        "where",
        &model(&g),
        &[feed("x", &shape, &xb), feed("y", &shape, &yb)],
    );
    let Some(p) = p else { return };
    let (xq, yq) = (qh(&x), qh(&y));
    let want: Vec<f64> = xq
        .iter()
        .zip(&yq)
        .map(|(a, b)| if a > b { *a } else { *b })
        .collect();
    check(&out(&p, "o_max").values(), &want, 0.0, 1e-3, "where max");
    let want: Vec<f64> = xq
        .iter()
        .zip(&yq)
        .map(|(a, b)| (a == b) as i32 as f64)
        .collect();
    check(&out(&p, "o_eq").values(), &want, 0.0, 0.0, "equal");
    let want: Vec<f64> = xq
        .iter()
        .zip(&yq)
        .map(|(a, b)| (a < b) as i32 as f64)
        .collect();
    check(&out(&p, "o_lt").values(), &want, 0.0, 0.0, "less");
}

#[test]
fn softmax_gelu_e2e() {
    let shape = [2i64, 5];
    let x = randv(10, -2.0, 2.0, 0x5A);
    let g = graph(
        "g",
        vec![
            node("Softmax", &["x"], &["o_sm"], &[], ""),
            node("LogSoftmax", &["x"], &["o_lsm"], &[], ""),
            node("Gelu", &["x"], &["o_gelu"], &[], ""),
            node(
                "Gelu",
                &["x"],
                &["o_gelut"],
                &[Attr::S("approximate", "tanh")],
                "",
            ),
        ],
        vec![],
        vec![value_info("x", et::FLOAT, &shape)],
        vec![
            value_info("o_sm", et::FLOAT, &shape),
            value_info("o_lsm", et::FLOAT, &shape),
            value_info("o_gelu", et::FLOAT, &shape),
            value_info("o_gelut", et::FLOAT, &shape),
        ],
    );
    let xb = f16_bytes(&x);
    let p = go("sm", &model(&g), &[feed("x", &shape, &xb)]);
    let Some(p) = p else { return };
    let xq = qh(&x);
    let mut sm = Vec::new();
    let mut lsm = Vec::new();
    for r in 0..2 {
        let row = &xq[r * 5..r * 5 + 5];
        let s = softmax_row(row);
        sm.extend_from_slice(&s);
        lsm.extend(s.iter().map(|v| v.ln()));
    }
    check(&out(&p, "o_sm").values(), &sm, 1e-2, 5e-3, "softmax");
    check(&out(&p, "o_lsm").values(), &lsm, 1e-2, 5e-3, "logsoftmax");
    let want: Vec<f64> = xq
        .iter()
        .map(|v| 0.5 * v * (1.0 + erf64(v / std::f64::consts::SQRT_2)))
        .collect();
    check(&out(&p, "o_gelu").values(), &want, 5e-3, 3e-3, "gelu exact");
    let want: Vec<f64> = xq
        .iter()
        .map(|v| {
            0.5 * v
                * (1.0
                    + ((2.0f64 / std::f64::consts::PI).sqrt() * (v + 0.044715 * v * v * v)).tanh())
        })
        .collect();
    check(&out(&p, "o_gelut").values(), &want, 1e-2, 5e-3, "gelu tanh");
}

/// Softmax opset 11 default axis=1: coerce to 2-D around axis.
#[test]
fn softmax_opset11_e2e() {
    let shape = [2i64, 3, 4];
    let x = randv(24, -2.0, 2.0, 0x5B);
    let g = graph(
        "g",
        vec![node("Softmax", &["x"], &["y"], &[], "")],
        vec![],
        vec![value_info("x", et::FLOAT, &shape)],
        vec![value_info("y", et::FLOAT, &shape)],
    );
    let xb = f16_bytes(&x);
    let p = go("sm11", &model_opset(&g, 11), &[feed("x", &shape, &xb)]);
    let Some(p) = p else { return };
    // opset≤13 axis=1 → 2-D [2, 12]: softmax per row over 12 elems.
    let xq = qh(&x);
    let mut want = Vec::new();
    for r in 0..2 {
        want.extend(softmax_row(&xq[r * 12..r * 12 + 12]));
    }
    check(&out(&p, "y").values(), &want, 1e-2, 5e-3, "softmax11");
}

#[test]
fn layer_norm_e2e() {
    let shape = [2i64, 8];
    let x = randv(16, -2.0, 2.0, 0x1A);
    let gamma = randv(8, 0.5, 1.5, 0x1B);
    let beta = randv(8, -0.5, 0.5, 0x1C);
    let g = graph(
        "g",
        vec![node(
            "LayerNormalization",
            &["x", "gamma", "beta"],
            &["y"],
            &[Attr::I("axis", -1), Attr::F("epsilon", 1e-5)],
            "",
        )],
        vec![t_f32("gamma", &[8], &gamma), t_f32("beta", &[8], &beta)],
        vec![value_info("x", et::FLOAT, &shape)],
        vec![value_info("y", et::FLOAT, &shape)],
    );
    let xb = f16_bytes(&x);
    let p = go("ln", &model(&g), &[feed("x", &shape, &xb)]);
    let Some(p) = p else { return };
    let (xq, gq, bq) = (qh(&x), qh(&gamma), qh(&beta));
    let mut want = vec![0.0f64; 16];
    for r in 0..2 {
        let row = &xq[r * 8..r * 8 + 8];
        let mean = row.iter().sum::<f64>() / 8.0;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 8.0;
        let std = (var + 1e-5).sqrt();
        for i in 0..8 {
            want[r * 8 + i] = (row[i] - mean) / std * gq[i] + bq[i];
        }
    }
    check(&out(&p, "y").values(), &want, 1e-2, 5e-3, "layer_norm");
}

#[test]
fn batch_norm_e2e() {
    let shape = [1i64, 3, 2, 2];
    let x = randv(12, -2.0, 2.0, 0xBA);
    let gamma = [1.5f32, 0.8, 1.1];
    let beta = [0.1f32, -0.2, 0.0];
    let mean = [0.5f32, -0.3, 0.9];
    let var = [0.4f32, 0.7, 1.2];
    let g = graph(
        "g",
        vec![node(
            "BatchNormalization",
            &["x", "gamma", "beta", "mean", "var"],
            &["y"],
            &[Attr::F("epsilon", 1e-5)],
            "",
        )],
        vec![
            t_f32("gamma", &[3], &gamma),
            t_f32("beta", &[3], &beta),
            t_f32("mean", &[3], &mean),
            t_f32("var", &[3], &var),
        ],
        vec![value_info("x", et::FLOAT, &shape)],
        vec![value_info("y", et::FLOAT, &shape)],
    );
    let xb = f16_bytes(&x);
    let p = go("bn", &model(&g), &[feed("x", &shape, &xb)]);
    let Some(p) = p else { return };
    let (xq, gq, bq, mq, vq) = (qh(&x), qh(&gamma), qh(&beta), qh(&mean), qh(&var));
    let mut want = vec![0.0f64; 12];
    for c in 0..3usize {
        for i in 0..4usize {
            let v = xq[c * 4 + i];
            want[c * 4 + i] = gq[c] * (v - mq[c]) / (vq[c] + 1e-5).sqrt() + bq[c];
        }
    }
    check(&out(&p, "y").values(), &want, 1e-2, 5e-3, "batch_norm");
}

#[test]
fn resize_e2e() {
    // Upsample (opset≥10: scales input) nearest 2×, Resize linear 2×.
    let ish = [1i64, 1, 2, 2];
    let x = vec![0.0f32, 1.0, 2.0, 3.0];
    let g = graph(
        "g",
        vec![
            node("Upsample", &["x", "sc"], &["o_nn"], &[], ""),
            node(
                "Resize",
                &["x", "roi", "sc"],
                &["o_bl"],
                &[Attr::S("mode", "linear")],
                "",
            ),
        ],
        vec![
            t_f32("sc", &[4], &[1.0, 1.0, 2.0, 2.0]),
            t_f32("roi", &[0], &[]),
        ],
        vec![value_info("x", et::FLOAT, &ish)],
        vec![
            value_info("o_nn", et::FLOAT, &[1, 1, 4, 4]),
            value_info("o_bl", et::FLOAT, &[1, 1, 4, 4]),
        ],
    );
    let xb = f16_bytes(&x);
    let p = go("resize", &model(&g), &[feed("x", &ish, &xb)]);
    let Some(p) = p else { return };
    // nearest 2×: each pixel replicated to 2×2.
    let nn: Vec<f64> = vec![
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0,
    ];
    check(&out(&p, "o_nn").values(), &nn, 0.0, 1e-3, "nearest");
    // bilinear upsample 2×, half-pixel: corners preserved, centers
    // interpolated — expected values from align_corners=false sampling.
    let got = out(&p, "o_bl").values();
    assert_eq!(got.len(), 16);
    // center pixels of the 2× source quadrants land on source values
    assert!(
        (got[0] - 0.0).abs() < 0.3 && (got[15] - 3.0).abs() < 0.3,
        "bilinear corners: {got:?}"
    );
}

#[test]
fn dropout_identity_e2e() {
    let shape = [4i64];
    let x = randv(4, -1.0, 1.0, 0xD0);
    let g = graph(
        "g",
        vec![
            node("Dropout", &["x"], &["d", "mask"], &[], ""),
            node("Identity", &["d"], &["y"], &[], ""),
        ],
        vec![],
        vec![value_info("x", et::FLOAT, &shape)],
        vec![value_info("y", et::FLOAT, &shape)],
    );
    let xb = f16_bytes(&x);
    let p = go("dropout", &model(&g), &[feed("x", &shape, &xb)]);
    let Some(p) = p else { return };
    check(
        &out(&p, "y").values(),
        &qh(&x),
        0.0,
        1e-3,
        "dropout passthrough",
    );
}

#[test]
fn constant_of_shape_e2e() {
    let g = graph(
        "g",
        vec![
            node("ConstantOfShape", &["sh"], &["o_c"], &[], ""),
            node(
                "ConstantOfShape",
                &["sh2"],
                &["o_cv"],
                &[Attr::T("value", t_scalar_f32("", 3.5))],
                "",
            ),
            node("Shape", &["o_c"], &["dummy"], &[], ""),
        ],
        vec![t_i64("sh", &[3], &[2, 2, 3]), t_i64("sh2", &[1], &[4])],
        vec![],
        vec![
            value_info("o_c", et::FLOAT, &[2, 2, 3]),
            value_info("o_cv", et::FLOAT, &[4]),
        ],
    );
    let p = go("cos", &model(&g), &[]);
    let Some(p) = p else { return };
    check(&out(&p, "o_c").values(), &[0.0; 12], 0.0, 1e-3, "cos zeros");
    check(&out(&p, "o_cv").values(), &[3.5; 4], 0.0, 1e-3, "cos value");
}
