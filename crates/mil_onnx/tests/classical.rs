//! Classical-ML tests: the JSON interchange schema (`convert_classical_json`)
//! and `ai.onnx.ml` nodes embedded in ONNX graphs, both verified end-to-end
//! through `coremlc` + `mil_infer` where available.
//!
//! Conversion (decode→MIL) is unconditional; execution is CoreML-gated.

mod common;

use common::*;
use mil_infer::Input;

/// Convert classical JSON bytes and (if coremlc exists) predict.
fn go_json(tag: &str, json: &[u8], feeds: &[Input<'_>]) -> Option<mil_infer::Prediction> {
    let built = mil_onnx::convert_classical_json(json)
        .unwrap_or_else(|e| panic!("{tag}: convert_classical_json: {e}"));
    run(tag, &built, feeds)
}

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

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax_row(v: &[f64]) -> Vec<f64> {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = v.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

// ================= JSON schema — rejection =================

#[test]
fn json_rejects_garbage() {
    assert!(mil_onnx::convert_classical_json(b"{not json").is_err());
    // valid json, missing kind
    let e = mil_onnx::convert_classical_json(br#"{"n_features": 3}"#)
        .err()
        .map(|e| e.to_string())
        .unwrap();
    assert!(e.contains("kind"), "{e}");
    // unknown kind
    let e = mil_onnx::convert_classical_json(br#"{"kind": "svm", "n_features": 3}"#)
        .err()
        .map(|e| e.to_string())
        .unwrap();
    assert!(e.contains("svm"), "{e}");
    // n_features <= 0
    assert!(mil_onnx::convert_classical_json(
        br#"{"kind": "glm_regressor", "n_features": 0, "weights": [1.0]}"#
    )
    .is_err());
}

#[test]
fn json_glm_rejects_bad_weights() {
    // weights length != n_features
    let e = mil_onnx::convert_classical_json(
        br#"{"kind": "glm_regressor", "n_features": 3, "weights": [1.0, 2.0]}"#,
    )
    .err()
    .map(|e| e.to_string())
    .unwrap();
    assert!(e.contains("weights"), "{e}");
    // intercept wrong length
    let e = mil_onnx::convert_classical_json(
        br#"{"kind": "glm_regressor", "n_features": 2,
            "weights": [1.0, 2.0], "intercept": [1.0, 2.0]}"#,
    )
    .err()
    .map(|e| e.to_string())
    .unwrap();
    assert!(e.contains("intercept"), "{e}");
    // unknown post_transform
    let e = mil_onnx::convert_classical_json(
        br#"{"kind": "glm_regressor", "n_features": 2,
            "weights": [1.0, 2.0], "post_transform": "relu"}"#,
    )
    .err()
    .map(|e| e.to_string())
    .unwrap();
    assert!(e.contains("relu"), "{e}");
}

#[test]
fn json_tree_rejects_schema_violations() {
    // dangling child reference
    let e = mil_onnx::convert_classical_json(
        br#"{"kind": "tree_ensemble_regressor", "n_features": 2,
            "trees": [{"nodes": [
                {"id": 0, "feature": 0, "threshold": 1.0, "mode": "leq",
                 "left": 1, "right": 99},
                {"id": 1, "value": [0.5]}
            ]}]}"#,
    )
    .err()
    .map(|e| e.to_string())
    .unwrap();
    assert!(e.contains("99"), "{e}");
    // unknown branch mode
    let e = mil_onnx::convert_classical_json(
        br#"{"kind": "tree_ensemble_regressor", "n_features": 2,
            "trees": [{"nodes": [
                {"id": 0, "feature": 0, "threshold": 1.0, "mode": "fuzzy",
                 "left": 1, "right": 2},
                {"id": 1, "value": [0.5]},
                {"id": 2, "value": [1.5]}
            ]}]}"#,
    )
    .err()
    .map(|e| e.to_string())
    .unwrap();
    assert!(e.contains("fuzzy"), "{e}");
    // leaf value width != n_outputs
    let e = mil_onnx::convert_classical_json(
        br#"{"kind": "tree_ensemble_classifier", "n_features": 2,
            "class_labels": [0, 1],
            "trees": [{"nodes": [{"id": 0, "value": [0.5]}]}]}"#,
    )
    .err()
    .map(|e| e.to_string())
    .unwrap();
    assert!(e.contains("1 elems, expected 2"), "{e}");
    // unknown aggregate
    let e = mil_onnx::convert_classical_json(
        br#"{"kind": "tree_ensemble_regressor", "n_features": 2,
            "aggregate": "max",
            "trees": [{"nodes": [{"id": 0, "value": [0.5]}]}]}"#,
    )
    .err()
    .map(|e| e.to_string())
    .unwrap();
    assert!(e.contains("max"), "{e}");
}

// ================= JSON GLM =================

#[test]
fn json_glm_regressor_e2e() {
    let json = br#"{
        "kind": "glm_regressor",
        "n_features": 3,
        "input_name": "features",
        "weights": [0.5, -1.0, 2.0],
        "intercept": [0.25]
    }"#;
    let x = randv(3, -2.0, 2.0, 0xA1);
    let xb = f16_bytes(&x);
    let p = go_json("glm_reg", json, &[feed("features", &[3], &xb)]);
    let Some(p) = p else { return };
    let xq = qh(&x);
    let want = vec![0.5 * xq[0] - xq[1] + 2.0 * xq[2] + 0.25];
    check(
        &out(&p, "scores").values(),
        &want,
        1e-2,
        5e-3,
        "glm regressor",
    );
    assert_eq!(out(&p, "scores").shape, vec![1]);
}

#[test]
fn json_glm_classifier_e2e() {
    // 2-class logistic classifier: label = argmax(sigmoid(scores)).
    let json = br#"{
        "kind": "glm_classifier",
        "n_features": 4,
        "weights": [[1.0, 0.0, -1.0, 0.5],
                    [-1.0, 0.25, 1.0, 0.0]],
        "intercept": [0.1, -0.2],
        "class_labels": [7, 42],
        "post_transform": "logistic"
    }"#;
    let x = randv(4, -1.5, 1.5, 0xA2);
    let xb = f16_bytes(&x);
    let p = go_json("glm_clf", json, &[feed("features", &[4], &xb)]);
    let Some(p) = p else { return };
    let xq = qh(&x);
    let s0 = xq[0] - xq[2] + 0.5 * xq[3] + 0.1;
    let s1 = -xq[0] + 0.25 * xq[1] + xq[2] - 0.2;
    let want = vec![sigmoid(s0), sigmoid(s1)];
    check(
        &out(&p, "scores").values(),
        &want,
        1e-2,
        5e-3,
        "glm clf scores",
    );
    let want_label = if sigmoid(s0) >= sigmoid(s1) {
        7.0
    } else {
        42.0
    };
    check(
        &out(&p, "label").values(),
        &[want_label],
        0.0,
        0.0,
        "glm clf label",
    );
}

#[test]
fn json_glm_softmax_probit_e2e() {
    // softmax over 3 classes.
    let json = br#"{
        "kind": "glm_classifier",
        "n_features": 2,
        "weights": [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]],
        "class_labels": [0, 1, 2],
        "post_transform": "softmax"
    }"#;
    let x = randv(2, -2.0, 2.0, 0xA3);
    let xb = f16_bytes(&x);
    let p = go_json("glm_sm", json, &[feed("features", &[2], &xb)]);
    if let Some(p) = p {
        let xq = qh(&x);
        let want = softmax_row(&[xq[0], xq[1], 0.5 * (xq[0] - xq[1])]);
        check(
            &out(&p, "scores").values(),
            &want,
            1e-2,
            5e-3,
            "glm softmax",
        );
    }
    // probit (Φ) on a regressor — exercises the erf path.
    let json = br#"{
        "kind": "glm_regressor",
        "n_features": 2,
        "weights": [0.8, -0.4],
        "intercept": [0.1],
        "post_transform": "probit"
    }"#;
    let p = go_json("glm_pb", json, &[feed("features", &[2], &xb)]);
    let Some(p) = p else { return };
    let xq = qh(&x);
    let z = 0.8 * xq[0] - 0.4 * xq[1] + 0.1;
    // Φ(z) = 0.5(1 + erf(z/√2)) — erf via Abramowitz-Stegun like ops_e2e.
    let zs = z * std::f64::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * zs.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-zs * zs).exp();
    let erf = if zs >= 0.0 { y } else { -y };
    let want = vec![0.5 * (1.0 + erf)];
    check(&out(&p, "scores").values(), &want, 5e-3, 5e-3, "glm probit");
}

// ================= JSON tree ensembles =================

/// A two-level tree: root splits feature 0 at 0.5 (leq); left leaf favours
/// class 0, right leaf favours class 1.
fn json_tree_clf(post: &str) -> Vec<u8> {
    format!(
        r#"{{
        "kind": "tree_ensemble_classifier",
        "n_features": 3,
        "n_outputs": 2,
        "class_labels": [10, 20],
        "post_transform": "{post}",
        "trees": [{{"nodes": [
            {{"id": 0, "feature": 0, "threshold": 0.5, "mode": "leq",
             "left": 1, "right": 2}},
            {{"id": 1, "value": [0.9, 0.1]}},
            {{"id": 2, "value": [0.2, 0.8]}}
        ]}}]
    }}"#
    )
    .into_bytes()
}

#[test]
fn json_tree_classifier_e2e() {
    // x0 <= 0.5 → left leaf [0.9, 0.1] → label 10.
    let json = json_tree_clf("softmax");
    let x = [0.1f32, 5.0, -1.0];
    let xb = f16_bytes(&x);
    let p = go_json("tree_clf_l", &json, &[feed("features", &[3], &xb)]);
    if let Some(p) = p {
        let want = softmax_row(&[0.9, 0.1]);
        check(
            &out(&p, "scores").values(),
            &want,
            1e-2,
            5e-3,
            "tree clf left",
        );
        check(
            &out(&p, "label").values(),
            &[10.0],
            0.0,
            0.0,
            "tree clf label",
        );
    }
    // x0 > 0.5 → right leaf → label 20.
    let x = [2.0f32, 0.0, 0.0];
    let xb = f16_bytes(&x);
    let p = go_json("tree_clf_r", &json, &[feed("features", &[3], &xb)]);
    let Some(p) = p else { return };
    let want = softmax_row(&[0.2, 0.8]);
    check(
        &out(&p, "scores").values(),
        &want,
        1e-2,
        5e-3,
        "tree clf right",
    );
    check(
        &out(&p, "label").values(),
        &[20.0],
        0.0,
        0.0,
        "tree clf label r",
    );
}

#[test]
fn json_tree_regressor_e2e() {
    // Three trees, mixed modes (leq on f0, gt on f1, lt on f2); sum aggregate.
    let json = br#"{
        "kind": "tree_ensemble_regressor",
        "n_features": 3,
        "n_outputs": 1,
        "base_values": [0.5],
        "trees": [
            {"weight": 1.0, "nodes": [
                {"id": 0, "feature": 0, "threshold": 1.0, "mode": "leq",
                 "left": 1, "right": 2},
                {"id": 1, "value": [2.0]},
                {"id": 2, "value": [-1.0]}
            ]},
            {"weight": 0.5, "nodes": [
                {"id": 0, "feature": 1, "threshold": -0.5, "mode": "gt",
                 "left": 1, "right": 2},
                {"id": 1, "value": [4.0]},
                {"id": 2, "value": [0.0]}
            ]},
            {"weight": 1.0, "nodes": [
                {"id": 0, "feature": 2, "threshold": 3.0, "mode": "lt",
                 "left": 1, "right": 2},
                {"id": 1, "value": [1.0]},
                {"id": 2, "value": [-2.0]}
            ]}
        ]
    }"#;
    // x = [0.5, 1.0, 5.0]: t0 left(+2), t1 1.0>-0.5 true→+0.5*4=+2, t2 5<3 false→-2
    let x = [0.5f32, 1.0, 5.0];
    let xb = f16_bytes(&x);
    let p = go_json("tree_reg", json, &[feed("features", &[3], &xb)]);
    let Some(p) = p else { return };
    let want = vec![0.5 + 2.0 + 2.0 - 2.0];
    check(
        &out(&p, "scores").values(),
        &want,
        1e-2,
        5e-3,
        "tree reg sum",
    );
}

#[test]
fn json_tree_average_and_stump_e2e() {
    // aggregate "average": two stump-only trees (j==0 const path).
    let json = br#"{
        "kind": "tree_ensemble_regressor",
        "n_features": 2,
        "n_outputs": 1,
        "base_values": [0.5],
        "aggregate": "average",
        "trees": [
            {"nodes": [{"id": 0, "value": [2.0]}]},
            {"nodes": [{"id": 0, "value": [4.0]}]}
        ]
    }"#;
    let x = [1.0f32, -1.0];
    let xb = f16_bytes(&x);
    let p = go_json("tree_stump", json, &[feed("features", &[2], &xb)]);
    let Some(p) = p else { return };
    // (2 + 4)/2 + 0.5 = 3.5 — input-independent.
    check(
        &out(&p, "scores").values(),
        &[3.5],
        1e-2,
        5e-3,
        "tree stumps avg",
    );
}

// ================= ai.onnx.ml nodes =================

#[test]
fn ml_linear_regressor_e2e() {
    let g = graph(
        "g",
        vec![node_dom(
            "LinearRegressor",
            &["x"],
            &["y"],
            &[
                Attr::Fs("coefficients", &[0.5, -1.0, 2.0]),
                Attr::Fs("intercepts", &[0.25]),
            ],
            "lr",
            "ai.onnx.ml",
        )],
        vec![],
        vec![value_info("x", et::FLOAT, &[1, 3])],
        vec![value_info("y", et::FLOAT, &[1, 1])],
    );
    let x = randv(3, -2.0, 2.0, 0xB1);
    let xb = f16_bytes(&x);
    let p = go("ml_lr", &model(&g), &[feed("x", &[1, 3], &xb)]);
    let Some(p) = p else { return };
    let xq = qh(&x);
    let want = vec![0.5 * xq[0] - xq[1] + 2.0 * xq[2] + 0.25];
    check(&out(&p, "y").values(), &want, 1e-2, 5e-3, "ml linreg");
}

#[test]
fn ml_linear_classifier_binary_e2e() {
    // Binary sklearn LR export: K=1 coefficient row, 2 labels → the
    // converter must synthesize the [-s, s] two-class score matrix.
    let g = graph(
        "g",
        vec![node_dom(
            "LinearClassifier",
            &["x"],
            &["y_label", "y_scores"],
            &[
                Attr::Fs("coefficients", &[1.0, -0.5]),
                Attr::Fs("intercepts", &[0.1]),
                Attr::Ints("classlabels_ints", &[0, 1]),
                Attr::S("post_transform", "SOFTMAX"),
            ],
            "lc",
            "ai.onnx.ml",
        )],
        vec![],
        vec![value_info("x", et::FLOAT, &[1, 2])],
        vec![
            value_info("y_label", et::INT64, &[1]),
            value_info("y_scores", et::FLOAT, &[1, 2]),
        ],
    );
    let x = randv(2, -2.0, 2.0, 0xB2);
    let xb = f16_bytes(&x);
    let p = go("ml_lc", &model(&g), &[feed("x", &[1, 2], &xb)]);
    let Some(p) = p else { return };
    let xq = qh(&x);
    let s = xq[0] - 0.5 * xq[1] + 0.1;
    let want = softmax_row(&[-s, s]);
    check(
        &out(&p, "y_scores").values(),
        &want,
        1e-2,
        5e-3,
        "ml linclf scores",
    );
    let want_label = if s >= 0.0 { 1.0 } else { 0.0 };
    check(
        &out(&p, "y_label").values(),
        &[want_label],
        0.0,
        0.0,
        "ml linclf label",
    );
}

#[test]
fn ml_tree_ensemble_classifier_e2e() {
    // One tree, two classes. Node 0: f0 <= 0.5 → node 1 else node 2.
    // class_* arrays map (tree,node,class) → leaf weight.
    let g = graph(
        "g",
        vec![node_dom(
            "TreeEnsembleClassifier",
            &["x"],
            &["y", "z"],
            &[
                Attr::Ints("nodes_treeids", &[0, 0, 0]),
                Attr::Ints("nodes_nodeids", &[0, 1, 2]),
                Attr::Ints("nodes_featureids", &[0, 0, 0]),
                Attr::Fs("nodes_values", &[0.5, 0.0, 0.0]),
                Attr::Ss("nodes_modes", &["BRANCH_LEQ", "LEAF", "LEAF"]),
                Attr::Ints("nodes_truenodeids", &[1, 0, 0]),
                Attr::Ints("nodes_falsenodeids", &[2, 0, 0]),
                Attr::Ints("class_treeids", &[0, 0, 0, 0]),
                Attr::Ints("class_nodeids", &[1, 1, 2, 2]),
                Attr::Ints("class_ids", &[0, 1, 0, 1]),
                Attr::Fs("class_weights", &[0.9, 0.1, 0.2, 0.8]),
                Attr::Ints("classlabels_ints", &[10, 20]),
                Attr::S("post_transform", "SOFTMAX"),
            ],
            "tec",
            "ai.onnx.ml",
        )],
        vec![],
        vec![value_info("x", et::FLOAT, &[1, 3])],
        vec![
            value_info("y", et::INT64, &[1]),
            value_info("z", et::FLOAT, &[1, 2]),
        ],
    );
    // x0 <= 0.5 → leaf 1 weights [0.9, 0.1].
    let x = [0.1f32, 9.0, -3.0];
    let xb = f16_bytes(&x);
    let p = go("ml_tec_l", &model(&g), &[feed("x", &[1, 3], &xb)]);
    if let Some(p) = p {
        let want = softmax_row(&[0.9, 0.1]);
        check(&out(&p, "z").values(), &want, 1e-2, 5e-3, "ml tec z");
        check(&out(&p, "y").values(), &[10.0], 0.0, 0.0, "ml tec y");
    }
    // x0 > 0.5 → leaf 2 [0.2, 0.8] → label 20.
    let x = [3.0f32, 0.0, 0.0];
    let xb = f16_bytes(&x);
    let p = go("ml_tec_r", &model(&g), &[feed("x", &[1, 3], &xb)]);
    let Some(p) = p else { return };
    let want = softmax_row(&[0.2, 0.8]);
    check(&out(&p, "z").values(), &want, 1e-2, 5e-3, "ml tec z r");
    check(&out(&p, "y").values(), &[20.0], 0.0, 0.0, "ml tec y r");
}

#[test]
fn ml_tree_ensemble_regressor_e2e() {
    // Two trees; AVERAGE aggregation over targets.
    let g = graph(
        "g",
        vec![node_dom(
            "TreeEnsembleRegressor",
            &["x"],
            &["y"],
            &[
                Attr::I("n_targets", 1),
                Attr::Ints("nodes_treeids", &[0, 0, 0, 1, 1, 1]),
                Attr::Ints("nodes_nodeids", &[0, 1, 2, 0, 1, 2]),
                Attr::Ints("nodes_featureids", &[0, 0, 0, 1, 0, 0]),
                Attr::Fs("nodes_values", &[1.0, 0.0, 0.0, -0.5, 0.0, 0.0]),
                Attr::Ss(
                    "nodes_modes",
                    &["BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_GT", "LEAF", "LEAF"],
                ),
                Attr::Ints("nodes_truenodeids", &[1, 0, 0, 1, 0, 0]),
                Attr::Ints("nodes_falsenodeids", &[2, 0, 0, 2, 0, 0]),
                Attr::Ints("target_treeids", &[0, 0, 1, 1]),
                Attr::Ints("target_nodeids", &[1, 2, 1, 2]),
                Attr::Ints("target_ids", &[0, 0, 0, 0]),
                Attr::Fs("target_weights", &[2.0, -1.0, 4.0, 0.0]),
                Attr::S("aggregate_function", "AVERAGE"),
                Attr::Fs("base_values", &[0.5]),
            ],
            "ter",
            "ai.onnx.ml",
        )],
        vec![],
        vec![value_info("x", et::FLOAT, &[1, 2])],
        vec![value_info("y", et::FLOAT, &[1, 1])],
    );
    // x = [0.5, 1.0]: t0 → leaf1 (+2), t1 → 1.0 > -0.5 → leaf1 (+4).
    // AVERAGE: (6)/2 = 3 + base 0.5 = 3.5.
    let x = [0.5f32, 1.0];
    let xb = f16_bytes(&x);
    let p = go("ml_ter", &model(&g), &[feed("x", &[1, 2], &xb)]);
    let Some(p) = p else { return };
    check(&out(&p, "y").values(), &[3.5], 1e-2, 5e-3, "ml ter avg");
}

#[test]
fn ml_normalizer_e2e() {
    for (norm, attr) in [
        ("L1", Attr::S("norm", "L1")),
        ("L2", Attr::S("norm", "L2")),
        ("MAX", Attr::S("norm", "MAX")),
    ] {
        let g = graph(
            "g",
            vec![node_dom(
                "Normalizer",
                &["x"],
                &["y"],
                &[attr],
                "norm",
                "ai.onnx.ml",
            )],
            vec![],
            vec![value_info("x", et::FLOAT, &[2, 3])],
            vec![value_info("y", et::FLOAT, &[2, 3])],
        );
        let x = randv(6, -3.0, 3.0, 0xB5);
        let xb = f16_bytes(&x);
        let p = go(
            &format!("norm_{norm}"),
            &model(&g),
            &[feed("x", &[2, 3], &xb)],
        );
        let Some(p) = p else { continue };
        let xq = qh(&x);
        let mut want = Vec::new();
        for r in 0..2 {
            let row = &xq[r * 3..r * 3 + 3];
            let d = match norm {
                "L1" => row.iter().map(|v| v.abs()).sum::<f64>(),
                "L2" => row.iter().map(|v| v * v).sum::<f64>().sqrt(),
                _ => row.iter().cloned().map(f64::abs).fold(0.0, f64::max),
            };
            want.extend(row.iter().map(|v| v / d));
        }
        check(&out(&p, "y").values(), &want, 2e-2, 1e-2, "normalizer");
    }
}

#[test]
fn ml_scaler_e2e() {
    let g = graph(
        "g",
        vec![node_dom(
            "Scaler",
            &["x"],
            &["y"],
            &[
                Attr::Fs("offset", &[1.0, -1.0, 0.5]),
                Attr::Fs("scale", &[2.0, 0.5, -1.0]),
            ],
            "sc",
            "ai.onnx.ml",
        )],
        vec![],
        vec![value_info("x", et::FLOAT, &[1, 3])],
        vec![value_info("y", et::FLOAT, &[1, 3])],
    );
    let x = randv(3, -2.0, 2.0, 0xB6);
    let xb = f16_bytes(&x);
    let p = go("ml_scaler", &model(&g), &[feed("x", &[1, 3], &xb)]);
    let Some(p) = p else { return };
    let xq = qh(&x);
    let off = [1.0, -1.0, 0.5];
    let sc = [2.0, 0.5, -1.0];
    let want: Vec<f64> = (0..3).map(|i| (xq[i] - off[i]) * sc[i]).collect();
    check(&out(&p, "y").values(), &want, 1e-2, 5e-3, "scaler");
}

#[test]
fn ml_unsupported_op_rejected() {
    let g = graph(
        "g",
        vec![node_dom(
            "OneHotEncoder",
            &["x"],
            &["y"],
            &[Attr::Fs("cats_floats", &[1.0, 2.0])],
            "ohe",
            "ai.onnx.ml",
        )],
        vec![],
        vec![value_info("x", et::FLOAT, &[1])],
        vec![value_info("y", et::FLOAT, &[2])],
    );
    let e = convert(&model(&g)).err().map(|e| e.to_string()).unwrap();
    assert!(e.contains("OneHotEncoder"), "{e}");
}
