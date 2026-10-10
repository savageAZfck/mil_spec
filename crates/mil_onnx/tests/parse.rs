//! Parser + rejection tests: malformed wire streams, unknown ops,
//! missing inputs — all must fail with structured `OnnxError`s.

mod common;

use common::*;
use mil_onnx::{convert_bytes, OnnxError};

/// Minimal valid model: `y = relu(x)` on shape [3].
fn relu_model() -> Vec<u8> {
    let g = graph(
        "g",
        vec![node("Relu", &["x"], &["y"], &[], "r0")],
        vec![],
        vec![value_info("x", et::FLOAT, &[3])],
        vec![value_info("y", et::FLOAT, &[3])],
    );
    model(&g)
}

#[test]
fn decode_minimal_model() {
    let m = mil_onnx::ModelProto::decode(&relu_model()).unwrap();
    assert_eq!(m.ir_version, 8);
    assert_eq!(m.producer, "mil_onnx test encoder");
    assert_eq!(m.opset(""), 13);
    assert_eq!(m.opset("ai.onnx.ml"), 3);
    assert_eq!(m.graph.nodes.len(), 1);
    assert_eq!(m.graph.nodes[0].op_type, "Relu");
    assert_eq!(m.graph.nodes[0].inputs, vec!["x"]);
    assert_eq!(m.graph.nodes[0].outputs, vec!["y"]);
    assert_eq!(m.graph.inputs[0].shape, Some(vec![mil_onnx::Dim::Value(3)]));
}

#[test]
fn decode_attrs_all_kinds() {
    let g = graph(
        "g",
        vec![node(
            "Conv",
            &["x", "w"],
            &["y"],
            &[
                Attr::Ints("strides", &[2, 2]),
                Attr::Ints("pads", &[0, 0, 0, 0]),
                Attr::I("group", 2),
                Attr::F("eps", 0.5),
                Attr::S("auto_pad", "SAME_UPPER"),
                Attr::Ss("names", &["a", "b"]),
            ],
            "c0",
        )],
        vec![t_f32("w", &[2, 4, 2, 2], &[0.0; 32])],
        vec![value_info("x", et::FLOAT, &[1, 4, 8, 8])],
        vec![value_info("y", et::FLOAT, &[1, 2, 4, 4])],
    );
    let m = mil_onnx::ModelProto::decode(&model(&g)).unwrap();
    let n = &m.graph.nodes[0];
    assert_eq!(n.op_type, "Conv");
    assert_eq!(n.attr_ints("strides"), vec![2, 2]);
    assert_eq!(n.attr_ints("pads"), vec![0, 0, 0, 0]);
    assert_eq!(n.attr_i("group", 0), 2);
    assert!((n.attr_f("eps", 0.0) - 0.5).abs() < 1e-6);
    assert_eq!(n.attr_str("auto_pad").as_deref(), Some("SAME_UPPER"));
    assert_eq!(n.name, "c0");
    let tp = &m.graph.initializers[0];
    assert_eq!(tp.dims, vec![2, 4, 2, 2]);
    assert_eq!(tp.data_type, et::FLOAT);
    assert_eq!(tp.as_f32().unwrap().len(), 32);
}

#[test]
fn decode_int_and_scalar_tensors() {
    let g = graph(
        "g",
        vec![],
        vec![
            t_i64("ia", &[3], &[1, -2, 3]),
            t_i64_packed_unused(),
            t_scalar_f32("sf", 2.5),
            t_f32_list("fl", &[2], &[0.5, 1.5]),
        ],
        vec![value_info("x", et::FLOAT, &[1])],
        vec![value_info("x", et::FLOAT, &[1])],
    );
    let m = mil_onnx::ModelProto::decode(&model(&g)).unwrap();
    let ia = m.graph.initializers[0].as_i64().unwrap();
    assert_eq!(ia, vec![1, -2, 3]);
    let packed = m.graph.initializers[1].as_i64().unwrap();
    assert_eq!(packed, vec![7, 8, 9]);
    let sf = m.graph.initializers[2].as_f32().unwrap();
    assert!((sf[0] - 2.5).abs() < 1e-6);
    let fl = m.graph.initializers[3].as_f32().unwrap();
    assert!((fl[0] - 0.5).abs() < 1e-6 && (fl[1] - 1.5).abs() < 1e-6);
}

/// packed int64 tensor (dims + data both packed).
fn t_i64_packed_unused() -> Vec<u8> {
    let mut m = Vec::new();
    f_packed_i64(1, &[3], &mut m);
    f_i(2, et::INT64 as i64, &mut m);
    f_packed_i64(7, &[7, 8, 9], &mut m);
    f_str(8, "ip", &mut m);
    m
}

#[test]
fn reject_garbage() {
    let junk: Vec<u8> = vec![0xff, 0xff, 0xff, 0xff, 0xff, 0x00];
    assert!(matches!(
        mil_onnx::ModelProto::decode(&junk),
        Err(OnnxError::Malformed { .. })
    ));
}

#[test]
fn reject_empty() {
    // An empty stream decodes to an empty message → missing graph.
    assert!(matches!(
        mil_onnx::ModelProto::decode(&[]),
        Err(OnnxError::Malformed { .. })
    ));
}

#[test]
fn reject_bad_wire_type() {
    // field 1, wire type 3 (SGROUP) — must be rejected, not skipped.
    let buf = vec![(1 << 3) | 3, 0x01];
    assert!(matches!(
        mil_onnx::ModelProto::decode(&buf),
        Err(OnnxError::Malformed { .. })
    ));
    // wire type 7 invalid too
    let buf = vec![(1 << 3) | 7, 0x01];
    assert!(matches!(
        mil_onnx::ModelProto::decode(&buf),
        Err(OnnxError::Malformed { .. })
    ));
}

#[test]
fn reject_truncated() {
    let good = relu_model();
    for cut in [1, 3, good.len() / 2, good.len() - 1] {
        assert!(
            mil_onnx::ModelProto::decode(&good[..cut]).is_err(),
            "truncated at {cut} should fail"
        );
    }
}

#[test]
fn reject_len_overrun() {
    // field 7 len=200 but only 2 bytes follow
    let mut buf = Vec::new();
    varint((7 << 3) | 2, &mut buf);
    varint(200, &mut buf);
    buf.extend_from_slice(&[0x0a, 0x00]);
    assert!(matches!(
        mil_onnx::ModelProto::decode(&buf),
        Err(OnnxError::Malformed { .. })
    ));
}

#[test]
fn reject_unknown_op() {
    let g = graph(
        "g",
        vec![node("Frobnicate", &["x"], &["y"], &[], "frob_node")],
        vec![],
        vec![value_info("x", et::FLOAT, &[3])],
        vec![value_info("y", et::FLOAT, &[3])],
    );
    match convert_bytes(&model(&g)) {
        Err(OnnxError::UnknownOp { op, node, domain }) => {
            assert_eq!(op, "Frobnicate");
            assert_eq!(node, "frob_node");
            assert_eq!(domain, "");
        }
        other => panic!(
            "expected UnknownOp, got {}",
            other
                .err()
                .map(|e| e.to_string())
                .unwrap_or_else(|| "<ok>".into())
        ),
    }
}

#[test]
fn reject_unknown_domain() {
    let g = graph(
        "g",
        vec![node_dom("Foo", &["x"], &["y"], &[], "n", "com.example")],
        vec![],
        vec![value_info("x", et::FLOAT, &[3])],
        vec![value_info("y", et::FLOAT, &[3])],
    );
    match convert_bytes(&model(&g)) {
        Err(OnnxError::UnknownOp { op, domain, .. }) => {
            assert_eq!(op, "Foo");
            assert_eq!(domain, "com.example");
        }
        other => panic!(
            "expected UnknownOp, got {}",
            other
                .err()
                .map(|e| e.to_string())
                .unwrap_or_else(|| "<ok>".into())
        ),
    }
}

#[test]
fn reject_missing_input() {
    // MatMul with W absent from both initializers and inputs.
    let g = graph(
        "g",
        vec![node("MatMul", &["x", "w_ghost"], &["y"], &[], "mm0")],
        vec![],
        vec![value_info("x", et::FLOAT, &[2, 3])],
        vec![value_info("y", et::FLOAT, &[2, 3])],
    );
    match convert_bytes(&model(&g)) {
        Err(OnnxError::MissingInput { name, node }) => {
            assert_eq!(name, "w_ghost");
            assert_eq!(node, "mm0");
        }
        other => panic!(
            "expected MissingInput, got {}",
            other
                .err()
                .map(|e| e.to_string())
                .unwrap_or_else(|| "<ok>".into())
        ),
    }
}

#[test]
fn reject_symbolic_dim() {
    let g = graph(
        "g",
        vec![node("Relu", &["x"], &["y"], &[], "")],
        vec![],
        vec![value_info_param("x", et::FLOAT, &["batch", "3"])],
        vec![value_info("y", et::FLOAT, &[4, 3])],
    );
    assert!(matches!(
        convert_bytes(&model(&g)),
        Err(OnnxError::Unsupported(_))
    ));
}

#[test]
fn reject_string_initializer() {
    // A STRING initializer materialized by an Add → Unsupported.
    let mut st = Vec::new();
    f_rep_i64(1, &[1], &mut st);
    f_i(2, et::STRING as i64, &mut st);
    f_len(6, b"hi", &mut st); // string_data
    f_str(8, "s", &mut st);
    let g = graph(
        "g",
        vec![node("Add", &["x", "s"], &["y"], &[], "a0")],
        vec![st],
        vec![value_info("x", et::FLOAT, &[1])],
        vec![value_info("y", et::FLOAT, &[1])],
    );
    assert!(matches!(
        convert_bytes(&model(&g)),
        Err(OnnxError::Unsupported(_))
    ));
}

#[test]
fn reject_non_static_output_dim() {
    let g = graph(
        "g",
        vec![node("Relu", &["x"], &["y"], &[], "")],
        vec![],
        vec![value_info("x", et::FLOAT, &[3])],
        vec![value_info_param("y", et::FLOAT, &["mystery"])],
    );
    assert!(matches!(
        convert_bytes(&model(&g)),
        Err(OnnxError::Unsupported(_))
    ));
}

#[test]
fn reject_dynamic_reshape() {
    // reshape spec as a *graph input* (not initializer) → Dynamic error.
    let g = graph(
        "g",
        vec![node("Reshape", &["x", "shape_in"], &["y"], &[], "r")],
        vec![],
        vec![
            value_info("x", et::FLOAT, &[2, 3]),
            value_info("shape_in", et::INT64, &[2]),
        ],
        vec![value_info("y", et::FLOAT, &[3, 2])],
    );
    assert!(matches!(
        convert_bytes(&model(&g)),
        Err(OnnxError::Dynamic(_))
    ));
}
