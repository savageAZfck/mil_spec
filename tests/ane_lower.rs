//! `ane_lower` — stripe-split + index-bake lowering tests.

const SRC: &str = r#"program(1.3)
{
    func main<ios19>(state<tensor<fp16, [4, 3, 64, 64]>> kv, tensor<int32, [1]> pos, tensor<fp16, [1, 8]> x) {
            tensor<fp16, [4, 3, 64, 64]> kv_0 = read_state(input = kv)[name = string("kv_0")];
            tensor<int32, [1]> b0 = const()[val = tensor<int32, [1]>([0])];
            tensor<int32, [1]> w2 = const()[val = tensor<int32, [1]>([2])];
            tensor<int32, [1]> s3 = const()[val = tensor<int32, [1]>([3])];
            tensor<int32, [1]> d64 = const()[val = tensor<int32, [1]>([64])];
            tensor<int32, [1]> five = const()[val = tensor<int32, [1]>([5])];
            tensor<int32, [1]> ax = const()[val = tensor<int32, [1]>([0])];
            tensor<fp16, [2, 3, 5, 64]> u0 = const()[val = tensor<fp16, [2, 3, 5, 64]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
            tensor<fp16, [2, 3, 5, 64]> u1 = const()[val = tensor<fp16, [2, 3, 5, 64]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(128)))];
            tensor<int32, [4]> l0_beg = concat(axis = ax, values = (b0, b0, pos, b0));
            tensor<int32, [1]> l0_p2 = add(x = pos, y = five);
            tensor<int32, [4]> l0_end = concat(axis = ax, values = (w2, s3, l0_p2, d64));
            tensor<int32, [4]> l1_beg = concat(axis = ax, values = (w2, b0, pos, b0));
            tensor<int32, [1]> l1_p2 = add(x = pos, y = five);
            tensor<int32, [1]> e4 = const()[val = tensor<int32, [1]>([4])];
            tensor<int32, [4]> l1_end = concat(axis = ax, values = (e4, s3, l1_p2, d64));
            tensor<fp16, [4, 3, 64, 64]> l0_su = slice_update(begin = l0_beg, end = l0_end, update = u0, x = kv_0);
            tensor<fp16, [4, 3, 64, 64]> l1_su = slice_update(begin = l1_beg, end = l1_end, update = u1, x = l0_su);
            tensor<int32, [4]> k_b = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];
            tensor<int32, [4]> k_e = const()[val = tensor<int32, [4]>([1, 3, 64, 64])];
            tensor<int32, [4]> v_b = const()[val = tensor<int32, [4]>([3, 0, 0, 0])];
            tensor<int32, [4]> v_e = const()[val = tensor<int32, [4]>([4, 3, 64, 64])];
            tensor<fp16, [1, 3, 64, 64]> kf = slice_by_index(begin = k_b, end = k_e, x = l0_su);
            tensor<fp16, [1, 3, 64, 64]> vf = slice_by_index(begin = v_b, end = v_e, x = l1_su);
            write_state(data = l1_su, input = kv)[name = string("kv_write_state")];
            tensor<fp16, [1, 8]> y = add(x = x, y = x);
        } -> (y);
}
"#;

#[test]
fn stripes_split_and_indices_bake() {
    let (out, rep) = mil_spec::ane_lower::lower_for_ane(SRC, 0).expect("lower");
    assert_eq!(rep.states_split, 1);
    assert_eq!(rep.updates_localized, 2);
    assert_eq!(rep.reads_localized, 2);
    assert_eq!(rep.writes_emitted, 2);
    assert!(rep.params_dropped >= 1, "pos should drop");
    // Two stripe states of width 2 each.
    assert!(out.contains("state<tensor<fp16, [2, 3, 64, 64]>> kv_0"));
    assert!(out.contains("state<tensor<fp16, [2, 3, 64, 64]>> kv_1"));
    // Each su has its own read → update → write triple.
    assert_eq!(out.matches("read_state(input = kv_0)").count(), 1);
    assert_eq!(out.matches("read_state(input = kv_1)").count(), 1);
    assert_eq!(
        out.matches("write_state(data = l0_su, input = kv_0)")
            .count(),
        1
    );
    assert_eq!(
        out.matches("write_state(data = l1_su, input = kv_1)")
            .count(),
        1
    );
    // Index math gone; begin/end baked to local consts.
    assert!(
        !out.contains(" l0_beg = concat"),
        "concat index math dropped"
    );
    assert!(!out.contains(" l0_p2 = add"), "pos add dropped");
    assert!(out.contains("> l0_su_lb = const()[val = tensor<int32, [4]>([0, 0, 0, 0])]"));
    assert!(out.contains("> l1_su_lb = const()[val = tensor<int32, [4]>([0, 0, 0, 0])]"));
    // v read rebased from global [3..4] to local [1..2] on stripe 1.
    assert!(out.contains("> vf_lb = const()[val = tensor<int32, [4]>([1, 0, 0, 0])]"));
    assert!(out.contains("> vf_le = const()[val = tensor<int32, [4]>([2, 3, 64, 64])]"));
    // su output type localized.
    assert!(out.contains("tensor<fp16, [2, 3, 64, 64]> l0_su = slice_update"));
    // ios18 forced, pos param gone, shared read/write gone.
    assert!(out.contains("func main<ios18>"));
    assert!(!out.contains("tensor<int32, [1]> pos,"));
    assert!(!out.contains("read_state(input = kv)"));
    assert!(!out.contains("write_state(data = l1_su, input = kv)"));
}

#[test]
fn non_tiling_state_is_an_error() {
    // Update regions [0,2) + [1,4) overlap — not a legal stripe split.
    let bad = SRC
        .replace("[3, 0, 0, 0]", "[1, 0, 0, 0]")
        .replace("(w2, b0, pos, b0)", "(b0, b0, pos, b0)");
    assert!(mil_spec::ane_lower::lower_for_ane(&bad, 0).is_err());
}
