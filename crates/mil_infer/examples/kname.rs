fn main() {
    for off in [64u64, 128, 8320, 8384, 8256] {
        println!(
            "off={off} -> {}",
            mil_infer::hwx::conv_kernel_name_ns(64, 64, off)
        );
    }
}
