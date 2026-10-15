use std::fs;

fn main() {
    // conv1x1 MIL with spatial W: x [1,64,1,8] -> z [1,64,1,8], identity weights
    let d = "/tmp/convroute";
    fs::create_dir_all(format!("{d}/weights")).unwrap();
    let mut blob: Vec<u8> = vec![0; 0x40];
    let mut data = Vec::new();
    data.extend_from_slice(&(0xdeadbeefu32).to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    let wsz = 64 * 64 * 2u64;
    data.extend_from_slice(&wsz.to_le_bytes());
    data.extend_from_slice(&0x80u64.to_le_bytes());
    data.resize(0x40, 0);
    for o in 0..64 {
        for i in 0..64 {
            let v = if i == o { 1.0f32 } else { 0.0 };
            data.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
    }
    blob.extend_from_slice(&data);
    fs::write(format!("{d}/weights/weight.bin"), &blob).unwrap();
    let mil = r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 64, 1, 8]> x) {
            tensor<fp16, [64, 64, 1, 1]> w = const()[val = tensor<fp16, [64, 64, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
            tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];
            tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];
            tensor<fp16, [1, 64, 1, 8]> z = conv(x = x, weight = w, strides = st, pad = pd)[name = string("z")];
    } -> (z);
}"#;
    fs::write(format!("{d}/model.mil"), mil).unwrap();
    match mil_infer::hwx::emit_from_mil(&format!("{d}/model.mil"), &format!("{d}/out.macho")) {
        Ok(img) => {
            fs::write(format!("{d}/out.macho"), &img).unwrap();
            println!("emitted {} bytes", img.len())
        }
        Err(e) => println!("ERR: {}", e),
    }
}
