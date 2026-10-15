//! chain_mixed_probe — compile a conv1x1 chain with arbitrary per-task
//! shapes through ANEC for the mixed-shape corpus. `cs` is the channel
//! list: `c0,c1,c2` emits tasks c0->c1 then c1->c2.
//!
//!   chain_mixed_probe <root> <c0,c1,...,ck>

use mil_infer::Anec;
use std::fs;
use std::path::Path;

fn main() {
    let mut a = std::env::args().skip(1);
    let root = Path::new(&a.next().unwrap_or("/tmp/chainmix".into())).to_path_buf();
    let cs: Vec<u32> = a
        .next()
        .unwrap()
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let k = cs.len() - 1;
    let tag = cs.iter().map(u32::to_string).collect::<Vec<_>>().join("x");
    let dir = root.join(format!("m{tag}"));
    fs::create_dir_all(dir.join("out")).unwrap();
    fs::create_dir_all(dir.join("weights")).unwrap();
    // weight.bin: 64B global hdr, then per-task record (64B hdr + data)
    let mut b = vec![0x0c, 0, 0, 0, 0x02, 0, 0, 0];
    b.resize(64, 0);
    let mut rec_offs = Vec::with_capacity(k);
    for (i, w) in cs.windows(2).enumerate() {
        rec_offs.push(b.len());
        let (ci, co) = (w[0] as usize, w[1] as usize);
        let payload = ci * co * 2;
        let data_off = b.len() + 64;
        b.extend_from_slice(&[0xef, 0xbe, 0xad, 0xde, 0x01, 0, 0, 0]);
        b.extend_from_slice(&(payload as u64).to_le_bytes());
        b.extend_from_slice(&(data_off as u64).to_le_bytes());
        b.resize(data_off, 0);
        for j in 0..(ci * co) as u32 {
            b.extend_from_slice(&(((j + (i as u32 + 1) * 7919) % 65536) as u16).to_le_bytes());
        }
    }
    fs::write(dir.join("weights/weight.bin"), &b).unwrap();
    let mut mil = format!(
        "program(1.3)\n{{\n    func main<ios18>(tensor<fp16, [1, {}, 1, 1]> x) {{\n",
        cs[0]
    );
    for (i, w) in cs.windows(2).enumerate() {
        let (ci, co) = (w[0], w[1]);
        mil += &format!("            tensor<fp16, [{co}, {ci}, 1, 1]> w{i} = const()[val = tensor<fp16, [{co}, {ci}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({})))];\n", rec_offs[i]);
    }
    mil += "            tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];\n            string pt = const()[val = string(\"valid\")];\n            tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n            tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];\n            int32 gr = const()[val = int32(1)];\n";
    for (i, w) in cs.windows(2).enumerate() {
        let (inp, outp) = (
            if i == 0 { "x".into() } else { format!("y{i}") },
            format!("y{}", i + 1),
        );
        mil += &format!("            tensor<fp16, [1, {}, 1, 1]> {outp} = conv(x = {inp}, weight = w{i}, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gr)[name = string(\"{outp}\")];\n", w[1]);
    }
    mil += &format!("    }} -> (y{k});\n}}\n");
    fs::write(dir.join("net.mil"), &mil).unwrap();
    let anec = Anec::open().expect("ANEC");
    match anec.compile(&dir.join("net.mil"), &dir.join("out"), "h16") {
        Ok(v) => println!(
            "{dir:?}: status={} dram={:?} errs={:?}",
            v.status, v.dram_usage, v.errors
        ),
        Err(e) => println!("ERR {e}"),
    }
}
