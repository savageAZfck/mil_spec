//! mm_kern_check — is the non-square matmul __KERN_0 stream the
//! conv weight_layout_ns transform of the [N,K] weight?
//!   mm_kern_check <artifact> <K> <N>
fn main() {
    let mut a = std::env::args().skip(1);
    let d = std::fs::read(a.next().unwrap()).unwrap();
    let (k, n): (usize, usize) = (
        a.next().unwrap().parse().unwrap(),
        a.next().unwrap().parse().unwrap(),
    );
    // __kern_0 section fileoff: locate via the __KERN_0 segment (foff at +48)
    let mut off = 0x20usize;
    let mut kern_fo = 0xc000usize;
    let mut kern_sz = 0usize;
    let nc = u32::from_le_bytes(d[0x10..0x14].try_into().unwrap());
    for _ in 0..nc {
        let (c, s) = (
            u32::from_le_bytes(d[off..off + 4].try_into().unwrap()),
            u32::from_le_bytes(d[off + 4..off + 8].try_into().unwrap()) as usize,
        );
        if c == 0x19 && &d[off + 8..off + 15] == b"__KERN_" {
            let ns = u32::from_le_bytes(d[off + 64..off + 68].try_into().unwrap()) as usize;
            for j in 0..ns {
                let s0 = off + 72 + 80 * j;
                if &d[s0..s0 + 7] == b"__kern_" {
                    kern_sz = u32::from_le_bytes(d[s0 + 40..s0 + 44].try_into().unwrap()) as usize;
                    kern_fo = u32::from_le_bytes(d[s0 + 48..s0 + 52].try_into().unwrap()) as usize;
                }
            }
        }
        off += s;
    }
    let w: Vec<u16> = (0..n * k).map(|i| i as u16).collect();
    let (st, loffs, lsz) = mil_infer::hwx::weight_layout_ns(&w, k, n);
    let mut mine = Vec::new();
    for v in &st {
        mine.extend_from_slice(&v.to_le_bytes());
    }
    let got = &d[kern_fo..kern_fo + kern_sz];
    let m: Vec<usize> = (0..got.len().min(mine.len()))
        .filter(|&i| got[i] != mine[i])
        .collect();
    println!(
        "kern@foff {kern_fo:#x} sz {kern_sz:#x}; stream {} B; diffs {} first {:?}; lanes {:?}/{:?}",
        mine.len(),
        m.len(),
        m.first(),
        loffs,
        lsz
    );
}
