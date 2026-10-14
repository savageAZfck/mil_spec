//! `hwx` — emit the ANE Mach-O (HWX) container directly, replacing
//! `ANECCompile` for supported graph shapes.
//!
//! The ANE accepts a custom object format: a `0xBEEFFACE`-magicked
//! Mach-O header followed by 15 load commands — `__PAGEZERO`, three
//! `__FVMLIB` segments, `__TEXT` (instruction stream + const pool),
//! three `0x40` notes, type-`0x4` task-descriptor commands for the
//! program and each I/O tensor, a type-`0x8` signature block, an
//! `rt.version` note, and a standard `LC_SYMTAB` whose string table
//! carries the tensor layout descriptors.
//!
//! The skeleton below is a captured reference compile of a two-input
//! elementwise graph; every byte has been field-mapped by diffing
//! artifacts across shapes (N = 4..=256) and opcodes (add/mul/sub).
//! The emitter patches the shape/opcode/name fields and regenerates
//! the variable-length tail (signature, symtab, strtab).

/// Canonical skeleton: `z = add(x, y)` over `fp16 [1,64,1,1]` for h16.
const SKELETON: &[u8] = include_bytes!("../assets/ew2_template.hwx");
/// Unary activation skeleton: `z = sigmoid(x)` over `fp16 [1,64,1,1]`.
/// Layout differs from elementwise: `__TEXT@0x4000`, a `__KERN_0`
/// LUT segment at `0xc000`, and one `__text` relocation.
const SKELETON_ULUT: &[u8] = include_bytes!("../assets/ulut_template.hwx");

/// 1x1-conv skeletons: `z = conv1x1(x, w)` over `fp16 [1,C,W,1]` with
/// a `[C,C,1,1]` constant weight, h16. All share the signature block
/// at `0x2688` (capacity `0x890` bytes of text) and the transformed
/// weight stream at `__KERN_0` file offset `0xc000`.
const SKELETON_CONV32: &[u8] = include_bytes!("../assets/conv32_skeleton.hwx");
const SKELETON_CONV64S1: &[u8] = include_bytes!("../assets/conv64s1_skeleton.hwx");
const SKELETON_CONV64S5: &[u8] = include_bytes!("../assets/conv64s5_skeleton.hwx");
const SKELETON_CONV64S8: &[u8] = include_bytes!("../assets/conv64s8p_skeleton.hwx");
const SKELETON_CONV128: &[u8] = include_bytes!("../assets/conv128_skeleton.hwx");

/// Square-matmul skeletons: `z = matmul(x, w, transpose_y)` over fp16
/// `[1,N,1,N]` with a `[1,N,1,N]` constant weight, h16. Unlike conv,
/// weights are stored verbatim inside `__TEXT`'s const pool — no
/// `__KERN_0` segment and no `__text` relocations.
/// `conv64s1` + `sigmoid` chain reference (ones weights): `__KERN_0`
/// carries sixteen `0x280`-byte bands of `[0x80 LUT][0x200 weights]`.
const SKELETON_CONV64_SIG: &[u8] = include_bytes!("../assets/conv64_sig_skeleton.hwx");
const SKELETON_MM32: &[u8] = include_bytes!("../assets/mm32_skeleton.hwx");
const SKELETON_MM64: &[u8] = include_bytes!("../assets/mm64_skeleton.hwx");
const SKELETON_MM128: &[u8] = include_bytes!("../assets/mm128_skeleton.hwx");
const SKELETON_MM256: &[u8] = include_bytes!("../assets/mm256_skeleton.hwx");

/// Elementwise ops sharing the two-input instruction template.
/// Value is the u32 opcode word at `__text+0xcc`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EwOp {
    /// `add` — elementwise sum.
    Add,
    /// `mul` — elementwise product.
    Mul,
    /// `sub` — elementwise difference.
    Sub,
    /// `minimum` — elementwise min.
    Min,
    /// `maximum` — elementwise max.
    Max,
}

impl EwOp {
    fn opcode(self) -> u32 {
        match self {
            EwOp::Add => 0x0008_0000,
            EwOp::Mul => 0x0008_0004,
            EwOp::Sub => 0x000c_0000,
            EwOp::Min => 0x0008_000c,
            EwOp::Max => 0x0008_0008,
        }
    }
}

/// u32 LE fields holding the channel count `N` outright.
const FLD_N: &[usize] = &[0x0c44, 0x198c, 0x26d4];
/// u32 LE fields holding `N/4` (byte-count/8 style descriptor fields).
const FLD_N_DIV4: &[usize] = &[
    0x00d9, 0x0171, 0x0209, 0x0c69, 0x0c89, 0x0c99, 0x19b1, 0x19d1, 0x19e1, 0x26f9, 0x2719, 0x2729,
];
/// u32 LE fields holding `N/128` (tile-group loop count).
const FLD_N_DIV128: &[usize] = &[0x0c08];
/// Offset of the u32 opcode word inside the instruction stream.
const OP_FIELD: usize = 0x80cc;

/// Two-op elementwise chain (`t = x ⊕ y ; z = t ⊗ x`) — same envelope
/// as the single-ew family, with a 0x1c0-byte fused instruction
/// record at 0x8000 in place of the 0x130-byte single record.
const SKELETON_EWCHAIN: &[u8] = include_bytes!("../assets/ew_chain_skeleton.hwx");

/// Tail-byte table for the `(tail_bytes, vec_count)` instruction
/// words. Encoded from the reference corpus across `N mod 64`.
fn tail_a0(n: u32) -> u8 {
    let r16 = n % 16;
    if r16 != 0 {
        return (r16 * 16) as u8;
    }
    match n % 64 {
        0 if n >= 64 => 0x20,
        32 => 0x10,
        _ => 0,
    }
}

fn tail_ac(n: u32) -> u8 {
    let r16 = n % 16;
    if r16 != 0 {
        return (r16 * 16) as u8;
    }
    match n % 64 {
        0 if n >= 64 => 0x40,
        32 => 0x10,
        _ => 0,
    }
}

fn tail_c4(n: u32) -> u8 {
    if n % 8 == 4 {
        return 0x80;
    }
    match n % 64 {
        0 if n >= 64 => 0x60,
        32 => 0x20,
        _ => 0,
    }
}

fn put_u32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

/// Emit a complete HWX image for `z = x ⊕ y` over fp16 `[1, N, 1, 1]`.
///
/// `mil_path` / `out_path` are recorded in the signature block's
/// compilation record, matching the reference format.
pub fn emit_elementwise2(op: EwOp, n: u32, mil_path: &str, out_path: &str) -> Vec<u8> {
    let mut img = SKELETON.to_vec();

    // ---- scalar shape fields -----------------------------------------
    for &f in FLD_N {
        put_u32(&mut img, f, n);
    }
    for &f in FLD_N_DIV4 {
        put_u32(&mut img, f, n / 4);
    }
    for &f in FLD_N_DIV128 {
        put_u32(&mut img, f, n / 128);
    }

    // ---- variable-length tail ----------------------------------------
    // Rebuild cmd[12] (signature), rt.version, symtab+strtab with our
    // own paths; then fix every offset that moves. This re-splices the
    // instruction region, so stream patches come after.
    rebuild_tail(&mut img, n, mil_path, out_path, &SKELETON[0x8000..]);

    // ---- instruction stream ------------------------------------------
    let v16 = n / 16;
    let v8 = n / 8;
    put_u32(&mut img, 0x8014, n / 128);
    put_u32(&mut img, 0x8048, n);
    put_u32(&mut img, 0x8058, n);
    let tail = if n % 16 != 0 { (n % 16) * 16 } else { 0 };
    for &f in &[0x80a4, 0x80a8, 0x80b8, 0x80bc] {
        put_u32(&mut img, f, tail | (v16 << 8));
    }
    for &f in &[0x80a0, 0x80b4] {
        put_u32(&mut img, f, tail_a0(n) as u32 | (v16 << 8));
    }
    put_u32(&mut img, 0x80ac, tail_ac(n) as u32 | (v16 << 8));
    put_u32(&mut img, 0x80c4, tail_c4(n) as u32 | (v8 << 8));
    put_u32(&mut img, OP_FIELD, op.opcode());

    img
}

/// Emit `t = x ⊕ y ; z = t ⊗ w` (w ∈ {x, y}) — a fused
/// two-elementwise-op instruction record (0x1c0 bytes at 0x8000) on
/// the single-ew envelope. `op2_t_first` selects the operand order
/// of the second op: `z = op2(t, w)` vs `z = op2(w, t)`;
/// `op2_free_y` selects which program input is the free operand.
///
/// Decoded field map (record-relative, all u32 LE):
///   +0x14            n/128            +0x40/+0x50/+0x138/+0x148  n
///   +0x168           n*2              +0xc8/+0x180               op opcodes
///   +0x8c..+0xb4     lane descriptors — two swapped arrangements:
///                    free=x → `71,72 / n*2 / v16 / ac / b4 / v16`,
///                    free=y → `72,71 / ac / b4 / n*2 / v16`
///                    where `ac = n*2&0xff (+0x40 if 64|n) | hi<<8`,
///                    `hi = v16 + n/128`, `b4 = f(n%64) | v16<<8`
///   +0x158..+0x178   op2 operand-order descriptor block
///   +0x178           `n*2 & 0xff | hi << 8`
pub fn emit_elementwise_chain(
    op1: EwOp,
    op2: EwOp,
    op2_t_first: bool,
    op2_free_y: bool,
    n: u32,
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    if n < 32 {
        return Err(format!(
            "elementwise chain n={n} is below the covered template (n>=32)"
        ));
    }
    let mut img = SKELETON_EWCHAIN.to_vec();

    // ---- scalar shape fields (same plane as the single-ew family) --
    for &f in FLD_N {
        put_u32(&mut img, f, n);
    }
    for &f in FLD_N_DIV4 {
        put_u32(&mut img, f, n / 4);
    }
    for &f in FLD_N_DIV128 {
        put_u32(&mut img, f, n / 128);
    }

    rebuild_tail(&mut img, n, mil_path, out_path, &SKELETON_EWCHAIN[0x8000..]);

    // ---- fused instruction record -----------------------------------
    let r = 0x8000usize;
    let v16 = n / 16;
    put_u32(&mut img, r + 0x14, n / 128);
    for &f in &[0x40usize, 0x50, 0x138, 0x148] {
        put_u32(&mut img, r + f, n);
    }
    put_u32(&mut img, r + 0x168, n * 2);
    put_u32(&mut img, r + 0x178, (n * 2 & 0xff) | ((v16 + n / 128) << 8));

    // lane descriptors: which program input is op2's free operand
    // swaps the two descriptor halves.
    let ac = (n * 2 & 0xff) + if n % 64 == 0 { 0x40 } else { 0 };
    let hi = v16 + n / 128;
    let b4 = if n >= 64 && n % 64 == 0 {
        0x20
    } else if n % 64 == 32 {
        0x10
    } else {
        0
    };
    for &f in &[0xa4usize, 0xa8, 0xb8, 0xbc] {
        put_u32(&mut img, r + f, v16 << 8);
    }
    if op2_free_y {
        img[r + 0x8c] = 0x72;
        img[r + 0x90] = 0x71;
        put_u32(&mut img, r + 0x98, ac | (hi << 8));
        put_u32(&mut img, r + 0xa0, b4 | (v16 << 8));
        put_u32(&mut img, r + 0xac, n * 2);
        put_u32(&mut img, r + 0xb4, v16 << 8);
    } else {
        put_u32(&mut img, r + 0x98, n * 2);
        put_u32(&mut img, r + 0xa0, v16 << 8);
        put_u32(&mut img, r + 0xac, ac | (hi << 8));
        put_u32(&mut img, r + 0xb4, b4 | (v16 << 8));
    }

    put_u32(&mut img, r + 0xc8, op1.opcode());
    put_u32(&mut img, r + 0x180, op2.opcode());

    // op2 operand-order descriptor block.
    if op2_t_first {
        img[r + 0x158..r + 0x15c].copy_from_slice(&[0x00, 0x01, 0x00, 0x00]);
        img[r + 0x160..r + 0x164].copy_from_slice(&[0x41, 0x10, 0xc0, 0x80]);
        img[r + 0x170..r + 0x178]
            .copy_from_slice(&[0x52, 0x90, 0x00, 0x00, 0x4a, 0x01, 0x00, 0x00]);
    } else {
        img[r + 0x158..r + 0x15c].copy_from_slice(&[0x00, 0x02, 0x00, 0x00]);
        img[r + 0x160..r + 0x164].copy_from_slice(&[0x42, 0x10, 0x03, 0xc0]);
        img[r + 0x170..r + 0x178]
            .copy_from_slice(&[0x4a, 0x01, 0x00, 0x00, 0x53, 0x10, 0x00, 0x00]);
    }

    Ok(img)
}

const SIG_FLAGS: &str = "-t h16\n\t\
--fno-fold-scale=true\n\t\
--fdram-allocator=ffreuse\n\t\
--fdram-tensor-priority=sizebyliverange\n\t\
--fl2-allocator=ffreuse\n\t\
--fl3-allocator=ffreuse\n\t\
--fl2-cache-mode=resident\n\t\
--fsignature=ident\n\t\
--fdisable-bonded-networks=true\n\t\
--enable-param-and-map-rtgraph-refactor=false\n\t\
--enable-work-stealing-for-bonded-networks=false\n\t\
--memcache-size=4194304\n\t\
--fspatial-split=disabled\n\t\
--fenable-circular-buffer-in-spatial-split=-1\n\t\
--fkernel-rewind=enabled\n\t\
--fuse-runtime-matmul-scaling=false\n\t\
--max-td-latency=10000.000000\n\t\
--ne-frequency=-1.000000\n\t\
--pstate-dcs=-1\n\t\
--pstate-soc=-1\n\t\
--bss-limit=-1\n\t\
--foptimize-ne-utilization=true\n\t\
--disable-cache-prefetch-mask=1\n\t\
--optimize-mutable-kernel-section=true\n\t\
--enable-summary-performance-stats=false\n\t\
--split-kernel-section=true\n\t\
--max-kernel-section-size=134217728\n\t\
--enable-context-switch-events=false\n\t\
--fdisable-ne-width-slicing=false\n\t\
--disable-hard-swish-opt=false\n\t\
--preserve-texture-fraction=false\n\t\
--max-td-count=-1\n\t\
--max-segment-size=-1\n\t\
--e4m3-overflow-setting=Saturate\n\t\
--enable-dram-inplace-allocation=false\n\t\
--enable-l2-batch-splitting=true\n\t\
--enable-global-cw-optimization=true\n\t\
--suppress-math-exception=false\n\t\
--enable-l2-cached-buffer=true\n\t\
--enable-low-effort-cp-allocation=false\n\t\
--enable-noise-reduction-tiling=false\n\t\
--dump-parallel-score=false\n\t\
--enable-function-inlining=true\n\t\
--fdisable-weight-file-size-check=false\n\t\
--compile_time_mutable=false\n\t\
--enable-afm-mlir-features=true\n\t\
--enable-stable-precision=false\n\t\
--enable-kernel-split-for-multi-palette-lut=false\n\t\
--enable-non-self-replaceable-persistence=false\n\t\
--enable-memcache-kernel-allocation=false\n\t\
--cost-model-cluster-threshold=85\n\t\
--enforce-user-slot-table-alloc-for-standard-cc\n\t\
--enable-matmul-kernel-pad-insertion=true\n\t\
--use-dp-allocator=false\n\t\
--enable-nonbonded-networks=false\n\t\
--generate-model-debug-data=true\n\t\
--enable-ne-input-transpose-fusion=true\n\t\
--llir-emit-bytecode=true\n\t\
--tmem-size=-1\n\t\
--Wl-undefined=fvmlib";

fn signature(mil_path: &str, out_path: &str) -> String {
    if std::env::var_os("HWX_SIG_ANEC").is_some() {
        // Reference-format signature — used only by the byte-equality
        // verifier, which compares against ANECCompile artifacts.
        return format!(
            "ANEC v1\nzin_ane_compiler v10.26.6\n\n Module ANEC: \n\t \
ModuleBundleName: com.apple.ANECompilerFramework\n\t \
ModuleVersion: 10.26.6\n\t \
ModuleCompilationFlags: \n\t{SIG_FLAGS}\n\t-i {mil_path}\n\t-o {out_path}\n"
        );
    }
    format!(
        "ANEC v1\nmil_infer hwx-emit v0.1\n\n Module MILC: \n\t \
ModuleBundleName: dev.badapple.mil\n\t \
ModuleVersion: 0.1\n\t \
ModuleCompilationFlags: \n\t{SIG_FLAGS}\n\t-i {mil_path}\n\t-o {out_path}\n"
    )
}

/// Scalar type-descriptor strings — identical across all programs.
const TYPE_SYMS: &[&str] = &[
    "void:t1=1",
    "int8:t2=r2;0;127",
    "uint8:t3=r1;0;255",
    "int16:t4=r1;-32768;32767",
    "float16:t5=r1;2;0",
    "float:t6=r1;4;0",
    "raw10:t7=r1;-512;511",
    "lut:t8=",
    "uint4:t9=r1;0;15",
    "uint6:t10=r1;0;63",
    "uint1:t11=r1;0,1",
    "uint2:t12=r1;0;3",
    "packed10:t13=r1;-512;511",
    "y10:t14=r1;-512;511",
    "uint16:t15=r1;0;65535",
    "e4m3:t16=r1;1;0",
    "int32:t17=r1;-2147483648;2147483647",
    "uint32:t18=r1;0;4294967295",
    "int64:t19=r1;-9223372036854775808;9223372036854775807",
    "uint64:t20=r1;0;18446744073709551615",
    "raw12:t21=r1;-512;511",
    "y12:t22=r1;-512;511",
    "int4:t23=r1;-8;7",
];

fn layout_desc(name: &str, base: u32, n: u32) -> String {
    format!(
        "{name}:t{base}=ar1;0;1;{b1}=s{ns}n:ar1;0;{n};{b2}=s64c:ar1;0;1;{b3}=s64h:ar1;0;1;{b4}=s2w:5",
        base = base,
        b1 = base + 1,
        ns = 64 * n,
        n = n,
        b2 = base + 2,
        b3 = base + 3,
        b4 = base + 4,
    )
}

/// Rebuild everything after the z-task-descriptor (file offset
/// `0x33f0`): signature command, `rt.version`, `LC_SYMTAB`, symbol
/// table, string table — then fix `sizeofcmds` in the header.
fn rebuild_tail(img: &mut Vec<u8>, n: u32, mil_path: &str, out_path: &str, post: &[u8]) {
    let sig = signature(mil_path, out_path);
    let sig_pad = (8 - (sig.len() + 1) % 8) % 8 + 1; // NUL + 8-align
    let csz = 8 + sig.len() + sig_pad;

    let mut tail = Vec::new();
    // cmd[12]: type 8, csz, signature text
    tail.extend_from_slice(&8u32.to_le_bytes());
    tail.extend_from_slice(&(csz as u32).to_le_bytes());
    tail.extend_from_slice(sig.as_bytes());
    tail.extend(std::iter::repeat(0u8).take(sig_pad));

    // cmd[13]: rt.version (0x31), fixed 0x30 — name[16], 0x28, 4, 0x10001
    tail.extend_from_slice(&[
        0x31, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, b'r', b't', b'.', b'v', b'e', b'r', b's',
        b'i', b'o', b'n', 0, 0, 0, 0, 0, 0, 0x28, 0x00, 0x00, 0x00, 0, 0, 0, 0, 0x04, 0, 0, 0, 0,
        0, 0, 0, 0x01, 0, 0x01, 0, 0, 0, 0, 0,
    ]);

    // ---- symtab + strtab -------------------------------------------------
    let symoff = 0x33f0 + tail.len() + 0x18;
    let io_syms: &[(&str, u8, u64)] = &[
        ("x", 1, 0x3000_0000),
        ("y", 2, 0x3000_4000),
        ("z", 3, 0x3000_8000),
    ];
    let layout_strs = [
        layout_desc("x", 24, n),
        layout_desc("y", 29, n),
        layout_desc("z", 34, n),
    ];

    // string table
    let mut strtab = vec![0u8];
    let mut strx = |s: &str| -> u32 {
        let off = strtab.len() as u32;
        strtab.extend_from_slice(s.as_bytes());
        strtab.push(0);
        off
    };
    let mut entries: Vec<(u32, u8, u8, u16, u64)> = Vec::new();
    for &(name, sect, val) in io_syms {
        entries.push((strx(name), 0x0f, sect, 0, val));
    }
    for ty in TYPE_SYMS {
        entries.push((strx(ty), 0x80, 0, 0, 0));
    }
    const LAYOUT_DESC: [u16; 3] = [0x18, 0x1d, 0x02];
    for (ls, &desc) in layout_strs.iter().zip(&LAYOUT_DESC) {
        entries.push((strx(ls), 0x20, 0, desc, 0));
    }

    let stroff = symoff + entries.len() * 16;
    // cmd[14]: LC_SYMTAB
    tail.extend_from_slice(&0x2u32.to_le_bytes());
    tail.extend_from_slice(&0x18u32.to_le_bytes());
    tail.extend_from_slice(&(symoff as u32).to_le_bytes());
    tail.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    tail.extend_from_slice(&(stroff as u32).to_le_bytes());
    tail.extend_from_slice(&(strtab.len() as u32).to_le_bytes());

    for (sx, ty, sec, desc, val) in entries {
        tail.extend_from_slice(&sx.to_le_bytes());
        tail.push(ty);
        tail.push(sec);
        tail.extend_from_slice(&desc.to_le_bytes());
        tail.extend_from_slice(&val.to_le_bytes());
    }
    tail.extend_from_slice(&strtab);

    // splice: replace [0x33f0 .. symtab_end] in the skeleton, keep the
    // instruction region at 0x8000+.
    img.truncate(0x33f0);
    img.extend_from_slice(&tail);
    img.resize(0x8000, 0);
    img.extend_from_slice(post);
    // header sizeofcmds = end-of-load-commands - 0x20
    let sizeofcmds = (0x33f0 + csz + 0x30 + 0x18 - 0x20) as u32;
    put_u32(img, 0x14, sizeofcmds);
}

/// Unary activation ops. The instruction stream is a generic
/// apply-LUT kernel — the operation is entirely determined by the
/// `0x80`-byte fp16 lookup table in `__KERN_0`.
///
/// Verified byte-identical to `ANECCompile` output at `N % 16 == 0`;
/// tail encodings for ragged shapes are unprobed in this template.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LutOp {
    /// `sigmoid` — 33-point grid over `[-8, 8]`.
    Sigmoid,
    /// `tanh` — same grid, saturating to `±1`.
    Tanh,
}

impl LutOp {
    fn lut(self) -> [u8; 0x80] {
        match self {
            LutOp::Sigmoid => [
                0xf8, 0xc8, 0x29, 0x48, 0x00, 0x00, 0x00, 0x3c, 0x7f, 0x0d, 0x87, 0x10, 0x77, 0x13,
                0x26, 0x16, 0x10, 0x19, 0x2b, 0x1c, 0xda, 0x1e, 0xa0, 0x21, 0x9b, 0x24, 0x81, 0x27,
                0x12, 0x2a, 0xdb, 0x2c, 0xa1, 0x2f, 0xd6, 0x31, 0x4e, 0x34, 0x0a, 0x36, 0x00, 0x38,
                0xfb, 0x38, 0xd9, 0x39, 0x8a, 0x3a, 0x0c, 0x3b, 0x65, 0x3b, 0x9f, 0x3b, 0xc4, 0x3b,
                0xdb, 0x3b, 0xe9, 0x3b, 0xf2, 0x3b, 0xf8, 0x3b, 0xfb, 0x3b, 0xfd, 0x3b, 0xfe, 0x3b,
                0xff, 0x3b, 0xff, 0x3b, 0xac, 0x09, 0x0c, 0x17, 0x3e, 0x16, 0xe6, 0x3b, 0x01, 0x00,
                0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00,
            ],
            LutOp::Tanh => [
                0x00, 0x00, 0x00, 0x44, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0xf5, 0x2f, 0xd6, 0x33,
                0xbc, 0x35, 0x65, 0x37, 0x70, 0x38, 0x15, 0x39, 0xa2, 0x39, 0x18, 0x3a, 0x79, 0x3a,
                0xc9, 0x3a, 0x0a, 0x3b, 0x3e, 0x3b, 0x67, 0x3b, 0x88, 0x3b, 0xa2, 0x3b, 0xb6, 0x3b,
                0xc6, 0x3b, 0xd3, 0x3b, 0xdd, 0x3b, 0xe5, 0x3b, 0xeb, 0x3b, 0xef, 0x3b, 0xf3, 0x3b,
                0xf6, 0x3b, 0xf8, 0x3b, 0xfa, 0x3b, 0xfb, 0x3b, 0xfc, 0x3b, 0xfd, 0x3b, 0xfe, 0x3b,
                0xfe, 0x3b, 0xff, 0x3b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3c, 0x03, 0x00,
                0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00,
            ],
        }
    }
}

/// Emit a complete HWX image for `z = act(x)` over fp16 `[1, N, 1, 1]`,
/// using the apply-LUT kernel template (`__KERN_0` carries the table).
pub fn emit_unary_lut(op: LutOp, n: u32, mil_path: &str, out_path: &str) -> Vec<u8> {
    let mut img = SKELETON_ULUT.to_vec();

    // ---- scalar shape fields -----------------------------------------
    for &f in &[0x0c24usize, 0x196c] {
        put_u32(&mut img, f, n);
    }
    for &f in &[
        0x00d9usize,
        0x0171,
        0x0c49,
        0x0c69,
        0x0c79,
        0x1991,
        0x19b1,
        0x19c1,
    ] {
        put_u32(&mut img, f, n / 4);
    }

    rebuild_tail_ulut(&mut img, op, n, mil_path, out_path);

    // ---- instruction stream (generic apply-LUT kernel) ----------------
    put_u32(&mut img, 0x4054, n);
    put_u32(&mut img, 0x4064, n);
    for &f in &[0x40a4usize, 0x40a8, 0x40ac, 0x40b4] {
        put_u32(&mut img, f, (n / 16) << 8);
    }

    // ---- LUT ----------------------------------------------------------
    img[0xc000..0xc080].copy_from_slice(&op.lut());

    img
}

/// Rebuild the unary template tail after the z task-descriptor
/// (`0x2688`): signature, `rt.version`, `LC_SYMTAB` (I/O + kernel-LUT
/// symbols + type table + layouts), string table, and the single
/// `__text` relocation entry at `align8(end-of-strtab)`.
fn rebuild_tail_ulut(img: &mut Vec<u8>, op: LutOp, n: u32, mil_path: &str, out_path: &str) {
    let sig = signature(mil_path, out_path);
    let sig_pad = (8 - (sig.len() + 1) % 8) % 8 + 1;
    let csz = 8 + sig.len() + sig_pad;

    let mut tail = Vec::new();
    tail.extend_from_slice(&8u32.to_le_bytes());
    tail.extend_from_slice(&(csz as u32).to_le_bytes());
    tail.extend_from_slice(sig.as_bytes());
    tail.extend(std::iter::repeat(0u8).take(sig_pad));

    tail.extend_from_slice(&[
        0x31, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, b'r', b't', b'.', b'v', b'e', b'r', b's',
        b'i', b'o', b'n', 0, 0, 0, 0, 0, 0, 0x28, 0x00, 0x00, 0x00, 0, 0, 0, 0, 0x04, 0, 0, 0, 0,
        0, 0, 0, 0x01, 0, 0x01, 0, 0, 0, 0, 0,
    ]);

    let symoff = 0x2688 + tail.len() + 0x18;
    // ANEC names the standalone apply-LUT kernel per op — a mined
    // constant, not a content hash (the same name appears at every N
    // for a given op).
    let kb = match op {
        LutOp::Sigmoid => "KB3B269BED6CC462DFC66AA7343285C01AFD65136F6C478452FF3605131C3FEBE",
        LutOp::Tanh => "KEEC44EE7EC029FB6826DF7CE273672621421E515063188B15C0A849E2323006F",
    };
    let kb_actlut = format!("{kb}_actlut");

    let mut strtab = vec![0u8];
    let mut strx = |s: &str| -> u32 {
        let off = strtab.len() as u32;
        strtab.extend_from_slice(s.as_bytes());
        strtab.push(0);
        off
    };
    let mut entries: Vec<(u32, u8, u8, u16, u64)> = Vec::new();
    entries.push((strx("x"), 0x0f, 1, 0, 0x3000_0000));
    entries.push((strx("z"), 0x0f, 2, 0, 0x3000_4000));
    entries.push((strx(&kb_actlut), 0x0f, 5, 2, 0x3001_0000));
    entries.push((strx(&kb), 0x0f, 5, 0, 0x3001_0080));
    for ty in TYPE_SYMS {
        entries.push((strx(ty), 0x80, 0, 0, 0));
    }
    entries.push((strx(&layout_desc("x", 24, n)), 0x20, 0, 0x18, 0));
    entries.push((strx(&layout_desc("z", 29, n)), 0x20, 0, 0x1d, 0));

    let stroff = symoff + entries.len() * 16;
    tail.extend_from_slice(&0x2u32.to_le_bytes());
    tail.extend_from_slice(&0x18u32.to_le_bytes());
    tail.extend_from_slice(&(symoff as u32).to_le_bytes());
    tail.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    tail.extend_from_slice(&(stroff as u32).to_le_bytes());
    tail.extend_from_slice(&(strtab.len() as u32).to_le_bytes());

    for (sx, ty, sec, desc, val) in entries {
        tail.extend_from_slice(&sx.to_le_bytes());
        tail.push(ty);
        tail.push(sec);
        tail.extend_from_slice(&desc.to_le_bytes());
        tail.extend_from_slice(&val.to_le_bytes());
    }
    tail.extend_from_slice(&strtab);

    // __text relocation: one entry at align8(strtab end), fixes the
    // instruction word at __text+0xf0 to reference __KERN_0 (sect 5).
    let reloff = 0x2688 + (tail.len() + 7) / 8 * 8;
    while 0x2688 + tail.len() < reloff {
        tail.push(0);
    }
    tail.extend_from_slice(&[0xf0, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x07]);
    put_u32(img, 0x218, reloff as u32); // __text reloff
                                        // nreloc at 0x21c is already 1 in the skeleton.

    img.truncate(0x2688);
    img.extend_from_slice(&tail);
    img.resize(0x4000, 0);
    img.extend_from_slice(&SKELETON_ULUT[0x4000..]);
    let sizeofcmds = (0x2688 + csz + 0x30 + 0x18 - 0x20) as u32;
    put_u32(img, 0x14, sizeofcmds);
}

/// ANE weight-layout interleave applied to `BLOBFILE` constants for
/// conv/matmul: the compiler embeds a banded transpose of the raw
/// `[C,C]` fp16 matrix in `__KERN_0`, not the row-major blob.
///
/// `w[oc][ic]` (index `oc*C + ic`) maps to
/// `dst[g*C*B + ic*B + m] = w[(g*B + m)*C + ic]` with `B = C/16` —
/// sixteen output-channel bands transposed independently. Verified
/// byte-exact against perm-weight corpus artifacts for `C = 32, 64,
/// 128` at spatial widths 1, 5, and 8.
///
/// (An earlier decode of this transform reported a leading pad plus a
/// trailing "wing"; that was an artifact of a malformed perm blob with
/// four leading pad elements, not a property of the transform.)
pub fn weight_interleave(w: &[u16], c: usize) -> Vec<u16> {
    debug_assert_eq!(w.len(), c * c);
    debug_assert_eq!(c % 16, 0);
    let b = c / 16;
    let mut out = Vec::with_capacity(c * c);
    for g in 0..16usize {
        for ic in 0..c {
            for m in 0..b {
                out.push(w[(g * b + m) * c + ic]);
            }
        }
    }
    out
}

/// Conv shape families with a captured skeleton. `(channels, spatial)`
/// selects the reference artifact whose instruction stream and task
/// descriptors are reused verbatim — the ANE conv kernel is generated
/// code whose length varies with shape, so coverage is per-skeleton
/// until the instruction encoding is fully decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvShape {
    /// `[1,32,1,1]` in, `[32,32,1,1]` weight.
    C32W1,
    /// `[1,64,1,1]` in, `[64,64,1,1]` weight.
    C64W1,
    /// `[1,64,5,1]` in, `[64,64,1,1]` weight.
    C64W5,
    /// `[1,64,8,1]` in, `[64,64,1,1]` weight.
    C64W8,
    /// `[1,128,1,1]` in, `[128,128,1,1]` weight.
    C128W1,
}

impl ConvShape {
    /// Channel count `C` of the `[C,C,1,1]` weight.
    pub fn channels(self) -> usize {
        match self {
            ConvShape::C32W1 => 32,
            ConvShape::C64W1 | ConvShape::C64W5 | ConvShape::C64W8 => 64,
            ConvShape::C128W1 => 128,
        }
    }

    fn skeleton(self) -> &'static [u8] {
        match self {
            ConvShape::C32W1 => SKELETON_CONV32,
            ConvShape::C64W1 => SKELETON_CONV64S1,
            ConvShape::C64W5 => SKELETON_CONV64S5,
            ConvShape::C64W8 => SKELETON_CONV64S8,
            ConvShape::C128W1 => SKELETON_CONV128,
        }
    }
}

/// End of the z task-descriptor: everything past `0x2688` (signature,
/// `rt.version`, `LC_SYMTAB`, strtab, the single `__text` relocation)
/// is regenerated so signature length changes re-offset correctly.
const CONV_TAIL_OFF: usize = 0x2688;
/// Transformed weight stream offset inside `__KERN_0`.
const CONV_WEIGHTS_OFF: usize = 0xc000;
/// End of the z task-descriptor for matmul skeletons.
const MM_TAIL_OFF: usize = 0x25f0;

/// Symbol-table entries + optional `__text` relocation payload,
/// extracted from a skeleton (kernel names, type table, layout strings
/// and the reloc target are all shape-family state we carry verbatim).
fn tail_meta(skel: &[u8]) -> (Vec<(String, u8, u8, u16, u64)>, Vec<[u8; 8]>) {
    // walk load commands for the type-2 LC_SYMTAB
    let ncmds = u32::from_le_bytes(skel[0x10..0x14].try_into().unwrap()) as usize;
    let mut off = 0x20usize;
    let (mut symoff, mut nsyms, mut stroff, mut strsize) = (0usize, 0usize, 0usize, 0usize);
    for _ in 0..ncmds {
        let cmd = u32::from_le_bytes(skel[off..off + 4].try_into().unwrap());
        let csz = u32::from_le_bytes(skel[off + 4..off + 8].try_into().unwrap()) as usize;
        if cmd == 2 {
            symoff = u32::from_le_bytes(skel[off + 8..off + 12].try_into().unwrap()) as usize;
            nsyms = u32::from_le_bytes(skel[off + 12..off + 16].try_into().unwrap()) as usize;
            stroff = u32::from_le_bytes(skel[off + 16..off + 20].try_into().unwrap()) as usize;
            strsize = u32::from_le_bytes(skel[off + 20..off + 24].try_into().unwrap()) as usize;
        }
        off += csz;
    }
    let strtab = &skel[stroff..stroff + strsize];
    let name = |o: u32| -> String {
        let o = o as usize;
        let e = strtab[o..].iter().position(|&b| b == 0).unwrap();
        String::from_utf8(strtab[o..o + e].to_vec()).unwrap()
    };
    let mut entries = Vec::with_capacity(nsyms);
    for i in 0..nsyms {
        let e = &skel[symoff + i * 16..symoff + i * 16 + 16];
        entries.push((
            name(u32::from_le_bytes(e[0..4].try_into().unwrap())),
            e[4],
            e[5],
            u16::from_le_bytes(e[6..8].try_into().unwrap()),
            u64::from_le_bytes(e[8..16].try_into().unwrap()),
        ));
    }
    let relocs = reloc_entries(skel, 0x218);
    (entries, relocs)
}

/// `__text` relocation entries, located through the `reloff`/`nreloc`
/// field pair at `reloff_field` (header `0x218` on single-section
/// skeletons; the `__text` section entry on multi-section families).
fn reloc_entries(skel: &[u8], reloff_field: usize) -> Vec<[u8; 8]> {
    let reloff = u32::from_le_bytes(skel[reloff_field..reloff_field + 4].try_into().unwrap()) as usize;
    let n = u32::from_le_bytes(skel[reloff_field + 4..reloff_field + 8].try_into().unwrap()) as usize;
    (0..n)
        .map(|i| skel[reloff + 8 * i..reloff + 8 * i + 8].try_into().unwrap())
        .collect()
}

/// Rebuild the variable tail after `tail_off` — signature command,
/// `rt.version`, `LC_SYMTAB`, symtab, strtab, then the `relocs`
/// `__text` relocation entries — and pad to the section's `0x4000`
/// alignment boundary. Returns the image prefix and the `__text`
/// section file offset (which floats past `0x4000` when the tail
/// overflows it); callers append the section content. Fixes
/// `sizeofcmds` and the `__text` reloff field.
fn build_tail_at(
    skel: &[u8],
    tail_off: usize,
    entries: &[(String, u8, u8, u16, u64)],
    relocs: &[[u8; 8]],
    mil_path: &str,
    out_path: &str,
) -> (Vec<u8>, usize) {
    build_tail_at_reloc(skel, tail_off, entries, relocs, mil_path, out_path, 0x218)
}

/// [`build_tail_at`] with an explicit `reloff_field` — the image offset
/// of the `reloff`/`nreloc` field pair that receives the rebuilt
/// relocation table address (`0x218` normally; the `__text` section
/// entry on the matmul family whose record floats past it).
fn build_tail_at_reloc(
    skel: &[u8],
    tail_off: usize,
    entries: &[(String, u8, u8, u16, u64)],
    relocs: &[[u8; 8]],
    mil_path: &str,
    out_path: &str,
    reloff_field: usize,
) -> (Vec<u8>, usize) {
    let sig = signature(mil_path, out_path);
    let sig_pad = (8 - (sig.len() + 1) % 8) % 8 + 1;
    let csz = 8 + sig.len() + sig_pad;

    let mut tail = Vec::new();
    tail.extend_from_slice(&8u32.to_le_bytes());
    tail.extend_from_slice(&(csz as u32).to_le_bytes());
    tail.extend_from_slice(sig.as_bytes());
    tail.extend(std::iter::repeat(0u8).take(sig_pad));

    tail.extend_from_slice(&[
        0x31, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, b'r', b't', b'.', b'v', b'e', b'r', b's',
        b'i', b'o', b'n', 0, 0, 0, 0, 0, 0, 0x28, 0x00, 0x00, 0x00, 0, 0, 0, 0, 0x04, 0, 0, 0, 0,
        0, 0, 0, 0x01, 0, 0x01, 0, 0, 0, 0, 0,
    ]);

    let symoff = tail_off + tail.len() + 0x18;
    let mut strtab = vec![0u8];
    let strxs: Vec<u32> = entries
        .iter()
        .map(|(n, ..)| {
            let o = strtab.len() as u32;
            strtab.extend_from_slice(n.as_bytes());
            strtab.push(0);
            o
        })
        .collect();
    let stroff = symoff + entries.len() * 16;
    tail.extend_from_slice(&0x2u32.to_le_bytes());
    tail.extend_from_slice(&0x18u32.to_le_bytes());
    tail.extend_from_slice(&(symoff as u32).to_le_bytes());
    tail.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    tail.extend_from_slice(&(stroff as u32).to_le_bytes());
    tail.extend_from_slice(&(strtab.len() as u32).to_le_bytes());
    for ((_, ty, sec, desc, val), &sx) in entries.iter().zip(&strxs) {
        tail.extend_from_slice(&sx.to_le_bytes());
        tail.push(*ty);
        tail.push(*sec);
        tail.extend_from_slice(&desc.to_le_bytes());
        tail.extend_from_slice(&val.to_le_bytes());
    }
    tail.extend_from_slice(&strtab);

    let mut img = skel[..tail_off].to_vec();
    if !relocs.is_empty() {
        let reloff = tail_off + (tail.len() + 7) / 8 * 8;
        while tail_off + tail.len() < reloff {
            tail.push(0);
        }
        for r in relocs {
            tail.extend_from_slice(r);
        }
        put_u32(&mut img, reloff_field, reloff as u32);
    }
    img.extend_from_slice(&tail);
    put_u32(&mut img, 0x14, (tail_off + csz + 0x30 + 0x18 - 0x20) as u32);
    let text_off = (img.len() + 0x3fff) & !0x3fff;
    img.resize(text_off, 0);
    (img, text_off)
}

/// Rebuild the tail and splice the skeleton's section content at
/// `0x4000` back on — the shared-layout fast path used by the
/// single-stream emitters.
fn rebuild_tail_at(
    skel: &[u8],
    tail_off: usize,
    entries: &[(String, u8, u8, u16, u64)],
    relocs: &[[u8; 8]],
    mil_path: &str,
    out_path: &str,
) -> Vec<u8> {
    let (mut img, text_off) = build_tail_at(skel, tail_off, entries, relocs, mil_path, out_path);
    debug_assert_eq!(text_off, 0x4000);
    img.extend_from_slice(&skel[0x4000..]);
    img
}

/// Emit a complete HWX image for `z = conv1x1(x, w)` on a skeleton
/// family. `weights` is the raw `[C,C]` fp16 matrix (row-major `u16`
/// elements); it is interleaved per [`weight_interleave`] into
/// `__KERN_0`. All other bytes come from the reference skeleton except
/// the signature's `-i`/`-o` paths — the tail is rebuilt so a longer
/// signature re-offsets symtab/strtab/relocation correctly.
pub fn emit_conv1x1(
    shape: ConvShape,
    weights: &[u16],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    let c = shape.channels();
    if weights.len() < c * c {
        return Err(format!("expected {} weights, got {}", c * c, weights.len()));
    }
    let weights = &weights[..c * c];
    let skel = shape.skeleton();
    let (entries, relocs) = tail_meta(skel);
    let mut img = rebuild_tail_at(skel, CONV_TAIL_OFF, &entries, &relocs, mil_path, out_path);

    let stream = weight_interleave(weights, c);
    for (i, v) in stream.iter().enumerate() {
        img[CONV_WEIGHTS_OFF + i * 2..CONV_WEIGHTS_OFF + i * 2 + 2]
            .copy_from_slice(&v.to_le_bytes());
    }
    Ok(img)
}

/// Emit a complete HWX image for the fused chain
/// `z = sigmoid(conv1x1(x, w))` — the real-model pattern — for `C=64`.
///
/// `__KERN_0` layout differs from bare conv: sixteen bands of
/// `0x280` bytes each, `[0x80 sigmoid LUT][0x200 band weights]`
/// spanning `0xc000..0xe800`. The LUT bytes come from the skeleton
/// (fixed 128-byte table, identical in every band); only the
/// `0x200`-byte weight tail of each band is spliced in.
pub fn emit_conv_sig(weights: &[u16], mil_path: &str, out_path: &str) -> Result<Vec<u8>, String> {
    emit_conv_act(FusedAct::Sigmoid, weights, mil_path, out_path)
}

/// Emit a complete HWX image for the single fused layer
/// `z = act(conv1x1(x, w))` for `C=64` and activation family `act`.
///
/// Each family has its own k=1 skeleton: sigmoid/tanh share the
/// `[0x80 LUT][0x200 weights]` band layout (record bytes identical,
/// differing only in LUT contents and kernel names); relu is the
/// LUT-free form with `0x200`-byte bands and a shorter descriptor.
pub fn emit_conv_act(
    act: FusedAct,
    weights: &[u16],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    const C: usize = 64;
    if weights.len() < C * C {
        return Err(format!("expected {} weights, got {}", C * C, weights.len()));
    }
    let weights = &weights[..C * C];
    let skel = act.skeleton1();
    let (entries, relocs) = tail_meta(skel);
    let mut img = rebuild_tail_at(skel, CONV_TAIL_OFF, &entries, &relocs, mil_path, out_path);

    let stream = weight_interleave(weights, C);
    let band = act.band_stride();
    let head = if act.lut().is_some() { 0x80 } else { 0 };
    for b in 0..16usize {
        let dst = CONV_WEIGHTS_OFF + b * band + head;
        for (i, v) in stream[b * 256..b * 256 + 256].iter().enumerate() {
            img[dst + i * 2..dst + i * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    Ok(img)
}

/// k=1 fused conv+tanh skeleton (identical record geometry to the
/// sigmoid form; carries the tanh LUT and kernel names).
const SKELETON_CONV64_TANH: &[u8] = include_bytes!("../assets/conv64_tanh_skeleton.hwx");
/// k=1 fused conv+relu skeleton — the LUT-free family: `0x200`
/// weight bands, `0x2000` `__KERN_0` payload, `0x11` descriptor
/// element count.
const SKELETON_CONV64_RELU: &[u8] = include_bytes!("../assets/conv64_relu_skeleton.hwx");

/// k=1 fused conv+gelu skeleton — sigmoid-layout family member
/// (gelu `EXACT` mode; `TANH_APPROXIMATION` not yet mined).
const SKELETON_CONV64_GELU: &[u8] = include_bytes!("../assets/conv64_gelu_skeleton.hwx");

/// Gelu band-head LUT, mined from `chain_conv_gelu`.
const GELU_LUT: [u8; 0x80] = {
    let t = SKELETON_CONV64_GELU;
    let mut o = [0u8; 0x80];
    let mut i = 0;
    while i < 0x80 {
        o[i] = t[CONV_WEIGHTS_OFF + i];
        i += 1;
    }
    o
};

/// k=1 fused conv+silu skeleton — the `mul(t, sigmoid(t))` triple
/// lowered to a single fused LUT task.
const SKELETON_CONV64_SILU: &[u8] = include_bytes!("../assets/conv64_silu_skeleton.hwx");

/// Silu band-head LUT, mined from `conv64_silu_skeleton`.
const SILU_LUT: [u8; 0x80] = {
    let t = SKELETON_CONV64_SILU;
    let mut o = [0u8; 0x80];
    let mut i = 0;
    while i < 0x80 {
        o[i] = t[CONV_WEIGHTS_OFF + i];
        i += 1;
    }
    o
};

/// k=1 conv+softmax skeleton — the multi-task record family: the
/// program decomposes into ~4 task descriptors (conv + softmax
/// reduce/normalize/output stages), `__KERN_0` is `0x2100` = 16
/// contiguous `0x200` weight chunks + a `0x100` aux region.
const SKELETON_CONV64_SM: &[u8] = include_bytes!("../assets/conv64_sm_skeleton.hwx");

/// Softmax-family kernel names by stream ordinal — only stream 0 is
/// mined (the conv task). The softmax program additionally carries
/// two `_actlut` symbols (exp/reciprocal tables) at `+0x2000`/`+0x2080`
/// in `__KERN_0`'s aux tail — skeleton-supplied for k=1 emission.
const KNAME_SOFTMAX: [&str; 8] = [
    "K94191614DE4DACE8C293DBDB924D98C1ADD491137600442C695C5B04E2D79E8C",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
];

/// k=2 fused conv+sigmoid chain skeleton — provides the `i=1` form
/// of the terminal record (the only depth where position 1 is last).
const SKELETON_CONV_SIG2: &[u8] = include_bytes!("../assets/conv_sig2_reference.hwx");
/// k=5 fused conv+sigmoid chain skeleton — provides the `i>=2`
/// interior and terminal record forms plus the program header.
const SKELETON_CONV_SIG5: &[u8] = include_bytes!("../assets/conv_sig5_skeleton.hwx");

// Record-stream geometry inside `__TEXT` section content (file
// offsets — the section maps verbatim at 0x4000).
/// Program header preceding the per-layer records.
const REC_HDR: (usize, usize) = (0x4000, 0x30);
/// Layer-0 record (prologue + first fused task).
const REC_FIRST: (usize, usize) = (0x4030, 0x190);
/// Interior record, position-1 form — positions `i>=2` insert an
/// extra release-buffer element into the sigmoid wiring, eating into
/// the record's trailing pad so the record stays `0x160` bytes.
const REC_INT: (usize, usize) = (0x41c0, 0x160);
/// Terminal record at position `i>=2` (`0x170` bytes incl. epilogue).
const REC_LAST: (usize, usize) = (0x45e0, 0x170);
/// Terminal record at position `i==1` (k=2 only; `0x16c` bytes).
const REC_LAST1: (usize, usize) = (0x41c0, 0x16c);

/// Patch the sixteen `21 00 |<i16>| 00` lane-index fields of a
/// conv+sig record to layer index `i` (u16 at +0x0e, stride 4).
/// The +0x30 slot is a different descriptor (`15 51`/`00 03`) and is
/// left alone — the table is +0x0e..+0x2e and +0x36..+0x4e.
fn patch_lane_idx(rec: &mut [u8], i: usize) {
    let v = (i as u16).to_le_bytes();
    for o in (0x0e..=0x4e).step_by(4) {
        if o == 0x32 {
            continue;
        }
        rec[o..o + 2].copy_from_slice(&v);
    }
}

/// Patch a conv+act record for its activation family: the
/// band-descriptor table (`+0x54` offset ramp `stride*(b+1)`, `+0x90`
/// sixteen stride entries) and the fused-op element count `u16` at
/// `41 12 02 80` marker +6 (`0x12` LUT families, `0x11` relu). For
/// sigmoid/tanh every write is a no-op against the sigmoid skeleton
/// bytes; relu rewrites both fields.
fn patch_record_act(rec: &mut [u8], act: FusedAct) {
    let stride = act.band_stride() as u32;
    for b in 0..15 {
        rec[0x54 + 4 * b..0x58 + 4 * b]
            .copy_from_slice(&(stride * (b as u32 + 1)).to_le_bytes());
    }
    for b in 0..16 {
        rec[0x90 + 4 * b..0x94 + 4 * b].copy_from_slice(&stride.to_le_bytes());
    }
    let mut m = 0;
    while m + 8 <= rec.len() {
        if rec[m..m + 4] == [0x41, 0x12, 0x02, 0x80] {
            rec[m + 6..m + 8].copy_from_slice(&(act.desc_count() as u16).to_le_bytes());
        }
        m += 1;
    }
}

/// Emit a complete HWX image for the k-layer fused chain
/// `x -> (conv1x1 -> sigmoid)^k -> z` sharing one `C=64` weight blob —
/// the transformer-FFN stacking pattern. Records are spliced
/// per-position: first (`0x190`), interior (`0x160`), terminal
/// (`0x16c`/`0x170` with epilogue). `__KERN_0` carries the same
/// banded `[LUT][weights]` payload as the single-layer emitter —
/// identical weights are deduplicated into one stream by the ANE
/// allocator, so the payload is independent of `k`.
///
/// Record-field model (decoded from ANEC k=1..5 references):
/// - `+0x00` position flag: `01` first/interior, `09` last
/// - `+0x08` output descriptor: `40 00` interior, `c0 02` last
/// - `+0x09` weight-stream release: `02` at the last use of the
///   record's stream, else `00`; `+0x0a` = stream ordinal + 1
/// - `+0x0e..+0x4e` lane indices = layer index `i`
/// - `+0x100` fused-sigmoid wiring: buffer release/alloc pairs
///   `(4(i-1), 0x10)` / `(4i, 0x10)` around a `52 10/90` gate word,
///   `41 10 04 80` header at `i==1`, `41 10 06 80` at `i>=2`
/// - u32 after each `44 95 80 23` marker = the record's
///   `__KERN_0` weight-stream offset (`ordinal * 0x2800`); also the
///   relocation target for that record
/// - `+0x140`/`+0x150`/`+0x158` (interior) and `+0x170`/`+0x180`/
///   `+0x188` (first): next-record-is-terminal markers
///   (`0x56/0x64`, `0x20/0x60`, `1/0`)
/// - terminal epilogue at `+0x128` with layer index at the
///   `e0 00 <i>` and `31 00 <i> 01` slots
///
/// FVMLIB scalars: `0x378` = `0x8000 + align(stream_len, 0x40)`,
/// `0xb88` = k, `0xbb8` = `0xe*k + 3` (k>=2), `0xb7c` per lookup
/// (byte-length of a payload descriptor list).
pub fn emit_conv_sig_n(
    layers: usize,
    weights: &[u16],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    emit_conv_act_n(FusedAct::Sigmoid, layers, weights, mil_path, out_path)
}

/// Shared-weight k-layer fused chain `x -> (conv1x1 -> act)^k -> z`.
pub fn emit_conv_act_n(
    act: FusedAct,
    layers: usize,
    weights: &[u16],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    if layers == 0 {
        return Err("layers must be >= 1".into());
    }
    let ws = vec![weights; layers];
    emit_conv_act_n_dw(act, layers, &ws, mil_path, out_path)
}

/// Fused activation families for conv+act chains — what the ANE
/// compiler folds into the conv kernel's tail. LUT families carry a
/// 128-byte table at the head of each weight band; relu has no table
/// and uses a denser `0x200` band stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedAct {
    Sigmoid,
    Tanh,
    Relu,
    Gelu,
    /// `mul(t, sigmoid(t))` — ANEC pattern-matches the triple into the
    /// same fused LUT kernel, so record geometry is sigmoid-identical
    /// and only the LUT bytes and kernel names differ.
    Silu,
    /// `softmax` — the multi-task record family: each layer emits
    /// three task descriptors (conv + sm-reduce + sm-final). The
    /// conv task uses `0x200` weight-only bands (contiguous in
    /// `__KERN_0`, plus a `0x100` exp/reciprocal LUT tail); shared
    /// chains emit via [`emit_conv_sm_n`], k=1 via `emit_conv_act`.
    Softmax,
}

impl FusedAct {
    /// Per-band stride in `__KERN_0`.
    fn band_stride(self) -> usize {
        match self {
            FusedAct::Relu | FusedAct::Softmax => 0x200,
            _ => 0x280,
        }
    }
    /// Bytes per `__KERN_0` weight stream (16 bands).
    fn stream_size(self) -> usize {
        16 * self.band_stride()
    }
    /// Element count in the fused-op descriptor (`41 12 02 80`+6):
    /// LUT activations carry one extra buffer element.
    fn desc_count(self) -> u8 {
        match self {
            FusedAct::Relu => 0x11,
            _ => 0x12,
        }
    }
    /// 128-byte activation LUT at the head of every band.
    fn lut(self) -> Option<&'static [u8; 0x80]> {
        match self {
            FusedAct::Sigmoid => Some(&SIGMOID_LUT),
            FusedAct::Tanh => Some(&TANH_LUT),
            FusedAct::Gelu => Some(&GELU_LUT),
            FusedAct::Silu => Some(&SILU_LUT),
            FusedAct::Relu | FusedAct::Softmax => None,
        }
    }
    /// Kernel names indexed by stream ordinal.
    fn knames(self) -> &'static [&'static str; 8] {
        match self {
            FusedAct::Sigmoid => &KERNEL_NAME_BY_STREAM,
            FusedAct::Tanh => &KNAME_TANH,
            FusedAct::Relu => &KNAME_RELU,
            FusedAct::Gelu => &KNAME_GELU,
            FusedAct::Silu => &KNAME_SILU,
            // multi-task family: no mined stream-name table yet
            FusedAct::Softmax => &KNAME_SOFTMAX,
        }
    }
    /// k=1 fused-op skeleton for this family.
    fn skeleton1(self) -> &'static [u8] {
        match self {
            FusedAct::Sigmoid => SKELETON_CONV64_SIG,
            FusedAct::Tanh => SKELETON_CONV64_TANH,
            FusedAct::Relu => SKELETON_CONV64_RELU,
            FusedAct::Gelu => SKELETON_CONV64_GELU,
            FusedAct::Silu => SKELETON_CONV64_SILU,
            FusedAct::Softmax => SKELETON_CONV64_SM,
        }
    }
}

/// Sigmoid band-head LUT (128 bytes), mined from ANEC artifacts.
const SIGMOID_LUT: [u8; 0x80] = {
    let skel = SKELETON_CONV64_SIG;
    let mut t = [0u8; 0x80];
    let mut i = 0;
    while i < 0x80 {
        t[i] = skel[CONV_WEIGHTS_OFF + i];
        i += 1;
    }
    t
};

/// Tanh band-head LUT, mined from `chain_conv_tanh`.
const TANH_LUT: [u8; 0x80] = {
    let t = SKELETON_CONV64_TANH;
    let mut o = [0u8; 0x80];
    let mut i = 0;
    while i < 0x80 {
        o[i] = t[CONV_WEIGHTS_OFF + i];
        i += 1;
    }
    o
};

/// `K<sha256>_ne_<lane>` kernel names for the `__KERN_0` weight-stream
/// band groups of the C=64 fused conv+sigmoid family, indexed by
/// stream ordinal (`offset / 0x2800`). The hash covers ANEC's
/// internal kernel descriptor keyed on the stream offset — verified
/// weight-content-independent (swapping blob contents across streams
/// does not move the names). Mined from ANEC dw-chain artifacts;
/// chains needing more streams must mine further names.
const KERNEL_NAME_BY_STREAM: [&str; 8] = [
    "K471CD71E4391E230B902F296B69E5C520A7450E528ADDDECBA7C4B89BF69E305",
    "K92076DC27A23A00AF80F5C7FEE82ECAE76AFCF0798A5DBC62E018B79689621F7",
    "KCB4FC07CDB3A236D799A29B19EE78706DD656A97C6F03E7A7305D33C1C28A7FA",
    "K4F0A8C558FC8858B7F053216004595AE14C49FF642AA70F4A776D04E80821457",
    "K00CB9FC85E0D61B8D56401323B3377D3491AFCE66043902D05283FD80853D74F",
    "KAC5658BCDBBC20FEAD05C3C7CA32BDA57E0C1E3022DA043890A0C582AD84029C",
    "KDAB1BD6B4EE32537508DF702D04D4D97BAB62348B1164420A810790C675939CF",
    "KB03EFB6AB1251DC6329B01C8A2A5077965107E4FB519A359D1D8D708B9A76A85",
];

/// Tanh-family kernel names by stream ordinal (`offset / 0x2800`).
const KNAME_TANH: [&str; 8] = [
    "KCB01050CFDA60841DC78E7E6A40A87B8039DCB162F0F39BB0F715F6990F260F3",
    "KCE028E8C30B8A99C6A4C72E60C2789515E4FF7012EC616D2787FF055B6167AFF",
    "KAD9265DE89E606BC87E8566DEB409C831F4416893F4937B8949034A0B371558F",
    "K5BB67FF751069E230327742FEC42A663831E4E52C2A64DC1A867B841385A731F",
    "K8C0C3043F8A840640A1DDE5FB157E3D889B48E67C40B83D5767116C46930374E",
    "K8948BDD324B566ABD2F26ADE90A4CFACAC888B9D7296537E3DFD533515EEEB5B",
    "KF2FDF237B05B88C3590A3E99C97C40FDC6722FA39D7F281F11C9EF706B79B6D7",
    "K53641C79165EDF191B443016DE9DA67CF42BB3FA6CFB06BF91651F8B6FF43006",
];

/// Relu-family kernel names by stream ordinal (`offset / 0x2000`).
const KNAME_RELU: [&str; 8] = [
    "K923BD8B34B9065A305AACB9E27DFF3177ED35CF5AF9CECF9DF7BB9EC94047DE2",
    "KBEE157892D70D61D719BB7D8071948C39FD62BEDDF9F9D2E98C1B7960B123BC0",
    "K7D06C0B1A2C4ACEBE4BA887179B991BA61D101751CEA5F5FDBF48E3516F5076B",
    "KE38618D47393A7FF76CDEAF92E1BA1ED7E362AC0D8424AFA1503A92480D0B80D",
    "K9E3C06093273BB1BCC5019995492143534B69A1097BDBFF43DCC65BDE487B0DA",
    "K5303A3B852F86C1D39B8B5A057E5C08C4C08CEC0D5BFDE9D451D1CFD5E30F759",
    "KF27D0F46516949E5A75C7C5EA6855C4DD90D1927F9D98AB0EFFFDB7D9D6F7437",
    "KD904D046DD3308218EF464C12742276B8E854770ACF6A21091C2F20B4A87001B",
];

/// Gelu-family kernel names by stream ordinal (`offset / 0x2800`).
const KNAME_GELU: [&str; 8] = [
    "K3D216797AB3DE2F3DF01E439006E24578C7FFEEA13233932238F3676268575D0",
    "KE8BE9FE86A0E25227C0528CC13E26C7853B941975120893A3A90E317705018C9",
    "KB47C60104FE03EF329310D96FE9D2EF91E20C9F85B85A565EE7065F71F381756",
    "KD7E2AD4EE920B2A3315E4022D3EAA7278B02EAE498CC9FE79E4F7A34338C62BD",
    "K2CC5BC9A41207938E891C6230A998FE9E9E5EC158F0F19DDF0580320CBD8D0D9",
    "K53233454AA2E9E748AD8A556A6BFB9C8FB61D0B2B72FF4F737C214C8EDBD1290",
    "K1FB68BF6CE04899ED637EBA1CABEC51781E671E0F6594C25A038B66CC44E11B6",
    "KD48F54858D73FDA7FBDF455FB1A66821CB9A1EDC6B71707E3FD55988DB581FB5",
];

/// Silu-family kernel names by stream ordinal (`offset / 0x2800`).
const KNAME_SILU: [&str; 8] = [
    "K5AF90A3CDAE04A725C406BD1F7984344F0F13AC3F9DE958B72B29E2090B191F3",
    "K566207E1FFD51D54617A6F5F235878E0896B798C2A53901208FECC6E6C500A75",
    "KE953D3D588EA3676297DCFF0E686EC2CEF6F5E7B6037903C793D2CE6734F6FF2",
    "K1956C4C1B88ED8ACB22F90636DD9D29ED7D34CC3041285F43CA81AE1AAA03032",
    "KBAC32A7AF798ED01D0FE6A85A731F9F72FB81F694B47A0A816E02DF94E1B4347",
    "KEE80A6BDAE435FA4C685114EAC216AA4AD7BB2BF6C0551070DB9EEA348ACD022",
    "K33DE195169E522F5DB9A50526EDE1D6C5FEB146EAAFCBF759F6E777B3FACFF8D",
    "KDE8904E907AD78C992BAA1A7E06423DF9BA6E974A0DE9FC00317CC635992FD1F",
];

/// Section-relative offset of each layer record's weight-stream
/// field — the u32 right after the `44 95 80 23` marker, which is
/// also the record's relocation `r_address`.
fn chain_stream_field(i: usize, layers: usize) -> usize {
    if i == 0 {
        0x180
    } else {
        let rec_rel = 0x1c0 + (i - 1) * 0x160;
        rec_rel
            + if i == layers - 1 {
                if i == 1 {
                    0x150
                } else {
                    0x154
                }
            } else if i == 1 {
                0x12c
            } else {
                0x130
            }
    }
}

/// Emit a complete HWX image for `(conv1x1 -> sigmoid)^k` where each
/// layer references its own `C=64` weight blob. Blobs are
/// deduplicated by content into `__KERN_0` band-group streams (one
/// `0x2800` group per unique blob, in first-use order); each record
/// carries its stream ordinal, a last-use release flag, and the
/// stream's byte offset in the relocated descriptor field.
///
/// Multi-stream artifacts restructure: the `__text` plane floats to
/// the next `0x4000` boundary when the expanded symtab/strtab/reloc
/// tail overflows `0x4000`, `__KERN_0` floats to the next boundary
/// after `__const`, and the symtab gains one `K<hash>_ne_<lane>`
/// name set per stream.
pub fn emit_conv_sig_n_dw(
    layers: usize,
    weights: &[&[u16]],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    emit_conv_act_n_dw(FusedAct::Sigmoid, layers, weights, mil_path, out_path)
}

/// Emit a complete HWX image for `(conv1x1 -> act)^k` with per-layer
/// weight blobs — the activation-family generalization of
/// `emit_conv_sig_n_dw`. Sigmoid and tanh share the LUT-band layout;
/// relu uses `0x200` weight-only bands, `0x2000` streams, and an
/// `0x11` fused-descriptor element count.
pub fn emit_conv_act_n_dw(
    act: FusedAct,
    layers: usize,
    weights: &[&[u16]],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    const C: usize = 64;
    if layers == 0 || weights.len() != layers {
        return Err(format!("expected {layers} weight blobs, got {}", weights.len()));
    }
    let weights: Vec<&[u16]> = weights
        .iter()
        .map(|w| -> Result<&[u16], String> {
            if w.len() < C * C {
                return Err(format!("expected {} weights, got {}", C * C, w.len()));
            }
            Ok(&w[..C * C])
        })
        .collect::<Result<_, _>>()?;
    let weights = &weights[..];
    if layers == 1 {
        return emit_conv_act(act, weights[0], mil_path, out_path);
    }
    if act == FusedAct::Softmax {
        return emit_conv_sm_n_dw(layers, weights, mil_path, out_path);
    }
    let stream_sz = act.stream_size();
    let band = act.band_stride();
    let knames = act.knames();
    // dedup blobs by content, first-use order
    let mut streams: Vec<&[u16]> = Vec::new();
    let mut ords = Vec::with_capacity(layers);
    for w in weights {
        match streams.iter().position(|s| *s == *w) {
            Some(o) => ords.push(o),
            None => {
                ords.push(streams.len());
                streams.push(w);
            }
        }
    }
    if streams.len() > knames.len() {
        return Err(format!(
            "{} distinct weight streams exceeds mined kernel-name table ({})",
            streams.len(),
            knames.len()
        ));
    }
    let mut last_use = vec![0usize; streams.len()];
    for (i, &o) in ords.iter().enumerate() {
        last_use[o] = i;
    }

    let stream_len = REC_HDR.1
        + REC_FIRST.1
        + REC_INT.1 * (layers - 2)
        + if layers == 2 { REC_LAST1.1 } else { REC_LAST.1 };
    if stream_len > 0x4000 {
        return Err(format!("chain of {layers} layers overflows the record plane"));
    }

    let skel = SKELETON_CONV_SIG5;
    let (skel_entries, skel_relocs) = tail_meta(skel);

    // symtab: [x, z] + ne_0 for each stream + per-stream ne_1..15 +
    // the scalar type-descriptor entries copied from the skeleton
    let mut entries = Vec::with_capacity(2 + 16 * streams.len() + skel_entries.len() - 18);
    entries.push(skel_entries[0].clone());
    entries.push(skel_entries[1].clone());
    // the `_ne_0` band symbol carries n_desc=2; `_ne_1..15` carry 0
    let tpl0 = skel_entries[2].clone();
    let tpln = skel_entries[3].clone();
    for (j, name) in knames.iter().enumerate().take(streams.len()) {
        entries.push((
            format!("{name}_ne_0"),
            tpl0.1,
            tpl0.2,
            tpl0.3,
            0x3001_0000 + stream_sz as u64 * j as u64,
        ));
    }
    for (j, name) in knames.iter().enumerate().take(streams.len()) {
        for b in 1..16u64 {
            entries.push((
                format!("{name}_ne_{b}"),
                tpln.1,
                tpln.2,
                tpln.3,
                0x3001_0000 + stream_sz as u64 * j as u64 + band as u64 * b,
            ));
        }
    }
    entries.extend_from_slice(&skel_entries[18..]);

    // one relocation per record at the weight-stream field
    let rflag: [u8; 4] = skel_relocs[0][4..8].try_into().unwrap();
    let relocs: Vec<[u8; 8]> = (0..layers)
        .map(|i| {
            let mut e = [0u8; 8];
            e[..4].copy_from_slice(&(chain_stream_field(i, layers) as u32).to_le_bytes());
            e[4..].copy_from_slice(&rflag);
            e
        })
        .collect();

    let (mut img, text_off) =
        build_tail_at(skel, CONV_TAIL_OFF, &entries, &relocs, mil_path, out_path);

    let mut rec = Vec::with_capacity(stream_len);
    rec.extend_from_slice(&skel[REC_HDR.0..REC_HDR.0 + REC_HDR.1]);

    // layer 0 — first record; its tail carries the next-terminal markers
    {
        let mut r = skel[REC_FIRST.0..REC_FIRST.0 + REC_FIRST.1].to_vec();
        patch_record_act(&mut r, act);
        let next_last = layers == 2;
        r[0x09] = if last_use[ords[0]] == 0 { 2 } else { 0 };
        r[0x0a] = ords[0] as u8 + 1;
        r[0x150..0x154].copy_from_slice(&((ords[0] * stream_sz) as u32).to_le_bytes());
        r[0x172..0x174].copy_from_slice(&(if next_last { 0x63u16 } else { 0x55u16 }).to_le_bytes());
        r[0x17b] = if next_last { 4 } else { 0 };
        r[0x180] = if next_last { 0x60 } else { 0x20 };
        r[0x18c..0x190].copy_from_slice(&(if next_last { 0u32 } else { 1 }).to_le_bytes());
        rec.extend_from_slice(&r);
    }

    // interior layers 1..layers-1 — sigmoid wiring is rebuilt per
    // position: [release (4(i-1),0x10)] [alloc (4i,0x10)] buffer pairs
    // around the `52 10 01 00` gate, marker at +0x11c (i==1) or +0x120.
    const SIG_MARK: [u8; 20] = [
        0x41, 0x12, 0x02, 0x80, 0x00, 0x3c, 0x12, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x44, 0x95, 0x80,
        0x23, 0x00, 0x00, 0x00, 0x00,
    ];
    for i in 1..layers - 1 {
        let mut r = skel[REC_INT.0..REC_INT.0 + REC_INT.1].to_vec();
        patch_lane_idx(&mut r, i);
        let mut w = Vec::with_capacity(0x34);
        if i == 1 {
            w.extend_from_slice(&[
                0x41, 0x10, 0x04, 0x80, 0x20, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x52, 0x10,
                0x01, 0x00, 0x40, 0x01, 0x00, 0x00,
            ]);
        } else {
            w.extend_from_slice(&[
                0x41, 0x10, 0x06, 0x80, 0x04, 0x01, 0x00, 0x00, 0x00, 4 * (i as u8 - 1), 0x00,
                0x00, 0x10, 0x00, 0x00, 0x00, 0x52, 0x10, 0x01, 0x00, 0x40, 0x01, 0x00, 0x00,
            ]);
        }
        w.extend_from_slice(&[0x00, 4 * i as u8, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00]);
        w.extend_from_slice(&SIG_MARK);
        r[0x100..0x100 + w.len()].copy_from_slice(&w);
        for b in &mut r[0x100 + w.len()..0x140] {
            *b = 0;
        }
        // stream-offset u32 follows the `44 95 80 23` marker
        let foff = if i == 1 { 0x12c } else { 0x130 };
        r[foff..foff + 4].copy_from_slice(&((ords[i] * stream_sz) as u32).to_le_bytes());
        r[0x09] = if last_use[ords[i]] == i { 2 } else { 0 };
        r[0x0a] = ords[i] as u8 + 1;
        let next_last = i + 1 == layers - 1;
        r[0x140..0x142].copy_from_slice(&(i as u16 + 1).to_le_bytes());
        r[0x142..0x144].copy_from_slice(&(if next_last { 0x64u16 } else { 0x56u16 }).to_le_bytes());
        for b in &mut r[0x144..0x150] {
            *b = 0;
        }
        r[0x14b] = if next_last { 4 } else { 0 };
        r[0x150] = if next_last { 0x60 } else { 0x20 };
        r[0x151] = 0xf8;
        r[0x152] = 0xff;
        for b in &mut r[0x153..0x15c] {
            *b = 0;
        }
        r[0x15c..0x160].copy_from_slice(&(if next_last { 0u32 } else { 1 }).to_le_bytes());
        // act patch last — the `41 12 02 80` marker lives inside the
        // rebuilt `w` wiring, not the template bytes
        patch_record_act(&mut r, act);
        rec.extend_from_slice(&r);
    }

    // terminal layer — record ends in the output/epilogue section;
    // layer index sits at `e0 00 <i>` (+0x142) and `31 00 <i> 01`.
    {
        let i = layers - 1;
        let foff = if i == 1 { 0x150 } else { 0x154 };
        let mut r = if i == 1 {
            let mut r = SKELETON_CONV_SIG2[REC_LAST1.0..REC_LAST1.0 + REC_LAST1.1].to_vec();
            r[0x13e] = i as u8; // e0 00 <i>
            r[0x16a] = i as u8; // 31 00 <i> 01
            r
        } else {
            let mut r = skel[REC_LAST.0..REC_LAST.0 + REC_LAST.1].to_vec();
            // buffer pair slots: `00 <v> 00 00` — value at byte +1
            r[0x109] = 4 * (i as u8 - 1);
            r[0x119] = 4 * i as u8;
            r[0x142] = i as u8; // e0 00 <i>
            r[0x16e] = i as u8; // 31 00 <i> 01
            r
        };
        patch_record_act(&mut r, act);
        patch_lane_idx(&mut r, i);
        r[0x09] = 2;
        r[0x0a] = ords[i] as u8 + 1;
        r[foff..foff + 4].copy_from_slice(&((ords[i] * stream_sz) as u32).to_le_bytes());
        rec.extend_from_slice(&r);
    }

    debug_assert_eq!(rec.len(), stream_len);
    img.extend_from_slice(&rec);
    // zero fill through the __const section, then pad __KERN_0 to the
    // next 0x4000 boundary and splice each stream's band group
    let const_end = text_off + (stream_len + 0x3f) / 0x40 * 0x40 + 0x4000;
    img.resize(const_end, 0);
    let kern_off = (const_end + 0x3fff) & !0x3fff;
    img.resize(kern_off + stream_sz * streams.len(), 0);
    for (j, w) in streams.iter().enumerate() {
        let stream = weight_interleave(w, C);
        for b in 0..16usize {
            let base = kern_off + j * stream_sz + b * band;
            // fixed 128-byte activation LUT, identical in every band
            // (relu carries no LUT: band == 0x200 weight bytes)
            let head = match act.lut() {
                Some(lut) => {
                    img[base..base + 0x80].copy_from_slice(lut);
                    0x80
                }
                None => 0,
            };
            for (i, v) in stream[b * 256..b * 256 + 256].iter().enumerate() {
                img[base + head + i * 2..base + head + i * 2 + 2]
                    .copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    img.resize(kern_off + (stream_sz * streams.len() + 0x3fff) & !0x3fff, 0);

    patch_chain_scalars(
        &mut img,
        layers,
        stream_len,
        streams.len(),
        stream_sz,
        text_off,
        kern_off,
    );
    Ok(img)
}

/// Layer-count FVMLIB/section scalars for fused conv+sigmoid chains
/// and the `__KERN_0` stream sizing for multi-stream (distinct
/// weight) layouts: `text_off` is the floated `__text` file offset,
/// `kern_off` the `__KERN_0` file offset.
fn patch_chain_scalars(
    img: &mut [u8],
    layers: usize,
    stream_len: usize,
    n_streams: usize,
    stream_sz: usize,
    text_off: usize,
    kern_off: usize,
) {
    let aligned = (stream_len + 0x3f) / 0x40 * 0x40;
    let kern_payload = stream_sz * n_streams;
    let kern_vsz = (kern_payload + 0x3fff) & !0x3fff;
    put_u32(img, 0x1c0, text_off as u32); // __TEXT segment fileoff
    put_u32(img, 0x208, stream_len as u32); // __text section size
    put_u32(img, 0x210, text_off as u32); // __text section fileoff
    put_u32(img, 0x21c, layers as u32); // __text nreloc
    img[0x250..0x252].copy_from_slice(&(0x8000u16 + aligned as u16).to_le_bytes()); // __const va
    put_u32(img, 0x260, (text_off + aligned) as u32); // __const fileoff
    put_u32(img, 0x2a0, kern_vsz as u32); // __KERN_0 segment vsize
    put_u32(img, 0x2a8, kern_off as u32); // __KERN_0 segment fileoff
    put_u32(img, 0x2b0, kern_vsz as u32); // __KERN_0 segment filesize
    img[0x2f0..0x2f8].copy_from_slice(&(kern_payload as u64).to_le_bytes()); // section size
    put_u32(img, 0x2f8, kern_off as u32); // __kern_0 section fileoff
    img[0x378..0x37a].copy_from_slice(&(0x8000u16 + aligned as u16).to_le_bytes());
    img[0xb7c..0xb7e].copy_from_slice(&chain_b7c(layers).to_le_bytes());
    img[0xb88..0xb8c].copy_from_slice(&(layers as u32).to_le_bytes());
    img[0xbb8..0xbba].copy_from_slice(&((0xe * layers + 3) as u16).to_le_bytes());
}

/// `0xb7c` — byte-length of the fused-op payload descriptor list.
/// Decoded from ANEC k=1..5 references; `0x1c + 0x58k` holds for
/// `k>=3`, the k=2 record set is one byte shorter.
fn chain_b7c(layers: usize) -> u16 {
    match layers {
        1 => 0x74,
        2 => 0xcb,
        k => (0x1c + 0x58 * k) as u16,
    }
}

/// k=3 conv+softmax chain skeleton — provides every multi-task
/// template: first-layer conv (`0x290`, carrying the input-DMA
/// prologue), interior conv (`0x270`), the `0xf0` softmax reduce
/// task, the `0x170` interior softmax-final task, and the `0x190`
/// terminal form with the output-writeback epilogue.
const SKELETON_CONV_SM3: &[u8] = include_bytes!("../assets/conv_sm3_skeleton.hwx");

/// Distinct-weight softmax kernel names by stream ordinal — the
/// conv-task descriptor hash in multi-stream programs (stream 0
/// uses `K9419..` only when it is the program's sole stream). Mined
/// from ANEC dw probes; ordinals beyond 3 need further mining.
const KNAME_SM_DW: [&str; 4] = [
    "KE3E0F4E952E53E653B75FE6AF44DD22939751A660FEF245DA535F7F503E9AC52",
    "K36F9FAA5AC1598F61E735E999A791F3AF94F26320DC08F4A47655BE0F56D009C",
    "K445C1528F57CD5C3F8981EF604408231AE7ABE5C085A6D3966F1B4E5B85CDB85",
    "K8352046B90FD96625782305C20104AE0C2A4CE2D9D985157FCA0FC20945A859E",
];

// Multi-task templates inside the sm3 `__text` section — section
// coordinates; the record stream is a 4-byte `01 00 05 00` prologue
// then per-layer task blocks (file offsets in the skeleton).
/// Program header + record prologue (`0x30` header, `01 00 05 00`).
const SM_PROG: (usize, usize) = (0x4000, 0x34);
/// Layer-0 conv task — `0x290` with the input-DMA prologue.
const SM_CONV_FIRST: (usize, usize) = (0x4034, 0x290);
/// Interior/terminal conv task — `0x270`.
const SM_CONV_INT: (usize, usize) = (0x4524, 0x270);
/// Softmax reduce task — `0xf0` (interior form; the layer-0 flag
/// differs at +0x95).
const SM_SMA: (usize, usize) = (0x4794, 0xf0);
/// Softmax-final task, interior form — `0x170`.
const SM_SMB_INT: (usize, usize) = (0x4884, 0x170);
/// Softmax-final task, terminal form — `0x190` with the
/// output-writeback epilogue.
const SM_SMB_TERM: (usize, usize) = (0x4d54, 0x190);

fn put_u16(b: &mut [u8], off: usize, v: u16) {
    b[off..off + 2].copy_from_slice(&v.to_le_bytes());
}

/// Producer-task-id column slots — u16 fields at +0x0a..+0x2a and
/// +0x32..+0x4a stride 4 (skipping the descriptor at +0x2e). The
/// softmax tasks carry a 17th slot at +0x52; in the conv task that
/// slot is a different descriptor and stays zero.
const TASK_COL: [usize; 17] = [
    0x0a, 0x0e, 0x12, 0x16, 0x1a, 0x1e, 0x22, 0x26, 0x2a, 0x32, 0x36, 0x3a, 0x3e, 0x42, 0x46,
    0x4a, 0x52,
];

/// Patch the producer-task-id slots of a task descriptor — the 16
/// shared column slots plus +0x52 for softmax (non-conv) tasks.
fn patch_task_column(t: &mut [u8], v: u16, n: usize) {
    for &o in &TASK_COL[..n] {
        put_u16(t, o, v);
    }
}

/// Emit a complete HWX image for `(conv1x1 -> softmax)^k` sharing one
/// `C=64` weight blob — the multi-task record family. Each layer
/// emits three task descriptors: the conv task (which internally
/// manages sub-task ids `6i+1..6i+3`: input DMA, compute, output),
/// a softmax reduce task (id `6i+4` consuming `6i+3`), and a
/// softmax-final task (id `6i+5` consuming `6i+4`). The next layer's
/// conv reads the previous layer's output task `6(i+1)`; the last
/// layer's final task grows an output-writeback epilogue.
///
/// Task-id model decoded from ANEC sm1..sm3 references:
/// - producer ids ride in a 17-slot u16 column (+0x0a..+0x52)
/// - task ids at header +0x06 and +0xcc
/// - conv flag +0x05: `02` on the last layer's conv, else `00`
/// - sm-a +0x95: `04` at layer 0 (no predecessor), else `08`
/// - sm-b +0xa1: `08` interior, `04` terminal; +0x14c = next layer's
///   conv-input task `6(i+1)` interior, `0x1140` output DMA terminal
/// - three relocations per layer: conv weight-stream u32, sm-a's
///   exp-LUT (`+0x2000`) and sm-b's recip-LUT (`+0x2080`) targets
///
/// `__KERN_0` = 16 contiguous `0x200` weight chunks (`0x2000`) plus
/// the `0x100` shared exp/reciprocal LUT tail — one stream total for
/// shared weights.
pub fn emit_conv_sm_n(
    layers: usize,
    weights: &[u16],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    emit_conv_sm_n_dw(layers, &vec![weights; layers], mil_path, out_path)
}

/// `emit_conv_sm_n` with per-layer weight blobs. Blobs are
/// deduplicated by content into `__KERN_0` weight streams; each
/// conv task carries its stream's last-use release flag (+0x05),
/// the deduplicated kernel id `6*first_use(ord)+1` (+0x06), and the
/// stream's byte offset (reloc'd u32). Multi-stream artifacts
/// interleave the per-stream `K<hash>_ne_0` symbols with the shared
/// actlut entries and float `__text`/`__KERN_0` to later pages.
///
/// Stream offsets: stream 0 spans `0x0..0x2100` (weights + the
/// program's shared exp/recip LUT tail); stream `j>0` spans
/// `0x100 + j*0x2000`. `__KERN_0` payload = `0x100 + n*0x2000`,
/// segment size = `align(payload, 0x4000)`, fileoff = `text_off +
/// 0x8000`.
pub fn emit_conv_sm_n_dw(
    layers: usize,
    weights: &[&[u16]],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    const C: usize = 64;
    if layers == 0 || weights.len() != layers {
        return Err(format!("expected {layers} weight blobs, got {}", weights.len()));
    }
    let weights: Vec<&[u16]> = weights
        .iter()
        .map(|w| -> Result<&[u16], String> {
            if w.len() < C * C {
                return Err(format!("expected {} weights, got {}", C * C, w.len()));
            }
            Ok(&w[..C * C])
        })
        .collect::<Result<_, _>>()?;
    let weights = &weights[..];
    // dedup blobs by content, first-use order
    let mut streams: Vec<&[u16]> = Vec::new();
    let mut ords = Vec::with_capacity(layers);
    for w in weights {
        match streams.iter().position(|s| *s == *w) {
            Some(o) => ords.push(o),
            None => {
                ords.push(streams.len());
                streams.push(w);
            }
        }
    }
    if streams.len() > KNAME_SM_DW.len() {
        return Err(format!(
            "{} distinct softmax weight streams exceeds the mined kernel-name table ({})",
            streams.len(),
            KNAME_SM_DW.len()
        ));
    }
    let n_streams = streams.len();
    // stream byte offset in __KERN_0: stream 0 owns the LUT tail
    let stream_off = |o: usize| if o == 0 { 0 } else { 0x100 + 0x2000 * o };
    let mut first_use = vec![usize::MAX; n_streams];
    let mut last_use = vec![0usize; n_streams];
    for (i, &o) in ords.iter().enumerate() {
        first_use[o] = first_use[o].min(i);
        last_use[o] = i;
    }

    let skel3 = SKELETON_CONV_SM3;
    let (sm_entries, sm_relocs) = tail_meta(SKELETON_CONV64_SM);
    let rflag: [u8; 4] = sm_relocs[0][4..8].try_into().unwrap();

    // symtab — [x, z] + per-stream header group (stream 0's ne_0
    // followed by the two `_actlut` entries) + per-stream lane groups
    // (the bare LUT-kernel symbols trail stream 0's lanes) + the
    // scalar type-descriptor tail
    let entries = if n_streams == 1 {
        sm_entries.clone()
    } else {
        let mut e = Vec::with_capacity(sm_entries.len() + 16 * (n_streams - 1));
        e.push(sm_entries[0].clone()); // x
        e.push(sm_entries[1].clone()); // z
        let tpl0 = sm_entries[2].clone();
        let tpln = sm_entries[5].clone();
        e.push((
            format!("{}_ne_0", KNAME_SM_DW[0]),
            tpl0.1,
            tpl0.2,
            tpl0.3,
            tpl0.4,
        ));
        e.push(sm_entries[3].clone()); // exp _actlut
        e.push(sm_entries[4].clone()); // recip _actlut
        for j in 1..n_streams {
            e.push((
                format!("{}_ne_0", KNAME_SM_DW[j]),
                tpl0.1,
                tpl0.2,
                tpl0.3,
                tpl0.4 + stream_off(j) as u64,
            ));
        }
        for j in 0..n_streams {
            for b in 1..16u64 {
                e.push((
                    format!("{}_ne_{b}", KNAME_SM_DW[j]),
                    tpln.1,
                    tpln.2,
                    tpln.3,
                    0x3001_0000 + stream_off(j) as u64 + 0x200 * b,
                ));
            }
            if j == 0 {
                e.push(sm_entries[20].clone()); // exp LUT kernel
                e.push(sm_entries[21].clone()); // recip LUT kernel
            }
        }
        e.extend_from_slice(&sm_entries[22..]); // scalar descriptors
        e
    };

    // record stream: prologue + per-layer [conv][sm-reduce][sm-final]
    let mut stream = skel3[SM_PROG.0..SM_PROG.0 + SM_PROG.1].to_vec();
    let mut reloc_offs = Vec::with_capacity(3 * layers);
    for i in 0..layers {
        // conv task — first-layer form carries the input-DMA prologue
        let co = stream.len() - SM_PROG.1 + 4;
        let (stream_u32_off, t) = if i == 0 {
            (0x14c, skel3[SM_CONV_FIRST.0..SM_CONV_FIRST.0 + SM_CONV_FIRST.1].to_vec())
        } else {
            let mut t =
                skel3[SM_CONV_INT.0..SM_CONV_INT.0 + SM_CONV_INT.1].to_vec();
            patch_task_column(&mut t, (6 * i) as u16, 16);
            put_u16(&mut t, 0x13c, (6 * i + 1) as u16);
            put_u16(&mut t, 0x1bc, (6 * i + 2) as u16);
            put_u16(&mut t, 0x24c, (6 * i + 3) as u16);
            (0x128, t)
        };
        let mut t = t;
        // last-use release flag + deduplicated kernel id + stream
        // offset (the reloc'd u32 after the `44 95 80 23` marker)
        t[0x05] = if last_use[ords[i]] == i { 0x02 } else { 0x00 };
        put_u16(&mut t, 0x06, (6 * first_use[ords[i]] + 1) as u16);
        t[stream_u32_off..stream_u32_off + 4]
            .copy_from_slice(&(stream_off(ords[i]) as u32).to_le_bytes());
        stream.extend_from_slice(&t);
        reloc_offs.push(co + stream_u32_off);

        // softmax reduce task
        let co = stream.len() - SM_PROG.1 + 4;
        let mut t = skel3[SM_SMA.0..SM_SMA.0 + SM_SMA.1].to_vec();
        put_u16(&mut t, 0x06, (4 + 6 * i) as u16);
        patch_task_column(&mut t, (3 + 6 * i) as u16, 17);
        t[0x95] = if i == 0 { 0x04 } else { 0x08 };
        put_u16(&mut t, 0xcc, (4 + 6 * i) as u16);
        stream.extend_from_slice(&t);
        reloc_offs.push(co + 0xb8);

        // softmax-final task — interior or terminal (writeback) form
        let co = stream.len() - SM_PROG.1 + 4;
        let last = i + 1 == layers;
        let (off, len) = if last { SM_SMB_TERM } else { SM_SMB_INT };
        let mut t = skel3[off..off + len].to_vec();
        put_u16(&mut t, 0x06, (5 + 6 * i) as u16);
        patch_task_column(&mut t, (4 + 6 * i) as u16, 17);
        put_u16(&mut t, 0xcc, (5 + 6 * i) as u16);
        if last {
            t[0xa1] = 0x04;
            // `31 00 <id> 01` epilogue slots — single bytes
            t[0x16e] = (5 + 6 * i) as u8;
            t[0x18e] = (5 + 6 * i) as u8;
        } else {
            t[0xa1] = 0x08;
            put_u16(&mut t, 0x14c, (6 * (i + 1)) as u16);
        }
        stream.extend_from_slice(&t);
        reloc_offs.push(co + 0xb8);
    }

    let relocs: Vec<[u8; 8]> = reloc_offs
        .iter()
        .map(|&o| {
            let mut e = [0u8; 8];
            // r_address is section-relative: prologue (0x30) + stream
            e[..4].copy_from_slice(&(o as u32 + 0x30).to_le_bytes());
            e[4..].copy_from_slice(&rflag);
            e
        })
        .collect();

    let (mut img, text_off) = build_tail_at(
        skel3,
        CONV_TAIL_OFF,
        &entries,
        &relocs,
        mil_path,
        out_path,
    );

    let section_size = stream.len();
    img.extend_from_slice(&stream);

    // __KERN_0 lands one 0x8000 page past __text regardless of
    // stream count: stream 0 owns offsets 0..0x2100 (weights + the
    // shared exp/recip LUT tail), streams j>0 pack contiguously
    let kern_off = text_off + 0x8000;
    let kern_payload = 0x100 + 0x2000 * n_streams;
    let kern_vsz = (kern_payload + 0x3fff) & !0x3fff;
    img.resize(kern_off + kern_vsz, 0);
    for (j, w) in streams.iter().enumerate() {
        let inter = weight_interleave(w, C);
        for b in 0..16usize {
            let base = kern_off + stream_off(j) + b * 0x200;
            for (i, v) in inter[b * 256..b * 256 + 256].iter().enumerate() {
                img[base + i * 2..base + i * 2 + 2]
                    .copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    // exp/reciprocal LUTs from the k=1 skeleton's aux tail
    // (kern fileoff 0xc000 + 0x2000)
    img[kern_off + 0x2000..kern_off + 0x2100]
        .copy_from_slice(&SKELETON_CONV64_SM[0xe000..0xe100][..]);

    // layer-count scalars for the multi-task family
    let aligned = (section_size + 0x3f) / 0x40 * 0x40;
    put_u32(&mut img, 0x1c0, text_off as u32);
    put_u32(&mut img, 0x208, section_size as u32);
    put_u32(&mut img, 0x210, text_off as u32);
    put_u32(&mut img, 0x21c, (3 * layers) as u32);
    img[0x250..0x252].copy_from_slice(&(0x8000u16 + aligned as u16).to_le_bytes());
    put_u32(&mut img, 0x260, (text_off + aligned) as u32);
    put_u32(&mut img, 0x2a0, kern_vsz as u32);
    put_u32(&mut img, 0x2a8, kern_off as u32);
    put_u32(&mut img, 0x2b0, kern_vsz as u32);
    img[0x2f0..0x2f8].copy_from_slice(&(kern_payload as u64).to_le_bytes());
    put_u32(&mut img, 0x2f8, kern_off as u32);
    img[0x378..0x37a].copy_from_slice(&(0x8000u16 + aligned as u16).to_le_bytes());
    img[0xb7c..0xb7e]
        .copy_from_slice(&((0x151 + 0x134 * (layers - 1)) as u16).to_le_bytes());
    img[0xb88..0xb8c].copy_from_slice(&((6 * layers) as u32).to_le_bytes());
    img[0xbb8..0xbba]
        .copy_from_slice(&((0x57 + 0x54 * (layers - 1)) as u16).to_le_bytes());
    Ok(img)
}

/// Layer-norm skeletons — `conv_ln1` k=1 task (`0x37c`), `conv_ln3`
/// terminal-even task, `conv_ln4` first/interior/terminal task
/// templates. `layer_norm` fuses into the conv task itself (no
/// separate reduce tasks, no LUT stream): each layer is one `0x330`
/// task carrying the conv+LN descriptor program, allocating five
/// task ids.
const SKELETON_CONV_LN1: &[u8] = include_bytes!("../assets/conv_ln1_skeleton.hwx");
const SKELETON_CONV_LN2: &[u8] = include_bytes!("../assets/conv_ln2_reference.hwx");
const SKELETON_CONV_LN3: &[u8] = include_bytes!("../assets/conv_ln3_skeleton.hwx");
const SKELETON_CONV_LN4: &[u8] = include_bytes!("../assets/conv_ln4_skeleton.hwx");

/// Distinct-weight layer-norm kernel names by stream ordinal (the
/// single-stream program always uses `K9419..`). Mined from ANEC
/// lndw2/3/4 probes; ordinals beyond 3 need further mining.
const KNAME_LN_DW: [&str; 4] = [
    "K3C23BA63F9FE7926259BE0CD49C610C288A7DB57FE6AFF2F2F0543F4EF3B5712",
    "KE3545219F139ACC28DC984868084DC69371098AC16C6143198F730D551E6D04F",
    "K49750E73342DC5A9AF618282C1FC3E0663E255CD326287258C267314D841D492",
    "KC2200A5DA40187D6913B22C3566BA0458BC425B33670C299DF9AB485B31C198E",
];

/// Program header + record prologue (`0x30` header, `01 00 05 00`).
const LN_PROG: (usize, usize) = (0x4000, 0x34);
/// k=1 whole task — `0x37c`, terminal+first combined.
const LN_K1: (usize, usize) = (0x4034, 0x37c);
/// Layer-0 task — `0x370` with the input-DMA insert at +0x130.
/// k=2 chains run the whole program bank-swapped, so L0 uses the
/// odd-parity variant captured from the ln2 reference.
const LN_FIRST: (usize, usize) = (0x4034, 0x370);
/// Layer-0 task, bank-swapped (k=2 programs only).
const LN_FIRST_ODD: (usize, usize) = (0x4034, 0x370);
/// Interior task at odd layer index — `0x330`.
const LN_INT_ODD: (usize, usize) = (0x43a4, 0x330);
/// Interior task at even layer index — `0x330` (bank-select bytes
/// alternate by layer parity).
const LN_INT_EVEN: (usize, usize) = (0x46d4, 0x330);
/// Terminal task at odd layer index — `0x350` (output writeback).
const LN_TERM_ODD: (usize, usize) = (0x4a04, 0x350);
/// Terminal task at even layer index — `0x350`.
const LN_TERM_EVEN: (usize, usize) = (0x46d4, 0x350);

/// Emit a complete HWX image for `(conv1x1 -> layer_norm)^k` sharing
/// one `C=64` weight blob. `layer_norm` fuses into the conv task —
/// one task per layer allocating five ids (`5i+1..5i+5`), so the
/// wiring grammar is a single-task variant of the softmax multi-task
/// family: 16-slot producer column = `5i`, own ids `5i+1..5i+4` at
/// +0x12c/+0x19c/+0x21c/+0x29c, next-layer input `5i+5` at +0x30c
/// (interior) or output-writeback refs `5i+4` at +0x32e/+0x34e
/// (terminal). Bank-select bytes (+0x188/+0x190/...) and the `04/05`
/// descriptor variants (+0x191/+0x1fd/+0x209/+0x281) alternate by
/// layer parity — SRAM ping-pong — captured as even/odd templates.
///
/// `__KERN_0` = `n` contiguous `0x2000` weight streams (no LUT tail);
/// relocation per layer targets the stream-offset u32 after the
/// task's `44 95 80 23` marker (+0x14c first, +0x124 otherwise).
pub fn emit_conv_ln_n(
    layers: usize,
    weights: &[u16],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    emit_conv_ln_n_dw(layers, &vec![weights; layers], mil_path, out_path)
}

/// `emit_conv_ln_n` with per-layer weight blobs — same dedup/stream
/// model as the other dw emitters: conv task id `5*first_use+1`
/// (+0x06), last-use release flag (+0x05), stream offset u32
/// `ord*0x2000`, per-stream `K<hash>_ne_*` symbol groups.
pub fn emit_conv_ln_n_dw(
    layers: usize,
    weights: &[&[u16]],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    const C: usize = 64;
    if layers == 0 || weights.len() != layers {
        return Err(format!("expected {layers} weight blobs, got {}", weights.len()));
    }
    let weights: Vec<&[u16]> = weights
        .iter()
        .map(|w| -> Result<&[u16], String> {
            if w.len() < C * C {
                return Err(format!("expected {} weights, got {}", C * C, w.len()));
            }
            Ok(&w[..C * C])
        })
        .collect::<Result<_, _>>()?;
    let weights = &weights[..];
    let mut streams: Vec<&[u16]> = Vec::new();
    let mut ords = Vec::with_capacity(layers);
    for w in weights {
        match streams.iter().position(|s| *s == *w) {
            Some(o) => ords.push(o),
            None => {
                ords.push(streams.len());
                streams.push(w);
            }
        }
    }
    if streams.len() > KNAME_LN_DW.len() {
        return Err(format!(
            "{} distinct layernorm weight streams exceeds the mined kernel-name table ({})",
            streams.len(),
            KNAME_LN_DW.len()
        ));
    }
    let n_streams = streams.len();
    let mut first_use = vec![usize::MAX; n_streams];
    let mut last_use = vec![0usize; n_streams];
    for (i, &o) in ords.iter().enumerate() {
        first_use[o] = first_use[o].min(i);
        last_use[o] = i;
    }

    let skel = SKELETON_CONV_LN4;
    let (ln_entries, ln_relocs) = tail_meta(SKELETON_CONV_LN1);
    let rflag: [u8; 4] = ln_relocs[0][4..8].try_into().unwrap();

    // symtab — shared uses the k=1 skeleton's entries verbatim; dw is
    // [x, z] + all streams' `_ne_0` (desc 2) + per-stream lanes 1..15
    // (desc 0) at `0x30010000 + ord*0x2000 + 0x200*b` + the scalar tail.
    let entries = if n_streams == 1 {
        ln_entries.clone()
    } else {
        let mut e = Vec::with_capacity(ln_entries.len() + 16 * (n_streams - 1));
        e.push(ln_entries[0].clone()); // x
        e.push(ln_entries[1].clone()); // z
        let tpl0 = ln_entries[2].clone();
        let tpln = ln_entries[3].clone();
        for j in 0..n_streams {
            e.push((
                format!("{}_ne_0", KNAME_LN_DW[j]),
                tpl0.1,
                tpl0.2,
                tpl0.3,
                0x3001_0000 + 0x2000 * j as u64,
            ));
        }
        for j in 0..n_streams {
            for b in 1..16u64 {
                e.push((
                    format!("{}_ne_{b}", KNAME_LN_DW[j]),
                    tpln.1,
                    tpln.2,
                    tpln.3,
                    0x3001_0000 + 0x2000 * j as u64 + 0x200 * b,
                ));
            }
        }
        e.extend_from_slice(&ln_entries[18..]); // scalar descriptors
        e
    };

    // record stream: prologue + one fused conv+LN task per layer;
    // prologue +0x12 is `0x62` for the single-task program, `0x63`
    // for multi-task chains
    let mut stream = skel[LN_PROG.0..LN_PROG.0 + LN_PROG.1].to_vec();
    stream[0x12] = if layers == 1 { 0x62 } else { 0x63 };
    let mut reloc_offs = Vec::with_capacity(layers);
    for i in 0..layers {
        let co = stream.len() - LN_PROG.1 + 4;
        let last = i + 1 == layers;
        // k=2 programs run the whole bank alternation swapped
        let even = (i % 2 == 0) != (layers == 2);
        let (mark_u32_off, src, len) = if layers == 1 {
            (0x148usize, SKELETON_CONV_LN1, LN_K1)
        } else if i == 0 {
            if layers == 2 {
                (0x14cusize, SKELETON_CONV_LN2, LN_FIRST_ODD)
            } else {
                (0x14cusize, skel, LN_FIRST)
            }
        } else if last {
            if even {
                (0x124usize, SKELETON_CONV_LN3, LN_TERM_EVEN)
            } else {
                (0x124usize, skel, LN_TERM_ODD)
            }
        } else if even {
            (0x124usize, skel, LN_INT_EVEN)
        } else {
            (0x124usize, skel, LN_INT_ODD)
        };
        let mut t = src[len.0..len.0 + len.1].to_vec();
        t[0x05] = if last_use[ords[i]] == i { 0x02 } else { 0x00 };
        put_u16(&mut t, 0x06, (5 * first_use[ords[i]] + 1) as u16);
        if i > 0 && layers > 1 {
            patch_task_column(&mut t, (5 * i) as u16, 16);
            put_u16(&mut t, 0x12c, (5 * i + 1) as u16);
            put_u16(&mut t, 0x19c, (5 * i + 2) as u16);
            put_u16(&mut t, 0x21c, (5 * i + 3) as u16);
            put_u16(&mut t, 0x29c, (5 * i + 4) as u16);
            // buffer-slot counters: family A = 4+(i-1)/2, family B
            // = 4+i/2 interior / 4+(i+2)/3 in the terminal task
            let ca = (4 + (i - 1) / 2) as u16;
            for &o in &[0x111usize, 0x189, 0x201, 0x27d, 0x28d] {
                put_u16(&mut t, o, ca);
            }
            if last {
                let cb = (4 + (i + 2) / 3) as u16;
                for &o in &[0x191usize, 0x1fd, 0x209, 0x281, 0x311] {
                    put_u16(&mut t, o, cb);
                }
                put_u16(&mut t, 0x32e, (5 * i + 4) as u16);
                // `xx 01` epilogue slot — single byte + trailing flag
                t[0x34e] = (5 * i + 4) as u8;
                put_u16(&mut t, 0x305, ca);
            } else {
                let cb = (4 + i / 2) as u16;
                for &o in &[0x191usize, 0x1fd, 0x209, 0x281] {
                    put_u16(&mut t, o, cb);
                }
                put_u16(&mut t, 0x2fd, ca);
                put_u16(&mut t, 0x30c, (5 * i + 5) as u16);
            }
        }
        t[mark_u32_off..mark_u32_off + 4]
            .copy_from_slice(&(0x2000 * ords[i] as u32).to_le_bytes());
        stream.extend_from_slice(&t);
        reloc_offs.push(co + mark_u32_off);
    }

    let relocs: Vec<[u8; 8]> = reloc_offs
        .iter()
        .map(|&o| {
            let mut e = [0u8; 8];
            e[..4].copy_from_slice(&(o as u32 + 0x30).to_le_bytes());
            e[4..].copy_from_slice(&rflag);
            e
        })
        .collect();

    let (mut img, text_off) = build_tail_at(
        skel,
        CONV_TAIL_OFF,
        &entries,
        &relocs,
        mil_path,
        out_path,
    );

    let section_size = stream.len();
    img.extend_from_slice(&stream);

    // __KERN_0 one 0x8000 page past __text; n contiguous 0x2000
    // weight streams, no LUT tail
    let kern_off = text_off + 0x8000;
    let kern_payload = 0x2000 * n_streams;
    let kern_vsz = (kern_payload + 0x3fff) & !0x3fff;
    img.resize(kern_off + kern_vsz, 0);
    for (j, w) in streams.iter().enumerate() {
        let inter = weight_interleave(w, C);
        for b in 0..16usize {
            let base = kern_off + 0x2000 * j + b * 0x200;
            for (i, v) in inter[b * 256..b * 256 + 256].iter().enumerate() {
                img[base + i * 2..base + i * 2 + 2]
                    .copy_from_slice(&v.to_le_bytes());
            }
        }
    }

    // layer-count scalars — the record-end fields use the
    // 0x40-aligned text size like the other families
    let aligned = (section_size + 0x3f) / 0x40 * 0x40;
    put_u32(&mut img, 0x1c0, text_off as u32);
    put_u32(&mut img, 0x208, section_size as u32);
    put_u32(&mut img, 0x210, text_off as u32);
    put_u32(&mut img, 0x21c, layers as u32);
    put_u32(&mut img, 0x250, 0x3000_8000 + aligned as u32);
    put_u32(&mut img, 0x260, text_off as u32 + aligned as u32);
    put_u32(&mut img, 0x2a0, kern_vsz as u32);
    put_u32(&mut img, 0x2a8, kern_off as u32);
    put_u32(&mut img, 0x2b0, kern_vsz as u32);
    img[0x2f0..0x2f8].copy_from_slice(&(kern_payload as u64).to_le_bytes());
    put_u32(&mut img, 0x2f8, kern_off as u32);
    put_u32(&mut img, 0x378, 0x3000_8000 + aligned as u32);
    img[0xb7c..0xb7e].copy_from_slice(
        &((0xec + 0xcc * (layers - 1) + if layers > 1 { 5 } else { 0 }) as u16)
            .to_le_bytes(),
    );
    img[0xb88..0xb8c].copy_from_slice(&((5 * layers) as u32).to_le_bytes());
    img[0xbb8..0xbba]
        .copy_from_slice(&((0x49 + 0x46 * (layers - 1)) as u16).to_le_bytes());
    Ok(img)
}

/// Rms-norm skeletons — `conv_rms1` k=1 task set, `conv_rms2` aux3
/// i=1-terminal, `conv_rms4` aux1 odd-terminal, `conv_rms5` main
/// template source (interior/even-terminal forms + tail + symtab).
/// `rms_norm` decomposes into four task descriptors per layer: a
/// conv task (managing ids `12i+1..12i+4`) and three aux tasks
/// binding the shared div/recip LUT streams.
const SKELETON_CONV_RMS1: &[u8] = include_bytes!("../assets/conv_rms1_skeleton.hwx");
const SKELETON_CONV_RMS2: &[u8] = include_bytes!("../assets/conv_rms2_skeleton.hwx");
const SKELETON_CONV_RMS4: &[u8] = include_bytes!("../assets/conv_rms4_skeleton.hwx");
const SKELETON_CONV_RMS5: &[u8] = include_bytes!("../assets/conv_rms5_skeleton.hwx");

/// Program header + record prologue.
const RMS_PROG: (usize, usize) = (0x4000, 0x34);
/// Layer-0 conv task — `0x340` with the input-DMA insert.
const RMS_CONV_FIRST: (usize, usize) = (0x4034, 0x340);
/// Interior conv task at odd layer index — `0x2d0`.
const RMS_CONV_INT: (usize, usize) = (0x48b4, 0x2d0);
/// Aux1 interior form A — `0x1f0` (odd layers and i<2).
const RMS_A1_F1: (usize, usize) = (0x4b84, 0x1f0);
/// Aux1 terminal form A — `0x1f0` (rms4 layer-3 terminal).
const RMS_A1_F1_TERM: (usize, usize) = (0x5b94, 0x1f0);
/// Aux1 interior form B — `0x1e0` (even layers i>=2; aux2-style
/// field layout).
const RMS_A1_F2: (usize, usize) = (0x5394, 0x1e0);
/// Aux1 terminal form B — `0x1e0` (rms5 layer-4 terminal).
const RMS_A1_F2_TERM: (usize, usize) = (0x63a4, 0x1e0);
/// Aux2 task — `0x1e0`, single form at all positions.
const RMS_A2: (usize, usize) = (0x4d74, 0x1e0);
/// Aux3 interior — `0x170`.
const RMS_A3_INT: (usize, usize) = (0x4f54, 0x170);
/// Aux3 terminal at i=1 — `0x17c` (rms2).
const RMS_A3_T1: (usize, usize) = (0x4f54, 0x17c);
/// Aux3 terminal at i>=2 — `0x18c` (rms5).
const RMS_A3_T2: (usize, usize) = (0x6764, 0x18c);
/// k=1 tasks — conv `0x210`, mini-aux `0xf0`, two `0x1e0` aux tasks,
/// `0x18c` output task. Tight id packing (1..10) and a third LUT
/// block in the kern tail make this a separate family.
const RMS_K1: [(usize, usize); 5] = [
    (0x4034, 0x210),
    (0x4244, 0xf0),
    (0x4334, 0x1e0),
    (0x4514, 0x1e0),
    (0x46f4, 0x18c),
];
/// k=1 task reloc field offsets (mark+4 per task).
const RMS_K1_RELOC: [usize; 5] = [0x14c, 0xc0, 0xb4, 0xac, 0xb8];

/// Emit a complete HWX image for `(conv1x1 -> rms_norm)^k` sharing
/// one `C=64` weight blob. The `rms_norm` decomp emits four tasks
/// per layer on the same task-DAG grammar as softmax but with a
/// 12-task-id stride: conv (`12i+1..12i+4`), aux1 (`12i+5..12i+7`,
/// producer `12i+4`), aux2 (`12i+8..12i+10`, producer `12i+7`),
/// aux3 (`12i+11`, producer `12i+10`, emits next-input `12(i+1)`).
///
/// Aux1 alternates two layouts: `0x1f0` (odd i and i<2) with id
/// slots at +0xcc/+0x14c/+0x1cc, `0x1e0` (even i>=2) with slots at
/// +0xbc/+0x13c/+0x1bc. Aux3 grows a writeback epilogue terminal
/// (`0x17c` at i=1, `0x18c` at i>=2). Each task carries one
/// relocation at mark+4: conv binds the weight stream, aux1/aux2
/// the div LUT (`+0x2000`), aux3 the rsqrt LUT (`+0x2080`).
///
/// `__KERN_0` = `0x100 + n_streams*0x2000`: stream 0 owns
/// `0x0..0x2100` (weights + the shared LUT tail), streams j>0 at
/// `0x100 + j*0x2000`. `__text` floats to `0x8000` multi-stream;
/// `__KERN_0` lands at `text_off + 0x4000` for chains (`+0x8000`
/// k=1, whose payload is `0x2180` with a third LUT block).
pub fn emit_conv_rms_n(
    layers: usize,
    weights: &[u16],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    emit_conv_rms_n_dw(layers, &vec![weights; layers], mil_path, out_path)
}

/// `emit_conv_rms_n` with per-layer weight blobs — conv tasks carry
/// the stream's last-use release flag (+0x05), deduplicated kernel
/// id `12*first_use(ord)+1` (+0x06), and the stream offset u32.
pub fn emit_conv_rms_n_dw(
    layers: usize,
    weights: &[&[u16]],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    const C: usize = 64;
    if layers == 0 || weights.len() != layers {
        return Err(format!("expected {layers} weight blobs, got {}", weights.len()));
    }
    let weights: Vec<&[u16]> = weights
        .iter()
        .map(|w| -> Result<&[u16], String> {
            if w.len() < C * C {
                return Err(format!("expected {} weights, got {}", C * C, w.len()));
            }
            Ok(&w[..C * C])
        })
        .collect::<Result<_, _>>()?;
    let weights = &weights[..];
    let mut streams: Vec<&[u16]> = Vec::new();
    let mut ords = Vec::with_capacity(layers);
    for w in weights {
        match streams.iter().position(|s| *s == *w) {
            Some(o) => ords.push(o),
            None => {
                ords.push(streams.len());
                streams.push(w);
            }
        }
    }
    if streams.len() > KNAME_SM_DW.len() {
        return Err(format!(
            "{} distinct weight streams exceeds the mined kernel-name table ({})",
            streams.len(),
            KNAME_SM_DW.len()
        ));
    }
    let n_streams = streams.len();
    let stream_off = |o: usize| if o == 0 { 0 } else { 0x100 + 0x2000 * o };
    let mut first_use = vec![usize::MAX; n_streams];
    let mut last_use = vec![0usize; n_streams];
    for (i, &o) in ords.iter().enumerate() {
        first_use[o] = first_use[o].min(i);
        last_use[o] = i;
    }

    // k=1 is a distinct compact family — five tasks, tight ids, an
    // extra LUT block, a 0x4000 second-__TEXT and different header
    // scalars; use its own skeleton for the header/tail template.
    let skel = if layers == 1 {
        SKELETON_CONV_RMS1
    } else {
        SKELETON_CONV_RMS5
    };
    let (rms_entries, rms_relocs) = tail_meta(skel);
    let rflag: [u8; 4] = rms_relocs[0][4..8].try_into().unwrap();

    let mut stream = skel[RMS_PROG.0..RMS_PROG.0 + RMS_PROG.1].to_vec();
    let mut reloc_offs = Vec::with_capacity(4 * layers + 1);

    if layers == 1 {
        // compact k=1 family — five tasks verbatim, tight ids
        for (t, &(off, len)) in RMS_K1.iter().enumerate() {
            let co = stream.len() - RMS_PROG.1 + 4;
            let mut task = skel[off..off + len].to_vec();
            if t == 0 {
                task[0x05] = 0x02;
                put_u16(&mut task, 0x06, 1);
                task[0x14c..0x150].copy_from_slice(&0u32.to_le_bytes());
            }
            stream.extend_from_slice(&task);
            reloc_offs.push(co + RMS_K1_RELOC[t]);
        }
    } else {
        for i in 0..layers {
            let last = i + 1 == layers;
            // conv task — first-layer form carries the input-DMA insert
            let co = stream.len() - RMS_PROG.1 + 4;
            let (ru32, mut t) = if i == 0 {
                (
                    0x14cusize,
                    skel[RMS_CONV_FIRST.0..RMS_CONV_FIRST.0 + RMS_CONV_FIRST.1].to_vec(),
                )
            } else {
                let mut t = skel[RMS_CONV_INT.0..RMS_CONV_INT.0 + RMS_CONV_INT.1].to_vec();
                patch_task_column(&mut t, (12 * i) as u16, 16);
                put_u16(&mut t, 0x12c, (12 * i + 1) as u16);
                put_u16(&mut t, 0x1ac, (12 * i + 2) as u16);
                put_u16(&mut t, 0x22c, (12 * i + 3) as u16);
                put_u16(&mut t, 0x2ac, (12 * i + 4) as u16);
                // bank-descriptor counters alternate by layer parity
                if i % 2 == 0 {
                    put_u16(&mut t, 0x278, 0x04);
                    put_u16(&mut t, 0x2ae, 0x38);
                } else {
                    put_u16(&mut t, 0x278, 0x44);
                    put_u16(&mut t, 0x2ae, 0x39);
                }
                (0x124, t)
            };
            t[0x05] = if last_use[ords[i]] == i { 0x02 } else { 0x00 };
            put_u16(&mut t, 0x06, (12 * first_use[ords[i]] + 1) as u16);
            t[ru32..ru32 + 4].copy_from_slice(&(stream_off(ords[i]) as u32).to_le_bytes());
            stream.extend_from_slice(&t);
            reloc_offs.push(co + ru32);

            // aux1 — two layouts: 0x1f0 at odd i and i<2, 0x1e0 at
            // even i>=2; terminal forms swap the writeback block
            let co = stream.len() - RMS_PROG.1 + 4;
            let f2 = i >= 2 && i % 2 == 0;
            let ((off, len), isrc) = match (f2, last) {
                (true, true) => (RMS_A1_F2_TERM, skel),
                (true, false) => (RMS_A1_F2, skel),
                (false, true) => (RMS_A1_F1_TERM, SKELETON_CONV_RMS4),
                (false, false) => (RMS_A1_F1, skel),
            };
            let mut t = isrc[off..off + len].to_vec();
            patch_task_column(&mut t, (12 * i + 4) as u16, 17);
            put_u16(&mut t, 0x06, (12 * i + 5) as u16);
            if f2 {
                put_u16(&mut t, 0xbc, (12 * i + 5) as u16);
                put_u16(&mut t, 0x13c, (12 * i + 6) as u16);
                put_u16(&mut t, 0x1bc, (12 * i + 7) as u16);
                reloc_offs.push(co + 0xb4);
            } else {
                put_u16(&mut t, 0xcc, (12 * i + 5) as u16);
                put_u16(&mut t, 0x14c, (12 * i + 6) as u16);
                put_u16(&mut t, 0x1cc, (12 * i + 7) as u16);
                reloc_offs.push(co + 0xb8);
            }
            stream.extend_from_slice(&t);

            // aux2 — single 0x1e0 form everywhere
            let co = stream.len() - RMS_PROG.1 + 4;
            let mut t = skel[RMS_A2.0..RMS_A2.0 + RMS_A2.1].to_vec();
            patch_task_column(&mut t, (12 * i + 7) as u16, 17);
            put_u16(&mut t, 0x06, (12 * i + 8) as u16);
            put_u16(&mut t, 0xbc, (12 * i + 8) as u16);
            put_u16(&mut t, 0x13c, (12 * i + 9) as u16);
            put_u16(&mut t, 0x1bc, (12 * i + 10) as u16);
            if last && i == 1 {
                // layer-1 terminal aux2 — release descriptors
                t[0x188] = 0x04;
                t[0x1be] = 0x38;
            }
            stream.extend_from_slice(&t);
            reloc_offs.push(co + 0xac);

            // aux3 — interior or terminal (writeback) form
            let co = stream.len() - RMS_PROG.1 + 4;
            let (toff, tlen, isrc) = if last {
                if i == 1 {
                    (RMS_A3_T1.0, RMS_A3_T1.1, SKELETON_CONV_RMS2)
                } else {
                    (RMS_A3_T2.0, RMS_A3_T2.1, skel)
                }
            } else {
                (RMS_A3_INT.0, RMS_A3_INT.1, skel)
            };
            let mut t = isrc[toff..toff + tlen].to_vec();
            patch_task_column(&mut t, (12 * i + 10) as u16, 17);
            put_u16(&mut t, 0x06, (12 * i + 11) as u16);
            if !(last && i == 1) {
                // T1 terminal has a fixed `60 f8` field at +0xcc
                put_u16(&mut t, 0xcc, (12 * i + 11) as u16);
            }
            if last && i >= 2 {
                put_u16(&mut t, 0x16a, (12 * i + 11) as u16);
                // `31 00 <id> 01` writeback slot — the id is one byte
                t[0x18a] = (12 * i + 11) as u8;
            }
            if !last {
                put_u16(&mut t, 0x14c, (12 * (i + 1)) as u16);
            }
            stream.extend_from_slice(&t);
            reloc_offs.push(co + if last && i == 1 { 0xb4 } else { 0xb8 });
        }
    }

    let relocs: Vec<[u8; 8]> = reloc_offs
        .iter()
        .map(|&o| {
            let mut e = [0u8; 8];
            e[..4].copy_from_slice(&(o as u32 + 0x30).to_le_bytes());
            e[4..].copy_from_slice(&rflag);
            e
        })
        .collect();

    // The second `__TEXT` section (`0x40` bytes of `00 1c` + zeros on
    // chains, a `0x4000` zero block on k=1) starts at the `0x80`-
    // aligned text end; symtab[0] binds its vmaddr and `__KERN_0`'s
    // vmaddr is the next `0x4000` boundary past it.
    let section_size = stream.len();
    let aligned = (section_size + 0x7f) & !0x7f;
    let sec2_addr = 0x3000_8000u64 + aligned as u64;
    let sec2_size = if layers == 1 { 0x4000u32 } else { 0x40 };
    let kern_addr = (sec2_addr + sec2_size as u64 + 0x3fff) & !0x3fff;

    // symtab — [fl, x, z] + [s0_ne_0, div/recip _actlut, s1..sn_ne_0]
    // + per-stream lane groups (the bare div/recip LUT syms trail
    // stream 0's lanes) + the scalar tail. n_values are `__KERN_0`
    // vmaddrs: ne_0 at stream_off, lanes at +0x200·b.
    let entries = if n_streams == 1 {
        let mut v = rms_entries.clone();
        if layers != 1 {
            v[0].4 = sec2_addr;
            let d = kern_addr.wrapping_sub(0x3000_c000);
            for e in v[3..23].iter_mut() {
                e.4 = e.4.wrapping_add(d);
            }
            if layers >= 6 {
                // deep shared chains bind the multi-stream ord-0
                // kernel name instead of the shared K9419 — the
                // _actlut entries at [4]/[5] are untouched
                v[3].0 = format!("{}_ne_0", KNAME_SM_DW[0]);
                for b in 1..16 {
                    v[5 + b].0 = format!("{}_ne_{b}", KNAME_SM_DW[0]);
                }
            }
        }
        v
    } else {
        let mut e = Vec::with_capacity(rms_entries.len() + 16 * (n_streams - 1));
        e.push(rms_entries[0].clone()); // fl
        e.push(rms_entries[1].clone()); // x
        e.push(rms_entries[2].clone()); // z
        e[0].4 = sec2_addr;
        let tpl0 = rms_entries[3].clone();
        let tpln = rms_entries[6].clone();
        for j in 0..n_streams {
            e.push((
                format!("{}_ne_0", KNAME_SM_DW[j]),
                tpl0.1,
                tpl0.2,
                tpl0.3,
                kern_addr + stream_off(j) as u64,
            ));
            if j == 0 {
                e.push((rms_entries[4].0.clone(), tpl0.1, tpl0.2, tpl0.3, kern_addr + 0x2000));
                e.push((rms_entries[5].0.clone(), tpl0.1, tpl0.2, tpl0.3, kern_addr + 0x2080));
            }
        }
        for j in 0..n_streams {
            for b in 1..16u64 {
                e.push((
                    format!("{}_ne_{b}", KNAME_SM_DW[j]),
                    tpln.1,
                    tpln.2,
                    tpln.3,
                    kern_addr + stream_off(j) as u64 + 0x200 * b,
                ));
            }
            if j == 0 {
                e.push((rms_entries[21].0.clone(), tpln.1, tpln.2, tpln.3, kern_addr + 0x2080));
                e.push((rms_entries[22].0.clone(), tpln.1, tpln.2, tpln.3, kern_addr + 0x2100));
            }
        }
        e.extend_from_slice(&rms_entries[23..]); // scalar descriptors
        e
    };

    let (mut img, text_off) = build_tail_at(
        skel,
        CONV_TAIL_OFF,
        &entries,
        &relocs,
        mil_path,
        out_path,
    );

    img.extend_from_slice(&stream);

    // __KERN_0 = weight streams + the shared div/recip LUT tail on
    // stream 0. Chains place it one 0x4000 page past __text; k=1
    // carries a third LUT block (payload 0x2180) and sits +0x8000.
    let (kern_off, kern_payload) = if layers == 1 {
        (text_off + 0x8000, 0x2180usize)
    } else {
        (text_off + 0x4000, 0x100 + 0x2000 * n_streams)
    };
    let kern_vsz = (kern_payload + 0x3fff) & !0x3fff;
    img.resize(kern_off + kern_vsz, 0);
    if layers != 1 {
        img[text_off + aligned + 1] = 0x1c;
    }
    for (j, w) in streams.iter().enumerate() {
        let inter = weight_interleave(w, C);
        for b in 0..16usize {
            let base = kern_off + stream_off(j) + b * 0x200;
            for (i, v) in inter[b * 256..b * 256 + 256].iter().enumerate() {
                img[base + i * 2..base + i * 2 + 2]
                    .copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    // LUT tail — 0x180 k=1 (three blocks) / 0x100 chains (two)
    if layers == 1 {
        img[kern_off + 0x2000..kern_off + 0x2180]
            .copy_from_slice(&SKELETON_CONV_RMS1[0xe000..0xe180][..]);
    } else {
        img[kern_off + 0x2000..kern_off + 0x2100]
            .copy_from_slice(&skel[0xa000..0xa100][..]);
    }

    // scalars
    let ntasks = if layers == 1 { 5 } else { 4 * layers };
    put_u32(&mut img, 0x1c0, text_off as u32);
    put_u32(&mut img, 0x208, section_size as u32);
    put_u32(&mut img, 0x210, text_off as u32);
    put_u32(&mut img, 0x21c, ntasks as u32);
    put_u32(&mut img, 0x250, sec2_addr as u32);
    put_u32(&mut img, 0x258, sec2_size);
    put_u32(&mut img, 0x260, (text_off + aligned) as u32);
    put_u32(&mut img, 0x298, kern_addr as u32);
    put_u32(&mut img, 0x2a0, kern_vsz as u32);
    put_u32(&mut img, 0x2a8, kern_off as u32);
    put_u32(&mut img, 0x2b0, kern_vsz as u32);
    put_u32(&mut img, 0x2e8, kern_addr as u32);
    img[0x2f0..0x2f8].copy_from_slice(&(kern_payload as u64).to_le_bytes());
    put_u32(&mut img, 0x2f8, kern_off as u32);
    put_u32(&mut img, 0x378, sec2_addr as u32);
    put_u32(&mut img, 0x3d8, kern_addr as u32);
    put_u32(
        &mut img,
        0xb7c,
        0x220 + (section_size as u32 - 0x880) / 4,
    );
    put_u32(&mut img, 0xb88, if layers == 1 { 0xb } else { 12 * layers as u32 });
    put_u32(
        &mut img,
        0xbb8,
        if layers == 1 {
            0x9d
        } else {
            0x153 + 0xa8 * (layers as u32 - 2)
        },
    );
    Ok(img)
}

/// Matmul shape families with a captured skeleton.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmShape {
    /// `[1,32,1,32]` operands.
    N32,
    /// `[1,64,1,64]` operands.
    N64,
    /// `[1,128,1,128]` operands.
    N128,
    /// `[1,256,1,256]` operands.
    N256,
}

impl MmShape {
    /// Operand width `N`.
    pub fn n(self) -> usize {
        match self {
            MmShape::N32 => 32,
            MmShape::N64 => 64,
            MmShape::N128 => 128,
            MmShape::N256 => 256,
        }
    }

    fn skeleton(self) -> &'static [u8] {
        match self {
            MmShape::N32 => SKELETON_MM32,
            MmShape::N64 => SKELETON_MM64,
            MmShape::N128 => SKELETON_MM128,
            MmShape::N256 => SKELETON_MM256,
        }
    }

    /// Weight data offset inside `__TEXT`'s const pool.
    fn weights_off(self) -> usize {
        match self {
            MmShape::N256 => 0x4340,
            _ => 0x4200,
        }
    }
}

/// Standalone `relu` skeleton over fp16 `[1,64,1,1]` — no LUT (relu is
/// a fixed clamp, not a table op): `0x104` `__text` record at `0x4000`,
/// zero-filled const pool, tail rebuilt after `0x25f0`.
const SKELETON_RELU64: &[u8] = include_bytes!("../assets/relu64_skeleton.hwx");
const RELU_TAIL_OFF: usize = 0x25f0;

/// Emit `z = relu(x)` `[1,64,1,1]` byte-identically — the standalone
/// (non-fused) relu program.
pub fn emit_relu(mil_path: &str, out_path: &str) -> Vec<u8> {
    let skel = SKELETON_RELU64;
    let (entries, relocs) = tail_meta(skel);
    rebuild_tail_at(skel, RELU_TAIL_OFF, &entries, &relocs, mil_path, out_path)
}

/// Activation-matmul skeleton: `z = matmul(x, y, transpose_y)` over
/// fp16 `[1,64,1,64]` with both operands as program inputs — the
/// attention QK^T / PV form. The record is a fixed `0x1fc` task
/// descriptor at `__text` fileoff `0x8000`; there are no weight
/// streams, no `__KERN_0`, and no relocations. The variable front
/// matter (signature + `rt.version` + `LC_SYMTAB`) ends at `0x33f0`.
const SKELETON_MM_ACT64: &[u8] = include_bytes!("../assets/mm_act64_skeleton.hwx");

/// Emit a complete HWX image for `z = matmul(x, y, transpose_y)` with
/// both operands activations. Unlike [`emit_matmul`] there is no
/// weight stream: the `__text` record and all-zero const pools are
/// copied verbatim and only the signature block is rebuilt.
pub fn emit_mm_act(mil_path: &str, out_path: &str) -> Vec<u8> {
    emit_mm_part(MmPart::Act64, mil_path, out_path)
}

/// `matmul -> softmax` fused skeleton — the QK^T+softmax attention
/// partition over fp16 `[1,64,1,64]` activation operands. The `0x4f0`
/// `__text` record composes the matmul task with a softmax tail; the
/// softmax LUT lives in `__KERN_0` at `0x10000` and is bound by one
/// `__text` relocation whose `reloff` field sits inside the `__text`
/// section entry at `0x2b0`.
const SKELETON_MMSM64: &[u8] = include_bytes!("../assets/mmsm64_skeleton.hwx");
/// `softmax -> matmul` fused skeleton — the softmax+PV attention
/// partition. Same envelope as `SKELETON_MMSM64` with an extra
/// `__DATA/__bss` segment; sig at `0x3540`, `__text` section entry at
/// `0x310` (`reloff` field `0x348`), record `0x570`.
const SKELETON_SMMM64: &[u8] = include_bytes!("../assets/smmm64_skeleton.hwx");
/// `matmul -> mul -> add -> softmax` — the masked+scaled QK^T
/// partition over `[1,64,1,64]` (a constant scale `mul` folds into the
/// fused task; `mm -> add -> softmax` produces the identical record).
const SKELETON_MMADDSM: &[u8] = include_bytes!("../assets/mmaddsm_skeleton.hwx");
/// `matmul -> mul -> softmax` over `[1,64,1,64]` with a constant
/// tensor scale operand.
const SKELETON_MMMULSM: &[u8] = include_bytes!("../assets/mmmulsm_skeleton.hwx");
/// 4-D activation-matmul over `[1,8,64,64]` — heads×seq attention
/// shape. `0x384` record, no relocs.
const SKELETON_MM4D: &[u8] = include_bytes!("../assets/mm4d_skeleton.hwx");
/// `matmul -> softmax` over `[1,8,64,64]` — the 4-D QK^T+softmax
/// partition.
const SKELETON_MMSM4D: &[u8] = include_bytes!("../assets/mmsm4d_skeleton.hwx");

/// Attention-partition skeleton variants — the compositions ANEC
/// accepts around a single matmul (two chained matmuls are refused:
/// `CompilationFailure`). Each variant is a distinct envelope:
/// signature offset, `__text` section-entry `reloff` field, and
/// section layout all differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmPart {
    /// `matmul(x, y)` `[1,64,1,64]` — bare PV/QK partition.
    Act64,
    /// `matmul -> softmax` `[1,64,1,64]` — QK^T+softmax.
    MmSm64,
    /// `softmax -> matmul` `[1,64,1,64]`.
    SmMm64,
    /// `matmul -> [mul ->] add -> softmax` `[1,64,1,64]` — masked
    /// QK^T+softmax (constant scale folds into the fused task).
    MmAddSm64,
    /// `matmul -> mul -> softmax` `[1,64,1,64]` — scaled.
    MmMulSm64,
    /// `matmul(x, y)` `[1,8,64,64]` — heads×seq PV/QK partition.
    Act4d,
    /// `matmul -> softmax` `[1,8,64,64]` — heads×seq QK^T+softmax.
    MmSm4d,
}

impl MmPart {
    /// `(skeleton, sig tail offset, __text section reloff field)`.
    fn spec(self) -> (&'static [u8], usize, usize) {
        match self {
            MmPart::Act64 => (SKELETON_MM_ACT64, 0x33f0, 0),
            MmPart::MmSm64 => (SKELETON_MMSM64, 0x3488, 0x2b0),
            MmPart::SmMm64 => (SKELETON_SMMM64, 0x3540, 0x348),
            MmPart::MmAddSm64 => (SKELETON_MMADDSM, 0x4288, 0x348),
            MmPart::MmMulSm64 => (SKELETON_MMMULSM, 0x3488, 0x2b0),
            MmPart::Act4d => (SKELETON_MM4D, 0x34a8, 0x348),
            MmPart::MmSm4d => (SKELETON_MMSM4D, 0x3540, 0x348),
        }
    }
}

/// Emit an attention-partition HWX image byte-identically on its
/// skeleton family. All variants place `__text` at fileoff `0x8000`;
/// everything before the signature command is skeleton-verbatim.
pub fn emit_mm_part(part: MmPart, mil_path: &str, out_path: &str) -> Vec<u8> {
    let (skel, tail_off, reloff_field) = part.spec();
    let (entries, _) = tail_meta(skel);
    let relocs = if reloff_field != 0 {
        reloc_entries(skel, reloff_field)
    } else {
        Vec::new()
    };
    let (mut img, text_off) = build_tail_at_reloc(
        skel, tail_off, &entries, &relocs, mil_path, out_path, reloff_field,
    );
    debug_assert_eq!(text_off, 0x8000);
    img.extend_from_slice(&skel[0x8000..]);
    img
}

/// Emit `p = softmax(matmul(x, y^T))` — both operands activations —
/// byte-identically for the `[1,64,1,64]` skeleton family.
pub fn emit_mm_sm(mil_path: &str, out_path: &str) -> Vec<u8> {
    emit_mm_part(MmPart::MmSm64, mil_path, out_path)
}

/// Emit `z = matmul(softmax(x), y^T)` — both operands activations —
/// byte-identically for the `[1,64,1,64]` skeleton family.
pub fn emit_sm_mm(mil_path: &str, out_path: &str) -> Vec<u8> {
    emit_mm_part(MmPart::SmMm64, mil_path, out_path)
}

/// Emit a complete HWX image for `z = matmul(x, w, transpose_y)` on a
/// skeleton family. `weights` is the raw `[N,N]` fp16 matrix stored
/// verbatim in `__TEXT`'s const pool (no interleave for matmul).
pub fn emit_matmul(
    shape: MmShape,
    weights: &[u16],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    let n = shape.n();
    if weights.len() < n * n {
        return Err(format!("expected {} weights, got {}", n * n, weights.len()));
    }
    let weights = &weights[..n * n];
    let skel = shape.skeleton();
    let (entries, relocs) = tail_meta(skel);
    let mut img = rebuild_tail_at(skel, MM_TAIL_OFF, &entries, &relocs, mil_path, out_path);

    let off = shape.weights_off();
    for (i, v) in weights.iter().enumerate() {
        img[off + i * 2..off + i * 2 + 2].copy_from_slice(&v.to_le_bytes());
    }
    Ok(img)
}

// ---------------------------------------------------------------------------
// MIL-driven dispatch — `milc compile --target ane --emit-hwx`.
// ---------------------------------------------------------------------------

/// A compute op extracted from MIL text: op type, result shape, and
/// the operand names we inspect (`x`/`y`/`weight`).
struct MilOp {
    op: String,
    /// Result tensor name (the lhs identifier).
    result: String,
    shape: Vec<i64>,
    args: Vec<(String, String)>,
}

/// Scan MIL text for compute ops (non-`const` declarations of the
/// form `tensor<dtype, [shape]> name = op(arg = operand, ...)`) and
/// blob-const bindings (`name = const()[..BLOBFILE(path = string("P"))]`).
fn mil_ops(text: &str) -> (Vec<MilOp>, Vec<(String, String)>) {
    let mut ops = Vec::new();
    let mut blobs = Vec::new();
    // Op declarations may wrap after `=` — join continuation lines so
    // `name =\n    op(...)` parses as one statement.
    let joined = {
        let mut out = String::with_capacity(text.len());
        for raw in text.lines() {
            let line = raw.trim_end();
            if out.ends_with(" =") && line.trim_start().starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
                out.push(' ');
                out.push_str(line.trim_start());
            } else {
                out.push('\n');
                out.push_str(line);
            }
        }
        out
    };
    for raw in joined.lines() {
        let line = raw.trim();
        let Some(eq) = line.find(" = ") else { continue };
        let (lhs, rhs) = line.split_at(eq);
        let rhs = &rhs[3..];
        const BLOB_PFX: &str = "BLOBFILE(path = string(\"";
        if let Some(blob_pos) = rhs.find(BLOB_PFX) {
            if let Some(name) = lhs.rsplit(' ').next() {
                let rest = &rhs[blob_pos + BLOB_PFX.len()..];
                if let Some(end) = rest.find('"') {
                    blobs.push((name.to_string(), rest[..end].to_string()));
                }
            }
            continue;
        }
        let Some(paren) = rhs.find('(') else { continue };
        let op = rhs[..paren].trim();
        if !op.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            || op == "const"
            || !line.contains("tensor<")
        {
            continue;
        }
        let shape = lhs
            .find('[')
            .and_then(|l| lhs.find(']').map(|r| (l, r)))
            .map(|(l, r)| {
                lhs[l + 1..r]
                    .split(',')
                    .filter_map(|s| s.trim().parse::<i64>().ok())
                    .collect()
            })
            .unwrap_or_default();
        // The op call's closing paren precedes `[name = ...]` or `;`
        // (a later rfind(')') would land inside `string("...")`).
        let close = rhs[paren..]
            .find(")[")
            .or_else(|| rhs[paren..].find(");"))
            .map(|i| paren + i)
            .unwrap_or(rhs.len());
        let arg_str = &rhs[paren + 1..close];
        let args = arg_str
            .split(',')
            .filter_map(|a| {
                let (k, v) = a.split_once('=')?;
                Some((k.trim().to_string(), v.trim().to_string()))
            })
            .collect();
        ops.push(MilOp {
            op: op.to_string(),
            result: lhs
                .rsplit(|c: char| c == ' ' || c == '>')
                .next()
                .unwrap_or("")
                .trim()
                .to_string(),
            shape,
            args,
        });
    }
    (ops, blobs)
}

/// Load the `u16` payload of a weight blob — deadbeef-headered blobs
/// carry data at byte 128; bare files are read from offset 0.
fn load_blob_u16(path: &std::path::Path) -> Result<Vec<u16>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let off = if b.len() > 132 && &b[64..68] == b"\xef\xbe\xad\xde" {
        128
    } else {
        0
    };
    Ok(b[off..]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Resolve a `BLOBFILE` path — `@model_path/` prefixes and relative
/// paths resolve against the MIL's directory.
fn resolve_blob<'a>(mil_dir: &'a std::path::Path, p: &'a str) -> std::path::PathBuf {
    let p = p.strip_prefix("@model_path/").unwrap_or(p);
    let pb = std::path::PathBuf::from(p);
    if pb.is_absolute() {
        pb
    } else {
        mil_dir.join(pb)
    }
}

/// Extract a scalar `fp16` const's bit pattern: `<ty> <name> =
/// const()[val = fp16(V)]`. Returns `None` if `name` isn't declared
/// as a scalar fp16 const (e.g. a blob tensor).
fn mil_scalar_f16(text: &str, name: &str) -> Option<u16> {
    for raw in text.lines() {
        let line = raw.trim();
        let Some(eq) = line.find(" = ") else { continue };
        if line[..eq].rsplit(|c: char| c == ' ' || c == '>').next() != Some(name) {
            continue;
        }
        let rhs = &line[eq + 3..];
        let v = rhs.find("fp16(")?;
        let end = rhs[v..].find(')')? + v;
        let f: f32 = rhs[v + 5..end].parse().ok()?;
        return Some(half::f16::from_f32(f).to_bits());
    }
    None
}

/// Emit a complete HWX image for a MIL program on a covered skeleton
/// family — the dispatch behind `milc compile --target ane --emit-hwx`.
/// Matches the op sequence against every verified family and loads
/// `BLOBFILE` weights itself; the returned bytes embed `mil_path` /
/// `out_path` in the signature block.
///
/// Covered: `conv1x1` (C=32/64/128, spatial 1/5/8), single `add`/`mul`/
/// `sub`/`sigmoid`/`tanh`, const-weight `matmul` (N=32..256),
/// activation `matmul` and its fused `softmax`/`add`/`mul` partitions
/// (`[1,64,1,64]` and `[1,8,64,64]`), and arbitrary-depth
/// `conv -> {sigmoid|tanh|relu|gelu|silu|softmax|layer_norm|rms_norm}`
/// chains (C=64) with shared/distinct/partially-deduplicated weights.
pub fn emit_from_mil(mil_path: &str, out_path: &str) -> Result<Vec<u8>, String> {
    let text = std::fs::read_to_string(mil_path).map_err(|e| format!("{mil_path}: {e}"))?;
    let mil_dir = std::path::Path::new(mil_path)
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();
    let (ops, blobs) = mil_ops(&text);
    if ops.is_empty() {
        return Err(format!("{mil_path}: no compute ops found"));
    }
    let blob_of = |name: &str| -> Option<String> {
        blobs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p.clone())
    };
    let arg = |op: &MilOp, k: &str| -> Option<String> {
        op.args
            .iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| v.clone())
    };
    let weights_of = |op: &MilOp| -> Result<Vec<u16>, String> {
        let wname = arg(op, "weight").ok_or("conv without weight operand")?;
        let path = blob_of(&wname).ok_or_else(|| format!("weight `{wname}` is not a blob const"))?;
        load_blob_u16(&resolve_blob(&mil_dir, &path))
    };
    let names: Vec<&str> = ops.iter().map(|o| o.op.as_str()).collect();
    let elems = |o: &MilOp| -> u32 {
        o.shape.iter().map(|d| (*d).max(1) as u32).product()
    };

    // ---- single-op families ----
    match names.as_slice() {
        ["conv"] => {
            let o = &ops[0];
            let (c, w) = (o.shape.get(1).copied().unwrap_or(0), o.shape.get(2).copied().unwrap_or(1));
            let shape = match (c, w) {
                (32, 1) => ConvShape::C32W1,
                (64, 1) => ConvShape::C64W1,
                (64, 5) => ConvShape::C64W5,
                (64, 8) => ConvShape::C64W8,
                (128, 1) => ConvShape::C128W1,
                _ => return Err(format!("conv shape {:?} not on a skeleton family", o.shape)),
            };
            return emit_conv1x1(shape, &weights_of(o)?, mil_path, out_path);
        }
        ["add"] => return Ok(emit_elementwise2(EwOp::Add, elems(&ops[0]), mil_path, out_path)),
        ["mul"] => return Ok(emit_elementwise2(EwOp::Mul, elems(&ops[0]), mil_path, out_path)),
        ["sub"] => return Ok(emit_elementwise2(EwOp::Sub, elems(&ops[0]), mil_path, out_path)),
        ["minimum"] => {
            return Ok(emit_elementwise2(EwOp::Min, elems(&ops[0]), mil_path, out_path))
        }
        ["maximum"] => {
            return Ok(emit_elementwise2(EwOp::Max, elems(&ops[0]), mil_path, out_path))
        }
        ["relu"] => return Ok(emit_relu(mil_path, out_path)),
        ["sigmoid"] => return Ok(emit_unary_lut(LutOp::Sigmoid, elems(&ops[0]), mil_path, out_path)),
        ["tanh"] => return Ok(emit_unary_lut(LutOp::Tanh, elems(&ops[0]), mil_path, out_path)),
        [a, b] => {
            let ew = |s: &str| -> Option<EwOp> {
                Some(match s {
                    "add" => EwOp::Add,
                    "mul" => EwOp::Mul,
                    "sub" => EwOp::Sub,
                    "minimum" => EwOp::Min,
                    "maximum" => EwOp::Max,
                    _ => return None,
                })
            };
            if let (Some(op1), Some(op2)) = (ew(a), ew(b)) {
                let (o1, o2) = (&ops[0], &ops[1]);
                if arg(o1, "x").as_deref() != Some("x")
                    || arg(o1, "y").as_deref() != Some("y")
                {
                    return Err("elementwise chain op1 must bind (x = x, y = y)".into());
                }
                let (ox, oy) = (arg(o2, "x"), arg(o2, "y"));
                let (t_first, free) = if ox.as_deref() == Some(o1.result.as_str()) {
                    (true, oy.as_deref())
                } else if oy.as_deref() == Some(o1.result.as_str()) {
                    (false, ox.as_deref())
                } else {
                    return Err("elementwise chain op2 must consume op1's result".into());
                };
                if !matches!(free, Some("x") | Some("y")) {
                    return Err(
                        "elementwise chain op2's free operand must be x or y".into(),
                    );
                }
                return emit_elementwise_chain(
                    op1,
                    op2,
                    t_first,
                    free == Some("y"),
                    elems(o2),
                    mil_path,
                    out_path,
                );
            }
        }
        _ => {}
    }

    // ---- matmul families ----
    let mm_part = |names: &[&str]| -> Option<MmPart> {
        let shape = &ops[0].shape;
        let is64 = shape == &[1, 64, 1, 64];
        let is4d = shape == &[1, 8, 64, 64];
        match (names, is64, is4d) {
            (["matmul"], true, _) => Some(MmPart::Act64),
            (["matmul"], _, true) => Some(MmPart::Act4d),
            (["matmul", "softmax"], true, _) => Some(MmPart::MmSm64),
            (["matmul", "softmax"], _, true) => Some(MmPart::MmSm4d),
            (["softmax", "matmul"], true, _) => Some(MmPart::SmMm64),
            (["matmul", "add", "softmax"] | ["matmul", "mul", "add", "softmax"], true, _) => {
                Some(MmPart::MmAddSm64)
            }
            (["matmul", "mul", "softmax"], true, _) => Some(MmPart::MmMulSm64),
            _ => None,
        }
    };
    if let ["matmul"] = names.as_slice() {
        // const-weight matmul → the MmShape families.
        let y = arg(&ops[0], "y").unwrap_or_default();
        if let Some(path) = blob_of(&y) {
            let n = ops[0].shape.get(1).copied().unwrap_or(0) as usize;
            let shape = match n {
                32 => MmShape::N32,
                64 => MmShape::N64,
                128 => MmShape::N128,
                256 => MmShape::N256,
                _ => return Err(format!("matmul shape {:?} not on a skeleton family", ops[0].shape)),
            };
            return emit_matmul(shape, &load_blob_u16(&resolve_blob(&mil_dir, &path))?, mil_path, out_path);
        }
    }
    if let Some(part) = mm_part(&names) {
        let mut img = emit_mm_part(part, mil_path, out_path);
        // A scalar fp16 `mul` on the matmul output folds into the
        // fused record as the scale operand at __text+0x1ac.
        if matches!(part, MmPart::MmAddSm64 | MmPart::MmMulSm64) {
            let mul = &ops[1];
            let free = if arg(mul, "x").as_deref() == Some(ops[0].result.as_str()) {
                arg(mul, "y")
            } else {
                arg(mul, "x")
            };
            if let Some(bits) = free.and_then(|f| mil_scalar_f16(&text, &f)) {
                img[0x81ac..0x81ae].copy_from_slice(&bits.to_le_bytes());
            }
        }
        return Ok(img);
    }

    // ---- conv chains: group aux ops after each conv ----
    if names.first() != Some(&"conv") || !names.iter().all(|n| {
        matches!(
            *n,
            "conv"
                | "sigmoid"
                | "tanh"
                | "relu"
                | "gelu"
                | "softmax"
                | "layer_norm"
                | "mul"
                | "abs"
                | "reduce_max"
                | "maximum"
                | "real_div"
                | "reduce_mean"
                | "add"
                | "rsqrt"
        )
    }) {
        return Err(format!(
            "op sequence {names:?} is not on a covered skeleton family"
        ));
    }
    let mut layers: Vec<(&MilOp, Vec<&str>)> = Vec::new();
    for (i, o) in ops.iter().enumerate() {
        if o.op == "conv" {
            layers.push((o, Vec::new()));
        } else if let Some(l) = layers.last_mut() {
            l.1.push(names[i]);
        }
    }
    let k = layers.len();
    let ws: Vec<Vec<u16>> = layers
        .iter()
        .map(|(o, _)| weights_of(o))
        .collect::<Result<_, _>>()?;
    let wrefs: Vec<&[u16]> = ws.iter().map(|w| w.as_slice()).collect();
    let aux: &[&str] = &layers[0].1;
    let uniform = layers.iter().all(|(_, a)| a == &aux);
    if !uniform {
        return Err("chain layers have non-uniform aux-op patterns".into());
    }
    match aux {
        [] => Err("plain conv chains are not on a covered family".into()),
        ["sigmoid"] => emit_conv_act_n_dw(FusedAct::Sigmoid, k, &wrefs, mil_path, out_path),
        ["tanh"] => emit_conv_act_n_dw(FusedAct::Tanh, k, &wrefs, mil_path, out_path),
        ["relu"] => emit_conv_act_n_dw(FusedAct::Relu, k, &wrefs, mil_path, out_path),
        ["gelu"] => emit_conv_act_n_dw(FusedAct::Gelu, k, &wrefs, mil_path, out_path),
        ["sigmoid", "mul"] => emit_conv_act_n_dw(FusedAct::Silu, k, &wrefs, mil_path, out_path),
        ["softmax"] => emit_conv_sm_n_dw(k, &wrefs, mil_path, out_path),
        ["layer_norm"] => emit_conv_ln_n_dw(k, &wrefs, mil_path, out_path),
        ["abs", "reduce_max", "maximum", "real_div", "mul", "reduce_mean", "real_div", "mul", "add", "rsqrt", "mul", "mul"] => {
            emit_conv_rms_n_dw(k, &wrefs, mil_path, out_path)
        }
        _ => Err(format!("aux-op pattern {aux:?} is not on a covered family")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serializes tests that toggle the process-global
    /// `HWX_SIG_ANEC` env flag.
    static SIG_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn emits_image_with_magic() {
        let img = emit_elementwise2(EwOp::Add, 64, "/tmp/x/net.mil", "/tmp/x/net.macho");
        assert_eq!(&img[0..4], &0xbeef_face_u32.to_le_bytes());
        assert_eq!(img.len(), 0x10000);
        assert_eq!(&img[0x8000], &SKELETON[0x8000]);
    }

    #[test]
    fn emits_unary_lut_image() {
        let img = emit_unary_lut(LutOp::Sigmoid, 64, "/tmp/x/net.mil", "/tmp/x/net.macho");
        assert_eq!(&img[0..4], &0xbeef_face_u32.to_le_bytes());
        assert_eq!(img.len(), 0x10000);
        assert_eq!(&img[0xc000..0xc002], &[0xf8, 0xc8]);
    }

    /// `assets/conv_perm_vectors.json` holds the `__KERN_0` streams ANE
    /// produced for `w[oc][ic] = oc*C + ic` at C = 32, 64, 128 — the
    /// transform's ground truth.
    #[test]
    fn weight_interleave_matches_anec() {
        let raw = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/conv_perm_vectors.json"
        ))
        .unwrap();
        let vec: serde_json::Value = serde_json::from_str(&raw).unwrap();
        for (c, key) in [
            (32usize, "32w1"),
            (64, "64w1"),
            (64, "64w5"),
            (64, "64w8"),
            (128, "128w1"),
        ] {
            let expect: Vec<u16> = vec[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u16)
                .collect();
            let src: Vec<u16> = (0..(c * c) as u16).collect();
            assert_eq!(weight_interleave(&src, c), expect, "{key}");
        }
    }

    /// Emitting a skeleton family with the corpus compile paths and
    /// perm weights must reproduce the ANEC artifact byte-for-byte.
    #[test]
    fn conv1x1_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        for (shape, case) in [
            (ConvShape::C32W1, "conv32p"),
            (ConvShape::C64W1, "conv64p"),
            (ConvShape::C64W5, "conv64s5p"),
            (ConvShape::C64W8, "conv64s8p"),
            (ConvShape::C128W1, "conv128p"),
        ] {
            let c = shape.channels();
            let weights: Vec<u16> = (0..(c * c) as u16).collect();
            std::env::set_var("HWX_SIG_ANEC", "1");
            let img = emit_conv1x1(
                shape,
                &weights,
                &format!("/tmp/ane_corpus/{case}/net.mil"),
                &format!("/tmp/ane_corpus/{case}/out/net.macho"),
            )
            .unwrap();
            std::env::remove_var("HWX_SIG_ANEC");
            let expect = std::fs::read(format!(
                "{}/assets/{case}_reference.hwx",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            let diffs: Vec<usize> = (0..expect.len().min(img.len()))
                .filter(|&i| img[i] != expect[i])
                .collect();
            assert!(
                diffs.is_empty() && img.len() == expect.len(),
                "{case}: {} diffs, first at {:#x}",
                diffs.len(),
                diffs.first().copied().unwrap_or(0)
            );
        }
    }

    /// The fused conv+sigmoid chain must reproduce the ANEC artifact
    /// byte-for-byte — weights land in the `[LUT|weights]` bands.
    #[test]
    fn conv_sig_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        let weights: Vec<u16> = (0..(64 * 64) as u16).collect();
        std::env::set_var("HWX_SIG_ANEC", "1");
        let img = emit_conv_sig(
            &weights,
            "/tmp/ane_corpus/chain_conv_sigp/net.mil",
            "/tmp/ane_corpus/chain_conv_sigp/out/net.macho",
        )
        .unwrap();
        std::env::remove_var("HWX_SIG_ANEC");
        let expect = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/conv64_sigp_reference.hwx"
        ))
        .unwrap();
        let diffs: Vec<usize> = (0..expect.len().min(img.len()))
            .filter(|&i| img[i] != expect[i])
            .collect();
        assert!(
            diffs.is_empty() && img.len() == expect.len(),
            "conv64_sigp: {} diffs, first at {:#x}",
            diffs.len(),
            diffs.first().copied().unwrap_or(0)
        );
    }

    /// The general k-layer fused conv+sigmoid splicer must reproduce
    /// the ANEC artifacts byte-for-byte for every captured depth —
    /// records, wiring, FVMLIB scalars, and the per-layer reloc table.
    /// Weights are recovered from each reference by inverting the
    /// `__KERN_0` band transform (`weight_interleave` is a bijection).
    #[test]
    fn conv_sig_chain_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        for (layers, asset, case) in [
            (2usize, "conv_sig2_reference.hwx", "chain_conv2_sig2"),
            (3, "conv_sig3_skeleton.hwx", "chain_conv3_sig3"),
            (4, "conv_sig4_reference.hwx", "chain4"),
            (5, "conv_sig5_skeleton.hwx", "chain5"),
        ] {
            let expect = std::fs::read(format!(
                "{}/assets/{asset}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            // invert dst[g*C*B + ic*B + m] = src[(g*B+m)*C + ic];
            // band b holds group g=b, so j = ic*B + m
            const C: usize = 64;
            const B: usize = C / 16;
            let mut weights = vec![0u16; C * C];
            for b in 0..16usize {
                let base = 0xc000 + b * 0x280 + 0x80;
                for j in 0..256usize {
                    let d = u16::from_le_bytes([expect[base + 2 * j], expect[base + 2 * j + 1]]);
                    weights[(b * B + j % B) * C + j / B] = d;
                }
            }
            std::env::set_var("HWX_SIG_ANEC", "1");
            let img = emit_conv_sig_n(
                layers,
                &weights,
                &format!("/tmp/ane_corpus/{case}/net.mil"),
                &format!("/tmp/ane_corpus/{case}/out/net.macho"),
            )
            .unwrap();
            std::env::remove_var("HWX_SIG_ANEC");
            let diffs: Vec<usize> = (0..expect.len().min(img.len()))
                .filter(|&i| img[i] != expect[i])
                .collect();
            assert!(
                diffs.is_empty() && img.len() == expect.len(),
                "{case}: {} diffs, first at {:#x}",
                diffs.len(),
                diffs.first().copied().unwrap_or(0)
            );
        }
    }

    /// Distinct-weight chains — one `__KERN_0` band-group stream per
    /// unique blob, per-record stream ordinals, floated `__text`/
    /// `__KERN_0` planes, and per-stream `K<hash>_ne_<lane>` symbol
    /// sets — must reproduce the ANEC artifacts byte-for-byte,
    /// including the partially-shared `[A, B, A]` layering.
    /// Weights regenerate the corpus pattern `w[i] = i + j*0x5555`.
    #[test]
    fn conv_sig_chain_dw_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        for (layers, n_blobs, asset, case) in [
            (2usize, 2usize, "conv_sig2_dw_reference.hwx", "chain2_dw"),
            (3, 3, "conv_sig3_dw_reference.hwx", "chain3_dw"),
            (3, 2, "conv_sig3_pdw_reference.hwx", "chain3_pdw"),
            (8, 8, "conv_sig8_dw_reference.hwx", "chain8_dw"),
        ] {
            let expect = std::fs::read(format!(
                "{}/assets/{asset}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            let weights: Vec<Vec<u16>> = (0..layers)
                .map(|i| {
                    let j = (i % n_blobs) as u32;
                    (0..64u32 * 64)
                        .map(|e| (e + j * 0x5555) as u16)
                        .collect()
                })
                .collect();
            let wrefs: Vec<&[u16]> = weights.iter().map(|w| w.as_slice()).collect();
            std::env::set_var("HWX_SIG_ANEC", "1");
            let img = emit_conv_sig_n_dw(
                layers,
                &wrefs,
                &format!("/tmp/ane_corpus/{case}/net.mil"),
                &format!("/tmp/ane_corpus/{case}/out/net.macho"),
            )
            .unwrap();
            std::env::remove_var("HWX_SIG_ANEC");
            let diffs: Vec<usize> = (0..expect.len().min(img.len()))
                .filter(|&i| img[i] != expect[i])
                .collect();
            assert!(
                diffs.is_empty() && img.len() == expect.len(),
                "{case}: {} diffs, first at {:#x}",
                diffs.len(),
                diffs.first().copied().unwrap_or(0)
            );
        }
    }

    /// Activation-family chains — relu (LUT-free `0x200` bands,
    /// `0x2000` streams, `0x11` descriptor count) and tanh (sigmoid
    /// layout, tanh LUT and kernel-name table) — must reproduce the
    /// ANEC artifacts byte-for-byte across k=1, shared chains, and
    /// the 8-stream distinct-weight case. Shared cases use the
    /// corpus ones blob; dw uses `w[i] = i + j*0x5555`.
    #[test]
    fn conv_act_chain_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        for (act, layers, n_blobs, asset, case) in [
            (FusedAct::Relu, 1usize, 1usize, "conv64_relu_skeleton.hwx", "chain_conv_relu"),
            (FusedAct::Relu, 2, 1, "conv_act2_relu_reference.hwx", "chain_conv2_relu2"),
            (FusedAct::Relu, 5, 1, "conv_sig5_relu_reference.hwx", "chain5_relu"),
            (FusedAct::Relu, 8, 8, "conv_act8_relu_dw_reference.hwx", "chain8_relu_dw"),
            (FusedAct::Tanh, 1, 1, "conv64_tanh_skeleton.hwx", "chain_conv_tanh"),
            (FusedAct::Tanh, 2, 1, "conv_act2_tanh_reference.hwx", "chain_conv2_tanh2"),
            (FusedAct::Tanh, 5, 1, "conv_sig5_tanh_reference.hwx", "chain5_tanh"),
            (FusedAct::Tanh, 8, 8, "conv_act8_tanh_dw_reference.hwx", "chain8_tanh_dw"),
            (FusedAct::Gelu, 1, 1, "conv64_gelu_skeleton.hwx", "chain_conv_gelu"),
            (FusedAct::Gelu, 2, 1, "conv_act2_gelu_reference.hwx", "chain_conv2_gelu2"),
            (FusedAct::Gelu, 5, 1, "conv_sig5_gelu_reference.hwx", "chain5_gelu"),
            (FusedAct::Gelu, 8, 8, "conv_act8_gelu_dw_reference.hwx", "chain8_gelu_dw"),
            (FusedAct::Silu, 1, 1, "conv64_silu_skeleton.hwx", "chain_conv_silu"),
            (FusedAct::Silu, 2, 1, "conv_act2_silu_reference.hwx", "chain_conv2_silu2"),
            (FusedAct::Silu, 5, 1, "conv_sig5_silu_reference.hwx", "chain5_silu"),
            (FusedAct::Silu, 8, 8, "conv_act8_silu_dw_reference.hwx", "chain8_silu_dw"),
            (FusedAct::Softmax, 1, 1, "conv_sm_perm_reference.hwx", "smp"),
        ] {
            let expect = std::fs::read(format!(
                "{}/assets/{asset}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            let weights: Vec<Vec<u16>> = (0..layers)
                .map(|i| {
                    let j = (i % n_blobs) as u32;
                    if case == "smp" {
                        // perm-weight softmax reference
                        (0..64u32 * 64).map(|e| e as u16).collect()
                    } else if n_blobs == 1 {
                        vec![0x3c00u16; 64 * 64]
                    } else {
                        (0..64u32 * 64).map(|e| (e + j * 0x5555) as u16).collect()
                    }
                })
                .collect();
            let wrefs: Vec<&[u16]> = weights.iter().map(|w| w.as_slice()).collect();
            std::env::set_var("HWX_SIG_ANEC", "1");
            let img = emit_conv_act_n_dw(
                act,
                layers,
                &wrefs,
                &format!("/tmp/ane_act/{case}/net.mil"),
                &format!("/tmp/ane_act/{case}/out/net.macho"),
            )
            .unwrap();
            std::env::remove_var("HWX_SIG_ANEC");
            let diffs: Vec<usize> = (0..expect.len().min(img.len()))
                .filter(|&i| img[i] != expect[i])
                .collect();
            assert!(
                diffs.is_empty() && img.len() == expect.len(),
                "{case}: {} diffs, first at {:#x}",
                diffs.len(),
                diffs.first().copied().unwrap_or(0)
            );
        }
    }

    /// Softmax multi-task chains — `(conv -> softmax)^k` decomposes
    /// into three task descriptors per layer (conv + sm-reduce +
    /// sm-final with the terminal writeback epilogue). Shared cases
    /// use all-ones weights; dw uses `w[i] = i + j*0x5555` and the
    /// pdw case reuses blob 0 at layer 2.
    #[test]
    fn conv_sm_chain_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        for (layers, n_blobs, asset, case) in [
            (2usize, 1usize, "conv_sm2_reference.hwx", "sm2"),
            (3, 1, "conv_sm3_skeleton.hwx", "sm3"),
            (4, 1, "conv_sm4_reference.hwx", "sm4"),
            (5, 1, "conv_sm5_reference.hwx", "sm5"),
            (2, 2, "conv_smdw2_reference.hwx", "smdw2"),
            (3, 3, "conv_smdw3_reference.hwx", "smdw3"),
            (4, 4, "conv_smdw4_reference.hwx", "smdw4"),
            (3, 2, "conv_smpdw_reference.hwx", "smpdw"),
        ] {
            let expect = std::fs::read(format!(
                "{}/assets/{asset}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            let weights: Vec<Vec<u16>> = (0..layers)
                .map(|i| {
                    let j = (i % n_blobs) as u32;
                    if n_blobs == 1 {
                        vec![0x3c00u16; 64 * 64]
                    } else {
                        (0..64u32 * 64).map(|e| (e + j * 0x5555) as u16).collect()
                    }
                })
                .collect();
            let wrefs: Vec<&[u16]> = weights.iter().map(|w| w.as_slice()).collect();
            std::env::set_var("HWX_SIG_ANEC", "1");
            let img = emit_conv_sm_n_dw(
                layers,
                &wrefs,
                &format!("/tmp/ane_sm/{case}/net.mil"),
                &format!("/tmp/ane_sm/{case}/out/net.macho"),
            )
            .unwrap();
            std::env::remove_var("HWX_SIG_ANEC");
            let diffs: Vec<usize> = (0..expect.len().min(img.len()))
                .filter(|&i| img[i] != expect[i])
                .collect();
            assert!(
                diffs.is_empty() && img.len() == expect.len(),
                "{case}: {} diffs, first at {:#x}",
                diffs.len(),
                diffs.first().copied().unwrap_or(0)
            );
        }
    }

    /// Layer-norm fused chains — `(conv -> layer_norm)^k` is one
    /// fused task per layer (5 task ids, bank-select parity
    /// alternation). Shared cases use all-ones weights; dw uses
    /// `w[i] = i + seeds[j]` with seeds `[1, 0x5555, 0xaaaa, 0x3333]`.
    #[test]
    fn conv_ln_chain_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        const SEEDS: [u32; 4] = [1, 0x5555, 0xaaaa, 0x3333];
        for (layers, n_blobs, asset, case) in [
            (1usize, 1usize, "conv_ln1_skeleton.hwx", "ln1"),
            (2, 1, "conv_ln2_reference.hwx", "ln2"),
            (3, 1, "conv_ln3_skeleton.hwx", "ln3"),
            (4, 1, "conv_ln4_skeleton.hwx", "ln4"),
            (5, 1, "conv_ln5_reference.hwx", "ln5"),
            (6, 1, "conv_ln6_reference.hwx", "ln6"),
            (2, 2, "conv_lndw2_reference.hwx", "lndw2"),
            (3, 3, "conv_lndw3_reference.hwx", "lndw3"),
            (4, 4, "conv_lndw4_reference.hwx", "lndw4"),
        ] {
            let expect = std::fs::read(format!(
                "{}/assets/{asset}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            let weights: Vec<Vec<u16>> = (0..layers)
                .map(|i| {
                    let j = i % n_blobs;
                    if n_blobs == 1 {
                        vec![0x3c00u16; 64 * 64]
                    } else {
                        (0..64u32 * 64).map(|e| (e + SEEDS[j]) as u16).collect()
                    }
                })
                .collect();
            let wrefs: Vec<&[u16]> = weights.iter().map(|w| w.as_slice()).collect();
            std::env::set_var("HWX_SIG_ANEC", "1");
            let img = emit_conv_ln_n_dw(
                layers,
                &wrefs,
                &format!("/tmp/ane_ln/{case}/net.mil"),
                &format!("/tmp/ane_ln/{case}/out/net.macho"),
            )
            .unwrap();
            std::env::remove_var("HWX_SIG_ANEC");
            let diffs: Vec<usize> = (0..expect.len().min(img.len()))
                .filter(|&i| img[i] != expect[i])
                .collect();
            assert!(
                diffs.is_empty() && img.len() == expect.len(),
                "{case}: {} diffs, first at {:#x}",
                diffs.len(),
                diffs.first().copied().unwrap_or(0)
            );
        }
    }

    /// Rms-norm multi-task chains — `(conv -> rms_norm)^k` decomposes
    /// into four task descriptors per layer on a 12-task-id stride
    /// (conv + div-max + square-mean + rsqrt-mul auxes), with the
    /// compact five-task k=1 family as a separate case. dw uses
    /// `w[i] = i + j*0x5555`; the pdw case reuses blob 0 at layer 2.
    #[test]
    fn conv_rms_chain_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        for (layers, n_blobs, asset, case, nested) in [
            (1usize, 1usize, "conv_rms1_skeleton.hwx", "rms1", false),
            (2, 1, "conv_rms2_skeleton.hwx", "rms2", false),
            (3, 1, "conv_rms3_reference.hwx", "rms3", false),
            (4, 1, "conv_rms4_skeleton.hwx", "rms4", false),
            (5, 1, "conv_rms5_skeleton.hwx", "rms5", false),
            (6, 1, "conv_rms6_reference.hwx", "rms6", true),
            (2, 2, "conv_rmsdw2_reference.hwx", "rmsdw2", false),
            (3, 3, "conv_rmsdw3_reference.hwx", "rmsdw3", true),
            (4, 4, "conv_rmsdw4_reference.hwx", "rmsdw4", true),
            (3, 2, "conv_rmspdw3_reference.hwx", "rmspdw3", true),
        ] {
            let expect = std::fs::read(format!(
                "{}/assets/{asset}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            const SEEDS: [u32; 4] = [0, 0x5555, 0xaaaa, 0x3333];
            let weights: Vec<Vec<u16>> = (0..layers)
                .map(|i| {
                    let j = i % n_blobs;
                    if n_blobs != 1 {
                        (0..64u32 * 64)
                            .map(|e| (e + SEEDS[j]) as u16)
                            .collect()
                    } else if case == "rms6" {
                        // the rms6 probe's weight.bin is the `w[i]=i`
                        // ramp, not ones
                        (0..64u32 * 64).map(|e| e as u16).collect()
                    } else {
                        vec![0x3c00u16; 64 * 64]
                    }
                })
                .collect();
            let wrefs: Vec<&[u16]> = weights.iter().map(|w| w.as_slice()).collect();
            let out = if nested {
                format!("/tmp/ane_rms/{case}/out/net.macho/net.macho")
            } else {
                format!("/tmp/ane_rms/{case}/out/net.macho")
            };
            std::env::set_var("HWX_SIG_ANEC", "1");
            let img = emit_conv_rms_n_dw(
                layers,
                &wrefs,
                &format!("/tmp/ane_rms/{case}/net.mil"),
                &out,
            )
            .unwrap();
            std::env::remove_var("HWX_SIG_ANEC");
            let diffs: Vec<usize> = (0..expect.len().min(img.len()))
                .filter(|&i| img[i] != expect[i])
                .collect();
            assert!(
                diffs.is_empty() && img.len() == expect.len(),
                "{case}: {} diffs, first at {:#x}",
                diffs.len(),
                diffs.first().copied().unwrap_or(0)
            );
        }
    }

    /// The fused two-elementwise-op family must likewise reproduce
    /// the ANEC artifacts byte-for-byte: both operand orders
    /// (`z = op2(t, x)` vs `z = op2(x, t)`), the free-operand-`y`
    /// descriptor swap, and the n=128 lane-descriptor ramp.
    #[test]
    fn ew_chain_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        for (op1, op2, t_first, free_y, n, mil, out, asset) in [
            (
                EwOp::Add,
                EwOp::Mul,
                true,
                false,
                64,
                "/tmp/ane_act/chain_add_mul/net.mil",
                "/tmp/ane_act/chain_add_mul/out/net.macho",
                "ew_chain_skeleton.hwx",
            ),
            (
                EwOp::Add,
                EwOp::Add,
                true,
                true,
                64,
                "/tmp/ane_act/chain_add_add/net.mil",
                "/tmp/ane_act/chain_add_add/out/net.macho",
                "ew_chain_add_add.hwx",
            ),
            (
                EwOp::Add,
                EwOp::Mul,
                true,
                false,
                128,
                "/tmp/ew_chain/n128_add_mul/net.mil",
                "/tmp/ew_chain/n128_add_mul/out/net.macho/net.macho",
                "ew_chain_n128.hwx",
            ),
            (
                EwOp::Add,
                EwOp::Mul,
                false,
                false,
                64,
                "/tmp/ew_chain/swap_add_mul/net.mil",
                "/tmp/ew_chain/swap_add_mul/out/net.macho/net.macho",
                "ew_chain_swap.hwx",
            ),
        ] {
            std::env::set_var("HWX_SIG_ANEC", "1");
            let img = emit_elementwise_chain(op1, op2, t_first, free_y, n, mil, out).unwrap();
            std::env::remove_var("HWX_SIG_ANEC");
            let expect = std::fs::read(format!("{}/assets/{asset}", env!("CARGO_MANIFEST_DIR"))).unwrap();
            let diffs: Vec<usize> = (0..expect.len().min(img.len()))
                .filter(|&i| img[i] != expect[i])
                .collect();
            assert!(
                diffs.is_empty() && img.len() == expect.len(),
                "{asset}: {} diffs, first at {:#x}",
                diffs.len(),
                diffs.first().copied().unwrap_or(0)
            );
        }
    }

    /// Matmul emission must likewise reproduce the ANEC artifact
    /// byte-for-byte on the perm-weight probes.
    #[test]
    fn matmul_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        for (shape, case) in [(MmShape::N32, "mm32p"), (MmShape::N64, "mm64p")] {
            let n = shape.n();
            let weights: Vec<u16> = (0..(n * n) as u16).collect();
            std::env::set_var("HWX_SIG_ANEC", "1");
            let img = emit_matmul(
                shape,
                &weights,
                &format!("/tmp/ane_corpus/{case}/net.mil"),
                &format!("/tmp/ane_corpus/{case}/out/net.macho"),
            )
            .unwrap();
            std::env::remove_var("HWX_SIG_ANEC");
            let expect = std::fs::read(format!(
                "{}/assets/{case}_reference.hwx",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            let diffs: Vec<usize> = (0..expect.len().min(img.len()))
                .filter(|&i| img[i] != expect[i])
                .collect();
            assert!(
                diffs.is_empty() && img.len() == expect.len(),
                "{case}: {} diffs, first at {:#x}",
                diffs.len(),
                diffs.first().copied().unwrap_or(0)
            );
        }
    }

    /// The activation-matmul attention family must likewise reproduce
    /// the ANEC artifacts byte-for-byte: `matmul(x, y)` with both
    /// operands program inputs, `matmul -> softmax`,
    /// `softmax -> matmul`, masked/scaled QK^T partitions, and the
    /// 4-D heads×seq forms — the partitions an attention block splits
    /// into (ANEC refuses two chained matmuls in one program).
    #[test]
    fn mm_act_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        for (part, case, dir) in [
            (MmPart::Act64, "mm_act64", "mm_act"),
            (MmPart::MmSm64, "mmsm64", "mmsm"),
            (MmPart::SmMm64, "smmm64", "smmm"),
            (MmPart::MmAddSm64, "mmaddsm", "mmaddsm"),
            (MmPart::MmMulSm64, "mmmulsm", "mmmulsm"),
            (MmPart::Act4d, "mm4d", "mm4d"),
            (MmPart::MmSm4d, "mmsm4d", "mmsm4d"),
        ] {
            std::env::set_var("HWX_SIG_ANEC", "1");
            let img = emit_mm_part(
                part,
                &format!("/tmp/ane_attn/{dir}/net.mil"),
                &format!("/tmp/ane_attn/{dir}/out/net.macho/net.macho"),
            );
            std::env::remove_var("HWX_SIG_ANEC");
            let expect = std::fs::read(format!(
                "{}/assets/{case}_skeleton.hwx",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            let diffs: Vec<usize> = (0..expect.len().min(img.len()))
                .filter(|&i| img[i] != expect[i])
                .collect();
            assert!(
                diffs.is_empty() && img.len() == expect.len(),
                "{case}: {} diffs, first at {:#x}",
                diffs.len(),
                diffs.first().copied().unwrap_or(0)
            );
        }
    }
}
