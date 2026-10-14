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
}

impl EwOp {
    fn opcode(self) -> u32 {
        match self {
            EwOp::Add => 0x0008_0000,
            EwOp::Mul => 0x0008_0004,
            EwOp::Sub => 0x000c_0000,
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
    rebuild_tail(&mut img, n, mil_path, out_path);

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
fn rebuild_tail(img: &mut Vec<u8>, n: u32, mil_path: &str, out_path: &str) {
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
    img.extend_from_slice(&SKELETON[0x8000..]);
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
    let kb = format!(
        "K{}",
        mil_spec::sha256::sha256_hex(&op.lut()).to_uppercase()
    );
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
