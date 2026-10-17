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
/// `W=8` convs at other channel counts — the record program and the
/// `__KERN_0` band layout both vary with `C` (`C=16` packs sixteen
/// weight bands at a `0x20`-u16 stride; `C>=32` is contiguous).
const SKELETON_CONV16S8: &[u8] = include_bytes!("../assets/conv16s8_skeleton.hwx");
const SKELETON_CONV32S8: &[u8] = include_bytes!("../assets/conv32s8_skeleton.hwx");
const SKELETON_CONV128S8: &[u8] = include_bytes!("../assets/conv128s8_skeleton.hwx");

/// Square-matmul skeletons: `z = matmul(x, w, transpose_y)` over fp16
/// `[1,N,1,N]` with a `[1,N,1,N]` constant weight, h16. Unlike conv,
/// weights are stored verbatim inside `__TEXT`'s const pool — no
/// `__KERN_0` segment and no `__text` relocations.
/// `conv64s1` + `sigmoid` chain reference (ones weights): `__KERN_0`
/// carries sixteen `0x280`-byte bands of `[0x80 LUT][0x200 weights]`.
const SKELETON_CONV64_SIG: &[u8] = include_bytes!("../assets/conv64_sig_skeleton.hwx");
const SKELETON_MM16: &[u8] = include_bytes!("../assets/mm16_skeleton.hwx");
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

fn put_u64(buf: &mut [u8], off: usize, v: u64) {
    buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// Emit a complete HWX image for `z = x ⊕ y` over fp16 `[1, N, 1, 1]`.
///
/// `mil_path` / `out_path` are recorded in the signature block's
/// compilation record, matching the reference format.
pub fn emit_elementwise2(op: EwOp, n: u32, mil_path: &str, out_path: &str) -> Vec<u8> {
    if n > 256 {
        return emit_elementwise2_big(op, n, mil_path, out_path);
    }
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

/// Big-n skeleton: ANEC's `[n,1,1,1]` dim-0 layout for `n > 256`
/// (the threshold is `n*64 > 0x4000` — one IOSurface page). Task
/// record is n-invariant except `+0x14` (tile count
/// `n/128 + n/1024`) and `+0x5c` (element dim `n`); opcode at
/// `0x80d0`. Verified against ANEC refs at n = 257, 300, 384, 512,
/// 768, 1024, 1280, 1536.
const SKELETON_EW2_BIG: &[u8] = include_bytes!("../assets/ew2_big_skeleton.hwx");

/// Emit `z = x ⊕ y` over fp16 `[n,1,1,1]` for `n > 256`. Surfaces are
/// `S = align(n*64, 0x4000)` bytes each; port VAs step by `S`; the
/// fourth region (bss/intermediate) sits at `3*S`.
fn emit_elementwise2_big(op: EwOp, n: u32, mil_path: &str, out_path: &str) -> Vec<u8> {
    let mut img = SKELETON_EW2_BIG.to_vec();
    let s = ((n as usize * 64 + 0x3fff) & !0x3fff) as u32;

    // segment vmsize fields (page-aligned) and filesize fields (raw)
    for &f in &[0x88usize, 0x120, 0x1b8] {
        put_u32(&mut img, f, s);
    }
    let raw = n * 64;
    for &f in &[0xd8usize, 0x170, 0x208, 0xc98, 0x19e0, 0x2728] {
        put_u32(&mut img, f, raw);
    }
    // symbol/descriptor VAs — 0x3000_0000 | k*S
    for &f in &[0x118usize, 0x168, 0x348, 0x3e8] {
        put_u32(&mut img, f, 0x3000_0000 | s);
    }
    for &f in &[0x1b0usize, 0x200, 0x368, 0x3f8] {
        put_u32(&mut img, f, 0x3000_0000 | (2 * s));
    }
    for &f in &[0x248usize, 0x298, 0x388, 0xb88] {
        put_u32(&mut img, f, 0x3000_0000 | (3 * s));
    }
    for &f in &[0x2e8usize, 0x398] {
        put_u32(&mut img, f, 0x3000_0000 | (3 * s + 0x140));
    }
    // element-dim fields
    for &f in &[0xc40usize, 0x1988, 0x26d0] {
        put_u32(&mut img, f, n);
    }
    // tile-group count (kernel loop bound)
    let tiles = n / 128 + n / 1024;
    put_u32(&mut img, 0xc08, tiles);

    rebuild_tail_g(
        &mut img,
        n,
        mil_path,
        out_path,
        &SKELETON_EW2_BIG[0x8000..],
        s,
        true,
    );

    // task-record fields — after the tail splice, which restores the
    // instruction region from the skeleton
    put_u32(&mut img, 0x8014, tiles);
    put_u32(&mut img, 0x805c, n);
    put_u32(&mut img, 0x80d0, op.opcode());
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
    put_u32(
        &mut img,
        r + 0x178,
        ((n * 2) & 0xff) | ((v16 + n / 128) << 8),
    );

    // lane descriptors: which program input is op2's free operand
    // swaps the two descriptor halves.
    let ac = ((n * 2) & 0xff) + if n % 64 == 0 { 0x40 } else { 0 };
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
    layout_desc_w(name, base, n, 1)
}

/// W-aware form: section bytes `stride64 * c` at `{b1}`, dim `c` at
/// `n`, channel stride `stride64` at `{b2}`/`{b3}`, spatial dim `w`
/// at `h`. For `w == 1` this reduces to the `s64`/`h=1` shape every
/// W=1 corpus artifact carries.
fn layout_desc_w(name: &str, base: u32, c: u32, w: u32) -> String {
    let stride64 = ((w as usize * 2 + 63) & !63) as u32;
    format!(
        "{name}:t{base}=ar1;0;1;{b1}=s{ns}n:ar1;0;{c};{b2}=s{st}c:ar1;0;1;{b3}=s{st}h:ar1;0;{w};{b4}=s2w:5",
        base = base,
        b1 = base + 1,
        ns = stride64 * c,
        c = c,
        b2 = base + 2,
        st = stride64,
        b3 = base + 3,
        w = w,
        b4 = base + 4,
    )
}

/// Big-n (`n > 256`) elementwise descriptors: the element axis moves
/// to dim-0 (`ar1;0;{n}` stride `s64`) and every other dim collapses
/// to extent 1 — the layout ANEC emits once `n*64` exceeds one page.
fn layout_desc_hi(name: &str, base: u32, n: u32) -> String {
    format!(
        "{name}:t{base}=ar1;0;{n};{b1}=s64n:ar1;0;1;{b2}=s64c:ar1;0;1;{b3}=s64h:ar1;0;1;{b4}=s2w:5",
        base = base,
        n = n,
        b1 = base + 1,
        b2 = base + 2,
        b3 = base + 3,
        b4 = base + 4,
    )
}

/// Rebuild everything after the z-task-descriptor (file offset
/// `0x33f0`): signature command, `rt.version`, `LC_SYMTAB`, symbol
/// table, string table — then fix `sizeofcmds` in the header.
fn rebuild_tail(img: &mut Vec<u8>, n: u32, mil_path: &str, out_path: &str, post: &[u8]) {
    rebuild_tail_g(img, n, mil_path, out_path, post, 0x4000, false)
}

/// Generic form: `va_step` is the per-port IOSurface stride in the
/// symbol table (0x4000 for the one-page small-n layout, `align(n*64,
/// 0x4000)` for the big-n dim-0 layout); `big` selects the dim-0
/// element-axis descriptor.
fn rebuild_tail_g(
    img: &mut Vec<u8>,
    n: u32,
    mil_path: &str,
    out_path: &str,
    post: &[u8],
    va_step: u32,
    big: bool,
) {
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
        ("y", 2, 0x3000_0000 | va_step as u64),
        ("z", 3, 0x3000_0000 | (2 * va_step as u64)),
    ];
    let layout_strs = if big {
        [
            layout_desc_hi("x", 24, n),
            layout_desc_hi("y", 29, n),
            layout_desc_hi("z", 34, n),
        ]
    } else {
        [
            layout_desc("x", 24, n),
            layout_desc("y", 29, n),
            layout_desc("z", 34, n),
        ]
    };

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
    entries.push((strx(kb), 0x0f, 5, 0, 0x3001_0080));
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
    let reloff = 0x2688 + tail.len().div_ceil(8) * 8;
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

/// General `__KERN_0` layout for a square `[C,C,1,1]` conv weight at
/// any `C`: output channels spread over sixteen lanes, the first
/// `C % 16` lanes taking one extra row; each lane stores its rows
/// transposed (`for ic { for m in rows }`) and pads to 64 bytes.
/// Returns the stream plus each lane's byte offset and padded size
/// (the record's lane tables). Reduces to [`weight_interleave`] when
/// `32 | C`. For `C > 512` rows are consumed in passes of at most 512
/// (32 per lane — the ANE lane-depth limit); each lane's stream is the
/// concat of per-pass `for ic { for m }` chunks. Verified byte-exact
/// against ANEC for 34 channel counts, `C = 8..1056`, including uneven
/// `C = 24, 33, 40, 56, 72, 80, 112, 520`.
pub fn weight_layout(w: &[u16], c: usize) -> (Vec<u16>, Vec<usize>, Vec<usize>) {
    weight_layout_ns(w, c, c)
}

/// Non-square form: `w` is a row-major `[cout, cin]` weight. Same 16
/// lane / 512-row-pass structure as [`weight_layout`] — the row dim is
/// `cout`, the inner `ic` loop runs the full `cin`. Verified
/// byte-exact against ANEC over ~150 swept `(cin, cout)` pairs
/// including partial lane rows (`cout % 16 != 0`), `cout < 32` and
/// multi-pass `cout > 512`.
pub fn weight_layout_ns(w: &[u16], cin: usize, cout: usize) -> (Vec<u16>, Vec<usize>, Vec<usize>) {
    debug_assert_eq!(w.len(), cin * cout);
    let (mut out, mut offs, mut sizes) = (Vec::with_capacity(cin * cout), Vec::new(), Vec::new());
    for g in 0..16 {
        offs.push(out.len() * 2);
        let mut base = 0;
        while base < cout {
            let pr = (cout - base).min(512);
            let (pq, pr2) = (pr / 16, pr % 16);
            let rows = pq + usize::from(g < pr2);
            let row0 = base + g * pq + g.min(pr2);
            for ic in 0..cin {
                for m in 0..rows {
                    out.push(w[(row0 + m) * cin + ic]);
                }
            }
            base += pr;
        }
        let bytes = out.len() * 2 - offs[g];
        let pad = (64 - bytes % 64) % 64;
        out.resize(out.len() + pad / 2, 0);
        sizes.push(bytes + pad);
    }
    (out, offs, sizes)
}

/// ANEC's conv kernel symbol name (`K<sha256>`) for a square
/// `[C,C,1,1]` weight const at `@model_path/weights/weight.bin`.
///
/// The name is SHA-256 over a 235-byte record ANEC builds from the
/// weight const — blob stem + file name, data offset and byte size, a
/// `desc` block with the weight dims, and a trailer carrying the
/// per-lane row count `ceil(C/16)` capped at 32 — the ANE lane-depth
/// limit, verified at C=640/768/1024 where `ceil(C/16)` exceeds it.
/// Recovered by tracing `CC_SHA256_Update` inside `ANECompiler` during
/// a live compile; it is independent of the weight values and of the
/// activation's spatial size. Verified against 19 swept channel counts
/// and the corpus conv32/64/128 artifacts (square weights only).
pub fn conv_kernel_name(c: u64, data_offset: u64) -> String {
    conv_kernel_name_ns(c, c, data_offset)
}

/// Non-square form: `[cout, cin, 1, 1]` weight. The desc block is the
/// weight's `[cout, cin]` dims with `cin` repeated as the row stride;
/// the lane-depth trailer keys off `cout`. Verified at (64,32) and
/// (64,256) against ANEC kernel symbols.
pub fn conv_kernel_name_ns(cin: u64, cout: u64, data_offset: u64) -> String {
    let mut s: Vec<u8> = Vec::with_capacity(235);
    s.extend_from_slice(b"weight");
    s.extend_from_slice(b"weight.bin0");
    s.extend_from_slice(&data_offset.to_le_bytes());
    s.extend_from_slice(&(cout * cin * 2).to_le_bytes());
    s.extend_from_slice(&[0; 5]);
    s.extend_from_slice(b"desc");
    s.extend_from_slice(&[0; 40]);
    for v in [cout, cin, 1, 1, 1, cin, 1, 1, 1, 1, 0, 0] {
        s.extend_from_slice(&v.to_le_bytes());
    }
    for v in [1u32, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0] {
        s.extend_from_slice(&v.to_le_bytes());
    }
    s.extend_from_slice(&cout.div_ceil(16).min(32).to_le_bytes());
    s.push(0);
    debug_assert_eq!(s.len(), 235);
    format!("K{}", mil_spec::sha256::sha256_hex(&s).to_uppercase())
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
    /// `[1,16,8,1]` in, `[16,16,1,1]` weight.
    C16W8,
    /// `[1,32,8,1]` in, `[32,32,1,1]` weight.
    C32W8,
    /// `[1,128,8,1]` in, `[128,128,1,1]` weight.
    C128W8,
}

impl ConvShape {
    /// Channel count `C` of the `[C,C,1,1]` weight.
    pub fn channels(self) -> usize {
        match self {
            ConvShape::C32W1 => 32,
            ConvShape::C64W1 | ConvShape::C64W5 | ConvShape::C64W8 => 64,
            ConvShape::C128W1 => 128,
            ConvShape::C16W8 => 16,
            ConvShape::C32W8 => 32,
            ConvShape::C128W8 => 128,
        }
    }

    fn skeleton(self) -> &'static [u8] {
        match self {
            ConvShape::C32W1 => SKELETON_CONV32,
            ConvShape::C64W1 => SKELETON_CONV64S1,
            ConvShape::C64W5 => SKELETON_CONV64S5,
            ConvShape::C64W8 => SKELETON_CONV64S8,
            ConvShape::C128W1 => SKELETON_CONV128,
            ConvShape::C16W8 => SKELETON_CONV16S8,
            ConvShape::C32W8 => SKELETON_CONV32S8,
            ConvShape::C128W8 => SKELETON_CONV128S8,
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
    let reloff =
        u32::from_le_bytes(skel[reloff_field..reloff_field + 4].try_into().unwrap()) as usize;
    let n =
        u32::from_le_bytes(skel[reloff_field + 4..reloff_field + 8].try_into().unwrap()) as usize;
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
        let reloff = tail_off + tail.len().div_ceil(8) * 8;
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

/// [`rebuild_tail_at`] for families whose `__text` section sits deeper
/// than `0x4000` (the attnv family keeps it at `0x8000`). The emitted
/// tail must still fit below `src_text_off` — the load-command
/// `fileoff` is not patched, so the splice offset must match.
fn rebuild_tail_at_text(
    skel: &[u8],
    tail_off: usize,
    src_text_off: usize,
    entries: &[(String, u8, u8, u16, u64)],
    relocs: &[[u8; 8]],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    let (mut img, text_off) = build_tail_at(skel, tail_off, entries, relocs, mil_path, out_path);
    if text_off > src_text_off {
        return Err(format!(
            "tail ({text_off:#x}) overflows __text fileoff {src_text_off:#x}"
        ));
    }
    img.resize(src_text_off, 0);
    img.extend_from_slice(&skel[src_text_off..]);
    Ok(img)
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
    if c == 16 {
        // C=16 pads each 16-wide band to a 0x20-u16 stride
        for b in 0..16usize {
            for (i, v) in stream[b * 16..b * 16 + 16].iter().enumerate() {
                let off = CONV_WEIGHTS_OFF + b * 0x40 + i * 2;
                img[off..off + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
    } else {
        for (i, v) in stream.iter().enumerate() {
            img[CONV_WEIGHTS_OFF + i * 2..CONV_WEIGHTS_OFF + i * 2 + 2]
                .copy_from_slice(&v.to_le_bytes());
        }
    }
    Ok(img)
}

/// `__text` encoding template for the narrow-channel regime
/// (`ceil(C/16) in 5..=8`): the same 0x84 lane table and tail fields as
/// the standard form, minus one instruction word (0x1cc vs 0x1d0) and
/// the `f128` field. Sliced from an ANEC `C=80` artifact.
const DYN_TEXT_NARROW: &[u8] = include_bytes!("../assets/conv_dyn_text_narrow.bin");

/// Per-channel scheduler scalars for [`emit_conv1x1_dyn`]:
/// `(f14, pad16, big)`. `f14` is the program-descriptor count stored in
/// the tagged field at blob `0xbe8` / text `0x14`; `pad16` is the extra
/// `*16` row padding applied to the text `0x158`/`0x168` copies; `big`
/// selects the `0x244404` text `0x124` flag. Inside the swept range
/// (`C in 32..=4096`) these come from
/// [`crate::conv_chan_table::CONV_CHAN_TABLE`], which stores ANEC's
/// observed values verbatim — a `u32::MAX` `f14` marks the `713..=720`
/// 59-symbol variant window. Beyond the table the fallback uses the
/// kernel byte size directly (chunk count ≈ `ceil(kern/0xC000)`, no row
/// pad, `big` once the payload exceeds 1 MiB).
fn conv_chan_params(c: usize, kern_sz: usize) -> (u32, usize, bool) {
    let table = crate::conv_chan_table::CONV_CHAN_TABLE;
    if (32..=4096).contains(&c) {
        let v = table[c - 32];
        if v != u32::MAX {
            return (v & 0xfff, ((v >> 12) & 0xf) as usize, v & 0x10000 != 0);
        }
    }
    conv_ns_fallback(c, c, kern_sz)
}

/// Fallback `(f14, pad16, big)` for shapes outside the probed tables:
/// `f14` sums the per-512-row-pass kernel chunk counts (exact for most
/// swept pairs; ANEC's residual rounding is irregular), `pad16` and
/// `big` key on the observed `kern > 1 MiB` transition.
fn conv_ns_fallback(cin: usize, cout: usize, kern_sz: usize) -> (u32, usize, bool) {
    let mut f14 = 0u32;
    let mut rem = cout;
    while rem > 0 {
        let pr = rem.min(512);
        f14 += ((pr * cin * 2).div_ceil(0xc000)) as u32;
        rem -= pr;
    }
    let big = kern_sz > 0x100000;
    (f14, if big { 3 } else { 0 }, big)
}

/// `(f14, pad16, big)` for a `(cin, cout)` pair: exact square values on
/// the diagonal, [`crate::conv_ns_table::CONV_NS_TABLE`] for probed
/// non-square pairs (multiples of 32 through 2048), structural
/// fallback otherwise.
fn conv_ns_params(cin: usize, cout: usize, kern_sz: usize) -> (u32, usize, bool) {
    if cin == cout {
        return conv_chan_params(cin, kern_sz);
    }
    if cin % 32 == 0 && cout % 32 == 0 && cin <= 2048 && cout <= 2048 {
        let v = crate::conv_ns_table::CONV_NS_TABLE[(cin / 32 - 1) * 64 + (cout / 32 - 1)];
        if v != u32::MAX {
            return (v & 0xfff, ((v >> 12) & 0xf) as usize, v & 0x10000 != 0);
        }
    }
    conv_ns_fallback(cin, cout, kern_sz)
}

/// Emit `z = conv1x1(x, w)` — `[1,C,1,1]` in, `[C,C,1,1]` weight — for
/// arbitrary `C >= 32`. Fully parametric: segment VAs, section sizes,
/// the 16-lane table, `LC_SYMTAB`, strtab, relocation and the three ANE
/// command blobs are all generated from `c` and `weights`; the two
/// `__text` encoding templates are the only bytes carried verbatim.
///
/// Field model recovered from a dense ANEC channel sweep
/// (`/tmp/cdense`+`/tmp/cdense2`, every `C` in 32..=4096):
/// - `__const`/`__data` slots are `align(C*64, 0x4000)` each; `__TEXT`
///   follows at `+2*slot`, `__KERN_0` at `text_va + 0x8000`.
/// - `__KERN_0` payload is [`weight_layout`]'s 16-lane stream; the text
///   lane table is the 15 lane-start offsets + 16 padded lane sizes.
/// - `f14` (text `0x14`, blob `0xbe8`), the `0x158`/`0x168` row pad and
///   the `0x124` big-kernel flag are ANEC scheduler internals — no
///   closed form survived fitting; they come from
///   [`crate::conv_chan_table::CONV_CHAN_TABLE`] inside the probed
///   range, with a documented `ceil(kern/0xC000)` fallback beyond it.
/// - `f128` (text `0x128`) buckets `ceil(C/16)` to 1/2/3/4/5.
/// - `C in 713..=720` uses a different 59-symbol program plan and is
///   rejected rather than mis-emitted.
/// - Blob fields: `C` at `0xc24`,`0x196c`; `C*64` at `0xc48`,`0xc68`,
///   `0xc78`,`0x1990`,`0x19b0`,`0x19c0`; VA copies at `0x348`/`0x3c8`
///   (`__data`), `0x3d8` (`__KERN_0`), `0x368`/`0xb68` (`__TEXT`),
///   `0x378` (const); `0xb7c` is a per-regime marker (0x74 std, 0x73
///   narrow).
pub fn emit_conv1x1_dyn(
    c: usize,
    weights: &[u16],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    emit_conv1x1_ns(c, c, weights, mil_path, out_path)
}

/// Non-square form: `z = conv1x1(x, w)` with x `[1,Cin,1,1]`, weight
/// `[Cout,Cin,1,1]`, z `[1,Cout,1,1]`. Same program plan as the square
/// emitter — every field the square code keyed on `C` splits cleanly:
/// input-side descriptors (`0x10c`, `0xc24`, the `0xc48` group, the
/// `0x158` `*16` group, `__const` size) take `cin`; output-side
/// descriptors (`0x11c`, `0x196c`, the `0x1990` group, `__data` size)
/// and the text regime (`f128`, narrow) take `cout`; the lane table
/// and kernel payload come from [`weight_layout_ns`]; the kernel name
/// from [`conv_kernel_name_ns`]. No `(cin,cout)` pair probed — including
/// the square-variant range 713..=720 in either dim — produced a
/// non-43-symbol plan, so the variant window only applies on the
/// diagonal. Requires `cin >= 32` and `cout >= 32`.
pub fn emit_conv1x1_ns(
    cin: usize,
    cout: usize,
    weights: &[u16],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    emit_conv1x1_w(cin, cout, 1, weights, mil_path, out_path)
}

/// Spatial form: `z = conv1x1(x, w)` with x `[1,Cin,1,W]` →
/// `z [1,Cout,1,W]`. W scaling (decoded from an ANEC W-sweep at
/// fixed C=64 plus non-square and narrow probes):
/// - DRAM per-channel row stride `stride64 = align(W*2, 64)`; SRAM
///   row stride `stride16 = align(W*2, 16)`.
/// - in/out section bytes `= stride64 * C` (replaces `C*64` — the
///   W=1 case, since `align(2,64) == 64`); segment vmsizes page-round.
/// - Descriptor blobs: `W` at `0xc2c`/`0x1974`, `stride64` at
///   `0xc50`/`0xc58`/`0x1998`/`0x19a0`, section bytes at the `0xc48`
///   / `0x1990` groups.
/// - Task record: `W` at `0x104`/`0x114`, `stride64` at `0x138`
///   (`0x134` narrow) and `0x17c` (`0x178` narrow), `stride16` at
///   `0x154` (`0x150`), `(cin+pad16)*stride16` at the `cx` group.
/// - `W > 64` clears the task flag's low bits (`0x200004` →
///   `0x200000` at `0x124`).
///
/// Since the input is channel-major `[Cin, W]`, `W` is the sequence/
/// batch dim: `conv1x1(x=[K,M], w[K,N])` computes `x[m] @ w^T`
/// per position — the M>1 matvec ANEC's own matmul template gets a
/// different (undecoded) record for. Requires `cin >= 32`,
/// `cout >= 32`.
pub fn emit_conv1x1_w(
    cin: usize,
    cout: usize,
    w: usize,
    weights: &[u16],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    if cin < 32 || cout < 32 {
        return Err(format!(
            "conv1x1 dyn: (Cin={cin},Cout={cout}) below the C>=32 template regimes"
        ));
    }
    if weights.len() < cin * cout {
        return Err(format!(
            "expected {} weights, got {}",
            cin * cout,
            weights.len()
        ));
    }
    let (stream, lane_off, lane_sz) = weight_layout_ns(&weights[..cin * cout], cin, cout);
    let kern_sz = stream.len() * 2;
    let q = cout.div_ceil(16);
    let narrow = (5..=8).contains(&q);
    let textlen: usize = if narrow { 0x1cc } else { 0x1d0 };
    // scheduler scalars — ANEC's planning heuristic is table-driven
    // inside the probed range; outside it falls back to the nearest
    // structural approximation (kern chunk count, no pad, big flag by
    // the 1 MiB boundary the window implies).
    let (f14, pad16, big) = conv_ns_params(cin, cout, kern_sz);
    if f14 == u32::MAX {
        return Err(format!(
            "conv1x1 dyn: C={cin} is in the 59-symbol variant window (713..=720) — different program plan"
        ));
    }
    let f128 = match q {
        ..=2 => 1u32,
        3 | 4 => 2,
        5..=8 => 3,
        9..=16 => 4,
        _ => 5,
    };
    // W-scaled strides: DRAM rows 64-byte aligned, SRAM rows 16-byte.
    let stride64 = (w * 2 + 63) & !63;
    let stride16 = (w * 2 + 15) & !15;
    let in_sz = stride64 * cin;
    let out_sz = stride64 * cout;
    let slot_in = (in_sz + 0x3fff) & !0x3fff;
    let slot_out = (out_sz + 0x3fff) & !0x3fff;
    let const_va = 0x3000_0000u64;
    let data_va = const_va + slot_in as u64;
    let text_va = data_va + slot_out as u64;
    let const2_va = text_va + textlen.div_ceil(0x40) as u64 * 0x40;
    let kern_va = text_va + 0x8000;
    let kern_vsz = (kern_sz + 0x3fff) & !0x3fff;

    // ---- header + segments + command blobs -------------------------
    let mut pre = SKELETON_CONV32[..CONV_TAIL_OFF].to_vec();
    put_u64(&mut pre, 0x88, slot_in as u64); // __const seg vmsize
    put_u64(&mut pre, 0xd0, const_va); // __const sect addr
    put_u64(&mut pre, 0xd8, in_sz as u64); // __const sect size
    put_u64(&mut pre, 0x118, data_va); // __data seg vmaddr
    put_u64(&mut pre, 0x120, slot_out as u64); // __data seg vmsize
    put_u64(&mut pre, 0x168, data_va); // __data sect addr
    put_u64(&mut pre, 0x170, out_sz as u64); // __data sect size
    put_u64(&mut pre, 0x1b0, text_va); // __TEXT seg vmaddr
    put_u64(&mut pre, 0x200, text_va); // __text sect addr
    put_u64(&mut pre, 0x208, textlen as u64); // __text sect size
    put_u64(&mut pre, 0x250, const2_va); // __TEXT __const sect addr
    put_u64(&mut pre, 0x298, kern_va); // __KERN_0 seg vmaddr
    put_u64(&mut pre, 0x2a0, kern_vsz as u64); // __KERN_0 seg vmsize
    put_u64(&mut pre, 0x2b0, kern_vsz as u64); // __KERN_0 seg filesize
    put_u64(&mut pre, 0x2e8, kern_va); // __kern_0 sect addr
    put_u64(&mut pre, 0x2f0, kern_sz as u64); // __kern_0 sect size

    // 0x40-cmd 2 @0x338 and cmd4 blob 1 @0x358 carry copies of the
    // section VAs; blob 2 @0xbf8 repeats the text VA
    put_u64(&mut pre, 0x348, data_va);
    put_u64(&mut pre, 0x3c8, data_va);
    put_u64(&mut pre, 0x3d8, kern_va);
    put_u32(&mut pre, 0x368, text_va as u32);
    put_u32(&mut pre, 0xb68, text_va as u32);
    put_u32(&mut pre, 0x378, const2_va as u32);
    put_u32(&mut pre, 0xb7c, if narrow { 0x73 } else { 0x74 });
    put_u32(&mut pre, 0xbe8, f14);
    // cmd4 blobs 2 @0xbf8 / 3 @0x1940: input/output dims + section byte
    // sizes — the 0xc?? group is the x-side (stride64*cin + row stride),
    // 0x19?? the z-side
    put_u32(&mut pre, 0xc24, cin as u32);
    put_u32(&mut pre, 0xc2c, w as u32);
    put_u32(&mut pre, 0x196c, cout as u32);
    put_u32(&mut pre, 0x1974, w as u32);
    for &o in &[0xc50usize, 0xc58, 0x1998, 0x19a0] {
        put_u32(&mut pre, o, stride64 as u32);
    }
    for &o in &[0xc48usize, 0xc68, 0xc78] {
        put_u32(&mut pre, o, in_sz as u32);
    }
    for &o in &[0x1990usize, 0x19b0, 0x19c0] {
        put_u32(&mut pre, o, out_sz as u32);
    }

    // ---- tail: signature, rt.version, LC_SYMTAB, strtab, reloc -----
    let kname = conv_kernel_name_ns(cin as u64, cout as u64, 0x80);
    let mut entries: Vec<(String, u8, u8, u16, u64)> = Vec::with_capacity(43);
    entries.push(("x".into(), 0x0f, 1, 0, const_va));
    entries.push(("z".into(), 0x0f, 2, 0, data_va));
    for i in 0..16 {
        entries.push((
            format!("{kname}_ne_{i}"),
            0x0f,
            5,
            if i == 0 { 2 } else { 0 },
            kern_va + lane_off[i] as u64,
        ));
    }
    for ty in TYPE_SYMS {
        entries.push(((*ty).into(), 0x80, 0, 0, 0));
    }
    entries.push((
        layout_desc_w("x", 24, cin as u32, w as u32),
        0x20,
        0,
        0x18,
        0,
    ));
    entries.push((
        layout_desc_w("z", 29, cout as u32, w as u32),
        0x20,
        0,
        0x1d,
        0,
    ));
    // one __text relocation pointing at the kernel reference word —
    // 0x1a0 in the std template, 0x19c in the narrower one
    let reloc_addr: u32 = if narrow { 0x19c } else { 0x1a0 };
    let relocs: Vec<[u8; 8]> = vec![[
        reloc_addr.to_le_bytes()[0],
        reloc_addr.to_le_bytes()[1],
        reloc_addr.to_le_bytes()[2],
        reloc_addr.to_le_bytes()[3],
        0x05,
        0,
        0,
        0x07,
    ]];

    let (mut img, text_off) = build_tail_at_reloc(
        &pre,
        CONV_TAIL_OFF,
        &entries,
        &relocs,
        mil_path,
        out_path,
        0x218,
    );
    // file offsets slide if the tail overflows 0x4000
    put_u32(&mut img, 0x1c0, text_off as u32); // __TEXT seg fileoff
    put_u32(&mut img, 0x210, text_off as u32); // __text sect offset
    let const2_fo = text_off + textlen.div_ceil(0x40) * 0x40;
    put_u32(&mut img, 0x260, const2_fo as u32); // __TEXT __const offset
    let kern_fo = text_off + 0x8000;
    put_u64(&mut img, 0x2a8, kern_fo as u64); // __KERN_0 seg fileoff
    put_u32(&mut img, 0x2f8, kern_fo as u32); // __kern_0 sect offset

    // ---- __text ----------------------------------------------------
    let mut text = if narrow {
        DYN_TEXT_NARROW.to_vec()
    } else {
        SKELETON_CONV32[0x4000..0x4000 + 0x1d0].to_vec()
    };
    put_u32(&mut text, 0x14, f14);
    for i in 0..15 {
        put_u32(&mut text, 0x84 + i * 4, lane_off[i + 1] as u32);
    }
    for i in 0..16 {
        put_u32(&mut text, 0xc0 + i * 4, lane_sz[i] as u32);
    }
    // spatial W: position count at 0x104/0x114, DRAM stride at
    // 0x138/0x17c (narrow: 0x134/0x178), SRAM stride at 0x154
    // (narrow: 0x150)
    put_u32(&mut text, 0x104, w as u32);
    put_u32(&mut text, 0x114, w as u32);
    put_u32(&mut text, 0x10c, cin as u32);
    put_u32(&mut text, 0x11c, cout as u32);
    let (s64a, s16, s64b) = if narrow {
        (0x134usize, 0x150usize, 0x178usize)
    } else {
        (0x138, 0x154, 0x17c)
    };
    put_u32(&mut text, s64a, stride64 as u32);
    put_u32(&mut text, s16, stride16 as u32);
    put_u32(&mut text, s64b, stride64 as u32);
    if !narrow {
        // W > 64 drops the flag's low nibble (0x200004 -> 0x200000)
        let flag = if w > 64 {
            0x200000
        } else if big {
            0x244404
        } else {
            0x200004
        };
        put_u32(&mut text, 0x124, flag);
        put_u32(&mut text, 0x128, f128);
    }
    let cx: usize = if narrow { 0x154 } else { 0x158 };
    // first and last of the four copies carry the padded input dim
    // scaled by the SRAM row stride
    put_u32(&mut text, cx, ((cin + pad16) * stride16) as u32);
    put_u32(&mut text, cx + 4, (cin * stride16) as u32);
    put_u32(&mut text, cx + 8, (cin * stride16) as u32);
    put_u32(&mut text, cx + 0x10, ((cin + pad16) * stride16) as u32);
    debug_assert_eq!(text.len(), textlen);

    // ---- section contents ------------------------------------------
    img.resize(text_off, 0);
    img.extend_from_slice(&text);
    img.resize(kern_fo, 0);
    for v in &stream {
        img.extend_from_slice(&v.to_le_bytes());
    }
    img.resize(kern_fo + kern_vsz, 0);
    Ok(img)
}

/// Non-square const-weight matmul skeleton: `matmul(x, w, transpose_y)`
/// with x `[1,1,1,K]`, w `[1,N,1,K]` → z `[1,1,1,N]` (the M=1
/// vector-matrix form transformer projections lower to). Same
/// 43-symbol program envelope as the conv family — the `__KERN_0`
/// stream is [`weight_layout_ns`]`(w, K, N)` byte-for-byte — but a
/// distinct `__text` record (0x24c) and kernel-name record (383 B).
const SKELETON_MM_NS: &[u8] = include_bytes!("../assets/mmns64x32_reference.hwx");

/// Narrow-output `__text` template (`ceil(N/16) in 5..=8`): the std
/// record minus the `0x1d8` f128 word (0x248 bytes), with the regime
/// marker `0x91418005` at `0x1c0`.
const MM_NS_TEXT_NARROW: &[u8] = include_bytes!("../assets/mm_ns_text_narrow.bin");

/// `(f14, pad, big)` scheduler scalars for the matmul plan: tabled
/// over the swept multiples-of-32 grid in
/// [`crate::mm_ns_table::MM_NS_TABLE`], structural fallback off-grid.
/// No `*16` input pad has been observed for this family (the text
/// `0x88` group is always `K*16`), so the table stores `f14` and the
/// big-kernel flag only.
fn mm_ns_params(k: usize, n: usize, kern_sz: usize) -> (u32, bool) {
    if k % 32 == 0 && n % 32 == 0 && k <= 2048 && n <= 2048 {
        let v = crate::mm_ns_table::MM_NS_TABLE[(k / 32 - 1) * 64 + (n / 32 - 1)];
        if v != u32::MAX {
            return (v & 0xfff, v & 0x10000 != 0);
        }
    }
    // fallback: f14 counts the kernel's full 0xC000 chunks (ANEC's
    // exact rounding is irregular — same heuristic class as conv).
    (
        kern_sz.saturating_sub(1) as u32 / 0xc000,
        kern_sz > 0x100000,
    )
}

/// ANEC's matmul kernel symbol name for a const weight `[1,N,1,K]`
/// at `@model_path/weights/weight.bin`: SHA-256 over a 383-byte record
/// — blob stem/file name/offset/size, a `desc` block carrying the
/// weight's `[N,K]` dims in the conv `[cout,cin,1,1]` arrangement,
/// then a second blob descriptor and a metadata block whose strides
/// scale with `K`,`N` (`N*256`,`K*256`,`K*N*256`), ending with the
/// per-lane row count `min(ceil(N/16),32)` packed `<<40`.
/// Verified against ANEC symbols at (64,32),(128,64),(256,64),
/// (64,256),(64,65),(64,513),(64,1024).
pub fn mm_kernel_name_ns(k: u64, n: u64, data_offset: u64) -> String {
    let mut s: Vec<u8> = Vec::with_capacity(383);
    s.extend_from_slice(b"weight");
    s.extend_from_slice(b"weight.bin0");
    s.extend_from_slice(&data_offset.to_le_bytes());
    s.extend_from_slice(&(n * k * 2).to_le_bytes());
    s.extend_from_slice(&[0; 5]);
    s.extend_from_slice(b"desc");
    s.extend_from_slice(&[0; 40]);
    for v in [n, k, 1, 1, 1, k, 1, 1, 1, 1, 0, 0] {
        s.extend_from_slice(&v.to_le_bytes());
    }
    for v in [1u32, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0] {
        s.extend_from_slice(&v.to_le_bytes());
    }
    s.extend_from_slice(b"weight.bin0");
    s.extend_from_slice(&data_offset.to_le_bytes());
    s.extend_from_slice(&(n * k * 2).to_le_bytes());
    for v in [
        0x10fu64,
        0x100,
        0x100,
        n * 256,
        k * 256,
        n * 256,
        k * 256,
        0x100,
        0x100,
        0x100,
        n * k * 256,
        n * k * 256,
        k * 256,
        k * 256,
        0x100,
        n.div_ceil(16).min(32) << 40,
    ] {
        s.extend_from_slice(&v.to_le_bytes());
    }
    s.extend_from_slice(&[0; 6]);
    debug_assert_eq!(s.len(), 383);
    format!("K{}", mil_spec::sha256::sha256_hex(&s).to_uppercase())
}

/// Emit `z = matmul(x, w, transpose_y)` for x `[1,1,1,K]`, const
/// weight `[1,N,1,K]` → z `[1,1,1,N]` — the M=1 non-square form every
/// transformer FFN/attention projection needs. `weights` is the raw
/// row-major `[N,K]` blob payload.
///
/// Field model (dense ANEC sweep over the `(K,N)` grid,
/// `/tmp/mmgrid`, plus per-shape probes):
/// - `__const` slot is `align(K*2, 0x4000)`, `__data` `align(N*64)`;
///   `__KERN_0` payload = [`weight_layout_ns`]`(w, K, N)` verbatim.
/// - `__text` is a fixed 0x24c record (0x248 in the narrow regime
///   `ceil(N/16) in 5..=8`, which drops the `0x1d8` f128 word and
///   shifts the tail down one u32): `K` at `0x40`/`0x50`/`0x1bc`,
///   `K*16` at `0x88`, `K*2` at `0x8c..0x98`, `N` at `0x1cc`, the
///   15 lane offsets at `0x134`, 16 lane sizes at `0x170`, the big
///   flag at `0x1d4`, `K*18` at `0x1f4` (`0x1f0` narrow).
/// - Blobs: `1` (M) at `0xc24`, `K` at `0xc2c`, `K*2` at the `0xc48`
///   group; `N` at `0x196c`, `N*64` at the `0x1990` group; regime
///   marker `0x93`/`0x92` at `0xb7c`; `f14` at `0xbe8`.
/// - Only the M=1 form is emitted — ANEC rejects `M>1` non-square
///   matmuls (status 1), and M=N squares go through [`emit_matmul`].
pub fn emit_matmul_ns(
    k: usize,
    n: usize,
    weights: &[u16],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    if k < 32 || n < 32 {
        return Err(format!(
            "matmul ns: (K={k},N={n}) below the probed template floor"
        ));
    }
    if weights.len() < k * n {
        return Err(format!("expected {} weights, got {}", k * n, weights.len()));
    }
    let (stream, lane_off, lane_sz) = weight_layout_ns(&weights[..k * n], k, n);
    let kern_sz = stream.len() * 2;
    let narrow = (5..=8).contains(&n.div_ceil(16));
    let textlen: usize = if narrow { 0x248 } else { 0x24c };
    let (f14, big) = mm_ns_params(k, n, kern_sz);
    if f14 == u32::MAX {
        return Err(format!("matmul ns: (K={k},N={n}) hit a variant window"));
    }
    let f128 = match n.div_ceil(16) {
        ..=2 => 1u32,
        3 | 4 => 2,
        9..=16 => 4,
        _ => 5,
    };
    // __const payload is the K-element input padded to 0x40; the text
    // stride group uses align(K*2, 0x10), and 0x90 additionally adds a
    // DMA guard when K*2 lands on a 0x200 boundary (0x10, or 0x20 at
    // 0x400 boundaries — ANEC cache-set spacing).
    let const_sz = (k * 2 + 0x3f) & !0x3f;
    let stride16 = (k * 2 + 0xf) & !0xf;
    let stride_g = if (k * 2) % 0x200 == 0 {
        stride16 + 0x10 + usize::from((k * 2) % 0x400 == 0) * 0x10
    } else {
        stride16
    };
    let slot_in = (const_sz + 0x3fff) & !0x3fff;
    let slot_out = (n * 64 + 0x3fff) & !0x3fff;
    let const_va = 0x3000_0000u64;
    let data_va = const_va + slot_in as u64;
    let text_va = data_va + slot_out as u64;
    let const2_va = text_va + textlen.div_ceil(0x40) as u64 * 0x40;
    let kern_va = text_va + 0x8000;
    let kern_vsz = (kern_sz + 0x3fff) & !0x3fff;

    // ---- header + segments + command blobs -------------------------
    let mut pre = SKELETON_MM_NS[..CONV_TAIL_OFF].to_vec();
    put_u64(&mut pre, 0x88, slot_in as u64); // __const seg vmsize
    put_u64(&mut pre, 0xd0, const_va); // __const sect addr
    put_u64(&mut pre, 0xd8, const_sz as u64); // __const sect size
    put_u64(&mut pre, 0x118, data_va); // __data seg vmaddr
    put_u64(&mut pre, 0x120, slot_out as u64); // __data seg vmsize
    put_u64(&mut pre, 0x168, data_va); // __data sect addr
    put_u64(&mut pre, 0x170, (n * 64) as u64); // __data sect size
    put_u64(&mut pre, 0x1b0, text_va); // __TEXT seg vmaddr
    put_u64(&mut pre, 0x200, text_va); // __text sect addr
    put_u64(&mut pre, 0x208, textlen as u64); // __text sect size
    put_u64(&mut pre, 0x250, const2_va); // __TEXT __const sect addr
    put_u64(&mut pre, 0x298, kern_va); // __KERN_0 seg vmaddr
    put_u64(&mut pre, 0x2a0, kern_vsz as u64); // __KERN_0 seg vmsize
    put_u64(&mut pre, 0x2b0, kern_vsz as u64); // __KERN_0 seg filesize
    put_u64(&mut pre, 0x2e8, kern_va); // __kern_0 sect addr
    put_u64(&mut pre, 0x2f0, kern_sz as u64); // __kern_0 sect size

    put_u64(&mut pre, 0x348, data_va);
    put_u64(&mut pre, 0x3c8, data_va);
    put_u64(&mut pre, 0x3d8, kern_va);
    put_u32(&mut pre, 0x368, text_va as u32);
    put_u32(&mut pre, 0xb68, text_va as u32);
    put_u32(&mut pre, 0x378, const2_va as u32);
    put_u32(&mut pre, 0xb7c, if narrow { 0x92 } else { 0x93 });
    put_u32(&mut pre, 0xbe8, f14);
    // x-side descriptors: M at 0xc24, K at 0xc2c, padded input byte
    // size (align(K*2, 0x40)) at the 0xc48 group
    put_u32(&mut pre, 0xc24, 1);
    put_u32(&mut pre, 0xc2c, k as u32);
    for &o in &[0xc48usize, 0xc50, 0xc58, 0xc68, 0xc78] {
        put_u32(&mut pre, o, const_sz as u32);
    }
    // z-side: N and N*64 (same layout as conv's Cout group)
    put_u32(&mut pre, 0x196c, n as u32);
    for &o in &[0x1990usize, 0x19b0, 0x19c0] {
        put_u32(&mut pre, o, (n * 64) as u32);
    }

    // ---- tail: signature, rt.version, LC_SYMTAB, strtab, reloc -----
    let kname = mm_kernel_name_ns(k as u64, n as u64, 0x80);
    let mut entries: Vec<(String, u8, u8, u16, u64)> = Vec::with_capacity(43);
    entries.push(("x".into(), 0x0f, 1, 0, const_va));
    entries.push(("z".into(), 0x0f, 2, 0, data_va));
    for i in 0..16 {
        entries.push((
            format!("{kname}_ne_{i}"),
            0x0f,
            5,
            if i == 0 { 2 } else { 0 },
            kern_va + lane_off[i] as u64,
        ));
    }
    for ty in TYPE_SYMS {
        entries.push(((*ty).into(), 0x80, 0, 0, 0));
    }
    // x layout: the three s-fields carry the padded input byte size
    // (align(K*2, 0x40)); the raw element count lands on `h` only.
    let xdesc = format!(
        "x:t24=ar1;0;1;25=s{}n:ar1;0;1;26=s{}c:ar1;0;1;27=s{}h:ar1;0;{};28=s2w:5",
        const_sz, const_sz, const_sz, k
    );
    entries.push((xdesc, 0x20, 0, 0x18, 0));
    entries.push((layout_desc("z", 29, n as u32), 0x20, 0, 0x1d, 0));
    let reloc_addr: u32 = if narrow { 0x22c } else { 0x230 };
    let relocs: Vec<[u8; 8]> = vec![[
        reloc_addr.to_le_bytes()[0],
        reloc_addr.to_le_bytes()[1],
        reloc_addr.to_le_bytes()[2],
        reloc_addr.to_le_bytes()[3],
        0x05,
        0,
        0,
        0x07,
    ]];

    let (mut img, text_off) = build_tail_at_reloc(
        &pre,
        CONV_TAIL_OFF,
        &entries,
        &relocs,
        mil_path,
        out_path,
        0x218,
    );
    put_u32(&mut img, 0x1c0, text_off as u32); // __TEXT seg fileoff
    put_u32(&mut img, 0x210, text_off as u32); // __text sect offset
    let const2_fo = text_off + textlen.div_ceil(0x40) * 0x40;
    put_u32(&mut img, 0x260, const2_fo as u32); // __TEXT __const offset
    let kern_fo = text_off + 0x8000;
    put_u64(&mut img, 0x2a8, kern_fo as u64); // __KERN_0 seg fileoff
    put_u32(&mut img, 0x2f8, kern_fo as u32); // __kern_0 sect offset

    // ---- __text ----------------------------------------------------
    // the narrow template is the std record minus the f128 word at
    // 0x1d8 — all tail fields sit one u32 lower.
    let shift = |o: usize| if narrow && o >= 0x1dc { o - 4 } else { o };
    let mut text = if narrow {
        MM_NS_TEXT_NARROW.to_vec()
    } else {
        SKELETON_MM_NS[0x4000..0x4000 + 0x24c].to_vec()
    };
    // f14 lands at text 0xc4 (and blob 0xbe8) — text 0x14 stays 0 for
    // this family.
    put_u32(&mut text, 0xc4, f14);
    put_u32(&mut text, 0x40, k as u32);
    put_u32(&mut text, 0x50, k as u32);
    put_u32(&mut text, 0x88, (k * 16) as u32);
    for &o in &[0x8cusize, 0x94, 0x98] {
        put_u32(&mut text, o, stride16 as u32);
    }
    put_u32(&mut text, 0x90, stride_g as u32);
    for i in 0..15 {
        put_u32(&mut text, 0x134 + i * 4, lane_off[i + 1] as u32);
    }
    for i in 0..16 {
        put_u32(&mut text, 0x170 + i * 4, lane_sz[i] as u32);
    }
    put_u32(&mut text, 0x1bc, k as u32);
    put_u32(&mut text, 0x1cc, n as u32);
    put_u32(&mut text, 0x1d4, if big { 0x244404 } else { 0x200004 });
    if !narrow {
        put_u32(&mut text, 0x1d8, f128);
    }
    put_u32(
        &mut text,
        shift(0x1f4),
        ((k * 18 + 0xf) & !0xf) as u32 + (stride_g - stride16) as u32,
    );
    debug_assert_eq!(text.len(), textlen);

    // ---- section contents ------------------------------------------
    img.resize(text_off, 0);
    img.extend_from_slice(&text);
    img.resize(kern_fo, 0);
    for v in &stream {
        img.extend_from_slice(&v.to_le_bytes());
    }
    img.resize(kern_fo + kern_vsz, 0);
    Ok(img)
}

// ------------------------------------------------------------------
// Multi-task composition: conv1x1 chains
// ------------------------------------------------------------------
//
// ANEC compiles a chain of K independent conv1x1 ops into ONE Mach-O:
// a single `__KERN_0` holding the concatenated per-task weight streams
// and a `__TEXT.__text` task table of per-task records (0x30-byte
// header, then first 0x190 / middle 0x160 / last 0x170-byte records —
// the i=1 record forms are one u32 shorter). The intermediate buffers
// use a small fixed set of descriptor ids: `0x55` for the first
// intermediate, `0x56` for middle ones, and `0x64` (`0x63` when it is
// the only intermediate) for the buffer feeding the last task, with
// `0xfff820`/`0xfff860`+`0x04000000` tail variants for the same split.
// Everything else is the single-op machinery: same 13-command
// envelope, `x`/output binds, per-task kernel families in the symtab
// (each task's name = [`conv_kernel_name_ns`] over its own blob data
// offset), `nreloc = K` pointing at each record's kernel-offset word.
//
// Templates are sliced from the captured `c64k4`/`c64k2` corpus; the
// emitter patches every task-index/chain-length-dependent field.

/// `c64k4` reference: `__text` at file offset `0x8000`, size `0x5f0`.
const SKELETON_CHAIN4: &[u8] = include_bytes!("../assets/chain/c64k4.hwx");

/// `__text` task-table header (`+0x2c` carries the multi-task flag).
const CHAIN4_TEXT: usize = 0x8000;
/// The signature/argv load command begins at `0x2690`; everything
/// before it is the command prefix (`pre`) reused verbatim.
const CHAIN_PRE_OFF: usize = 0x2690;

/// Narrow-channel regime: `f128 = ceil(cout/16)` in `5..=8`. Narrow-cout
/// task records drop the f128 word (everything below `+0xf8` shifts up
/// one u32) and carry the `0x91418005` marker instead of `0x93418005`.
fn chain_narrow(cout: usize) -> bool {
    (65..=128).contains(&cout)
}

/// Emit a complete HWX image for a chain of `conv1x1` tasks where
/// task `i` consumes task `i-1`'s output — the multi-task analogue of
/// [`emit_conv1x1_ns`]. `shapes[i] = (cin, cout)`; `weights` holds
/// each task's `[cin,cout]` fp16 matrix concatenated; `data_offs[i]`
/// is task `i`'s weight payload offset inside the referenced
/// `weights/weight.bin` (hashed into its kernel family name).
/// `out_name` is the network output symbol (`y{k}` in the corpus).
///
/// Verified byte-for-byte against ANEC for same-shape chains
/// (C=64 at depths 2..6,8) and the mixed `64→32→64` pair.
pub fn emit_conv_chain(
    shapes: &[(usize, usize)],
    weights: &[u16],
    data_offs: &[u64],
    out_name: &str,
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    let k = shapes.len();
    if k < 2 {
        return Err("chain emitter needs >=2 tasks".into());
    }
    if data_offs.len() != k {
        return Err("data_offs must carry one entry per task".into());
    }
    let need: usize = shapes.iter().map(|&(ci, co)| ci * co).sum();
    if weights.len() < need {
        return Err(format!("expected {need} weights, got {}", weights.len()));
    }

    // per-task kernel stream + lane tables
    let mut streams: Vec<Vec<u16>> = Vec::with_capacity(k);
    let mut lane_offs: Vec<Vec<usize>> = Vec::with_capacity(k);
    let mut lane_szs: Vec<Vec<usize>> = Vec::with_capacity(k);
    let mut kern_off: Vec<usize> = Vec::with_capacity(k);
    let mut off = 0usize;
    let mut wcur = 0usize;
    for &(ci, co) in shapes {
        kern_off.push(off);
        let (s, lo, ls) = weight_layout_ns(&weights[wcur..wcur + ci * co], ci, co);
        off += s.len() * 2;
        wcur += ci * co;
        streams.push(s);
        lane_offs.push(lo);
        lane_szs.push(ls);
    }
    let kern_bytes = off;
    // Network I/O channel counts drive the FVMLIB buffer sizes and the
    // VA cascade: each FVMLIB segment is max(0x4000, c*0x40) and every
    // downstream section base follows.
    let c_in = shapes[0].0 as u32;
    let c_out = shapes[k - 1].1 as u32;
    let fv_in = (c_in * 0x40).max(0x4000) as u64;
    let fv_out = (c_out * 0x40).max(0x4000) as u64;
    let out_va = 0x3000_0000u64 + fv_in;
    let text_va = out_va + fv_out;
    let kern_va = text_va + 0x8000;
    // per-descriptor page-count word (desc block +4 and blob 0xbe8):
    // only nonzero once the input FVMLIB region spans a full 0x4000
    // page (c_in * 0x40 >= 0x4000, i.e. c_in >= 256).
    let f14 = c_in * 0x40 / 0x4000;
    let desc_flag = |cout: usize| -> u32 {
        if f14 != 0 {
            (cout as u32 * 0x40).div_ceil(0x4000)
        } else {
            0
        }
    };
    let f128_of = |cout: usize| -> u32 {
        match cout.div_ceil(16) {
            ..=2 => 1,
            3 | 4 => 2,
            9..=16 => 4,
            _ => 5,
        }
    };

    // ---- intermediate buffer pool -----------------------------------
    // Buffer `j` is task j's output (y_{j+1}), live during tasks
    // {j, j+1}; slot size = cout*16. If the intermediates are not
    // already in non-increasing size order, ANEC packs them into a
    // descending-size arena; otherwise offsets are plain cumulative
    // (no reuse). Reuse is single-slot: a buffer joins the *most
    // recently created* slot iff every tenant is liveness-disjoint
    // (|m - b| > 1); otherwise a new slot is carved at the arena end.
    let n_int = k - 1;
    let isz: Vec<usize> = (0..n_int).map(|j| shapes[j].1 * 16).collect();
    let mut order: Vec<usize> = (0..n_int).collect();
    order.sort_by(|&a, &b| isz[b].cmp(&isz[a]));
    let reordered = order.iter().enumerate().any(|(p, &b)| p != b);
    let mut pool_off = vec![0usize; n_int];
    // whether a buffer's slot was created after the pool's first reuse
    // event (controls the `0x120` word in long input descriptors on
    // buffers after y1).
    let mut slot_after_reuse = vec![false; n_int];
    if !reordered {
        let mut acc = 0usize;
        for j in 0..n_int {
            pool_off[j] = acc;
            acc += isz[j];
        }
    } else {
        // (offset, created_post_reuse, tenants)
        let mut slots: Vec<(usize, bool, Vec<usize>)> = Vec::new();
        let mut end = 0usize;
        let mut reused = false;
        for &b in &order {
            let mut hit = false;
            if let Some((so, created_post, occ)) = slots.last_mut() {
                if occ.iter().all(|&m| usize::abs_diff(m, b) > 1) {
                    pool_off[b] = *so;
                    slot_after_reuse[b] = *created_post;
                    occ.push(b);
                    hit = true;
                }
            }
            if !hit {
                pool_off[b] = end;
                end += isz[b];
                slot_after_reuse[b] = reused;
                slots.push((end - isz[b], reused, vec![b]));
            }
            if !reused && slots.last().is_some_and(|s| s.2.len() > 1) {
                reused = true;
            }
        }
    }
    // ---- __text: header + task records ------------------------------
    // The task-table header's +0x10 hi16 encodes the x descriptor id.
    // Empirically (all corpus chains): 0x62 when the first intermediate
    // is narrow-class and no larger than... over (y1, y2):
    //   narrow(y1): y2 <= y1 -> 0x62, y2 > y1 -> 0x63
    //   else:       y2 >  y1 -> 0x64, else    -> 0x63
    // k=2 has a single intermediate: 0x62 iff narrow(y1).
    let hdr10: u32 = {
        let y1 = shapes[0].1;
        if k == 2 {
            if chain_narrow(y1) {
                0x62
            } else {
                0x63
            }
        } else {
            let y2 = shapes[1].1;
            if chain_narrow(y1) {
                if y2 <= y1 {
                    0x62
                } else {
                    0x63
                }
            } else if y2 > y1 {
                0x64
            } else {
                0x63
            }
        }
    };
    let mut text = SKELETON_CHAIN4[CHAIN4_TEXT..CHAIN4_TEXT + 0x30].to_vec();
    put_u32(&mut text, 0x10, hdr10 << 16);
    put_u32(&mut text, 0x14, f14);
    put_u32(&mut text, 0x2c, 1); // multi-task flag
    let head = &SKELETON_CHAIN4[CHAIN4_TEXT + 0x1c0..CHAIN4_TEXT + 0x1c0 + 0xf8];
    let mut relocs: Vec<[u8; 8]> = Vec::with_capacity(k);
    let mut rec_base = 0x30usize;
    let push = |rec: &mut Vec<u8>, v: u32| rec.extend_from_slice(&v.to_le_bytes());
    for i in 0..k {
        let (ci, co) = shapes[i];
        let last = i == k - 1;
        let narrow = chain_narrow(co);
        let feeder = i + 1 == k - 1;
        // input buffer y_i = intermediate i-1; output y_{i+1} = i.
        let in_off = if i == 0 { 0 } else { pool_off[i - 1] };
        let out_off = if last { 0 } else { pool_off[i] };

        let mut rec = head.to_vec();
        put_u32(&mut rec, 0x00, if last { 0x0005_0009 } else { 0x0005_0001 });
        put_u32(
            &mut rec,
            0x08,
            0x10240 + (i as u32) * 0x10000 + if last { 0x80 } else { 0 },
        );
        for j in 0..16 {
            let o = 0x0c + j * 4 + if j >= 9 { 4 } else { 0 };
            put_u32(&mut rec, o, 0x21 + (i as u32) * 0x10000);
        }
        for j in 0..15 {
            put_u32(&mut rec, 0x54 + j * 4, lane_offs[i][j + 1] as u32);
        }
        for j in 0..16 {
            put_u32(&mut rec, 0x90 + j * 4, lane_szs[i][j] as u32);
        }
        put_u32(&mut rec, 0xdc, ci as u32);
        put_u32(
            &mut rec,
            0xe0,
            if narrow { 0x9141_8005 } else { 0x9341_8005 },
        );
        put_u32(&mut rec, 0xec, co as u32);
        if !narrow {
            push(&mut rec, f128_of(co));
        }
        push(&mut rec, 0x0010_0000);

        let reloc_off;
        if i == 0 {
            for v in [
                0x8008_1342u32,
                0xce,
                0x40,
                0x135a,
                0x0100_2031,
                0x0000_9041,
                0x0050_0172,
                0x0050_0130,
            ] {
                push(&mut rec, v);
            }
            push(
                &mut rec,
                if out_off == 0 {
                    0xd007_9044
                } else {
                    0xf007_9044
                },
            );
            push(&mut rec, (out_off + co * 16) as u32);
            push(&mut rec, 0x10);
            for _ in 0..3 {
                push(&mut rec, (ci * 16) as u32);
            }
            push(&mut rec, 0x0050_0170);
            if out_off != 0 {
                push(&mut rec, out_off as u32);
            }
            push(&mut rec, 0x10);
            for v in [0x8002_1241u32, 0x0010_3c00, 0x0000_3c00, 0x2380_9544] {
                push(&mut rec, v);
            }
            reloc_off = rec.len();
            push(&mut rec, kern_off[i] as u32);
            for v in [0u32, 0x2280_9344, 0, 0, 0x2200_1340, 0x21] {
                push(&mut rec, v);
            }
            while rec.len() < 0x170 {
                push(&mut rec, 0);
            }
            debug_assert_eq!(rec.len(), 0x170);
        } else {
            // input descriptor: short form at pool offset 0, long form
            // otherwise. The long form's second word turns 0x120 when
            // the buffer's slot was created after the first reuse.
            if in_off == 0 {
                for v in [0x8004_1041u32, 0x120, 0x10] {
                    push(&mut rec, v);
                }
            } else {
                let w2 = if i > 1 && slot_after_reuse[i - 1] {
                    0x120
                } else {
                    0x104
                };
                for v in [0x8006_1041u32, w2, in_off as u32, 0x10] {
                    push(&mut rec, v);
                }
            }
            if last {
                push(&mut rec, 0x0000_9052);
                push(&mut rec, 0x14a);
                push(&mut rec, (in_off + ci * 16) as u32);
                for v in [
                    0x8002_1241u32,
                    0x0010_3c00,
                    0x0000_3c00,
                    0x8080_1445,
                    0x40,
                    0x0130_2031,
                    0x8311_9640,
                    0x12,
                    0x00a0_00a0,
                    ((i as u32) << 16) | 0xe0,
                    0,
                    0x007f_0000,
                    0x2001_0701,
                    0x2380_9544,
                ] {
                    push(&mut rec, v);
                }
                reloc_off = rec.len();
                push(&mut rec, kern_off[i] as u32);
                for v in [0u32, 0x2300_9442, 0, 0, 0x2200_1440] {
                    push(&mut rec, v);
                }
                push(&mut rec, 0x0100_0031 + (i as u32) * 0x10000);
            } else {
                if out_off == 0 {
                    for v in [0x8001_1052u32, 0x140, 0x10] {
                        push(&mut rec, v);
                    }
                } else {
                    for v in [0x0001_1052u32, 0x140, out_off as u32, 0x10] {
                        push(&mut rec, v);
                    }
                }
                for v in [0x8002_1241u32, 0x0010_3c00, 0x0000_3c00, 0x2380_9544] {
                    push(&mut rec, v);
                }
                reloc_off = rec.len();
                push(&mut rec, kern_off[i] as u32);
            }
        }

        if !last {
            // intermediate-buffer descriptor block at a fixed tail
            // offset. Narrow records sit 0x10 lower — except when both
            // the input and output descriptors carry pool offsets
            // (long/long), which restores the wide-record length.
            let base = if i == 0 {
                0x170
            } else if narrow && !(in_off != 0 && out_off != 0) {
                0x130
            } else {
                0x140
            };
            while rec.len() < base {
                push(&mut rec, 0);
            }
            let oid = if feeder {
                // buffer consumed by the last task: 0x63 when it grew
                // past the previous intermediate or the output is
                // narrow-class; 0x64 otherwise. k=2 runs one class lower.
                let cond =
                    chain_narrow(shapes[k - 1].1) || (k >= 3 && shapes[k - 2].1 > shapes[k - 3].1);
                (if k == 2 { 0x63u32 } else { 0x64u32 }) - cond as u32
            } else if i == 0 && k == 3 {
                // single intermediate besides the feeder (k=3):
                // 0x54 when the feeder is narrow-class and at least as
                // large as this buffer, else 0x55.
                if chain_narrow(shapes[1].1) && shapes[1].1 >= shapes[0].1 {
                    0x54
                } else {
                    0x55
                }
            } else {
                // Word offset of task `i+1`'s completion descriptor:
                // 0x53 base + 1 if task `i+1` carries the f128 word
                // (`!chain_narrow(co_{i+1})`) + 1 if task `i+1`'s input
                // descriptor is long (`out_off != 0`) + 1 if task
                // `i+1`'s output descriptor is long (`pool_off[i+1] != 0`).
                0x53 + (out_off != 0) as u32
                    + (pool_off[i + 1] != 0) as u32
                    + (!chain_narrow(shapes[i + 1].1)) as u32
            };
            push(&mut rec, (oid << 16) | (i as u32 + 1));
            push(&mut rec, desc_flag(shapes[i].1));
            push(&mut rec, if feeder { 0x0400_0000 } else { 0 });
            push(&mut rec, 0);
            push(&mut rec, if feeder { 0xfff8_60 } else { 0xfff8_20 });
            push(&mut rec, 0);
            push(&mut rec, 0);
            push(&mut rec, if feeder { 0 } else { 1 });
        }
        relocs.push([
            (rec_base + reloc_off) as u8,
            ((rec_base + reloc_off) >> 8) as u8,
            ((rec_base + reloc_off) >> 16) as u8,
            ((rec_base + reloc_off) >> 24) as u8,
            0x05,
            0,
            0,
            0x07,
        ]);
        rec_base += rec.len();
        text.extend_from_slice(&rec);
    }
    let textlen = text.len();

    // ---- symtab entries ---------------------------------------------
    let mut entries: Vec<(String, u8, u8, u16, u64)> = Vec::with_capacity(27 + 16 * k);
    entries.push(("x".into(), 0x0f, 1, 0, 0x3000_0000));
    entries.push((out_name.into(), 0x0f, 2, 0, out_va));
    // ANEC orders the table x, out, then every family's primary `ne_0`
    // (desc=2), then each family's remaining `ne_1..15` lanes.
    let knames: Vec<String> = (0..k)
        .map(|i| conv_kernel_name_ns(shapes[i].0 as u64, shapes[i].1 as u64, data_offs[i]))
        .collect();
    for (i, kn) in knames.iter().enumerate() {
        entries.push((
            format!("{kn}_ne_0"),
            0x0f,
            5,
            2,
            kern_va + (kern_off[i] + lane_offs[i][0]) as u64,
        ));
    }
    for (i, kn) in knames.iter().enumerate() {
        for j in 1..16 {
            entries.push((
                format!("{kn}_ne_{j}"),
                0x0f,
                5,
                0,
                kern_va + (kern_off[i] + lane_offs[i][j]) as u64,
            ));
        }
    }
    for ty in TYPE_SYMS {
        entries.push(((*ty).into(), 0x80, 0, 0, 0));
    }
    entries.push((layout_desc("x", 24, shapes[0].0 as u32), 0x20, 0, 0x18, 0));
    entries.push((
        layout_desc(out_name, 29, shapes[k - 1].1 as u32),
        0x20,
        0,
        0x1d,
        0,
    ));

    // ---- envelope: prefix + generated tail ---------------------------
    let const2_span = textlen.div_ceil(0x40) * 0x40;
    let kern_vsz = kern_bytes.div_ceil(0x4000) * 0x4000;
    let pre = &SKELETON_CHAIN4[..CHAIN_PRE_OFF];
    let (mut img, text_off) = build_tail_at_reloc(
        pre,
        CHAIN_PRE_OFF,
        &entries,
        &relocs,
        mil_path,
        out_path,
        0x218,
    );
    let const2_fo = text_off + const2_span;
    let kern_fo = text_off + 0x8000;
    let const2_va = text_va + const2_span as u64;

    // FVMLIB buffer segments: cmd1 = network input, cmd2 = network
    // output — section size = channels * 0x40 (verified C=32/64/64→32→32)
    put_u64(&mut img, 0x88, fv_in); // __FVMLIB(in) seg vmsize
    put_u64(&mut img, 0xd8, (c_in * 0x40) as u64);
    put_u64(&mut img, 0x118, out_va); // __FVMLIB(out) seg vmaddr
    put_u64(&mut img, 0x120, fv_out); // __FVMLIB(out) seg vmsize
    put_u64(&mut img, 0x168, out_va); // __data sect addr
    put_u64(&mut img, 0x170, (c_out * 0x40) as u64);
    put_u64(&mut img, 0x1b0, text_va); // __TEXT seg vmaddr
    put_u64(&mut img, 0x1c0, text_off as u64); // __TEXT seg fileoff
    put_u64(&mut img, 0x200, text_va); // __text sect addr
    put_u64(&mut img, 0x208, textlen as u64); // __text sect size
    put_u32(&mut img, 0x210, text_off as u32); // __text sect offset
    put_u32(&mut img, 0x21c, k as u32); // __text nreloc
    put_u64(&mut img, 0x250, const2_va); // __const sect addr
    put_u32(&mut img, 0x260, const2_fo as u32); // __const sect offset
    put_u64(&mut img, 0x298, kern_va); // __KERN_0 seg vmaddr
    put_u64(&mut img, 0x2a0, kern_vsz as u64); // __KERN_0 vmsize
    put_u64(&mut img, 0x2a8, kern_fo as u64); // __KERN_0 fileoff
    put_u64(&mut img, 0x2b0, kern_vsz as u64); // __KERN_0 filesize
    put_u64(&mut img, 0x2e8, kern_va); // __kern_0 sect addr
    put_u64(&mut img, 0x2f0, kern_bytes as u64); // __kern_0 sect size
    put_u32(&mut img, 0x2f8, kern_fo as u32); // __kern_0 sect offset
                                              // bind1 (network output) +0x18 carries a symbol-table reference:
                                              // (0x30 + k) in the high byte — same `k + 0x30` as cmd 0x4 #3.
    put_u32(&mut img, 0x348, out_va as u32); // cmd 0x4 #0: out VA
    put_u32(&mut img, 0x350, ((0x30 + k as u32) << 8) | 0x79);
    put_u32(&mut img, 0x368, text_va as u32); // cmd 0x4 #0: text VA
    put_u32(&mut img, 0x378, const2_va as u32); // cmd 0x4 #1: const VA
    put_u32(&mut img, 0x3c8, out_va as u32); // cmd 0x4 #?: out VA
    put_u32(&mut img, 0x3d8, kern_va as u32); // cmd 0x4 #?: kern VA
    put_u32(&mut img, 0xb68, text_va as u32); // cmd 0x4 #1: text VA
    put_u32(&mut img, 0xbe8, f14); // cmd 0x4 #1: io page count
    put_u32(&mut img, 0xb7c, (textlen / 4) as u32); // cmd 0x4 #1: text words
    put_u32(&mut img, 0xb88, k as u32); // cmd 0x4 #1: task count
    put_u32(&mut img, 0xbb8, (14 * k + 3) as u32); // cmd 0x4 #1: sched count
                                                   // cmd 0x4 #1 input side: channel count + three buffer-size copies
    put_u32(&mut img, 0xc24, c_in);
    for o in [0xc48usize, 0xc68, 0xc78] {
        put_u32(&mut img, o, c_in * 0x40);
    }
    // cmd 0x4 #3 output side: channel count + three buffer-size copies
    put_u32(&mut img, 0x196c, c_out);
    for o in [0x1990usize, 0x19b0, 0x19c0] {
        put_u32(&mut img, o, c_out * 0x40);
    }
    put_u32(&mut img, 0x267c, 0x4030_7900 + (k as u32) * 0x10000);
    put_u32(&mut img, 0x2688, (0x30 + k) as u32);

    // ---- section contents --------------------------------------------
    img.resize(text_off, 0);
    img.extend_from_slice(&text);
    img.resize(kern_fo, 0);
    for s in &streams {
        for v in s {
            img.extend_from_slice(&v.to_le_bytes());
        }
    }
    img.resize(kern_fo + kern_vsz, 0);
    Ok(img)
}

/// Same-shape convenience wrapper over [`emit_conv_chain`].
pub fn emit_conv_chain64(
    k: usize,
    weights: &[u16],
    out_name: &str,
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    emit_conv_chain(
        &vec![(64usize, 64usize); k],
        weights,
        &(0..k).map(|i| 0x80 + i as u64 * 0x2040).collect::<Vec<_>>(),
        out_name,
        mil_path,
        out_path,
    )
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
        rec[0x54 + 4 * b..0x58 + 4 * b].copy_from_slice(&(stride * (b as u32 + 1)).to_le_bytes());
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
        return Err(format!(
            "expected {layers} weight blobs, got {}",
            weights.len()
        ));
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
        return Err(format!(
            "chain of {layers} layers overflows the record plane"
        ));
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
                0x41,
                0x10,
                0x06,
                0x80,
                0x04,
                0x01,
                0x00,
                0x00,
                0x00,
                4 * (i as u8 - 1),
                0x00,
                0x00,
                0x10,
                0x00,
                0x00,
                0x00,
                0x52,
                0x10,
                0x01,
                0x00,
                0x40,
                0x01,
                0x00,
                0x00,
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
    let const_end = text_off + stream_len.div_ceil(0x40) * 0x40 + 0x4000;
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
                img[base + head + i * 2..base + head + i * 2 + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    img.resize(
        (kern_off + (stream_sz * streams.len() + 0x3fff)) & !0x3fff,
        0,
    );

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
    let aligned = stream_len.div_ceil(0x40) * 0x40;
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
/// from ANEC dw probes; ordinals beyond 5 need further mining.
const KNAME_SM_DW: [&str; 12] = [
    "KE3E0F4E952E53E653B75FE6AF44DD22939751A660FEF245DA535F7F503E9AC52",
    "K36F9FAA5AC1598F61E735E999A791F3AF94F26320DC08F4A47655BE0F56D009C",
    "K445C1528F57CD5C3F8981EF604408231AE7ABE5C085A6D3966F1B4E5B85CDB85",
    "K8352046B90FD96625782305C20104AE0C2A4CE2D9D985157FCA0FC20945A859E",
    "K85542F51A780FBCE6FF5BED1B91438097147F4C3080AD4F938E620367C9D2EA3",
    "K54AF82C5871EA2863186C5298F4A8364871F89349D053806768B5829341185D5",
    "K0AA87BB51004D6284B7AE5CDA663CC8198C38BA3BF9D5C1154C5A348327EF014",
    "KDB0CCC61329C942785BA98FF3791D13EE53D966FFDC7CD2A4A1C96AD69B6FBA6",
    "K81BF0E9988805F7BAF6D3D574EC5CDE61B7C0F51AF4A235DE581EA539E2ABC66",
    "K80C4CCDDB1B6B3E1E9D5B5AB255039AB8CA529A0CDB3591F69734D877EEB75D2",
    "KEB1128FF3601A42210AE7DAB8F9EA0A9F0658244A7F78C002C028ABDF85B6037",
    "K6502428E710051C8E12E86D0756DD39A411400A54AECA6D3F0528BD8FAAFAD8E",
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
    0x0a, 0x0e, 0x12, 0x16, 0x1a, 0x1e, 0x22, 0x26, 0x2a, 0x32, 0x36, 0x3a, 0x3e, 0x42, 0x46, 0x4a,
    0x52,
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
        return Err(format!(
            "expected {layers} weight blobs, got {}",
            weights.len()
        ));
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
    if streams.len() > 6 {
        return Err(format!(
            "{} distinct softmax weight streams exceeds the verified table (6)",
            streams.len()
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
            (
                0x14c,
                skel3[SM_CONV_FIRST.0..SM_CONV_FIRST.0 + SM_CONV_FIRST.1].to_vec(),
            )
        } else {
            let mut t = skel3[SM_CONV_INT.0..SM_CONV_INT.0 + SM_CONV_INT.1].to_vec();
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

    let (mut img, text_off) =
        build_tail_at(skel3, CONV_TAIL_OFF, &entries, &relocs, mil_path, out_path);

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
                img[base + i * 2..base + i * 2 + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    // exp/reciprocal LUTs from the k=1 skeleton's aux tail
    // (kern fileoff 0xc000 + 0x2000)
    img[kern_off + 0x2000..kern_off + 0x2100]
        .copy_from_slice(&SKELETON_CONV64_SM[0xe000..0xe100][..]);

    // layer-count scalars for the multi-task family
    let aligned = section_size.div_ceil(0x40) * 0x40;
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
    img[0xb7c..0xb7e].copy_from_slice(&((0x151 + 0x134 * (layers - 1)) as u16).to_le_bytes());
    img[0xb88..0xb8c].copy_from_slice(&((6 * layers) as u32).to_le_bytes());
    img[0xbb8..0xbba].copy_from_slice(&((0x57 + 0x54 * (layers - 1)) as u16).to_le_bytes());
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
/// lndw2-6 probes; ordinals beyond 5 need further mining.
const KNAME_LN_DW: [&str; 6] = [
    "K3C23BA63F9FE7926259BE0CD49C610C288A7DB57FE6AFF2F2F0543F4EF3B5712",
    "KE3545219F139ACC28DC984868084DC69371098AC16C6143198F730D551E6D04F",
    "K49750E73342DC5A9AF618282C1FC3E0663E255CD326287258C267314D841D492",
    "KC2200A5DA40187D6913B22C3566BA0458BC425B33670C299DF9AB485B31C198E",
    "K97766E9CC7D38B5F48C560C7A007B1F77335519988EE1ECCC57EADC0FDEE3AFC",
    "K3E3F259488E410992DE415DA6A4746060519BB624D148CBA18B8C2327A0E01F1",
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
        return Err(format!(
            "expected {layers} weight blobs, got {}",
            weights.len()
        ));
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
                let cb = (4 + i.div_ceil(3)) as u16;
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
        t[mark_u32_off..mark_u32_off + 4].copy_from_slice(&(0x2000 * ords[i] as u32).to_le_bytes());
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

    let (mut img, text_off) =
        build_tail_at(skel, CONV_TAIL_OFF, &entries, &relocs, mil_path, out_path);

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
                img[base + i * 2..base + i * 2 + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
    }

    // layer-count scalars — the record-end fields use the
    // 0x40-aligned text size like the other families
    let aligned = section_size.div_ceil(0x40) * 0x40;
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
        &((0xec + 0xcc * (layers - 1) + if layers > 1 { 5 } else { 0 }) as u16).to_le_bytes(),
    );
    img[0xb88..0xb8c].copy_from_slice(&((5 * layers) as u32).to_le_bytes());
    img[0xbb8..0xbba].copy_from_slice(&((0x49 + 0x46 * (layers - 1)) as u16).to_le_bytes());
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
/// Extended-family skeletons — ANEC switches `(conv -> rms_norm)^k`
/// depthwise programs to a 13-task-id-stride record with per-layer
/// `__KERN_0` lanes (no weight-stream dedup) once `layers >= 5` and
/// `distinct >= 2`. The 8-layer artifact supplies every task template
/// except the terminal+marker aux3, which only occurs when `k % 3`
/// leaves a partial trailing triple-group on the last layer — the
/// 7-layer artifact provides it.
const SKELETON_CONV_RMSX7: &[u8] = include_bytes!("../assets/conv_rmsdw7_skeleton.hwx");
const SKELETON_CONV_RMSX8: &[u8] = include_bytes!("../assets/conv_rmsdw8_skeleton.hwx");

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
        return Err(format!(
            "expected {layers} weight blobs, got {}",
            weights.len()
        ));
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
    // `layers >= 5` with more than one distinct weight stream selects
    // the extended family: no dedup (one `__KERN_0` lane per layer),
    // 13-id stride, wide aux3 schedule tables, a per-layer ramp table
    // in the second `__TEXT` section, and ordinal-positioned kernel
    // names. (Shared-weight chains — one distinct stream — stay on the
    // compact path at any depth.)
    if layers >= 5 && streams.len() > 1 {
        return emit_conv_rms_xdw(layers, weights, mil_path, out_path);
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
                e.push((
                    rms_entries[4].0.clone(),
                    tpl0.1,
                    tpl0.2,
                    tpl0.3,
                    kern_addr + 0x2000,
                ));
                e.push((
                    rms_entries[5].0.clone(),
                    tpl0.1,
                    tpl0.2,
                    tpl0.3,
                    kern_addr + 0x2080,
                ));
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
                e.push((
                    rms_entries[21].0.clone(),
                    tpln.1,
                    tpln.2,
                    tpln.3,
                    kern_addr + 0x2080,
                ));
                e.push((
                    rms_entries[22].0.clone(),
                    tpln.1,
                    tpln.2,
                    tpln.3,
                    kern_addr + 0x2100,
                ));
            }
        }
        e.extend_from_slice(&rms_entries[23..]); // scalar descriptors
        e
    };

    let (mut img, text_off) =
        build_tail_at(skel, CONV_TAIL_OFF, &entries, &relocs, mil_path, out_path);

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
                img[base + i * 2..base + i * 2 + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    // LUT tail — 0x180 k=1 (three blocks) / 0x100 chains (two)
    if layers == 1 {
        img[kern_off + 0x2000..kern_off + 0x2180]
            .copy_from_slice(&SKELETON_CONV_RMS1[0xe000..0xe180][..]);
    } else {
        img[kern_off + 0x2000..kern_off + 0x2100].copy_from_slice(&skel[0xa000..0xa100][..]);
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
    put_u32(&mut img, 0xb7c, 0x220 + (section_size as u32 - 0x880) / 4);
    put_u32(
        &mut img,
        0xb88,
        if layers == 1 { 0xb } else { 12 * layers as u32 },
    );
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

/// `__text` section (file offset, size) of a multi-section skeleton —
/// the second entry of the third segment command (fixed head offset
/// `0x1e0` on the conv/rms chain families).
fn xtext_bounds(skel: &[u8]) -> (usize, usize) {
    let s = 0x1e0usize;
    let off = u32::from_le_bytes(skel[s + 0x30..s + 0x34].try_into().unwrap()) as usize;
    let size = u32::from_le_bytes(skel[s + 0x28..s + 0x2c].try_into().unwrap()) as usize;
    (off, size)
}

/// Task-descriptor head offsets inside an extended-family `__text`.
/// Every task head is `XX 00 40 15 c0 ff 40 02` and the first sits at
/// `0x32`; tasks are emitted four per layer (conv, aux1, aux2, aux3).
fn xtask_starts(text: &[u8]) -> Vec<usize> {
    let pat = [0x40u8, 0x15, 0xc0, 0xff, 0x40, 0x02];
    let mut v = Vec::new();
    let mut i = 0x32usize;
    while let Some(p) = text[i..].windows(6).position(|w| w == pat) {
        v.push(i + p - 2);
        i += p + 6;
    }
    v
}

/// Extended multi-lane `(conv -> rms_norm)^k` depthwise family —
/// `layers >= 5` with ≥2 distinct weight streams. ANEC drops weight
/// deduplication entirely: layer `i` binds `__KERN_0` lane `i` keyed
/// by the ordinal-position kernel name `KNAME_SM_DW[i]`, the record
/// runs on a 13-task-id stride, aux3 carries a wide schedule table
/// (four forms: interior `0x220`, marker `0x230`, terminal `0x23a`,
/// terminal+marker `0x24a`), and the second `__TEXT` section grows to
/// `0x40 + 0x100*layers` holding a constant ramp lane per layer. The
/// record is stream-content-invariant — the task bytes depend only on
/// the layer count.
///
/// Marker rule: a layer is "marked" iff `i == 3*floor(layers/3)` and
/// `layers % 3 != 0` — the first layer of the trailing partial
/// triple-group. Marked layers patch aux2 (`+0x18a`=0x44, `+0x1c0`=0x39,
/// task-rel) and select the wide aux3 forms.
fn emit_conv_rms_xdw(
    layers: usize,
    weights: &[&[u16]],
    mil_path: &str,
    out_path: &str,
) -> Result<Vec<u8>, String> {
    const C: usize = 64;
    if layers > KNAME_SM_DW.len() {
        return Err(format!(
            "{layers} layers exceeds the mined ordinal kernel-name table ({})",
            KNAME_SM_DW.len()
        ));
    }
    let skel = SKELETON_CONV_RMSX8;
    let (toff8, tsz8) = xtext_bounds(skel);
    let text8 = &skel[toff8..toff8 + tsz8];
    let (toff7, tsz7) = xtext_bounds(SKELETON_CONV_RMSX7);
    let text7 = &SKELETON_CONV_RMSX7[toff7..toff7 + tsz7];
    let st8 = xtask_starts(text8);
    let st7 = xtask_starts(text7);
    debug_assert_eq!(st8.len(), 32);
    debug_assert_eq!(st7.len(), 28);

    // Emitted slice of a donor task — the 2-byte head (`05 00`/`01 00`)
    // is carried by the preceding slice's tail (or the prologue for
    // task 0), so slices run `[start+2, next_start+2)`; the program's
    // last task has no successor and emits body-only.
    let task = |text: &[u8], st: &[usize], t: usize| -> Vec<u8> {
        let s = st[t];
        let e = if t + 1 < st.len() {
            st[t + 1] + 2
        } else {
            text.len()
        };
        text[s + 2..e].to_vec()
    };

    let (x_entries, x_relocs) = tail_meta(skel);
    let rflag: [u8; 4] = x_relocs[0][4..8].try_into().unwrap();

    let mut stream = text8[..0x34].to_vec();
    let mut reloc_offs = Vec::with_capacity(4 * layers + 1);
    let mk = 3 * (layers / 3);

    for i in 0..layers {
        let last = i + 1 == layers;
        let marked = mk < layers && i == mk;

        // conv — layer 0 keeps the input-DMA prologue form; interior
        // layers carry no stream binding at all in this family
        let co = stream.len() - 0x34 + 4;
        if i == 0 {
            stream.extend_from_slice(&task(text8, &st8, 0));
            reloc_offs.push(co + 0x14c);
        } else {
            let mut t = task(text8, &st8, 4);
            patch_task_column(&mut t, (13 * i) as u16, 16);
            put_u16(&mut t, 0x06, (13 * i + 1) as u16);
            put_u16(&mut t, 0x125, (0x20 * i + 1) as u16);
            put_u16(&mut t, 0x12c, (13 * i + 1) as u16);
            put_u16(&mut t, 0x1ac, (13 * i + 2) as u16);
            put_u16(&mut t, 0x22c, (13 * i + 3) as u16);
            put_u16(&mut t, 0x2ac, (13 * i + 4) as u16);
            if last {
                // `00 05` writeback tags → `80 04`
                put_u16(&mut t, 0x21c, 0x0480);
                put_u16(&mut t, 0x288, 0x0480);
                put_u16(&mut t, 0x294, 0x0480);
            }
            stream.extend_from_slice(&t);
            reloc_offs.push(co + 0x124);
        }

        // aux1 — single 0x1f0 form; terminal adds release descriptors
        let co = stream.len() - 0x34 + 4;
        let mut t = task(text8, &st8, 5);
        patch_task_column(&mut t, (13 * i + 4) as u16, 17);
        put_u16(&mut t, 0x06, (13 * i + 5) as u16);
        put_u16(&mut t, 0xcc, (13 * i + 5) as u16);
        put_u16(&mut t, 0x14c, (13 * i + 6) as u16);
        put_u16(&mut t, 0x1cc, (13 * i + 7) as u16);
        if last {
            put_u16(&mut t, 0x94, 0x0480);
            put_u16(&mut t, 0xa0, 0x0490);
            put_u16(&mut t, 0x130, 0x0490);
        }
        stream.extend_from_slice(&t);
        reloc_offs.push(co + 0xb8);

        // aux2 — marker layers re-tag the aux3 lane producers
        let co = stream.len() - 0x34 + 4;
        let mut t = task(text8, &st8, 6);
        patch_task_column(&mut t, (13 * i + 7) as u16, 17);
        put_u16(&mut t, 0x06, (13 * i + 8) as u16);
        put_u16(&mut t, 0xbc, (13 * i + 8) as u16);
        put_u16(&mut t, 0x13c, (13 * i + 9) as u16);
        put_u16(&mut t, 0x1bc, (13 * i + 10) as u16);
        if marked {
            t[0x188] = 0x44;
            t[0x1be] = 0x39;
        }
        if last {
            for o in [0x8cusize, 0x94, 0x120, 0x124, 0x128, 0x1a4, 0x1a8] {
                put_u16(&mut t, o, 0x0480);
            }
        }
        stream.extend_from_slice(&t);
        reloc_offs.push(co + 0xac);

        // aux3 — four structural forms; interior emits `13i+13` (the
        // next layer's producer), terminal forms carry no emit id
        let co = stream.len() - 0x34 + 4;
        let (mut t, roff) = match (last, marked) {
            (false, false) => (task(text8, &st8, 7), 0xb4usize),
            (false, true) => (task(text8, &st8, 27), 0xb8),
            (true, false) => (task(text8, &st8, 31), 0xb4),
            (true, true) => (task(text7, &st7, 27), 0xb8),
        };
        patch_task_column(&mut t, (13 * i + 10) as u16, 17);
        put_u16(&mut t, 0x06, (13 * i + 11) as u16);
        match (last, marked) {
            (false, false) => {
                put_u16(&mut t, 0xbc, (13 * i + 11) as u16);
                put_u16(&mut t, 0x13c, (13 * i + 12) as u16);
                put_u16(&mut t, 0x1e1, i as u16);
                put_u16(&mut t, 0x1ee, (13 * i + 12) as u16);
                put_u16(&mut t, 0x1fc, (13 * i + 13) as u16);
            }
            (false, true) => {
                put_u16(&mut t, 0xcc, (13 * i + 11) as u16);
                put_u16(&mut t, 0x14c, (13 * i + 12) as u16);
                put_u16(&mut t, 0x1f1, i as u16);
                put_u16(&mut t, 0x1fe, (13 * i + 12) as u16);
                put_u16(&mut t, 0x20c, (13 * i + 13) as u16);
            }
            (true, false) => {
                put_u16(&mut t, 0xbc, (13 * i + 11) as u16);
                put_u16(&mut t, 0x13c, (13 * i + 12) as u16);
                put_u16(&mut t, 0x202, (13 * i + 12) as u16);
                put_u16(&mut t, 0x215, i as u16);
                put_u16(&mut t, 0x22e, (13 * i + 12) as u16);
                // trailing byte is a `01` tag, not the id high byte
                t[0x236] = (13 * i + 12) as u8;
            }
            (true, true) => {
                put_u16(&mut t, 0xcc, (13 * i + 11) as u16);
                put_u16(&mut t, 0x14c, (13 * i + 12) as u16);
                put_u16(&mut t, 0x212, (13 * i + 12) as u16);
                put_u16(&mut t, 0x225, i as u16);
                put_u16(&mut t, 0x23e, (13 * i + 12) as u16);
                t[0x246] = (13 * i + 12) as u8;
            }
        }
        stream.extend_from_slice(&t);
        reloc_offs.push(co + roff);
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

    // The second `__TEXT` section holds a `0x40`-byte `00 1c` header
    // then one `0x100` lane per layer — `0x80` zeros plus a constant
    // u16 ramp `0x1234..0x1273`. The symtab names each lane `TT<out>`
    // after the layer's output tensor (s{i+2} .. z by corpus
    // convention).
    let section_size = stream.len();
    let aligned = (section_size + 0x3f) & !0x3f;
    let sec2_addr = 0x3000_8000u64 + aligned as u64;
    let sec2_size = 0x40u32 + 0x100 * layers as u32;
    let kern_addr = (sec2_addr + sec2_size as u64 + 0x3fff) & !0x3fff;
    let kern_payload = 0x100 + 0x2000 * layers;
    let kern_vsz = (kern_payload + 0x3fff) & !0x3fff;
    let stream_off = |o: usize| if o == 0 { 0 } else { 0x100 + 0x2000 * o };

    // symtab — [fl, x, z] + per-lane ne_0 (lane 0 trails the two
    // `_actlut` names) + per-layer `TT` lane descriptors + per-lane
    // ne_1..ne_15 groups (bare LUT names trail lane 0) + scalars.
    let mut e = Vec::with_capacity(17 * layers + 32);
    e.push(x_entries[0].clone());
    e.push(x_entries[1].clone());
    e.push(x_entries[2].clone());
    e[0].4 = sec2_addr;
    let tpl0 = x_entries[3].clone();
    let ttt = x_entries[13].clone();
    let tpln = x_entries[21].clone();
    for j in 0..layers {
        e.push((
            format!("{}_ne_0", KNAME_SM_DW[j]),
            tpl0.1,
            tpl0.2,
            tpl0.3,
            kern_addr + stream_off(j) as u64,
        ));
        if j == 0 {
            e.push((
                x_entries[4].0.clone(),
                tpl0.1,
                tpl0.2,
                tpl0.3,
                kern_addr + 0x2000,
            ));
            e.push((
                x_entries[5].0.clone(),
                tpl0.1,
                tpl0.2,
                tpl0.3,
                kern_addr + 0x2080,
            ));
        }
    }
    // per-layer output names from the MIL: each following conv's `x`
    // operand, then the function output
    let layer_out: Vec<String> = std::fs::read_to_string(mil_path)
        .ok()
        .and_then(|t| {
            let io = mil_io_sig(&t)?;
            let xs: Vec<String> = mil_ops(&t)
                .0
                .iter()
                .filter(|o| o.op == "conv")
                .skip(1)
                .filter_map(|o| {
                    o.args
                        .iter()
                        .find(|(k, _)| k == "x")
                        .map(|(_, v)| v.clone())
                })
                .collect();
            (xs.len() + 1 == layers).then(|| xs.into_iter().chain([io.out]).collect())
        })
        .unwrap_or_else(|| {
            (0..layers)
                .map(|j| {
                    if j + 1 == layers {
                        "z".into()
                    } else {
                        format!("s{}", j + 2)
                    }
                })
                .collect()
        });
    for (j, nm) in layer_out.iter().enumerate() {
        e.push((
            format!("TT{nm}_ew_to_goc_xfm_1_fuse_goc_const_xfm_0"),
            ttt.1,
            ttt.2,
            ttt.3,
            sec2_addr + 0x40 + 0x100 * j as u64,
        ));
    }
    for j in 0..layers {
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
            e.push((
                x_entries[36].0.clone(),
                tpln.1,
                tpln.2,
                tpln.3,
                kern_addr + 0x2080,
            ));
            e.push((
                x_entries[37].0.clone(),
                tpln.1,
                tpln.2,
                tpln.3,
                kern_addr + 0x2100,
            ));
        }
    }
    e.extend_from_slice(&x_entries[x_entries.len() - 25..]);

    let (mut img, text_off) = build_tail_at(skel, CONV_TAIL_OFF, &e, &relocs, mil_path, out_path);
    // `__KERN_0` file offset mirrors the vmaddr delta — the next
    // 0x4000 page past the end of the second `__TEXT` section.
    let kern_off = text_off + (kern_addr - 0x3000_8000) as usize;
    img.extend_from_slice(&stream);
    img.resize(kern_off + kern_vsz, 0);
    img[text_off + aligned + 1] = 0x1c;
    for j in 0..layers {
        let lane = text_off + aligned + 0x40 + 0x100 * j + 0x80;
        for k in 0..64u16 {
            img[lane + 2 * k as usize..lane + 2 * k as usize + 2]
                .copy_from_slice(&(0x1234u16 + k).to_le_bytes());
        }
    }
    for (j, w) in weights.iter().enumerate() {
        let inter = weight_interleave(w, C);
        for b in 0..16usize {
            let base = kern_off + stream_off(j) + b * 0x200;
            for (i, v) in inter[b * 256..b * 256 + 256].iter().enumerate() {
                img[base + i * 2..base + i * 2 + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    // div/recip LUT tail on lane 0
    img[kern_off + 0x2000..kern_off + 0x2100].copy_from_slice(&skel[0x12000..0x12100]);

    // scalars — the `__text` segment spans through `__KERN_0`'s
    // page-aligned file offset
    let seg_span = kern_off - text_off;
    put_u32(&mut img, 0x1b8, seg_span as u32);
    put_u32(&mut img, 0x1c8, seg_span as u32);
    put_u32(&mut img, 0x1c0, text_off as u32);
    put_u32(&mut img, 0x208, section_size as u32);
    put_u32(&mut img, 0x210, text_off as u32);
    put_u32(&mut img, 0x21c, (4 * layers) as u32);
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
    put_u32(&mut img, 0xb7c, 0x220 + (section_size as u32 - 0x880) / 4);
    put_u32(&mut img, 0xb88, 13 * layers as u32);
    put_u32(&mut img, 0xbb8, 0x3 + 0xb6 * layers as u32);
    Ok(img)
}

/// Matmul shape families with a captured skeleton.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmShape {
    /// `[1,16,1,16]` operands — sub-tile family: the const pool
    /// holds sixteen `0x40`-byte bands of `[0x20 weights][0x20 pad]`.
    N16,
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
            MmShape::N16 => 16,
            MmShape::N32 => 32,
            MmShape::N64 => 64,
            MmShape::N128 => 128,
            MmShape::N256 => 256,
        }
    }

    fn skeleton(self) -> &'static [u8] {
        match self {
            MmShape::N16 => SKELETON_MM16,
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
/// Same partition over `[1,32,1,32]` — smaller lane footprint
/// (`0x1f4` record) but the identical envelope and symtab plan.
const SKELETON_MM_ACT32: &[u8] = include_bytes!("../assets/mm_act32_skeleton.hwx");

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
    /// `matmul(x, y)` `[1,32,1,32]` — bare PV/QK partition.
    Act32,
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
            MmPart::Act32 => (SKELETON_MM_ACT32, 0x33f0, 0),
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
        skel,
        tail_off,
        &entries,
        &relocs,
        mil_path,
        out_path,
        reloff_field,
    );
    debug_assert_eq!(text_off, 0x8000);
    img.extend_from_slice(&skel[0x8000..]);
    img
}

/// Broadcast-vector×matrix activation matmul (`matmul(x=[1,1,1,K],
/// y=[1,R,1,K], transpose_y=true)`): ANEC lowers it to a sliding
/// row-dot engine whose channel-plane output satisfies
/// `z[c][h] = a·V[c+h]` — the `c=0` row is the full matvec
/// `z[j] = Σ_w a[w]·V[j][w]`. This is both decode-time attention
/// contractions: `a=q` against `V=K[t][d]` produces all scores in one
/// program, and `a=scores` against `V=Vᵀ[j][t]` produces `attn·V`
/// (the value cache is stored transposed — a column-append).
/// Exec-verified at (R,K) ∈ {(64,64),(64,256),(512,64)} with 0
/// mismatches.
const SKELETON_AVW: &[u8] = include_bytes!("../assets/avw_ft64_skeleton.hwx");

/// Emit the broadcast matvec `z[j] = Σ_w a[w]·v[j][w]` for
/// `a=[1,1,1,k]`, `v=[1,r,1,k]` (ANEC `matmul(transpose_y)` form).
/// Every shape-dependent word was recovered by field-diffing ANEC
/// artifacts across 13 `(r,k)` grid points; see the field map below.
/// `k`/`r` are padded to 64 by the caller — zero lanes contribute
/// nothing to the contraction.
pub fn emit_avw(r: u32, k: u32, mil_path: &str, out_path: &str) -> Vec<u8> {
    assert!(r >= 1 && k >= 1);
    let k2 = k * 2; // a bytes / v row stride
    let vbytes = r * k * 2;
    let v_vm = (vbytes + 0x3fff) & !0x3fff;
    let zbytes = r * 64;
    let z_vm = (zbytes + 0x3fff) & !0x3fff;
    let z_va = 0x3000_8000 + (v_vm - 0x4000) as u64; // z seg after v
    let tiles = r * k / 2048 + r / 128 + k / 128;
    let f8014 = r / 256 + r / 512 + k / 256 + k / 1024 + r.min(k) / 128;
    // 0x802c is a low-B-bits lane mask over the c+h sliding window.
    let b = (r / 64 + k / 64 + r.min(k) / 128).max(1);
    assert!(b < 32, "avw lane mask overflows at r={r} k={k}");
    let mask = (1u32 << b) - 1;
    let small = k < 64;

    let mut img = SKELETON_AVW.to_vec();
    let put = |img: &mut Vec<u8>, off: usize, v: u32| {
        img[off..off + 4].copy_from_slice(&v.to_le_bytes());
    };
    // a surface = K*2, v = R*K*2 (+0x4000-aligned vmsize), z = R*64.
    put(&mut img, 0xd8, k2);
    put(&mut img, 0x120, v_vm);
    put(&mut img, 0x170, vbytes);
    // z segment at 0x4000+v_vm; the __text/weights segment follows it
    // (+0x200 lands on its entry header).
    put(&mut img, 0x1b8, z_vm);
    for off in [0x1b0usize, 0x200, 0x368, 0x3f8] {
        put(&mut img, off, 0x3000_0000u32.wrapping_add(0x4000 + v_vm));
    }
    for off in [0x248usize, 0x298, 0x388, 0xb88] {
        put(
            &mut img,
            off,
            0x3000_0000u32.wrapping_add(0x4000 + v_vm + z_vm),
        );
    }
    for off in [0x2e8usize, 0x398] {
        put(
            &mut img,
            off,
            0x3000_0000u32.wrapping_add(0x4200 + v_vm + z_vm),
        );
    }
    put(&mut img, 0x208, zbytes);
    put(&mut img, 0x2a0, if small { 0x1f0 } else { 0x1f8 });
    put(&mut img, 0xb9c, if small { 0x7c } else { 0x7e });
    put(&mut img, 0xc08, tiles);
    put(&mut img, 0xc4c, k);
    for off in [0xc68usize, 0xc70, 0xc78, 0xc88, 0xc98] {
        put(&mut img, off, k2);
    }
    put(&mut img, 0x198c, r);
    put(&mut img, 0x1994, k);
    for off in [0x19b0usize, 0x19d0, 0x19e0] {
        put(&mut img, off, vbytes);
    }
    put(&mut img, 0x19b8, k2);
    put(&mut img, 0x19c0, k2);
    put(&mut img, 0x26d4, r);
    for off in [0x26f8usize, 0x2718, 0x2728] {
        put(&mut img, off, zbytes);
    }
    // Tail: entries carry the three layout-desc strings and z's VA.
    let (mut entries, relocs) = tail_meta(SKELETON_AVW);
    let a_desc =
        format!("a:t24=ar1;0;1;25=s{k2}n:ar1;0;1;26=s{k2}c:ar1;0;1;27=s{k2}h:ar1;0;{k};28=s2w:5");
    let v_desc = format!(
        "v:t29=ar1;0;1;30=s{vbytes}n:ar1;0;{r};31=s{k2}c:ar1;0;1;32=s{k2}h:ar1;0;{k};33=s2w:5"
    );
    let z_desc =
        format!("z:t34=ar1;0;1;35=s{zbytes}n:ar1;0;{r};36=s64c:ar1;0;1;37=s64h:ar1;0;1;38=s2w:5");
    for e in entries.iter_mut() {
        if e.0 == "z" {
            e.4 = z_va;
        } else if e.0.starts_with("a:t24") {
            e.0 = a_desc.clone();
        } else if e.0.starts_with("v:t29") {
            e.0 = v_desc.clone();
        } else if e.0.starts_with("z:t34") {
            e.0 = z_desc.clone();
        }
    }
    let (mut img, text_off) =
        build_tail_at_reloc(&img, 0x33f0, &entries, &relocs, mil_path, out_path, 0x218);
    debug_assert_eq!(text_off, 0x8000);
    img.extend_from_slice(&SKELETON_AVW[0x8000..]);
    // __text task record fields ride at file offset 0x8000.
    put(&mut img, 0x8014, f8014);
    put(&mut img, 0x802c, mask);
    put(&mut img, 0x8038, k);
    put(&mut img, 0x8048, k);
    put(&mut img, 0x8054, r);
    for off in [0x8074usize, 0x80fc, 0x8144] {
        put(&mut img, off, k2);
    }
    // lane-stride fields saturate at 128 elements (0x100 bytes).
    let k2_sat = k.min(128) * 2;
    for off in [0x8090usize, 0x8094, 0x8098, 0x809c] {
        put(&mut img, off, k2_sat);
    }
    put(&mut img, 0x808c, r * k * 16);
    put(&mut img, 0x8194, r * k * 16);
    put(&mut img, 0x80ac, k * 16);
    put(&mut img, 0x8188, k * 16);
    put(&mut img, 0x80d4, tiles);
    put(&mut img, 0x8154, k);
    put(&mut img, 0x8168, r);
    img
}

/// Standalone `softmax(x=[1,1,1,T], axis=-1)` — the attention score
/// normalizer. ANEC lowers it to a multi-task record (exp / reduce /
/// broadcast-divide sub-tasks); the record layout differs across T
/// regimes, so we carry per-T skeletons. A decode pipeline keeps the
/// score window padded to a fixed T_max with masked `-inf` lanes, so
/// one size covers all cache lengths.
const SKELETON_SM64: &[u8] = include_bytes!("../assets/sm64_skeleton.hwx");
const SKELETON_SM256: &[u8] = include_bytes!("../assets/sm256_skeleton.hwx");
const SKELETON_SM1024: &[u8] = include_bytes!("../assets/sm1024_skeleton.hwx");

/// Emit a standalone softmax over `n` elements for n ∈ {64, 256, 1024}.
/// Sig tail at `0x2688`, `__text` at `0x4000` on this family.
pub fn emit_softmax(n: u32, mil_path: &str, out_path: &str) -> Result<Vec<u8>, String> {
    let skel = match n {
        64 => SKELETON_SM64,
        256 => SKELETON_SM256,
        1024 => SKELETON_SM1024,
        _ => return Err(format!("softmax n={n} not in {{64,256,1024}}")),
    };
    let (entries, relocs) = tail_meta(skel);
    Ok(rebuild_tail_at(
        skel, 0x2688, &entries, &relocs, mil_path, out_path,
    ))
}

/// Channel-axis softmax `softmax(x=[1,T,1,1], axis=1)` — the attention
/// score normalizer in **strided** layout: `x[t]`/`z[t]` sit at byte
/// `t*64`, matching `emit_avw`'s output planes exactly, so the score
/// surface adopts straight into this program's input.
const SKELETON_SMC64: &[u8] = include_bytes!("../assets/smc64_skeleton.hwx");
const SKELETON_SMC256: &[u8] = include_bytes!("../assets/smc256_skeleton.hwx");
const SKELETON_SMC1024: &[u8] = include_bytes!("../assets/smc1024_skeleton.hwx");

/// Emit channel-axis softmax over `t` positions for t ∈ {64,256,1024}.
pub fn emit_softmax_ch(t: u32, mil_path: &str, out_path: &str) -> Result<Vec<u8>, String> {
    let skel = match t {
        64 => SKELETON_SMC64,
        256 => SKELETON_SMC256,
        1024 => SKELETON_SMC1024,
        _ => return Err(format!("softmax_ch t={t} not in {{64,256,1024}}")),
    };
    let (entries, relocs) = tail_meta(skel);
    Ok(rebuild_tail_at(
        skel, 0x2688, &entries, &relocs, mil_path, out_path,
    ))
}

/// Attention output contraction `z[j] = Σ_t s[t]·v[t][j]`, emitted as
/// `matmul(x=s[1,1,T,1], y=v[1,1,T,D], transpose_x)`. `s[t]` reads at
/// `t*64` — byte-identical to the channel-axis softmax output, so the
/// chain `scores → softmax → attn·V` adopts surfaces end to end.
const SKELETON_AV64X32: &[u8] = include_bytes!("../assets/attnv_64x32_skeleton.hwx");
const SKELETON_AV64X64: &[u8] = include_bytes!("../assets/attnv_64x64_skeleton.hwx");
const SKELETON_AV64X128: &[u8] = include_bytes!("../assets/attnv_64x128_skeleton.hwx");
const SKELETON_AV256X64: &[u8] = include_bytes!("../assets/attnv_256x64_skeleton.hwx");
const SKELETON_AV1024X64: &[u8] = include_bytes!("../assets/attnv_1024x64_skeleton.hwx");
const SKELETON_AV1024X128: &[u8] = include_bytes!("../assets/attnv_1024x128_skeleton.hwx");

/// Outer product `z[c][j] = e[c]·r[j]` — `matmul(x=e[1,T,1,1],
/// y=r[1,1,1,K])` → `z[1,T,1,K]`. With `e` a one-hot this scatters row
/// `r` into row `t` of a `[1,T,1,K]` frame — the on-device KV-cache row
/// append (`cache += outer(e_t, kv_t)` via a flat ew add).
const SKELETON_OUTER64: &[u8] = include_bytes!("../assets/outer64_skeleton.hwx");
const SKELETON_OUTER256: &[u8] = include_bytes!("../assets/outer256_skeleton.hwx");
const SKELETON_OUTER512: &[u8] = include_bytes!("../assets/outer512_skeleton.hwx");
const SKELETON_OUTER1024: &[u8] = include_bytes!("../assets/outer1024_skeleton.hwx");

/// Emit the outer-product row scatter for `(t, d)` on the skeleton
/// table — `t` is the cache/frame length, `d` the channel width.
pub fn emit_outer_t(t: u32, d: u32, mil_path: &str, out_path: &str) -> Result<Vec<u8>, String> {
    let skel = match (t, d) {
        (64, 64) => SKELETON_OUTER64,
        (256, 64) => SKELETON_OUTER256,
        (512, 64) => SKELETON_OUTER512,
        (1024, 64) => SKELETON_OUTER1024,
        _ => return Err(format!("outer ({t},{d}) off skeleton table")),
    };
    let (entries, relocs) = tail_meta(skel);
    // __text at 0x8000 like the attnv family
    rebuild_tail_at_text(skel, 0x33f0, 0x8000, &entries, &relocs, mil_path, out_path)
        .map_err(|e| e.to_string())
}

/// Emit the outer-product row scatter for `t=d=64`.
pub fn emit_outer(mil_path: &str, out_path: &str) -> Vec<u8> {
    emit_outer_t(64, 64, mil_path, out_path).expect("outer64 tail fits under __text")
}

/// Packed 4-D elementwise add `z = x + y` on `[1,64,1,64]` — element
/// `(c,j)` at `c*128+j*2`, byte-identical layout to the outer-product
/// output and the `avw`/`attn_v` cache frames, so it drops into the
/// KV-append slot (`cache += outer(e_t, kv_t)` in place via aliasing).
const SKELETON_ADD4D64: &[u8] = include_bytes!("../assets/add4d64_skeleton.hwx");
const SKELETON_ADD4D256: &[u8] = include_bytes!("../assets/add4d256_skeleton.hwx");
const SKELETON_ADD4D512: &[u8] = include_bytes!("../assets/add4d512_skeleton.hwx");
const SKELETON_ADD4D1024: &[u8] = include_bytes!("../assets/add4d1024_skeleton.hwx");

/// Packed 4-D add on `[1,T,1,D]` for `(t,d)` on the skeleton table.
pub fn emit_add4d(t: u32, d: u32, mil_path: &str, out_path: &str) -> Result<Vec<u8>, String> {
    let skel = match (t, d) {
        (64, 64) => SKELETON_ADD4D64,
        (256, 64) => SKELETON_ADD4D256,
        (512, 64) => SKELETON_ADD4D512,
        (1024, 64) => SKELETON_ADD4D1024,
        _ => return Err(format!("add4d ({t},{d}) off skeleton table")),
    };
    let (entries, relocs) = tail_meta(skel);
    rebuild_tail_at_text(skel, 0x33f0, 0x8000, &entries, &relocs, mil_path, out_path)
        .map_err(|e| e.to_string())
}

pub fn emit_add4d64(mil_path: &str, out_path: &str) -> Vec<u8> {
    emit_add4d(64, 64, mil_path, out_path).expect("add4d64 tail fits under __text")
}

/// Strided `[1,64,1,1]` add — the causal-mask op: `scores += mask`
/// where mask[t] = 0 for t ≤ now, -inf (fp16 `-65504`) beyond. Strides
/// match the `avw`/`smc` score surface exactly.
const SKELETON_MASKADD64: &[u8] = include_bytes!("../assets/maskadd64_skeleton.hwx");
const SKELETON_MASKADD256: &[u8] = include_bytes!("../assets/maskadd256_skeleton.hwx");

/// Strided `[1,T,1,1]` add for `t` on the skeleton table.
pub fn emit_maskadd(t: u32, mil_path: &str, out_path: &str) -> Result<Vec<u8>, String> {
    let skel = match t {
        64 => SKELETON_MASKADD64,
        256 => SKELETON_MASKADD256,
        _ => return Err(format!("maskadd t={t} off skeleton table")),
    };
    let (entries, relocs) = tail_meta(skel);
    rebuild_tail_at_text(skel, 0x33f0, 0x8000, &entries, &relocs, mil_path, out_path)
        .map_err(|e| e.to_string())
}

pub fn emit_maskadd64(mil_path: &str, out_path: &str) -> Vec<u8> {
    emit_maskadd(64, mil_path, out_path).expect("maskadd64 tail fits under __text")
}

/// Layout repack `transpose(x=[1,64,1,1], perm=[0,2,3,1])` →
/// `z[1,1,1,64]` — strided channel-planes → contiguous w-packed.
/// Adapts `emit_matmul_ns` (channel-strided out) onto the contiguous
/// vector operands `emit_avw`/`emit_outer` expect.
const SKELETON_REPACK64: &[u8] = include_bytes!("../assets/repack64_skeleton.hwx");

pub fn emit_repack64(mil_path: &str, out_path: &str) -> Vec<u8> {
    let (entries, relocs) = tail_meta(SKELETON_REPACK64);
    rebuild_tail_at(
        SKELETON_REPACK64,
        0x25f0,
        &entries,
        &relocs,
        mil_path,
        out_path,
    )
}

/// 128-wide repack `transpose(x=[1,128,1,1], perm=[0,2,3,1])` →
/// `z[1,1,1,128]` — FFN intermediate width counterpart of
/// [`emit_repack64`].
const SKELETON_RPK128: &[u8] = include_bytes!("../assets/rpk128_skeleton.hwx");

pub fn emit_repack128(mil_path: &str, out_path: &str) -> Vec<u8> {
    emit_repack(128, mil_path, out_path).expect("repack128 tail fits")
}

/// 256-wide repack — same transpose task at FFN width 256.
const SKELETON_RPK256: &[u8] = include_bytes!("../assets/rpk256_skeleton.hwx");
const SKELETON_RPK512: &[u8] = include_bytes!("../assets/rpk512_skeleton.hwx");
const SKELETON_RPK1024: &[u8] = include_bytes!("../assets/rpk1024_skeleton.hwx");

/// Strided `[1,N,1,1]` -> contiguous `[1,1,1,N]` repack for `n` on the
/// skeleton table `{64,128,256}`.
pub fn emit_repack(n: u32, mil_path: &str, out_path: &str) -> Result<Vec<u8>, String> {
    let skel = match n {
        64 => SKELETON_REPACK64,
        128 => SKELETON_RPK128,
        256 => SKELETON_RPK256,
        512 => SKELETON_RPK512,
        1024 => SKELETON_RPK1024,
        _ => return Err(format!("repack n={n} off skeleton table")),
    };
    let (entries, relocs) = tail_meta(skel);
    Ok(rebuild_tail_at(
        skel, 0x25f0, &entries, &relocs, mil_path, out_path,
    ))
}

/// Strided `sigmoid(x=[1,128,1,1])` — elementwise on the channel axis,
/// same plane layout as the strided vectors; fits `gate·σ(gate)` SiLU
/// without leaving the strided currency.
const SKELETON_SIG128: &[u8] = include_bytes!("../assets/sig128_skeleton.hwx");
const SKELETON_SIG256: &[u8] = include_bytes!("../assets/sig256_skeleton.hwx");
const SKELETON_SIG512: &[u8] = include_bytes!("../assets/sig512_skeleton.hwx");
const SKELETON_SIG1024: &[u8] = include_bytes!("../assets/sig1024_skeleton.hwx");

/// Strided sigmoid `[1,N,1,1]` for `n` on the skeleton table
/// `{128,256}`.
pub fn emit_sigmoid_strided(n: u32, mil_path: &str, out_path: &str) -> Result<Vec<u8>, String> {
    let skel = match n {
        128 => SKELETON_SIG128,
        256 => SKELETON_SIG256,
        512 => SKELETON_SIG512,
        1024 => SKELETON_SIG1024,
        _ => return Err(format!("sigmoid strided n={n} off skeleton table")),
    };
    let (entries, relocs) = tail_meta(skel);
    Ok(rebuild_tail_at(
        skel, 0x2688, &entries, &relocs, mil_path, out_path,
    ))
}

pub fn emit_sigmoid128(mil_path: &str, out_path: &str) -> Vec<u8> {
    let (entries, relocs) = tail_meta(SKELETON_SIG128);
    rebuild_tail_at(
        SKELETON_SIG128,
        0x2688,
        &entries,
        &relocs,
        mil_path,
        out_path,
    )
}

/// Strided `mul(x=[1,128,1,1], y=[1,128,1,1])` — SiLU's multiply.
const SKELETON_MUL128: &[u8] = include_bytes!("../assets/mul128_skeleton.hwx");

pub fn emit_mul128(mil_path: &str, out_path: &str) -> Vec<u8> {
    let (entries, relocs) = tail_meta(SKELETON_MUL128);
    rebuild_tail_at_text(
        SKELETON_MUL128,
        0x33f0,
        0x8000,
        &entries,
        &relocs,
        mil_path,
        out_path,
    )
    .expect("mul128 tail fits under __text")
}

/// Emit the attn·V matvec for `(t, d)` pairs on the skeleton table.
pub fn emit_attn_v(t: u32, d: u32, mil_path: &str, out_path: &str) -> Result<Vec<u8>, String> {
    let skel = match (t, d) {
        (64, 32) => SKELETON_AV64X32,
        (64, 64) => SKELETON_AV64X64,
        (64, 128) => SKELETON_AV64X128,
        (256, 64) => SKELETON_AV256X64,
        (1024, 64) => SKELETON_AV1024X64,
        (1024, 128) => SKELETON_AV1024X128,
        _ => return Err(format!("attn_v (t={t},d={d}) not on skeleton table")),
    };
    let (entries, relocs) = tail_meta(skel);
    // this family keeps __text at fileoff 0x8000, not 0x4000
    rebuild_tail_at_text(skel, 0x34a8, 0x8000, &entries, &relocs, mil_path, out_path)
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
    if shape == MmShape::N16 {
        // sub-tile pool: 16 weights per 0x40-byte band
        for (i, v) in weights.iter().enumerate() {
            let o = off + (i / 16) * 0x40 + (i % 16) * 2;
            img[o..o + 2].copy_from_slice(&v.to_le_bytes());
        }
    } else {
        for (i, v) in weights.iter().enumerate() {
            img[off + i * 2..off + i * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    Ok(img)
}

// ---------------------------------------------------------------------------
// MIL-driven dispatch — `milc compile --target ane --emit-hwx`.
// ---------------------------------------------------------------------------

/// A compute op extracted from MIL text: op type, result shape, and
/// the operand names we inspect (`x`/`y`/`weight`).
#[derive(Clone, Debug)]
pub(crate) struct MilOp {
    pub(crate) op: String,
    /// Result tensor name (the lhs identifier).
    pub(crate) result: String,
    pub(crate) shape: Vec<i64>,
    pub(crate) args: Vec<(String, String)>,
}

/// Scan MIL text for compute ops (non-`const` declarations of the
/// form `tensor<dtype, [shape]> name = op(arg = operand, ...)`) and
/// blob-const bindings (`name = const()[..BLOBFILE(path = string("P"))]`).
pub(crate) fn mil_ops(text: &str) -> (Vec<MilOp>, Vec<(String, String)>) {
    let mut ops = Vec::new();
    let mut blobs = Vec::new();
    // Op declarations may wrap after `=` — join continuation lines so
    // `name =\n    op(...)` parses as one statement.
    let joined = {
        let mut out = String::with_capacity(text.len());
        for raw in text.lines() {
            let line = raw.trim_end();
            if out.ends_with(" =")
                && line
                    .trim_start()
                    .starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
            {
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
            .zip(lhs.find(']'))
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
        let result = lhs
            .rsplit([' ', '>'])
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        if op.starts_with("constexpr_") {
            blobs.push((result, "__constexpr__".to_string()));
            continue;
        }
        ops.push(MilOp {
            op: op.to_string(),
            result,
            shape,
            args,
        });
    }
    (ops, blobs)
}

/// Parse the `func main<...>(tensor<fp16, [dims]> name, ...)` signature
/// into `(name, shape)` pairs — operand shapes for ops whose inputs are
/// function parameters (matmul's M lives on `x`, not the result).
pub(crate) fn func_input_shapes(text: &str) -> Vec<(String, Vec<i64>)> {
    let mut out = Vec::new();
    // the func signature may wrap across lines — accumulate until `{`
    let mut in_sig = false;
    for raw in text.lines() {
        let line = raw.trim();
        if !in_sig {
            if !line.contains("func ") {
                continue;
            }
            in_sig = true;
        }
        let (pre, done) = match line.find('{') {
            Some(p) => (&line[..p], true),
            None => (line, false),
        };
        let mut rest = pre;
        while let Some(t) = rest.find("tensor<") {
            let after = &rest[t + 7..];
            let Some(l) = after.find('[') else { break };
            let Some(r) = after[l..].find(']') else { break };
            let shape: Vec<i64> = after[l + 1..l + r]
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            let name = after[l + r + 1..]
                .trim_start_matches(['>', ' ', ','])
                .split([' ', ',', ')'])
                .next()
                .unwrap_or("")
                .to_string();
            out.push((name, shape));
            rest = &after[l + r + 1..];
        }
        if done {
            break;
        }
    }
    out
}

#[derive(Clone, Debug)]
struct BlobDecl {
    dtype: String,
    shape: Vec<usize>,
    path: String,
    offset: u64,
}

fn load_blob_bytes_at(
    path: &std::path::Path,
    offset: u64,
    raw_len: usize,
) -> Result<Vec<u8>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let off = offset as usize;
    if off + 64 <= b.len() && &b[off..off + 4] == b"\xef\xbe\xad\xde" {
        let sz = u64::from_le_bytes(b[off + 8..off + 16].try_into().unwrap()) as usize;
        let doff = u64::from_le_bytes(b[off + 16..off + 24].try_into().unwrap()) as usize;
        if doff + sz > b.len() {
            return Err(format!("{}: truncated blob at {off}", path.display()));
        }
        return Ok(b[doff..doff + sz].to_vec());
    }
    if off == 0 && b.len() > 132 && &b[64..68] == b"\xef\xbe\xad\xde" {
        let sz = u64::from_le_bytes(b[72..80].try_into().unwrap()) as usize;
        let doff = u64::from_le_bytes(b[80..88].try_into().unwrap()) as usize;
        if doff + sz <= b.len() {
            return Ok(b[doff..doff + sz].to_vec());
        }
        return Ok(b[128..].to_vec());
    }
    if raw_len > 0 && off + raw_len <= b.len() {
        Ok(b[off..off + raw_len].to_vec())
    } else if off <= b.len() {
        Ok(b[off..].to_vec())
    } else {
        Err(format!(
            "{}: offset {off} past EOF {}",
            path.display(),
            b.len()
        ))
    }
}

fn decode_blob_f32(d: &BlobDecl, mil_dir: &std::path::Path) -> Result<Vec<f32>, String> {
    let numel: usize = if d.shape.is_empty() {
        0
    } else {
        d.shape.iter().product()
    };
    let raw_len = match d.dtype.as_str() {
        "int8" | "uint8" => numel,
        "uint4" | "int4" => numel.div_ceil(2),
        _ => numel * 2,
    };
    let raw = load_blob_bytes_at(&resolve_blob(mil_dir, &d.path), d.offset, raw_len)?;
    Ok(match d.dtype.as_str() {
        "int8" => raw.iter().map(|&b| (b as i8) as f32).collect(),
        "uint8" => raw.iter().map(|&b| b as f32).collect(),
        "uint4" => {
            let mut v = Vec::with_capacity(raw.len() * 2);
            for &b in &raw {
                v.push((b & 0x0f) as f32);
                v.push((b >> 4) as f32);
            }
            if numel > 0 {
                v.truncate(numel);
            }
            v
        }
        "int4" => {
            let s4 = |n: u8| (((n as i8) << 4) >> 4) as f32;
            let mut v = Vec::with_capacity(raw.len() * 2);
            for &b in &raw {
                v.push(s4(b & 0x0f));
                v.push(s4(b >> 4));
            }
            if numel > 0 {
                v.truncate(numel);
            }
            v
        }
        _ => raw
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
    })
}

/// Resolve a MIL weight operand by name — supports direct `BLOBFILE`
/// `const()` declarations (with `offset = uint64(...)` in multi-blob
/// `weight.bin` files or bare files), `constexpr_blockwise_shift_scale`
/// (per-channel int8, blockwise Q8_0 int8, blockwise Q4_0 uint4/int8
/// with offset), and `constexpr_lut_to_dense` (palette4).
pub(crate) fn load_mil_weight_u16(
    text: &str,
    mil_dir: &std::path::Path,
    target: &str,
) -> Result<Vec<u16>, String> {
    let joined = {
        let mut out = String::with_capacity(text.len());
        for raw in text.lines() {
            let line = raw.trim_end();
            if out.ends_with(" =")
                && line
                    .trim_start()
                    .starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
            {
                out.push(' ');
                out.push_str(line.trim_start());
            } else {
                out.push('\n');
                out.push_str(line);
            }
        }
        out
    };
    let mut decls: std::collections::HashMap<String, BlobDecl> = std::collections::HashMap::new();
    let mut cexprs: std::collections::HashMap<String, (String, Vec<(String, String)>)> =
        std::collections::HashMap::new();
    for raw in joined.lines() {
        let line = raw.trim();
        let Some(eq) = line.find(" = ") else { continue };
        let (lhs, rhs) = line.split_at(eq);
        let rhs = &rhs[3..];
        let name = lhs
            .rsplit([' ', '>'])
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        let dtype = lhs
            .find("tensor<")
            .and_then(|t| {
                let after = &lhs[t + 7..];
                after
                    .find([',', '>'])
                    .map(|e| after[..e].trim().to_string())
            })
            .unwrap_or_else(|| "fp16".to_string());
        let shape: Vec<usize> = lhs
            .find('[')
            .zip(lhs.find(']'))
            .map(|(l, r)| {
                lhs[l + 1..r]
                    .split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .collect()
            })
            .unwrap_or_default();
        const BLOB_PFX: &str = "BLOBFILE(path = string(\"";
        if let Some(bp) = rhs.find(BLOB_PFX) {
            let rest = &rhs[bp + BLOB_PFX.len()..];
            if let Some(end) = rest.find('"') {
                let path = rest[..end].to_string();
                let offset = rhs
                    .find("offset = uint64(")
                    .and_then(|p| {
                        let a = &rhs[p + 16..];
                        a.find(')').and_then(|e| a[..e].trim().parse::<u64>().ok())
                    })
                    .unwrap_or(0);
                decls.insert(
                    name,
                    BlobDecl {
                        dtype,
                        shape,
                        path,
                        offset,
                    },
                );
            }
            continue;
        }
        let Some(paren) = rhs.find('(') else { continue };
        let op = rhs[..paren].trim();
        if op.starts_with("constexpr_") {
            let close = rhs[paren..]
                .find(")[")
                .or_else(|| rhs[paren..].find(");"))
                .map(|i| paren + i)
                .unwrap_or(rhs.len());
            let args: Vec<(String, String)> = rhs[paren + 1..close]
                .split(',')
                .filter_map(|a| {
                    let (k, v) = a.split_once('=')?;
                    Some((k.trim().to_string(), v.trim().to_string()))
                })
                .collect();
            cexprs.insert(name, (op.to_string(), args));
        }
    }
    if let Some(d) = decls.get(target) {
        let numel: usize = if d.shape.is_empty() {
            0
        } else {
            d.shape.iter().product()
        };
        let raw = load_blob_bytes_at(&resolve_blob(mil_dir, &d.path), d.offset, numel * 2)?;
        return Ok(raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect());
    }
    if let Some((op, args)) = cexprs.get(target) {
        let arg = |k: &str| -> Option<&str> {
            args.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
        };
        match op.as_str() {
            "constexpr_blockwise_shift_scale" => {
                let dname = arg("data").ok_or("constexpr_blockwise_shift_scale missing data")?;
                let sname = arg("scale").ok_or("constexpr_blockwise_shift_scale missing scale")?;
                let dd = decls
                    .get(dname)
                    .ok_or_else(|| format!("data `{dname}` not a blob const"))?;
                let sd = decls
                    .get(sname)
                    .ok_or_else(|| format!("scale `{sname}` not a blob const"))?;
                let dv = decode_blob_f32(dd, mil_dir)?;
                let sv = decode_blob_f32(sd, mil_dir)?;
                let ov = match arg("offset") {
                    Some(oname) => {
                        let od = decls
                            .get(oname)
                            .ok_or_else(|| format!("offset `{oname}` not a blob const"))?;
                        Some(decode_blob_f32(od, mil_dir)?)
                    }
                    None => None,
                };
                let mut out = Vec::with_capacity(dv.len());
                if dd.shape.len() == sd.shape.len() && !dd.shape.is_empty() {
                    let r = dd.shape.len();
                    let mut d_str = vec![1usize; r];
                    let mut s_str = vec![1usize; r];
                    for m in (0..r - 1).rev() {
                        d_str[m] = d_str[m + 1] * dd.shape[m + 1].max(1);
                        s_str[m] = s_str[m + 1] * sd.shape[m + 1].max(1);
                    }
                    let blk: Vec<usize> = (0..r)
                        .map(|m| (dd.shape[m].max(1) / sd.shape[m].max(1)).max(1))
                        .collect();
                    for (idx, &x) in dv.iter().enumerate() {
                        let mut rem = idx;
                        let mut s_idx = 0usize;
                        for m in 0..r {
                            let c = rem / d_str[m];
                            rem %= d_str[m];
                            let sc = (c / blk[m]).min(sd.shape[m].max(1) - 1);
                            s_idx += sc * s_str[m];
                        }
                        let z = ov.as_ref().map(|o| o[s_idx]).unwrap_or(0.0);
                        out.push(half::f16::from_f32((x - z) * sv[s_idx]).to_bits());
                    }
                } else {
                    let blk = (dv.len() / sv.len().max(1)).max(1);
                    for (idx, &x) in dv.iter().enumerate() {
                        let s_idx = (idx / blk).min(sv.len() - 1);
                        let z = ov.as_ref().map(|o| o[s_idx]).unwrap_or(0.0);
                        out.push(half::f16::from_f32((x - z) * sv[s_idx]).to_bits());
                    }
                }
                return Ok(out);
            }
            "constexpr_lut_to_dense" => {
                let iname = arg("indices").ok_or("constexpr_lut_to_dense missing indices")?;
                let lname = arg("lut").ok_or("constexpr_lut_to_dense missing lut")?;
                let id = decls
                    .get(iname)
                    .ok_or_else(|| format!("indices `{iname}` not a blob const"))?;
                let ld = decls
                    .get(lname)
                    .ok_or_else(|| format!("lut `{lname}` not a blob const"))?;
                let codes = decode_blob_f32(id, mil_dir)?;
                let lut = decode_blob_f32(ld, mil_dir)?;
                let rows = id.shape.first().copied().unwrap_or(1).max(1);
                let row_len = (codes.len() / rows).max(1);
                let mut out = Vec::with_capacity(codes.len());
                for (i, &c) in codes.iter().enumerate() {
                    let r = (i / row_len).min(rows - 1);
                    let k = (c as usize) & 0x0f;
                    out.push(half::f16::from_f32(lut[r * 16 + k]).to_bits());
                }
                return Ok(out);
            }
            other => return Err(format!("unsupported constexpr `{other}`")),
        }
    }
    Err(format!(
        "weight `{target}` is not a blob or constexpr const"
    ))
}

/// Resolve a `BLOBFILE` path — `@model_path/` prefixes and relative
/// paths resolve against the MIL's directory.
pub(crate) fn resolve_blob<'a>(mil_dir: &'a std::path::Path, p: &'a str) -> std::path::PathBuf {
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
pub(crate) fn mil_scalar_f16(text: &str, name: &str) -> Option<u16> {
    for raw in text.lines() {
        let line = raw.trim();
        let Some(eq) = line.find(" = ") else { continue };
        if line[..eq].rsplit([' ', '>']).next() != Some(name) {
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

/// Resolve a scalar `bool` const's value: `bool <name> =
/// const()[val = bool(true|false)]`. `None` when not a bool const.
pub(crate) fn mil_scalar_bool(text: &str, name: &str) -> Option<bool> {
    for raw in text.lines() {
        let line = raw.trim();
        let Some(eq) = line.find(" = ") else { continue };
        if line[..eq].rsplit([' ', '>']).next() != Some(name) {
            continue;
        }
        let rhs = &line[eq + 3..];
        let v = rhs.find("bool(")?;
        let end = rhs[v..].find(')')? + v;
        return Some(&rhs[v + 5..end] == "true");
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
///
/// Real MIL tensor names are substituted into the emitted artifact —
/// arbitrary `func main` io names work, including names that reorder
/// the alphabetical section ranks.
pub fn emit_from_mil(mil_path: &str, out_path: &str) -> Result<Vec<u8>, String> {
    let img = emit_from_mil_inner(mil_path, out_path)?;
    let text = std::fs::read_to_string(mil_path).map_err(|e| format!("{mil_path}: {e}"))?;
    let Some(mut io) = mil_io_sig(&text) else {
        return Ok(img);
    };
    let (ops, _) = mil_ops(&text);
    for o in &ops {
        let keys: &[&str] = match o.op.as_str() {
            "matmul" => &["x", "y"],
            "conv" => &["x"],
            _ => continue,
        };
        for (k, v) in &o.args {
            let Some(i) = io.ins.iter().position(|n| n == v) else {
                continue;
            };
            let spatial = io.in_dims[i].iter().skip(2).product::<usize>() > 1;
            if keys.contains(&k.as_str()) && (o.op == "matmul" || spatial) {
                io.relaid.push(v.clone());
            }
        }
    }
    // skeleton-name -> MIL-name role convention: skeletons were captured
    // from MILs whose io names were x/y/m/p/z/sc; each name's role is a
    // fixed DAG operand slot shared by every family using it.
    let arg_of = |opname: &str, key: &str| -> Option<String> {
        ops.iter().find(|o| o.op == opname).and_then(|o| {
            o.args
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
        })
    };
    let first_arg = |key: &str| -> Option<String> {
        ops.first().and_then(|o| {
            o.args
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
        })
    };
    let out_name = io.out.clone();
    let resolve = move |skel: &str| -> Option<String> {
        match skel {
            "x" | "a" => first_arg("x"),
            "y" | "v" => first_arg("y").or_else(|| arg_of("matmul", "y")),
            "m" => arg_of("add", "y"),
            "sc" => arg_of("mul", "y"),
            "z" | "p" => Some(out_name.clone()),
            _ => None,
        }
    };
    apply_io_names(&img, &io, &resolve)
}

fn emit_from_mil_inner(mil_path: &str, out_path: &str) -> Result<Vec<u8>, String> {
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
        op.args.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone())
    };
    let weights_of = |op: &MilOp| -> Result<Vec<u16>, String> {
        let wname = arg(op, "weight").ok_or("conv without weight operand")?;
        load_mil_weight_u16(&text, &mil_dir, &wname)
    };
    let names: Vec<&str> = ops.iter().map(|o| o.op.as_str()).collect();
    let elems = |o: &MilOp| -> u32 { o.shape.iter().map(|d| (*d).max(1) as u32).product() };

    // ---- single-op families ----
    match names.as_slice() {
        ["conv"] => {
            let o = &ops[0];
            let (c, h, w) = (
                o.shape.get(1).copied().unwrap_or(0),
                o.shape.get(2).copied().unwrap_or(1),
                o.shape.get(3).copied().unwrap_or(1),
            );
            // `[1,Cout,1,W]` conv1x1 with both dims >=32 goes through
            // the parametric emitter — byte-verified against ANEC for
            // every square `C` in 32..=4096 and a dense non-square
            // `(Cin,Cout)` grid, plus the spatial W sweep (W is the
            // sequence dim: the emitted program is an M=W channel-major
            // matvec). The square 713..=720 variant window still errors
            // cleanly.
            if h == 1 && c >= 32 {
                let weights = weights_of(o)?;
                let cout = c as usize;
                if weights.len() % cout == 0 {
                    let cin = weights.len() / cout;
                    if cin >= 32 {
                        return emit_conv1x1_w(cin, cout, w as usize, &weights, mil_path, out_path);
                    }
                }
            }
            let w = h;
            let shape = match (c, w) {
                (64, 5) => ConvShape::C64W5,
                (64, 8) => ConvShape::C64W8,
                (16, 8) => ConvShape::C16W8,
                (32, 8) => ConvShape::C32W8,
                (128, 8) => ConvShape::C128W8,
                _ => return Err(format!("conv shape {:?} not on a skeleton family", o.shape)),
            };
            return emit_conv1x1(shape, &weights_of(o)?, mil_path, out_path);
        }
        ["add"] => {
            return Ok(emit_elementwise2(
                EwOp::Add,
                elems(&ops[0]),
                mil_path,
                out_path,
            ))
        }
        ["mul"] => {
            return Ok(emit_elementwise2(
                EwOp::Mul,
                elems(&ops[0]),
                mil_path,
                out_path,
            ))
        }
        ["sub"] => {
            return Ok(emit_elementwise2(
                EwOp::Sub,
                elems(&ops[0]),
                mil_path,
                out_path,
            ))
        }
        ["minimum"] => {
            return Ok(emit_elementwise2(
                EwOp::Min,
                elems(&ops[0]),
                mil_path,
                out_path,
            ))
        }
        ["maximum"] => {
            return Ok(emit_elementwise2(
                EwOp::Max,
                elems(&ops[0]),
                mil_path,
                out_path,
            ))
        }
        ["softmax"] => {
            let n = elems(&ops[0]);
            return emit_softmax(n, mil_path, out_path);
        }
        ["relu"] => return Ok(emit_relu(mil_path, out_path)),
        ["sigmoid"] => {
            return Ok(emit_unary_lut(
                LutOp::Sigmoid,
                elems(&ops[0]),
                mil_path,
                out_path,
            ))
        }
        ["tanh"] => {
            return Ok(emit_unary_lut(
                LutOp::Tanh,
                elems(&ops[0]),
                mil_path,
                out_path,
            ))
        }
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
                if arg(o1, "x").as_deref() != Some("x") || arg(o1, "y").as_deref() != Some("y") {
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
                    return Err("elementwise chain op2's free operand must be x or y".into());
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
        let is32 = shape == &[1, 32, 1, 32];
        let is64 = shape == &[1, 64, 1, 64];
        let is4d = shape == &[1, 8, 64, 64];
        match (names, is32, is64, is4d) {
            (["matmul"], true, _, _) => Some(MmPart::Act32),
            (["matmul"], _, true, _) => Some(MmPart::Act64),
            (["matmul"], _, _, true) => Some(MmPart::Act4d),
            (["matmul", "softmax"], _, true, _) => Some(MmPart::MmSm64),
            (["matmul", "softmax"], _, _, true) => Some(MmPart::MmSm4d),
            (["softmax", "matmul"], _, true, _) => Some(MmPart::SmMm64),
            (["matmul", "add", "softmax"] | ["matmul", "mul", "add", "softmax"], _, true, _) => {
                Some(MmPart::MmAddSm64)
            }
            (["matmul", "mul", "softmax"], _, true, _) => Some(MmPart::MmMulSm64),
            _ => None,
        }
    };
    if let ["matmul"] = names.as_slice() {
        // const-weight matmul → the MmShape families (square `[1,N,1,N]`
        // activation) or the parametric M=1 non-square emitter
        // (`[1,1,1,K] x [1,N,1,K]`, transpose_y).
        let y = arg(&ops[0], "y").unwrap_or_default();
        if blob_of(&y).is_some() {
            let o = &ops[0].shape;
            let m = o.get(1).copied().unwrap_or(0) as usize;
            let n = o.get(3).copied().unwrap_or(0) as usize;
            // ANEC's matmul grammar carries the sequence dim on the
            // *input*: `x=[1,1,M,K]` (or 2-D `[M,K]`), `z=[1,1,1,N]`.
            let xname = arg(&ops[0], "x").unwrap_or_default();
            let xs = func_input_shapes(&text)
                .into_iter()
                .find(|(n, _)| n == &xname)
                .map(|(_, s)| s)
                .unwrap_or_default();
            let seq = if xs.len() == 2 {
                xs[0].max(1) as usize
            } else {
                xs.get(2).copied().unwrap_or(1).max(1) as usize
            };
            let w = load_mil_weight_u16(&text, &mil_dir, &y)?;
            let ty = match arg(&ops[0], "transpose_y").as_deref() {
                Some(v) => mil_scalar_bool(&text, v).unwrap_or(v == "true"),
                None => false,
            };
            if m == 1 && ty && n >= 32 && w.len() % n == 0 {
                let k = w.len() / n;
                if k >= 32 {
                    if seq == 1 {
                        return emit_matmul_ns(k, n, &w, mil_path, out_path);
                    }
                    // M>1 `[M,K]x[K,N]`: the conv1x1 spatial task is the
                    // same math with the channel-major contract — x as
                    // `[K,M]` (one 64B-stride channel row per K), z as
                    // `[N,M]` positions — which is also how interior
                    // buffers flow through conv chains anyway.
                    return emit_conv1x1_w(k, n, seq, &w, mil_path, out_path);
                }
            }
            let shape = match m {
                16 => MmShape::N16,
                32 => MmShape::N32,
                64 => MmShape::N64,
                128 => MmShape::N128,
                256 => MmShape::N256,
                _ => {
                    return Err(format!(
                        "matmul shape {:?} not on a skeleton family",
                        ops[0].shape
                    ))
                }
            };
            return emit_matmul(shape, &w, mil_path, out_path);
        }
        // Broadcast-vector×matrix (`avw`): `matmul(x=[1,1,1,K],
        // y=[1,R,1,K], transpose_y)` — both operands runtime. The
        // engine computes `z[c][h]=a·V[c+h]`; z row 0 is the matvec.
        // Covers decode-time QKᵀ scores (`a=q`, `V=K[t][d]`) and
        // `attn·V` (`a=scores`, `V=Vᵀ[j][t]` — transposed cache).
        let (xname, yname) = (arg(&ops[0], "x"), arg(&ops[0], "y"));
        let sigs = func_input_shapes(&text);
        let shape_of = |n: &Option<String>| -> Vec<i64> {
            sigs.iter()
                .find(|(nm, _)| Some(nm) == n.as_ref().map(|s| s))
                .map(|(_, s)| s.clone())
                .unwrap_or_default()
        };
        let (xs, ys) = (shape_of(&xname), shape_of(&yname));
        let ty = match arg(&ops[0], "transpose_y").as_deref() {
            Some(v) => mil_scalar_bool(&text, v).unwrap_or(v == "true"),
            None => false,
        };
        if ty
            && xs.len() == 4
            && ys.len() == 4
            && xs[..3] == [1, 1, 1]
            && ys[0] == 1
            && ys[2] == 1
            && xs[3] == ys[3]
        {
            return Ok(emit_avw(
                ys[1].max(1) as u32,
                xs[3].max(1) as u32,
                mil_path,
                out_path,
            ));
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
    if names.first() != Some(&"conv")
        || !names.iter().all(|n| {
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
        })
    {
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
    let uniform = layers.iter().all(|(_, a)| a == aux);
    if !uniform {
        return Err("chain layers have non-uniform aux-op patterns".into());
    }
    match aux {
        [] => {
            let mut shapes = Vec::with_capacity(k);
            let mut flat_w = Vec::new();
            let mut offs = Vec::with_capacity(k);
            for (i, (o, _)) in layers.iter().enumerate() {
                let cout = o.shape.get(1).copied().unwrap_or(0).max(1) as usize;
                let cin = ws[i].len() / cout;
                shapes.push((cin, cout));
                flat_w.extend_from_slice(&ws[i]);
                offs.push((128 + i * (cin * cout * 2 + 64)) as u64);
            }
            let out_name = &layers.last().unwrap().0.result;
            emit_conv_chain(&shapes, &flat_w, &offs, out_name, mil_path, out_path)
        }
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

// ---------------------------------------------------------------------------
// IO-name substitution
// ---------------------------------------------------------------------------

/// Real io names parsed from a MIL `func main` signature: inputs in
/// declaration order plus the single output.
#[derive(Debug, Default)]
struct IoSig {
    ins: Vec<String>,
    out: String,
    /// Declared shape of each input (parallel to `ins`).
    in_dims: Vec<Vec<usize>>,
    /// Inputs ANEC re-lays out internally — direct `matmul` operands
    /// and spatial (`H*W > 1`) `conv` inputs. Their symtab heads trail
    /// the rest (see [`apply_io_names`]).
    relaid: Vec<String>,
}

/// Parse `func main<…>(tensor<..> a, tensor<..> b) { .. } -> (out)` —
/// parameter names are the last token of each `tensor<..> name` entry.
fn mil_io_sig(text: &str) -> Option<IoSig> {
    let fpos = text.find("func main")?;
    let paren = text[fpos..].find('(')? + fpos;
    let (mut depth, mut close) = (0usize, paren + 1);
    for (i, c) in text[paren..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = paren + i;
                    break;
                }
            }
            _ => {}
        }
    }
    let mut ins = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in text[paren + 1..close].chars() {
        match c {
            '<' | '[' | '(' => depth += 1,
            '>' | ']' | ')' => depth -= 1,
            ',' if depth == 0 => {
                ins.push(std::mem::take(&mut cur));
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    ins.push(cur);
    let (ins, in_dims): (Vec<String>, Vec<Vec<usize>>) = ins
        .iter()
        .filter_map(|p| {
            let p = p.trim();
            if p.is_empty() {
                return None;
            }
            let dims = p
                .find('[')
                .zip(p.find(']'))
                .map(|(a, b)| {
                    p[a + 1..b]
                        .split(',')
                        .filter_map(|d| d.trim().parse().ok())
                        .collect()
                })
                .unwrap_or_default();
            p.rsplit([' ', '>']).next().map(|n| (n.to_string(), dims))
        })
        .unzip();
    let arrow = text[close..].find("->")? + close;
    let open = text[arrow + 2..].find('(')? + arrow + 2;
    let shut = text[open..].find(')')? + open;
    let out = text[open + 1..shut].split(',').next()?.trim().to_string();
    if out.is_empty() {
        return None;
    }
    Some(IoSig {
        ins,
        out,
        in_dims,
        relaid: Vec::new(),
    })
}

/// Shift position-derived field ids inside a layout-descriptor body:
/// every `N=` token whose id lies in `[lo, lo+4]` moves by `delta`.
/// Shape dims (`ar1;0;64`, `s128`, `s8192n`) never match `N=` form.
fn rebase_desc(body: &str, lo: u32, delta: i32) -> String {
    let b = body.as_bytes();
    let mut out = String::with_capacity(body.len());
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let j = i + b[i..].iter().take_while(|c| c.is_ascii_digit()).count();
            if body[j..].starts_with('=') {
                let n: u32 = body[i..j].parse().unwrap_or(0);
                if (lo..lo + 5).contains(&n) {
                    out.push_str(&(n as i32 + delta).to_string());
                    i = j;
                    continue;
                }
            }
            out.push_str(&body[i..j]);
            i = j;
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

/// Substitute real MIL tensor names into a just-emitted artifact,
/// rebuilding the whole name plane the way ANEC does for arbitrary
/// names:
///
/// * `n_sect` = alphabetical rank among io names plus the family's
///   fixed offset for hidden section classes; `n_value` keeps the
///   family's `base + (sec-1)*stride` addressing (fit from the
///   skeleton's io heads),
/// * type-`0x40` section records re-sort by name and carry the new
///   addresses,
/// * cmd-`0x4` name blocks resize (`main\0name[@output]\0name\0` in an
///   `align8` region, capacity `>= 0x10`) and re-sort — inputs
///   alphabetical, output last,
/// * symtab io heads fill the skeleton's io slots as two alphabetical
///   runs — everything else, then the inputs ANEC re-lays out
///   (`IoSig::relaid`); layout descriptors re-sort alphabetically with
///   type bases `t24+5i` and `n_desc = (0x18+5i)&0x1f`,
/// * task-record `4_9_` binding fields re-key to
///   `0x2200 + (sec-1)*0x80` (`0x2080` scalar bindings untouched),
/// * `sizeofcmds`/`symoff`/`stroff`/`reloff` recompute as resized
///   commands shift the whole prefix.
///
/// Byte-identical to ANEC for arbitrary renames — the head grouping
/// rule is verified over every rank permutation of 13 families (conv
/// 1x1/spatial, activation and norm chains, mm/mmsm/mm4d/mmaddsm/smmm).
fn apply_io_names(
    img: &[u8],
    io: &IoSig,
    resolve: &dyn Fn(&str) -> Option<String>,
) -> Result<Vec<u8>, String> {
    let rd32 = |o: usize| u32::from_le_bytes(img[o..o + 4].try_into().unwrap()) as usize;
    let ncmds = rd32(0x10);

    // ---- walk load commands -------------------------------------------
    let mut cmd_offs: Vec<(u32, usize, usize)> = Vec::with_capacity(ncmds); // (cmd, off, sz)
    let (mut symoff, mut nsyms, mut stroff, mut strsize) = (0usize, 0usize, 0usize, 0usize);
    let (mut text_off, mut text_sz) = (0usize, 0usize);
    let (mut reloff, mut nreloc, mut reloff_field) = (0usize, 0usize, 0usize);
    let mut off = 0x20usize;
    for _ in 0..ncmds {
        let cmd = rd32(off) as u32;
        let sz = rd32(off + 4);
        match cmd {
            2 => {
                symoff = rd32(off + 8);
                nsyms = rd32(off + 12);
                stroff = rd32(off + 16);
                strsize = rd32(off + 20);
            }
            0x19 => {
                for s in 0..rd32(off + 0x40) {
                    let e = off + 0x48 + s * 0x50;
                    if &img[e..e + 6] == b"__text" {
                        text_sz = rd32(e + 0x28);
                        text_off = rd32(e + 0x30);
                        if rd32(e + 0x3c) > 0 {
                            reloff = rd32(e + 0x38);
                            nreloc = rd32(e + 0x3c);
                            reloff_field = e + 0x38;
                        }
                    }
                }
            }
            _ => {}
        }
        cmd_offs.push((cmd, off, sz));
        off += sz;
    }
    if nreloc == 0 && rd32(0x21c) > 0 && rd32(0x21c) < 0x20 {
        // single-section skeletons carry the pair in the header
        reloff = rd32(0x218);
        nreloc = rd32(0x21c);
        reloff_field = 0x218;
    }
    if text_off == 0 || symoff == 0 {
        return Err("apply_io_names: malformed image (no __text or LC_SYMTAB)".into());
    }

    // ---- symtab ---------------------------------------------------------
    let strtab = &img[stroff..stroff + strsize];
    let sym_name = |sx: usize| -> String {
        let e = strtab[sx..].iter().position(|&b| b == 0).unwrap();
        String::from_utf8_lossy(&strtab[sx..sx + e]).into_owned()
    };
    // (name, ty, sect, desc, val)
    let mut syms: Vec<(String, u8, u8, u16, u64)> = Vec::with_capacity(nsyms);
    for i in 0..nsyms {
        let e = symoff + i * 16;
        syms.push((
            sym_name(rd32(e)),
            img[e + 4],
            img[e + 5],
            u16::from_le_bytes(img[e + 6..e + 8].try_into().unwrap()),
            u64::from_le_bytes(img[e + 8..e + 16].try_into().unwrap()),
        ));
    }

    // ---- cmd-0x4 blocks: discover the skeleton io names ------------------
    // name region = trailing `main\0name\0name\0` / `main\0name@output\0name\0`
    enum Blk {
        Main,
        In { name: String },
        Out { name: String },
    }
    let mut blocks: Vec<(usize, Blk)> = Vec::new(); // (cmd index, kind)
    for (ci, &(cmd, coff, csz)) in cmd_offs.iter().enumerate() {
        if cmd != 4 {
            continue;
        }
        let b = &img[coff..coff + csz];
        let Some(mpos) = b.windows(5).rposition(|w| w == b"main\0") else {
            continue;
        };
        let rest = &b[mpos + 5..];
        let end = rest.iter().position(|&c| c == 0).unwrap_or(rest.len());
        let first = String::from_utf8_lossy(&rest[..end]).into_owned();
        let kind = if first.is_empty() {
            Blk::Main
        } else if let Some(n) = first.strip_suffix("@output") {
            Blk::Out {
                name: n.to_string(),
            }
        } else {
            Blk::In { name: first }
        };
        blocks.push((ci, kind));
    }
    let skel_ins: Vec<String> = blocks
        .iter()
        .filter_map(|(_, k)| match k {
            Blk::In { name } => Some(name.clone()),
            _ => None,
        })
        .collect();
    let Some((_, Blk::Out { name: skel_out })) =
        blocks.iter().find(|(_, k)| matches!(k, Blk::Out { .. }))
    else {
        return Ok(img.to_vec()); // no io blocks — nothing to substitute
    };
    let skel_out = skel_out.clone();

    // ---- resolve each skeleton name to its real MIL name -----------------
    let new_name = |skel: &str| -> Option<String> { resolve(skel) };
    let mut new_ins: Vec<String> = Vec::with_capacity(skel_ins.len());
    for s in &skel_ins {
        let resolved = new_name(s).or_else(|| {
            if skel_ins.len() == 1 {
                io.ins.first().cloned()
            } else {
                io.ins.iter().find(|i| *i == s).cloned()
            }
        });
        let Some(n) = resolved else {
            return Err(format!(
                "apply_io_names: no MIL input for skeleton input `{s}`"
            ));
        };
        new_ins.push(n);
    }
    let new_out = new_name(&skel_out).unwrap_or_else(|| io.out.clone());
    if new_ins == skel_ins && new_out == skel_out {
        return Ok(img.to_vec()); // already named correctly
    }
    let io_names: Vec<String> = skel_ins.iter().cloned().chain([skel_out.clone()]).collect();
    let new_all: Vec<String> = new_ins.iter().cloned().chain([new_out.clone()]).collect();
    if new_all.iter().any(|n| n.len() > 15 || n.contains('\0')) {
        return Err("io name exceeds the 15-char type-0x40 name field".into());
    }
    if new_all
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len()
        != new_all.len()
    {
        return Err("io names must be distinct".into());
    }

    // ---- new section ranks ----------------------------------------------
    // Skeleton io heads carry the family's sec assignment; io secs must
    // be contiguous (hidden classes take fixed secs outside the range).
    let io_secs: Vec<u8> = syms
        .iter()
        .filter(|(n, ty, s, _, _)| *ty == 0x0f && *s != 0 && io_names.contains(n))
        .map(|(_, _, s, _, _)| *s)
        .collect();
    if io_secs.len() != io_names.len() {
        return Err("apply_io_names: io head symbols not found in symtab".into());
    }
    let sec_lo = *io_secs.iter().min().unwrap();
    let mut sorted_secs = io_secs.clone();
    sorted_secs.sort_unstable();
    if sorted_secs
        .iter()
        .enumerate()
        .any(|(i, &s)| s != sec_lo + i as u8)
    {
        return Err("apply_io_names: non-contiguous io sec range".into());
    }
    let io_off = sec_lo as u32 - 1;
    // n_value = vbase + (sec-1)*vstride, fit over the io heads
    let io_heads: Vec<(u8, u64)> = syms
        .iter()
        .filter(|(n, ty, s, _, _)| *ty == 0x0f && *s != 0 && io_names.contains(n))
        .map(|(_, _, s, _, v)| (*s, *v))
        .collect();
    let (vbase, vstride) = if io_heads.len() >= 2 {
        let mut hh = io_heads.clone();
        hh.sort();
        let (s1, v1) = hh[0];
        let (s2, v2) = hh[1];
        let stride = (v2 - v1) / (s2 - s1) as u64;
        (v1 - stride * (s1 as u64 - 1), stride)
    } else {
        (0x3000_0000, 0x4000)
    };
    // new sec for every io name = alpha rank + family offset
    let mut alpha: Vec<&String> = new_all.iter().collect();
    alpha.sort();
    let new_sec = |name: &str| -> u8 {
        (alpha.iter().position(|n| n.as_str() == name).unwrap() + 1) as u8 + io_off as u8
    };
    // sec -> skeleton io name (for re-keying record bindings)
    let sec_name: Vec<(u8, String)> = syms
        .iter()
        .filter(|(n, ty, s, _, _)| *ty == 0x0f && *s != 0 && io_names.contains(n))
        .map(|(n, _, s, _, _)| (*s, n.clone()))
        .collect();

    // ---- rebuild cmd-0x40 section records (alpha-sorted by new name) -----
    // record: cmd,sz | +8: 0x18 | +0x10: vmaddr | name @0x18 in a field
    // of `align8(len+1)` bytes (min 8) — the record grows with the name.
    let mut c40: Vec<Vec<u8>> = Vec::new();
    for &(cmd, coff, csz) in &cmd_offs {
        if cmd != 0x40 {
            continue;
        }
        let rec = &img[coff..coff + csz];
        let name = String::from_utf8_lossy(&rec[0x18..])
            .split('\0')
            .next()
            .unwrap_or("")
            .to_string();
        // rename + re-address: io names get their new sec's vmaddr;
        // non-io entries keep the skeleton's name/address.
        let (nm, addr) = if let Some(pos) = io_names.iter().position(|n| *n == name) {
            let nn = new_all[pos].clone();
            let a = vbase + vstride * (new_sec(&nn) as u64 - 1);
            (nn, a)
        } else {
            (
                new_name(&name).unwrap_or_else(|| name.clone()),
                u64::from_le_bytes(rec[0x10..0x18].try_into().unwrap()),
            )
        };
        let nb = nm.as_bytes();
        let csz_new = 0x18 + ((nb.len() + 1 + 7) & !7).max(8);
        let mut rec = Vec::with_capacity(csz_new);
        rec.extend_from_slice(&0x40u32.to_le_bytes());
        rec.extend_from_slice(&(csz_new as u32).to_le_bytes());
        rec.extend_from_slice(&0x18u64.to_le_bytes());
        rec.extend_from_slice(&addr.to_le_bytes());
        rec.extend_from_slice(nb);
        rec.resize(csz_new, 0);
        c40.push(rec);
    }
    {
        // re-sort the rebuilt records by their embedded name
        let key = |r: &Vec<u8>| {
            String::from_utf8_lossy(&r[0x18..])
                .split('\0')
                .next()
                .unwrap_or("")
                .to_string()
        };
        c40.sort_by_key(key);
    }

    // ---- permute per-tensor segments by new sec order --------------------
    // A tensor segment = LC_SEGMENT_64 whose single section's vmaddr is an
    // io address. Section content (sectname/prot/flags) is role-bound and
    // re-sorts with the tensor's new sec; addresses stay position-bound.
    let mut tensor_segs: Vec<(usize, u8)> = Vec::new(); // (cmd index, skel sec)
    for (ci, &(cmd, coff, _)) in cmd_offs.iter().enumerate() {
        if cmd != 0x19 || rd32(coff + 0x40) != 1 {
            continue;
        }
        let addr = u64::from_le_bytes(img[coff + 0x68..coff + 0x70].try_into().unwrap());
        if io_heads.iter().any(|(_, v)| *v == addr) {
            let sec = io_heads.iter().find(|(_, v)| *v == addr).unwrap().0;
            tensor_segs.push((ci, sec));
        }
    }
    // skeleton sec -> the io role's skeleton seg cmd index
    let seg_at_skel_sec = |s: u8| -> Option<usize> {
        tensor_segs
            .iter()
            .find(|(_, ss)| *ss == s)
            .map(|(ci, _)| *ci)
    };

    // ---- rebuild cmd-0x4 blocks (ins alpha-sorted, output last) ---------
    // Rebuild each block's name tail; blocks move to the cmd slots in
    // new-alpha order.
    let tail_and_pad = |body: &[u8], tail_start: usize, region: &[u8]| -> Vec<u8> {
        let cap = (region.len().max(0x10) + 7) & !7;
        let mut b = body[..tail_start].to_vec();
        b.extend_from_slice(region);
        b.resize(tail_start + cap, 0);
        let sz = b.len() as u32;
        b[4..8].copy_from_slice(&sz.to_le_bytes());
        // +0x7c interior field = block-relative offset of the second
        // name occurrence: tail_start + len("main\0") + len(first) + 1
        let first_len = region[5..].iter().position(|&c| c == 0).unwrap_or(0);
        let ptr = (tail_start + 5 + first_len + 1) as u32;
        b[0x7c..0x80].copy_from_slice(&ptr.to_le_bytes());
        b
    };
    let mut in_blks: Vec<(String, Vec<u8>)> = Vec::new();
    let mut out_blk: Option<Vec<u8>> = None;
    for &(ci, ref kind) in &blocks {
        let (_, coff, csz) = cmd_offs[ci];
        let b = &img[coff..coff + csz];
        let mpos = b.windows(5).rposition(|w| w == b"main\0").unwrap();
        match kind {
            Blk::Main => {}
            Blk::In { name } => {
                let pos = io_names.iter().position(|n| n == name).unwrap();
                let nn = &new_all[pos];
                let mut region = b"main\0".to_vec();
                region.extend_from_slice(nn.as_bytes());
                region.push(0);
                region.extend_from_slice(nn.as_bytes());
                region.push(0);
                in_blks.push((nn.clone(), tail_and_pad(b, mpos, &region)));
            }
            Blk::Out { .. } => {
                let mut region = b"main\0".to_vec();
                region.extend_from_slice(new_out.as_bytes());
                region.extend_from_slice(b"@output\0");
                region.extend_from_slice(new_out.as_bytes());
                region.push(0);
                out_blk = Some(tail_and_pad(b, mpos, &region));
            }
        }
    }
    in_blks.sort_by(|a, b| a.0.cmp(&b.0));

    // ---- rebuild the nlist ----------------------------------------------
    // io heads: substitute name/sec/val in place (skeleton slot order is
    // the family's canonical order); layout descriptors rebuild at the
    // tail of the nlist in new-alpha order.
    struct Desc {
        skel_name: String,
        body: String, // "t<base>=<rest>"
        skel_base: u32,
    }
    let mut descs: Vec<Desc> = Vec::new();
    let mut entries: Vec<(u32, u8, u8, u16, u64)> = Vec::with_capacity(nsyms); // strx filled later
    let mut strtab_new = vec![0u8];
    let strx = |s: &str, t: &mut Vec<u8>| -> u32 {
        let o = t.len() as u32;
        t.extend_from_slice(s.as_bytes());
        t.push(0);
        o
    };
    // io head slots fill in ANEC's order: two alphabetical runs, inputs
    // ANEC re-lays out (matmul operands, spatial conv inputs) trailing.
    let mut head_order: Vec<String> = new_all.clone();
    head_order.sort_by_key(|n| (io.relaid.contains(n), n.clone()));
    let mut head_order = head_order.into_iter();
    for (name, ty, sect, desc, val) in &syms {
        if io_names.contains(name) {
            let nn = head_order.next().unwrap();
            let sec = new_sec(&nn);
            let val = vbase + vstride * (sec as u64 - 1);
            let sx = strx(&nn, &mut strtab_new);
            entries.push((sx, *ty, sec, *desc, val));
        } else if *ty == 0x20 {
            if let Some((nm, rest)) = name.split_once(':') {
                if let Some(tbase) = rest
                    .strip_prefix('t')
                    .and_then(|r| r.split('=').next().and_then(|d| d.parse::<u32>().ok()))
                {
                    descs.push(Desc {
                        skel_name: nm.to_string(),
                        body: rest[tbase.to_string().len() + 1 + 1..].to_string(),
                        skel_base: tbase,
                    });
                    continue;
                }
            }
            let sx = strx(name, &mut strtab_new);
            entries.push((sx, *ty, *sect, *desc, *val));
        } else {
            // non-io heads (const scalars like sc/fl) and kernel/type
            // syms: rename only if the resolver knows the role. Layout-
            // convert symbols embed the skeleton output name
            // (`TOOO<out>__@convert_input…`).
            let embedded = name.strip_prefix("TOOO").and_then(|rest| {
                io_names.iter().zip(&new_all).find_map(|(sk, nw)| {
                    let tail = rest.strip_prefix(sk.as_str())?;
                    tail.starts_with("__@").then(|| format!("TOOO{nw}{tail}"))
                })
            });
            let nn = embedded
                .or_else(|| new_name(name))
                .unwrap_or_else(|| name.clone());
            let sx = strx(&nn, &mut strtab_new);
            entries.push((sx, *ty, *sect, *desc, *val));
        }
    }
    // layout descriptors in new-alpha order; a desc's tensor set is the
    // union of io names and any non-io desc carriers.
    let mut desc_pos: Vec<String> = descs
        .iter()
        .map(|d| {
            if let Some(pos) = io_names.iter().position(|n| *n == d.skel_name) {
                new_all[pos].clone()
            } else {
                new_name(&d.skel_name).unwrap_or_else(|| d.skel_name.clone())
            }
        })
        .collect();
    desc_pos.sort();
    for nm in &desc_pos {
        // find which skeleton desc this position's tensor carries
        let d = descs
            .iter()
            .find(|d| {
                let nn = if let Some(pos) = io_names.iter().position(|n| *n == d.skel_name) {
                    &new_all[pos]
                } else {
                    nm
                };
                nn == nm
            })
            .unwrap();
        let i_new = desc_pos.iter().position(|n| n == nm).unwrap() as u32;
        let base = 24 + 5 * i_new;
        let body = rebase_desc(&d.body, d.skel_base, base as i32 - d.skel_base as i32);
        let full = format!("{nm}:t{base}={body}");
        let sx = strx(&full, &mut strtab_new);
        entries.push((sx, 0x20, 0, (0x18 + 5 * i_new as u16) & 0x1f, 0));
    }

    // ---- reassemble ------------------------------------------------------
    let mut new_cmds: Vec<u8> = Vec::new();
    let mut c40i = 0usize;
    let mut inbi = 0usize;
    let mut symtab_cmd_pos = 0usize;
    for (ci, &(cmd, coff, csz)) in cmd_offs.iter().enumerate() {
        match cmd {
            0x40 => {
                new_cmds.extend_from_slice(&c40[c40i]);
                c40i += 1;
            }
            0x19 if tensor_segs.iter().any(|(tci, _)| *tci == ci) => {
                // this slot's sec -> the role whose NEW sec lands here ->
                // that role's skeleton segment body, with the slot's
                // position-bound fields restored
                let slot_sec = tensor_segs.iter().find(|(tci, _)| *tci == ci).unwrap().1;
                let role_j = (0..io_names.len()).find(|&j| new_sec(&new_all[j]) == slot_sec);
                let src = role_j
                    .and_then(|j| {
                        let skel_nm = &io_names[j];
                        let sksec = sec_name.iter().find(|(_, n)| n == skel_nm)?.0;
                        seg_at_skel_sec(sksec)
                    })
                    .map(|sci| cmd_offs[sci]);
                match src {
                    Some((_, soff, ssz)) if ssz == csz => {
                        let mut seg = img[soff..soff + ssz].to_vec();
                        // position-bound: seg addrs/sizes + nsects/flags +
                        // the section's segname/addr and offset..nreloc;
                        // the section size (+0x70) is the tensor's own
                        for (a, b) in [(0x18, 0x38), (0x40, 0x48), (0x58, 0x70), (0x78, 0x88)] {
                            seg[a..b].copy_from_slice(&img[coff + a..coff + b]);
                        }
                        new_cmds.extend_from_slice(&seg);
                    }
                    _ => new_cmds.extend_from_slice(&img[coff..coff + csz]),
                }
            }
            4 => match &blocks.iter().find(|(bci, _)| *bci == ci).map(|(_, k)| k) {
                Some(Blk::In { .. }) => {
                    new_cmds.extend_from_slice(&in_blks[inbi].1);
                    inbi += 1;
                }
                Some(Blk::Out { .. }) => new_cmds.extend_from_slice(out_blk.as_ref().unwrap()),
                _ => new_cmds.extend_from_slice(&img[coff..coff + csz]),
            },
            2 => {
                symtab_cmd_pos = new_cmds.len();
                new_cmds.extend_from_slice(&[0u8; 0x18]); // patched below
            }
            _ => new_cmds.extend_from_slice(&img[coff..coff + csz]),
        }
    }
    let cmds_end = 0x20 + new_cmds.len();
    let symoff_new = cmds_end;
    let stroff_new = symoff_new + entries.len() * 16;
    new_cmds[symtab_cmd_pos..symtab_cmd_pos + 0x18].copy_from_slice(
        &[
            2u32.to_le_bytes(),
            0x18u32.to_le_bytes(),
            (symoff_new as u32).to_le_bytes(),
            (entries.len() as u32).to_le_bytes(),
            (stroff_new as u32).to_le_bytes(),
            (strtab_new.len() as u32).to_le_bytes(),
        ]
        .concat(),
    );

    let mut out = img[..0x20].to_vec();
    put_u32(&mut out, 0x14, new_cmds.len() as u32);
    out.extend_from_slice(&new_cmds);
    for (sx, ty, sec, desc, val) in &entries {
        out.extend_from_slice(&sx.to_le_bytes());
        out.push(*ty);
        out.push(*sec);
        out.extend_from_slice(&desc.to_le_bytes());
        out.extend_from_slice(&val.to_le_bytes());
    }
    out.extend_from_slice(&strtab_new);
    if nreloc > 0 {
        let rl = (out.len() + 7) & !7;
        out.resize(rl, 0);
        out.extend_from_slice(&img[reloff..reloff + 8 * nreloc]);
        put_u32(&mut out, reloff_field, rl as u32);
    }
    // pad to __text — emitted layouts carry only zero padding between
    // the variable region and the section data; bail if a family hides
    // real bytes there
    let old_end = (stroff + strsize).max(reloff + 8 * nreloc);
    if img[old_end..text_off].iter().any(|&b| b != 0) {
        return Err("apply_io_names: data between strtab and __text unhandled".into());
    }
    if out.len() > text_off {
        return Err("apply_io_names: rebuilt prefix overflows __text".into());
    }
    out.resize(text_off, 0);

    // ---- patch task-record binding fields in __text ----------------------
    // `4_ 9_ <u16>` entries bind tensors by io rank (sec minus the
    // family's hidden-section offset): 0x2200 + rank*0x80.
    let mut text = img[text_off..].to_vec();
    let end = text_sz.min(text.len());
    for i in 2..end.saturating_sub(1) {
        if text[i - 2] & 0xf0 != 0x40 || text[i - 1] & 0xf0 != 0x90 {
            continue;
        }
        let v = u16::from_le_bytes([text[i], text[i + 1]]);
        if v < 0x2200 || (v - 0x2200) % 0x80 != 0 || v == 0x2200 {
            continue;
        }
        let sec = ((v - 0x2200) / 0x80) as u8 + io_off as u8;
        let Some(skel) = sec_name.iter().find(|(s, _)| *s == sec).map(|(_, n)| n) else {
            continue;
        };
        let pos = io_names.iter().position(|n| n == skel).unwrap();
        let rank = new_sec(&new_all[pos]) - io_off as u8;
        let nv = 0x2200u16 + rank as u16 * 0x80;
        text[i..i + 2].copy_from_slice(&nv.to_le_bytes());
    }
    out.extend_from_slice(&text);
    Ok(out)
}

// ---------------------------------------------------------------------------
// HWX -> MIL + weight.bin Decompiler
// ---------------------------------------------------------------------------

/// Exact inverse of [`weight_layout_ns`]: de-interleaves the 16-lane,
/// 64-byte-padded `__KERN_0` stream back into the row-major `[cout, cin]`
/// fp16 weight matrix, returning `(weights, u16_words_consumed)`.
pub fn weight_unlayout_ns(stream: &[u16], cin: usize, cout: usize) -> (Vec<u16>, usize) {
    let mut w = vec![0u16; cin * cout];
    let mut cur = 0usize;
    for g in 0..16 {
        let g_start = cur;
        let mut base = 0;
        while base < cout {
            let pr = (cout - base).min(512);
            let (pq, pr2) = (pr / 16, pr % 16);
            let rows = pq + usize::from(g < pr2);
            let row0 = base + g * pq + g.min(pr2);
            for ic in 0..cin {
                for m in 0..rows {
                    if cur < stream.len() {
                        w[(row0 + m) * cin + ic] = stream[cur];
                    }
                    cur += 1;
                }
            }
            base += pr;
        }
        let bytes = (cur - g_start) * 2;
        let pad = (64 - bytes % 64) % 64;
        cur += pad / 2;
    }
    (w, cur)
}

/// One lifted operation from a compiled `.hwx` Mach-O binary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecompiledOp {
    Conv1x1 {
        cin: usize,
        cout: usize,
        spatial: usize,
        weight_idx: usize,
    },
    MatmulConst {
        k: usize,
        n: usize,
        weight_idx: usize,
    },
    MatmulAvw {
        r: usize,
        k: usize,
    },
    MatmulOuter {
        t: usize,
        d: usize,
    },
    MatmulAttnV {
        t: usize,
        d: usize,
    },
    Elementwise {
        op: String,
    },
    Unary {
        op: String,
    },
    Softmax {
        axis: i32,
    },
    Transpose,
}

/// Complete lifted representation of a precompiled ANE `.hwx` Mach-O:
/// IO signatures, decoded task operations, de-interleaved `[cout, cin]`
/// fp16 weight tensors, and reconstructed MIL source text.
#[derive(Clone, Debug)]
pub struct DecompiledHwx {
    pub inputs: Vec<(String, [usize; 4])>,
    pub outputs: Vec<(String, [usize; 4])>,
    pub ops: Vec<DecompiledOp>,
    /// De-interleaved row-major `[cout, cin]` fp16 weights per weight-bearing op.
    pub weights: Vec<Vec<u16>>,
    /// Reconstructed `program(1.3)` MIL text.
    pub mil: String,
}

fn get_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

/// Decompile a precompiled ANE `.hwx`/`.macho` binary back into its
/// [`DecompiledHwx`] (IO descriptors, task sequence, de-interleaved
/// `__KERN_0` fp16 weights, and reconstructed MIL program).
pub fn decompile_hwx(img: &[u8], weight_dir: &str) -> Result<DecompiledHwx, String> {
    if img.len() < 0x200 || img[0..4] != 0xbeef_face_u32.to_le_bytes() {
        return Err("not an ANE HWX Mach-O (missing 0xbeefface magic)".into());
    }
    let ncmds = get_u32(img, 0x10) as usize;
    let mut lc = 0x20usize;
    let mut text_range = (0usize, 0usize);
    let mut kern_range = (0usize, 0usize);
    for _ in 0..ncmds {
        if lc + 8 > img.len() {
            break;
        }
        let cmd = get_u32(img, lc);
        let cmdsz = get_u32(img, lc + 4) as usize;
        if cmd == 0x19 && lc + 72 <= img.len() {
            let nsects = get_u32(img, lc + 64) as usize;
            for si in 0..nsects {
                let s = lc + 72 + si * 80;
                if s + 80 > img.len() {
                    break;
                }
                let sname = std::str::from_utf8(&img[s..s + 16])
                    .unwrap_or("")
                    .trim_end_matches('\0');
                let sz = u64::from_le_bytes(img[s + 40..s + 48].try_into().unwrap()) as usize;
                let fo = get_u32(img, s + 48) as usize;
                match sname {
                    "__text" => text_range = (fo, sz),
                    "__kern_0" => kern_range = (fo, sz),
                    _ => {}
                }
            }
        }
        if cmdsz == 0 {
            break;
        }
        lc += cmdsz;
    }

    let (entries, _) = tail_meta(img);
    let mut io_syms = Vec::new();
    for (name, n_type, _n_sect, n_desc, _) in &entries {
        if *n_type == 0x20 && name.contains("ar1;0;") {
            let sym = name.split(':').next().unwrap_or("x").to_string();
            let dv: Vec<usize> = name
                .split("ar1;0;")
                .skip(1)
                .filter_map(|s| s.split(';').next()?.parse().ok())
                .collect();
            if dv.len() >= 4 {
                let dims = if dv[0] > 1 && dv[1] == 1 {
                    [1, dv[0], dv[2], dv[3]]
                } else {
                    [dv[0], dv[1], dv[2], dv[3]]
                };
                io_syms.push((sym, dims, *n_desc));
            }
        }
    }
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for (idx, (sym, dims, desc)) in io_syms.iter().enumerate() {
        if *desc == 0x1d || (idx + 1 == io_syms.len() && outputs.is_empty()) {
            outputs.push((sym.clone(), *dims));
        } else {
            inputs.push((sym.clone(), *dims));
        }
    }
    if inputs.is_empty() {
        inputs.push(("x".into(), [1, 64, 1, 1]));
    }
    if outputs.is_empty() {
        outputs.push(("z".into(), [1, 64, 1, 1]));
    }

    let text = if text_range.0 + text_range.1 <= img.len() && text_range.1 > 0 {
        &img[text_range.0..text_range.0 + text_range.1]
    } else {
        &[]
    };
    let mut ops = Vec::new();
    let mut weights = Vec::new();

    if kern_range.1 > 0 && kern_range.0 + kern_range.1 <= img.len() {
        let kern_u16: Vec<u16> = img[kern_range.0..kern_range.0 + kern_range.1]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let k_tasks = if img.len() > 0xb8c {
            get_u32(img, 0xb88).max(1) as usize
        } else {
            1
        };
        let multi_flag = text.len() > 0x30 && get_u32(text, 0x2c) == 1;
        if k_tasks > 1 && multi_flag {
            let mut kcur = 0usize;
            for ti in 0..k_tasks {
                let want_hdr =
                    0x10240u32 + (ti as u32) * 0x10000 + if ti + 1 == k_tasks { 0x80 } else { 0 };
                let rec_off = (0x30..text.len().saturating_sub(0xf0))
                    .step_by(4)
                    .find(|&o| get_u32(text, o + 8) == want_hdr)
                    .ok_or_else(|| format!("task {ti} record header not found in __text"))?;
                let cin = get_u32(text, rec_off + 0xdc) as usize;
                let cout = get_u32(text, rec_off + 0xec) as usize;
                let (w, used) = weight_unlayout_ns(&kern_u16[kcur..], cin, cout);
                kcur += used;
                let widx = weights.len();
                weights.push(w);
                ops.push(DecompiledOp::Conv1x1 {
                    cin,
                    cout,
                    spatial: 1,
                    weight_idx: widx,
                });
            }
        } else {
            let in0 = inputs[0].1;
            let out0 = outputs[0].1;
            if in0[1] == 1 && in0[3] > 1 {
                let (k, n) = (in0[3], out0[1].max(out0[3]));
                let (w, _) = weight_unlayout_ns(&kern_u16, k, n);
                weights.push(w);
                ops.push(DecompiledOp::MatmulConst {
                    k,
                    n,
                    weight_idx: 0,
                });
            } else {
                let (cin, cout, spatial) = (in0[1], out0[1], in0[3].max(in0[2]).max(1));
                let (w, _) = weight_unlayout_ns(&kern_u16, cin, cout);
                weights.push(w);
                ops.push(DecompiledOp::Conv1x1 {
                    cin,
                    cout,
                    spatial,
                    weight_idx: 0,
                });
            }
        }
    } else if inputs.len() >= 2 {
        let (s0, s1, so) = (inputs[0].1, inputs[1].1, outputs[0].1);
        let avw_op = if text.len() > 0xd4 {
            get_u32(text, 0xd0)
        } else {
            0
        };
        if matches!(avw_op, 0x4a0001 | 0x480001) {
            let r = get_u32(img, 0x198c) as usize;
            let k = get_u32(img, 0x1994) as usize;
            ops.push(DecompiledOp::MatmulAvw { r, k });
        } else if s0[1] > 1 && s0[3] == 1 && s1[3] > 1 && so[1] == s0[1] && so[3] == s1[3] {
            ops.push(DecompiledOp::MatmulOuter { t: so[1], d: so[3] });
        } else if (s0[1] > 1 || s0[2] > 1)
            && (s1[1] > 1 || s1[2] > 1)
            && so[1] == 1
            && so[3] == s1[3]
        {
            ops.push(DecompiledOp::MatmulAttnV {
                t: s1[1].max(s1[2]),
                d: s1[3],
            });
        } else {
            let opc = if s0[1] > 256 && text.len() > 0xd4 {
                get_u32(text, 0xd0)
            } else if text.len() > 0xd0 {
                get_u32(text, 0xcc)
            } else {
                0x0001_0000
            };
            let op = match opc {
                x if x == EwOp::Sub.opcode() => "sub",
                x if x == EwOp::Mul.opcode() => "mul",
                x if x == EwOp::Min.opcode() => "minimum",
                x if x == EwOp::Max.opcode() => "maximum",
                _ => "add",
            };
            ops.push(DecompiledOp::Elementwise { op: op.into() });
        }
    } else {
        let (s0, so) = (inputs[0].1, outputs[0].1);
        if s0[1] > 1 && s0[3] == 1 && so[1] == 1 && so[3] == s0[1] {
            ops.push(DecompiledOp::Transpose);
        } else if text_range.0 == 0x4000 && (so[3] > 1 || text.len() > 0x300) {
            ops.push(DecompiledOp::Softmax {
                axis: if so[1] > 1 { 1 } else { -1 },
            });
        } else {
            ops.push(DecompiledOp::Unary {
                op: "sigmoid".into(),
            });
        }
    }

    // Reconstruct executable MIL text
    let mut mil = String::from("program(1.3)\n{\n    func main<ios18>(");
    for (i, (nm, d)) in inputs.iter().enumerate() {
        if i > 0 {
            mil.push_str(", ");
        }
        mil.push_str(&format!(
            "tensor<fp16, [{}, {}, {}, {}]> {nm}",
            d[0], d[1], d[2], d[3]
        ));
    }
    mil.push_str(") {\n");
    let out_sym = &outputs[0].0;
    let out_d = outputs[0].1;
    for (i, op) in ops.iter().enumerate() {
        let last = i + 1 == ops.len();
        let res = if last {
            out_sym.clone()
        } else {
            format!("y{}", i + 1)
        };
        let xin = if i == 0 {
            inputs[0].0.clone()
        } else {
            format!("y{i}")
        };
        match op {
            DecompiledOp::Conv1x1 {
                cin,
                cout,
                spatial,
                weight_idx,
            } => {
                if i == 0 {
                    mil.push_str("        tensor<string, []> pt = const()[val = string(\"valid\")];\n        tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];\n        tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n        tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];\n        tensor<int32, []> gr = const()[val = int32(1)];\n");
                }
                mil.push_str(&format!(
                    "        tensor<fp16, [{cout}, {cin}, 1, 1]> W{weight_idx} = const()[val = tensor<fp16, [{cout}, {cin}, 1, 1]>(BLOBFILE(path = string(\"{weight_dir}/w{weight_idx}.bin\"), offset = uint64(0)))];\n        tensor<fp16, [1, {cout}, 1, {spatial}]> {res} = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W{weight_idx}, x = {xin})[name = string(\"{res}\")];\n"
                ));
            }
            DecompiledOp::MatmulConst { k, n, weight_idx } => {
                mil.push_str(&format!(
                    "        bool bf = const()[val = bool(false)];\n        bool bt = const()[val = bool(true)];\n        tensor<fp16, [1, {n}, 1, {k}]> W{weight_idx} = const()[val = tensor<fp16, [1, {n}, 1, {k}]>(BLOBFILE(path = string(\"{weight_dir}/w{weight_idx}.bin\"), offset = uint64(0)))];\n        tensor<fp16, [1, 1, 1, {n}]> {res} = matmul(transpose_x = bf, transpose_y = bt, x = {xin}, y = W{weight_idx})[name = string(\"{res}\")];\n"
                ));
            }
            DecompiledOp::MatmulAvw { r, .. } => {
                let yin = &inputs[1].0;
                mil.push_str(&format!(
                    "        tensor<bool, []> bf = const()[val = bool(false)];\n        tensor<bool, []> bt = const()[val = bool(true)];\n        tensor<fp16, [1, {r}, 1, 1]> {res} = matmul(transpose_x = bf, transpose_y = bt, x = {xin}, y = {yin})[name = string(\"{res}\")];\n"
                ));
            }
            DecompiledOp::MatmulOuter { t, d } => {
                let yin = &inputs[1].0;
                mil.push_str(&format!(
                    "        tensor<bool, []> bt = const()[val = bool(true)];\n        tensor<fp16, [1, {t}, 1, {d}]> {res} = matmul(transpose_x = bt, x = {xin}, y = {yin})[name = string(\"{res}\")];\n"
                ));
            }
            DecompiledOp::MatmulAttnV { d, .. } => {
                let yin = &inputs[1].0;
                mil.push_str(&format!(
                    "        tensor<bool, []> bt = const()[val = bool(true)];\n        tensor<fp16, [1, 1, 1, {d}]> {res} = matmul(transpose_x = bt, x = {xin}, y = {yin})[name = string(\"{res}\")];\n"
                ));
            }
            DecompiledOp::Elementwise { op } => {
                let yin = &inputs[1].0;
                mil.push_str(&format!(
                    "        tensor<fp16, [{}, {}, {}, {}]> {res} = {op}(x = {xin}, y = {yin})[name = string(\"{res}\")];\n",
                    out_d[0], out_d[1], out_d[2], out_d[3]
                ));
            }
            DecompiledOp::Unary { op } => {
                mil.push_str(&format!(
                    "        tensor<fp16, [{}, {}, {}, {}]> {res} = {op}(x = {xin})[name = string(\"{res}\")];\n",
                    out_d[0], out_d[1], out_d[2], out_d[3]
                ));
            }
            DecompiledOp::Softmax { axis } => {
                mil.push_str(&format!(
                    "        tensor<int32, []> ax = const()[val = int32({axis})];\n        tensor<fp16, [{}, {}, {}, {}]> {res} = softmax(axis = ax, x = {xin})[name = string(\"{res}\")];\n",
                    out_d[0], out_d[1], out_d[2], out_d[3]
                ));
            }
            DecompiledOp::Transpose => {
                mil.push_str(&format!(
                    "        tensor<fp16, [{}, {}, {}, {}]> {res} = transpose(x = {xin})[name = string(\"{res}\")];\n",
                    out_d[0], out_d[1], out_d[2], out_d[3]
                ));
            }
        }
    }
    mil.push_str(&format!("    }} -> ({out_sym});\n}}\n"));

    Ok(DecompiledHwx {
        inputs,
        outputs,
        ops,
        weights,
        mil,
    })
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
        let _g = SIG_LOCK.lock().unwrap();
        let img = emit_elementwise2(EwOp::Add, 64, "/tmp/x/net.mil", "/tmp/x/net.macho");
        assert_eq!(&img[0..4], &0xbeef_face_u32.to_le_bytes());
        assert_eq!(img.len(), 0x10000);
        assert_eq!(&img[0x8000], &SKELETON[0x8000]);
    }

    #[test]
    fn emits_unary_lut_image() {
        let _g = SIG_LOCK.lock().unwrap();
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

    /// `weight_layout` reproduces ANEC's `__KERN_0` stream for uneven
    /// channel counts (C=40, 80: lanes of unequal row counts) from the
    /// sweep's `0x3c00 + (i*7 % 0x3ff)` weights, and collapses to
    /// `weight_interleave` whenever 16 divides C.
    #[test]
    fn weight_layout_matches_anec_any_c() {
        let raw = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/conv_uneven_kern.json"
        ))
        .unwrap();
        let vec: serde_json::Value = serde_json::from_str(&raw).unwrap();
        for c in [40usize, 80] {
            let w: Vec<u16> = (0..c * c)
                .map(|i| 0x3c00 + ((i * 7) % 0x3ff) as u16)
                .collect();
            let expect: Vec<u16> = vec[c.to_string()]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u16)
                .collect();
            assert_eq!(weight_layout(&w, c).0, expect, "C={c}");
        }
        // (C=16 lanes are 32 bytes and pad to 64 — the padded-band form
        // `weight_interleave` doesn't model; C>=32 lanes are 64-aligned.)
        for c in [32usize, 64, 128] {
            let w: Vec<u16> = (0..(c * c) as u32).map(|i| i as u16).collect();
            assert_eq!(weight_layout(&w, c).0, weight_interleave(&w, c), "C={c}");
        }
    }

    /// Computed kernel names match the ones ANEC embedded in the
    /// captured conv skeletons — no lookup table.
    #[test]
    fn conv_kernel_name_matches_skeletons() {
        for (c, skel) in [
            (32u64, SKELETON_CONV32),
            (64, SKELETON_CONV64S1),
            (64, SKELETON_CONV64S8),
            (128, SKELETON_CONV128),
            (16, SKELETON_CONV16S8),
            (128, SKELETON_CONV128S8),
        ] {
            let name = conv_kernel_name(c, 0x80);
            assert!(
                skel.windows(name.len()).any(|w| w == name.as_bytes()),
                "C={c}: {name} not in skeleton"
            );
        }
    }

    /// Emitting a skeleton family with the corpus compile paths and
    /// perm weights must reproduce the ANEC artifact byte-for-byte.
    #[test]
    fn conv1x1_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        for (shape, case, weights, rel) in [
            (ConvShape::C32W1, "conv32p", vec![0u16; 0], false),
            (ConvShape::C64W1, "conv64p", vec![0u16; 0], false),
            (ConvShape::C64W5, "conv64s5p", vec![0u16; 0], false),
            (ConvShape::C64W8, "conv64s8p", vec![0u16; 0], false),
            (ConvShape::C128W1, "conv128p", vec![0u16; 0], false),
            // fp16 ramp weights (`conv_ramp` corpus case)
            (
                ConvShape::C16W8,
                "conv_ramp",
                (0..256u32)
                    .map(|i| half::f16::from_f32((i as f32 - 4.0).max(0.0)).to_bits())
                    .collect(),
                false,
            ),
            // perm weights; refs embed relative sig paths
            (
                ConvShape::C32W8,
                "conv_perm32",
                (0..1024u32).map(|i| (i as u16).saturating_sub(4)).collect(),
                true,
            ),
            (
                ConvShape::C128W8,
                "conv_perm128",
                (0..16384u32)
                    .map(|i| (i as u16).saturating_sub(4))
                    .collect(),
                true,
            ),
        ] {
            let c = shape.channels();
            let weights = if weights.is_empty() {
                (0..(c * c) as u16).collect::<Vec<u16>>()
            } else {
                weights
            };
            let (mil, out) = if rel {
                (format!("{case}/net.mil"), format!("{case}/out/net.macho"))
            } else {
                (
                    format!("/tmp/ane_corpus/{case}/net.mil"),
                    format!("/tmp/ane_corpus/{case}/out/net.macho"),
                )
            };
            std::env::set_var("HWX_SIG_ANEC", "1");
            let img = emit_conv1x1(shape, &weights, &mil, &out).unwrap();
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

    /// Non-square conv1x1: `emit_conv1x1_ns(64, 32)` reproduces the
    /// captured ANEC artifact byte-for-byte — the `Cin`/`Cout` field
    /// split, the lane stream, the kernel-name hash and all metadata.
    #[test]
    fn conv1x1_ns_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        let weights: Vec<u16> = (0..64 * 32).map(|i| i as u16).collect();
        std::env::set_var("HWX_SIG_ANEC", "1");
        let img = emit_conv1x1_ns(
            64,
            32,
            &weights,
            "/tmp/ane_corpus/convns64x32/net.mil",
            "/tmp/ane_corpus/convns64x32/out/net.macho",
        )
        .unwrap();
        std::env::remove_var("HWX_SIG_ANEC");
        let expect = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/convns64x32_reference.hwx"
        ))
        .unwrap();
        assert_eq!(
            img,
            expect,
            "convns64x32: first diff at {:#x}",
            img.iter()
                .zip(&expect)
                .position(|(a, b)| a != b)
                .map(|i| i as i64)
                .unwrap_or(-1)
        );
    }

    /// Non-square const-weight matmul: `emit_matmul_ns(64, 32)`
    /// reproduces the captured ANEC artifact byte-for-byte — the
    /// M=1 program plan, lane stream, 383-byte kernel-name hash and
    /// all metadata.
    #[test]
    fn matmul_ns_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        let weights: Vec<u16> = (0..64 * 32).map(|i| i as u16).collect();
        std::env::set_var("HWX_SIG_ANEC", "1");
        let img = emit_matmul_ns(
            64,
            32,
            &weights,
            "/tmp/ane_corpus/mmns64x32/net.mil",
            "/tmp/ane_corpus/mmns64x32/out/net.macho",
        )
        .unwrap();
        std::env::remove_var("HWX_SIG_ANEC");
        let expect = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/mmns64x32_reference.hwx"
        ))
        .unwrap();
        assert_eq!(
            img,
            expect,
            "mmns64x32: first diff at {:#x}",
            img.iter()
                .zip(&expect)
                .position(|(a, b)| a != b)
                .map(|i| i as i64)
                .unwrap_or(-1)
        );
    }

    /// Multi-task program emission: `k` chained conv1x1(64→64) ops
    /// in one Mach-O — task-table records, per-task kernel families,
    /// `nreloc = k`, intermediate-buffer descriptor classes. Verified
    /// byte-for-byte against the ANEC chain corpus (weights are the
    /// probe ramp `w[j] = (j + i*7919) % 65536` per task `i`).
    #[test]
    fn conv_chain64_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        std::env::set_var("HWX_SIG_ANEC", "1");
        for k in [2usize, 3, 4] {
            let mut weights = Vec::with_capacity(k * 4096);
            for i in 0..k {
                for j in 0..4096u32 {
                    weights.push(((j + i as u32 * 7919) % 65536) as u16);
                }
            }
            let img = emit_conv_chain64(
                k,
                &weights,
                &format!("y{k}"),
                &format!("/tmp/chain/c64k{k}/net.mil"),
                &format!("/tmp/chain/c64k{k}/out/net.macho"),
            )
            .unwrap();
            let expect = std::fs::read(format!(
                "{}/assets/chain/c64k{k}.hwx",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            assert_eq!(
                img.len(),
                expect.len(),
                "c64k{k}: len {} vs {}",
                img.len(),
                expect.len()
            );
            assert_eq!(
                img,
                expect,
                "c64k{k}: first diff at {:#x}",
                img.iter()
                    .zip(&expect)
                    .position(|(a, b)| a != b)
                    .map(|i| i as i64)
                    .unwrap_or(-1)
            );
        }
        std::env::remove_var("HWX_SIG_ANEC");
    }

    /// Arbitrary io names on the conv family: same-sort, reordered,
    /// and long names must reproduce the ANEC artifacts byte-for-byte
    /// — including the name-sort-sensitive sec ranks, section records,
    /// segment permutations, name-table resizes and record bindings.
    #[test]
    fn io_names_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        let weights: Vec<u16> = (0..64 * 64).map(|i| i as u16).collect();
        std::env::set_var("HWX_SIG_ANEC", "1");
        for (probe, i_name, o_name) in [
            ("same", "bb", "cc"),
            ("alpha", "aa", "zz"),
            ("reorder", "zz", "aa"),
            ("long", "input_xx", "output_zz"),
            ("r2", "yy", "xx"),
            ("r3", "nn", "mm"),
        ] {
            let mil = format!("/tmp/ioname/{probe}/net.mil");
            let out = format!("/tmp/ioname/{probe}/out/net.macho");
            let img = emit_conv1x1(ConvShape::C64W8, &weights, &mil, &out).unwrap();
            // conv64s8 input is spatial ([1,64,8,1]) — relaid
            let io = IoSig {
                ins: vec![i_name.to_string()],
                out: o_name.to_string(),
                in_dims: vec![vec![1, 64, 8, 1]],
                relaid: vec![i_name.to_string()],
            };
            let (inn, outn) = (i_name.to_string(), o_name.to_string());
            let img = apply_io_names(&img, &io, &move |s| match s {
                "x" => Some(inn.clone()),
                "z" => Some(outn.clone()),
                _ => None,
            })
            .unwrap();
            let expect = std::fs::read(format!(
                "{}/assets/ioname_{probe}_reference.hwx",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            let diffs: Vec<usize> = (0..expect.len().min(img.len()))
                .filter(|&i| img[i] != expect[i])
                .collect();
            assert!(
                diffs.is_empty() && img.len() == expect.len(),
                "{probe}: {} diffs, first at {:#x}",
                diffs.len(),
                diffs.first().copied().unwrap_or(0)
            );
        }
        std::env::remove_var("HWX_SIG_ANEC");
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
            let expect =
                std::fs::read(format!("{}/assets/{asset}", env!("CARGO_MANIFEST_DIR"))).unwrap();
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
            let expect =
                std::fs::read(format!("{}/assets/{asset}", env!("CARGO_MANIFEST_DIR"))).unwrap();
            let weights: Vec<Vec<u16>> = (0..layers)
                .map(|i| {
                    let j = (i % n_blobs) as u32;
                    (0..64u32 * 64).map(|e| (e + j * 0x5555) as u16).collect()
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
            (
                FusedAct::Relu,
                1usize,
                1usize,
                "conv64_relu_skeleton.hwx",
                "chain_conv_relu",
            ),
            (
                FusedAct::Relu,
                2,
                1,
                "conv_act2_relu_reference.hwx",
                "chain_conv2_relu2",
            ),
            (
                FusedAct::Relu,
                5,
                1,
                "conv_sig5_relu_reference.hwx",
                "chain5_relu",
            ),
            (
                FusedAct::Relu,
                8,
                8,
                "conv_act8_relu_dw_reference.hwx",
                "chain8_relu_dw",
            ),
            (
                FusedAct::Tanh,
                1,
                1,
                "conv64_tanh_skeleton.hwx",
                "chain_conv_tanh",
            ),
            (
                FusedAct::Tanh,
                2,
                1,
                "conv_act2_tanh_reference.hwx",
                "chain_conv2_tanh2",
            ),
            (
                FusedAct::Tanh,
                5,
                1,
                "conv_sig5_tanh_reference.hwx",
                "chain5_tanh",
            ),
            (
                FusedAct::Tanh,
                8,
                8,
                "conv_act8_tanh_dw_reference.hwx",
                "chain8_tanh_dw",
            ),
            (
                FusedAct::Gelu,
                1,
                1,
                "conv64_gelu_skeleton.hwx",
                "chain_conv_gelu",
            ),
            (
                FusedAct::Gelu,
                2,
                1,
                "conv_act2_gelu_reference.hwx",
                "chain_conv2_gelu2",
            ),
            (
                FusedAct::Gelu,
                5,
                1,
                "conv_sig5_gelu_reference.hwx",
                "chain5_gelu",
            ),
            (
                FusedAct::Gelu,
                8,
                8,
                "conv_act8_gelu_dw_reference.hwx",
                "chain8_gelu_dw",
            ),
            (
                FusedAct::Silu,
                1,
                1,
                "conv64_silu_skeleton.hwx",
                "chain_conv_silu",
            ),
            (
                FusedAct::Silu,
                2,
                1,
                "conv_act2_silu_reference.hwx",
                "chain_conv2_silu2",
            ),
            (
                FusedAct::Silu,
                5,
                1,
                "conv_sig5_silu_reference.hwx",
                "chain5_silu",
            ),
            (
                FusedAct::Silu,
                8,
                8,
                "conv_act8_silu_dw_reference.hwx",
                "chain8_silu_dw",
            ),
            (FusedAct::Softmax, 1, 1, "conv_sm_perm_reference.hwx", "smp"),
        ] {
            let expect =
                std::fs::read(format!("{}/assets/{asset}", env!("CARGO_MANIFEST_DIR"))).unwrap();
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
        for (layers, n_blobs, asset, case, nested) in [
            (2usize, 1usize, "conv_sm2_reference.hwx", "sm2", false),
            (3, 1, "conv_sm3_skeleton.hwx", "sm3", false),
            (4, 1, "conv_sm4_reference.hwx", "sm4", false),
            (5, 1, "conv_sm5_reference.hwx", "sm5", false),
            (2, 2, "conv_smdw2_reference.hwx", "smdw2", false),
            (3, 3, "conv_smdw3_reference.hwx", "smdw3", false),
            (4, 4, "conv_smdw4_reference.hwx", "smdw4", false),
            (5, 5, "conv_smdw5_reference.hwx", "smdw5", true),
            (6, 6, "conv_smdw6_reference.hwx", "smdw6", true),
            (3, 2, "conv_smpdw_reference.hwx", "smpdw", false),
        ] {
            let expect =
                std::fs::read(format!("{}/assets/{asset}", env!("CARGO_MANIFEST_DIR"))).unwrap();
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
            let (mil, out) = if nested {
                (
                    format!("/tmp/dwmine/{case}/net.mil"),
                    format!("/tmp/dwmine/{case}/out/net.macho/net.macho"),
                )
            } else {
                (
                    format!("/tmp/ane_sm/{case}/net.mil"),
                    format!("/tmp/ane_sm/{case}/out/net.macho"),
                )
            };
            std::env::set_var("HWX_SIG_ANEC", "1");
            let img = emit_conv_sm_n_dw(layers, &wrefs, &mil, &out).unwrap();
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
        for (layers, n_blobs, asset, case, ramp) in [
            (1usize, 1usize, "conv_ln1_skeleton.hwx", "ln1", false),
            (2, 1, "conv_ln2_reference.hwx", "ln2", false),
            (3, 1, "conv_ln3_skeleton.hwx", "ln3", false),
            (4, 1, "conv_ln4_skeleton.hwx", "ln4", false),
            (5, 1, "conv_ln5_reference.hwx", "ln5", false),
            (6, 1, "conv_ln6_reference.hwx", "ln6", false),
            (2, 2, "conv_lndw2_reference.hwx", "lndw2", false),
            (3, 3, "conv_lndw3_reference.hwx", "lndw3", false),
            (4, 4, "conv_lndw4_reference.hwx", "lndw4", false),
            (5, 5, "conv_lndw5_reference.hwx", "lndw5", true),
            (6, 6, "conv_lndw6_reference.hwx", "lndw6", true),
        ] {
            let expect =
                std::fs::read(format!("{}/assets/{asset}", env!("CARGO_MANIFEST_DIR"))).unwrap();
            let weights: Vec<Vec<u16>> = (0..layers)
                .map(|i| {
                    let j = i % n_blobs;
                    if n_blobs == 1 {
                        vec![0x3c00u16; 64 * 64]
                    } else if ramp {
                        // the lndw5/6 probes use the `j*0x5555` ramp
                        (0..64u32 * 64)
                            .map(|e| (e + j as u32 * 0x5555) as u16)
                            .collect()
                    } else {
                        (0..64u32 * 64).map(|e| (e + SEEDS[j]) as u16).collect()
                    }
                })
                .collect();
            let wrefs: Vec<&[u16]> = weights.iter().map(|w| w.as_slice()).collect();
            let (mil, out) = if ramp {
                (
                    format!("/tmp/dwmine/{case}/net.mil"),
                    format!("/tmp/dwmine/{case}/out/net.macho/net.macho"),
                )
            } else {
                (
                    format!("/tmp/ane_ln/{case}/net.mil"),
                    format!("/tmp/ane_ln/{case}/out/net.macho"),
                )
            };
            std::env::set_var("HWX_SIG_ANEC", "1");
            let img = emit_conv_ln_n_dw(layers, &wrefs, &mil, &out).unwrap();
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
            let expect =
                std::fs::read(format!("{}/assets/{asset}", env!("CARGO_MANIFEST_DIR"))).unwrap();
            const SEEDS: [u32; 4] = [0, 0x5555, 0xaaaa, 0x3333];
            let weights: Vec<Vec<u16>> = (0..layers)
                .map(|i| {
                    let j = i % n_blobs;
                    if n_blobs != 1 {
                        (0..64u32 * 64).map(|e| (e + SEEDS[j]) as u16).collect()
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

    /// Rank-changing renames on multi-io families, end to end through
    /// `emit_from_mil`: io symtab heads form two alphabetical runs
    /// (relaid matmul operands trail), section sizes follow their
    /// tensor, bindings key on io rank past hidden sections (mm4d),
    /// and the matmul `y` role resolves past a leading softmax (smmm).
    /// The MIL is staged at the path ANEC's signature embeds.
    #[test]
    fn io_name_head_grouping_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        let assets = format!("{}/assets/ioperm", env!("CARGO_MANIFEST_DIR"));
        for case in [
            "p_dcba",
            "p_adbc",
            "mmsm_cab",
            "mm4d_cab",
            "y_smmm_cab",
            "y_mmsm4d_bca",
            "y_mm_act_cba",
        ] {
            let dir = format!("/tmp/nlperm/{case}");
            std::fs::create_dir_all(format!("{dir}/out")).unwrap();
            std::fs::copy(format!("{assets}/{case}.mil"), format!("{dir}/net.mil")).unwrap();
            let expect = std::fs::read(format!("{assets}/{case}_reference.hwx")).unwrap();
            std::env::set_var("HWX_SIG_ANEC", "1");
            let img = emit_from_mil(&format!("{dir}/net.mil"), &format!("{dir}/out/net.macho"));
            std::env::remove_var("HWX_SIG_ANEC");
            let img = img.unwrap();
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

    /// Extended rms-norm family (`layers >= 5 && distinct >= 2`): one
    /// `__KERN_0` lane per layer with no dedup, 13-id stride, wide
    /// aux3 schedule tables with the marker lane at layer `3*(k/3)`.
    /// Covers k=5..12 (interior/terminal x marker forms, ordinals up
    /// to 11) plus partial-dup and true-dup stream patterns.
    #[test]
    fn conv_rms_extended_byte_identical() {
        let _g = SIG_LOCK.lock().unwrap();
        // probe weight generators, by blob index `j`
        let pat = |p: u8, j: u32, e: u32| -> u16 {
            match p {
                b'A' => (j.wrapping_mul(0x5555) + e) as u16,
                b'B' => ((((j + 1) * 0x5555 + e) & 0xffff) & 0x3bff) as u16,
                b'C' => (0x3c00 + ((j * 37 + e * 7) & 0x3ff)) as u16,
                _ => (0x3c00 + (((j + 1) * 37 + e * 7) & 0x3ff)) as u16,
            }
        };
        let seq = |k: usize| (0..k).collect::<Vec<_>>();
        for (asset, case, p, blobs, nested) in [
            ("conv_rmsdw5_reference.hwx", "rmsdw5", b'A', seq(5), true),
            ("conv_rmsdw6_reference.hwx", "rmsdw6", b'A', seq(6), true),
            ("conv_rmsdw7_skeleton.hwx", "rmsdw7", b'B', seq(7), false),
            ("conv_rmsdw8_skeleton.hwx", "rmsdw8", b'B', seq(8), false),
            ("conv_rmsdw9_reference.hwx", "rmsdw9", b'D', seq(9), false),
            (
                "conv_rmsdw10_reference.hwx",
                "rmsdw10",
                b'D',
                seq(10),
                false,
            ),
            (
                "conv_rmsdw11_reference.hwx",
                "rmsdw11",
                b'C',
                seq(11),
                false,
            ),
            (
                "conv_rmsdw12_reference.hwx",
                "rmsdw12",
                b'C',
                seq(12),
                false,
            ),
            (
                "conv_rmspdw5_2_reference.hwx",
                "rmspdw5_2",
                b'C',
                vec![0, 1, 0, 1, 0],
                false,
            ),
            (
                "conv_rmspdw6_4_reference.hwx",
                "rmspdw6_4",
                b'C',
                vec![0, 1, 2, 3, 0, 1],
                false,
            ),
        ] {
            let expect =
                std::fs::read(format!("{}/assets/{asset}", env!("CARGO_MANIFEST_DIR"))).unwrap();
            let weights: Vec<Vec<u16>> = blobs
                .iter()
                .map(|&j| (0..64u32 * 64).map(|e| pat(p, j as u32, e)).collect())
                .collect();
            let wrefs: Vec<&[u16]> = weights.iter().map(|w| w.as_slice()).collect();
            let out = if nested {
                format!("/tmp/dwmine/{case}/out/net.macho/net.macho")
            } else {
                format!("/tmp/dwmine/{case}/out/net.macho")
            };
            std::env::set_var("HWX_SIG_ANEC", "1");
            let img = emit_conv_rms_n_dw(
                blobs.len(),
                &wrefs,
                &format!("/tmp/dwmine/{case}/net.mil"),
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
        // beyond the mined ordinal table: clear error, no bad bytes
        let (w0, w1) = (vec![0u16; 64 * 64], vec![1u16; 64 * 64]);
        let many: Vec<&[u16]> = (0..13)
            .map(|i| {
                if i % 2 == 0 {
                    w0.as_slice()
                } else {
                    w1.as_slice()
                }
            })
            .collect();
        assert!(emit_conv_rms_n_dw(13, &many, "/tmp/x.mil", "/tmp/x.out").is_err());
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
            let expect =
                std::fs::read(format!("{}/assets/{asset}", env!("CARGO_MANIFEST_DIR"))).unwrap();
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
        for (shape, case, ramp) in [
            (MmShape::N16, "mm_ramp16", true),
            (MmShape::N32, "mm32p", false),
            (MmShape::N64, "mm64p", false),
        ] {
            let n = shape.n();
            // `mm_ramp16` weights are an fp16 ramp after a 4-element pad
            let weights: Vec<u16> = if ramp {
                (0..n * n)
                    .map(|i| half::f16::from_f32((i as f32 - 4.0).max(0.0)).to_bits())
                    .collect()
            } else {
                (0..(n * n) as u16).collect()
            };
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
            (MmPart::Act32, "mm_act32", "mm_act32"),
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

    #[test]
    fn constexpr_dequant_folding_and_hw_exec() {
        let dir = std::env::temp_dir().join(format!("deq_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let (n, k) = (64usize, 64usize);
        // 1) Q8_0 blockwise (block=32): int8 data [64, 64], fp16 scale [64, 2]
        let q8: Vec<i8> = (0..n * k)
            .map(|i| (((i * 37 + 11) % 251) as i32 - 125) as i8)
            .collect();
        let q8_bytes: Vec<u8> = q8.iter().map(|&c| c as u8).collect();
        let sc8: Vec<f32> = (0..n * 2)
            .map(|i| 0.002 * ((i * 13 % 47) as f32 + 1.0))
            .collect();
        let sc8_bytes: Vec<u8> = sc8
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();

        // 2) Q4_0 blockwise (block=32): uint4 packed nibbles [64, 64], fp16 scale [64, 2], uint4 offset=8 [64, 2]
        let nibs: Vec<u8> = (0..n * k).map(|i| ((i * 29 + 5) % 16) as u8).collect();
        let q4_packed: Vec<u8> = (0..n * k / 2)
            .map(|j| nibs[2 * j] | (nibs[2 * j + 1] << 4))
            .collect();
        let off4_packed: Vec<u8> = vec![0x88u8; n]; // 64*2 nibbles = 64 bytes

        // 3) Palette4: uint4 indices [64, 64], fp16 LUT [64, 1, 16, 1]
        let lut_f32: Vec<f32> = (0..n * 16).map(|i| (i as f32 * 0.005) - 1.5).collect();
        let lut_bytes: Vec<u8> = lut_f32
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();

        // Pack all blobs into a single deadbeef weight.bin at distinct offsets
        let mut wb = mil_spec::WeightBin::new();
        let o_q8 = wb.put("q8", mil_spec::DType::Int8, &[64, 64], &q8_bytes);
        let o_sc8 = wb.put("sc8", mil_spec::DType::Fp16, &[64, 2], &sc8_bytes);
        let o_q4 = wb.put("q4", mil_spec::DType::Uint4, &[64, 64], &q4_packed);
        let o_off4 = wb.put("off4", mil_spec::DType::Uint4, &[64, 2], &off4_packed);
        let o_lut = wb.put("lut", mil_spec::DType::Fp16, &[64, 1, 16, 1], &lut_bytes);
        std::fs::write(dir.join("weight.bin"), wb.finish()).unwrap();

        let mil = format!(
            r#"program(1.3)
{{
    func main<ios18>(tensor<fp16, [1, 1, 1, 64]> x) {{
        tensor<int8, [64, 64]> W8_q = const()[val = tensor<int8, [64, 64]>(BLOBFILE(path = string("weight.bin"), offset = uint64({o_q8})))];
        tensor<fp16, [64, 2]> W8_s = const()[val = tensor<fp16, [64, 2]>(BLOBFILE(path = string("weight.bin"), offset = uint64({o_sc8})))];
        tensor<fp16, [64, 64]> W8 = constexpr_blockwise_shift_scale(data = W8_q, scale = W8_s)[name = string("W8")];
        tensor<uint4, [64, 64]> W4_q = const()[val = tensor<uint4, [64, 64]>(BLOBFILE(path = string("weight.bin"), offset = uint64({o_q4})))];
        tensor<uint4, [64, 2]> W4_o = const()[val = tensor<uint4, [64, 2]>(BLOBFILE(path = string("weight.bin"), offset = uint64({o_off4})))];
        tensor<fp16, [64, 64]> W4 = constexpr_blockwise_shift_scale(data = W4_q, scale = W8_s, offset = W4_o)[name = string("W4")];
        tensor<fp16, [64, 1, 16, 1]> Wp_l = const()[val = tensor<fp16, [64, 1, 16, 1]>(BLOBFILE(path = string("weight.bin"), offset = uint64({o_lut})))];
        tensor<fp16, [64, 64]> Wp = constexpr_lut_to_dense(indices = W4_q, lut = Wp_l)[name = string("Wp")];
        tensor<bool, []> bt = const()[val = bool(true)];
        tensor<fp16, [1, 1, 1, 64]> z = matmul(x = x, y = W4, transpose_y = bt)[name = string("z")];
    }} -> (z);
}}
"#
        );
        let mil_path = dir.join("m.mil");
        std::fs::write(&mil_path, &mil).unwrap();

        let got8 = load_mil_weight_u16(&mil, &dir, "W8").unwrap();
        let got4 = load_mil_weight_u16(&mil, &dir, "W4").unwrap();
        let gotp = load_mil_weight_u16(&mil, &dir, "Wp").unwrap();
        assert_eq!(got8.len(), n * k);
        assert_eq!(got4.len(), n * k);
        assert_eq!(gotp.len(), n * k);
        for r in 0..n {
            for c in 0..k {
                let idx = r * k + c;
                let s = half::f16::from_f32(sc8[r * 2 + c / 32]).to_f32();
                let w8_ref = half::f16::from_f32(q8[idx] as f32 * s).to_bits();
                let w4_ref = half::f16::from_f32((nibs[idx] as f32 - 8.0) * s).to_bits();
                let wp_ref = half::f16::from_f32(lut_f32[r * 16 + nibs[idx] as usize]).to_bits();
                assert_eq!(got8[idx], w8_ref, "W8[{r},{c}]");
                assert_eq!(got4[idx], w4_ref, "W4[{r},{c}]");
                assert_eq!(gotp[idx], wp_ref, "Wp[{r},{c}]");
            }
        }

        // Emit HWX through emit_from_mil and execute on ANE
        let hwx_path = dir.join("m.macho");
        let img = emit_from_mil(mil_path.to_str().unwrap(), hwx_path.to_str().unwrap()).unwrap();
        std::fs::write(&hwx_path, &img).unwrap();
        let mut prog = crate::espresso::EspressoProgram::load_precompiled(&hwx_path, &[]).unwrap();
        let xv: Vec<f32> = (0..k).map(|i| (i as f32 * 0.03) - 0.9).collect();
        let xd = prog.input_descs[0].clone();
        for i in 0..k {
            let o = xd.offset(0, 0, 0, 0, i as u64);
            prog.input_mut(0)[o..o + 2].copy_from_slice(&half::f16::from_f32(xv[i]).to_le_bytes());
        }
        prog.eval().unwrap();
        let zd = prog.output_descs[0].clone();
        let zb = prog.output(0);
        for j in 0..n {
            let o = zd.offset(0, 0, j as u64, 0, 0);
            let hw = half::f16::from_bits(u16::from_le_bytes([zb[o], zb[o + 1]])).to_f32();
            let want: f32 = (0..k)
                .map(|i| {
                    let x16 = half::f16::from_f32(xv[i]).to_f32();
                    let w16 = half::f16::from_bits(got4[j * k + i]).to_f32();
                    x16 * w16
                })
                .sum();
            assert!(
                (hw - want).abs() < 0.02 + want.abs() * 0.02,
                "z[{j}]: hw={hw} want={want}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Numeric parity harness: for every core family (`conv1x1` square,
    /// non-square, and spatial `W>1`, `matmul_ns`, `avw` broadcast
    /// matvec, `elementwise` small + big-n, `sigmoid`/`tanh`, `softmax`,
    /// and fused `conv -> silu` chains), emit natively via
    /// `emit_from_mil`, compile the same `.mil` via `Anec`, execute both
    /// on ANE hardware, and verify bit-exact parity (`emitted == anec`)
    /// plus agreement with the CPU f32 reference.
    #[test]
    fn numeric_parity_emitted_vs_anec_vs_cpu_ref() {
        let _g = SIG_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!("parity_suite_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let ane = crate::Anec::open().expect("Anec::open");

        let w64: Vec<f32> = (0..64 * 64)
            .map(|i| (((i * 19 + 7) % 97) as f32 - 48.0) * 0.005)
            .collect();
        let w64_128: Vec<f32> = (0..128 * 64)
            .map(|i| (((i * 23 + 3) % 89) as f32 - 44.0) * 0.005)
            .collect();
        let write_f16 = |p: &std::path::Path, shape: &[i64], v: &[f32]| {
            let b: Vec<u8> = v
                .iter()
                .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
                .collect();
            let mut wb = mil_spec::WeightBin::new();
            wb.put("w", mil_spec::DType::Fp16, shape, &b);
            std::fs::write(p, wb.finish()).unwrap();
        };
        write_f16(&root.join("w64.bin"), &[64, 64, 1, 1], &w64);
        write_f16(&root.join("w64_128.bin"), &[128, 64, 1, 1], &w64_128);
        write_f16(&root.join("w64_128_2d.bin"), &[128, 64], &w64_128);

        let cases: Vec<(&str, String)> = vec![
            (
                "conv_sq64",
                format!(
                    r#"program(1.3)
{{
    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {{
        tensor<string, []> pt = const()[val = string("valid")];
        tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];
        tensor<int32, []> gr = const()[val = int32(1)];
        tensor<fp16, [64, 64, 1, 1]> W = const()[val = tensor<fp16, [64, 64, 1, 1]>(BLOBFILE(path = string("{}/w64.bin"), offset = uint64(64)))];
        tensor<fp16, [1, 64, 1, 1]> z = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x)[name = string("z")];
    }} -> (z);
}}"#,
                    root.display()
                ),
            ),
            (
                "conv_ns_64_128",
                format!(
                    r#"program(1.3)
{{
    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {{
        tensor<string, []> pt = const()[val = string("valid")];
        tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];
        tensor<int32, []> gr = const()[val = int32(1)];
        tensor<fp16, [128, 64, 1, 1]> W = const()[val = tensor<fp16, [128, 64, 1, 1]>(BLOBFILE(path = string("{}/w64_128.bin"), offset = uint64(64)))];
        tensor<fp16, [1, 128, 1, 1]> z = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x)[name = string("z")];
    }} -> (z);
}}"#,
                    root.display()
                ),
            ),
            (
                "matmul_ns_64_128",
                format!(
                    r#"program(1.3)
{{
    func main<ios18>(tensor<fp16, [1, 1, 1, 64]> x) {{
        bool bf = const()[val = bool(false)];
        bool bt = const()[val = bool(true)];
        tensor<fp16, [1, 128, 1, 64]> W = const()[val = tensor<fp16, [1, 128, 1, 64]>(BLOBFILE(path = string("{}/w64_128.bin"), offset = uint64(64)))];
        tensor<fp16, [1, 1, 1, 128]> z = matmul(transpose_x = bf, transpose_y = bt, x = x, y = W)[name = string("z")];
    }} -> (z);
}}"#,
                    root.display()
                ),
            ),
            (
                "avw_128_64",
                r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 1, 1, 64]> x, tensor<fp16, [1, 128, 1, 64]> y) {
        tensor<bool, []> bf = const()[val = bool(false)];
        tensor<bool, []> bt = const()[val = bool(true)];
        tensor<fp16, [1, 128, 1, 1]> z = matmul(transpose_x = bf, transpose_y = bt, x = x, y = y)[name = string("z")];
    } -> (z);
}"#
                .to_string(),
            ),
            (
                "ew_add_64",
                r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x, tensor<fp16, [1, 64, 1, 1]> y) {
        tensor<fp16, [1, 64, 1, 1]> z = add(x = x, y = y)[name = string("z")];
    } -> (z);
}"#
                .to_string(),
            ),
            (
                "ew_mul_512",
                r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 512, 1, 1]> x, tensor<fp16, [1, 512, 1, 1]> y) {
        tensor<fp16, [1, 512, 1, 1]> z = mul(x = x, y = y)[name = string("z")];
    } -> (z);
}"#
                .to_string(),
            ),
            (
                "sigmoid_64",
                r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {
        tensor<fp16, [1, 64, 1, 1]> z = sigmoid(x = x)[name = string("z")];
    } -> (z);
}"#
                .to_string(),
            ),
            (
                "conv_spatial_64w8",
                format!(
                    r#"program(1.3)
{{
    func main<ios18>(tensor<fp16, [1, 64, 1, 8]> x) {{
        tensor<string, []> pt = const()[val = string("valid")];
        tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];
        tensor<int32, []> gr = const()[val = int32(1)];
        tensor<fp16, [64, 64, 1, 1]> W = const()[val = tensor<fp16, [64, 64, 1, 1]>(BLOBFILE(path = string("{}/w64.bin"), offset = uint64(64)))];
        tensor<fp16, [1, 64, 1, 8]> z = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x)[name = string("z")];
    }} -> (z);
}}"#,
                    root.display()
                ),
            ),
            (
                "tanh_64",
                r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {
        tensor<fp16, [1, 64, 1, 1]> z = tanh(x = x)[name = string("z")];
    } -> (z);
}"#
                .to_string(),
            ),
            (
                "softmax_64",
                r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 1, 1, 64]> x) {
        tensor<int32, []> ax = const()[val = int32(-1)];
        tensor<fp16, [1, 1, 1, 64]> z = softmax(axis = ax, x = x)[name = string("z")];
    } -> (z);
}"#
                .to_string(),
            ),
        ];

        for (name, mil_text) in &cases {
            let cdir = root.join(name);
            let adir = cdir.join("anec");
            std::fs::create_dir_all(&adir).unwrap();
            let mil_path = cdir.join("net.mil");
            std::fs::write(&mil_path, mil_text).unwrap();

            let emit_path = cdir.join("emitted.macho");
            let img = emit_from_mil(mil_path.to_str().unwrap(), emit_path.to_str().unwrap())
                .unwrap_or_else(|e| panic!("{name} emit_from_mil: {e}"));
            std::fs::write(&emit_path, &img).unwrap();

            let v = ane
                .compile(&mil_path, &adir, "h16g")
                .unwrap_or_else(|e| panic!("{name} anec: {e}"));
            let anec_path = v.artifact.expect("anec artifact");

            let mut p_emit =
                crate::espresso::EspressoProgram::load_precompiled(&emit_path, &[]).unwrap();
            let mut p_anec =
                crate::espresso::EspressoProgram::load_precompiled(&anec_path, &[]).unwrap();
            assert_eq!(p_emit.inputs.len(), p_anec.inputs.len(), "{name} inputs");

            for port in 0..p_emit.inputs.len() {
                let d = p_emit.input_descs[port].clone();
                for n in 0..d.n.max(1) {
                    for c in 0..d.k.max(1) {
                        for h in 0..d.h.max(1) {
                            for w in 0..d.w.max(1) {
                                let idx =
                                    (n * 131 + c * 17 + h * 7 + w * 3 + port as u64 * 29) % 41;
                                let val =
                                    half::f16::from_f32((idx as f32 - 20.0) * 0.04).to_le_bytes();
                                let o = d.offset(n, 0, c, h, w);
                                if o + 2 <= p_emit.input_mut(port).len() {
                                    p_emit.input_mut(port)[o..o + 2].copy_from_slice(&val);
                                    p_anec.input_mut(port)[o..o + 2].copy_from_slice(&val);
                                }
                            }
                        }
                    }
                }
            }
            p_emit.eval().unwrap();
            p_anec.eval().unwrap();

            let od = p_emit.output_descs[0].clone();
            let b_emit = p_emit.output(0);
            let b_anec = p_anec.output(0);
            for n in 0..od.n.max(1) {
                for c in 0..od.k.max(1) {
                    for h in 0..od.h.max(1) {
                        for w in 0..od.w.max(1) {
                            let o = od.offset(n, 0, c, h, w);
                            if o + 2 <= b_emit.len() && o + 2 <= b_anec.len() {
                                let ve = u16::from_le_bytes([b_emit[o], b_emit[o + 1]]);
                                let va = u16::from_le_bytes([b_anec[o], b_anec[o + 1]]);
                                assert_eq!(
                                    ve, va,
                                    "{name} ({n},{c},{h},{w}): emitted={:#06x} anec={:#06x}",
                                    ve, va
                                );
                            }
                        }
                    }
                }
            }
        }

        // Verify C in 713..=720 (the 59-symbol ANEC window) executes on
        // ANE hardware via the standard 43-symbol parametric emitter.
        let c716 = 716usize;
        let w716: Vec<u16> = (0..c716 * c716)
            .map(|i| half::f16::from_f32((((i * 13 + 5) % 61) as f32 - 30.0) * 0.002).to_bits())
            .collect();
        let p716 = root.join("c716.macho");
        let img716 = emit_conv1x1_dyn(
            c716,
            &w716,
            root.join("c716.mil").to_str().unwrap(),
            p716.to_str().unwrap(),
        )
        .unwrap();
        std::fs::write(&p716, &img716).unwrap();
        let mut prog716 = crate::espresso::EspressoProgram::load_precompiled(&p716, &[]).unwrap();
        let d_in = prog716.input_descs[0].clone();
        for c in 0..c716 {
            let o = d_in.offset(0, 0, c as u64, 0, 0);
            let v = half::f16::from_f32(((c % 19) as f32 - 9.0) * 0.05).to_le_bytes();
            prog716.input_mut(0)[o..o + 2].copy_from_slice(&v);
        }
        prog716.eval().unwrap();
        let d_out = prog716.output_descs[0].clone();
        let b_out = prog716.output(0);
        for j in (0..c716).step_by(64) {
            let o = d_out.offset(0, 0, j as u64, 0, 0);
            let hw = half::f16::from_bits(u16::from_le_bytes([b_out[o], b_out[o + 1]])).to_f32();
            let want: f32 = (0..c716)
                .map(|i| {
                    let xv = half::f16::from_f32(((i % 19) as f32 - 9.0) * 0.05).to_f32();
                    let wv = half::f16::from_bits(w716[j * c716 + i]).to_f32();
                    xv * wv
                })
                .sum();
            assert!(
                (hw - want).abs() < 0.03 + want.abs() * 0.03,
                "c716[{j}]: hw={hw} want={want}"
            );
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn roundtrip_compile_and_decompile_hwx() {
        let _g = SIG_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("decompile_rt_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // 1) Non-square conv1x1 (64 -> 128): compile -> decompile -> verify bit-exact weights + re-emit byte-identity
        let w_orig: Vec<u16> = (0..128 * 64)
            .map(|i| half::f16::from_f32((((i * 19 + 7) % 83) as f32 - 41.0) * 0.01).to_bits())
            .collect();
        let hwx1 = emit_conv1x1_ns(64, 128, &w_orig, "/tmp/m.mil", "/tmp/m.macho").unwrap();
        let dec1 = decompile_hwx(&hwx1, dir.to_str().unwrap()).unwrap();
        assert_eq!(
            dec1.ops,
            vec![DecompiledOp::Conv1x1 {
                cin: 64,
                cout: 128,
                spatial: 1,
                weight_idx: 0
            }]
        );
        assert_eq!(dec1.weights.len(), 1);
        assert_eq!(
            dec1.weights[0], w_orig,
            "decompiled conv1x1 weights bit-exact"
        );

        // Write decompiled weights + MIL and re-compile via emit_from_mil: must produce identical HWX bytes
        let wb0: Vec<u8> = dec1.weights[0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(dir.join("w0.bin"), &wb0).unwrap();
        let mil1_path = dir.join("dec1.mil");
        std::fs::write(&mil1_path, &dec1.mil).unwrap();
        let hwx1_re = emit_from_mil(mil1_path.to_str().unwrap(), "/tmp/m.macho").unwrap();
        let hwx1_expected = emit_conv1x1_ns(
            64,
            128,
            &w_orig,
            mil1_path.to_str().unwrap(),
            "/tmp/m.macho",
        )
        .unwrap();
        assert_eq!(
            hwx1_re, hwx1_expected,
            "recompiled decompiled MIL matches HWX byte-for-byte"
        );

        // 2) Multi-task mixed-shape conv_chain (64 -> 96 -> 32 -> 64): decompile recovers all 3 task shapes & weights
        let shapes = [(64usize, 96usize), (96, 32), (32, 64)];
        let mut flat_w = Vec::new();
        let mut per_task_w = Vec::new();
        for (ti, &(ci, co)) in shapes.iter().enumerate() {
            let tw: Vec<u16> = (0..ci * co)
                .map(|i| {
                    half::f16::from_f32((((i * 13 + ti * 29) % 71) as f32 - 35.0) * 0.01).to_bits()
                })
                .collect();
            flat_w.extend_from_slice(&tw);
            per_task_w.push(tw);
        }
        let offs = [
            128u64,
            128 + 64 * 96 * 2 + 64,
            128 + (64 * 96 + 96 * 32) * 2 + 128,
        ];
        let hwx_chain =
            emit_conv_chain(&shapes, &flat_w, &offs, "y3", "/tmp/c.mil", "/tmp/c.macho").unwrap();
        let dec_chain = decompile_hwx(&hwx_chain, dir.to_str().unwrap()).unwrap();
        assert_eq!(dec_chain.ops.len(), 3);
        for (ti, &(ci, co)) in shapes.iter().enumerate() {
            assert_eq!(
                dec_chain.ops[ti],
                DecompiledOp::Conv1x1 {
                    cin: ci,
                    cout: co,
                    spatial: 1,
                    weight_idx: ti
                }
            );
            assert_eq!(
                dec_chain.weights[ti], per_task_w[ti],
                "chain task {ti} weights bit-exact"
            );
        }

        // 3) Elementwise mul & avw broadcast matvec decompilation
        let hwx_mul = emit_elementwise2(EwOp::Mul, 64, "/tmp/e.mil", "/tmp/e.macho");
        let dec_mul = decompile_hwx(&hwx_mul, dir.to_str().unwrap()).unwrap();
        assert_eq!(
            dec_mul.ops,
            vec![DecompiledOp::Elementwise { op: "mul".into() }]
        );

        let hwx_avw = emit_avw(128, 64, "/tmp/a.mil", "/tmp/a.macho");
        let dec_avw = decompile_hwx(&hwx_avw, dir.to_str().unwrap()).unwrap();
        assert_eq!(dec_avw.ops, vec![DecompiledOp::MatmulAvw { r: 128, k: 64 }]);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
