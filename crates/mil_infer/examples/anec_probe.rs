//! anec_probe — talk to ANECompiler.framework directly, no CoreML.
//!
//! dlopens the private framework (dyld resolves it from the shared
//! cache) and walks the exported C surface discovered by haruspex:
//!   ANECGetMPSDialectSupportedVersion   (maj, min, patch) -> int
//!   ANECGetCompilerFileFormat           (libc++ std::string&) -> int
//!   ANECGetDeviceProperty               (CFString name, CFDictionaryRef*)
//!   ANECUnitValidatorCreate             (ctx, CFString arch, validator**)
//!   ANECUnitValidatorCreateWithParams   (ctx, arch, params_dict, out*)
//!   ANEC<Op>LayerDescInitialize         zero-inits a desc buffer
//!   ANECValidate*                       verdicts on a layer desc
//!
//! Signatures recovered by disassembly (haruspex cache disasm):
//!   - ANECUnitValidatorCreate builds {TargetArchitecture: arg1} and
//!     forwards to WithParams(x0, arg1, dict, arg2) — so arg1 is the
//!     target-arch CFString and arg2 the out-param.
//!   - Valid arch names carved from __cstring: h11 h12 h13 h13g h14
//!     h14g h14c h15 h15g h15c h16 h16g h16c h16s h17 h17a h17c h17d
//!     h17g h17s h18 h18g h19
//!   - ANECGetDeviceProperty converts arg0 CFString to std::string,
//!     looks the name up (rc 10 = unknown), stores a CFDictionary at
//!     *arg1 on success.

use std::ffi::{c_char, c_void, CString};

#[link(name = "objc")]
#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn dlopen(path: *const c_char, mode: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn CFStringCreateWithCString(
        alloc: *const c_void,
        cstr: *const c_char,
        encoding: u32,
    ) -> *mut c_void;
    fn CFShow(obj: *const c_void);
    fn CFDictionaryCreateMutable(
        alloc: *const c_void,
        capacity: i64,
        key_cbs: *const c_void,
        val_cbs: *const c_void,
    ) -> *mut c_void;
    fn CFDictionarySetValue(dict: *mut c_void, key: *const c_void, val: *const c_void);
    fn CFArrayCreate(
        alloc: *const c_void,
        values: *const *const c_void,
        n: i64,
        cbs: *const c_void,
    ) -> *mut c_void;

    static _NSConcreteGlobalBlock: c_void;
    static kCFTypeDictionaryKeyCallBacks: c_void;
    static kCFTypeDictionaryValueCallBacks: c_void;
    static kCFTypeArrayCallBacks: c_void;
}

const RTLD_NOW: i32 = 2;
const UTF8: u32 = 0x0800_0100;

fn sym(h: *mut c_void, name: &str) -> *mut c_void {
    let c = CString::new(name).unwrap();
    unsafe { dlsym(h, c.as_ptr()) }
}

fn cfstr(s: &str) -> *mut c_void {
    let c = CString::new(s).unwrap();
    unsafe { CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), UTF8) }
}

/// libc++ std::string, short-string form (arm64): bytes[0..22]=data,
/// byte[23] = size<<1 (bit7 clear).
fn cxx_string(s: &str) -> [u8; 24] {
    assert!(s.len() <= 22);
    let mut b = [0u8; 24];
    b[..s.len()].copy_from_slice(s.as_bytes());
    b[23] = (s.len() as u8) << 1;
    b
}

fn main() {
    let path = CString::new("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler")
        .unwrap();
    let h = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
    if h.is_null() {
        eprintln!("dlopen failed — framework not loadable from this process");
        std::process::exit(1);
    }
    println!("[ok] dlopen ANECompiler");

    // 1. Dialect versions: int fn(u64*,u64*,u64*).
    for name in [
        "ANECGetMPSDialectSupportedVersion",
        "ANECGetMPSSPIDialectSupportedVersion",
        "ANECGetValidateNetworkSupportedVersion",
    ] {
        let p = sym(h, name);
        if p.is_null() {
            println!("[--] {name}: not exported");
            continue;
        }
        let f: unsafe extern "C" fn(*mut u64, *mut u64, *mut u64) -> i32 =
            unsafe { std::mem::transmute(p) };
        let (mut a, mut b, mut c) = (0u64, 0u64, 0u64);
        let rc = unsafe { f(&mut a, &mut b, &mut c) };
        println!("[ver] {name}: rc={rc} out=({a:#x}, {b:#x}, {c:#x})");
    }

    // 2. File-format enum from format names (libc++ string arg).
    let p = sym(h, "ANECGetCompilerFileFormat");
    if !p.is_null() {
        let f: unsafe extern "C" fn(*const u8) -> i32 = unsafe { std::mem::transmute(p) };
        for cand in [
            "mps", "mpssp", "mil", "milbin", "espresso", "program", "text", "binary", "plist",
            "zin",
        ] {
            let s = cxx_string(cand);
            let rc = unsafe { f(s.as_ptr()) };
            println!("[fmt] {cand:10} -> {rc}");
        }
    }

    // 3. Device properties: ANECGetDeviceProperty(CFString, CFDictionaryRef*)
    let p = sym(h, "ANECGetDeviceProperty");
    if !p.is_null() {
        let f: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> u64 =
            unsafe { std::mem::transmute(p) };
        for key in [
            "TargetArchitecture",
            "Architecture",
            "DeviceName",
            "DeviceClass",
            "BoardId",
            "ProductType",
            "ane",
            "ANE",
        ] {
            let mut out: *mut c_void = std::ptr::null_mut();
            let rc = unsafe { f(cfstr(key), &mut out) };
            print!("[dev] {key:20} rc={rc}");
            if !out.is_null() {
                print!(" dict=");
                unsafe { CFShow(out) };
            } else {
                println!();
            }
        }
    }

    // 4. Validator via ANECUnitValidatorCreate(ctx, arch, out*).
    //    Create wraps WithParams with {TargetArchitecture: arch}
    //    internally — arg1 is the arch string, arg2 the out-param.
    let p = sym(h, "ANECUnitValidatorCreate");
    let mut validator: u64 = 0;
    let mut good_arch = String::new();
    if !p.is_null() {
        let f: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut u64) -> u64 =
            unsafe { std::mem::transmute(p) };
        for arch in [
            "h11", "h12", "h13", "h13g", "h14", "h14g", "h14c", "h15", "h15g", "h15c", "h16",
            "h16g", "h16c", "h16s", "h17", "h17a", "h17c", "h17d", "h17g", "h17s", "h18", "h18g",
            "h19",
        ] {
            let mut out: u64 = 0;
            let rc = unsafe { f(std::ptr::null_mut(), cfstr(arch), &mut out) };
            println!("[val] Create(0,\"{arch}\",&out): rc={rc} v={out:#x}");
            if out != 0 && validator == 0 {
                validator = out;
                good_arch = arch.into();
            }
        }
    }

    // 5. With a live validator, poke a descriptor validate call.
    let do_desc = std::env::var("ANEC_DESC").is_ok();
    if validator != 0 && do_desc {
        println!("[val] live validator for arch {good_arch}");
        let ip = sym(h, "ANECRingBufferWriterLayerDescInitialize");
        let mut desc = vec![0u8; 0x200];
        if !ip.is_null() {
            let init: unsafe extern "C" fn(*mut u8) = unsafe { std::mem::transmute(ip) };
            unsafe { init(desc.as_mut_ptr()) };
            println!("[desc] RBW desc zeroed 0xd8+ bytes");
        }
        for name in ["ANECValidateRingBufferWriterLayer", "ANECValidate"] {
            let vp = sym(h, name);
            if vp.is_null() {
                continue;
            }
            let vf: unsafe extern "C" fn(u64, *const u8, *mut u64, *mut u64, *mut u64) -> i32 =
                unsafe { std::mem::transmute(vp) };
            let (mut e1, mut e2, mut e3) = (0u64, 0u64, 0u64);
            let rc = unsafe { vf(validator, desc.as_ptr(), &mut e1, &mut e2, &mut e3) };
            println!("[val] {name}(v,desc,...): rc={rc} e=({e1:#x},{e2:#x},{e3:#x})");
        }
        let dp = sym(h, "ANECUnitValidatorDelete");
        if !dp.is_null() {
            let d: unsafe extern "C" fn(u64) = unsafe { std::mem::transmute(dp) };
            unsafe { d(validator) };
            println!("[val] validator deleted");
        }
    } else {
        println!("[val] no validator — WithParams params dict still unknown");
    }

    // 6. The dict path — ANECValidate(input_dict, params_dict, out).
    //    Schema recovered from the input-dict parser at 0x20aca1eac:
    //      input:  { InputNetworks: [ { NetworkName, NetworkMILPath |
    //                NetworkPlistPath | NetworkSourcePath } ] }
    //      params: { TargetArchitecture: "h16g", <300 option keys> }
    fn mdict() -> *mut c_void {
        unsafe {
            CFDictionaryCreateMutable(
                std::ptr::null(),
                0,
                &raw const kCFTypeDictionaryKeyCallBacks as *const c_void,
                &raw const kCFTypeDictionaryValueCallBacks as *const c_void,
            )
        }
    }
    fn dset(d: *mut c_void, k: &str, v: *const c_void) {
        unsafe { CFDictionarySetValue(d, cfstr(k) as *const c_void, v as *const c_void) }
    }
    let mil = std::env::args()
        .nth(1)
        .unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/assets/chain/c64k1.mil").into());
    let (dir0, base) = mil.rsplit_once('/').unwrap();
    let dir = format!("{dir0}/");
    let net = mdict();
    dset(net, "NetworkName", cfstr("probe_net"));
    dset(net, "NetworkMILPath", cfstr(&dir));
    dset(net, "NetworkMILName", cfstr(base));
    dset(net, "NetworkSourcePath", cfstr(&dir));
    dset(net, "NetworkSourceFileName", cfstr(base));
    dset(net, "OutputFilePath", cfstr("/tmp/anec_out/"));
    dset(net, "OutputFileName", cfstr("probe_net.macho"));
    dset(net, "NetworkJITShapesPath", cfstr(&dir));
    dset(net, "Path", cfstr("/tmp/anec_out"));
    dset(net, "NetworkJITShapesName", cfstr(base));
    dset(net, "Version", cfstr("1.0.1"));
    dset(net, "TargetArchitecture", cfstr("h16g"));
    let empty_arr = unsafe {
        CFArrayCreate(
            std::ptr::null(),
            std::ptr::null(),
            0,
            &raw const kCFTypeArrayCallBacks as *const c_void,
        )
    };
    let net_mil = mdict();
    dset(net_mil, "NetworkName", cfstr("probe_mil"));
    dset(net_mil, "NetworkMILPath", cfstr(&dir));
    dset(net_mil, "NetworkMILName", cfstr(base));
    let nets_mil = unsafe {
        CFArrayCreate(
            std::ptr::null(),
            [net_mil as *const c_void].as_ptr(),
            1,
            &raw const kCFTypeArrayCallBacks as *const c_void,
        )
    };
    let input_mil = mdict();
    dset(input_mil, "InputNetworks", nets_mil);
    dset(input_mil, "ExternConstants", empty_arr);
    dset(input_mil, "FileConstants", empty_arr);
    dset(input_mil, "OutputFilePath", cfstr("/tmp/anec_out/"));
    dset(input_mil, "OutputFileName", cfstr("probe_mil.macho"));
    let outdir = "/tmp/anec_out";
    let _ = std::fs::create_dir_all(outdir);
    let nets = unsafe {
        CFArrayCreate(
            std::ptr::null(),
            [net as *const c_void].as_ptr(),
            1,
            &raw const kCFTypeArrayCallBacks as *const c_void,
        )
    };
    let inner = mdict();
    dset(inner, "TargetArchitecture", cfstr("h16g"));
    dset(inner, "PlannedBufferReuse", cfstr("reuse"));
    let input = mdict();
    dset(input, "InputNetworks", nets);
    let input_nets_key = mdict();
    dset(input_nets_key, "Networks", nets);
    dset(input, "Params", inner);
    dset(input, "TargetArchitecture", cfstr("h16g"));
    dset(input, "OptionsFilePath", cfstr("/tmp/anec_opts.plist"));
    dset(input, "OutputFilePath", cfstr("/tmp/anec_out/"));
    dset(input, "OutputFileName", cfstr("probe_net.macho"));
    dset(input, "ExternConstants", empty_arr);
    dset(input, "FileConstants", empty_arr);
    dset(input, "OutputFilePath", cfstr("/tmp/anec_out/"));
    dset(input, "OutputFileName", cfstr("probe_net.macho"));
    let params = mdict();
    dset(params, "TargetArchitecture", cfstr("h16g"));
    let params_opts = mdict();
    dset(
        params_opts,
        "OptionsFilePath",
        cfstr("/tmp/anec_opts.plist"),
    );
    let params_both = mdict();
    dset(params_both, "TargetArchitecture", cfstr("h16g"));
    let arch_arr = unsafe {
        CFArrayCreate(
            std::ptr::null(),
            [cfstr("h16g") as *const c_void].as_ptr(),
            1,
            &raw const kCFTypeArrayCallBacks as *const c_void,
        )
    };
    let params_arr = mdict();
    dset(params_arr, "TargetArchitecture", arch_arr);
    dset(
        params_both,
        "OptionsFilePath",
        cfstr("/tmp/anec_opts.plist"),
    );

    println!("[dict] input:");
    unsafe { CFShow(input) };
    println!("[dict] params:");
    unsafe { CFShow(params) };

    // Each variant runs in a forked child: a crash isolates that case.
    extern "C" {
        fn fork() -> i32;
        fn _exit(s: i32) -> !;
        fn waitpid(p: i32, s: *mut i32, o: i32) -> i32;
    }
    // 6b. Canonical construction: ANECCreateCompilerInputDictionary
    //     takes a libc++ std::string& (path) -> the input dict itself.
    //     libc++ arm64 long-form: {ptr, size, cap|MSB} when byte23<0.
    let mkpath = |s: &str| -> ([u8; 24], CString) {
        let c = CString::new(s).unwrap();
        let mut b = [0u8; 24];
        b[..8].copy_from_slice(&(c.as_ptr() as usize).to_le_bytes());
        b[8..16].copy_from_slice(&s.len().to_le_bytes());
        let cap = s.len() | (1usize << 63);
        b[16..24].copy_from_slice(&cap.to_le_bytes());
        (b, c)
    };
    let cd = sym(h, "ANECCreateCompilerInputDictionary");
    let mut built_input: *mut c_void = std::ptr::null_mut();
    if !cd.is_null() {
        let (pb, _keep) = mkpath(&mil);
        let f: unsafe extern "C" fn(*const u8) -> *mut c_void = unsafe { std::mem::transmute(cd) };
        built_input = unsafe { f(pb.as_ptr()) };
        println!("[dict] CreateCompilerInputDictionary({mil}) -> {built_input:?}");
        if !built_input.is_null() {
            unsafe { CFShow(built_input) };
        }
    }
    let vp = sym(h, "ANECValidate");
    let cp = sym(h, "ANECCompile");
    let empty_i = mdict();
    let empty_p = mdict();
    let i_nets_only = mdict();
    dset(i_nets_only, "InputNetworks", nets);
    let cases: Vec<(&str, *mut c_void, *mut c_void)> = vec![
        ("empty,empty", empty_i, empty_p),
        ("input,empty", input, empty_p),
        ("input,params", input, params),
        ("nets_only,params", i_nets_only, params),
        ("input,archsections", input, params),
        ("mildict,params", input_mil, params),
        ("netskey,params", input_nets_key, params),
        ("input,nullparams", input, std::ptr::null_mut()),
        ("input,optsonly", input, params_opts),
        ("input,both", input, params_both),
        ("input,arrarch", input, params_arr),
        ("built,empty", built_input, empty_p),
        ("built,params", built_input, params),
    ];
    // ANECValidate/Compile take a completion BLOCK as arg2:
    //   block(ANECStatus status, CFDictionaryRef result)
    #[repr(C)]
    struct Block {
        isa: *const c_void,
        flags: i32,
        rsvd: i32,
        invoke: *const c_void,
        desc: *const c_void,
    }
    #[repr(C)]
    struct BlockDesc {
        rsvd: usize,
        size: usize,
    }
    unsafe extern "C" fn on_complete(_b: *mut c_void, status: u64, dict: *mut c_void) {
        println!("  [cb] status={status:#x} dict={dict:?}");
        if !dict.is_null() {
            unsafe { CFShow(dict) };
        }
    }
    static DESC: BlockDesc = BlockDesc { rsvd: 0, size: 32 };
    unsafe impl Sync for Block {}
    static BLK: Block = Block {
        isa: &raw const _NSConcreteGlobalBlock,
        flags: 1 << 30,
        rsvd: 0,
        invoke: on_complete as *const c_void,
        desc: &raw const DESC as *const c_void,
    };
    for (tag, i, p) in cases {
        for (fname, fp) in [
            ("ANECCompileOnline", sym(h, "ANECCompileOnline")),
            ("ANECCompileOffline", sym(h, "ANECCompileOffline")),
            ("ANECCompileJIT", sym(h, "ANECCompileJIT")),
            ("ANECValidate", vp),
            ("ANECCompile", cp),
        ] {
            if fp.is_null() {
                continue;
            }
            let pid = unsafe { fork() };
            if pid == 0 {
                let f: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void) -> i64 =
                    unsafe { std::mem::transmute(fp) };
                let rc = unsafe { f(i, p, &raw const BLK as *const c_void) };
                println!("[dict] {fname}({tag}): rc={rc}");
                unsafe { _exit(0) };
            }
            let mut st: i32 = 0;
            unsafe { waitpid(pid, &mut st, 0) };
            if st & 0x7f != 0 {
                println!("[dict] {fname}({tag}): CRASH sig={}", st & 0x7f);
            }
            for pat in [
                "/tmp/anec_out/probe_net.macho",
                "/tmp/anec_out/probe_mil.macho",
                "/tmp/anec_outprobe_net.macho",
                "/tmp/anec_outprobe_mil.macho",
            ] {
                if let Ok(md) = std::fs::metadata(pat) {
                    println!("    -> artifact {pat} {} bytes", md.len());
                }
            }
        }
    }
}
