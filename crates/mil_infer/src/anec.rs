//! `anec` — direct MIL → ANE compilation through `ANECompiler.framework`.
//!
//! Bypasses CoreML entirely: builds the private CFDictionary schema
//! recovered by haruspex (`InputNetworks` / `NetworkMILPath` /
//! `NetworkMILName` / `TargetArchitecture`), invokes `ANECValidate` /
//! `ANECCompile` with a real Objective-C completion block, and turns the
//! result dictionary into typed Rust values.
//!
//! The MIL text must be the *ANE-eligible* dialect — float tensor compute
//! only, const index args, per-state read→slice_update→write triples.
//! See `ANE_FRONTEND_SPEC.md` at the repo root for the full grammar.
//!
//! This is a private, undocumented, unstable API. Every call here is
//! contained behind `Result`; malformed dicts historically crashed the
//! process, so the schema below is deliberately exact.

use std::ffi::{c_char, c_void, CString};
use std::path::{Path, PathBuf};

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn dlopen(path: *const c_char, mode: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn CFStringCreateWithCString(
        alloc: *const c_void,
        cstr: *const c_char,
        encoding: u32,
    ) -> *mut c_void;
    fn CFStringGetCStringPtr(s: *const c_void, encoding: u32) -> *const c_char;
    fn CFStringGetLength(s: *const c_void) -> i64;
    fn CFStringGetBytes(
        s: *const c_void,
        range: CFRange,
        encoding: u32,
        fill: u8,
        buf: *mut u8,
        limit: i64,
        used: *mut i64,
    ) -> i64;
    fn CFCopyDescription(obj: *const c_void) -> *mut c_void;
    fn CFRelease(obj: *const c_void);
    fn CFDictionaryCreateMutable(
        alloc: *const c_void,
        capacity: i64,
        key_cbs: *const c_void,
        val_cbs: *const c_void,
    ) -> *mut c_void;
    fn CFDictionarySetValue(dict: *mut c_void, key: *const c_void, val: *const c_void);
    fn CFDictionaryGetValue(dict: *const c_void, key: *const c_void) -> *const c_void;
    fn CFArrayCreate(
        alloc: *const c_void,
        values: *const *const c_void,
        n: i64,
        cbs: *const c_void,
    ) -> *mut c_void;
    fn CFArrayGetCount(arr: *const c_void) -> i64;
    fn CFArrayGetValueAtIndex(arr: *const c_void, idx: i64) -> *const c_void;
    fn CFNumberGetValue(num: *const c_void, ty: i32, out: *mut c_void) -> bool;
    fn CFNumberCreate(alloc: *const c_void, ty: i32, val: *const c_void) -> *mut c_void;
    fn CFGetTypeID(obj: *const c_void) -> u64;
    fn CFStringGetTypeID() -> u64;
    fn CFNumberGetTypeID() -> u64;
    fn CFArrayGetTypeID() -> u64;
    static _NSConcreteGlobalBlock: c_void;
    static kCFTypeDictionaryKeyCallBacks: c_void;
    static kCFTypeDictionaryValueCallBacks: c_void;
    static kCFTypeArrayCallBacks: c_void;
}

const RTLD_NOW: i32 = 2;
const UTF8: u32 = 0x0800_0100;
const CF_NUMBER_SINT64: i32 = 4;
const FRAMEWORK: &str = "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler";

#[repr(C)]
struct CFRange {
    loc: i64,
    len: i64,
}

/// All ANE target architectures accepted by `ANECUnitValidatorCreate`
/// (carved from the framework's target table).
pub const ARCHES: &[&str] = &[
    "h11", "h12", "h13", "h13g", "h14", "h14g", "h14c", "h15", "h15g", "h15c", "h16", "h16g",
    "h16c", "h16s", "h17", "h17a", "h17c", "h17d", "h17g", "h17s", "h18", "h18g", "h19",
];

/// Errors opening or driving the compiler.
#[derive(Debug)]
pub enum AnecError {
    /// `dlopen` failed — the private framework isn't loadable here.
    LoadFailed,
    /// A required export is missing on this OS build.
    MissingSymbol(&'static str),
    /// The mil path has no parent dir / file name (can't build the dict).
    BadPath(PathBuf),
}

impl std::fmt::Display for AnecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnecError::LoadFailed => write!(f, "ANECompiler.framework failed to load"),
            AnecError::MissingSymbol(s) => write!(f, "missing export {s}"),
            AnecError::BadPath(p) => write!(f, "no dir/name split for {}", p.display()),
        }
    }
}

impl std::error::Error for AnecError {}

/// Structured verdict from `ANECValidate` / `ANECCompile`.
pub struct Verdict {
    /// Raw return code of the entry point (0 = clean).
    pub rc: i64,
    /// Callback status (0x0 = clean).
    pub status: u64,
    /// `ErrorList` entries (e.g. `InvalidMILProgram`) or message strings.
    pub errors: Vec<String>,
    /// `ModelMaxDramUsage` when planned.
    pub dram_usage: Option<u64>,
    /// Full result dictionary, verbatim (plist text).
    pub detail: String,
    /// Mach-O path written by a compile call, if any.
    pub artifact: Option<PathBuf>,
}

/// A loaded `ANECompiler.framework` handle.
pub struct Anec {
    validate: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void) -> i64,
    compile: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void) -> i64,
}

// ---- CoreFoundation helpers -------------------------------------------------

fn cfstr(s: &str) -> *mut c_void {
    let c = CString::new(s).unwrap();
    unsafe { CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), UTF8) }
}

fn cf_to_string(s: *const c_void) -> String {
    unsafe {
        let fast = CFStringGetCStringPtr(s, UTF8);
        if !fast.is_null() {
            return std::ffi::CStr::from_ptr(fast)
                .to_string_lossy()
                .into_owned();
        }
        let len = CFStringGetLength(s);
        let mut buf = vec![0u8; (len as usize + 1) * 2];
        let mut used: i64 = 0;
        let got = CFStringGetBytes(
            s,
            CFRange { loc: 0, len },
            UTF8,
            b'?',
            buf.as_mut_ptr(),
            buf.len() as i64,
            &mut used,
        );
        if got > 0 {
            buf.truncate(used as usize);
            String::from_utf8_lossy(&buf).into_owned()
        } else {
            String::new()
        }
    }
}

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
    unsafe { CFDictionarySetValue(d, cfstr(k) as *const c_void, v) }
}

fn dset_str(d: *mut c_void, k: &str, v: &str) {
    dset(d, k, cfstr(v) as *const c_void);
}

fn dset_num(d: *mut c_void, k: &str, v: i64) {
    unsafe {
        let n = CFNumberCreate(
            std::ptr::null(),
            CF_NUMBER_SINT64,
            &v as *const i64 as *const c_void,
        );
        CFDictionarySetValue(d, cfstr(k) as *const c_void, n);
    }
}

fn carray(items: &[*const c_void]) -> *mut c_void {
    unsafe {
        CFArrayCreate(
            std::ptr::null(),
            items.as_ptr(),
            items.len() as i64,
            &raw const kCFTypeArrayCallBacks as *const c_void,
        )
    }
}

fn dict_str(d: *const c_void, key: &str) -> *const c_void {
    unsafe { CFDictionaryGetValue(d, cfstr(key) as *const c_void) }
}

fn array_strings(arr: *const c_void) -> Vec<String> {
    let mut out = Vec::new();
    unsafe {
        let n = CFArrayGetCount(arr);
        for i in 0..n {
            let v = CFArrayGetValueAtIndex(arr, i);
            if !v.is_null() && CFGetTypeID(v) == CFStringGetTypeID() {
                out.push(cf_to_string(v));
            }
        }
    }
    out
}

fn describe(obj: *const c_void) -> String {
    unsafe {
        let d = CFCopyDescription(obj);
        if d.is_null() {
            return String::new();
        }
        let s = cf_to_string(d);
        CFRelease(d);
        s
    }
}

// ---- Completion block --------------------------------------------------------

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

struct Cb {
    status: u64,
    errors: Vec<String>,
    dram: Option<u64>,
    detail: String,
}

static mut CB_SLOT: Option<Cb> = None;

unsafe extern "C" fn on_complete(_b: *mut c_void, status: u64, dict: *mut c_void) {
    let mut cb = Cb {
        status,
        errors: Vec::new(),
        dram: None,
        detail: String::new(),
    };
    if !dict.is_null() {
        let errs = dict_str(dict, "ErrorList");
        if !errs.is_null() && unsafe { CFGetTypeID(errs) } == unsafe { CFArrayGetTypeID() } {
            cb.errors = array_strings(errs);
        }
        let dram = dict_str(dict, "ModelMaxDramUsage");
        if !dram.is_null() && unsafe { CFGetTypeID(dram) } == unsafe { CFNumberGetTypeID() } {
            let mut v: i64 = 0;
            if unsafe {
                CFNumberGetValue(dram, CF_NUMBER_SINT64, &mut v as *mut i64 as *mut c_void)
            } {
                cb.dram = Some(v as u64);
            }
        }
        cb.detail = describe(dict);
    }
    unsafe { *std::ptr::addr_of_mut!(CB_SLOT) = Some(cb) };
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

// ---- Public API ---------------------------------------------------------------

impl Anec {
    /// dlopen the private framework and resolve the entry points.
    pub fn open() -> Result<Self, AnecError> {
        let path = CString::new(FRAMEWORK).unwrap();
        let h = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
        if h.is_null() {
            return Err(AnecError::LoadFailed);
        }
        let get = |name: &'static str| -> Result<_, AnecError> {
            let c = CString::new(name).unwrap();
            let p = unsafe { dlsym(h, c.as_ptr()) };
            if p.is_null() {
                Err(AnecError::MissingSymbol(name))
            } else {
                Ok(unsafe {
                    std::mem::transmute::<
                        _,
                        unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void) -> i64,
                    >(p)
                })
            }
        };
        let entry: &'static str = match std::env::var("ANEC_ENTRY").as_deref() {
            Ok("online") => "ANECCompileOnline",
            Ok("offline") => "ANECCompileOffline",
            Ok("jit") => "ANECCompileJIT",
            Ok("fc") => "ANECCompileWithFunctionCall",
            _ => "ANECCompile",
        };
        Ok(Anec {
            validate: get("ANECValidate")?,
            compile: get(entry)?,
        })
    }

    /// Build the input dictionary for one MIL file.
    ///
    /// `mil` must sit beside any referenced weights (`@model_path`
    /// resolves against its containing dir). `out_dir` gets the Mach-O
    /// on compile; the dict joins path+name by literal concat, so the
    /// dir string must end in `/`.
    fn input_dict(mil: &Path, out_dir: &Path, arch: &str) -> Result<*mut c_void, AnecError> {
        // `parent()` of a bare filename is "" — resolve to "." so
        // NetworkMILPath points at cwd, not "/".
        let dir = match mil.parent() {
            Some(d) if !d.as_os_str().is_empty() => d,
            _ => Path::new("."),
        };
        let base = mil
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| AnecError::BadPath(mil.to_path_buf()))?;
        let dir_s = format!("{}/", dir.display());
        let out_s = format!("{}/", out_dir.display());
        let out_name = base.replace(".mil", ".macho");

        let net = mdict();
        dset_str(net, "NetworkName", base.trim_end_matches(".mil"));
        dset_str(net, "NetworkMILPath", &dir_s);
        dset_str(net, "NetworkMILName", base);
        dset_str(net, "NetworkSourcePath", &dir_s);
        dset_str(net, "NetworkSourceFileName", base);
        dset_str(net, "NetworkJITShapesPath", &dir_s);
        dset_str(net, "NetworkJITShapesName", base);
        dset_str(net, "Version", "1.0.1");
        dset_str(net, "TargetArchitecture", arch);

        let input = mdict();
        dset(
            input,
            "InputNetworks",
            carray(&[net as *const c_void]) as *const c_void,
        );
        dset(input, "ExternConstants", carray(&[]) as *const c_void);
        dset(input, "FileConstants", carray(&[]) as *const c_void);
        dset_str(input, "OutputFilePath", &out_s);
        dset_str(input, "OutputFileName", &out_name);
        dset_str(input, "TargetArchitecture", arch);
        // ANEC_OPT_INPUT: k=v,k2=v2 injected into the input dict — some
        // options (e.g. perf-stats enablement) live here, not params.
        if let Ok(v) = std::env::var("ANEC_OPT_INPUT") {
            for kv in v.split(',') {
                if let Some((k, n)) = kv.split_once('=') {
                    if let Ok(iv) = n.parse::<i64>() {
                        dset_num(input, k, iv);
                    } else {
                        dset_str(input, k, n);
                    }
                }
            }
        }
        Ok(input)
    }

    fn params_dict(arch: &str) -> *mut c_void {
        let p = mdict();
        dset_str(p, "TargetArchitecture", arch);
        if let Ok(mask) = std::env::var("ANEC_DEBUG_MASK") {
            if let Ok(v) = mask.parse::<i64>() {
                dset_num(p, "DebugMask", v);
            }
        }
        if std::env::var_os("ANEC_DEBUG_COMPILE").is_some() {
            dset_num(p, "CompileANEProgramForDebugging", 1);
        }
        if let Ok(v) = std::env::var("ANEC_OPT") {
            for kv in v.split(',') {
                if let Some((k, n)) = kv.split_once('=') {
                    if let Ok(iv) = n.parse::<i64>() {
                        dset_num(p, k, iv);
                    } else {
                        dset_str(p, k, n);
                    }
                }
            }
        }
        p
    }

    fn call(
        f: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void) -> i64,
        mil: &Path,
        out_dir: &Path,
        arch: &str,
    ) -> Result<Verdict, AnecError> {
        let input = Self::input_dict(mil, out_dir, arch)?;
        let params = Self::params_dict(arch);
        unsafe { *std::ptr::addr_of_mut!(CB_SLOT) = None };
        let rc = unsafe { f(input, params, &raw const BLK as *const c_void) };
        let cb = unsafe { (*std::ptr::addr_of_mut!(CB_SLOT)).take() }.unwrap_or(Cb {
            status: u64::MAX,
            errors: vec!["no completion callback".into()],
            dram: None,
            detail: String::new(),
        });
        let base = mil.file_name().and_then(|s| s.to_str()).unwrap_or_default();
        let cand = out_dir.join(base.replace(".mil", ".macho"));
        let artifact = cand.exists().then_some(cand);
        Ok(Verdict {
            rc,
            status: cb.status,
            errors: cb.errors,
            dram_usage: cb.dram,
            detail: cb.detail,
            artifact,
        })
    }

    /// Validate `mil` against `arch` — no artifact written; returns the
    /// compiler's plan (`dram_usage`, live-layout details inside `detail`).
    pub fn validate(&self, mil: &Path, arch: &str) -> Result<Verdict, AnecError> {
        let dir = std::env::temp_dir().join("anec_validate");
        let _ = std::fs::create_dir_all(&dir);
        Self::call(self.validate, mil, &dir, arch)
    }

    /// Compile `mil` for `arch`; writes `<name>.macho` into `out_dir`.
    pub fn compile(&self, mil: &Path, out_dir: &Path, arch: &str) -> Result<Verdict, AnecError> {
        let _ = std::fs::create_dir_all(out_dir);
        Self::call(self.compile, mil, out_dir, arch)
    }
}
