//! `mil_compile` — turn a `.mlpackage` into a `.mlmodelc`.
//!
//! Two backends, same API:
//!
//! - [`Backend::CoreMlFramework`] — in-process `MLModel compileModelAtURL:error:`
//!   via the Objective-C runtime. No `xcrun`, no subprocess, no toolchain
//!   version matrix. This is the "native Rust" path: the same code
//!   `coremlc` calls, called directly. macOS-only.
//! - [`Backend::Subprocess`] — shells out to `xcrun coremlc`. Works anywhere
//!   Xcode CLT is installed and on CI where the framework may not be
//!   linkable.
//!
//! [`compile`] picks the framework when available and falls back to the
//! subprocess. Both return the located `.mlmodelc` path — the compiler
//! chooses the output directory name and we discover it, not the other
//! way around, so a rename by `coremlc` never silently loses the artifact.
//!
//! # Errors
//!
//! Every failure carries the stage that produced it (`"ffi"`, `"xcrun"`,
//! `"discover"`), the raw stderr/localizedDescription when there is one,
//! and a human-readable message. `milc` prints these verbatim — a
//! `coremlc` rejection already contains the line-and-op you need.

use std::fmt;
use std::path::{Path, PathBuf};

/// Which backend produced the artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// `MLModel compileModelAtURL:` in-process.
    CoreMlFramework,
    /// `xcrun coremlc` subprocess.
    Subprocess,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Backend::CoreMlFramework => "coreml.framework",
            Backend::Subprocess => "coremlc",
        })
    }
}

/// A successful compile.
#[derive(Clone, Debug)]
pub struct CompiledModel {
    /// The produced `.mlmodelc` directory.
    pub path: PathBuf,
    /// Backend that produced it.
    pub backend: Backend,
    /// `coremlc` stdout (empty on the FFI path).
    pub stdout: String,
    /// `coremlc` stderr (empty on the FFI path).
    pub stderr: String,
}

/// A compile failure.
#[derive(Clone, Debug)]
pub struct CompileError {
    /// `"ffi"` | `"xcrun"` | `"discover"` | `"io"`.
    pub stage: &'static str,
    /// Human-readable error.
    pub message: String,
    /// Raw compiler stderr or NSError description, when available.
    pub detail: Option<String>,
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.detail {
            Some(d) if !d.is_empty() => {
                write!(
                    f,
                    "compile failed at {}: {}\n{}",
                    self.stage, self.message, d
                )
            }
            _ => write!(f, "compile failed at {}: {}", self.stage, self.message),
        }
    }
}

impl std::error::Error for CompileError {}

impl From<std::io::Error> for CompileError {
    fn from(e: std::io::Error) -> Self {
        CompileError {
            stage: "io",
            message: e.to_string(),
            detail: None,
        }
    }
}

type Result<T> = std::result::Result<T, CompileError>;

/// Compile a `.mlpackage` at `pkg`, outputting to `out_dir`.
///
/// Tries the in-process framework first (macOS); on any FFI failure —
/// missing class, NSError, non-Apple build — falls back to `xcrun coremlc`
/// and reports which backend won in the result.
pub fn compile(pkg: &Path, out_dir: &Path) -> Result<CompiledModel> {
    match compile_with(Backend::CoreMlFramework, pkg, out_dir) {
        Ok(m) => Ok(m),
        Err(ffi_err) => {
            // Fall back to the subprocess and report if that also fails.
            match compile_with(Backend::Subprocess, pkg, out_dir) {
                Ok(m) => Ok(m),
                Err(sub_err) => Err(CompileError {
                    stage: "xcrun",
                    message: format!(
                        "framework backend failed ({}); coremlc also failed: {}",
                        ffi_err.message, sub_err.message
                    ),
                    detail: sub_err.detail,
                }),
            }
        }
    }
}

/// Compile with an explicit backend.
pub fn compile_with(backend: Backend, pkg: &Path, out_dir: &Path) -> Result<CompiledModel> {
    match backend {
        Backend::CoreMlFramework => ffi::compile_ffi(pkg, out_dir),
        Backend::Subprocess => compile_subprocess(pkg, out_dir),
    }
}

/// Find `coremlc` via `xcrun -f`. Returns `None` off-macOS or without CLT.
pub fn coremlc_path() -> Option<PathBuf> {
    let out = std::process::Command::new("xcrun")
        .args(["-f", "coremlc"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(PathBuf::from(s))
    }
}

// ---------- subprocess backend ----------

fn compile_subprocess(pkg: &Path, out_dir: &Path) -> Result<CompiledModel> {
    let coremlc = coremlc_path().ok_or(CompileError {
        stage: "xcrun",
        message: "xcrun -f coremlc found nothing — install Xcode CLT".into(),
        detail: None,
    })?;
    std::fs::create_dir_all(out_dir)?;
    let out = std::process::Command::new(&coremlc)
        .arg("compile")
        .arg(pkg)
        .arg(out_dir)
        .output()?;
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    if !out.status.success() {
        return Err(CompileError {
            stage: "xcrun",
            message: format!("coremlc exited {}", out.status.code().unwrap_or(-1)),
            detail: Some(if stderr.is_empty() { stdout } else { stderr }),
        });
    }
    let path = discover_mlmodelc(out_dir, pkg)?;
    Ok(CompiledModel {
        path,
        backend: Backend::Subprocess,
        stdout,
        stderr,
    })
}

// ---------- .mlmodelc discovery ----------

/// Locate the `.mlmodelc` a compile produced. `coremlc` names it after
/// the package stem, but doesn't promise that — accept any directory
/// ending in `.mlmodelc` under `out_dir`, preferring the stem match.
fn discover_mlmodelc(out_dir: &Path, pkg: &Path) -> Result<PathBuf> {
    let stem = pkg
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let mut candidates: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(out_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("mlmodelc") && path.is_dir() {
            candidates.push(path);
        }
    }
    if let Some(hit) = candidates
        .iter()
        .find(|p| p.file_stem().and_then(|s| s.to_str()) == Some(stem.as_str()))
    {
        return Ok(hit.clone());
    }
    if candidates.len() == 1 {
        return Ok(candidates.remove(0));
    }
    Err(CompileError {
        stage: "discover",
        message: format!(
            "expected one .mlmodelc under {}, found {}",
            out_dir.display(),
            candidates.len()
        ),
        detail: None,
    })
}

// ---------- in-process CoreML.framework backend ----------

/// FFI backend. macOS-only — compiled out elsewhere so the crate still
/// builds on Linux CI.
#[cfg(target_os = "macos")]
mod ffi {
    use super::*;
    use std::ffi::CString;
    use std::os::raw::c_char;
    use std::os::raw::c_void;

    type Id = *mut c_void;
    type Sel = *mut c_void;

    // The Objective-C runtime. `#[link(name = "objc")]` pulls in libobjc;
    // `dlopen` resolves from libSystem/libdl with no extra link. We declare
    // `objc_msgSend` as a plain extern fn and transmute per-call to the
    // needed signature — the standard pattern before the objc2 crates;
    // variadic dispatch makes the transmute unavoidable.
    #[link(name = "objc")]
    extern "C" {
        fn objc_getClass(name: *const c_char) -> Id;
        fn sel_registerName(name: *const c_char) -> Sel;
        fn objc_msgSend();
        fn objc_autoreleasePoolPush() -> *mut c_void;
        fn objc_autoreleasePoolPop(pool: *mut c_void);
        fn dlopen(path: *const c_char, mode: i32) -> *mut c_void;
    }

    const RTLD_NOW: i32 = 2;

    fn cls(name: &str) -> Result<Id> {
        let c = CString::new(name).unwrap();
        let v = unsafe { objc_getClass(c.as_ptr()) };
        if v.is_null() {
            Err(CompileError {
                stage: "ffi",
                message: format!(
                    "objc class {} not found — is CoreML.framework loaded?",
                    name
                ),
                detail: None,
            })
        } else {
            Ok(v)
        }
    }

    fn sel(name: &str) -> Sel {
        let c = CString::new(name).unwrap();
        unsafe { sel_registerName(c.as_ptr()) }
    }

    /// Cast objc_msgSend to a concrete signature and call it.
    /// Safety: signatures must match the Objective-C method's ABI exactly.
    /// All calls here are on 64-bit Apple Silicon/Intel where every arg is
    /// a pointer; each helper fixes one arity so the transmute is correct.
    unsafe fn msg0(recv: Id, s: Sel) -> Id {
        let f: unsafe extern "C" fn(Id, Sel) -> Id =
            std::mem::transmute(objc_msgSend as *const () as usize);
        f(recv, s)
    }
    unsafe fn msg1(recv: Id, s: Sel, a: Id) -> Id {
        let f: unsafe extern "C" fn(Id, Sel, Id) -> Id =
            std::mem::transmute(objc_msgSend as *const () as usize);
        f(recv, s, a)
    }
    unsafe fn msg2(recv: Id, s: Sel, a: Id, b: Id) -> Id {
        let f: unsafe extern "C" fn(Id, Sel, Id, Id) -> Id =
            std::mem::transmute(objc_msgSend as *const () as usize);
        f(recv, s, a, b)
    }

    /// `+[NSString stringWithUTF8String:]`
    fn ns_string(s: &str) -> Result<Id> {
        let c = CString::new(s).map_err(|_| CompileError {
            stage: "ffi",
            message: "path contains interior NUL".into(),
            detail: None,
        })?;
        Ok(unsafe {
            msg1(
                cls("NSString")?,
                sel("stringWithUTF8String:"),
                c.as_ptr() as Id,
            )
        })
    }

    /// `-[NSString UTF8String]` → owned String.
    fn utf8(ns: Id) -> Option<String> {
        if ns.is_null() {
            return None;
        }
        let p = unsafe { msg0(ns, sel("UTF8String")) } as *const c_char;
        if p.is_null() {
            return None;
        }
        let c = unsafe { std::ffi::CStr::from_ptr(p) };
        Some(c.to_string_lossy().into_owned())
    }

    /// `+[NSURL fileURLWithPath:]`
    fn file_url(path: &str) -> Result<Id> {
        Ok(unsafe { msg1(cls("NSURL")?, sel("fileURLWithPath:"), ns_string(path)?) })
    }

    /// NSError description string.
    fn err_desc(err: Id) -> String {
        if err.is_null() {
            return "unknown NSError".into();
        }
        let desc = unsafe { msg0(err, sel("localizedDescription")) };
        utf8(desc).unwrap_or_else(|| "unknown NSError".into())
    }

    /// Compile via `+[MLModel compileModelAtURL:error:]`.
    ///
    /// `url` must be a file URL to a `.mlpackage`. On success the return is
    /// an NSURL pointing at a temporary `.mlmodelc`; we copy it to `out_dir`
    /// so the caller controls the artifact's lifetime.
    pub(super) fn compile_ffi(pkg: &Path, out_dir: &Path) -> Result<CompiledModel> {
        let pool = unsafe { objc_autoreleasePoolPush() };
        let r = compile_ffi_inner(pkg, out_dir);
        unsafe { objc_autoreleasePoolPop(pool) };
        r
    }

    fn compile_ffi_inner(pkg: &Path, out_dir: &Path) -> Result<CompiledModel> {
        // Force the framework to load — objc_getClass returns nil until
        // the framework is mapped.
        let fw = CString::new("/System/Library/Frameworks/CoreML.framework/CoreML").unwrap();
        let handle = unsafe { dlopen(fw.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            return Err(CompileError {
                stage: "ffi",
                message: "dlopen(CoreML.framework) failed".into(),
                detail: None,
            });
        }

        let pkg_str = pkg.to_str().ok_or_else(|| CompileError {
            stage: "ffi",
            message: "package path is not UTF-8".into(),
            detail: None,
        })?;
        let url = file_url(pkg_str)?;

        let ml_model = cls("MLModel")?;
        let mut err: Id = std::ptr::null_mut();
        let err_ptr = &mut err as *mut Id as Id;
        let compiled = unsafe { msg2(ml_model, sel("compileModelAtURL:error:"), url, err_ptr) };
        if compiled.is_null() {
            return Err(CompileError {
                stage: "ffi",
                message: "MLModel compileModelAtURL returned nil".into(),
                detail: Some(err_desc(err)),
            });
        }

        // `[NSURL path]` → NSString
        let path_ns = unsafe { msg0(compiled, sel("path")) };
        let src = utf8(path_ns).ok_or_else(|| CompileError {
            stage: "ffi",
            message: "could not read compiled-model path".into(),
            detail: None,
        })?;
        let src = PathBuf::from(src);

        // The framework produced a temp .mlmodelc — move it under out_dir
        // with the package's stem so discovery is deterministic.
        std::fs::create_dir_all(out_dir)?;
        let stem = pkg.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        let dst = out_dir.join(format!("{}.mlmodelc", stem));
        if dst.exists() {
            std::fs::remove_dir_all(&dst)?;
        }
        std::fs::rename(&src, &dst).map_err(|e| CompileError {
            stage: "ffi",
            message: format!(
                "could not move {} → {}: {}",
                src.display(),
                dst.display(),
                e
            ),
            detail: None,
        })?;

        Ok(CompiledModel {
            path: dst,
            backend: Backend::CoreMlFramework,
            stdout: String::new(),
            stderr: String::new(),
        })
    }
}

/// Non-macOS stub — the FFI backend reports unavailable so callers fall
/// back to the subprocess.
#[cfg(not(target_os = "macos"))]
mod ffi {
    use super::*;
    pub(super) fn compile_ffi(_pkg: &Path, _out_dir: &Path) -> Result<CompiledModel> {
        Err(CompileError {
            stage: "ffi",
            message: "in-process compile requires macOS".into(),
            detail: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_mlmodelc_by_stem() {
        let root = std::env::temp_dir().join(format!("mc_disc_{}", std::process::id()));
        let out = root.join("out");
        std::fs::create_dir_all(&out).unwrap();
        let pkg = root.join("thing.mlpackage");
        std::fs::create_dir_all(&pkg).unwrap();
        std::fs::create_dir_all(out.join("thing.mlmodelc")).unwrap();
        let got = discover_mlmodelc(&out, &pkg).unwrap();
        assert!(got.ends_with("thing.mlmodelc"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn discover_errors_on_ambiguity() {
        let root = std::env::temp_dir().join(format!("mc_amb_{}", std::process::id()));
        let out = root.join("out");
        std::fs::create_dir_all(&out).unwrap();
        let pkg = root.join("a.mlpackage");
        std::fs::create_dir_all(&pkg).unwrap();
        std::fs::create_dir_all(out.join("x.mlmodelc")).unwrap();
        std::fs::create_dir_all(out.join("y.mlmodelc")).unwrap();
        assert!(discover_mlmodelc(&out, &pkg).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn coremlc_path_resolves_on_macos() {
        if cfg!(target_os = "macos") {
            let p = coremlc_path();
            // Only assert when CLT is actually installed.
            if let Some(p) = p {
                assert!(p.to_string_lossy().contains("coremlc"));
            }
        }
    }

    /// End-to-end subprocess compile — runs only when coremlc exists.
    /// The package is a trivial add graph; compile must accept it.
    #[test]
    fn subprocess_compiles_real_package() {
        if coremlc_path().is_none() {
            return;
        }
        let root = std::env::temp_dir().join(format!("mc_e2e_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let pkg = root.join("probe.mlpackage");

        let mut b = mil_spec::Block::new();
        let y = b.add("x", "x", &[1, 4, 1, 1], "y");
        b.outputs = vec![y];
        let spec = mil_spec::encode_model(
            &[mil_spec::Feature {
                name: "x".into(),
                shape: vec![1, 4, 1, 1],
                dtype: mil_spec::DType::Fp16,
                is_state: false,
            }],
            &[mil_spec::Feature {
                name: "y".into(),
                shape: vec![1, 4, 1, 1],
                dtype: mil_spec::DType::Fp16,
                is_state: false,
            }],
            &[],
            &b,
            &[mil_spec::NVT {
                name: "x".into(),
                ty: mil_spec::ValueType::Tensor(mil_spec::TensorType::f16(&[1, 4, 1, 1])),
            }],
            &mil_spec::ModelMeta::new(8, "CoreML5"),
        );
        mil_spec::write_mlpackage(&pkg, &spec, None).unwrap();

        let out_dir = root.join("out");
        let r = compile_with(Backend::Subprocess, &pkg, &out_dir);
        assert!(r.is_ok(), "coremlc rejected a valid graph: {:?}", r.err());
        let m = r.unwrap();
        assert!(m.path.ends_with("probe.mlmodelc"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
