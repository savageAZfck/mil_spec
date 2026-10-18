//! `e5rt` — execute models on the ANE through Espresso's `e5rt_*` C API.
//!
//! `dlopen` + `dlsym` only — no CoreML, no entitlement. The compile call
//! sends MIL to `aned` which runs the full MIL→MLIR→LLIR→ANECIR→HWX
//! pipeline and signs the program; after that every eval is
//! bind→encode→`execute_sync` on the hardware lane.
//!
//! ```text
//! let p = e5rt::Program::compile(mil_path, DeviceMask::ANE)?;
//! p.set_input_f16("x", &[f16s]);
//! p.execute()?;
//! let out = p.output_f16("z")?;
//! ```
//!
//! Call sequence recovered empirically (ane artifacts + probing); all
//! create entry points take the out-param FIRST except
//! `e5rt_e5_compiler_compile`, where the library out-param is last.

use std::ffi::{c_char, c_void, CString};
use std::path::Path;

#[link(name = "dl")]
extern "C" {}
extern "C" {
    fn dlopen(path: *const c_char, mode: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}

const RTLD_NOW: i32 = 2;
const ESPRESSO: &str = "/System/Library/PrivateFrameworks/Espresso.framework/Espresso";

/// e5rt error code; 0 = success.
type Rc = i64;

#[derive(Debug)]
pub enum E5Error {
    LoadFailed,
    Missing(&'static str),
    Call(&'static str, i64),
    Io(std::io::Error),
    Parse(String),
}

impl std::fmt::Display for E5Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            E5Error::LoadFailed => write!(f, "Espresso.framework failed to load"),
            E5Error::Missing(s) => write!(f, "missing e5rt symbol {s}"),
            E5Error::Call(s, rc) => write!(f, "{s} → {rc}"),
            E5Error::Io(e) => write!(f, "{e}"),
            E5Error::Parse(s) => write!(f, "{s}"),
        }
    }
}
impl std::error::Error for E5Error {}
impl From<std::io::Error> for E5Error {
    fn from(e: std::io::Error) -> Self {
        E5Error::Io(e)
    }
}

/// Compute-device bitmask for `set_compute_device_types_mask`.
#[derive(Clone, Copy)]
pub enum DeviceMask {
    /// BNNS (CPU).
    Cpu = 0x1,
    /// MPSGraph (GPU).
    Gpu = 0x2,
    /// Neural Engine only.
    Ane = 0x4,
    /// Planner picks (prefers ANE).
    Auto = 0x7,
}

/// Compiler configuration: where `aned` stages its cache bundle and
/// whether a compile forces a fresh build or may reuse what is on disk.
pub struct CompileCfg {
    /// `e5rt_e5_compiler_config_options_set_cache_bundle_location` target.
    pub cache_dir: std::path::PathBuf,
    /// `e5rt_e5_compiler_options_set_force_recompilation`.
    pub force_recompile: bool,
    /// `e5rt_e5_compiler_options_set_enable_profiling` — bakes
    /// perf-stats descriptors into the compiled program.
    pub profiling: bool,
}

impl Default for CompileCfg {
    fn default() -> Self {
        Self {
            cache_dir: std::env::temp_dir().join("e5_cache"),
            force_recompile: true,
            profiling: std::env::var_os("E5_PROFILE").is_some(),
        }
    }
}

macro_rules! e5sym {
    ($h:expr, $name:literal, ($($a:ty),*) -> $r:ty) => {{
        let c = CString::new($name).unwrap();
        let p = unsafe { dlsym($h, c.as_ptr()) };
        if p.is_null() {
            return Err(E5Error::Missing($name));
        }
        unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($a),*) -> $r>(p) }
    }};
}

pub(crate) struct Api {
    cfg_create: unsafe extern "C" fn(*mut *mut c_void) -> Rc,
    cfg_set_cache: unsafe extern "C" fn(*mut c_void, *const c_char) -> Rc,
    comp_create_cfg: unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> Rc,
    opts_create: unsafe extern "C" fn(*mut *mut c_void) -> Rc,
    opts_set_mask: unsafe extern "C" fn(*mut c_void, u64) -> Rc,
    opts_set_recomp: unsafe extern "C" fn(*mut c_void, i32) -> Rc,
    opts_set_segmenter: unsafe extern "C" fn(*mut c_void, *const c_char) -> Rc,
    opts_set_profiling: unsafe extern "C" fn(*mut c_void, i32) -> Rc,
    compile: unsafe extern "C" fn(*mut c_void, *const c_char, *mut c_void, *mut *mut c_void) -> Rc,
    lib_num_fns: unsafe extern "C" fn(*mut c_void, *mut u64) -> Rc,
    lib_fn_names: unsafe extern "C" fn(*mut c_void, u64, *mut *const c_char) -> Rc,
    lib_create: unsafe extern "C" fn(*const c_char, *mut *mut c_void) -> Rc,
    lib_retain_fn: unsafe extern "C" fn(*mut c_void, *const c_char, *mut *mut c_void) -> Rc,
    op_opts_create: unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> Rc,
    op_opts_set_name: unsafe extern "C" fn(*mut c_void, *const c_char) -> Rc,
    op_opts_set_intermed: unsafe extern "C" fn(*mut c_void, i32) -> Rc,
    op_create: unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> Rc,
    op_in_port: unsafe extern "C" fn(*mut c_void, *const c_char, *mut *mut c_void) -> Rc,
    op_out_port: unsafe extern "C" fn(*mut c_void, *const c_char, *mut *mut c_void) -> Rc,
    op_inout_port: unsafe extern "C" fn(*mut c_void, *const c_char, *mut *mut c_void) -> Rc,
    op_num_inputs: unsafe extern "C" fn(*mut c_void, *mut u64) -> Rc,
    op_input_names: unsafe extern "C" fn(*mut c_void, u64, *mut *const c_char) -> Rc,
    op_num_outputs: unsafe extern "C" fn(*mut c_void, *mut u64) -> Rc,
    op_output_names: unsafe extern "C" fn(*mut c_void, u64, *mut *const c_char) -> Rc,
    op_num_inouts: unsafe extern "C" fn(*mut c_void, *mut u64) -> Rc,
    op_inout_names: unsafe extern "C" fn(*mut c_void, u64, *mut *const c_char) -> Rc,
    port_tensor_desc: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> Rc,
    td_num_elems: unsafe extern "C" fn(*mut c_void, *mut u64) -> Rc,
    #[allow(dead_code)]
    td_size: unsafe extern "C" fn(*mut c_void, *mut u64) -> Rc,
    td_release: unsafe extern "C" fn(*mut *mut c_void) -> Rc,
    #[allow(dead_code)]
    op_prepare: unsafe extern "C" fn(*mut c_void) -> Rc,
    port_bind: unsafe extern "C" fn(*mut c_void, *mut c_void) -> Rc,
    buf_alloc: unsafe extern "C" fn(*mut *mut c_void, u64, u32) -> Rc,
    buf_ptr: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> Rc,
    buf_size: unsafe extern "C" fn(*mut c_void, *mut u64) -> Rc,
    stream_create: unsafe extern "C" fn(*mut *mut c_void) -> Rc,
    stream_encode: unsafe extern "C" fn(*mut c_void, *mut c_void) -> Rc,
    stream_exec: unsafe extern "C" fn(*mut c_void) -> Rc,
}

impl Api {
    pub(crate) fn open() -> Result<Self, E5Error> {
        let path = CString::new(ESPRESSO).unwrap();
        let h = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
        if h.is_null() {
            return Err(E5Error::LoadFailed);
        }
        Ok(Api {
            cfg_create: e5sym!(h, "e5rt_e5_compiler_config_options_create", (*mut *mut c_void) -> Rc),
            cfg_set_cache: e5sym!(h, "e5rt_e5_compiler_config_options_set_cache_bundle_location", (*mut c_void, *const c_char) -> Rc),
            comp_create_cfg: e5sym!(h, "e5rt_e5_compiler_create_with_config", (*mut *mut c_void, *mut c_void) -> Rc),
            opts_create: e5sym!(h, "e5rt_e5_compiler_options_create", (*mut *mut c_void) -> Rc),
            opts_set_mask: e5sym!(h, "e5rt_e5_compiler_options_set_compute_device_types_mask", (*mut c_void, u64) -> Rc),
            opts_set_recomp: e5sym!(h, "e5rt_e5_compiler_options_set_force_recompilation", (*mut c_void, i32) -> Rc),
            opts_set_segmenter: e5sym!(h, "e5rt_e5_compiler_options_set_segmenter", (*mut c_void, *const c_char) -> Rc),
            opts_set_profiling: e5sym!(h, "e5rt_e5_compiler_options_set_enable_profiling", (*mut c_void, i32) -> Rc),
            compile: e5sym!(h, "e5rt_e5_compiler_compile", (*mut c_void, *const c_char, *mut c_void, *mut *mut c_void) -> Rc),
            lib_num_fns: e5sym!(h, "e5rt_program_library_get_num_functions", (*mut c_void, *mut u64) -> Rc),
            lib_fn_names: e5sym!(h, "e5rt_program_library_get_function_names", (*mut c_void, u64, *mut *const c_char) -> Rc),
            lib_create: e5sym!(h, "e5rt_program_library_create", (*const c_char, *mut *mut c_void) -> Rc),
            lib_retain_fn: e5sym!(h, "e5rt_program_library_retain_program_function", (*mut c_void, *const c_char, *mut *mut c_void) -> Rc),
            op_opts_create: e5sym!(h, "e5rt_precompiled_compute_op_create_options_create_with_program_function", (*mut *mut c_void, *mut c_void) -> Rc),
            op_opts_set_name: e5sym!(h, "e5rt_precompiled_compute_op_create_options_set_operation_name", (*mut c_void, *const c_char) -> Rc),
            op_opts_set_intermed: e5sym!(h, "e5rt_precompiled_compute_op_create_options_set_allocate_intermediate_buffers", (*mut c_void, i32) -> Rc),
            op_create: e5sym!(h, "e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options", (*mut *mut c_void, *mut c_void) -> Rc),
            op_in_port: e5sym!(h, "e5rt_execution_stream_operation_retain_input_port", (*mut c_void, *const c_char, *mut *mut c_void) -> Rc),
            op_out_port: e5sym!(h, "e5rt_execution_stream_operation_retain_output_port", (*mut c_void, *const c_char, *mut *mut c_void) -> Rc),
            op_inout_port: e5sym!(h, "e5rt_execution_stream_operation_retain_inout_port", (*mut c_void, *const c_char, *mut *mut c_void) -> Rc),
            op_num_inouts: e5sym!(h, "e5rt_execution_stream_operation_get_num_inouts", (*mut c_void, *mut u64) -> Rc),
            op_inout_names: e5sym!(h, "e5rt_execution_stream_operation_get_inout_names", (*mut c_void, u64, *mut *const c_char) -> Rc),
            op_num_inputs: e5sym!(h, "e5rt_execution_stream_operation_get_num_inputs", (*mut c_void, *mut u64) -> Rc),
            op_input_names: e5sym!(h, "e5rt_execution_stream_operation_get_input_names", (*mut c_void, u64, *mut *const c_char) -> Rc),
            op_num_outputs: e5sym!(h, "e5rt_execution_stream_operation_get_num_outputs", (*mut c_void, *mut u64) -> Rc),
            op_output_names: e5sym!(h, "e5rt_execution_stream_operation_get_output_names", (*mut c_void, u64, *mut *const c_char) -> Rc),
            port_tensor_desc: e5sym!(h, "e5rt_io_port_retain_tensor_desc", (*mut c_void, *mut *mut c_void) -> Rc),
            td_num_elems: e5sym!(h, "e5rt_tensor_desc_get_num_elements", (*mut c_void, *mut u64) -> Rc),
            td_size: e5sym!(h, "e5rt_tensor_desc_get_size", (*mut c_void, *mut u64) -> Rc),
            td_release: e5sym!(h, "e5rt_tensor_desc_release", (*mut *mut c_void) -> Rc),
            op_prepare: e5sym!(h, "e5rt_execution_stream_operation_prepare_op_for_encode", (*mut c_void) -> Rc),
            port_bind: e5sym!(h, "e5rt_io_port_bind_buffer_object", (*mut c_void, *mut c_void) -> Rc),
            buf_alloc: e5sym!(h, "e5rt_buffer_object_alloc", (*mut *mut c_void, u64, u32) -> Rc),
            buf_ptr: e5sym!(h, "e5rt_buffer_object_get_data_ptr", (*mut c_void, *mut *mut c_void) -> Rc),
            buf_size: e5sym!(h, "e5rt_buffer_object_get_size", (*mut c_void, *mut u64) -> Rc),
            stream_create: e5sym!(h, "e5rt_execution_stream_create", (*mut *mut c_void) -> Rc),
            stream_encode: e5sym!(h, "e5rt_execution_stream_encode_operation", (*mut c_void, *mut c_void) -> Rc),
            stream_exec: e5sym!(h, "e5rt_execution_stream_execute_sync", (*mut c_void) -> Rc),
        })
    }
}

impl Api {
    /// Retain an input/output/inout port on `op`.
    pub(crate) fn port(
        &self,
        op: *mut c_void,
        kind: PortKind,
        name: &str,
    ) -> Result<*mut c_void, E5Error> {
        let f = match kind {
            PortKind::In => self.op_in_port,
            PortKind::Out => self.op_out_port,
            PortKind::InOut => self.op_inout_port,
        };
        let mut p = std::ptr::null_mut();
        check(unsafe { f(op, cstr(name).as_ptr(), &mut p) }, "retain_port")?;
        Ok(p)
    }

    /// Port names by kind.
    pub(crate) fn port_names(&self, op: *mut c_void, kind: PortKind) -> Vec<String> {
        let (num, get) = match kind {
            PortKind::In => (self.op_num_inputs, self.op_input_names),
            PortKind::Out => (self.op_num_outputs, self.op_output_names),
            PortKind::InOut => (self.op_num_inouts, self.op_inout_names),
        };
        unsafe {
            let mut n = 0u64;
            if num(op, &mut n) != 0 || n == 0 {
                return vec![];
            }
            let mut ptrs = vec![std::ptr::null::<c_char>(); n as usize];
            if get(op, n, ptrs.as_mut_ptr()) != 0 {
                return vec![];
            }
            ptrs.iter()
                .filter(|p| !p.is_null())
                .map(|p| std::ffi::CStr::from_ptr(*p).to_string_lossy().into_owned())
                .collect()
        }
    }

    /// Element count for a port's tensor descriptor.
    pub(crate) fn port_elems(&self, port: *mut c_void) -> u64 {
        unsafe {
            let mut td = std::ptr::null_mut();
            if (self.port_tensor_desc)(port, &mut td) != 0 || td.is_null() {
                return 0;
            }
            let mut elems = 0u64;
            let _ = (self.td_num_elems)(td, &mut elems);
            let _ = (self.td_release)(&mut td);
            elems
        }
    }

    /// Allocate a CPU-type buffer of `bytes`.
    pub(crate) fn buf(&self, bytes: u64) -> Result<*mut c_void, E5Error> {
        let mut b = std::ptr::null_mut();
        check(
            unsafe { (self.buf_alloc)(&mut b, bytes, 0) },
            "buffer_object_alloc",
        )?;
        Ok(b)
    }

    /// Bind `buf` to `port`.
    pub(crate) fn bind(&self, port: *mut c_void, buf: *mut c_void) -> Result<(), E5Error> {
        check(unsafe { (self.port_bind)(port, buf) }, "io_port_bind")
    }

    /// Host data pointer for a buffer.
    pub(crate) fn ptr(&self, buf: *mut c_void) -> Result<*mut u16, E5Error> {
        let mut p = std::ptr::null_mut();
        check(
            unsafe { (self.buf_ptr)(buf, &mut p) },
            "buffer_get_data_ptr",
        )?;
        Ok(p as *mut u16)
    }

    /// New execution stream.
    pub(crate) fn stream(&self) -> Result<*mut c_void, E5Error> {
        let mut s = std::ptr::null_mut();
        check(unsafe { (self.stream_create)(&mut s) }, "stream_create")?;
        Ok(s)
    }

    pub(crate) fn encode(&self, stream: *mut c_void, op: *mut c_void) -> Result<(), E5Error> {
        check(
            unsafe { (self.stream_encode)(stream, op) },
            "encode_operation",
        )
    }

    pub(crate) fn exec(&self, stream: *mut c_void) -> Result<(), E5Error> {
        check(unsafe { (self.stream_exec)(stream) }, "execute_sync")
    }
}

/// Which of an op's three port families.
#[derive(Clone, Copy)]
pub(crate) enum PortKind {
    In,
    Out,
    InOut,
}

fn check(rc: Rc, what: &'static str) -> Result<(), E5Error> {
    if rc == 0 {
        Ok(())
    } else {
        Err(E5Error::Call(what, rc))
    }
}

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

/// A bound port: retained IOPort + its backing BufferObject.
struct Bound {
    #[allow(dead_code)]
    port: *mut c_void,
    buf: *mut c_void,
    bytes: u64,
}

/// Every port the op declares, with element counts from its own
/// tensor descriptors.
#[derive(Debug, Default)]
pub struct PortMap {
    pub inputs: Vec<(String, u64)>,
    pub inouts: Vec<(String, u64)>,
    pub outputs: Vec<(String, u64)>,
}

/// A compiled, bound, encoded program — compile once, eval many.
pub struct Program {
    api: Api,
    stream: *mut c_void,
    inputs: Vec<(String, Bound)>,
    /// InOut (state) ports — readable and writable, resident across calls.
    pub inouts: Vec<String>,
    inout_bounds: Vec<(String, Bound)>,
    outputs: Vec<(String, Bound)>,
    /// Function names exported by the compiled library.
    pub functions: Vec<String>,
}

impl Program {
    /// Compile `mil` (path to a `.mil` text file, or `.mlmodelc`) through
    /// `aned` for the given device, then build the bound op + stream.
    ///
    /// `inputs`/`inouts`/`outputs`: `(port_name, n_elems)` fp16 ports to
    /// bind. Stateful models: `state<>` params are *inout* ports —
    /// read and written in place by the engine across calls.
    /// Compile `mil` for `mask` and return (op, function_names) on `api`.
    pub(crate) fn make_op_on(
        api: &Api,
        mil: &Path,
        mask: DeviceMask,
    ) -> Result<(*mut c_void, Vec<String>), E5Error> {
        Self::make_op_on_cfg(api, mil, mask, &CompileCfg::default())
    }

    /// Like `make_op_on` but with explicit cache location / recompilation.
    pub(crate) fn make_op_on_cfg(
        api: &Api,
        mil: &Path,
        mask: DeviceMask,
        cc: &CompileCfg,
    ) -> Result<(*mut c_void, Vec<String>), E5Error> {
        unsafe {
            let mut cfg = std::ptr::null_mut();
            check((api.cfg_create)(&mut cfg), "config_options_create")?;
            let _ = std::fs::create_dir_all(&cc.cache_dir);
            let cache = cstr(&cc.cache_dir.to_string_lossy());
            check(
                (api.cfg_set_cache)(cfg, cache.as_ptr()),
                "set_cache_bundle_location",
            )?;
            let mut comp = std::ptr::null_mut();
            check((api.comp_create_cfg)(&mut comp, cfg), "e5_compiler_create")?;

            let mut opts = std::ptr::null_mut();
            check((api.opts_create)(&mut opts), "compiler_options_create")?;
            check(
                (api.opts_set_mask)(opts, mask as u64),
                "set_compute_device_types_mask",
            )?;
            check(
                (api.opts_set_recomp)(opts, cc.force_recompile as i32),
                "set_force_recompilation",
            )?;
            check(
                (api.opts_set_segmenter)(opts, cstr("graph").as_ptr()),
                "set_segmenter",
            )?;
            if cc.profiling {
                check((api.opts_set_profiling)(opts, 1), "set_enable_profiling")?;
            }

            let mil_c = cstr(&mil.to_string_lossy());
            let mut lib = std::ptr::null_mut();
            check(
                (api.compile)(comp, mil_c.as_ptr(), opts, &mut lib),
                "e5_compiler_compile",
            )?;
            Self::make_op_from_lib(api, lib)
        }
    }

    /// Turn an already-created program library into a bound op: enumerate
    /// functions, retain `main`, build the precompiled compute op.
    fn make_op_from_lib(
        api: &Api,
        lib: *mut c_void,
    ) -> Result<(*mut c_void, Vec<String>), E5Error> {
        unsafe {
            // Enumerate exported functions (diagnostics).
            let mut n = 0u64;
            let _ = (api.lib_num_fns)(lib, &mut n);
            let mut names = vec![std::ptr::null::<c_char>(); n as usize];
            let functions = if n > 0 && (api.lib_fn_names)(lib, n, names.as_mut_ptr()) == 0 {
                names
                    .iter()
                    .filter(|p| !p.is_null())
                    .map(|p| std::ffi::CStr::from_ptr(*p).to_string_lossy().into_owned())
                    .collect()
            } else {
                vec![]
            };

            let mut func = std::ptr::null_mut();
            check(
                (api.lib_retain_fn)(lib, cstr("main").as_ptr(), &mut func),
                "retain_program_function(main)",
            )?;
            let mut oo = std::ptr::null_mut();
            check((api.op_opts_create)(&mut oo, func), "op_create_options")?;
            check(
                (api.op_opts_set_name)(oo, cstr("main").as_ptr()),
                "op_set_name",
            )?;
            check(
                (api.op_opts_set_intermed)(oo, 1),
                "op_set_intermediate_buffers",
            )?;
            let mut op = std::ptr::null_mut();
            check((api.op_create)(&mut op, oo), "op_create")?;
            Ok((op, functions))
        }
    }

    /// Build an op directly from an on-disk e5 bundle (`H16G.bundle` /
    /// `.e5` dir) — the precompiled path, no MIL compile at all.
    pub(crate) fn make_op_from_bundle(
        api: &Api,
        bundle: &Path,
    ) -> Result<(*mut c_void, Vec<String>), E5Error> {
        unsafe {
            let mut lib = std::ptr::null_mut();
            check(
                (api.lib_create)(cstr(&bundle.to_string_lossy()).as_ptr(), &mut lib),
                "program_library_create",
            )?;
            Self::make_op_from_lib(api, lib)
        }
    }

    fn bind_ports(
        api: &Api,
        op: *mut c_void,
        port_fn: unsafe extern "C" fn(*mut c_void, *const c_char, *mut *mut c_void) -> Rc,
        names: &[(String, u64)],
    ) -> Result<Vec<(String, Bound)>, E5Error> {
        names
            .iter()
            .map(|(n, e)| {
                let mut port = std::ptr::null_mut();
                check(
                    unsafe { port_fn(op, cstr(n).as_ptr(), &mut port) },
                    "retain_port",
                )?;
                let mut buf = std::ptr::null_mut();
                let bytes = e * 2;
                check(
                    unsafe { (api.buf_alloc)(&mut buf, bytes, 0) },
                    "buffer_object_alloc",
                )?;
                check(unsafe { (api.port_bind)(port, buf) }, "io_port_bind")?;
                Ok((n.clone(), Bound { port, buf, bytes }))
            })
            .collect()
    }

    /// Names/sizes for every port the op declares, discovered from the
    /// op's own descriptors — no caller-supplied port list needed.
    fn port_map(op: *mut c_void, api: &Api) -> PortMap {
        let names = |num: unsafe extern "C" fn(*mut c_void, *mut u64) -> Rc,
                     get: unsafe extern "C" fn(*mut c_void, u64, *mut *const c_char) -> Rc|
         -> Vec<String> {
            unsafe {
                let mut n = 0u64;
                if num(op, &mut n) != 0 || n == 0 {
                    return vec![];
                }
                let mut ptrs = vec![std::ptr::null::<c_char>(); n as usize];
                if get(op, n, ptrs.as_mut_ptr()) != 0 {
                    return vec![];
                }
                ptrs.iter()
                    .filter(|p| !p.is_null())
                    .map(|p| std::ffi::CStr::from_ptr(*p).to_string_lossy().into_owned())
                    .collect()
            }
        };
        let size_of =
            |port_fn: unsafe extern "C" fn(*mut c_void, *const c_char, *mut *mut c_void) -> Rc,
             name: &str|
             -> u64 {
                unsafe {
                    let mut port = std::ptr::null_mut();
                    if port_fn(op, cstr(name).as_ptr(), &mut port) != 0 || port.is_null() {
                        return 0;
                    }
                    let mut td = std::ptr::null_mut();
                    if (api.port_tensor_desc)(port, &mut td) != 0 || td.is_null() {
                        return 0;
                    }
                    let mut elems = 0u64;
                    let _ = (api.td_num_elems)(td, &mut elems);
                    let _ = (api.td_release)(&mut td);
                    elems
                }
            };
        let inputs = names(api.op_num_inputs, api.op_input_names);
        let outputs = names(api.op_num_outputs, api.op_output_names);
        let inouts = names(api.op_num_inouts, api.op_inout_names);
        PortMap {
            inputs: inputs
                .iter()
                .map(|n| (n.clone(), size_of(api.op_in_port, n)))
                .collect(),
            outputs: outputs
                .iter()
                .map(|n| (n.clone(), size_of(api.op_out_port, n)))
                .collect(),
            inouts: inouts
                .iter()
                .map(|n| (n.clone(), size_of(api.op_inout_port, n)))
                .collect(),
        }
    }

    /// Compile + self-configuring bind: port names and element counts
    /// are read back from the op's tensor descriptors.
    pub fn compile_auto(mil: &Path, mask: DeviceMask) -> Result<(Self, PortMap), E5Error> {
        Self::compile_auto_cfg(mil, mask, &CompileCfg::default())
    }

    /// Load a precompiled e5 bundle directory (`H16G.bundle` or the `.e5`
    /// bundle dir) and bind it for execution — no compilation.
    pub fn from_e5_bundle(bundle: &Path) -> Result<(Self, PortMap), E5Error> {
        let api = Api::open()?;
        let (op, functions) = Self::make_op_from_bundle(&api, bundle)?;
        let map = Self::port_map(op, &api);
        let ins = Self::bind_ports(&api, op, api.op_in_port, &map.inputs)?;
        let inouts = Self::bind_ports(&api, op, api.op_inout_port, &map.inouts)?;
        let outs = Self::bind_ports(&api, op, api.op_out_port, &map.outputs)?;
        unsafe {
            let mut stream = std::ptr::null_mut();
            check((api.stream_create)(&mut stream), "stream_create")?;
            check((api.stream_encode)(stream, op), "encode_operation")?;
            Ok((
                Program {
                    api,
                    stream,
                    inputs: ins,
                    inouts: inouts.iter().map(|(n, _)| n.clone()).collect(),
                    inout_bounds: inouts,
                    outputs: outs,
                    functions,
                },
                map,
            ))
        }
    }

    /// `compile_auto` with explicit cache location / recompilation control.
    pub fn compile_auto_cfg(
        mil: &Path,
        mask: DeviceMask,
        cc: &CompileCfg,
    ) -> Result<(Self, PortMap), E5Error> {
        let api = Api::open()?;
        let (op, functions) = Self::make_op_on_cfg(&api, mil, mask, cc)?;
        let map = Self::port_map(op, &api);
        let ins = Self::bind_ports(&api, op, api.op_in_port, &map.inputs)?;
        let inouts = Self::bind_ports(&api, op, api.op_inout_port, &map.inouts)?;
        let outs = Self::bind_ports(&api, op, api.op_out_port, &map.outputs)?;
        unsafe {
            let mut stream = std::ptr::null_mut();
            check((api.stream_create)(&mut stream), "stream_create")?;
            check((api.stream_encode)(stream, op), "encode_operation")?;
            Ok((
                Program {
                    api,
                    stream,
                    inputs: ins,
                    inouts: inouts.iter().map(|(n, _)| n.clone()).collect(),
                    inout_bounds: inouts,
                    outputs: outs,
                    functions,
                },
                map,
            ))
        }
    }

    pub fn compile(
        mil: &Path,
        mask: DeviceMask,
        inputs: &[(&str, u64)],
        outputs: &[(&str, u64)],
        inouts: &[(&str, u64)],
    ) -> Result<Self, E5Error> {
        Self::compile_cfg(mil, mask, inputs, outputs, inouts, &CompileCfg::default())
    }

    /// `compile` with explicit cache location / recompilation control.
    pub fn compile_cfg(
        mil: &Path,
        mask: DeviceMask,
        inputs: &[(&str, u64)],
        outputs: &[(&str, u64)],
        inouts: &[(&str, u64)],
        cc: &CompileCfg,
    ) -> Result<Self, E5Error> {
        let api = Api::open()?;
        let (op, functions) = Self::make_op_on_cfg(&api, mil, mask, cc)?;
        unsafe {
            let ins = Self::bind_ports(
                &api,
                op,
                api.op_in_port,
                &inputs
                    .iter()
                    .map(|(n, e)| (n.to_string(), *e))
                    .collect::<Vec<_>>(),
            )?;
            let inouts = Self::bind_ports(
                &api,
                op,
                api.op_inout_port,
                &inouts
                    .iter()
                    .map(|(n, e)| (n.to_string(), *e))
                    .collect::<Vec<_>>(),
            )?;
            let outs = Self::bind_ports(
                &api,
                op,
                api.op_out_port,
                &outputs
                    .iter()
                    .map(|(n, e)| (n.to_string(), *e))
                    .collect::<Vec<_>>(),
            )?;

            let mut stream = std::ptr::null_mut();
            check((api.stream_create)(&mut stream), "stream_create")?;
            check((api.stream_encode)(stream, op), "encode_operation")?;
            Ok(Program {
                api,
                stream,
                inputs: ins,
                inouts: inouts.iter().map(|(n, _)| n.clone()).collect(),
                inout_bounds: inouts,
                outputs: outs,
                functions,
            })
        }
    }

    fn data_ptr(&self, b: &Bound) -> Result<*mut u16, E5Error> {
        let mut p = std::ptr::null_mut();
        check(
            unsafe { (self.api.buf_ptr)(b.buf, &mut p) },
            "buffer_get_data_ptr",
        )?;
        Ok(p as *mut u16)
    }

    /// Write fp16 (as raw `u16` half-bits) into a bound input port.
    pub fn set_input_f16(&self, name: &str, data: &[u16]) -> Result<(), E5Error> {
        let b = self
            .inputs
            .iter()
            .chain(self.inout_bounds.iter())
            .find(|(n, _)| n == name)
            .map(|(_, b)| b)
            .ok_or(E5Error::Call("set_input: no such port", -1))?;
        if data.len() as u64 * 2 > b.bytes {
            return Err(E5Error::Call("set_input: overflow", -1));
        }
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.data_ptr(b)?, data.len()) };
        Ok(())
    }

    /// Read fp16 bits from a bound output port.
    pub fn output_f16(&self, name: &str) -> Result<Vec<u16>, E5Error> {
        let b = self
            .outputs
            .iter()
            .chain(self.inout_bounds.iter())
            .find(|(n, _)| n == name)
            .map(|(_, b)| b)
            .ok_or(E5Error::Call("output: no such port", -1))?;
        let mut n = 0u64;
        unsafe { (self.api.buf_size)(b.buf, &mut n) };
        let p = self.data_ptr(b)?;
        Ok(unsafe { std::slice::from_raw_parts(p, (n / 2) as usize) }.to_vec())
    }

    /// Run the encoded stream synchronously on the hardware lane.
    pub fn execute(&self) -> Result<(), E5Error> {
        check(
            unsafe { (self.api.stream_exec)(self.stream) },
            "execute_sync",
        )
    }
}

unsafe impl Send for Program {}
unsafe impl Sync for Program {}
