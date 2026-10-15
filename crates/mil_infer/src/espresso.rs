//! `espresso` — execute precompiled ANE programs through `aned`'s
//! deployment path (`_ANEModel` / `_ANEClient` over XPC).
//!
//! This is the loader the OS itself uses for the `.hwx` programs baked
//! into frameworks: a directory holding
//!
//! ```text
//! <name>.espresso.net                  (JSON graph)
//! <name>.espresso.shape                (JSON tensor shapes)
//! <name>.espresso.weights              (weight blob index)
//! <name>.H16.espresso.hwx              (ANE Mach-O — ours, not Apple's)
//! <name>.H16.espresso.precompilation_info
//! ```
//!
//! `_ANEModel modelAtURL:` opens the directory, `aned` takes the
//! precompiled path (`isPreCompiledModel=1`), and evaluation goes
//! through `_ANERequest` objects carrying `_ANEIOSurfaceObject`
//! buffers. No MIL compile, no entitlement — `sharedConnection` is
//! enough.
//!
//! ```text
//! let m = EspressoProgram::load(dir, surface_bytes)?;
//! m.input_mut(0)[..128].copy_from_slice(&fp16_lane_data);
//! m.eval()?;
//! let z = m.output(0);
//! ```
//!
//! Recovered empirically (see `examples/gate0.rs ownexec`): `aned`
//! loads a self-compiled ANE Mach-O and reports its real IO geometry;
//! `evaluateRealTimeWithModel:` then executes it on the engine.

use std::ffi::{c_char, c_void, CString};
use std::path::Path;

extern "C" {
    fn dlopen(path: *const c_char, mode: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}

#[link(name = "objc")]
extern "C" {
    fn objc_getClass(name: *const c_char) -> Id;
    fn sel_registerName(name: *const c_char) -> Sel;
    fn objc_msgSend();
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

const RTLD_NOW: i32 = 2;
const ANE_FW: &str =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";
const IOSURFACE: &str = "/System/Library/Frameworks/IOSurface.framework/IOSurface";
const FOUNDATION: &str = "/System/Library/Frameworks/Foundation.framework/Foundation";
const COREFOUNDATION: &str = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation";

type Id = *mut c_void;
type Sel = *mut c_void;

#[derive(Debug)]
pub enum EspError {
    FrameworkLoad,
    MissingClass(&'static str),
    MissingSel(&'static str),
    Msg(String),
}

impl std::fmt::Display for EspError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EspError::FrameworkLoad => write!(f, "framework dlopen failed"),
            EspError::MissingClass(c) => write!(f, "ObjC class {c} not registered"),
            EspError::MissingSel(s) => write!(f, "symbol {s} failed to resolve"),
            EspError::Msg(s) => write!(f, "{s}"),
        }
    }
}
impl std::error::Error for EspError {}

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn selp(name: &str) -> Sel {
    unsafe { sel_registerName(cstr(name).as_ptr()) }
}

/// `objc_msgSend` typed by signature.
macro_rules! msg {
    (($($t:ty),*) -> $r:ty) => {
        std::mem::transmute::<usize, unsafe extern "C" fn(Id, Sel $(, $t)*) -> $r>(
            objc_msgSend as *const () as usize,
        )
    };
}

/// ObjC class handles we drive.
struct Objc {
    ane_model: Id,
    ane_client: Id,
    ane_request: Id,
    ane_surf: Id,
    ns_string: Id,
    ns_url: Id,
    ns_array: Id,
    ns_number: Id,
    ns_dict: Id,
}

impl Objc {
    fn open() -> Result<Self, EspError> {
        for fw in [FOUNDATION, IOSURFACE, ANE_FW] {
            if unsafe { dlopen(cstr(fw).as_ptr(), RTLD_NOW) }.is_null() {
                return Err(EspError::FrameworkLoad);
            }
        }
        let cls = |n: &'static str| -> Result<Id, EspError> {
            let c = unsafe { objc_getClass(cstr(n).as_ptr()) };
            if c.is_null() {
                Err(EspError::MissingClass(n))
            } else {
                Ok(c)
            }
        };
        Ok(Objc {
            ane_model: cls("_ANEModel")?,
            ane_client: cls("_ANEClient")?,
            ane_request: cls("_ANERequest")?,
            ane_surf: cls("_ANEIOSurfaceObject")?,
            ns_string: cls("NSString")?,
            ns_url: cls("NSURL")?,
            ns_array: cls("NSArray")?,
            ns_number: cls("NSNumber")?,
            ns_dict: cls("NSDictionary")?,
        })
    }

    fn ns_string(&self, s: &str) -> Id {
        unsafe {
            let f = msg!((Id) -> Id);
            f(
                self.ns_string,
                selp("stringWithUTF8String:"),
                cstr(s).as_ptr() as Id,
            )
        }
    }
    fn ns_url(&self, path: &Path) -> Id {
        unsafe {
            let s = self.ns_string(&path.to_string_lossy());
            let f = msg!((Id) -> Id);
            f(self.ns_url, selp("fileURLWithPath:"), s)
        }
    }
    fn ns_int(&self, v: u64) -> Id {
        unsafe {
            let f = msg!((u64) -> Id);
            f(self.ns_number, selp("numberWithUnsignedLongLong:"), v)
        }
    }
    fn ns_arr(&self, objs: &[Id]) -> Id {
        unsafe {
            let f = msg!((*const Id, usize) -> Id);
            f(
                self.ns_array,
                selp("arrayWithObjects:count:"),
                objs.as_ptr(),
                objs.len(),
            )
        }
    }
    fn empty_dict(&self) -> Id {
        unsafe {
            let f = msg!(() -> Id);
            f(self.ns_dict, selp("dictionary"))
        }
    }

    /// `NSDictionary dictionaryWithObjects:forKeys:count:` for one pair.
    fn one_dict(&self, key: &str, val: Id) -> Id {
        unsafe {
            let k = self.ns_string(key);
            msg!((*const Id, *const Id, usize) -> Id)(
                self.ns_dict,
                selp("dictionaryWithObjects:forKeys:count:"),
                &val as *const Id,
                &k as *const Id,
                1,
            )
        }
    }

    /// `dictionaryWithObjects:forKeys:count:` over (key, u64) pairs.
    fn num_dict(&self, pairs: &[(&str, u64)]) -> Id {
        unsafe {
            let keys: Vec<Id> = pairs.iter().map(|(k, _)| self.ns_string(k)).collect();
            let vals: Vec<Id> = pairs.iter().map(|(_, v)| self.ns_int(*v)).collect();
            msg!((*const Id, *const Id, usize) -> Id)(
                self.ns_dict,
                selp("dictionaryWithObjects:forKeys:count:"),
                vals.as_ptr(),
                keys.as_ptr(),
                pairs.len(),
            )
        }
    }

    /// `NSMutableDictionary setObject:forKey:`.
    fn dict_set(&self, dict: Id, key: &str, val: Id) {
        unsafe {
            let k = self.ns_string(key);
            msg!((Id, Id) -> ())(dict, selp("setObject:forKey:"), val, k);
        }
    }

    /// `objectForKey:` on an NSDictionary (returns null for missing keys).
    fn dict_get(&self, dict: Id, key: &str) -> Id {
        if dict.is_null() {
            return std::ptr::null_mut();
        }
        unsafe {
            let k = self.ns_string(key);
            msg!((Id) -> Id)(dict, selp("objectForKey:"), k)
        }
    }

    /// `allKeys` → Vec<String>.
    fn dict_keys(&self, dict: Id) -> Vec<String> {
        if dict.is_null() {
            return vec![];
        }
        unsafe {
            let keys = msg!(() -> Id)(dict, selp("allKeys"));
            let n = msg!(() -> usize)(keys, selp("count"));
            (0..n)
                .map(|i| {
                    let k = msg!((usize) -> Id)(keys, selp("objectAtIndex:"), i);
                    ns_str(k)
                })
                .collect()
        }
    }

    /// `objectAtIndex:` on an NSArray.
    fn arr_at(&self, arr: Id, i: usize) -> Id {
        if arr.is_null() {
            return std::ptr::null_mut();
        }
        unsafe { msg!((usize) -> Id)(arr, selp("objectAtIndex:"), i) }
    }
    fn arr_count(&self, arr: Id) -> usize {
        if arr.is_null() {
            return 0;
        }
        unsafe { msg!(() -> usize)(arr, selp("count")) }
    }
    /// NSNumber → u64.
    fn num(&self, n: Id) -> u64 {
        if n.is_null() {
            return 0;
        }
        unsafe { msg!(() -> u64)(n, selp("unsignedLongLongValue")) }
    }
}

fn retain(o: Id) -> Id {
    unsafe { msg!(() -> Id)(o, selp("retain")) }
}
fn release(o: Id) {
    unsafe { msg!(() -> ())(o, selp("release")) }
}

/// IOSurface/CoreFoundation C entry points via `dlsym`.
struct IoApi {
    get_base: unsafe extern "C" fn(Id) -> *mut c_void,
    alloc_size: unsafe extern "C" fn(Id) -> usize,
    cf_release: unsafe extern "C" fn(Id),
}

impl IoApi {
    fn open() -> Result<Self, EspError> {
        let hi = unsafe { dlopen(cstr(IOSURFACE).as_ptr(), RTLD_NOW) };
        let hc = unsafe { dlopen(cstr(COREFOUNDATION).as_ptr(), RTLD_NOW) };
        if hi.is_null() || hc.is_null() {
            return Err(EspError::FrameworkLoad);
        }
        let sym = |h: *mut c_void, n: &'static str| -> Result<*mut c_void, EspError> {
            let p = unsafe { dlsym(h, cstr(n).as_ptr()) };
            if p.is_null() {
                Err(EspError::MissingSel(n))
            } else {
                Ok(p)
            }
        };
        Ok(IoApi {
            get_base: unsafe { std::mem::transmute(sym(hi, "IOSurfaceGetBaseAddress")?) },
            alloc_size: unsafe { std::mem::transmute(sym(hi, "IOSurfaceGetAllocSize")?) },
            cf_release: unsafe { std::mem::transmute(sym(hc, "CFRelease")?) },
        })
    }
}

/// One IO symbol's layout as `aned` decoded it from the hwx
/// (`NetworkStatusList` entry). Strides are in bytes; element `(b, d,
/// c, h, w)` lands at
/// `b*batch_stride + d*depth_stride + c*plane_stride + h*row_stride + w*elem`.
#[derive(Debug, Clone, Default)]
pub struct TensorDesc {
    /// Symbol name from the package `.net` (`x`, `z@output`, ...).
    pub name: String,
    /// Element type string (`Float16`, ...).
    pub ty: String,
    /// Batches.
    pub n: u64,
    /// Channels (the `k` dimension).
    pub k: u64,
    /// Height.
    pub h: u64,
    /// Width.
    pub w: u64,
    /// Bytes per channel plane.
    pub plane_stride: u64,
    /// Bytes per row.
    pub row_stride: u64,
    /// Bytes per batch.
    pub batch_stride: u64,
    /// Bytes per depth step.
    pub depth_stride: u64,
    /// Element size in bytes (2 for fp16).
    pub elem: u64,
}

impl TensorDesc {
    /// Byte offset of element `(b, c, h, w)` (depth folded into batch
    /// for the flat 4-D case — `d` rides `depth_stride` if present).
    pub fn offset(&self, b: u64, d: u64, c: u64, h: u64, w: u64) -> usize {
        (b * self.batch_stride
            + d * self.depth_stride
            + c * self.plane_stride
            + h * self.row_stride
            + w * self.elem) as usize
    }

    /// Total bytes the IOSurface needs (batches are the outermost dim).
    pub fn bytes(&self) -> usize {
        let span = self
            .batch_stride
            .max(self.depth_stride * self.n.max(1))
            .max(self.plane_stride * self.k.max(1))
            .max(self.row_stride * self.h.max(1))
            .max(self.elem * self.w.max(1));
        (span.max(16384)) as usize
    }
}

/// One program IO buffer: an IOSurface plus its `_ANEIOSurfaceObject`
/// wrapper (what `_ANERequest` wants in its arrays).
pub struct Surface {
    /// Raw `IOSurfaceRef` (+1, released on drop).
    surf: Id,
    /// Retained `_ANEIOSurfaceObject`.
    wrap: Id,
    len: usize,
}

/// A precompiled Espresso package loaded into `aned`, buffers bound.
pub struct EspressoProgram {
    objc: Objc,
    io: IoApi,
    model: Id,
    client: Id,
    /// Input surfaces in symbol order (`inputSyms` indexes 0..n).
    pub inputs: Vec<Surface>,
    /// Output surfaces in symbol order.
    pub outputs: Vec<Surface>,
    /// Per-symbol input layouts from `NetworkStatusList`.
    pub input_descs: Vec<TensorDesc>,
    /// Per-symbol output layouts.
    pub output_descs: Vec<TensorDesc>,
    /// Procedure name → procedureIndex map
    /// (`kANEFModelProcedureNameToIDMapKey`).
    pub procedures: Vec<(String, u64)>,
    /// `numInputs` / `numOutputs` as `aned` reported from the hwx.
    pub num_inputs: u64,
    pub num_outputs: u64,
    /// `aned`'s program handle.
    pub program_handle: u64,
}

impl EspressoProgram {
    /// Load the package directory through `_ANEClient sharedConnection`
    /// and allocate one IOSurface per IO symbol, sized from the model's
    /// own `NetworkStatusList` descriptors.
    pub fn load(dir: &Path) -> Result<Self, EspError> {
        let pool = unsafe { objc_autoreleasePoolPush() };
        let r = Self::load_inner(dir, false, &[]);
        unsafe { objc_autoreleasePoolPop(pool) };
        r
    }

    /// Load a bare `.hwx`/`.macho` file as a precompiled model:
    /// `modelAtURL:` on the file plus `loadModel` options
    /// `{ANEFModelType: ANEFModelPreCompiled}` — the trigger aned uses
    /// for OS-shipped Espresso packages (found via the
    /// `kANEFModelPreCompiledValue` error path).
    /// `extra`: additional numeric `loadModel` option keys
    /// (`kANEFIntermediateBufferSizeHint`, `kANEFSkipPreparePhaseKey`, ...).
    pub fn load_precompiled(hwx: &Path, extra: &[(&str, u64)]) -> Result<Self, EspError> {
        let pool = unsafe { objc_autoreleasePoolPush() };
        let r = Self::load_inner(hwx, true, extra);
        unsafe { objc_autoreleasePoolPop(pool) };
        r
    }

    /// Parse `modelAttributes` into names, tensor descriptors and the
    /// procedure map.
    fn parse_attrs(
        objc: &Objc,
        model: Id,
    ) -> (Vec<TensorDesc>, Vec<TensorDesc>, Vec<(String, u64)>) {
        unsafe {
            let attrs = msg!((Id) -> Id)(
                model,
                selp("valueForKey:"),
                objc.ns_string("modelAttributes"),
            );
            let mut in_descs = Vec::new();
            let mut out_descs = Vec::new();
            let mut procs = Vec::new();
            if attrs.is_null() {
                return (in_descs, out_descs, procs);
            }
            let parse_list = |key: &str| -> Vec<TensorDesc> {
                let nsl = objc.dict_get(attrs, "NetworkStatusList");
                let first = objc.arr_at(nsl, 0);
                let list = objc.dict_get(first, key);
                let n = objc.arr_count(list);
                (0..n)
                    .map(|i| {
                        let d = objc.arr_at(list, i);
                        let g = |k: &str| objc.num(objc.dict_get(d, k));
                        TensorDesc {
                            name: ns_str(objc.dict_get(d, "Symbol")),
                            ty: ns_str(objc.dict_get(d, "Type")),
                            n: g("Batches"),
                            k: g("Channels"),
                            h: g("Height"),
                            w: g("Width"),
                            plane_stride: g("PlaneStride"),
                            row_stride: g("RowStride"),
                            batch_stride: g("BatchStride"),
                            depth_stride: g("DepthStride"),
                            elem: 2,
                        }
                    })
                    .collect()
            };
            in_descs = parse_list("LiveInputList");
            out_descs = parse_list("LiveOutputList");
            let desc = objc.dict_get(attrs, "ANEFModelDescription");
            let pmap = objc.dict_get(desc, "kANEFModelProcedureNameToIDMapKey");
            for name in objc.dict_keys(pmap) {
                procs.push((name.clone(), objc.num(objc.dict_get(pmap, &name))));
            }
            (in_descs, out_descs, procs)
        }
    }

    fn load_inner(path: &Path, precompiled: bool, extra: &[(&str, u64)]) -> Result<Self, EspError> {
        let objc = Objc::open()?;
        let io = IoApi::open()?;
        unsafe {
            let url = objc.ns_url(path);
            let key = objc.ns_string("probe");
            let model = msg!((Id, Id) -> Id)(objc.ane_model, selp("modelAtURL:key:"), url, key);
            if model.is_null() {
                return Err(EspError::Msg("_ANEModel modelAtURL: returned nil".into()));
            }
            retain(model);
            let client = msg!(() -> Id)(objc.ane_client, selp("sharedConnection"));
            if client.is_null() {
                release(model);
                return Err(EspError::Msg("_ANEClient sharedConnection nil".into()));
            }
            retain(client);
            let opts = if precompiled {
                let mut keys: Vec<Id> = vec![objc.ns_string("kANEFModelType")];
                let mut vals: Vec<Id> = vec![objc.ns_string("kANEFModelPreCompiled")];
                for (k, v) in extra {
                    keys.push(objc.ns_string(k));
                    vals.push(objc.ns_int(*v));
                }
                msg!((*const Id, *const Id, usize) -> Id)(
                    objc.ns_dict,
                    selp("dictionaryWithObjects:forKeys:count:"),
                    vals.as_ptr(),
                    keys.as_ptr(),
                    keys.len(),
                )
            } else {
                objc.empty_dict()
            };
            let mut err: Id = std::ptr::null_mut();
            let f = msg!((Id, Id, u64, *mut Id) -> bool);
            let ok = f(
                client,
                selp("loadModel:options:qos:error:"),
                model,
                opts,
                0,
                &mut err,
            );
            if !ok {
                let msg = err_desc(err);
                release(model);
                release(client);
                return Err(EspError::Msg(format!("loadModel failed: {msg}")));
            }
            let num_inputs = u64v(model, "numInputs");
            let num_outputs = u64v(model, "numOutputs");
            let program_handle = u64v(model, "programHandle");
            let (input_descs, output_descs, procedures) = Self::parse_attrs(&objc, model);

            let mk = |objc: &Objc, bytes: usize| -> Result<Surface, EspError> {
                let f = msg!((usize, usize, usize) -> Id);
                let surf = f(
                    objc.ane_surf,
                    selp("createIOSurfaceWithWidth:pixel_size:height:"),
                    bytes / 2,
                    2,
                    1,
                );
                if surf.is_null() {
                    return Err(EspError::Msg("createIOSurface failed".into()));
                }
                let wrap = msg!((Id) -> Id)(objc.ane_surf, selp("objectWithIOSurface:"), surf);
                if wrap.is_null() {
                    return Err(EspError::Msg("objectWithIOSurface nil".into()));
                }
                retain(wrap);
                Ok(Surface {
                    surf,
                    wrap,
                    len: bytes,
                })
            };
            let mut inputs = Vec::new();
            for i in 0..num_inputs as usize {
                let bytes = input_descs.get(i).map(|d| d.bytes()).unwrap_or(1 << 20);
                inputs.push(mk(&objc, bytes)?);
            }
            let mut outputs = Vec::new();
            for i in 0..num_outputs as usize {
                let bytes = output_descs.get(i).map(|d| d.bytes()).unwrap_or(1 << 20);
                outputs.push(mk(&objc, bytes)?);
            }
            Ok(EspressoProgram {
                objc,
                io,
                model,
                client,
                inputs,
                outputs,
                input_descs,
                output_descs,
                procedures,
                num_inputs,
                num_outputs,
                program_handle,
            })
        }
    }

    fn surf_bytes(&self, s: &Surface) -> (*mut u8, usize) {
        unsafe {
            let base = (self.io.get_base)(s.surf);
            let n = (self.io.alloc_size)(s.surf).min(s.len);
            (base as *mut u8, n)
        }
    }

    /// Raw bytes of input surface `i` (write tensor data in the
    /// program's stride layout before `eval`).
    pub fn input_mut(&mut self, i: usize) -> &mut [u8] {
        let (p, n) = self.surf_bytes(&self.inputs[i]);
        unsafe { std::slice::from_raw_parts_mut(p, n) }
    }

    /// Raw bytes of output surface `i` after `eval`.
    pub fn output(&self, i: usize) -> &[u8] {
        let (p, n) = self.surf_bytes(&self.outputs[i]);
        unsafe { std::slice::from_raw_parts(p, n) }
    }

    /// Surface index of input symbol `name` (`x`, `c`, ...).
    pub fn input_named(&self, name: &str) -> Option<usize> {
        self.input_descs.iter().position(|d| d.name == name)
    }

    /// Surface index of output symbol `name`.
    pub fn output_named(&self, name: &str) -> Option<usize> {
        self.output_descs.iter().position(|d| d.name == name)
    }

    /// Evaluate procedure index `proc` (0 = first procedure; see
    /// [`EspressoProgram::procedures`]).
    pub fn eval_proc(&self, proc: u64) -> Result<(), EspError> {
        let pool = unsafe { objc_autoreleasePoolPush() };
        let r = self.eval_inner(proc);
        unsafe { objc_autoreleasePoolPop(pool) };
        r
    }

    /// Evaluate the first procedure.
    pub fn eval(&self) -> Result<(), EspError> {
        self.eval_proc(0)
    }

    fn eval_inner(&self, proc: u64) -> Result<(), EspError> {
        unsafe {
            let in_objs: Vec<Id> = self.inputs.iter().map(|s| s.wrap).collect();
            let out_objs: Vec<Id> = self.outputs.iter().map(|s| s.wrap).collect();
            let in_idx: Vec<Id> = (0..self.inputs.len() as u64)
                .map(|i| self.objc.ns_int(i))
                .collect();
            let out_idx: Vec<Id> = (0..self.outputs.len() as u64)
                .map(|i| self.objc.ns_int(i))
                .collect();
            let proc_idx = self.objc.ns_int(proc);
            let ins = self.objc.ns_arr(&in_objs);
            let ins_i = self.objc.ns_arr(&in_idx);
            let outs = self.objc.ns_arr(&out_objs);
            let outs_i = self.objc.ns_arr(&out_idx);
            let req = msg!((Id, Id, Id, Id, Id) -> Id)(
                self.objc.ane_request,
                selp("requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:"),
                ins,
                ins_i,
                outs,
                outs_i,
                proc_idx,
            );
            if req.is_null() {
                return Err(EspError::Msg("_ANERequest init failed".into()));
            }
            let valid = msg!(() -> bool)(req, selp("validate"));
            if !valid {
                return Err(EspError::Msg("_ANERequest validate failed".into()));
            }
            let opts = self.objc.empty_dict();
            let mut err: Id = std::ptr::null_mut();
            let ok = msg!((Id, Id, Id, *mut Id) -> bool)(
                self.client,
                selp("evaluateRealTimeWithModel:options:request:error:"),
                self.model,
                opts,
                req,
                &mut err,
            );
            if ok {
                Ok(())
            } else {
                Err(EspError::Msg(format!("evaluate failed: {}", err_desc(err))))
            }
        }
    }
}

/// NSString → String via `UTF8String` ("" for null/non-strings).
fn ns_str(o: Id) -> String {
    if o.is_null() {
        return String::new();
    }
    unsafe {
        let p = msg!(() -> *const c_char)(o, selp("UTF8String"));
        if p.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

fn u64v(o: Id, name: &str) -> u64 {
    unsafe {
        let ns = objc_getClass(cstr("NSString").as_ptr());
        let k = msg!((Id) -> Id)(ns, selp("stringWithUTF8String:"), cstr(name).as_ptr() as Id);
        let v = msg!((Id) -> Id)(o, selp("valueForKey:"), k);
        if v.is_null() {
            0
        } else {
            msg!(() -> u64)(v, selp("unsignedLongLongValue"))
        }
    }
}

fn err_desc(err: Id) -> String {
    if err.is_null() {
        return "unknown".into();
    }
    unsafe {
        let d = msg!(() -> Id)(err, selp("localizedDescription"));
        if d.is_null() {
            return "NSError".into();
        }
        let p = msg!(() -> *const c_char)(d, selp("UTF8String"));
        if p.is_null() {
            "NSError".into()
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

impl Drop for EspressoProgram {
    fn drop(&mut self) {
        for s in self.inputs.iter().chain(self.outputs.iter()) {
            release(s.wrap);
            unsafe { (self.io.cf_release)(s.surf) };
        }
        release(self.model);
        release(self.client);
    }
}

unsafe impl Send for EspressoProgram {}
unsafe impl Sync for EspressoProgram {}
