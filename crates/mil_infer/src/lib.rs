//! `mil_infer` — run compiled CoreML models in-process.
//!
//! The missing half of the toolchain: everything so far proves a spec
//! *compiles*; this proves it *computes*. `mil_infer` loads a `.mlmodelc`
//! through `MLModel`, marshals `MLMultiArray` inputs, calls predict, and
//! hands back raw fp16 output — the same calls Xcode's preview makes,
//! with no Python and no subprocess.
//!
//! Two things this unlocks that `coremltools` can't do for you:
//!
//! 1. **Forced-unit measurement.** [`ComputeUnits::CpuAndNeuralEngine`]
//!    pins the run to the ANE — the latency number `mil_lint`'s
//!    dispatch estimate predicts. `ComputeUnits::CpuAndGpu` excludes it.
//!    Comparing the two *proves* where the graph lands instead of
//!    trusting the performance report.
//! 2. **Numerical conformance.** `mil_verify` can finally check that a
//!    converted model produces the same logits as a reference — the
//!    difference between "it compiled" and "it's correct."
//!
//! macOS only — the rest of the toolchain is portable, this crate is
//! the piece that touches silicon.

use std::fmt;
use std::path::Path;
use std::time::{Duration, Instant};

/// `.milshards` bundle loading and chained prediction.
pub mod shard;
pub use shard::{ShardManifest, ShardedModel};

/// Which hardware units the model is allowed to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeUnits {
    /// CPU + GPU + ANE (scheduler's choice).
    All = 0,
    /// CPU only.
    CpuOnly = 1,
    /// CPU + GPU, no ANE.
    CpuAndGpu = 2,
    /// CPU + ANE, no GPU. The number that matters for the drafter.
    CpuAndNeuralEngine = 3,
}

impl ComputeUnits {
    /// The `MLComputeUnits` raw value Core ML expects
    /// (`MLModelConfiguration.computeUnits`): `cpuOnly = 0`,
    /// `cpuAndGPU = 1`, `all = 2`, `cpuAndNeuralEngine = 3`. This is
    /// deliberately an explicit `match`, not the enum discriminant — the
    /// discriminants here were once passed through verbatim and every
    /// setting but `CpuAndNeuralEngine` selected the wrong hardware.
    pub fn ml_raw_value(self) -> i64 {
        match self {
            ComputeUnits::CpuOnly => 0,
            ComputeUnits::CpuAndGpu => 1,
            ComputeUnits::All => 2,
            ComputeUnits::CpuAndNeuralEngine => 3,
        }
    }
}

/// A compute device Core ML can place an operation on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Device {
    /// `MLCPUComputeDevice`.
    Cpu,
    /// `MLGPUComputeDevice`.
    Gpu,
    /// `MLNeuralEngineComputeDevice`.
    NeuralEngine,
}

impl Device {
    /// Short label (`cpu` / `gpu` / `ane`).
    pub fn label(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Gpu => "gpu",
            Device::NeuralEngine => "ane",
        }
    }
}

/// Where Core ML's compute plan (`MLComputePlan`, macOS 14.4+) places
/// one ML Program operation.
#[derive(Clone, Debug)]
pub struct OpPlacement {
    /// Function the op lives in (`main`).
    pub function: String,
    /// MIL op type (`conv`, `softmax`, ...).
    pub op_type: String,
    /// First output's name (empty if the op has none).
    pub output: String,
    /// The device Core ML would run it on. `None` for ops with no
    /// device usage (compile-time consts).
    pub preferred: Option<Device>,
    /// Every device that supports the op.
    pub supported: Vec<Device>,
    /// `MLComputePlanCost.weight` — the op's share of estimated cost
    /// (weights sum to ~1 over the program). `None` when unavailable.
    pub cost: Option<f64>,
}

/// Per-device totals over a plan: op count and summed cost weight.
pub fn summarize_plan(ops: &[OpPlacement]) -> std::collections::BTreeMap<Device, (usize, f64)> {
    let mut m = std::collections::BTreeMap::new();
    for o in ops {
        if let Some(d) = o.preferred {
            let e = m.entry(d).or_insert((0usize, 0f64));
            e.0 += 1;
            e.1 += o.cost.unwrap_or(0.0);
        }
    }
    m
}

/// Ask Core ML where it would place every operation of a compiled
/// ML Program (`.mlmodelc`) under `units` — `MLComputePlan`, the same
/// data Xcode's performance report shows. Requires macOS 14.4+.
///
/// This is *Core ML's plan*, not a trace of an actual run, but it is the
/// authoritative statement of which device each op is scheduled on.
pub fn compute_plan(compiled: &Path, units: ComputeUnits) -> Result<Vec<OpPlacement>> {
    #[cfg(target_os = "macos")]
    {
        imp::compute_plan(compiled, units)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (compiled, units);
        err("plan", "compute_plan requires macOS")
    }
}

/// An inference error.
#[derive(Debug)]
pub struct InferError {
    /// Stage: `"load"`, `"input"`, `"predict"`, `"output"`.
    pub stage: &'static str,
    /// Human-readable message.
    pub message: String,
}

impl fmt::Display for InferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "infer {}: {}", self.stage, self.message)
    }
}
impl std::error::Error for InferError {}

type Result<T> = std::result::Result<T, InferError>;

fn err<T>(stage: &'static str, msg: impl Into<String>) -> Result<T> {
    Err(InferError {
        stage,
        message: msg.into(),
    })
}

/// One named tensor input.
#[derive(Clone, Debug)]
pub struct Input<'a> {
    /// Feature name.
    pub name: &'a str,
    /// Shape.
    pub shape: &'a [i64],
    /// Data — fp16/int32 LE bytes matching `dtype`.
    pub data: &'a [u8],
    /// Element dtype.
    pub dtype: mil_spec::DType,
}

/// One named output tensor.
#[derive(Clone, Debug)]
pub struct Output {
    /// Feature name.
    pub name: String,
    /// Shape.
    pub shape: Vec<i64>,
    /// `MLMultiArrayDataType` code (fp16 = 65552, fp32 = 65568, int32 = 131104).
    pub dtype_code: i64,
    /// Raw element bytes (little-endian).
    pub data: Vec<u8>,
}

impl Output {
    /// Decode as f32 values (fp16/fp32/int32 outputs).
    pub fn values(&self) -> Vec<f32> {
        match self.dtype_code {
            65552 => self
                .data
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            65568 => self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            131104 => self
                .data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// A prediction result.
#[derive(Clone, Debug)]
pub struct Prediction {
    /// Output tensors, in model-description order.
    pub outputs: Vec<Output>,
    /// Wall time of the predict call.
    pub latency: Duration,
}

// ======================== macOS implementation ========================

#[cfg(target_os = "macos")]
mod imp {
    use super::*;
    use std::ffi::CString;
    use std::os::raw::{c_char, c_void};
    use std::sync::atomic::{AtomicUsize, Ordering};

    type Id = *mut c_void;
    type Sel = *mut c_void;

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
            err("load", format!("objc class {name} missing"))
        } else {
            Ok(v)
        }
    }
    fn sel(name: &str) -> Sel {
        let c = CString::new(name).unwrap();
        unsafe { sel_registerName(c.as_ptr()) }
    }

    unsafe fn msg0(r: Id, s: Sel) -> Id {
        let f: unsafe extern "C" fn(Id, Sel) -> Id =
            std::mem::transmute(objc_msgSend as *const () as usize);
        f(r, s)
    }
    unsafe fn msg1(r: Id, s: Sel, a: Id) -> Id {
        let f: unsafe extern "C" fn(Id, Sel, Id) -> Id =
            std::mem::transmute(objc_msgSend as *const () as usize);
        f(r, s, a)
    }
    unsafe fn msg2(r: Id, s: Sel, a: Id, b: Id) -> Id {
        let f: unsafe extern "C" fn(Id, Sel, Id, Id) -> Id =
            std::mem::transmute(objc_msgSend as *const () as usize);
        f(r, s, a, b)
    }
    unsafe fn msg3(r: Id, s: Sel, a: Id, b: Id, c: Id) -> Id {
        let f: unsafe extern "C" fn(Id, Sel, Id, Id, Id) -> Id =
            std::mem::transmute(objc_msgSend as *const () as usize);
        f(r, s, a, b, c)
    }
    unsafe fn msg1_i64(r: Id, s: Sel, a: i64) -> Id {
        let f: unsafe extern "C" fn(Id, Sel, i64) -> Id =
            std::mem::transmute(objc_msgSend as *const () as usize);
        f(r, s, a)
    }
    unsafe fn msg0_i64(r: Id, s: Sel) -> i64 {
        let f: unsafe extern "C" fn(Id, Sel) -> i64 =
            std::mem::transmute(objc_msgSend as *const () as usize);
        f(r, s)
    }
    unsafe fn msg0_usize(r: Id, s: Sel) -> usize {
        let f: unsafe extern "C" fn(Id, Sel) -> usize =
            std::mem::transmute(objc_msgSend as *const () as usize);
        f(r, s)
    }
    unsafe fn msg1_usize(r: Id, s: Sel, a: usize) -> Id {
        let f: unsafe extern "C" fn(Id, Sel, usize) -> Id =
            std::mem::transmute(objc_msgSend as *const () as usize);
        f(r, s, a)
    }
    unsafe fn msg0_ptr(r: Id, s: Sel) -> *mut u8 {
        let f: unsafe extern "C" fn(Id, Sel) -> *mut u8 =
            std::mem::transmute(objc_msgSend as *const () as usize);
        f(r, s)
    }

    /// Minimal Objective-C block literal — a `_NSConcreteGlobalBlock`
    /// (no captures, static storage, invoked synchronously by
    /// `-[MLState getMultiArrayForStateNamed:handler:]`).
    #[repr(C)]
    struct StateBlock {
        isa: *const c_void,
        flags: i32,
        reserved: i32,
        invoke: unsafe extern "C" fn(*const StateBlock, Id),
        descriptor: *const StateBlockDesc,
    }
    #[repr(C)]
    struct StateBlockDesc {
        reserved: u64,
        size: u64,
    }
    // Both are read-only statics — the pointers point at other statics.
    unsafe impl Sync for StateBlock {}
    unsafe impl Sync for StateBlockDesc {}

    extern "C" {
        /// libSystem's global-block class (always linked).
        static _NSConcreteGlobalBlock: c_void;
    }

    static STATE_BLOCK_DESC: StateBlockDesc = StateBlockDesc {
        reserved: 0,
        size: std::mem::size_of::<StateBlock>() as u64,
    };

    /// `BLOCK_IS_GLOBAL` — see `<Block_private.h>`.
    const BLOCK_IS_GLOBAL: i32 = 1 << 28;

    static ZERO_STATE_BLOCK: StateBlock = StateBlock {
        isa: unsafe { &_NSConcreteGlobalBlock } as *const c_void as *const c_void,
        flags: BLOCK_IS_GLOBAL,
        reserved: 0,
        invoke: zero_multiarray,
        descriptor: &STATE_BLOCK_DESC,
    };

    /// `MLMultiArrayDataType` code → element size in bytes.
    /// Values from `<CoreML/MLMultiArray.h>` (0x10000|bits floats,
    /// 0x20000|bits ints). Unknown codes return `None` — callers must
    /// never guess an element size.
    pub(crate) fn ml_dtype_size(code: i64) -> Option<usize> {
        match code {
            65552 => Some(2),  // Float16 = 0x10000 | 16
            65568 => Some(4),  // Float32/Float = 0x10000 | 32
            65600 => Some(8),  // Double/Float64 = 0x10000 | 64
            131104 => Some(4), // Int32 = 0x20000 | 32
            131080 => Some(1), // Int8 = 0x20000 | 8
            _ => None,
        }
    }

    /// State buffers skipped by `zero_state` due to unrecognised
    /// dtype/layout — recorded so the skip is visible, never guessed.
    static ZERO_STATE_SKIPPED: AtomicUsize = AtomicUsize::new(0);

    /// `array` is an NSArray<NSNumber>; element `i` as i64.
    unsafe fn nsnum_at(arr: Id, i: usize) -> Option<i64> {
        let n = unsafe { msg1_usize(arr, sel("objectAtIndex:"), i) };
        if n.is_null() {
            None
        } else {
            Some(unsafe { msg0_i64(n, sel("integerValue")) })
        }
    }

    /// Block body: memset the handed MLMultiArray's data to zero.
    /// Byte extent = (Σ (shapeᵢ−1)·strideᵢ + 1)·esz — the span the view
    /// actually addresses, which equals `count·esz` only for contiguous
    /// arrays. Unknown dtype, missing shape/strides, or a negative
    /// stride → skip and record; never guess an element size.
    unsafe extern "C" fn zero_multiarray(_blk: *const StateBlock, array: Id) {
        if array.is_null() {
            return;
        }
        let ptr = unsafe { msg0_ptr(array, sel("dataPointer")) };
        if ptr.is_null() {
            return;
        }
        let dt = unsafe { msg0_i64(array, sel("dataType")) };
        let Some(esz) = ml_dtype_size(dt) else {
            ZERO_STATE_SKIPPED.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let shape = unsafe { msg0(array, sel("shape")) };
        let strides = unsafe { msg0(array, sel("strides")) };
        if shape.is_null() || strides.is_null() {
            ZERO_STATE_SKIPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let nd = unsafe { msg0_usize(shape, sel("count")) };
        // span in elements covered by the view, starting at dataPointer
        let mut span: i64 = 1;
        let mut ok = true;
        for i in 0..nd {
            let (Some(d), Some(st)) = (unsafe { nsnum_at(shape, i) }, unsafe {
                nsnum_at(strides, i)
            }) else {
                ok = false;
                break;
            };
            if d < 0 || st < 0 {
                ok = false;
                break;
            }
            span = span.saturating_add(d.saturating_sub(1).saturating_mul(st));
        }
        if !ok || span < 0 {
            ZERO_STATE_SKIPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let bytes = (span as usize).saturating_mul(esz);
        unsafe { std::ptr::write_bytes(ptr, 0, bytes) };
    }

    fn ns_string(s: &str) -> Result<Id> {
        let c = CString::new(s).map_err(|_| InferError {
            stage: "input",
            message: "name has interior NUL".into(),
        })?;
        Ok(unsafe {
            msg1(
                cls("NSString")?,
                sel("stringWithUTF8String:"),
                c.as_ptr() as Id,
            )
        })
    }
    fn utf8(ns: Id) -> Option<String> {
        if ns.is_null() {
            return None;
        }
        let p = unsafe { msg0(ns, sel("UTF8String")) } as *const c_char;
        if p.is_null() {
            None
        } else {
            Some(
                unsafe { std::ffi::CStr::from_ptr(p) }
                    .to_string_lossy()
                    .into_owned(),
            )
        }
    }
    fn file_url(path: &str) -> Result<Id> {
        Ok(unsafe { msg1(cls("NSURL")?, sel("fileURLWithPath:"), ns_string(path)?) })
    }
    fn err_desc(e: Id) -> String {
        if e.is_null() {
            return "unknown NSError".into();
        }
        utf8(unsafe { msg0(e, sel("localizedDescription")) })
            .unwrap_or_else(|| "unknown NSError".into())
    }
    fn ns_number_i64(v: i64) -> Result<Id> {
        Ok(unsafe { msg1_i64(cls("NSNumber")?, sel("numberWithLongLong:"), v) })
    }
    /// NSArray of NSNumbers from i64 dims.
    fn shape_array(dims: &[i64]) -> Result<Id> {
        let mut nums = Vec::with_capacity(dims.len());
        for &d in dims {
            nums.push(ns_number_i64(d)?);
        }
        Ok(unsafe {
            msg2(
                cls("NSArray")?,
                sel("arrayWithObjects:count:"),
                nums.as_ptr() as Id,
                nums.len() as Id,
            )
        })
    }

    fn ensure_coreml() -> Result<()> {
        let p = CString::new("/System/Library/Frameworks/CoreML.framework/CoreML").unwrap();
        let h = unsafe { dlopen(p.as_ptr(), RTLD_NOW) };
        if h.is_null() {
            err("load", "dlopen(CoreML.framework) failed")
        } else {
            Ok(())
        }
    }

    /// `DType` → `MLMultiArrayDataType` (the same codes `DType::array` emits).
    fn ml_dtype(dt: mil_spec::DType) -> i64 {
        match dt {
            mil_spec::DType::Fp16 => 65552,
            mil_spec::DType::Fp32 => 65568,
            mil_spec::DType::Int32 => 131104,
            mil_spec::DType::Int64 => 131136,
            _ => 65552,
        }
    }

    // ---------------- MLComputePlan ----------------

    #[link(name = "objc")]
    extern "C" {
        fn object_getClass(obj: Id) -> Id;
        fn class_getName(cls: Id) -> *const c_char;
    }

    unsafe fn msg0_f64(r: Id, s: Sel) -> f64 {
        let f: unsafe extern "C" fn(Id, Sel) -> f64 =
            std::mem::transmute(objc_msgSend as *const () as usize);
        f(r, s)
    }

    /// `NSString` → Rust `String` (empty for nil).
    fn ns_to_string(ns: Id) -> String {
        utf8(ns).unwrap_or_default()
    }

    fn class_name_of(obj: Id) -> String {
        if obj.is_null() {
            return String::new();
        }
        let c = unsafe { class_getName(object_getClass(obj)) };
        if c.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(c) }
                .to_string_lossy()
                .into_owned()
        }
    }

    fn device_of(obj: Id) -> Option<Device> {
        let n = class_name_of(obj);
        if n.contains("NeuralEngine") {
            Some(Device::NeuralEngine)
        } else if n.contains("GPU") {
            Some(Device::Gpu)
        } else if n.contains("CPU") {
            Some(Device::Cpu)
        } else {
            None
        }
    }

    /// Completion-handler block: `void (^)(MLComputePlan*, NSError*)`.
    /// A global block (no captures); results go through `PLAN_SLOT`,
    /// and `PLAN_CALL` serializes callers so there is one in-flight
    /// request per process.
    #[repr(C)]
    struct PlanBlock {
        isa: *const c_void,
        flags: i32,
        reserved: i32,
        invoke: unsafe extern "C" fn(*const PlanBlock, Id, Id),
        descriptor: *const StateBlockDesc,
    }
    unsafe impl Sync for PlanBlock {}

    struct PlanResult {
        plan: usize,
        error: Option<String>,
    }
    static PLAN_SLOT: std::sync::Mutex<Option<PlanResult>> = std::sync::Mutex::new(None);
    static PLAN_READY: std::sync::Condvar = std::sync::Condvar::new();
    static PLAN_CALL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    static PLAN_BLOCK_DESC: StateBlockDesc = StateBlockDesc {
        reserved: 0,
        size: std::mem::size_of::<PlanBlock>() as u64,
    };

    static PLAN_BLOCK: PlanBlock = PlanBlock {
        isa: unsafe { &_NSConcreteGlobalBlock } as *const c_void as *const c_void,
        flags: BLOCK_IS_GLOBAL,
        reserved: 0,
        invoke: plan_done,
        descriptor: &PLAN_BLOCK_DESC,
    };

    unsafe extern "C" fn plan_done(_blk: *const PlanBlock, plan: Id, error: Id) {
        let pool = objc_autoreleasePoolPush();
        let res = if plan.is_null() {
            PlanResult {
                plan: 0,
                error: Some(err_desc(error)),
            }
        } else {
            // outlive the handler's autorelease pool
            msg0(plan, sel("retain"));
            PlanResult {
                plan: plan as usize,
                error: None,
            }
        };
        objc_autoreleasePoolPop(pool);
        *PLAN_SLOT.lock().unwrap() = Some(res);
        PLAN_READY.notify_all();
    }

    unsafe fn nsarray_each(arr: Id, mut f: impl FnMut(Id)) {
        if arr.is_null() {
            return;
        }
        let n = msg0_usize(arr, sel("count"));
        for i in 0..n {
            f(msg1_usize(arr, sel("objectAtIndex:"), i));
        }
    }

    /// Walk `ops` (an `NSArray<MLModelStructureProgramOperation*>`),
    /// recursing into nested blocks.
    unsafe fn walk_ops(plan: Id, func: &str, ops: Id, out: &mut Vec<OpPlacement>) {
        nsarray_each(ops, |op| {
            let op_type = ns_to_string(msg0(op, sel("operatorName")));
            let outs = msg0(op, sel("outputs"));
            let mut output = String::new();
            if !outs.is_null() && msg0_usize(outs, sel("count")) > 0 {
                let nv = msg1_usize(outs, sel("objectAtIndex:"), 0);
                output = ns_to_string(msg0(nv, sel("name")));
            }
            let usage = msg1(plan, sel("computeDeviceUsageForMLProgramOperation:"), op);
            let (preferred, supported) = if usage.is_null() {
                (None, Vec::new())
            } else {
                let mut sup = Vec::new();
                nsarray_each(msg0(usage, sel("supportedComputeDevices")), |d| {
                    if let Some(dev) = device_of(d) {
                        sup.push(dev);
                    }
                });
                (device_of(msg0(usage, sel("preferredComputeDevice"))), sup)
            };
            let cost_obj = msg1(plan, sel("estimatedCostOfMLProgramOperation:"), op);
            let cost = if cost_obj.is_null() {
                None
            } else {
                Some(msg0_f64(cost_obj, sel("weight")))
            };
            out.push(OpPlacement {
                function: func.to_string(),
                op_type,
                output,
                preferred,
                supported,
                cost,
            });
            nsarray_each(msg0(op, sel("blocks")), |blk| {
                walk_ops(plan, func, msg0(blk, sel("operations")), out);
            });
        });
    }

    pub fn compute_plan(compiled: &Path, units: ComputeUnits) -> Result<Vec<OpPlacement>> {
        let pool = unsafe { objc_autoreleasePoolPush() };
        let r = compute_plan_inner(compiled, units);
        unsafe { objc_autoreleasePoolPop(pool) };
        r
    }

    fn compute_plan_inner(compiled: &Path, units: ComputeUnits) -> Result<Vec<OpPlacement>> {
        ensure_coreml()?;
        let plan_cls = unsafe { objc_getClass(c"MLComputePlan".as_ptr()) };
        if plan_cls.is_null() {
            return err("plan", "MLComputePlan unavailable (needs macOS 14.4+)");
        }
        let url = file_url(compiled.to_str().ok_or_else(|| InferError {
            stage: "plan",
            message: "path not utf8".into(),
        })?)?;
        let cfg = unsafe { msg0(cls("MLModelConfiguration")?, sel("new")) };
        unsafe { msg1_i64(cfg, sel("setComputeUnits:"), units.ml_raw_value()) };

        let _one_at_a_time = PLAN_CALL.lock().unwrap();
        *PLAN_SLOT.lock().unwrap() = None;
        unsafe {
            msg3(
                plan_cls,
                sel("loadContentsOfURL:configuration:completionHandler:"),
                url,
                cfg,
                &PLAN_BLOCK as *const PlanBlock as Id,
            );
        }
        let res = {
            let mut slot = PLAN_SLOT.lock().unwrap();
            let deadline = Instant::now() + Duration::from_secs(300);
            loop {
                if let Some(r) = slot.take() {
                    break r;
                }
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return err("plan", "timed out waiting for MLComputePlan");
                }
                slot = PLAN_READY.wait_timeout(slot, left).unwrap().0;
            }
        };
        if let Some(e) = res.error {
            return err("plan", e);
        }
        let plan = res.plan as Id;
        let mut out = Vec::new();
        unsafe {
            let st = msg0(plan, sel("modelStructure"));
            let prog = msg0(st, sel("program"));
            if prog.is_null() {
                msg0(plan, sel("release"));
                return err("plan", "not an ML Program (no program structure)");
            }
            let funcs = msg0(prog, sel("functions"));
            let keys = msg0(funcs, sel("allKeys"));
            nsarray_each(keys, |k| {
                let f = msg1(funcs, sel("objectForKey:"), k);
                let block = msg0(f, sel("block"));
                walk_ops(
                    plan,
                    &ns_to_string(k),
                    msg0(block, sel("operations")),
                    &mut out,
                );
            });
            msg0(plan, sel("release"));
        }
        Ok(out)
    }
    /// A loaded `.mlmodelc`.
    pub struct Model {
        model: Id,
    }

    /// An allocated MLState for KV models.
    pub struct State {
        state: Id,
    }

    impl Model {
        /// Load a compiled `.mlmodelc` directory.
        pub fn load(path: &Path, units: ComputeUnits) -> Result<Model> {
            let pool = unsafe { objc_autoreleasePoolPush() };
            let r = Self::load_inner(path, units);
            unsafe { objc_autoreleasePoolPop(pool) };
            r
        }
        fn load_inner(path: &Path, units: ComputeUnits) -> Result<Model> {
            ensure_coreml()?;
            let url = file_url(
                path.to_str()
                    .ok_or("path not utf8")
                    .map_err(|m| InferError {
                        stage: "load",
                        message: m.into(),
                    })?,
            )?;

            // configuration with the requested compute units
            let cfg = unsafe { msg0(cls("MLModelConfiguration")?, sel("new")) };
            unsafe {
                msg1_i64(cfg, sel("setComputeUnits:"), units.ml_raw_value());
            }

            let mut e: Id = std::ptr::null_mut();
            let ep = &mut e as *mut Id as Id;
            let m = unsafe {
                msg3(
                    cls("MLModel")?,
                    sel("modelWithContentsOfURL:configuration:error:"),
                    url,
                    cfg,
                    ep,
                )
            };
            if m.is_null() {
                return err("load", err_desc(e));
            }
            // retain — the pool that wrapped us is about to pop
            unsafe { msg0(m, sel("retain")) };
            Ok(Model { model: m })
        }

        /// Fresh `MLState` — one per sequence if you don't want cross-call
        /// KV bleed. Buffers are zeroed: CoreML does not guarantee
        /// initialized state, and uninitialized garbage (observed to
        /// contain Inf) poisons masked attention — `prob(0) * Inf = NaN`
        /// leaks through the v-matmul into live positions.
        pub fn new_state(&self) -> Result<State> {
            let responds: bool = unsafe {
                let f: unsafe extern "C" fn(Id, Sel, Sel) -> bool =
                    std::mem::transmute(objc_msgSend as *const () as usize);
                f(self.model, sel("respondsToSelector:"), sel("newState"))
            };
            if !responds {
                return err(
                    "predict",
                    "MLModel has no newState — model has no state features or OS too old",
                );
            }
            let s = unsafe { msg0(self.model, sel("newState")) };
            if s.is_null() {
                return err("predict", "newState returned nil");
            }
            unsafe { msg0(s, sel("retain")) };
            self.zero_state(s);
            Ok(State { state: s })
        }

        /// memset every declared state buffer to zero.
        fn zero_state(&self, state: Id) {
            let before = ZERO_STATE_SKIPPED.load(Ordering::Relaxed);
            let desc = unsafe { msg0(self.model, sel("modelDescription")) };
            if desc.is_null() {
                return;
            }
            let dict = unsafe { msg0(desc, sel("stateDescriptionsByName")) };
            if dict.is_null() {
                return;
            }
            let keys = unsafe { msg0(dict, sel("allKeys")) };
            if keys.is_null() {
                return;
            }
            let n = unsafe { msg0_usize(keys, sel("count")) };
            for i in 0..n {
                let name = unsafe { msg1_usize(keys, sel("objectAtIndex:"), i) };
                if name.is_null() {
                    continue;
                }
                unsafe {
                    msg2(
                        state,
                        sel("getMultiArrayForStateNamed:handler:"),
                        name,
                        &ZERO_STATE_BLOCK as *const StateBlock as Id,
                    )
                };
            }
            let skipped = ZERO_STATE_SKIPPED.load(Ordering::Relaxed) - before;
            if skipped > 0 {
                eprintln!(
                    "mil_infer: {skipped} state buffer(s) left unzeroed — unrecognised dtype/layout"
                );
            }
        }

        /// Build one MLMultiArray from an Input.
        fn make_array(input: &Input) -> Result<Id> {
            let shape = shape_array(input.shape)?;
            let mut e: Id = std::ptr::null_mut();
            let ep = &mut e as *mut Id as Id;
            let allocd = unsafe { msg0(cls("MLMultiArray")?, sel("alloc")) };
            let a = unsafe {
                msg3(
                    allocd,
                    sel("initWithShape:dataType:error:"),
                    shape,
                    ml_dtype(input.dtype) as Id,
                    ep,
                )
            };
            if a.is_null() {
                return err("input", err_desc(e));
            }
            // copy bytes into dataPointer
            let ptr = unsafe { msg0_ptr(a, sel("dataPointer")) };
            if ptr.is_null() {
                return err("input", "dataPointer null");
            }
            unsafe { std::ptr::copy_nonoverlapping(input.data.as_ptr(), ptr, input.data.len()) };
            Ok(a)
        }

        /// Wrap inputs in an MLDictionaryFeatureProvider.
        fn make_provider(inputs: &[Input]) -> Result<Id> {
            let mut keys = Vec::with_capacity(inputs.len());
            let mut vals = Vec::with_capacity(inputs.len());
            for inp in inputs {
                let arr = Self::make_array(inp)?;
                let fv = unsafe {
                    msg1(
                        cls("MLFeatureValue")?,
                        sel("featureValueWithMultiArray:"),
                        arr,
                    )
                };
                if fv.is_null() {
                    return err("input", format!("featureValue nil for {}", inp.name));
                }
                keys.push(ns_string(inp.name)?);
                vals.push(fv);
            }
            let dict = unsafe {
                msg3(
                    cls("NSDictionary")?,
                    sel("dictionaryWithObjects:forKeys:count:"),
                    vals.as_ptr() as Id,
                    keys.as_ptr() as Id,
                    vals.len() as Id,
                )
            };
            if dict.is_null() {
                return err("input", "could not build feature dict");
            }
            let allocd = unsafe { msg0(cls("MLDictionaryFeatureProvider")?, sel("alloc")) };
            let mut e: Id = std::ptr::null_mut();
            let ep = &mut e as *mut Id as Id;
            let prov = unsafe { msg2(allocd, sel("initWithDictionary:error:"), dict, ep) };
            if prov.is_null() {
                err("input", err_desc(e))
            } else {
                Ok(prov)
            }
        }

        /// Predict on the model's default state.
        pub fn predict(&self, inputs: &[Input]) -> Result<Prediction> {
            self.predict_with_state(None, inputs)
        }

        /// Predict with an explicit state (for KV models — state persists
        /// across calls on the same `State`).
        pub fn predict_with_state(
            &self,
            state: Option<&State>,
            inputs: &[Input],
        ) -> Result<Prediction> {
            let pool = unsafe { objc_autoreleasePoolPush() };
            let r = self.predict_inner(state, inputs);
            unsafe { objc_autoreleasePoolPop(pool) };
            r
        }

        fn predict_inner(&self, state: Option<&State>, inputs: &[Input]) -> Result<Prediction> {
            let prov = Self::make_provider(inputs)?;
            let mut e: Id = std::ptr::null_mut();
            let ep = &mut e as *mut Id as Id;
            let t0 = Instant::now();
            let out_prov = match state {
                Some(s) => {
                    let responds: bool = unsafe {
                        let f: unsafe extern "C" fn(Id, Sel, Sel) -> bool =
                            std::mem::transmute(objc_msgSend as *const () as usize);
                        f(
                            self.model,
                            sel("respondsToSelector:"),
                            sel("predictionFromFeatures:usingState:error:"),
                        )
                    };
                    if !responds {
                        return err(
                            "predict",
                            "predictionFromFeatures:usingState:error: missing — needs macOS 15+",
                        );
                    }
                    unsafe {
                        msg3(
                            self.model,
                            sel("predictionFromFeatures:usingState:error:"),
                            prov,
                            s.state,
                            ep,
                        )
                    }
                }
                None => unsafe { msg2(self.model, sel("predictionFromFeatures:error:"), prov, ep) },
            };
            let latency = t0.elapsed();
            if out_prov.is_null() {
                return err("predict", err_desc(e));
            }

            // output names — featureNames is an NSSet; allObjects → NSArray
            let names_set = unsafe { msg0(out_prov, sel("featureNames")) };
            let names_ns = unsafe { msg0(names_set, sel("allObjects")) };
            let count = unsafe { msg0_usize(names_ns, sel("count")) };
            let mut outputs = Vec::with_capacity(count);
            for i in 0..count {
                let name_ns = unsafe { msg1_i64(names_ns, sel("objectAtIndex:"), i as i64) };
                let name = utf8(name_ns).unwrap_or_default();
                let fv = unsafe { msg1(out_prov, sel("featureValueForName:"), name_ns) };
                if fv.is_null() {
                    continue;
                }
                let ma = unsafe { msg0(fv, sel("multiArrayValue")) };
                if ma.is_null() {
                    continue;
                }
                let n_el = unsafe { msg0_i64(ma, sel("count")) } as usize;
                // shape
                let shp = unsafe { msg0(ma, sel("shape")) };
                let nd = unsafe { msg0_usize(shp, sel("count")) };
                let mut shape = Vec::with_capacity(nd);
                for d in 0..nd {
                    let num = unsafe { msg1_i64(shp, sel("objectAtIndex:"), d as i64) };
                    shape.push(unsafe { msg0_i64(num, sel("longLongValue")) });
                }
                // strides — GPU-produced arrays are NOT guaranteed
                // contiguous (padded rows have been observed on the
                // CpuAndGpu path); honor the declared element strides.
                let strd = unsafe { msg0(ma, sel("strides")) };
                let mut strides = Vec::with_capacity(nd);
                for d in 0..nd {
                    let num = unsafe { msg1_i64(strd, sel("objectAtIndex:"), d as i64) };
                    strides.push(unsafe { msg0_i64(num, sel("longLongValue")) });
                }
                // element size from the array's own dataType — same
                // exact-code table as zero_multiarray; an unknown dtype
                // is skipped, never guessed (a wrong size over-reads).
                let dt = unsafe { msg0_i64(ma, sel("dataType")) };
                let Some(el) = ml_dtype_size(dt) else {
                    continue;
                };
                let ptr = unsafe { msg0_ptr(ma, sel("dataPointer")) } as *const u8;
                let bytes = n_el * el;
                let mut data = vec![0u8; bytes];
                // Map row-major flat index → strided source offset.
                for flat in 0..n_el {
                    let mut rem = flat as i64;
                    let mut src: isize = 0;
                    for d in (0..nd).rev() {
                        let dim = shape[d].max(1);
                        let idx = rem % dim;
                        rem /= dim;
                        src += (idx * strides[d]) as isize;
                    }
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            ptr.offset(src * el as isize),
                            data.as_mut_ptr().add(flat * el),
                            el,
                        )
                    };
                }
                outputs.push(Output {
                    name,
                    shape,
                    dtype_code: dt,
                    data,
                });
            }
            Ok(Prediction { outputs, latency })
        }
    }

    impl Drop for Model {
        fn drop(&mut self) {
            unsafe { msg0(self.model, sel("release")) };
        }
    }
    impl Drop for State {
        fn drop(&mut self) {
            unsafe { msg0(self.state, sel("release")) };
        }
    }
}

/// A loaded compiled model — macOS.
#[cfg(target_os = "macos")]
pub use imp::Model;
#[cfg(target_os = "macos")]
pub use imp::State;

/// Non-macOS stub — `load` reports the platform constraint.
#[cfg(not(target_os = "macos"))]
pub struct Model;
#[cfg(not(target_os = "macos"))]
impl Model {
    /// Always fails off-macOS.
    pub fn load(_path: &Path, _units: ComputeUnits) -> Result<Model> {
        err("load", "mil_infer requires macOS")
    }
    /// Always fails off-macOS.
    pub fn predict(&self, _inputs: &[Input]) -> Result<Prediction> {
        err("predict", "mil_infer requires macOS")
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use half::f16;
    use mil_spec::{Block, TensorType, ValueType};

    /// `MLComputeUnits` in `<CoreML/MLModelConfiguration.h>`:
    /// cpuOnly = 0, cpuAndGPU = 1, all = 2, cpuAndNeuralEngine = 3.
    #[test]
    fn compute_units_map_to_ml_compute_units() {
        assert_eq!(ComputeUnits::CpuOnly.ml_raw_value(), 0);
        assert_eq!(ComputeUnits::CpuAndGpu.ml_raw_value(), 1);
        assert_eq!(ComputeUnits::All.ml_raw_value(), 2);
        assert_eq!(ComputeUnits::CpuAndNeuralEngine.ml_raw_value(), 3);
    }

    /// `ml_dtype_size` covers every `MLMultiArrayDataType` in the SDK
    /// enum and returns `None` for anything else — zeroing must never
    /// guess an element size (a wrong guess overruns the buffer).
    #[test]
    fn dtype_size_map_matches_sdk() {
        assert_eq!(imp::ml_dtype_size(65552), Some(2)); // Float16
        assert_eq!(imp::ml_dtype_size(65568), Some(4)); // Float32/Float
        assert_eq!(imp::ml_dtype_size(65600), Some(8)); // Double/Float64
        assert_eq!(imp::ml_dtype_size(131104), Some(4)); // Int32
        assert_eq!(imp::ml_dtype_size(131080), Some(1)); // Int8
                                                         // codes the previous table got wrong / made up
        assert_eq!(imp::ml_dtype_size(65584), None);
        assert_eq!(imp::ml_dtype_size(131072), None);
        assert_eq!(imp::ml_dtype_size(65792), None);
        assert_eq!(imp::ml_dtype_size(131136), None); // no Int64 in the enum
        assert_eq!(imp::ml_dtype_size(0), None);
    }

    /// End-to-end: build y = x + x in mil_spec, package, compile through
    /// mil_compile (coremlc or the FFI path), load via FFI, predict —
    /// output must equal 2x. This is the closed loop: the toolchain's
    /// own artifact executed on Apple's runtime and checked numerically.
    #[test]
    fn predict_add_model() {
        if std::process::Command::new("xcrun")
            .args(["-f", "coremlc"])
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            return; // no coremlc on this machine
        }
        let dir = std::env::temp_dir().join("mil_infer_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // y = x + x on a [1,4,1,1] fp16 tensor.
        let mut b = Block::new();
        let y = b.add("x", "x", &[1, 4, 1, 1], "y");
        b.outputs = vec![y];
        let inputs = [mil_spec::Feature {
            name: "x".into(),
            shape: vec![1, 4, 1, 1],
            dtype: mil_spec::DType::Fp16,
            is_state: false,
        }];
        let outputs = [mil_spec::Feature {
            name: "y".into(),
            shape: vec![1, 4, 1, 1],
            dtype: mil_spec::DType::Fp16,
            is_state: false,
        }];
        let fn_inputs = [mil_spec::NVT {
            name: "x".into(),
            ty: ValueType::Tensor(TensorType::f16(&[1, 4, 1, 1])),
        }];
        let spec = mil_spec::encode_model(
            &inputs,
            &outputs,
            &[],
            &b,
            &fn_inputs,
            &mil_spec::ModelMeta::new(10, "CoreML9"),
        );
        let pkg = dir.join("add.mlpackage");
        mil_spec::write_mlpackage(&pkg, &spec, None).unwrap();

        let comp = dir.join("compiled");
        let compiled = mil_compile::compile(&pkg, &comp).unwrap();

        let model = Model::load(&compiled.path, ComputeUnits::All).unwrap();
        let xs: Vec<f16> = [1.0, 2.0, -3.5, 0.25]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let mut xb = Vec::new();
        for v in &xs {
            xb.extend_from_slice(&v.to_le_bytes());
        }
        let p = model
            .predict(&[Input {
                name: "x",
                shape: &[1, 4, 1, 1],
                data: &xb,
                dtype: mil_spec::DType::Fp16,
            }])
            .unwrap();
        assert_eq!(p.outputs.len(), 1);
        let y = &p.outputs[0];
        assert_eq!(y.shape, vec![1, 4, 1, 1]);
        let v = y.values();
        assert_eq!(v.len(), 4);
        assert!((v[0] - 2.0).abs() < 1e-3);
        assert!((v[1] - 4.0).abs() < 1e-3);
        assert!((v[2] + 7.0).abs() < 1e-3);
        assert!((v[3] - 0.5).abs() < 1e-3);
    }

    /// `Block::rms_norm` must survive fp16 on the real runtime: large
    /// activations (xs² would overflow fp16 unscaled), an all-zero row
    /// (must be exactly 0, matching HF), and an embedding-scale row
    /// (~0.01 RMS — the fixed `1/sqrt(d)` prescale pushed x² into fp16
    /// subnormals and produced inf/NaN on Qwen-class embeddings).
    #[test]
    fn rms_norm_fp16_edge_rows() {
        if std::process::Command::new("xcrun")
            .args(["-f", "coremlc"])
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            return; // no coremlc on this machine
        }
        let dir = std::env::temp_dir().join("mil_infer_rmsnorm_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let d: i64 = 8;
        let n: i64 = 3;
        let eps = 1e-6f32;
        let wvals = [1.0f32, 0.5, 2.0, 1.5, 1.0, 0.25, 1.0, 0.75];

        let mut b = Block::new();
        let w = b.op(
            "const",
            vec![],
            vec![("w", ValueType::Tensor(TensorType::f16(&[d, 1, 1])))],
            vec![("val".into(), mil_spec::Value::f16s(&[d, 1, 1], &wvals))],
        )[0]
        .clone();
        let y = b.rms_norm("x", &w, d, eps, &[n, d, 1, 1], "n");
        b.outputs = vec![y];

        let inputs = [mil_spec::Feature {
            name: "x".into(),
            shape: vec![n, d, 1, 1],
            dtype: mil_spec::DType::Fp16,
            is_state: false,
        }];
        let outputs = [mil_spec::Feature {
            name: "n_out".into(),
            shape: vec![n, d, 1, 1],
            dtype: mil_spec::DType::Fp16,
            is_state: false,
        }];
        let fn_inputs = [mil_spec::NVT {
            name: "x".into(),
            ty: ValueType::Tensor(TensorType::f16(&[n, d, 1, 1])),
        }];
        let spec = mil_spec::encode_model(
            &inputs,
            &outputs,
            &[],
            &b,
            &fn_inputs,
            &mil_spec::ModelMeta::new(10, "CoreML9"),
        );
        let pkg = dir.join("n.mlpackage");
        mil_spec::write_mlpackage(&pkg, &spec, None).unwrap();
        let compiled = mil_compile::compile(&pkg, &dir.join("n.compiled")).unwrap();

        // row 0: large (x/√8)² ≈ 5e5 overflows fp16 under a fixed
        // prescale. row 1: zeros → exactly 0. row 2: ~0.01 RMS —
        // (x/√8)² ≈ 1e-5, fp16 subnormal territory.
        let rows = [
            [
                2000.0f32, -2000.0, 2000.0, -2000.0, 1000.0, -1000.0, 500.0, -500.0,
            ],
            [0.0; 8],
            [0.012, -0.012, 0.008, -0.008, 0.004, -0.004, 0.002, -0.002],
        ];
        let mut xb = Vec::new();
        for row in &rows {
            for &v in row {
                xb.extend_from_slice(&f16::from_f32(v).to_le_bytes());
            }
        }
        let model = Model::load(&compiled.path, ComputeUnits::All).unwrap();
        let p = model
            .predict(&[Input {
                name: "x",
                shape: &[n, d, 1, 1],
                data: &xb,
                dtype: mil_spec::DType::Fp16,
            }])
            .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let out = p.outputs[0].values();
        for (r, row) in rows.iter().enumerate() {
            let ms: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / d as f64;
            let inv = 1.0f64 / (ms + eps as f64).sqrt();
            for (c, &v) in row.iter().enumerate() {
                let want = (v as f64 * inv * wvals[c] as f64) as f32;
                let got = out[r * d as usize + c];
                if r == 1 {
                    assert_eq!(got, 0.0, "all-zero row must produce 0 (c={c})");
                } else {
                    assert!(got.is_finite(), "row {r} c {c}: {got} (want {want})");
                    assert!(
                        (got - want).abs() <= 0.02 + 0.03 * want.abs(),
                        "row {r} c {c}: got {got}, want {want} (fp16 tol)"
                    );
                }
            }
        }
    }
}
