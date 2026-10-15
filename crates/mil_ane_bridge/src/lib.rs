// The dylib filename must stay `libBadAppleANEBridge` for the loader.
#![allow(non_snake_case)]

//! `libBadAppleANEBridge` — the `bad_apple_ane_*` ABI Bad Apple's
//! `ane_core.rs` resolves via dlopen, implemented on mil_infer's direct
//! e5rt path. Load order puts `BADAPPLE_ANE_BRIDGE` first, so pointing it
//! at this dylib swaps the whole ANE brain onto the sovereign path with
//! the CoreML bridge kept as untouched fallback.

use std::ffi::{c_char, c_void, CStr};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use mil_infer::ane_decode::Decoder;
use mil_infer::e5rt::{DeviceMask, Program};

/// Opaque handle — the Box owns the decoder plus load/prewarm timings
/// `bad_apple_ane_selection_latency_us` reports.
struct Handle {
    decoder: Decoder,
    load_us: u64,
    prewarm_us: Mutex<u64>,
}

fn handle(h: *mut c_void) -> Option<&'static Handle> {
    if h.is_null() {
        None
    } else {
        Some(unsafe { &*(h as *const Handle) })
    }
}

/// Mutable access for decoder calls. Callers serialize through the
/// engine's lock (same contract as the Swift bridge's Unmanaged handle),
/// so at most one `&mut` is live at a time.
fn mut_handle(h: *mut c_void) -> Option<&'static mut Handle> {
    if h.is_null() {
        None
    } else {
        Some(unsafe { &mut *(h as *mut Handle) })
    }
}

fn dir_of(path: *const c_char) -> Option<PathBuf> {
    if path.is_null() {
        return None;
    }
    let s = unsafe { CStr::from_ptr(path) }.to_str().ok()?;
    // Manifest path or bare directory — both resolve to the shard dir.
    let p = PathBuf::from(s);
    if p.is_dir() {
        Some(p)
    } else {
        p.parent().map(|d| d.to_path_buf())
    }
}

/// `tokens` is a borrowed slice — feed all but the last without
/// projecting, project the last, return its argmax (-1 on failure).
#[no_mangle]
pub extern "C" fn bad_apple_ane_predict_next(
    h: *mut c_void,
    tokens: *const i32,
    count: usize,
) -> i32 {
    let Some(h) = mut_handle(h) else { return -1 };
    if tokens.is_null() || count == 0 {
        return -1;
    }
    let toks = unsafe { std::slice::from_raw_parts(tokens, count) };
    let dec = &mut h.decoder;
    for (i, &t) in toks.iter().enumerate() {
        if t < 0 {
            return -1;
        }
        match dec.push(t as u32, i + 1 == count) {
            Ok(Some(tok)) => return tok as i32,
            Ok(None) => {}
            Err(_) => return -1,
        }
    }
    -1
}

#[no_mangle]
pub extern "C" fn bad_apple_ane_prefill(h: *mut c_void, tokens: *const i32, count: i32) -> bool {
    let Some(h) = mut_handle(h) else { return false };
    if tokens.is_null() || count <= 0 {
        return false;
    }
    let toks = unsafe { std::slice::from_raw_parts(tokens, count as usize) };
    let dec = &mut h.decoder;
    for &t in toks {
        if t < 0 || dec.push(t as u32, false).is_err() {
            return false;
        }
    }
    true
}

#[no_mangle]
pub extern "C" fn bad_apple_ane_create(path: *const c_char) -> *mut c_void {
    let Some(dir) = dir_of(path) else {
        return std::ptr::null_mut();
    };
    let t = Instant::now();
    match Decoder::open_dir(&dir, DeviceMask::Ane) {
        Ok(decoder) => {
            let load_us = t.elapsed().as_micros() as u64;
            Box::into_raw(Box::new(Handle {
                decoder,
                load_us,
                prewarm_us: Mutex::new(0),
            })) as *mut c_void
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn bad_apple_ane_destroy(h: *mut c_void) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h as *mut Handle) });
    }
}

#[no_mangle]
pub extern "C" fn bad_apple_ane_reset(h: *mut c_void) -> bool {
    let Some(h) = mut_handle(h) else { return false };
    h.decoder.reset().is_ok()
}

/// One throwaway decode step — warms the stream, states reset after.
#[no_mangle]
pub extern "C" fn bad_apple_ane_prewarm(h: *mut c_void) -> bool {
    let Some(this) = mut_handle(h) else {
        return false;
    };
    let t = Instant::now();
    let ok = this.decoder.push(0, false).is_ok() && this.decoder.reset().is_ok();
    if let Ok(mut u) = this.prewarm_us.lock() {
        *u = t.elapsed().as_micros() as u64;
    }
    ok
}

/// All of it runs on the engine — the ratio callers use for reporting.
#[no_mangle]
pub extern "C" fn bad_apple_ane_placement_ratio(_h: *mut c_void) -> f64 {
    if _h.is_null() {
        -1.0
    } else {
        1.0
    }
}

/// MLComputeUnits.all == 2.
#[no_mangle]
pub extern "C" fn bad_apple_ane_compute_units_raw_value(h: *mut c_void) -> i32 {
    if h.is_null() {
        -1
    } else {
        2
    }
}

#[no_mangle]
pub extern "C" fn bad_apple_ane_selection_latency_us(
    h: *mut c_void,
    load_us: *mut u64,
    prewarm_us: *mut u64,
) -> bool {
    let Some(h) = handle(h) else { return false };
    if !load_us.is_null() {
        unsafe { *load_us = h.load_us };
    }
    if !prewarm_us.is_null() {
        *unsafe { &mut *prewarm_us } = *h.prewarm_us.lock().unwrap_or_else(|e| e.into_inner());
    }
    true
}

/// Artifact-level placement probe — everything is ANE here.
#[no_mangle]
pub extern "C" fn bad_apple_coreml_placement_ratio(_path: *const c_char, _units: i32) -> f64 {
    1.0
}

/// Compile one shard MIL through e5rt and time `iterations` steps —
/// same contract as the bridge's shard probe.
#[no_mangle]
pub extern "C" fn bad_apple_ane_probe_shard(
    path: *const c_char,
    _units: i32,
    iterations: usize,
    avg_us: *mut u64,
) -> bool {
    if path.is_null() || iterations == 0 {
        return false;
    }
    let s = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };
    let p = PathBuf::from(s);
    let mil = if p.is_dir() { p.join("model.mil") } else { p };
    let Ok((prog, map)) = Program::compile_auto(&mil, DeviceMask::Ane) else {
        return false;
    };
    for (n, e) in &map.inputs {
        let _ = prog.set_input_f16(n, &vec![0x3c00u16; *e as usize]);
    }
    for (n, e) in &map.inouts {
        let _ = prog.set_input_f16(n, &vec![0u16; *e as usize]);
    }
    let t = Instant::now();
    for _ in 0..iterations {
        if prog.execute().is_err() {
            return false;
        }
    }
    if !avg_us.is_null() {
        unsafe { *avg_us = (t.elapsed().as_micros() as u64) / iterations as u64 };
    }
    true
}

/// Mesh seam: `bytes` = [n, hidden] fp16 rows; each advances the
/// frontier, the last is projected. -1 on failure.
#[no_mangle]
pub extern "C" fn bad_apple_ane_forward_hidden(
    h: *mut c_void,
    bytes: *const c_void,
    count: usize,
) -> i32 {
    let Some(h) = mut_handle(h) else { return -1 };
    if bytes.is_null() || count == 0 {
        return -1;
    }
    let dec = &mut h.decoder;
    let row = dec.hidden_bytes();
    if row == 0 || count % row != 0 {
        return -1;
    }
    let rows = count / row;
    let data = unsafe { std::slice::from_raw_parts(bytes as *const u8, count) };
    for i in 0..rows {
        match dec.push_hidden(&data[i * row..(i + 1) * row], i + 1 == rows) {
            Ok(Some(tok)) => return tok as i32,
            Ok(None) => {}
            Err(_) => return -1,
        }
    }
    -1
}

/// Per-row argmax variant for speculative verification.
#[no_mangle]
pub extern "C" fn bad_apple_ane_forward_hidden_multi(
    h: *mut c_void,
    bytes: *const c_void,
    count: usize,
    out_tokens: *mut i32,
    out_capacity: usize,
) -> i32 {
    let Some(h) = mut_handle(h) else { return -1 };
    if bytes.is_null() || count == 0 || out_tokens.is_null() {
        return -1;
    }
    let dec = &mut h.decoder;
    let row = dec.hidden_bytes();
    if row == 0 || count % row != 0 {
        return -1;
    }
    let rows = count / row;
    let data = unsafe { std::slice::from_raw_parts(bytes as *const u8, count) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_tokens, out_capacity) };
    for i in 0..rows.min(out_capacity) {
        match dec.push_hidden(&data[i * row..(i + 1) * row], true) {
            Ok(Some(tok)) => out[i] = tok as i32,
            _ => return -1,
        }
    }
    rows.min(out_capacity) as i32
}

#[no_mangle]
pub extern "C" fn bad_apple_ane_rewind(h: *mut c_void, position: i32) -> bool {
    let Some(h) = mut_handle(h) else { return false };
    if position < 0 {
        return false;
    }
    h.decoder.rewind(position as usize).is_ok()
}

#[no_mangle]
pub extern "C" fn bad_apple_ane_position(h: *mut c_void) -> i32 {
    match handle(h) {
        Some(h) => h.decoder.position() as i32,
        None => -1,
    }
}
