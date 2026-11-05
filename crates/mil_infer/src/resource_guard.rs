//! Emit-time resource guard.
//!
//! Per-node ANE compilation writes one `.macho` per graph node — a
//! d=512/L=4 brain is ~135k programs (~9 GiB). Unguarded, a runaway
//! compile can fill the disk, exhaust the system open-file table, or
//! spike memory pressure until the host organism starves. This guard
//! samples the output volume, the kernel file table, and memory
//! pressure at a bounded interval and fails the emit with a clear
//! error *before* resources are exhausted — turning a silent system
//! failure into a loud, attributable one.
//!
//! The vnode table is deliberately not sampled: on 16 GiB hosts
//! `kern.num_vnodes` sits at `kern.maxvnodes` as steady state — the
//! kernel recycles it continuously — so it carries no exhaustion
//! signal. `kern.num_files`/`kern.maxfiles` is the table whose
//! exhaustion produced the ENFILE that killed the identity agent.
//!
//! Env knobs (all optional):
//!   MIL_EMIT_GUARD=0               disable the guard entirely
//!   MIL_EMIT_GUARD_MS=<ms>         check interval (default 5000)
//!   MIL_EMIT_MIN_FREE_BYTES=<n>    disk floor (default 2 GiB)
//!   MIL_EMIT_MAX_FILE_PCT=<n>      open-file table ceiling % (default 90)
//!   MIL_EMIT_MIN_MEM_FREE_PCT=<n>  memory free-% floor (default 10)

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static LAST_CHECK_MS: AtomicU64 = AtomicU64::new(0);

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Bytes available to unprivileged writers on the volume holding `path`.
fn avail_bytes(path: &str) -> Option<u64> {
    unsafe {
        let c = std::ffi::CString::new(path).ok()?;
        let mut st: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c.as_ptr(), &mut st) != 0 {
            return None;
        }
        Some(st.f_bavail as u64 * st.f_frsize as u64)
    }
}

/// `kern.num_files` / `kern.maxfiles` — tolerates 4- or 8-byte ints.
#[cfg(target_os = "macos")]
fn sysctl_uint(name: &str) -> Option<u64> {
    unsafe {
        let c = std::ffi::CString::new(name).ok()?;
        let mut buf = [0u8; 16];
        let mut len = buf.len() as libc::size_t;
        if libc::sysctlbyname(
            c.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
        match len {
            4 => Some(u32::from_ne_bytes(buf[..4].try_into().ok()?) as u64),
            8 => Some(u64::from_ne_bytes(buf[..8].try_into().ok()?)),
            _ => None,
        }
    }
}

/// System-wide memory free percentage, from the same `memory_pressure`
/// source the organism's own dashboard reads.
#[cfg(target_os = "macos")]
fn mem_free_pct() -> Option<f64> {
    let out = std::process::Command::new("/usr/bin/memory_pressure")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .find(|l| l.contains("free percentage"))
        .and_then(|l| {
            l.split_whitespace()
                .find_map(|t| t.trim_end_matches('%').parse::<f64>().ok())
        })
}

/// Sample disk, the open-file table, and memory pressure at a bounded
/// interval; fail
/// the emit when any resource is near exhaustion. The first call after
/// each interval performs the checks — a failure aborts the compile
/// immediately via `?`, so no sticky state is needed.
pub fn check(out_dir: &str) -> Result<(), String> {
    if std::env::var("MIL_EMIT_GUARD").map(|v| v == "0").unwrap_or(false) {
        return Ok(());
    }
    let interval = env_u64("MIL_EMIT_GUARD_MS", 5000);
    let now = now_ms();
    let last = LAST_CHECK_MS.load(Ordering::Relaxed);
    if now.saturating_sub(last) < interval {
        return Ok(());
    }
    LAST_CHECK_MS.store(now, Ordering::Relaxed);

    let min_free = env_u64("MIL_EMIT_MIN_FREE_BYTES", 2 << 30);
    if min_free > 0 {
        if let Some(free) = avail_bytes(out_dir) {
            if free < min_free {
                return Err(format!(
                    "emit guard: {free} bytes free under {out_dir} < floor {min_free} — \
                     refusing to emit .macho programs; free disk space and retry"
                ));
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        let max_pct = env_u64("MIL_EMIT_MAX_FILE_PCT", 90);
        if max_pct > 0 {
            if let (Some(used), Some(max)) =
                (sysctl_uint("kern.num_files"), sysctl_uint("kern.maxfiles"))
            {
                if max > 0 && used * 100 / max > max_pct {
                    return Err(format!(
                        "emit guard: file table {used}/{max} ({}%) >= {max_pct}% — \
                         refusing to emit; system open-file table saturated",
                        used * 100 / max
                    ));
                }
            }
        }

        let min_mem_free = env_f64("MIL_EMIT_MIN_MEM_FREE_PCT", 10.0);
        if min_mem_free > 0.0 {
            if let Some(free_pct) = mem_free_pct() {
                if free_pct < min_mem_free {
                    return Err(format!(
                        "emit guard: memory {free_pct:.0}% free < {min_mem_free}% floor — \
                         refusing to emit; system under memory pressure"
                    ));
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Env knobs are process-global; run the fail-closed checks in one
    /// test so parallel tests can't race the variables.
    #[test]
    fn guard_fails_closed_on_exhausted_resources() {
        // interval 0 → every call performs the checks
        std::env::set_var("MIL_EMIT_GUARD_MS", "0");

        // isolate the disk check: disable file-table + memory floors
        std::env::set_var("MIL_EMIT_MAX_FILE_PCT", "0");
        std::env::set_var("MIL_EMIT_MIN_MEM_FREE_PCT", "0");
        std::env::set_var("MIL_EMIT_MIN_FREE_BYTES", u64::MAX.to_string());
        let err = check("/tmp").expect_err("disk floor must fail closed");
        assert!(err.contains("emit guard"), "{err}");
        assert!(err.contains("bytes free"), "{err}");

        // isolate the memory check (macOS only — memory_pressure exists)
        #[cfg(target_os = "macos")]
        {
            std::env::set_var("MIL_EMIT_MIN_FREE_BYTES", "0");
            std::env::set_var("MIL_EMIT_MIN_MEM_FREE_PCT", "100.0");
            let err = check("/tmp").expect_err("memory floor must fail closed");
            assert!(err.contains("memory"), "{err}");
        }

        std::env::remove_var("MIL_EMIT_GUARD_MS");
        std::env::remove_var("MIL_EMIT_MAX_FILE_PCT");
        std::env::remove_var("MIL_EMIT_MIN_MEM_FREE_PCT");
        std::env::remove_var("MIL_EMIT_MIN_FREE_BYTES");
    }
}
