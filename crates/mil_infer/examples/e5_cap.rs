//! `e5_cap` — compile a MIL via the e5 compiler (into a cache dir), then
//! scan our own address space for the deployed Mach-O program image the
//! compiler produced (it is shipped to aned via shared memory, never
//! written to our filesystem).

use mil_infer::e5rt::{CompileCfg, DeviceMask, Program};
use std::path::PathBuf;

type MachPort = u32;
extern "C" {
    fn mach_task_self() -> MachPort;
    fn mach_vm_region(
        task: MachPort,
        addr: *mut u64,
        size: *mut u64,
        flavor: i32,
        info: *mut u32,
        count: *mut u32,
        object: *mut MachPort,
    ) -> i32;
    fn mach_vm_read_overwrite(
        task: MachPort,
        addr: u64,
        size: u64,
        data: u64,
        out_size: *mut u64,
    ) -> i32;
}

const VM_REGION_BASIC_INFO_64: i32 = 9;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mil = PathBuf::from(&a[1]);
    let cache = a
        .get(2)
        .map(PathBuf::from)
        .unwrap_or(PathBuf::from("/tmp/e5cap"));
    let cc = CompileCfg {
        cache_dir: cache.clone(),
        force_recompile: true,
    };
    match Program::compile_auto_cfg(&mil, DeviceMask::Ane, &cc) {
        Ok((_p, map)) => println!("compiled: {map:?}"),
        Err(e) => println!("compile failed: {e:?}"),
    }

    // scan regions >16KB for the ANE Mach-O magic (ce fa ef be)
    let task = unsafe { mach_task_self() };
    let mut addr: u64 = 0;
    let mut found = 0;
    loop {
        let mut size: u64 = 0;
        let mut info = [0u32; 16];
        let mut count: u32 = 16 * 4;
        let mut obj: MachPort = 0;
        let kr = unsafe {
            mach_vm_region(
                task,
                &mut addr,
                &mut size,
                VM_REGION_BASIC_INFO_64,
                info.as_mut_ptr(),
                &mut count,
                &mut obj,
            )
        };
        if kr != 0 {
            break;
        }
        if (0x8000..=64 * 1024 * 1024).contains(&size) {
            let mut buf = vec![0u8; size as usize];
            let mut got: u64 = 0;
            let kr2 = unsafe {
                mach_vm_read_overwrite(task, addr, size, buf.as_mut_ptr() as u64, &mut got)
            };
            if kr2 == 0 && got == size {
                let magic = [0xceu8, 0xfa, 0xef, 0xbe];
                let mut i = 0usize;
                let l = buf.len();
                while i + 4 <= l {
                    if buf[i..i + 4] == magic {
                        // plausible header: cputype/filetype sane
                        let path = format!("/tmp/cap_vm_{addr:x}_{i:x}.bin");
                        std::fs::write(&path, &buf[i..]).unwrap();
                        println!("macho magic @ region {addr:#x}+{i:#x} sz={size:#x} -> {path}");
                        found += 1;
                    }
                    i += 8;
                }
            }
        }
        addr += size;
    }
    println!("scan done found={found}");
}
