//! `ane_strings` — scan a loaded image's __TEXT,__cstring for keys
//! matching a substring. Used to hunt private-framework option names
//! (e.g. the ANEC compile param that enables perf-stats descriptors)
//! without pulling files out of the dyld cache.

use std::ffi::{c_char, c_void, CStr};

extern "C" {
    fn dlopen(path: *const c_char, mode: i32) -> *mut c_void;
    fn _dyld_image_count() -> u32;
    fn _dyld_get_image_name(i: u32) -> *const c_char;
    fn _dyld_get_image_header(i: u32) -> *const c_void;
    fn _dyld_get_image_vmaddr_slide(i: u32) -> isize;
}

#[repr(C)]
struct SegCmd {
    cmd: u32,
    cmdsize: u32,
    segname: [u8; 16],
    vmaddr: u64,
    vmsize: u64,
    fileoff: u64,
    filesize: u64,
    maxprot: i32,
    initprot: i32,
    nsects: u32,
    flags: u32,
}

#[repr(C)]
struct Sect {
    sectname: [u8; 16],
    segname: [u8; 16],
    addr: u64,
    size: u64,
    offset: u32,
    align: u32,
    reloff: u32,
    nreloc: u32,
    flags: u32,
    r1: u32,
    r2: u32,
    r3: u32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image = args.get(1).map(|s| s.as_str()).unwrap_or("ANECompiler");
    let needle = args.get(2).map(|s| s.to_lowercase()).unwrap_or_default();
    unsafe {
        for fw in [
            "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            "/System/Library/PrivateFrameworks/Espresso.framework/Espresso",
        ] {
            let c = std::ffi::CString::new(fw).unwrap();
            dlopen(c.as_ptr(), 2);
        }
        for i in 0.._dyld_image_count() {
            let name = CStr::from_ptr(_dyld_get_image_name(i)).to_string_lossy();
            if !name.contains(image) {
                continue;
            }
            let hdr = _dyld_get_image_header(i) as *const u8;
            let slide = _dyld_get_image_vmaddr_slide(i);
            if hdr.is_null() {
                continue;
            }
            // 64-bit mach header: ncmds at +16, sizeofcmds +20, cmds +32
            let ncmds = *(hdr.add(16) as *const u32);
            let mut off = hdr.add(32);
            for _ in 0..ncmds {
                let cmd = *(off as *const u32);
                let cs = *(off.add(4) as *const u32);
                if cmd == 0x19 {
                    let seg = &*(off as *const SegCmd);
                    let mut so = off.add(72);
                    for _ in 0..seg.nsects {
                        let s = &*(so as *const Sect);
                        let sn =
                            CStr::from_ptr(s.sectname.as_ptr() as *const c_char).to_string_lossy();
                        let gn =
                            CStr::from_ptr(s.segname.as_ptr() as *const c_char).to_string_lossy();
                        if gn == "__TEXT" && (sn == "__cstring" || sn == "__objc_methname") {
                            let base = (s.addr as isize + slide) as *const u8;
                            let end = base.add(s.size as usize);
                            let mut p = base;
                            let mut start = base;
                            while p < end {
                                if *p == 0 {
                                    let len = p as usize - start as usize;
                                    if len >= 4 && len < 200 {
                                        let raw = std::slice::from_raw_parts(start, len);
                                        if raw.iter().all(|b| (0x20..0x7f).contains(b)) {
                                            if let Ok(t) = std::str::from_utf8(raw) {
                                                if needle.is_empty()
                                                    || t.to_lowercase().contains(&needle)
                                                {
                                                    println!("{t}");
                                                }
                                            }
                                        }
                                    }
                                    start = p.add(1);
                                }
                                p = p.add(1);
                            }
                        }
                        so = so.add(80);
                    }
                }
                off = off.add(cs as usize);
            }
            eprintln!("[scanned {name}]");
        }
    }
}
