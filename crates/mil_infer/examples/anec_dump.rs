//! Dump the mapped ANECompiler image (dyld-cache backed) for offline disassembly.
use std::ffi::{c_char, c_uint, c_void, CString};
use std::fs;

#[link(name = "dl", kind = "dylib")]
unsafe extern "C" {
    fn dlopen(path: *const c_char, mode: i32) -> *mut c_void;
}
unsafe extern "C" {
    fn _dyld_image_count() -> c_uint;
    fn _dyld_get_image_name(idx: c_uint) -> *const c_char;
    fn _dyld_get_image_header(idx: c_uint) -> *const c_void;
}

#[repr(C)]
struct MachHeader64 {
    magic: u32,
    cputype: i32,
    cpusubtype: i32,
    filetype: u32,
    ncmds: u32,
    sizeofcmds: u32,
    flags: u32,
    reserved: u32,
}
#[repr(C)]
struct SegCmd64 {
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

fn main() {
    unsafe {
        let p = CString::new("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler")
            .unwrap();
        let h = dlopen(p.as_ptr(), 2);
        assert!(!h.is_null());
        let n = _dyld_image_count();
        for i in 0..n {
            let name = std::ffi::CStr::from_ptr(_dyld_get_image_name(i)).to_string_lossy();
            if !name.contains("ANECompiler") {
                continue;
            }
            let mh = _dyld_get_image_header(i) as *const MachHeader64;
            let hdr = &*mh;
            println!("{} hdr={:p} ncmds={}", name, mh, hdr.ncmds);
            let mut lc = (mh as usize) + 32;
            for _ in 0..hdr.ncmds {
                let cmd = *(lc as *const u32);
                let cmdsize = *((lc + 4) as *const u32);
                if cmd == 0x19 {
                    // LC_SEGMENT_64
                    let sg = &*(lc as *const SegCmd64);
                    let seg = std::str::from_utf8(&sg.segname)
                        .unwrap()
                        .trim_end_matches('\0');
                    println!(
                        "  {} vmaddr={:#x} vmsize={:#x} fileoff={:#x} filesz={:#x}",
                        seg, sg.vmaddr, sg.vmsize, sg.fileoff, sg.filesize
                    );
                    let data = std::slice::from_raw_parts((lc - 32) as *const u8, 0); // placeholder
                    let _ = data;
                    // image base = mh; segment contents at mh + (vmaddr - first_text_vmaddr)?
                    // In dyld cache, image is mapped contiguously from mh upward by fileoff order?
                    // Actually segments are at mh + fileoff only for file-aligned images; cache images use vmaddr.
                    // Simpler: dump the whole mapped range: find min vmaddr (== hdr addr) and max end.
                }
                lc += cmdsize as usize;
            }
            // dump from hdr, total = max(vmaddr+vmsize) - vmaddr_of_first_seg
            let mut lc = (mh as usize) + 32;
            for _ in 0..hdr.ncmds {
                let cmd = *(lc as *const u32);
                let cmdsize = *((lc + 4) as *const u32);
                if cmd == 0x19 {
                    let sg = &*(lc as *const SegCmd64);
                    let seg = std::str::from_utf8(&sg.segname)
                        .unwrap()
                        .trim_end_matches('\0')
                        .to_string();
                    if sg.vmsize > 0
                        && (seg == "__TEXT" || seg == "__DATA_CONST" || seg == "__AUTH_CONST")
                    {
                        let img =
                            std::slice::from_raw_parts(sg.vmaddr as *const u8, sg.vmsize as usize);
                        let path = format!(
                            "/tmp/anecompiler_{}.bin",
                            seg.trim_matches('_').replace("_", "")
                        );
                        fs::write(&path, img).unwrap();
                        println!(
                            "wrote {} {} bytes @ vmaddr {:#x}",
                            path,
                            img.len(),
                            sg.vmaddr
                        );
                    }
                }
                lc += cmdsize as usize;
            }
            println!("hdr va={:#x}", mh as usize as u64);
            break;
        }
    }
}
