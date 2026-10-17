//! Gate 0 probe for the custom ANE backend: can the runtime execute an
//! HWX that `aned` did not produce this run?
//!
//! ```text
//! gate0 probe <dir>          write the const-add MIL probe to <dir>/probe.mil
//! gate0 run <mil> <cache> <0|1>
//!     compile via e5rt into <cache> (force_recompile flag) and execute
//! gate0 inventory <dir>      recursive file list: size + fnv1a64 hash
//! gate0 tamper <dir> <from_hex> <to_hex>
//!     byte-pattern replace across every file under <dir>
//! ```

use mil_infer::e5rt::{CompileCfg, DeviceMask, Program};
use mil_infer::espresso::EspressoProgram;
use std::path::{Path, PathBuf};

const N: usize = 64;

fn probe_mil() -> String {
    format!(
        "program(1.3)\n{{\n    func main<ios18>(tensor<fp16, [1, {N}, 1, 1]> x) {{\n        tensor<fp16, [1, {N}, 1, 1]> c = const()[val = tensor<fp16, [1, {N}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n        tensor<fp16, [1, {N}, 1, 1]> z = add(x = x, y = c)[name = string(\"z\")];\n    }} -> (z);\n}}\n"
    )
}

fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

fn f32_of(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32()
}

fn fnv(path: &Path) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    if let Ok(d) = std::fs::read(path) {
        for b in d {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn inventory(dir: &Path) {
    let mut stack = vec![dir.to_path_buf()];
    let mut files = Vec::new();
    while let Some(d) = stack.pop() {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    files.push(p);
                }
            }
        }
    }
    files.sort();
    for f in &files {
        let size = std::fs::metadata(f).map(|m| m.len()).unwrap_or(0);
        println!("{size:>10}  {:016x}  {}", fnv(f), f.display());
    }
}

fn tamper(dir: &Path, from: &[u8], to: &[u8]) {
    assert_eq!(from.len(), to.len());
    let mut total = 0usize;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let Ok(mut data) = std::fs::read(&p) else {
                continue;
            };
            let mut n = 0;
            let mut i = 0;
            while i + from.len() <= data.len() {
                if data[i..i + from.len()] == *from {
                    data[i..i + to.len()].copy_from_slice(to);
                    n += 1;
                    i += to.len();
                } else {
                    i += 1;
                }
            }
            if n > 0 {
                std::fs::write(&p, &data).unwrap();
                println!("{n:>6} replacements in {}", p.display());
                total += n;
            }
        }
    }
    println!("total {total}");
}

fn run(mil: &Path, cache: &Path, force: bool) {
    let cfg = CompileCfg {
        cache_dir: cache.to_path_buf(),
        force_recompile: force,
    };
    let t0 = std::time::Instant::now();
    let (prog, map) = Program::compile_auto_cfg(mil, DeviceMask::Ane, &cfg).expect("compile");
    let compile_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!("functions: {:?}", prog.functions);
    println!(
        "inputs: {:?}, outputs: {:?}, inouts: {:?}",
        map.inputs, map.outputs, map.inouts
    );
    for (name, n) in &map.inputs {
        prog.set_input_f16(name, &vec![f16(1.0); *n as usize])
            .unwrap();
    }
    let t1 = std::time::Instant::now();
    prog.execute().unwrap();
    let exec_us = t1.elapsed().as_secs_f64() * 1e6;
    for (name, _) in &map.outputs {
        let out = prog.output_f16(name).unwrap();
        let vals: Vec<f32> = out.iter().take(8).map(|b| f32_of(*b)).collect();
        println!("compile {compile_ms:.1} ms, exec {exec_us:.0} us, {name}[0..8] = {vals:?}");
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        "probe" => {
            let dir = PathBuf::from(&a[2]);
            let wdir = dir.join("weights");
            std::fs::create_dir_all(&wdir).unwrap();
            let p = dir.join("probe.mil");
            std::fs::write(&p, probe_mil()).unwrap();
            // weight.bin: 64 B global header + deadbeef chunks.
            // chunk = { magic u32, type u32, len u64, data_off u64, zero pad
            //           to 64 } with data immediately after; BLOBFILE
            // offsets point at chunk headers.
            let data: Vec<u8> = [f16(2.5); N].iter().flat_map(|b| b.to_le_bytes()).collect();
            let mut blob = Vec::new();
            blob.extend_from_slice(&[0x0c, 0, 0, 0, 0x02, 0, 0, 0]);
            blob.resize(64, 0);
            blob.extend_from_slice(&[0xef, 0xbe, 0xad, 0xde, 0x01, 0, 0, 0]);
            blob.extend_from_slice(&(data.len() as u64).to_le_bytes());
            blob.extend_from_slice(&128u64.to_le_bytes());
            blob.resize(128, 0);
            blob.extend_from_slice(&data);
            std::fs::write(wdir.join("weight.bin"), &blob).unwrap();
            println!(
                "wrote {} + weights/weight.bin ({} fp16 2.5)",
                p.display(),
                N
            );
        }
        "run" => run(
            Path::new(&a[2]),
            Path::new(&a[3]),
            a[4].parse::<u8>().unwrap() == 1,
        ),
        "inventory" => inventory(Path::new(&a[2])),
        "libprobe" => {
            // libprobe <bundle-dir> — probe e5rt_program_library_create arg types
            extern "C" {
                fn dlopen(path: *const std::ffi::c_char, mode: i32) -> *mut std::ffi::c_void;
                fn dlsym(
                    h: *mut std::ffi::c_void,
                    n: *const std::ffi::c_char,
                ) -> *mut std::ffi::c_void;
            }
            type V = *mut std::ffi::c_void;
            unsafe {
                let esp = dlopen(
                    b"/System/Library/PrivateFrameworks/Espresso.framework/Espresso\0".as_ptr()
                        as *const _,
                    2,
                );
                let cf = dlopen(
                    b"/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation\0".as_ptr()
                        as *const _,
                    2,
                );
                assert!(!esp.is_null() && !cf.is_null());
                let create: extern "C" fn(V, V) -> i64 =
                    std::mem::transmute(dlsym(esp, b"e5rt_program_library_create\0".as_ptr() as _));
                let cfstr_new: extern "C" fn(V, *const i8, u32) -> V =
                    std::mem::transmute(dlsym(cf, b"CFStringCreateWithCString\0".as_ptr() as _));
                let cfurl_new: extern "C" fn(V, V, V) -> V = std::mem::transmute(dlsym(
                    cf,
                    b"CFURLCreateWithFileSystemPath\0".as_ptr() as _,
                ));
                let cpath = std::ffi::CString::new(a[2].as_str()).unwrap();
                let cfstr = cfstr_new(std::ptr::null_mut(), cpath.as_ptr(), 0x08000100);
                let cfurl = cfurl_new(std::ptr::null_mut(), cfstr, std::ptr::null_mut());
                println!("cfstr={cfstr:?} cfurl={cfurl:?}");
                let mut lib: V = std::ptr::null_mut();
                let plib = &mut lib as *mut V;
                for (name, a0, a1) in [
                    ("(cfstr,&lib)", cfstr, plib as V),
                    ("(&lib,cfstr)", plib as V, cfstr),
                    ("(cfurl,&lib)", cfurl, plib as V),
                    ("(&lib,cfurl)", plib as V, cfurl),
                    ("(cpath,&lib)", cpath.as_ptr() as V, plib as V),
                    ("(&lib,cpath)", plib as V, cpath.as_ptr() as V),
                    ("(cfstr,0)", cfstr, std::ptr::null_mut()),
                ] {
                    lib = std::ptr::null_mut();
                    let rc = create(a0, a1);
                    println!("{name} -> rc={rc} lib={lib:?}");
                }
            }
        }
        "anesvc" => {
            // anesvc <hwx-path> <slot-hex> [flag-hex]
            //   open ANE device via ANEServices, build the 0xd78 ProgramCreate
            //   wire struct, put <hwx-path> in the char[0x400] slot at <slot>,
            //   optionally set flag byte at <flag>, and call ProgramCreate.
            extern "C" {
                fn dlopen(path: *const std::ffi::c_char, mode: i32) -> *mut std::ffi::c_void;
                fn dlsym(
                    h: *mut std::ffi::c_void,
                    n: *const std::ffi::c_char,
                ) -> *mut std::ffi::c_void;
            }
            type V = *mut std::ffi::c_void;
            unsafe {
                let h = dlopen(
                    b"/System/Library/PrivateFrameworks/ANEServices.framework/ANEServices\0"
                        .as_ptr() as *const _,
                    2,
                );
                assert!(!h.is_null());
                let dev_open: extern "C" fn(V, V, V, V) -> i32 =
                    std::mem::transmute(dlsym(h, b"ANEServicesDeviceOpen\0".as_ptr() as _));
                let prog_create: extern "C" fn(V, V, V) -> i64 =
                    std::mem::transmute(dlsym(h, b"ANEServicesProgramCreate\0".as_ptr() as _));
                // device params: u32 type at +0 — try value from argv[5] or 0
                let ptype: u32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);
                let mut params = [0u8; 64];
                params[0..4].copy_from_slice(&ptype.to_le_bytes());
                let mut dev: V = std::ptr::null_mut();
                let rc = dev_open(
                    &mut dev as *mut V as V,
                    params.as_mut_ptr() as V,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                );
                println!("DeviceOpen rc={rc} dev={dev:?}");
                if rc != 0 {
                    return;
                }
                let mut args = vec![0u8; 0xd78];
                let slot = usize::from_str_radix(a[3].trim_start_matches("0x"), 16).unwrap();
                let path = a[2].as_bytes();
                args[slot..slot + path.len()].copy_from_slice(path);
                if let Some(fl) = a.get(4).filter(|s| !s.is_empty()) {
                    let fo = usize::from_str_radix(fl.trim_start_matches("0x"), 16).unwrap();
                    args[fo] = 1;
                }
                let mut out = [0u8; 0xe8];
                let rc2 = prog_create(dev, args.as_mut_ptr() as V, out.as_mut_ptr() as V);
                println!("ProgramCreate rc={rc2:#x} out[0..16]={:02x?}", &out[..16]);
                // chain into CreateNewInstance — that's where aned actually loads
                let cni: Option<extern "C" fn(V, V, V) -> i64> = {
                    let p = dlsym(h, b"ANEServicesProgramCreateNewInstance\0".as_ptr() as _);
                    (!p.is_null()).then(|| std::mem::transmute(p))
                };
                if let Some(cni) = cni {
                    let mut inst = [0u8; 0xe8];
                    let rc3 = cni(dev, out.as_mut_ptr() as V, inst.as_mut_ptr() as V);
                    println!(
                        "CreateNewInstance rc={rc3:#x} inst[0..16]={:02x?}",
                        &inst[..16]
                    );
                }
            }
        }
        "libexec" => {
            // libexec <bundle-dir> — load a precompiled e5 bundle directly
            let (prog, map) = Program::from_e5_bundle(Path::new(&a[2])).expect("from_e5_bundle");
            println!("functions: {:?}", prog.functions);
            println!(
                "inputs: {:?}, outputs: {:?}, inouts: {:?}",
                map.inputs, map.outputs, map.inouts
            );
            for (name, n) in &map.inputs {
                prog.set_input_f16(name, &vec![f16(1.0); *n as usize])
                    .unwrap();
            }
            let t1 = std::time::Instant::now();
            prog.execute().unwrap();
            let exec_us = t1.elapsed().as_secs_f64() * 1e6;
            for (name, _) in &map.outputs {
                let out = prog.output_f16(name).unwrap();
                let vals: Vec<f32> = out.iter().take(8).map(|b| f32_of(*b)).collect();
                println!("exec {exec_us:.0} us, {name}[0..8] = {vals:?}");
            }
        }
        "anec" => {
            // anec <mil> <outdir> [arch] — compile MIL through anec.
            let arch = a.get(4).map(String::as_str).unwrap_or("h16");
            let anec = mil_infer::anec::Anec::open().expect("anec open");
            let v = anec
                .compile(Path::new(&a[2]), &PathBuf::from(&a[3]), arch)
                .expect("anec compile");
            println!(
                "anec: rc={} status={:#x} dram={:?} artifact={:?} errors={:?}",
                v.rc, v.status, v.dram_usage, v.artifact, v.errors
            );
        }
        "mkpkg" => {
            // mkpkg <dir> — build a complete Espresso package for the
            // two-input add graph: anec compiles add2.mil to the ANE
            // Mach-O, then we emit the deployment envelope.
            let dir = PathBuf::from(&a[2]);
            let out = dir.join("out");
            let pkg = dir.join("model");
            std::fs::create_dir_all(&out).unwrap();
            std::fs::create_dir_all(&pkg).unwrap();
            let mil = dir.join("add2.mil");
            std::fs::write(
                &mil,
                format!(
                    "program(1.3)\n{{\n    func main<ios18>(tensor<fp16, [1, {N}, 1, 1]> x, tensor<fp16, [1, {N}, 1, 1]> y) {{\n        tensor<fp16, [1, {N}, 1, 1]> z = add(x = x, y = y)[name = string(\"z\")];\n    }} -> (z);\n}}\n"
                ),
            )
            .unwrap();
            let anec = mil_infer::anec::Anec::open().expect("anec open");
            let v = anec.compile(&mil, &out, "h16").expect("anec compile");
            println!(
                "anec: rc={} status={:#x} dram={:?} artifact={:?}",
                v.rc, v.status, v.dram_usage, v.artifact
            );
            let macho = out.join("add2.macho");
            assert!(macho.exists(), "no add2.macho emitted");
            std::fs::copy(&macho, pkg.join("model.H16.espresso.hwx")).unwrap();
            std::fs::write(
                pkg.join("model.espresso.net"),
                "{\n \"storage\": \"model.espresso.weights\",\n \"layers\": [\n  {\"name\":\"add0\",\"type\":\"elementwise\",\"bottom\":\"x,y\",\"top\":\"z\",\"operation\":0,\"alpha\":1,\"beta\":0,\"fused_relu\":0,\"weights\":{},\"debug_info\":\"\"}\n ],\n \"properties\": {},\n \"analyses\": {},\n \"format_version\": 200,\n \"metadata_in_weights\": []\n}\n",
            )
            .unwrap();
            std::fs::write(
                pkg.join("model.espresso.shape"),
                format!("{{\"layer_shapes\":{{\n \"x\":{{\"n\":1,\"k\":{N},\"h\":1,\"w\":1,\"_rank\":4}},\n \"y\":{{\"n\":1,\"k\":{N},\"h\":1,\"w\":1,\"_rank\":4}},\n \"z\":{{\"n\":1,\"k\":{N},\"h\":1,\"w\":1,\"_rank\":4}}\n}}}}\n"),
            )
            .unwrap();
            std::fs::write(pkg.join("model.espresso.weights"), [0u8; 64]).unwrap();
            std::fs::write(
                pkg.join("model.H16.espresso.precompilation_info"),
                "{\"ANE_COMPILER\":{\"kernel_validation_status\":{\"visit_order\":[],\"to_fallback\":[],\"any_fallback\":0,\"unsupported\":[]}}}\n",
            )
            .unwrap();
            println!("package at {}", pkg.display());
        }
        "ownexec" => {
            // ownexec <pkg-dir> — load our own Espresso package through
            // _ANEClient and execute the add graph on the ANE, using the
            // strides `aned` itself reports (no hardcoded layout).
            let mut m = EspressoProgram::load(Path::new(&a[2])).expect("load");
            println!(
                "loaded: ph={} inputs={:?} outputs={:?} procs={:?}",
                m.program_handle,
                m.input_descs.iter().map(|d| &d.name).collect::<Vec<_>>(),
                m.output_descs.iter().map(|d| &d.name).collect::<Vec<_>>(),
                m.procedures
            );
            for (i, v) in [(0usize, f16(1.0)), (1, f16(2.5))] {
                let d = m.input_descs[i].clone();
                let b = m.input_mut(i);
                for k in 0..d.k {
                    let off = d.offset(0, 0, k, 0, 0);
                    b[off..off + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
            let t = std::time::Instant::now();
            m.eval().expect("eval");
            let us = t.elapsed().as_secs_f64() * 1e6;
            let d = m.output_descs[0].clone();
            let out = m.output(0);
            let want = f16(3.5);
            let good = (0..d.k)
                .filter(|k| {
                    let off = d.offset(0, 0, *k, 0, 0);
                    u16::from_le_bytes([out[off], out[off + 1]]) == want
                })
                .count();
            let lanes: Vec<f32> = (0..4)
                .map(|k| {
                    let off = d.offset(0, 0, k, 0, 0);
                    f32_of(u16::from_le_bytes([out[off], out[off + 1]]))
                })
                .collect();
            println!(
                "eval {us:.0} us, z[0..4] = {lanes:?}, lanes==3.5: {good}/{}",
                d.k
            );
        }
        "tamper" => {
            let hex = |s: &str| -> Vec<u8> {
                (0..s.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
                    .collect()
            };
            tamper(Path::new(&a[2]), &hex(&a[3]), &hex(&a[4]));
        }
        "dladdr" => {
            #[repr(C)]
            struct DlInfo {
                fname: *const std::ffi::c_char,
                fbase: *mut std::ffi::c_void,
                sname: *const std::ffi::c_char,
                saddr: *mut std::ffi::c_void,
            }
            extern "C" {
                fn dlopen(path: *const std::ffi::c_char, mode: i32) -> *mut std::ffi::c_void;
                fn dlsym(
                    h: *mut std::ffi::c_void,
                    n: *const std::ffi::c_char,
                ) -> *mut std::ffi::c_void;
                fn dladdr(p: *mut std::ffi::c_void, i: *mut DlInfo) -> i32;
            }
            let path = std::ffi::CString::new(a[2].as_str()).unwrap();
            unsafe {
                let h = dlopen(path.as_ptr(), 2);
                assert!(!h.is_null());
                // list every e5rt_* export we can find via the symtab of a
                // handful of known names + brute probing is overkill; just
                // resolve the binary's real path first.
                let sym = std::ffi::CString::new("e5rt_e5_compiler_compile").unwrap();
                let p = dlsym(h, sym.as_ptr());
                let mut i = std::mem::zeroed::<DlInfo>();
                if dladdr(p, &mut i) != 0 && !i.fname.is_null() {
                    println!(
                        "real path: {}",
                        std::ffi::CStr::from_ptr(i.fname).to_string_lossy()
                    );
                }
            }
        }
        "symbytes" => {
            // symbytes <image-path> <symbol> <n-bytes> <out-file>
            extern "C" {
                fn dlopen(path: *const std::ffi::c_char, mode: i32) -> *mut std::ffi::c_void;
                fn dlsym(
                    h: *mut std::ffi::c_void,
                    n: *const std::ffi::c_char,
                ) -> *mut std::ffi::c_void;
            }
            let path = std::ffi::CString::new(a[2].as_str()).unwrap();
            let sym = std::ffi::CString::new(a[3].as_str()).unwrap();
            let n: usize = a[4].parse().unwrap();
            unsafe {
                let h = dlopen(path.as_ptr(), 2);
                assert!(!h.is_null());
                let p = dlsym(h, sym.as_ptr());
                assert!(!p.is_null(), "symbol not found");
                let bytes = std::slice::from_raw_parts(p as *const u8, n);
                std::fs::write(&a[5], bytes).unwrap();
                println!("{} @ {:p} -> {} ({} bytes)", a[3], p, a[5], n);
            }
        }
        "images" => {
            extern "C" {
                fn dlopen(path: *const std::ffi::c_char, mode: i32) -> *mut std::ffi::c_void;
                fn _dyld_image_count() -> u32;
                fn _dyld_get_image_name(i: u32) -> *const std::ffi::c_char;
            }
            let path = std::ffi::CString::new(a[2].as_str()).unwrap();
            unsafe {
                assert!(!dlopen(path.as_ptr(), 2).is_null());
                let mut names: Vec<String> = (0.._dyld_image_count())
                    .map(|i| {
                        let n = _dyld_get_image_name(i);
                        if n.is_null() {
                            String::new()
                        } else {
                            std::ffi::CStr::from_ptr(n).to_string_lossy().into_owned()
                        }
                    })
                    .collect();
                names.sort();
                for n in names.iter().filter(|n| {
                    let l = n.to_lowercase();
                    l.contains("ane")
                        || l.contains("neural")
                        || l.contains("espresso")
                        || l.contains("mil")
                }) {
                    println!("{n}");
                }
            }
        }
        "dumpimg" => {
            // carve a dyld-loaded image out of our own address space:
            // dlopen <a[2]>, then write its file-backed segments to <a[3]>.
            extern "C" {
                fn dlopen(path: *const std::ffi::c_char, mode: i32) -> *mut std::ffi::c_void;
                fn _dyld_image_count() -> u32;
                fn _dyld_get_image_name(i: u32) -> *const std::ffi::c_char;
                fn _dyld_get_image_header(i: u32) -> *const u8;
                fn _dyld_get_image_vmaddr_slide(i: u32) -> isize;
            }
            let cstr = |p: *const std::ffi::c_char| {
                unsafe { std::ffi::CStr::from_ptr(p) }
                    .to_string_lossy()
                    .into_owned()
            };
            let path = std::ffi::CString::new(a[2].as_str()).unwrap();
            unsafe {
                assert!(!dlopen(path.as_ptr(), 2).is_null());
                let want = a[2].as_str();
                let short = want.rsplit('/').next().unwrap_or(want).to_lowercase();
                let mut found = None;
                for i in 0.._dyld_image_count() {
                    let n = _dyld_get_image_name(i);
                    if !n.is_null() && cstr(n).to_lowercase().contains(&short) {
                        println!("image {i}: {}", cstr(n));
                        if found.is_none() {
                            found = Some(i);
                        }
                    }
                }
                let i = found.expect("image not loaded");
                let hdr = _dyld_get_image_header(i);
                let b = std::slice::from_raw_parts(hdr, 1 << 16);
                let slide = _dyld_get_image_vmaddr_slide(i);
                let ncmds = u32::from_le_bytes(b[16..20].try_into().unwrap()) as usize;
                let sizeofcmds = u32::from_le_bytes(b[20..24].try_into().unwrap()) as usize;
                assert!(32 + sizeofcmds <= b.len(), "header cmds fit in window");
                let mut off = 32usize;
                let mut vmbuf = Vec::new();
                let mut segs: Vec<(u64, u64, u64, u64, String)> = Vec::new();
                for _ in 0..ncmds {
                    let cmd = u32::from_le_bytes(b[off..off + 4].try_into().unwrap());
                    let size = u32::from_le_bytes(b[off + 4..off + 8].try_into().unwrap()) as usize;
                    if cmd == 0x19 {
                        let name = std::str::from_utf8(&b[off + 8..off + 24])
                            .unwrap_or("?")
                            .trim_end_matches('\0')
                            .to_string();
                        let vmaddr = u64::from_le_bytes(b[off + 24..off + 32].try_into().unwrap());
                        let vmsize = u64::from_le_bytes(b[off + 32..off + 40].try_into().unwrap());
                        let fileoff = u64::from_le_bytes(b[off + 40..off + 48].try_into().unwrap());
                        let filesize =
                            u64::from_le_bytes(b[off + 48..off + 56].try_into().unwrap());
                        if name != "__PAGEZERO" && filesize > 0 {
                            segs.push((vmaddr, vmsize, fileoff, filesize, name));
                        }
                    }
                    off += size;
                }
                let total = segs.iter().map(|s| s.2 + s.3).max().unwrap_or(0) as usize;
                vmbuf.resize(total, 0);
                let preferred = segs
                    .iter()
                    .find(|s| s.4 == "__TEXT")
                    .map(|s| s.0)
                    .unwrap_or(0);
                assert_eq!(slide as u64, hdr as u64 - preferred, "slide sanity");
                for (vmaddr, _vmsize, fileoff, filesize, name) in &segs {
                    let src = hdr
                        .wrapping_sub(preferred as usize)
                        .wrapping_add(*vmaddr as usize);
                    std::ptr::copy_nonoverlapping(
                        src,
                        vmbuf.as_mut_ptr().add(*fileoff as usize),
                        *filesize as usize,
                    );
                    println!(
                        "{name}: vm {vmaddr:#x} +{_vmsize:#x} -> file {fileoff:#x} +{filesize:#x}"
                    );
                }
                std::fs::write(&a[3], &vmbuf).unwrap();
                println!("wrote {} ({} bytes)", a[3], vmbuf.len());
            }
        }
        other => eprintln!("unknown subcommand {other}"),
    }
}
