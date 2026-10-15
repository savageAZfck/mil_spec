//! st_compile <mil> — compile a stateful MIL through e5rt and list every
//! artifact the cache produces (the stateful Mach-O spec).

use mil_infer::e5rt::{CompileCfg, DeviceMask, Program};
use std::path::{Path, PathBuf};

fn walk(d: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(d).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            walk(&p, out);
        } else {
            out.push(p);
        }
    }
}

fn main() {
    let mil = PathBuf::from(std::env::args().nth(1).unwrap());
    let cache = PathBuf::from("/tmp/e5_st_cache");
    let _ = std::fs::remove_dir_all(&cache);
    let cc = CompileCfg {
        cache_dir: cache.clone(),
        force_recompile: true,
    };
    // stateful: use compile_auto which discovers inout ports itself
    match Program::compile_auto_cfg(&mil, DeviceMask::Ane, &cc) {
        Ok((prog, map)) => {
            println!(
                "compiled. inputs={:?} outputs={:?} inouts={:?}",
                map.inputs, map.outputs, map.inouts
            );
            prog.execute().expect("exec");
            println!("exec ok");
        }
        Err(e) => println!("compile ERR: {e:?}"),
    }
    let mut files = Vec::new();
    if cache.exists() {
        walk(&cache, &mut files);
    }
    for f in &files {
        println!("  {} ({} B)", f.display(), f.metadata().unwrap().len());
    }
}
