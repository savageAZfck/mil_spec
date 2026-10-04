//! Compile a text IR program into a .mlpackage.
//!
//! The IR is name-based SSA — ops take value names, not nested
//! expressions. Validation happens at compile time: unknown names,
//! shape mismatches, bad permutations, undeclared states all fail
//! before any bytes hit the wire.

use mil_spec::ir::compile;
use mil_spec::{encode_model, write_mlpackage, ModelMeta};

fn main() {
    // A tiny stateful program: read a KV state, concat the step, write
    // it back, emit the merged tensor.
    let src = r#"
        # one decode step against a KV cache state
        state kv: fp16[1,8,64,64]
        input step: fp16[1,8,1,64]
        cur = read_state(kv)
        merged = concat([cur, step], 2)
        write_state(kv, merged)
        output merged
    "#;

    let prog = match compile(src) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("compile failed: {e}");
            std::process::exit(1);
        }
    };

    println!("compiled: {} ops", prog.block.ops.len());
    println!(
        "inputs:  {:?}",
        prog.inputs.iter().map(|f| &f.name).collect::<Vec<_>>()
    );
    println!(
        "states:  {:?}",
        prog.states.iter().map(|f| &f.name).collect::<Vec<_>>()
    );
    println!(
        "outputs: {:?}",
        prog.outputs.iter().map(|f| &f.name).collect::<Vec<_>>()
    );

    let spec = encode_model(
        &prog.inputs,
        &prog.outputs,
        &prog.states,
        &prog.block,
        &prog.fn_inputs,
        &ModelMeta::new(8, "CoreML5").description("ir-compiled stateful probe"),
    );
    let dir = std::env::temp_dir().join("mil_ir_demo.mlpackage");
    let _ = std::fs::remove_dir_all(&dir);
    write_mlpackage(&dir, &spec, None).unwrap();
    println!("wrote {}", dir.display());
    println!("compile with: xcrun coremlc compile {} .", dir.display());
}
