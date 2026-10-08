//! Hardware automaton — experiment 1 of the silicon-duel series.
//!
//! Compiles an Aho-Corasick DFA into a *stateful* MIL graph and runs it on
//! Apple's runtime as a real model: the transition table is a const weight,
//! the DFA state is an `MLState`, each predict steps one byte:
//!
//!   read_state cur → idx = cur*256 + byte → gather(table, idx) = next
//!   → gather(verdict, next) = hit → write_state(next)
//!
//! A deterministic string matcher expressed as matmul-class silicon ops —
//! a workload the Neural Engine was never designed for, running on the
//! stack nobody else emits to.
//!
//! Usage: cargo run -p mil_infer --example dfa --release

#[cfg(target_os = "macos")]
mod imp {
    use mil_infer::{ComputeUnits, Input, Model};
    use mil_machines::Automaton;
    use mil_spec::DType;

    pub fn run() -> Result<(), String> {
        let patterns: Vec<&[u8]> = vec![
            b"password",
            b"api_key",
            b"secret",
            b"ssh-rsa",
            b"BEGIN PRIVATE",
        ];
        let ac = Automaton::build(&patterns);
        println!(
            "dfa: {} states, table {} B",
            ac.states(),
            ac.states() * 256 * 4
        );

        let dir = std::env::temp_dir().join("mil_dfa");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let pkg = dir.join("dfa.mlpackage");
        mil_machines::write_dfa_package(&pkg, &ac).map_err(|e| e.to_string())?;
        let compiled =
            mil_compile::compile(&pkg, &dir.join("compiled")).map_err(|e| e.to_string())?;
        println!(
            "compiled {} via {}",
            compiled.path.display(),
            compiled.backend
        );

        let model = Model::load(&compiled.path, ComputeUnits::CpuAndNeuralEngine)
            .map_err(|e| e.to_string())?;
        let state = model.new_state().map_err(|e| e.to_string())?;

        let stream: Vec<u8> = b"hello world my password is hunter2 and the api_key=xyz not a secret, ssh-rsa ok BEGIN PRIVATE"
            .to_vec();
        let want = ac.run(&stream);
        let mut got = Vec::new();
        let mut t_all = std::time::Duration::ZERO;
        for (i, &byte) in stream.iter().enumerate() {
            let data = (byte as i32).to_le_bytes();
            let p = model
                .predict_with_state(
                    Some(&state),
                    &[Input {
                        name: "byte",
                        shape: &[1],
                        data: &data,
                        dtype: DType::Int32,
                    }],
                )
                .map_err(|e| format!("step {i}: {e}"))?;
            t_all += p.latency;
            let hit = p
                .outputs
                .iter()
                .find(|o| o.name == "hit")
                .and_then(|o| {
                    o.data
                        .get(..4)
                        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                })
                .unwrap_or(-1);
            got.push(hit);
        }
        let matches = got.iter().filter(|&&h| h == 1).count();
        let same = got == want;
        println!(
            "streamed {} bytes → {} hit positions (software: {}) — {} in {:.1}ms total ({:.0}µs/step)",
            stream.len(),
            matches,
            want.iter().filter(|&&h| h == 1).count(),
            if same { "MATCH" } else { "MISMATCH" },
            t_all.as_secs_f64() * 1e3,
            t_all.as_secs_f64() * 1e6 / stream.len() as f64,
        );
        if !same {
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                if g != w {
                    println!(
                        "  first mismatch at byte {} ('{}'): ane={} sw={}",
                        i, stream[i] as char, g, w
                    );
                    break;
                }
            }
            return Err("hardware automaton disagrees with software DFA".into());
        }
        println!("verdict: the automaton ran on the model runtime, byte-exact");
        let _ = pkg;
        Ok(())
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    if let Err(e) = imp::run() {
        eprintln!("dfa: {e}");
        std::process::exit(1);
    }
    #[cfg(not(target_os = "macos"))]
    eprintln!("dfa: macOS only");
}
