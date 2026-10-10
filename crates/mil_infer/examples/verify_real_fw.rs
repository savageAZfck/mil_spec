//! Verifies the REAL Bad Apple firewall artifact
//! (/var/lib/bad_apple/firewall.mlmodelc, built by `badapple firewall
//! compile`) against the software automaton over the same pattern set.

use mil_infer::{ComputeUnits, Input, Model};
use mil_machines::Automaton;
use mil_spec::DType;
use std::path::Path;

fn main() {
    let artifact = Path::new("/var/lib/bad_apple/firewall.mlmodelc");
    if !artifact.exists() {
        println!("skip: no Bad Apple firewall artifact — run `badapple firewall compile` first");
        return;
    }
    let pats: Vec<&[u8]> = [
        "sk-",
        "ssh-rsa",
        "-----begin",
        "-----end",
        "begin private key",
        "begin openssh private key",
    ]
    .iter()
    .map(|s| s.as_bytes())
    .collect();
    let ac = Automaton::build(&pats);

    let model = Model::load(artifact, ComputeUnits::CpuAndNeuralEngine).unwrap();
    let state = model.new_state().unwrap();

    let stream = "ok SK-LIVE-12345 -----BEGIN RSA and ssh-rsa AAAA safe".to_lowercase();
    let mut hw = vec![];
    for &b in stream.as_bytes() {
        let data = (b as i32).to_le_bytes();
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
            .unwrap();
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
        hw.push(hit);
    }
    let sw = ac.run(stream.as_bytes());
    let hwn: Vec<usize> = hw
        .iter()
        .enumerate()
        .filter(|(_, &h)| h != 0)
        .map(|(i, _)| i)
        .collect();
    let swn: Vec<usize> = sw
        .iter()
        .enumerate()
        .filter(|(_, &h)| h != 0)
        .map(|(i, _)| i)
        .collect();
    println!("hw hits at {hwn:?}");
    println!("sw hits at {swn:?}");
    println!(
        "{}",
        if hw == sw {
            "MATCH — real firewall verified on ANE"
        } else {
            "MISMATCH"
        }
    );
    std::process::exit(if hw == sw { 0 } else { 1 });
}
