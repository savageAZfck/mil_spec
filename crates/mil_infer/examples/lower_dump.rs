fn main() {
    let src = include_str!("../../../tests/ane_lower.rs");
    // extract the SRC const between r#" and "#;
    let i = src.find("const SRC").unwrap();
    let j = src[i..].find("r#\"").unwrap() + i + 3;
    let k = src[j..].find("\"#;").unwrap() + j;
    let mil = &src[j..k];
    let (out, rep) = mil_spec::ane_lower::lower_for_ane(mil, 0).expect("lower");
    println!("REPORT: {:?}", rep);
    std::fs::write("/tmp/st1/lowered.mil", &out).unwrap();
    println!("{}", out);
}
