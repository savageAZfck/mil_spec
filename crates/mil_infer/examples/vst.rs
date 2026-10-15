fn main() {
    let a: Vec<String> = std::env::args().collect();
    let anec = mil_infer::anec::Anec::open().expect("ANEC");
    let v = anec.validate(std::path::Path::new(&a[1]), "h16").unwrap();
    println!("validate status={} errs={:?}", v.status, v.errors);
}
