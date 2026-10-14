fn main() {
    let anec = mil_infer::Anec::open().unwrap();
    let args: Vec<String> = std::env::args().collect();
    let v = anec
        .compile(std::path::Path::new(&args[1]), std::path::Path::new(&args[2]), "h16")
        .unwrap();
    println!("status={} dram={:?} err={:?}", v.status, v.dram_usage, v.errors);
}
