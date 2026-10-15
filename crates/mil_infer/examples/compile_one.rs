fn main() {
    let anec = mil_infer::Anec::open().unwrap();
    let args: Vec<String> = std::env::args().collect();
    let v = anec
        .compile(
            std::path::Path::new(&args[1]),
            std::path::Path::new(&args[2]),
            args.get(3).map(|s| s.as_str()).unwrap_or("h16"),
        )
        .unwrap();
    println!(
        "status={} dram={:?} err={:?}",
        v.status, v.dram_usage, v.errors
    );
    println!("DETAIL:\n{}", v.detail);
}
