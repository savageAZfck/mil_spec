// graph_warm — emit one graph dir and write its plan cache.
// Run once per layer dir so the malloc spike of a full-model emit
// stays inside a single process lifetime: the multi-layer emit
// path accumulates freed arena bytes (~15 GB over 40k nodes) and
// gets jetsam-killed on 16 GiB hosts; one layer is a few hundred MB.
//
//   graph_warm /path/to/layer_dir/net.mil

fn main() {
    let mil = std::env::args()
        .nth(1)
        .expect("usage: graph_warm <dir>/net.mil");
    let dir = std::path::Path::new(&mil)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".".into());
    let t0 = std::time::Instant::now();
    match mil_infer::graph::compile_graph(&mil, &dir) {
        Ok(plan) => println!(
            "{dir}: {} nodes, {} externals — {:.1}s",
            plan.nodes.len(),
            plan.externals.len(),
            t0.elapsed().as_secs_f64()
        ),
        Err(e) => {
            eprintln!("{dir}: {e}");
            std::process::exit(1);
        }
    }
}
