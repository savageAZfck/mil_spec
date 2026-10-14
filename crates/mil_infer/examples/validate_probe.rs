// quick: call Anec::validate on conv1 to see detailed verdict
fn main() {
    let a = mil_infer::anec::Anec::open().unwrap();
    let v = a
        .validate(
            std::path::Path::new("/tmp/ane_dissect/conv1/net.mil"),
            "h16",
        )
        .unwrap();
    println!(
        "rc={} status={:#x}\nerrors={:?}\ndetail={}",
        v.rc, v.status, v.errors, v.detail
    );
}
