//! Keeps `testdata/tiny.bin`, the model used by the Rust, C and C++ tests,
//! in sync with the training code.

use std::fs;
use std::path::Path;

use augur_train::build::{self, BuildOptions, Smoothing};
use augur_train::compile::{self, CompileOptions};
use augur_train::sort::TempDir;

#[test]
fn testdata_is_current() {
    let testdata = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata");
    let tmp = TempDir::new(&std::env::temp_dir()).unwrap();
    let dir = tmp.file("testdata");
    fs::create_dir(&dir).unwrap();
    let arpa = dir.join("tiny.arpa");
    let model = dir.join("tiny.bin");

    build::run(&BuildOptions {
        corpus: testdata.join("corpus.txt"),
        output: arpa.clone(),
        smoothing: Smoothing::Mkn,
        min_count: vec![1, 1, 1],
        vocab_size: 100_000,
        alpha: 0.4,
        memory: 1 << 24,
        tmp_dir: dir.clone(),
    })
    .unwrap();
    compile::run(&CompileOptions {
        arpa,
        output: model.clone(),
    })
    .unwrap();

    let built = fs::read(&model).unwrap();
    let committed = testdata.join("tiny.bin");
    if std::env::var_os("AUGUR_BLESS").is_some() {
        fs::write(&committed, &built).unwrap();
    } else {
        assert!(
            fs::read(&committed).is_ok_and(|c| c == built),
            "testdata/tiny.bin is out of date; regenerate it with \
             `AUGUR_BLESS=1 cargo test -p augur-train testdata_is_current`"
        );
    }
}
