//! Regenerates `include/augur.h` from the crate's `extern "C"` items.

use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    // Give the shared library a SONAME, so that programs linked against an
    // installed copy find it by name rather than by its build path. (On
    // Apple platforms rustc already sets the install name to @rpath/….)
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if matches!(
        target_os.as_str(),
        "linux" | "android" | "freebsd" | "netbsd" | "openbsd" | "dragonfly" | "illumos"
    ) {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libaugur_capi.so");
    }

    let config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml"))
        .expect("failed to read cbindgen.toml");
    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .expect("failed to generate C header")
        // Only rewrites the file when its contents change.
        .write_to_file(crate_dir.join("include").join("augur.h"));
}
