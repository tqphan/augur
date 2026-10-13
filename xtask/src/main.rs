//! Developer tasks. Run `cargo xtask help` for usage.

use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const USAGE: &str = "\
usage: cargo xtask <task>

tasks:
  ci            fmt check, clippy, tests, C header check
  sdk [CONFIG]  configure, build and test the C/C++ SDK with CMake, then
                install it into build/sdk-install and test a separate
                find_package(Augur) project against it (CONFIG defaults
                to Debug)
  check-header  fail if crates/augur-capi/include/augur.h differs from git

environment:
  CMAKE         path to cmake (default: `cmake` on PATH); ctest is taken
                from the same directory";

const HEADER: &str = "crates/augur-capi/include/augur.h";

type TaskResult = Result<(), String>;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("ci") => ci(),
        Some("sdk") => sdk(args.get(1).map_or("Debug", String::as_str)),
        Some("check-header") => check_header(),
        Some("help" | "-h" | "--help") => {
            println!("{USAGE}");
            Ok(())
        }
        _ => Err(USAGE.to_owned()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn ci() -> TaskResult {
    run(cargo().args(["fmt", "--all", "--check"]))?;
    run(cargo().args([
        "clippy",
        "--workspace",
        "--all-targets",
        "--",
        "-D",
        "warnings",
    ]))?;
    run(cargo().args(["test", "--workspace"]))?;
    check_header()
}

fn sdk(config: &str) -> TaskResult {
    let root = workspace_root();
    let build_dir = root.join("build").join("sdk");
    let (cmake, ctest) = match env::var_os("CMAKE") {
        Some(path) => {
            let cmake = PathBuf::from(path);
            let ctest = cmake.with_file_name(format!("ctest{}", env::consts::EXE_SUFFIX));
            (cmake, ctest)
        }
        None => (PathBuf::from("cmake"), PathBuf::from("ctest")),
    };

    run(Command::new(&cmake)
        .arg("-S")
        .arg(&root)
        .arg("-B")
        .arg(&build_dir)
        .arg(format!("-DCMAKE_BUILD_TYPE={config}")))?;
    run(Command::new(&cmake).arg("--build").arg(&build_dir).args([
        "--config",
        config,
        "--parallel",
    ]))?;
    run(Command::new(&ctest)
        .arg("--test-dir")
        .arg(&build_dir)
        .args(["-C", config, "--output-on-failure"]))?;

    // Install into a throwaway prefix inside build/ (never a system
    // location) and build a separate project against it with
    // find_package(Augur), linking the C++ wrapper statically and shared.
    let prefix = root.join("build").join("sdk-install");
    remove_dir(&prefix)?;
    run(Command::new(&cmake)
        .arg("--install")
        .arg(&build_dir)
        .args(["--config", config, "--prefix"])
        .arg(&prefix))?;
    for cpp_shared in ["OFF", "ON"] {
        let consumer_dir = root
            .join("build")
            .join(format!("consumer-cpp-shared-{}", cpp_shared.to_lowercase()));
        remove_dir(&consumer_dir)?;
        run(Command::new(&cmake)
            .arg("-S")
            .arg(root.join("bindings").join("tests").join("consumer"))
            .arg("-B")
            .arg(&consumer_dir)
            .arg(format!("-DCMAKE_PREFIX_PATH={}", prefix.display()))
            .arg(format!("-DCMAKE_BUILD_TYPE={config}"))
            .arg(format!("-DAUGUR_CPP_SHARED={cpp_shared}")))?;
        run(Command::new(&cmake)
            .arg("--build")
            .arg(&consumer_dir)
            .args(["--config", config, "--parallel"]))?;
        run(Command::new(&ctest)
            .arg("--test-dir")
            .arg(&consumer_dir)
            .args(["-C", config, "--output-on-failure"]))?;
    }
    Ok(())
}

/// Removes a directory under build/ and everything in it, if it exists.
fn remove_dir(dir: &Path) -> TaskResult {
    assert!(dir.starts_with(workspace_root().join("build")));
    match std::fs::remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("cannot remove {}: {e}", dir.display())),
    }
}

fn check_header() -> TaskResult {
    // Building augur-capi regenerates the header via build.rs.
    run(cargo().args(["build", "--package", "augur-capi"]))?;
    run(Command::new("git").args(["diff", "--exit-code", "--", HEADER]))
        .map_err(|_| format!("{HEADER} is out of date; commit the regenerated header"))
}

fn cargo() -> Command {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    let mut command = Command::new(cargo);
    command.current_dir(workspace_root());
    command
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives inside the workspace")
        .to_path_buf()
}

fn run(command: &mut Command) -> TaskResult {
    eprintln!("$ {command:?}");
    let status = command
        .status()
        .map_err(|e| format!("failed to start {}: {e}", command.get_program().display()))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("command failed with {status}"))
    }
}
