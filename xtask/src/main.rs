//! Developer tasks. Run `cargo xtask help` for usage.

use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const USAGE: &str = "\
usage: cargo xtask <task>

tasks:
  ci            fmt check, clippy, tests, C header check
  sdk [CONFIG]  configure, build and test the C/C++ SDK with CMake
                (CONFIG defaults to Debug)
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
    run(Command::new(ctest).arg("--test-dir").arg(&build_dir).args([
        "-C",
        config,
        "--output-on-failure",
    ]))
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
