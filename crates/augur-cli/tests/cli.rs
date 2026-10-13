//! Runs the `augur` binary.

use std::path::PathBuf;
use std::process::{Command, Output};

fn tiny_model() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny.bin")
}

fn augur(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_augur"))
        .args(args)
        .output()
        .expect("augur runs")
}

fn stdout(output: &Output) -> String {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout.clone()).unwrap()
}

#[test]
fn predicts() {
    let model = tiny_model();
    let model = model.to_str().unwrap();
    let out = stdout(&augur(&["predict", "-m", model, "-n", "2", "I live in "]));
    let words: Vec<&str> = out
        .lines()
        .map(|l| l.split_whitespace().nth(1).unwrap())
        .collect();
    assert_eq!(words, ["London", "Paris"]);
    assert!(out.contains("ngram"));

    let out = stdout(&augur(&["predict", "--json", "-m", model, "I live in Lon"]));
    let json: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(json[0]["text"], "London");
    assert_eq!(json[0]["sources"][0], "ngram");
}

#[test]
fn shows_model_info() {
    let model = tiny_model();
    let out = stdout(&augur(&["info", model.to_str().unwrap()]));
    assert!(out.contains("order       3"), "{out}");
    assert!(out.contains("1-grams"));

    let out = stdout(&augur(&["info", "--json", model.to_str().unwrap()]));
    let json: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(json["order"], 3);
    assert_eq!(json["ngrams"].as_array().unwrap().len(), 3);
    assert_eq!(json["meta"]["model.smoothing"], "modified-kneser-ney");
}

#[test]
fn evaluates_a_text() {
    let model = tiny_model();
    let text = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/corpus.txt");
    let out = stdout(&augur(&[
        "eval",
        "--json",
        "-m",
        model.to_str().unwrap(),
        "--learn",
        text.to_str().unwrap(),
    ]));
    let json: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert!(json["words"].as_u64().unwrap() > 50);
    assert!(json["keystroke_savings"].as_f64().unwrap() > 0.3);
    assert_eq!(json["learn"], true);

    let out = stdout(&augur(&[
        "eval",
        "-m",
        model.to_str().unwrap(),
        text.to_str().unwrap(),
    ]));
    assert!(out.contains("keystroke savings"));
}

#[test]
fn reports_errors() {
    let out = augur(&["info", "missing.bin"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).starts_with("error: "));

    let model = tiny_model();
    let out = augur(&[
        "predict",
        "-m",
        model.to_str().unwrap(),
        "--weights",
        "1,2",
        "the ",
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("three values"));
}
