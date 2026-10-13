//! `augur`: word prediction from the command line.

mod eval;

use std::collections::BTreeMap;
use std::error::Error;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use augur::{Context, Engine, EngineOptions, Model};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;

use crate::eval::EvalOptions;

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Print JSON instead of text.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Suggest words for the text before the cursor.
    Predict(PredictArgs),
    /// Show what a model contains.
    Info(InfoArgs),
    /// Measure keystroke savings by simulating typing a text.
    Eval(EvalArgs),
}

#[derive(Args)]
struct EngineArgs {
    /// Binary language model (e.g. en.bin).
    #[arg(short, long)]
    model: PathBuf,
    /// Language, as a BCP 47 tag.
    #[arg(long, default_value = "en")]
    lang: String,
    /// Relative weights of the n-gram, user and recency predictors
    /// (default: 0.6,0.3,0.1).
    #[arg(long, value_delimiter = ',', value_name = "NGRAM,USER,RECENCY")]
    weights: Option<Vec<f32>>,
}

impl EngineArgs {
    fn options(&self) -> Result<EngineOptions> {
        let options = EngineOptions::new();
        match self.weights.as_deref() {
            None => Ok(options),
            Some(&[ngram, user, recency]) => Ok(options.weights(ngram, user, recency)?),
            Some(_) => Err("--weights needs three values: NGRAM,USER,RECENCY".into()),
        }
    }

    fn engine(&self, options: &EngineOptions) -> Result<Engine> {
        Ok(Engine::with_options(&self.lang, &self.model, options)?)
    }
}

#[derive(Args)]
struct PredictArgs {
    #[command(flatten)]
    engine: EngineArgs,
    /// Text before the cursor; a trailing space asks for the next word.
    past: String,
    /// Text after the cursor.
    #[arg(long, default_value = "")]
    future: String,
    /// Number of suggestions.
    #[arg(short = 'n', long, default_value_t = 5)]
    count: usize,
    /// User model file to predict from (read, not changed).
    #[arg(long)]
    user: Option<PathBuf>,
}

#[derive(Args)]
struct InfoArgs {
    /// Binary language model (e.g. en.bin).
    model: PathBuf,
}

#[derive(Args)]
struct EvalArgs {
    #[command(flatten)]
    engine: EngineArgs,
    /// UTF-8 text to type, one document per line; `-` reads standard input.
    text: PathBuf,
    /// Number of suggestions shown at a time.
    #[arg(short = 'n', long, default_value_t = 5)]
    count: usize,
    /// Learn each sentence after typing it (in memory only).
    #[arg(long)]
    learn: bool,
    /// Stop after this many words.
    #[arg(long)]
    max_words: Option<u64>,
}

#[derive(Serialize)]
struct SuggestionOut<'a> {
    text: &'a str,
    prob: f32,
    sources: Vec<&'a str>,
}

fn predict(args: &PredictArgs, json: bool) -> Result<()> {
    let mut options = args.engine.options()?.learning(false);
    if let Some(user) = &args.user {
        options = options.user_store(user);
    }
    let engine = args.engine.engine(&options)?;
    let names = engine.predictor_names();
    let suggestions = engine.predict(Context::new(&args.past, &args.future), args.count);
    let out: Vec<SuggestionOut<'_>> = suggestions
        .iter()
        .map(|s| SuggestionOut {
            text: &s.text,
            prob: s.prob,
            sources: (0..names.len())
                .filter(|&i| s.sources.contains(i))
                .map(|i| names[i])
                .collect(),
        })
        .collect();

    if json {
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }
    let width = out
        .iter()
        .map(|s| s.text.chars().count())
        .max()
        .unwrap_or(0);
    for (rank, s) in out.iter().enumerate() {
        println!(
            "{:>2}  {:<width$}  {:.4}  {}",
            rank + 1,
            s.text,
            s.prob,
            s.sources.join(",")
        );
    }
    Ok(())
}

#[derive(Serialize)]
struct InfoOut {
    file: PathBuf,
    bytes: u64,
    order: usize,
    vocabulary: usize,
    ngrams: Vec<u64>,
    meta: BTreeMap<String, String>,
}

fn info(args: &InfoArgs, json: bool) -> Result<()> {
    let bytes = std::fs::metadata(&args.model)?.len();
    let model = Model::open(&args.model)?;
    let out = InfoOut {
        file: args.model.clone(),
        bytes,
        order: model.order(),
        vocabulary: model.vocab_len(),
        ngrams: model.counts().to_vec(),
        meta: model.meta().iter().cloned().collect(),
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }
    println!("file        {} ({})", out.file.display(), human_size(bytes));
    println!("order       {}", out.order);
    println!("vocabulary  {}", out.vocabulary);
    for (n, count) in (1..).zip(&out.ngrams) {
        println!("{n}-grams     {count}");
    }
    if !out.meta.is_empty() {
        let width = out.meta.keys().map(String::len).max().unwrap_or(0);
        println!("metadata");
        for (key, value) in &out.meta {
            println!("  {key:<width$}  {value}");
        }
    }
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1000.0 && unit + 1 < UNITS.len() {
        size /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{size:.1} {}", UNITS[unit])
    }
}

fn read_text(path: &Path) -> Result<String> {
    if path == Path::new("-") {
        let mut text = String::new();
        std::io::stdin().read_to_string(&mut text)?;
        Ok(text)
    } else {
        std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()).into())
    }
}

fn eval(args: &EvalArgs, json: bool) -> Result<()> {
    if args.count == 0 {
        return Err("--count must be at least 1".into());
    }
    let text = read_text(&args.text)?;
    let engine = args.engine.engine(&args.engine.options()?)?;
    let report = eval::evaluate(
        &engine,
        &text,
        &EvalOptions {
            suggestions: args.count,
            learn: args.learn,
            max_words: args.max_words,
        },
    );

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    let r = &report;
    println!(
        "words              {}{}",
        r.words,
        if r.learn { " (learning)" } else { "" }
    );
    println!(
        "keystrokes         {} without prediction, {} with {} suggestions",
        r.keystrokes_without, r.keystrokes_with, r.suggestions
    );
    println!("keystroke savings  {:.2}%", r.keystroke_savings * 100.0);
    println!(
        "next word          top-1 {:.2}%, top-{} {:.2}%",
        r.next_word_top1 * 100.0,
        r.suggestions,
        r.next_word_top_n * 100.0
    );
    println!(
        "predictions        {}, mean {:.0} µs, p50 {:.0} µs, p99 {:.0} µs",
        r.predictions, r.mean_us, r.p50_us, r.p99_us
    );
    Ok(())
}

fn run(cli: &Cli) -> Result<()> {
    match &cli.command {
        Command::Predict(args) => predict(args, cli.json),
        Command::Info(args) => info(args, cli.json),
        Command::Eval(args) => eval(args, cli.json),
    }
}

fn main() -> ExitCode {
    match run(&Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn tiny_model() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny.bin")
    }

    #[test]
    fn cli_is_consistent() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    #[test]
    fn formats_sizes() {
        assert_eq!(human_size(999), "999 B");
        assert_eq!(human_size(35_200_000), "35.2 MB");
        assert_eq!(human_size(1_500), "1.5 KB");
    }

    #[test]
    fn checks_weights() {
        let args = |weights: Option<Vec<f32>>| EngineArgs {
            model: tiny_model(),
            lang: "en".into(),
            weights,
        };
        assert!(args(None).options().is_ok());
        assert!(args(Some(vec![1.0, 0.0, 0.0])).options().is_ok());
        assert!(args(Some(vec![1.0, 0.0])).options().is_err());
        assert!(args(Some(vec![-1.0, 1.0, 0.0])).options().is_err());
    }
}
