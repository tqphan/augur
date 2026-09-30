//! `augur-train`: builds augur language models from text corpora.

use std::path::PathBuf;
use std::process::ExitCode;

use augur_train::Result;
use augur_train::build::{self, BuildOptions, MAX_ORDER, Smoothing};
use augur_train::compile::{self, CompileOptions};
use augur_train::extract::{self, ExtractOptions};
use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Worker threads (default: one per CPU).
    #[arg(long, global = true)]
    threads: Option<usize>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Extract a tokenized corpus from a MediaWiki XML dump.
    Extract(ExtractArgs),
    /// Build an n-gram model (ARPA) from a tokenized corpus.
    Build(BuildArgs),
    /// Convert an ARPA model to augur's binary format.
    Compile(CompileArgs),
}

#[derive(Args)]
struct CompileArgs {
    /// ARPA model; `.casing` and `.meta` files next to it are packed in if
    /// they exist.
    arpa: PathBuf,
    /// Output binary model (e.g. en.bin).
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Args)]
struct ExtractArgs {
    /// Dump file (e.g. simplewiki-YYYYMMDD-pages-articles.xml.bz2).
    dump: PathBuf,
    /// Output corpus; `.casing` and `.meta` files are written next to it.
    #[arg(short, long)]
    output: PathBuf,
    /// Language of the dump.
    #[arg(long, default_value = "en")]
    language: String,
    /// Drop sentences with fewer words.
    #[arg(long, default_value_t = 3)]
    min_words: usize,
}

#[derive(Args)]
struct BuildArgs {
    /// Corpus with one tokenized sentence per line.
    corpus: PathBuf,
    /// Output ARPA file; `.casing` and `.meta` files are written next to it.
    #[arg(short, long)]
    output: PathBuf,
    /// N-gram order.
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u8).range(1..=MAX_ORDER as i64))]
    order: u8,
    /// Smoothing method.
    #[arg(long, value_enum, default_value_t = Smoothing::Mkn)]
    smoothing: Smoothing,
    /// Vocabulary size, not counting <s>, </s> and <unk>.
    #[arg(long, default_value_t = 100_000)]
    vocab: usize,
    /// Minimum count per order, comma-separated from unigrams; the last
    /// value repeats for higher orders. Unigrams must be 1.
    #[arg(long, value_delimiter = ',', default_value = "1,2,2")]
    min_count: Vec<u64>,
    /// Back-off factor for Stupid Backoff.
    #[arg(long, default_value_t = 0.4)]
    alpha: f64,
    /// Approximate memory budget, e.g. 512M or 4G.
    #[arg(long, default_value = "1G", value_parser = parse_size)]
    mem: usize,
    /// Directory for temporary files (default: system temp directory).
    #[arg(long)]
    tmp_dir: Option<PathBuf>,
}

fn parse_size(text: &str) -> Result<usize, String> {
    let text = text.trim();
    let (digits, shift) = match text.chars().last().map(|c| c.to_ascii_uppercase()) {
        Some('K') => (&text[..text.len() - 1], 10),
        Some('M') => (&text[..text.len() - 1], 20),
        Some('G') => (&text[..text.len() - 1], 30),
        _ => (text, 0),
    };
    let value: usize = digits
        .trim()
        .parse()
        .map_err(|_| format!("invalid size {text:?}"))?;
    value
        .checked_mul(1 << shift)
        .ok_or_else(|| format!("size {text:?} is too large"))
}

/// Expands `--min-count` to one value per order and validates it.
fn min_counts(values: &[u64], order: usize) -> Result<Vec<u64>> {
    let last = *values
        .last()
        .ok_or("--min-count needs at least one value")?;
    let mut counts: Vec<u64> = values.iter().copied().take(order).collect();
    counts.resize(order, last);
    if counts[0] != 1 {
        return Err("--min-count for unigrams must be 1 (use --vocab to limit unigrams)".into());
    }
    if counts.windows(2).any(|w| w[1] < w[0]) {
        return Err("--min-count values must not decrease with the order".into());
    }
    Ok(counts)
}

fn run(cli: Cli) -> Result<()> {
    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?;
    }
    match cli.command {
        Command::Extract(args) => extract::run(&ExtractOptions {
            dump: args.dump,
            output: args.output,
            language: args.language,
            min_words: args.min_words,
        }),
        Command::Build(args) => build::run(&BuildOptions {
            corpus: args.corpus,
            output: args.output,
            smoothing: args.smoothing,
            min_count: min_counts(&args.min_count, usize::from(args.order))?,
            vocab_size: args.vocab,
            alpha: args.alpha,
            memory: args.mem,
            tmp_dir: args.tmp_dir.unwrap_or_else(std::env::temp_dir),
        }),
        Command::Compile(args) => compile::run(&CompileOptions {
            arpa: args.arpa,
            output: args.output,
        }),
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
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

    #[test]
    fn parses_sizes() {
        assert_eq!(parse_size("512"), Ok(512));
        assert_eq!(parse_size("2k"), Ok(2048));
        assert_eq!(parse_size("1G"), Ok(1 << 30));
        assert!(parse_size("x").is_err());
    }

    #[test]
    fn expands_min_counts() {
        assert_eq!(min_counts(&[1, 2], 4).unwrap(), [1, 2, 2, 2]);
        assert_eq!(min_counts(&[1, 2, 3], 2).unwrap(), [1, 2]);
        assert!(min_counts(&[2], 3).is_err());
        assert!(min_counts(&[1, 3, 2], 3).is_err());
    }

    #[test]
    fn cli_is_consistent() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
