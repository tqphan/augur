//! `augur-train extract`: turns a Wikipedia dump into a tokenized corpus.
//!
//! Outputs, next to `--output` (e.g. `corpus.txt`):
//!
//! - `corpus.txt`: one sentence per line, normalized tokens separated by
//!   spaces;
//! - `corpus.casing`: `word<TAB>Form` for words usually written with
//!   capitals inside a sentence (e.g. `london<TAB>London`);
//! - `corpus.meta`: provenance (source file, SHA-1, license, attribution)
//!   and statistics.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::sync_channel;
use std::thread;

use augur_core::{LanguagePack, Token};
use augur_lang_en::EnglishPack;
use augur_text::canonicalize;
use rayon::prelude::*;

use crate::Result;
use crate::dump::{DumpReader, Page};
use crate::meta::Meta;
use crate::wikitext;

/// Pages sent to the worker threads at a time.
const BATCH: usize = 64;

/// Settings for [`run`].
#[derive(Debug, Clone)]
pub struct ExtractOptions {
    /// MediaWiki XML dump, `.xml` or `.xml.bz2`.
    pub dump: PathBuf,
    /// Output corpus.
    pub output: PathBuf,
    /// Language tag of the dump.
    pub language: String,
    /// Sentences with fewer words are dropped.
    pub min_words: usize,
}

/// Returns the language pack for a language tag.
///
/// # Errors
///
/// If the language is not supported.
pub fn language_pack(tag: &str) -> Result<Box<dyn LanguagePack>> {
    match tag {
        "en" => Ok(Box::new(EnglishPack::new())),
        _ => Err(format!("unsupported language {tag:?} (supported: en)").into()),
    }
}

/// How often a word appears inside sentences, and in which capitalized
/// forms.
#[derive(Debug, Default)]
struct CaseCounts {
    total: u64,
    forms: HashMap<String, u64>,
}

/// The processed text of one article.
#[derive(Debug, Default)]
struct Article {
    lines: String,
    sentences: u64,
    tokens: u64,
    /// Every word inside a sentence (not first), with its written form if
    /// that differs from the normalized word.
    inner_words: Vec<(String, Option<String>)>,
}

fn process(page: &Page, pack: &dyn LanguagePack, min_words: usize) -> Option<Article> {
    if wikitext::is_disambiguation(&page.text) {
        return None;
    }
    let text = wikitext::to_plain_text(&page.text);
    let spans = pack.tokenizer().tokenize_spans(&text);
    let mut article = Article::default();
    for sentence in spans.split(|span| span.token == Token::SentenceStart) {
        if sentence.len() < min_words.max(1) {
            continue;
        }
        article.sentences += 1;
        for (index, span) in sentence.iter().enumerate() {
            let Token::Word(word) = &span.token else {
                continue;
            };
            article.tokens += 1;
            if index > 0 {
                article.lines.push(' ');
            }
            article.lines.push_str(word);
            if index > 0 && word != augur_text::NUMBER {
                let written = canonicalize(&text[span.span.clone()]);
                let form = (written != word.as_str()).then(|| written.into_owned());
                article.inner_words.push((word.clone(), form));
            }
        }
        article.lines.push('\n');
    }
    Some(article)
}

/// Statistics of a dump pass.
#[derive(Debug, Default)]
struct DumpStats {
    pages: u64,
    articles: u64,
}

/// Runs the extraction.
///
/// # Errors
///
/// On unreadable input, malformed XML or I/O errors.
pub fn run(options: &ExtractOptions) -> Result<()> {
    let pack = language_pack(&options.language)?;
    let (mut reader, hash) = DumpReader::open(&options.dump)?;
    let source_bytes = std::fs::metadata(&options.dump)?.len();

    let (sender, receiver) = sync_channel::<Result<Vec<Page>>>(4);
    let producer = thread::spawn(move || {
        let mut stats = DumpStats::default();
        let mut batch = Vec::with_capacity(BATCH);
        loop {
            match reader.next_page() {
                Ok(Some(page)) => {
                    stats.pages += 1;
                    if page.namespace != 0 || page.redirect {
                        continue;
                    }
                    stats.articles += 1;
                    batch.push(page);
                    if batch.len() == BATCH && sender.send(Ok(std::mem::take(&mut batch))).is_err()
                    {
                        break;
                    }
                }
                Ok(None) => {
                    let _ = sender.send(Ok(batch));
                    break;
                }
                Err(e) => {
                    let _ = sender.send(Err(e));
                    break;
                }
            }
        }
        stats
    });

    let mut corpus = BufWriter::with_capacity(1 << 20, File::create(&options.output)?);
    let mut casing: HashMap<String, CaseCounts> = HashMap::new();
    let (mut sentences, mut tokens, mut used, mut done) = (0u64, 0u64, 0u64, 0u64);
    for batch in receiver {
        let batch = batch?;
        let articles: Vec<Option<Article>> = batch
            .par_iter()
            .map(|page| process(page, pack.as_ref(), options.min_words))
            .collect();
        done += batch.len() as u64;
        for article in articles.into_iter().flatten() {
            used += 1;
            sentences += article.sentences;
            tokens += article.tokens;
            corpus.write_all(article.lines.as_bytes())?;
            for (word, form) in article.inner_words {
                let counts = casing.entry(word).or_default();
                counts.total += 1;
                if let Some(form) = form {
                    *counts.forms.entry(form).or_default() += 1;
                }
            }
        }
        if done >= 10_000 && done % 10_000 < BATCH as u64 {
            eprint!("\r{done} articles, {sentences} sentences, {tokens} tokens");
        }
    }
    eprintln!("\r{done} articles, {sentences} sentences, {tokens} tokens");
    corpus.flush()?;
    let stats = producer.join().map_err(|_| "dump reader thread panicked")?;

    let casing_entries = write_casing(&casing, &options.output.with_extension("casing"))?;

    let mut meta = Meta::new();
    meta.set("tool", concat!("augur-train ", env!("CARGO_PKG_VERSION")));
    meta.set("language", &options.language);
    describe_source(&mut meta, &options.dump, source_bytes, &hash.hex());
    meta.set("extract.min_words", options.min_words);
    meta.set("extract.pages", stats.pages);
    meta.set("extract.articles", stats.articles);
    meta.set("extract.articles_used", used);
    meta.set("extract.sentences", sentences);
    meta.set("extract.tokens", tokens);
    meta.set("extract.casing_entries", casing_entries);
    meta.write(&options.output.with_extension("meta"))?;
    Ok(())
}

/// Writes `word<TAB>Form` for words whose most common form inside
/// sentences is capitalized in more than half of their occurrences.
fn write_casing(casing: &HashMap<String, CaseCounts>, path: &Path) -> Result<u64> {
    let mut entries: Vec<(&str, &str)> = casing
        .iter()
        .filter_map(|(word, counts)| {
            let (form, &count) = counts
                .forms
                .iter()
                .max_by(|(fa, ca), (fb, cb)| ca.cmp(cb).then_with(|| fb.cmp(fa)))?;
            (count * 2 > counts.total).then_some((word.as_str(), form.as_str()))
        })
        .collect();
    entries.sort_unstable();
    let mut out = BufWriter::new(File::create(path)?);
    for (word, form) in &entries {
        writeln!(out, "{word}\t{form}")?;
    }
    out.flush()?;
    Ok(entries.len() as u64)
}

/// Records the source file and, for standard Wikimedia dump names such as
/// `simplewiki-20260901-pages-articles.xml.bz2`, the wiki, dump date,
/// license and attribution.
fn describe_source(meta: &mut Meta, dump: &Path, bytes: u64, sha1: &str) {
    let name = dump
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    meta.set("source.file", &name);
    meta.set("source.bytes", bytes);
    meta.set("source.sha1", sha1);
    let mut parts = name.split('-');
    let (Some(wiki), Some(date)) = (parts.next(), parts.next()) else {
        return;
    };
    let Some(language) = wiki.strip_suffix("wiki") else {
        return;
    };
    if language.is_empty() || date.len() != 8 || !date.bytes().all(|b| b.is_ascii_digit()) {
        return;
    }
    let domain = format!("{language}.wikipedia.org");
    meta.set("source.wiki", wiki);
    meta.set("source.dump_date", date);
    meta.set(
        "source.url",
        format!("https://dumps.wikimedia.org/{wiki}/{date}/{name}"),
    );
    meta.set("license", "CC-BY-SA-4.0");
    meta.set(
        "attribution",
        format!(
            "Derived from Wikipedia (https://{domain}) by Wikipedia contributors, dump {date}, licensed under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)"
        ),
    );
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::sort::TempDir;

    const DUMP: &str = r"<mediawiki>
  <page><title>London</title><ns>0</ns><revision><text>'''London''' is the capital of [[England]]. It has 9 million people.
Many people visit London every year.
== References ==
{{reflist}}</text></revision></page>
  <page><title>Talk:London</title><ns>1</ns><revision><text>Talk page text here.</text></revision></page>
  <page><title>Londres</title><ns>0</ns><redirect title='London'/><revision><text>#REDIRECT [[London]]</text></revision></page>
  <page><title>Mercury</title><ns>0</ns><revision><text>'''Mercury''' may mean:
{{disambiguation}}</text></revision></page>
  <page><title>Rain</title><ns>0</ns><revision><text>Rain falls in London. Short one.</text></revision></page>
</mediawiki>";

    #[test]
    fn extracts_corpus_casing_and_meta() {
        let tmp = TempDir::new(&std::env::temp_dir()).unwrap();
        let dump = tmp
            .file("simplewiki-20260901-pages-articles")
            .with_extension("xml");
        fs::write(&dump, DUMP).unwrap();
        let output = tmp.file("corpus").with_extension("txt");
        run(&ExtractOptions {
            dump: dump.clone(),
            output: output.clone(),
            language: "en".into(),
            min_words: 3,
        })
        .unwrap();

        assert_eq!(
            fs::read_to_string(&output).unwrap(),
            "london is the capital of england\n\
             it has <num> million people\n\
             many people visit london every year\n\
             rain falls in london\n"
        );
        assert_eq!(
            fs::read_to_string(output.with_extension("casing")).unwrap(),
            "england\tEngland\nlondon\tLondon\n"
        );
        let meta = Meta::read_if_exists(&output.with_extension("meta")).unwrap();
        assert_eq!(meta.get("extract.pages"), Some("5"));
        assert_eq!(meta.get("extract.articles"), Some("3"));
        assert_eq!(meta.get("extract.articles_used"), Some("2"));
        assert_eq!(meta.get("extract.sentences"), Some("4"));
        assert_eq!(
            meta.get("source.bytes"),
            Some(DUMP.len().to_string().as_str())
        );
        assert_eq!(meta.get("source.sha1").map(str::len), Some(40));
        assert_eq!(
            Path::new(meta.get("source.file").unwrap()).extension(),
            Some("xml".as_ref())
        );
    }

    #[test]
    fn describes_wikimedia_sources() {
        let mut meta = Meta::new();
        describe_source(
            &mut meta,
            Path::new("dumps/simplewiki-20260901-pages-articles.xml.bz2"),
            10,
            "abc",
        );
        assert_eq!(meta.get("source.wiki"), Some("simplewiki"));
        assert_eq!(meta.get("source.dump_date"), Some("20260901"));
        assert_eq!(
            meta.get("source.url"),
            Some(
                "https://dumps.wikimedia.org/simplewiki/20260901/simplewiki-20260901-pages-articles.xml.bz2"
            )
        );
        assert!(
            meta.get("attribution")
                .unwrap()
                .contains("https://simple.wikipedia.org")
        );
        assert_eq!(meta.get("license"), Some("CC-BY-SA-4.0"));
    }
}
