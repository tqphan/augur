//! Writes ARPA back-off language model files.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use super::count::Vocab;
use crate::Result;
use crate::sort::TempDir;

/// Log probability written for impossible events such as predicting `<s>`.
const LOG_ZERO: f64 = -99.0;

/// Collects the n-gram sections in temporary files, then assembles them.
pub(super) struct ArpaWriter {
    sections: Vec<Section>,
}

struct Section {
    path: PathBuf,
    file: BufWriter<File>,
    entries: u64,
}

impl ArpaWriter {
    /// Creates a writer for a model of the given order.
    ///
    /// # Errors
    ///
    /// If temporary files cannot be created.
    pub(super) fn new(order: usize, tmp: &TempDir) -> Result<Self> {
        let sections = (0..order)
            .map(|_| -> Result<Section> {
                let path = tmp.file("section");
                Ok(Section {
                    file: BufWriter::with_capacity(1 << 20, File::create(&path)?),
                    path,
                    entries: 0,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self { sections })
    }

    /// Writes one n-gram with its probability and optional back-off weight
    /// (both as plain, not log, values). A probability of 0 is written as
    /// `-99`.
    ///
    /// # Errors
    ///
    /// On write errors.
    pub(super) fn entry(
        &mut self,
        vocab: &Vocab,
        ids: &[u32],
        prob: f64,
        backoff: Option<f64>,
    ) -> Result<()> {
        let section = &mut self.sections[ids.len() - 1];
        section.entries += 1;
        let file = &mut section.file;
        write_log(file, prob)?;
        file.write_all(b"\t")?;
        for (i, &id) in ids.iter().enumerate() {
            if i > 0 {
                file.write_all(b" ")?;
            }
            file.write_all(vocab.word(id).as_bytes())?;
        }
        if let Some(backoff) = backoff {
            file.write_all(b"\t")?;
            write_log(file, backoff)?;
        }
        file.write_all(b"\n")?;
        Ok(())
    }

    /// Number of entries written for each order.
    pub(super) fn counts(&self) -> Vec<u64> {
        self.sections.iter().map(|s| s.entries).collect()
    }

    /// Writes the complete ARPA file to `path`.
    ///
    /// # Errors
    ///
    /// On I/O errors.
    pub(super) fn finish(self, path: &Path) -> Result<Vec<u64>> {
        let counts = self.counts();
        let mut out = BufWriter::with_capacity(1 << 20, File::create(path)?);
        writeln!(out, "\\data\\")?;
        for (n, count) in counts.iter().enumerate() {
            writeln!(out, "ngram {}={count}", n + 1)?;
        }
        for (n, mut section) in self.sections.into_iter().enumerate() {
            section.file.flush()?;
            drop(section.file);
            writeln!(out, "\n\\{}-grams:", n + 1)?;
            io::copy(&mut BufReader::new(File::open(&section.path)?), &mut out)?;
        }
        writeln!(out, "\n\\end\\")?;
        out.flush()?;
        Ok(counts)
    }
}

fn write_log(out: &mut impl Write, value: f64) -> io::Result<()> {
    if value > 0.0 {
        write!(out, "{:.6}", value.log10())
    } else {
        write!(out, "{LOG_ZERO}")
    }
}
