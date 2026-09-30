//! `augur-train compile`: converts an ARPA model, with its `.casing` and
//! `.meta` sidecar files, into augur's binary model format.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::path::PathBuf;

use augur_lm::{Arpa, write_model};

use crate::Result;
use crate::meta::Meta;

/// Settings for [`run`].
#[derive(Debug, Clone)]
pub struct CompileOptions {
    /// ARPA file; `.casing` and `.meta` files next to it are used if they
    /// exist.
    pub arpa: PathBuf,
    /// Output binary model.
    pub output: PathBuf,
}

/// Compiles the model.
///
/// # Errors
///
/// On I/O errors or an invalid ARPA file.
pub fn run(options: &CompileOptions) -> Result<()> {
    eprintln!("Reading {}", options.arpa.display());
    let arpa = Arpa::read(BufReader::with_capacity(
        1 << 20,
        File::open(&options.arpa)?,
    ))?;

    let casing_path = options.arpa.with_extension("casing");
    let mut casing = Vec::new();
    if casing_path.exists() {
        for line in BufReader::new(File::open(&casing_path)?).lines() {
            if let Some((word, form)) = line?.split_once('\t') {
                casing.push((word.to_owned(), form.to_owned()));
            }
        }
    }

    let mut meta = Meta::read_if_exists(&options.arpa.with_extension("meta"))?;
    meta.set("model.format", "augur");
    meta.set("model.quantization", "8-bit");
    if let Some(name) = options.arpa.file_name() {
        meta.set("compile.source", name.to_string_lossy());
    }

    let mut out = BufWriter::new(File::create(&options.output)?);
    let stats = write_model(&arpa, meta.entries(), &casing, &mut out)?;
    eprintln!(
        "Wrote {} ({} bytes, {:?} n-grams, {} casing entries)",
        options.output.display(),
        stats.bytes,
        stats.counts,
        stats.casing_entries
    );
    if stats.casing_skipped > 0 {
        eprintln!(
            "{} casing entries skipped (not in the vocabulary)",
            stats.casing_skipped
        );
    }
    eprintln!(
        "Mean quantization error (log10): probabilities {:?}, back-offs {:?}",
        stats.prob_error, stats.backoff_error
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use augur_lm::Model;

    use super::*;
    use crate::sort::TempDir;

    #[test]
    fn compiles_with_sidecars() {
        let tmp = TempDir::new(&std::env::temp_dir()).unwrap();
        let arpa = tmp.file("model").with_extension("arpa");
        fs::write(
            &arpa,
            "\\data\\\nngram 1=3\nngram 2=1\n\n\\1-grams:\n-99\t<s>\t-0.1\n-0.5\tlondon\t-0.2\n\
             -0.4\tis\t0\n\n\\2-grams:\n-0.3\t<s> london\n\n\\end\\\n",
        )
        .unwrap();
        fs::write(
            arpa.with_extension("casing"),
            "london\tLondon\nparis\tParis\n",
        )
        .unwrap();
        fs::write(
            arpa.with_extension("meta"),
            "source.license = CC BY-SA 4.0\nmodel.format = arpa\n",
        )
        .unwrap();
        let output = tmp.file("model").with_extension("bin");

        run(&CompileOptions {
            arpa: arpa.clone(),
            output: output.clone(),
        })
        .unwrap();

        let model = Model::open(&output).unwrap();
        assert_eq!(model.counts(), [3, 1]);
        assert_eq!(model.casing(model.id("london").unwrap()), Some("London"));
        assert_eq!(model.meta_value("source.license"), Some("CC BY-SA 4.0"));
        assert_eq!(model.meta_value("model.format"), Some("augur"));
        assert_eq!(
            model.meta_value("compile.source"),
            Some(arpa.file_name().unwrap().to_str().unwrap())
        );
    }
}
