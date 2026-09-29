//! Provenance sidecar files (`*.meta`): `key = value` lines with `#`
//! comments, in insertion order.

use std::fmt::Display;
use std::fs;
use std::path::Path;

use crate::Result;

/// Ordered `key = value` metadata.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Meta {
    entries: Vec<(String, String)>,
}

impl Meta {
    /// Empty metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads a metadata file, or returns empty metadata if it does not
    /// exist.
    ///
    /// # Errors
    ///
    /// If the file exists but cannot be read.
    pub fn read_if_exists(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }
        Ok(Self::parse(&fs::read_to_string(path)?))
    }

    fn parse(text: &str) -> Self {
        let entries = text
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .filter_map(|line| line.split_once(" = "))
            .map(|(key, value)| (key.trim().to_owned(), value.trim().to_owned()))
            .collect();
        Self { entries }
    }

    /// Sets `key`, replacing an existing value.
    pub fn set(&mut self, key: &str, value: impl Display) {
        let value = value.to_string().replace(['\n', '\r'], " ");
        match self.entries.iter_mut().find(|(k, _)| k == key) {
            Some(entry) => entry.1 = value,
            None => self.entries.push((key.to_owned(), value)),
        }
    }

    /// The value of `key`.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Writes the metadata to `path`.
    ///
    /// # Errors
    ///
    /// On I/O errors.
    pub fn write(&self, path: &Path) -> Result<()> {
        let mut text = String::from("# augur-train provenance and settings\n");
        for (key, value) in &self.entries {
            text.push_str(key);
            text.push_str(" = ");
            text.push_str(value);
            text.push('\n');
        }
        fs::write(path, text)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let mut meta = Meta::new();
        meta.set("a", 1);
        meta.set("b.c", "x = y");
        meta.set("a", "two\nlines");
        let path = std::env::temp_dir().join(format!("augur-meta-{}.meta", std::process::id()));
        meta.write(&path).unwrap();
        let read = Meta::read_if_exists(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(read, meta);
        assert_eq!(read.get("a"), Some("two lines"));
        assert_eq!(read.get("b.c"), Some("x = y"));
    }
}
