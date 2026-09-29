//! Streams pages out of a MediaWiki XML dump.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;
use std::sync::{Arc, Mutex};

use bzip2::read::MultiBzDecoder;
use quick_xml::Reader;
use quick_xml::escape::resolve_xml_entity;
use quick_xml::events::Event;
use sha1::{Digest, Sha1};

use crate::Result;

/// One page of the dump.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Page {
    /// Page title.
    pub title: String,
    /// Namespace number; articles are 0.
    pub namespace: i64,
    /// Whether the page is a redirect.
    pub redirect: bool,
    /// Wikitext of the (last) revision.
    pub text: String,
}

/// Reads pages from a dump, `.xml` or `.xml.bz2`.
pub struct DumpReader {
    reader: Reader<Box<dyn BufRead + Send>>,
    buf: Vec<u8>,
}

/// SHA-1 of the bytes read so far from a file, shared with its reader.
#[derive(Clone, Default)]
pub struct FileHash(Arc<Mutex<Sha1>>);

impl FileHash {
    /// Hex digest of everything read so far.
    ///
    /// # Panics
    ///
    /// If a reader thread panicked while holding the hash.
    pub fn hex(&self) -> String {
        let digest = self.0.lock().expect("hash lock").clone().finalize();
        digest.iter().fold(String::with_capacity(40), |mut hex, b| {
            let _ = write!(hex, "{b:02x}");
            hex
        })
    }
}

struct HashingReader<R> {
    inner: R,
    hash: FileHash,
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hash.0.lock().expect("hash lock").update(&buf[..n]);
        Ok(n)
    }
}

impl DumpReader {
    /// Opens a dump file, decompressing it if its name ends in `.bz2`.
    /// The returned hash covers the file as stored on disk.
    ///
    /// # Errors
    ///
    /// If the file cannot be opened.
    pub fn open(path: &Path) -> Result<(Self, FileHash)> {
        let hash = FileHash::default();
        let file = HashingReader {
            inner: File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?,
            hash: hash.clone(),
        };
        let file = BufReader::with_capacity(1 << 20, file);
        let compressed = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("bz2"));
        let input: Box<dyn BufRead + Send> = if compressed {
            Box::new(BufReader::with_capacity(1 << 20, MultiBzDecoder::new(file)))
        } else {
            Box::new(file)
        };
        Ok((Self::new(input), hash))
    }

    /// Reads a dump from any buffered input.
    pub fn new(input: Box<dyn BufRead + Send>) -> Self {
        Self {
            reader: Reader::from_reader(input),
            buf: Vec::with_capacity(1 << 16),
        }
    }

    /// Reads the next page, or `None` at the end of the dump.
    ///
    /// # Errors
    ///
    /// On malformed XML or I/O errors.
    pub fn next_page(&mut self) -> Result<Option<Page>> {
        #[derive(PartialEq)]
        enum Field {
            None,
            Title,
            Namespace,
            Text,
        }
        let mut page: Option<Page> = None;
        let mut field = Field::None;
        let mut value = String::new();
        loop {
            self.buf.clear();
            let event = self
                .reader
                .read_event_into(&mut self.buf)
                .map_err(|e| format!("XML error at byte {}: {e}", self.reader.buffer_position()))?;
            match event {
                Event::Start(start) => match (start.local_name().as_ref(), &mut page) {
                    ("page", _) => page = Some(Page::default()),
                    ("title", Some(_)) => field = Field::Title,
                    ("ns", Some(_)) => field = Field::Namespace,
                    ("text", Some(_)) => field = Field::Text,
                    ("redirect", Some(page)) => page.redirect = true,
                    _ => {}
                },
                Event::Empty(start) => {
                    if let ("redirect", Some(page)) = (start.local_name().as_ref(), &mut page) {
                        page.redirect = true;
                    }
                }
                Event::Text(text) if field != Field::None => value.push_str(&text.xml10_content()),
                Event::CData(data) if field != Field::None => {
                    value.push_str(&data.xml10_content());
                }
                Event::GeneralRef(reference) if field != Field::None => {
                    if let Ok(Some(c)) = reference.resolve_char_ref() {
                        value.push(c);
                    } else if let Some(s) = resolve_xml_entity(&reference) {
                        value.push_str(s);
                    }
                }
                Event::End(end) => {
                    let name = end.local_name();
                    let name = name.as_ref();
                    if let Some(page) = page.as_mut() {
                        match (&field, name) {
                            (Field::Title, "title") => page.title = std::mem::take(&mut value),
                            (Field::Namespace, "ns") => {
                                page.namespace = value.trim().parse().unwrap_or(-1);
                                value.clear();
                            }
                            (Field::Text, "text") => page.text = std::mem::take(&mut value),
                            _ => {}
                        }
                    }
                    if matches!(name, "title" | "ns" | "text") {
                        field = Field::None;
                    }
                    if name == "page" && page.is_some() {
                        return Ok(page);
                    }
                }
                Event::Eof => return Ok(None),
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DUMP: &str = r#"<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <siteinfo><sitename>Wikipedia</sitename></siteinfo>
  <page>
    <title>Cat &amp; Dog</title>
    <ns>0</ns>
    <revision><id>1</id><text bytes="10" xml:space="preserve">A &lt;b&gt; &#233;
line</text></revision>
  </page>
  <page>
    <title>Kitty</title>
    <ns>0</ns>
    <redirect title="Cat" />
    <revision><text>#REDIRECT [[Cat]]</text></revision>
  </page>
  <page>
    <title>Talk:Cat</title>
    <ns>1</ns>
    <revision><text /></revision>
  </page>
</mediawiki>"#;

    fn pages(input: &'static [u8]) -> Vec<Page> {
        let mut reader = DumpReader::new(Box::new(input));
        let mut pages = Vec::new();
        while let Some(page) = reader.next_page().unwrap() {
            pages.push(page);
        }
        pages
    }

    #[test]
    fn reads_pages() {
        let pages = pages(DUMP.as_bytes());
        assert_eq!(pages.len(), 3);
        assert_eq!(pages[0].title, "Cat & Dog");
        assert_eq!(pages[0].namespace, 0);
        assert!(!pages[0].redirect);
        assert_eq!(pages[0].text, "A <b> \u{e9}\nline");
        assert!(pages[1].redirect);
        assert_eq!(pages[2].namespace, 1);
        assert_eq!(pages[2].text, "");
    }
}
