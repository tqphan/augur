//! External merge sort of fixed-size binary records.
//!
//! Records are byte strings of a fixed length whose first `key_len` bytes are
//! the sort key, compared bytewise. Word ids are stored big-endian so that
//! bytewise order equals numeric order. Records that do not fit in the memory
//! budget are sorted in runs, spilled to disk and merged.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::slice::ParallelSliceMut;

use crate::Result;

/// Folds the payload of `other` into `acc`; both have the same key.
pub type Combine = fn(acc: &mut [u8], other: &[u8]);

/// A temporary directory removed when dropped.
#[derive(Debug)]
pub struct TempDir {
    path: PathBuf,
    next_file: AtomicU64,
}

impl TempDir {
    /// Creates a fresh directory inside `parent`.
    ///
    /// # Errors
    ///
    /// If the directory cannot be created.
    pub fn new(parent: &Path) -> Result<Self> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        let path = parent.join(format!("augur-train-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&path).map_err(|e| format!("cannot create {}: {e}", path.display()))?;
        Ok(Self {
            path,
            next_file: AtomicU64::new(0),
        })
    }

    /// A new, unused file path in the directory.
    pub fn file(&self, name: &str) -> PathBuf {
        let n = self.next_file.fetch_add(1, Ordering::Relaxed);
        self.path.join(format!("{name}-{n}.tmp"))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// Collects records and returns them sorted by key.
pub struct Sorter<'t> {
    tmp: &'t TempDir,
    name: &'static str,
    record_len: usize,
    key_len: usize,
    combine: Option<Combine>,
    buffer: Vec<u8>,
    max_buffer: usize,
    runs: Vec<PathBuf>,
}

impl<'t> Sorter<'t> {
    /// Creates a sorter that keeps about `memory` bytes in memory.
    ///
    /// # Panics
    ///
    /// If `record_len` is 0 or smaller than `key_len`.
    pub fn new(
        tmp: &'t TempDir,
        name: &'static str,
        record_len: usize,
        key_len: usize,
        memory: usize,
        combine: Option<Combine>,
    ) -> Self {
        assert!(key_len <= record_len && record_len > 0);
        // Sorting a run needs, per record, the buffered record, a 16-byte
        // slice reference and a copy in the sorted output.
        let records = (memory / (2 * record_len + 16)).max(1024);
        Self {
            tmp,
            name,
            record_len,
            key_len,
            combine,
            buffer: Vec::new(),
            max_buffer: records * record_len,
            runs: Vec::new(),
        }
    }

    /// Adds a record.
    ///
    /// # Errors
    ///
    /// If a run cannot be written to disk.
    pub fn push(&mut self, record: &[u8]) -> Result<()> {
        debug_assert_eq!(record.len(), self.record_len);
        self.buffer.extend_from_slice(record);
        if self.buffer.len() >= self.max_buffer {
            self.spill()?;
        }
        Ok(())
    }

    /// Sorts `buffer` by key and combines records with equal keys.
    fn sort_buffer(&self, buffer: &[u8]) -> Vec<u8> {
        let key_len = self.key_len;
        let mut records: Vec<&[u8]> = buffer.chunks_exact(self.record_len).collect();
        records.par_sort_unstable_by(|a, b| a[..key_len].cmp(&b[..key_len]));
        let mut sorted: Vec<u8> = Vec::with_capacity(buffer.len());
        let mut last_start: Option<usize> = None;
        for record in records {
            if let (Some(combine), Some(start)) = (self.combine, last_start)
                && sorted[start..start + key_len] == record[..key_len]
            {
                combine(&mut sorted[start..], record);
                continue;
            }
            last_start = Some(sorted.len());
            sorted.extend_from_slice(record);
        }
        sorted
    }

    fn spill(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let sorted = self.sort_buffer(&self.buffer);
        self.buffer.clear();
        let path = self.tmp.file(self.name);
        let mut file = BufWriter::with_capacity(1 << 20, File::create(&path)?);
        file.write_all(&sorted)?;
        file.flush()?;
        self.runs.push(path);
        Ok(())
    }

    /// Finishes collecting and returns the records in key order.
    ///
    /// # Errors
    ///
    /// If runs cannot be written or reopened.
    pub fn finish(mut self) -> Result<Sorted> {
        if self.runs.is_empty() {
            let sorted = self.sort_buffer(&self.buffer);
            return Ok(Sorted {
                record_len: self.record_len,
                key_len: self.key_len,
                combine: self.combine,
                source: Source::Memory {
                    data: sorted,
                    pos: 0,
                },
                current: vec![0; self.record_len],
            });
        }
        self.spill()?;
        let mut runs = Vec::with_capacity(self.runs.len());
        for path in &self.runs {
            let mut run = Run {
                reader: BufReader::with_capacity(1 << 16, File::open(path)?),
                record: vec![0; self.record_len],
                done: false,
            };
            run.advance()?;
            runs.push(run);
        }
        Ok(Sorted {
            record_len: self.record_len,
            key_len: self.key_len,
            combine: self.combine,
            source: Source::Runs(runs),
            current: vec![0; self.record_len],
        })
    }
}

struct Run {
    reader: BufReader<File>,
    record: Vec<u8>,
    done: bool,
}

impl Run {
    fn advance(&mut self) -> Result<()> {
        match self.reader.read_exact(&mut self.record) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
                self.done = true;
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }
}

enum Source {
    Memory { data: Vec<u8>, pos: usize },
    Runs(Vec<Run>),
}

/// Sorted records, read one at a time.
pub struct Sorted {
    record_len: usize,
    key_len: usize,
    combine: Option<Combine>,
    source: Source,
    current: Vec<u8>,
}

impl Sorted {
    /// The next record in key order, or `None` at the end.
    ///
    /// # Errors
    ///
    /// If a run cannot be read.
    pub fn next_record(&mut self) -> Result<Option<&[u8]>> {
        let key_len = self.key_len;
        match &mut self.source {
            Source::Memory { data, pos } => {
                if *pos >= data.len() {
                    return Ok(None);
                }
                let record = &data[*pos..*pos + self.record_len];
                *pos += self.record_len;
                Ok(Some(record))
            }
            Source::Runs(runs) => {
                let Some(first) = min_run(runs, key_len) else {
                    return Ok(None);
                };
                self.current.copy_from_slice(&runs[first].record);
                runs[first].advance()?;
                if let Some(combine) = self.combine {
                    while let Some(next) = min_run(runs, key_len) {
                        if runs[next].record[..key_len] != self.current[..key_len] {
                            break;
                        }
                        combine(&mut self.current, &runs[next].record);
                        runs[next].advance()?;
                    }
                }
                Ok(Some(&self.current))
            }
        }
    }
}

/// Index of the run with the smallest current key.
fn min_run(runs: &[Run], key_len: usize) -> Option<usize> {
    runs.iter()
        .enumerate()
        .filter(|(_, run)| !run.done)
        .min_by(|(_, a), (_, b)| a.record[..key_len].cmp(&b.record[..key_len]))
        .map(|(index, _)| index)
}

/// Adds the little-endian `u64` at `offset` of `other` into `acc`.
pub fn add_u64_at<const OFFSET: usize>(acc: &mut [u8], other: &[u8]) {
    let a = read_u64(&acc[OFFSET..]);
    let b = read_u64(&other[OFFSET..]);
    acc[OFFSET..OFFSET + 8].copy_from_slice(&(a + b).to_le_bytes());
}

/// Reads a little-endian `u64` from the start of `bytes`.
///
/// # Panics
///
/// If `bytes` is shorter than 8 bytes.
pub fn read_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes"))
}

/// Reads a little-endian `f64` from the start of `bytes`.
///
/// # Panics
///
/// If `bytes` is shorter than 8 bytes.
pub fn read_f64(bytes: &[u8]) -> f64 {
    f64::from_le_bytes(bytes[..8].try_into().expect("8 bytes"))
}

/// Reads a big-endian word id from the start of `bytes`.
///
/// # Panics
///
/// If `bytes` is shorter than 4 bytes.
pub fn read_id(bytes: &[u8]) -> u32 {
    u32::from_be_bytes(bytes[..4].try_into().expect("4 bytes"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(key: u32, count: u64) -> [u8; 12] {
        let mut r = [0; 12];
        r[..4].copy_from_slice(&key.to_be_bytes());
        r[4..].copy_from_slice(&count.to_le_bytes());
        r
    }

    fn collect(mut sorted: Sorted) -> Vec<(u32, u64)> {
        let mut out = Vec::new();
        while let Some(r) = sorted.next_record().unwrap() {
            out.push((read_id(r), read_u64(&r[4..])));
        }
        out
    }

    fn run(memory: usize) {
        let tmp = TempDir::new(&std::env::temp_dir()).unwrap();
        let mut sorter = Sorter::new(&tmp, "test", 12, 4, memory, Some(add_u64_at::<4>));
        // 5000 records over 100 keys, in a scrambled order.
        for i in 0..5000u32 {
            sorter.push(&record((i * 37) % 100, 1)).unwrap();
        }
        let spilled = !sorter.runs.is_empty();
        let out = collect(sorter.finish().unwrap());
        assert_eq!(out.len(), 100);
        assert!(
            out.iter()
                .enumerate()
                .all(|(i, &(k, c))| usize::try_from(k).unwrap() == i && c == 50)
        );
        assert_eq!(spilled, memory < 5000 * 40);
    }

    #[test]
    fn sorts_and_combines_in_memory() {
        run(1 << 20);
    }

    #[test]
    fn sorts_and_combines_across_runs() {
        run(0); // minimum run size of 1024 records forces spilling
    }

    #[test]
    fn temp_dir_is_removed() {
        let tmp = TempDir::new(&std::env::temp_dir()).unwrap();
        let path = tmp.path.clone();
        File::create(tmp.file("x")).unwrap();
        drop(tmp);
        assert!(!path.exists());
    }
}
