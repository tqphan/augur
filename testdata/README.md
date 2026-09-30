# Test data

`tiny.bin` is a small English model used by the Rust, C and C++ tests. It
is built from `corpus.txt` (a hand-written corpus) and `corpus.casing`
with `augur-train build --min-count 1` followed by `augur-train compile`.

The `testdata_is_current` test in `augur-train` fails when `tiny.bin`
differs from what the current code produces; run it with `AUGUR_BLESS=1`
to regenerate the file:

```sh
AUGUR_BLESS=1 cargo test -p augur-train testdata_is_current
```
