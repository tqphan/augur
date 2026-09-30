//! C API for augur.
//!
//! The header `include/augur.h` is generated from this crate by cbindgen at
//! build time. Conventions for every exported function:
//!
//! - Names are prefixed with `augur_`.
//! - Objects are opaque handles created by `augur_*_new` (or returned by a
//!   call such as `augur_predict`) and released by the matching
//!   `augur_*_free`, which accepts NULL.
//! - Fallible functions return an [`augur_status`]; on failure,
//!   `augur_last_error()` describes the problem. Panics never cross the FFI
//!   boundary.
//! - Strings are UTF-8, passed as pointer plus length in bytes; the pointer
//!   may be NULL when the length is 0.

#![allow(unsafe_code, non_camel_case_types)]

use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

use augur::{Context, Engine, Error};

/// Version of the C ABI described by this header. Incremented on every
/// incompatible change; compare with `augur_abi_version()` at runtime.
pub const AUGUR_ABI_VERSION: u32 = 0;

/// Result code returned by fallible functions.
pub type augur_status = i32;

/// The call succeeded.
pub const AUGUR_STATUS_OK: augur_status = 0;
/// An argument was null, out of range, or not valid UTF-8, or the language
/// is not supported.
pub const AUGUR_STATUS_INVALID_ARGUMENT: augur_status = 1;
/// An I/O error occurred, e.g. while opening a model.
pub const AUGUR_STATUS_IO: augur_status = 2;
/// An internal error or panic occurred inside the library.
pub const AUGUR_STATUS_INTERNAL: augur_status = 3;
/// A model file is corrupted, has an unsupported format, or is for another
/// language.
pub const AUGUR_STATUS_INVALID_MODEL: augur_status = 4;

/// A prediction engine. Thread-safe: one engine may be used from several
/// threads at once.
pub struct augur_engine {
    engine: Engine,
}

/// A list of suggestions returned by `augur_predict`, most likely first.
pub struct augur_suggestions {
    items: Vec<Item>,
}

struct Item {
    /// The text followed by a NUL byte.
    text: Box<str>,
    prob: f32,
}

/// Returns the ABI version of the loaded library.
#[unsafe(no_mangle)]
pub extern "C" fn augur_abi_version() -> u32 {
    AUGUR_ABI_VERSION
}

/// Returns the library version as a static, NUL-terminated string, e.g.
/// `"0.1.0"`. The caller must not free it.
#[unsafe(no_mangle)]
pub extern "C" fn augur_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Returns a static, NUL-terminated description of `status`. Unknown values
/// return `"unknown status"`. The caller must not free it.
#[unsafe(no_mangle)]
pub extern "C" fn augur_status_str(status: augur_status) -> *const c_char {
    let s: &'static std::ffi::CStr = match status {
        AUGUR_STATUS_OK => c"ok",
        AUGUR_STATUS_INVALID_ARGUMENT => c"invalid argument",
        AUGUR_STATUS_IO => c"I/O error",
        AUGUR_STATUS_INTERNAL => c"internal error",
        AUGUR_STATUS_INVALID_MODEL => c"invalid model",
        _ => c"unknown status",
    };
    s.as_ptr()
}

/// Returns a NUL-terminated message describing the most recent failed call
/// on the calling thread, or an empty string if no call has failed. The
/// string is owned by the library and stays valid until the next failing
/// call on the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn augur_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Creates an engine for `language` (a BCP-47 tag such as `"en"` or
/// `"en-US"`) from the n-gram model file at `model_path`, and stores it in
/// `*out`. Release it with `augur_engine_free`.
///
/// Returns `AUGUR_STATUS_INVALID_ARGUMENT` for a malformed or unsupported
/// language, `AUGUR_STATUS_IO` if the model cannot be read and
/// `AUGUR_STATUS_INVALID_MODEL` if it is invalid.
///
/// # Safety
///
/// `language` and `model_path` must point to at least `language_len` and
/// `model_path_len` readable bytes (or be NULL with length 0), and `out`
/// must be a valid pointer to write to.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn augur_engine_new(
    language: *const c_char,
    language_len: usize,
    model_path: *const c_char,
    model_path_len: usize,
    out: *mut *mut augur_engine,
) -> augur_status {
    guard(|| {
        let out = out_arg(out, "out")?;
        // SAFETY: the caller guarantees the pointers and lengths.
        let language = unsafe { str_arg(language, language_len, "language") }?;
        let model_path = unsafe { str_arg(model_path, model_path_len, "model_path") }?;
        let engine = Engine::from_model(language, model_path).map_err(Failure::from)?;
        // SAFETY: `out` is non-null and the caller guarantees it is valid.
        unsafe { out.write(Box::into_raw(Box::new(augur_engine { engine }))) };
        Ok(())
    })
}

/// Releases an engine. NULL is ignored.
///
/// # Safety
///
/// `engine` must be NULL or a pointer from `augur_engine_new` that has not
/// been freed, and no other thread may be using it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn augur_engine_free(engine: *mut augur_engine) {
    if !engine.is_null() {
        // SAFETY: the caller passes ownership of a live engine.
        drop(unsafe { Box::from_raw(engine) });
    }
}

/// Predicts up to `max` words for the text around the cursor: `past` is
/// the text before it and `future` the text after it (may be empty). The
/// result is stored in `*out`, even when empty; release it with
/// `augur_suggestions_free`.
///
/// # Safety
///
/// `engine` must be a live engine; `past` and `future` must point to at
/// least `past_len` and `future_len` readable bytes (or be NULL with length
/// 0); `out` must be a valid pointer to write to.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn augur_predict(
    engine: *const augur_engine,
    past: *const c_char,
    past_len: usize,
    future: *const c_char,
    future_len: usize,
    max: usize,
    out: *mut *mut augur_suggestions,
) -> augur_status {
    guard(|| {
        let out = out_arg(out, "out")?;
        // SAFETY: the caller guarantees the pointers and lengths.
        let engine = unsafe { engine_arg(engine) }?;
        let past = unsafe { str_arg(past, past_len, "past") }?;
        let future = unsafe { str_arg(future, future_len, "future") }?;
        let items = engine
            .predict(Context::new(past, future), max)
            .into_iter()
            .map(|s| Item {
                text: format!("{}\0", s.text).into_boxed_str(),
                prob: s.prob,
            })
            .collect();
        // SAFETY: `out` is non-null and the caller guarantees it is valid.
        unsafe { out.write(Box::into_raw(Box::new(augur_suggestions { items }))) };
        Ok(())
    })
}

/// Feeds committed text to the engine so that predictors can learn from
/// it. (The n-gram predictor is static; this matters once learning
/// predictors are added.)
///
/// # Safety
///
/// `engine` must be a live engine and `text` must point to at least
/// `text_len` readable bytes (or be NULL with length 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn augur_learn(
    engine: *const augur_engine,
    text: *const c_char,
    text_len: usize,
) -> augur_status {
    guard(|| {
        // SAFETY: the caller guarantees the pointers and lengths.
        let engine = unsafe { engine_arg(engine) }?;
        let text = unsafe { str_arg(text, text_len, "text") }?;
        engine.learn(text);
        Ok(())
    })
}

/// Number of suggestions in `list` (0 for NULL).
///
/// # Safety
///
/// `list` must be NULL or a live list from `augur_predict`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn augur_suggestions_len(list: *const augur_suggestions) -> usize {
    // SAFETY: the caller guarantees `list` is NULL or live.
    unsafe { list.as_ref() }.map_or(0, |list| list.items.len())
}

/// Stores the text of suggestion `index` in `*text` (UTF-8, also
/// NUL-terminated) and its length in bytes, without the NUL, in `*len`.
/// The text stays valid until the list is freed.
///
/// # Safety
///
/// `list` must be a live list from `augur_predict`; `text` and `len` must
/// be valid pointers to write to.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn augur_suggestions_text(
    list: *const augur_suggestions,
    index: usize,
    text: *mut *const c_char,
    len: *mut usize,
) -> augur_status {
    guard(|| {
        let text = out_arg(text, "text")?;
        let len = out_arg(len, "len")?;
        // SAFETY: the caller guarantees `list` is NULL or live.
        let item = unsafe { item_arg(list, index) }?;
        // SAFETY: both pointers are non-null and the caller guarantees
        // they are valid.
        unsafe {
            text.write(item.text.as_ptr().cast());
            len.write(item.text.len() - 1);
        }
        Ok(())
    })
}

/// Stores the probability of suggestion `index`, in `0..=1`, in `*prob`.
///
/// # Safety
///
/// `list` must be a live list from `augur_predict`; `prob` must be a valid
/// pointer to write to.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn augur_suggestions_prob(
    list: *const augur_suggestions,
    index: usize,
    prob: *mut f32,
) -> augur_status {
    guard(|| {
        let prob = out_arg(prob, "prob")?;
        // SAFETY: the caller guarantees `list` is NULL or live.
        let item = unsafe { item_arg(list, index) }?;
        // SAFETY: `prob` is non-null and the caller guarantees it is valid.
        unsafe { prob.write(item.prob) };
        Ok(())
    })
}

/// Releases a suggestion list. NULL is ignored.
///
/// # Safety
///
/// `list` must be NULL or a pointer from `augur_predict` that has not been
/// freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn augur_suggestions_free(list: *mut augur_suggestions) {
    if !list.is_null() {
        // SAFETY: the caller passes ownership of a live list.
        drop(unsafe { Box::from_raw(list) });
    }
}

// ---- helpers ----

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_last_error(message: &str) {
    let message = CString::new(message.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = message);
}

/// A failed call: status and message for `augur_last_error`.
struct Failure {
    status: augur_status,
    message: String,
}

impl Failure {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: AUGUR_STATUS_INVALID_ARGUMENT,
            message: message.into(),
        }
    }
}

impl From<Error> for Failure {
    fn from(error: Error) -> Self {
        let status = match &error {
            Error::InvalidLanguageTag(_) | Error::UnsupportedLanguage(_) => {
                AUGUR_STATUS_INVALID_ARGUMENT
            }
            Error::Io(_) => AUGUR_STATUS_IO,
            Error::InvalidModel(_) => AUGUR_STATUS_INVALID_MODEL,
            _ => AUGUR_STATUS_INTERNAL,
        };
        Self {
            status,
            message: error.to_string(),
        }
    }
}

/// Runs `f`, turning failures and panics into a status and last error.
fn guard(f: impl FnOnce() -> Result<(), Failure>) -> augur_status {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => AUGUR_STATUS_OK,
        Ok(Err(failure)) => {
            set_last_error(&failure.message);
            failure.status
        }
        Err(panic) => {
            let detail = panic
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown panic");
            set_last_error(&format!("internal error: {detail}"));
            AUGUR_STATUS_INTERNAL
        }
    }
}

fn out_arg<T>(ptr: *mut T, name: &str) -> Result<*mut T, Failure> {
    if ptr.is_null() {
        Err(Failure::invalid(format!("{name} is NULL")))
    } else {
        Ok(ptr)
    }
}

/// # Safety
///
/// `ptr` must point to `len` readable bytes that outlive `'a`, or `len`
/// must be 0.
unsafe fn str_arg<'a>(ptr: *const c_char, len: usize, name: &str) -> Result<&'a str, Failure> {
    if len == 0 {
        return Ok("");
    }
    if ptr.is_null() {
        return Err(Failure::invalid(format!(
            "{name} is NULL but its length is {len}"
        )));
    }
    // SAFETY: guaranteed by the caller.
    let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), len) };
    std::str::from_utf8(bytes).map_err(|_| Failure::invalid(format!("{name} is not valid UTF-8")))
}

/// # Safety
///
/// `engine` must be NULL or a live engine that outlives `'a`.
unsafe fn engine_arg<'a>(engine: *const augur_engine) -> Result<&'a Engine, Failure> {
    // SAFETY: guaranteed by the caller.
    unsafe { engine.as_ref() }
        .map(|e| &e.engine)
        .ok_or_else(|| Failure::invalid("engine is NULL"))
}

/// # Safety
///
/// `list` must be NULL or a live list that outlives `'a`.
unsafe fn item_arg<'a>(list: *const augur_suggestions, index: usize) -> Result<&'a Item, Failure> {
    // SAFETY: guaranteed by the caller.
    let list = unsafe { list.as_ref() }.ok_or_else(|| Failure::invalid("list is NULL"))?;
    list.items.get(index).ok_or_else(|| {
        Failure::invalid(format!(
            "index {index} is out of range for {} suggestions",
            list.items.len()
        ))
    })
}

#[cfg(test)]
mod tests;
