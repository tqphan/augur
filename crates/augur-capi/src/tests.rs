use std::ffi::CStr;
use std::ptr;

use super::*;

const TINY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny.bin");

fn last_error() -> String {
    unsafe { CStr::from_ptr(augur_last_error()) }
        .to_str()
        .unwrap()
        .to_owned()
}

fn new_engine(language: &str, path: &str) -> Result<*mut augur_engine, augur_status> {
    let mut engine = ptr::null_mut();
    let status = unsafe {
        augur_engine_new(
            language.as_ptr().cast(),
            language.len(),
            path.as_ptr().cast(),
            path.len(),
            &raw mut engine,
        )
    };
    if status == AUGUR_STATUS_OK {
        assert!(!engine.is_null());
        Ok(engine)
    } else {
        assert!(engine.is_null());
        Err(status)
    }
}

/// Predicts through the C API and reads the list back.
fn predict(
    engine: *const augur_engine,
    past: &str,
    future: &str,
    max: usize,
) -> Vec<(String, f32)> {
    let mut list = ptr::null_mut();
    let status = unsafe {
        augur_predict(
            engine,
            past.as_ptr().cast(),
            past.len(),
            future.as_ptr().cast(),
            future.len(),
            max,
            &raw mut list,
        )
    };
    assert_eq!(status, AUGUR_STATUS_OK);
    let len = unsafe { augur_suggestions_len(list) };
    let items = (0..len)
        .map(|i| {
            let (mut text, mut text_len, mut prob) = (ptr::null(), 0, 0.0);
            unsafe {
                assert_eq!(
                    augur_suggestions_text(list, i, &raw mut text, &raw mut text_len),
                    0
                );
                assert_eq!(augur_suggestions_prob(list, i, &raw mut prob), 0);
                // The text is also NUL-terminated.
                let c = CStr::from_ptr(text).to_str().unwrap();
                assert_eq!(c.len(), text_len);
                (c.to_owned(), prob)
            }
        })
        .collect();
    unsafe { augur_suggestions_free(list) };
    items
}

#[test]
fn version_matches_package() {
    let version = unsafe { CStr::from_ptr(augur_version()) };
    assert_eq!(version.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
}

#[test]
fn status_strings() {
    let s = |status| unsafe { CStr::from_ptr(augur_status_str(status)) };
    assert_eq!(s(AUGUR_STATUS_OK), c"ok");
    assert_eq!(s(AUGUR_STATUS_INVALID_MODEL), c"invalid model");
    assert_eq!(s(-1), c"unknown status");
}

#[test]
fn predicts_and_learns() {
    let engine = new_engine("en", TINY).unwrap();
    let out = predict(engine, "I live in ", "", 2);
    let words: Vec<&str> = out.iter().map(|(w, _)| w.as_str()).collect();
    assert_eq!(words, ["London", "Paris"]);
    assert!(out[0].1 > out[1].1 && out[0].1 <= 1.0);

    assert_eq!(predict(engine, "I live in ", " London", 1)[0].0, "Paris");
    assert!(predict(engine, "the ", "", 0).is_empty());
    // NULL with length 0 is an empty string.
    let mut list = ptr::null_mut();
    let status = unsafe { augur_predict(engine, ptr::null(), 0, ptr::null(), 0, 3, &raw mut list) };
    assert_eq!(status, AUGUR_STATUS_OK);
    assert_eq!(unsafe { augur_suggestions_len(list) }, 3);
    unsafe { augur_suggestions_free(list) };

    let text = "I love my dog.";
    assert_eq!(
        unsafe { augur_learn(engine, text.as_ptr().cast(), text.len()) },
        AUGUR_STATUS_OK
    );
    unsafe { augur_engine_free(engine) };
}

#[test]
fn reports_errors() {
    assert_eq!(new_engine("xx", TINY), Err(AUGUR_STATUS_INVALID_ARGUMENT));
    assert!(
        last_error().contains("unsupported language"),
        "{}",
        last_error()
    );
    assert_eq!(
        new_engine("en_US", TINY),
        Err(AUGUR_STATUS_INVALID_ARGUMENT)
    );
    assert_eq!(new_engine("en", "does-not-exist.bin"), Err(AUGUR_STATUS_IO));
    let not_a_model = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
    assert_eq!(
        new_engine("en", not_a_model),
        Err(AUGUR_STATUS_INVALID_MODEL)
    );
    assert!(
        last_error().contains("not an augur language model"),
        "{}",
        last_error()
    );

    let bad_utf8 = [0xffu8];
    let mut engine = ptr::null_mut();
    let status =
        unsafe { augur_engine_new(bad_utf8.as_ptr().cast(), 1, ptr::null(), 0, &raw mut engine) };
    assert_eq!(status, AUGUR_STATUS_INVALID_ARGUMENT);
    assert!(last_error().contains("language is not valid UTF-8"));
    let status = unsafe { augur_engine_new(ptr::null(), 2, ptr::null(), 0, &raw mut engine) };
    assert_eq!(status, AUGUR_STATUS_INVALID_ARGUMENT);
    assert_eq!(
        unsafe { augur_engine_new(c"en".as_ptr(), 2, ptr::null(), 0, ptr::null_mut()) },
        AUGUR_STATUS_INVALID_ARGUMENT
    );
    assert_eq!(last_error(), "out is NULL");

    let mut list = ptr::null_mut();
    let status = unsafe {
        augur_predict(
            ptr::null(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            3,
            &raw mut list,
        )
    };
    assert_eq!(status, AUGUR_STATUS_INVALID_ARGUMENT);
    assert!(list.is_null());

    let engine = new_engine("en", TINY).unwrap();
    let status = unsafe { augur_predict(engine, ptr::null(), 0, ptr::null(), 0, 1, &raw mut list) };
    assert_eq!(status, AUGUR_STATUS_OK);
    let mut prob = 0.0;
    assert_eq!(
        unsafe { augur_suggestions_prob(list, 1, &raw mut prob) },
        AUGUR_STATUS_INVALID_ARGUMENT
    );
    assert!(last_error().contains("out of range"));
    assert_eq!(unsafe { augur_suggestions_len(ptr::null()) }, 0);
    unsafe {
        augur_suggestions_free(list);
        augur_suggestions_free(ptr::null_mut());
        augur_engine_free(engine);
        augur_engine_free(ptr::null_mut());
    }
}

#[test]
fn last_error_is_per_thread() {
    assert_eq!(new_engine("xx", TINY), Err(AUGUR_STATUS_INVALID_ARGUMENT));
    let other = std::thread::spawn(last_error).join().unwrap();
    assert_eq!(other, "");
}

#[test]
fn panics_become_internal_errors() {
    let status = guard(|| panic!("boom"));
    assert_eq!(status, AUGUR_STATUS_INTERNAL);
    assert_eq!(last_error(), "internal error: boom");
}
