//! C API for augur.
//!
//! The header `include/augur.h` is generated from this crate by cbindgen at
//! build time. Conventions for every exported function:
//!
//! - Names are prefixed with `augur_`.
//! - Objects are opaque handles created by `augur_*_new` and released by the
//!   matching `augur_*_free`.
//! - Fallible functions return an [`augur_status`]; panics never cross the
//!   FFI boundary.
//! - Strings are UTF-8, passed as pointer plus length.

#![allow(unsafe_code, non_camel_case_types)]

use std::ffi::c_char;

/// Version of the C ABI described by this header. Incremented on every
/// incompatible change; compare with `augur_abi_version()` at runtime.
pub const AUGUR_ABI_VERSION: u32 = 0;

/// Result code returned by fallible functions.
pub type augur_status = i32;

/// The call succeeded.
pub const AUGUR_STATUS_OK: augur_status = 0;
/// An argument was null, out of range, or not valid UTF-8.
pub const AUGUR_STATUS_INVALID_ARGUMENT: augur_status = 1;
/// An I/O error occurred, e.g. while opening a model.
pub const AUGUR_STATUS_IO: augur_status = 2;
/// An internal error or panic occurred inside the library.
pub const AUGUR_STATUS_INTERNAL: augur_status = 3;

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
        _ => c"unknown status",
    };
    s.as_ptr()
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::*;

    #[test]
    fn version_matches_package() {
        let version = unsafe { CStr::from_ptr(augur_version()) };
        assert_eq!(version.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn status_strings() {
        let s = |status| unsafe { CStr::from_ptr(augur_status_str(status)) };
        assert_eq!(s(AUGUR_STATUS_OK), c"ok");
        assert_eq!(s(-1), c"unknown status");
    }
}
