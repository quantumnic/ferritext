//! C FFI bindings for ferritext OCR engine.
//!
//! All functions return JSON strings that must be freed with `ferritext_free`.

use ferritext_core::{output, OcrEngine};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

/// Opaque engine handle.
pub struct FerritextEngine {
    engine: OcrEngine,
}

/// Initialize a new OCR engine with random (untrained) weights.
///
/// # Safety
/// Returns a heap-allocated engine that must be freed with `ferritext_engine_free`.
#[no_mangle]
pub unsafe extern "C" fn ferritext_init(lang: *const c_char) -> *mut FerritextEngine {
    let language = if lang.is_null() {
        "eng".to_string()
    } else {
        match CStr::from_ptr(lang).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => "eng".to_string(),
        }
    };

    let engine = OcrEngine::new_random(&language);
    Box::into_raw(Box::new(FerritextEngine { engine }))
}

/// Run OCR on an image file. Returns a JSON string.
///
/// # Safety
/// `engine` must be a valid pointer from `ferritext_init`.
/// `path` must be a valid null-terminated UTF-8 string.
/// The returned string must be freed with `ferritext_free`.
#[no_mangle]
pub unsafe extern "C" fn ferritext_ocr_file(
    engine: *const FerritextEngine,
    path: *const c_char,
) -> *mut c_char {
    if engine.is_null() || path.is_null() {
        return to_error_json("null pointer");
    }

    let engine = &(*engine).engine;
    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => return to_error_json("invalid UTF-8 path"),
    };

    match engine.ocr_file(path_str) {
        Ok(result) => {
            let json = output::to_json(&result);
            CString::new(json)
                .map(|s| s.into_raw())
                .unwrap_or(ptr::null_mut())
        }
        Err(e) => to_error_json(&e.to_string()),
    }
}

/// Run OCR on raw image bytes. Returns a JSON string.
///
/// # Safety
/// `engine` must be a valid pointer from `ferritext_init`.
/// `data` must point to `len` valid bytes.
/// The returned string must be freed with `ferritext_free`.
#[no_mangle]
pub unsafe extern "C" fn ferritext_ocr_bytes(
    engine: *const FerritextEngine,
    data: *const u8,
    len: usize,
) -> *mut c_char {
    if engine.is_null() || data.is_null() {
        return to_error_json("null pointer");
    }

    let engine = &(*engine).engine;
    let slice = std::slice::from_raw_parts(data, len);

    match engine.ocr_bytes(slice) {
        Ok(result) => {
            let json = output::to_json(&result);
            CString::new(json)
                .map(|s| s.into_raw())
                .unwrap_or(ptr::null_mut())
        }
        Err(e) => to_error_json(&e.to_string()),
    }
}

/// Free a string returned by ferritext functions.
///
/// # Safety
/// `ptr` must be a pointer returned by a ferritext function, or null.
#[no_mangle]
pub unsafe extern "C" fn ferritext_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

/// Free an engine handle.
///
/// # Safety
/// `engine` must be a pointer from `ferritext_init`, or null.
#[no_mangle]
pub unsafe extern "C" fn ferritext_engine_free(engine: *mut FerritextEngine) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}

/// Get the library version. The returned string is static and must NOT be freed.
#[no_mangle]
pub extern "C" fn ferritext_version() -> *const c_char {
    // Safety: this is a static string
    c"0.1.0".as_ptr()
}

unsafe fn to_error_json(msg: &str) -> *mut c_char {
    let json = format!("{{\"error\":\"{}\"}}", msg.replace('"', "\\\""));
    CString::new(json)
        .map(|s| s.into_raw())
        .unwrap_or(ptr::null_mut())
}
