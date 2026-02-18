//! Python bindings for ferritext OCR engine via PyO3.

use ferritext_core::{output, OcrEngine};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python OCR engine wrapper.
#[pyclass]
struct PyOcrEngine {
    engine: OcrEngine,
}

#[pymethods]
impl PyOcrEngine {
    #[new]
    #[pyo3(signature = (lang = "eng"))]
    fn new(lang: &str) -> Self {
        Self {
            engine: OcrEngine::new_random(lang),
        }
    }

    /// Run OCR on an image file path.
    fn ocr(&self, image_path: &str) -> PyResult<String> {
        match self.engine.ocr_file(image_path) {
            Ok(result) => Ok(output::to_json(&result)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Run OCR on raw image bytes.
    fn ocr_bytes(&self, data: &[u8]) -> PyResult<String> {
        match self.engine.ocr_bytes(data) {
            Ok(result) => Ok(output::to_json(&result)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the engine language.
    fn language(&self) -> &str {
        self.engine.language()
    }

    /// Get library version.
    #[staticmethod]
    fn version() -> &'static str {
        OcrEngine::version()
    }
}

/// Run OCR on an image file (convenience function).
#[pyfunction]
fn ocr(image_path: &str) -> PyResult<String> {
    let engine = OcrEngine::new_random("eng");
    match engine.ocr_file(image_path) {
        Ok(result) => Ok(output::to_json(&result)),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

/// Run OCR on raw image bytes (convenience function).
#[pyfunction]
fn ocr_bytes(data: &[u8]) -> PyResult<String> {
    let engine = OcrEngine::new_random("eng");
    match engine.ocr_bytes(data) {
        Ok(result) => Ok(output::to_json(&result)),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

/// ferritext Python module.
#[pymodule]
fn ferritext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOcrEngine>()?;
    m.add_function(wrap_pyfunction!(ocr, m)?)?;
    m.add_function(wrap_pyfunction!(ocr_bytes, m)?)?;
    Ok(())
}
