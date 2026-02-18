#pragma once

/// @file ferritext.hpp
/// @brief Header-only C++ wrapper for ferritext OCR engine.

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

extern "C" {
struct FerritextEngine;
FerritextEngine *ferritext_init(const char *lang);
char *ferritext_ocr_file(const FerritextEngine *engine, const char *path);
char *ferritext_ocr_bytes(const FerritextEngine *engine, const uint8_t *data,
                          size_t len);
void ferritext_free(char *ptr);
void ferritext_engine_free(FerritextEngine *engine);
const char *ferritext_version();
}

namespace ferritext {

/// RAII wrapper for ferritext C strings.
class OcrResult {
public:
  explicit OcrResult(char *json) : json_(json) {}
  ~OcrResult() {
    if (json_)
      ferritext_free(json_);
  }

  OcrResult(OcrResult &&other) noexcept : json_(other.json_) {
    other.json_ = nullptr;
  }
  OcrResult &operator=(OcrResult &&other) noexcept {
    if (this != &other) {
      if (json_)
        ferritext_free(json_);
      json_ = other.json_;
      other.json_ = nullptr;
    }
    return *this;
  }

  OcrResult(const OcrResult &) = delete;
  OcrResult &operator=(const OcrResult &) = delete;

  /// Get the JSON result string.
  std::string json() const { return json_ ? std::string(json_) : "{}"; }

private:
  char *json_;
};

/// RAII OCR engine.
class OcrEngine {
public:
  explicit OcrEngine(const std::string &lang = "eng") {
    engine_ = ferritext_init(lang.c_str());
    if (!engine_)
      throw std::runtime_error("Failed to initialize ferritext engine");
  }

  ~OcrEngine() {
    if (engine_)
      ferritext_engine_free(engine_);
  }

  OcrEngine(OcrEngine &&other) noexcept : engine_(other.engine_) {
    other.engine_ = nullptr;
  }
  OcrEngine &operator=(OcrEngine &&other) noexcept {
    if (this != &other) {
      if (engine_)
        ferritext_engine_free(engine_);
      engine_ = other.engine_;
      other.engine_ = nullptr;
    }
    return *this;
  }

  OcrEngine(const OcrEngine &) = delete;
  OcrEngine &operator=(const OcrEngine &) = delete;

  /// Run OCR on a file.
  OcrResult ocr_file(const std::string &path) const {
    char *json = ferritext_ocr_file(engine_, path.c_str());
    return OcrResult(json);
  }

  /// Run OCR on raw bytes.
  OcrResult ocr_bytes(const uint8_t *data, size_t len) const {
    char *json = ferritext_ocr_bytes(engine_, data, len);
    return OcrResult(json);
  }

  /// Get library version.
  static std::string version() { return ferritext_version(); }

private:
  FerritextEngine *engine_ = nullptr;
};

} // namespace ferritext
