"""Example usage of ferritext Python bindings."""

import json

# pip install ferritext
import ferritext


def main():
    # Create engine
    engine = ferritext.PyOcrEngine(lang="eng")
    print(f"ferritext v{ferritext.PyOcrEngine.version()}")

    # OCR a file
    result_json = engine.ocr("test_image.png")
    result = json.loads(result_json)

    print(f"Text: {result.get('text', '')}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    print(f"Words: {len(result.get('words', []))}")

    # Or use the convenience function
    result_json = ferritext.ocr("test_image.png")
    print(result_json)


if __name__ == "__main__":
    main()
