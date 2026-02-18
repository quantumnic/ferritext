/// Example usage of ferritext C++ bindings.

#include "../bindings/cpp/ferritext.hpp"
#include <iostream>

int main() {
    try {
        ferritext::OcrEngine engine("eng");
        std::cout << "ferritext v" << ferritext::OcrEngine::version() << std::endl;

        auto result = engine.ocr_file("test_image.png");
        std::cout << "Result: " << result.json() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
