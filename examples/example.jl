# Example usage of ferritext Julia bindings.

include("../bindings/julia/Ferritext.jl")
using .Ferritext

println("ferritext v$(Ferritext.version())")

engine = OcrEngine("eng")
result = ocr_file(engine, "test_image.png")
println("Result: $result")
