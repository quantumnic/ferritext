"""
    Ferritext

Julia bindings for the ferritext OCR engine.
"""
module Ferritext

export OcrEngine, ocr_file, ocr_bytes, version

const LIB_NAME = "libferritext_ffi"

mutable struct OcrEngine
    ptr::Ptr{Cvoid}

    function OcrEngine(lang::String="eng")
        ptr = ccall((:ferritext_init, LIB_NAME), Ptr{Cvoid}, (Cstring,), lang)
        engine = new(ptr)
        finalizer(engine) do e
            if e.ptr != C_NULL
                ccall((:ferritext_engine_free, LIB_NAME), Cvoid, (Ptr{Cvoid},), e.ptr)
                e.ptr = C_NULL
            end
        end
        return engine
    end
end

"""
    ocr_file(engine::OcrEngine, path::String) -> String

Run OCR on an image file. Returns JSON string.
"""
function ocr_file(engine::OcrEngine, path::String)::String
    result_ptr = ccall((:ferritext_ocr_file, LIB_NAME), Ptr{Cchar},
                       (Ptr{Cvoid}, Cstring), engine.ptr, path)
    result = unsafe_string(result_ptr)
    ccall((:ferritext_free, LIB_NAME), Cvoid, (Ptr{Cchar},), result_ptr)
    return result
end

"""
    ocr_bytes(engine::OcrEngine, data::Vector{UInt8}) -> String

Run OCR on raw image bytes. Returns JSON string.
"""
function ocr_bytes(engine::OcrEngine, data::Vector{UInt8})::String
    result_ptr = ccall((:ferritext_ocr_bytes, LIB_NAME), Ptr{Cchar},
                       (Ptr{Cvoid}, Ptr{UInt8}, Csize_t),
                       engine.ptr, data, length(data))
    result = unsafe_string(result_ptr)
    ccall((:ferritext_free, LIB_NAME), Cvoid, (Ptr{Cchar},), result_ptr)
    return result
end

"""
    version() -> String

Get the ferritext library version.
"""
function version()::String
    ptr = ccall((:ferritext_version, LIB_NAME), Ptr{Cchar}, ())
    return unsafe_string(ptr)
end

end # module
