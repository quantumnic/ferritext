using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Ferritext
{
    /// <summary>
    /// P/Invoke bindings for ferritext native OCR engine.
    /// </summary>
    public class OcrEngine : IDisposable
    {
        private IntPtr _engine;
        private bool _disposed;

        [DllImport("ferritext_ffi", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr ferritext_init(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string lang);

        [DllImport("ferritext_ffi", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr ferritext_ocr_file(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string path);

        [DllImport("ferritext_ffi", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr ferritext_ocr_bytes(
            IntPtr engine, IntPtr data, UIntPtr len);

        [DllImport("ferritext_ffi", CallingConvention = CallingConvention.Cdecl)]
        private static extern void ferritext_free(IntPtr ptr);

        [DllImport("ferritext_ffi", CallingConvention = CallingConvention.Cdecl)]
        private static extern void ferritext_engine_free(IntPtr engine);

        [DllImport("ferritext_ffi", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr ferritext_version();

        public OcrEngine(string language = "eng")
        {
            _engine = ferritext_init(language);
            if (_engine == IntPtr.Zero)
                throw new InvalidOperationException("Failed to initialize ferritext engine");
        }

        /// <summary>Run OCR on an image file.</summary>
        public string OcrFile(string path)
        {
            IntPtr result = ferritext_ocr_file(_engine, path);
            return ConsumeString(result);
        }

        /// <summary>Run OCR on raw image bytes.</summary>
        public string OcrBytes(byte[] data)
        {
            var handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                IntPtr result = ferritext_ocr_bytes(
                    _engine,
                    handle.AddrOfPinnedObject(),
                    (UIntPtr)data.Length);
                return ConsumeString(result);
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>Get the library version.</summary>
        public static string Version()
        {
            IntPtr ptr = ferritext_version();
            return Marshal.PtrToStringUTF8(ptr) ?? "";
        }

        private static string ConsumeString(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero) return "{}";
            string result = Marshal.PtrToStringUTF8(ptr) ?? "{}";
            ferritext_free(ptr);
            return result;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed && _engine != IntPtr.Zero)
            {
                ferritext_engine_free(_engine);
                _engine = IntPtr.Zero;
            }
            _disposed = true;
        }

        ~OcrEngine() => Dispose(false);
    }
}
