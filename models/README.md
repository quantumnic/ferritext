# Models

## Base Latin Model

The base Latin model supports the following character sets:
- ASCII (printable characters)
- Extended Latin (German ÄÖÜäöüß, French àâéèêëîïôùûüç, Spanish ñ, etc.)

### Architecture
- **Input**: Grayscale, 32px height, variable width
- **CNN**: 4 conv blocks (1→32→64→128→128), 3×3 kernels, BatchNorm, ReLU, MaxPool 2×2
- **RNN**: 1 BiLSTM layer, 128 hidden units
- **Output**: CTC decoder, beam search (width=5)
- **Parameters**: ~2M (~8MB fp32, ~4MB int8 quantized)

### Model Format (`.ftxt`)

Binary format:
```
[4 bytes] Magic: "FTXT"
[4 bytes] Version: u32 LE
[4 bytes] Num layers: u32 LE
For each layer:
  [4 bytes] Layer type: u32 LE (0=conv, 1=bn, 2=lstm, 3=linear)
  [4 bytes] Data length in bytes: u32 LE
  [N bytes] Float32 LE weights
```

### Training

Models can be trained using standard PyTorch/TensorFlow and exported to `.ftxt` format.
A training script will be provided in a future release.

### Download

Pre-trained models will be available at:
```
https://github.com/quantumnic/ferritext/releases/download/v0.1.0/latin-base.ftxt
```
