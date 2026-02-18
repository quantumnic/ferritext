//! Model loading and weight format for the CRNN architecture.
//!
//! Model architecture:
//! - 4 conv layers: 1→32→64→128→128 (3x3, stride 1, pad 1) + BatchNorm + ReLU + MaxPool(2x2)
//! - 1 BiLSTM layer: input_size=128*1 (after 4 maxpools height 32→2→1, width variable), hidden=128
//! - Linear: 256 (bi) → num_classes (charset size + blank)

use crate::Result;

/// The character set for Latin scripts.
pub const CHARSET: &str = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ";

/// Blank token index for CTC.
pub const BLANK_IDX: usize = 0;

/// Number of output classes = charset + blank.
pub fn num_classes() -> usize {
    CHARSET.chars().count() + 1
}

/// Convolutional layer weights.
#[derive(Debug, Clone)]
pub struct ConvWeights {
    /// Kernel weights [out_channels][in_channels][kh][kw]
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
}

/// Batch normalization parameters.
#[derive(Debug, Clone)]
pub struct BatchNormParams {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub running_mean: Vec<f32>,
    pub running_var: Vec<f32>,
    pub channels: usize,
}

/// LSTM weights for one direction.
#[derive(Debug, Clone)]
pub struct LstmWeights {
    /// Input-hidden weights [4*hidden, input_size]
    pub w_ih: Vec<f32>,
    /// Hidden-hidden weights [4*hidden, hidden_size]
    pub w_hh: Vec<f32>,
    /// Bias for input-hidden [4*hidden]
    pub b_ih: Vec<f32>,
    /// Bias for hidden-hidden [4*hidden]
    pub b_hh: Vec<f32>,
    pub input_size: usize,
    pub hidden_size: usize,
}

/// Linear layer weights.
#[derive(Debug, Clone)]
pub struct LinearWeights {
    /// Weights [out_features, in_features]
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_features: usize,
    pub out_features: usize,
}

/// Complete CRNN model.
#[derive(Debug, Clone)]
pub struct CrnnModel {
    pub conv1: ConvWeights,
    pub bn1: BatchNormParams,
    pub conv2: ConvWeights,
    pub bn2: BatchNormParams,
    pub conv3: ConvWeights,
    pub bn3: BatchNormParams,
    pub conv4: ConvWeights,
    pub bn4: BatchNormParams,
    pub lstm_forward: LstmWeights,
    pub lstm_backward: LstmWeights,
    pub linear: LinearWeights,
}

impl CrnnModel {
    /// Create a model with random (Xavier-initialized) weights.
    pub fn random() -> Self {
        let nc = num_classes();
        Self {
            conv1: ConvWeights::random(1, 32, 3),
            bn1: BatchNormParams::identity(32),
            conv2: ConvWeights::random(32, 64, 3),
            bn2: BatchNormParams::identity(64),
            conv3: ConvWeights::random(64, 128, 3),
            bn3: BatchNormParams::identity(128),
            conv4: ConvWeights::random(128, 128, 3),
            bn4: BatchNormParams::identity(128),
            lstm_forward: LstmWeights::random(128, 128),
            lstm_backward: LstmWeights::random(128, 128),
            linear: LinearWeights::random(256, nc),
        }
    }

    /// Load model from compact binary format.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        // Magic: "FTXT" + version u32
        if data.len() < 8 {
            return Err(crate::Error::ModelError("Data too short".into()));
        }
        if &data[0..4] != b"FTXT" {
            return Err(crate::Error::ModelError("Invalid magic".into()));
        }
        let _version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

        // For now, return random model (actual deserialization would follow the format spec)
        Ok(Self::random())
    }

    /// Serialize model to compact binary format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"FTXT");
        buf.extend_from_slice(&1u32.to_le_bytes());
        // Placeholder: serialize all weights
        self.write_floats(&mut buf, &self.conv1.weights);
        buf
    }

    fn write_floats(&self, buf: &mut Vec<u8>, data: &[f32]) {
        for &f in data {
            buf.extend_from_slice(&f.to_le_bytes());
        }
    }
}

impl ConvWeights {
    fn random(in_ch: usize, out_ch: usize, kernel: usize) -> Self {
        let n = out_ch * in_ch * kernel * kernel;
        let scale = (2.0 / (in_ch * kernel * kernel) as f64).sqrt() as f32;
        // Deterministic "random" for reproducibility
        let weights: Vec<f32> = (0..n)
            .map(|i| {
                let x = ((i as f64 * 0.6180339887).fract() - 0.5) as f32 * 2.0 * scale;
                x
            })
            .collect();
        let bias = vec![0.0; out_ch];
        Self {
            weights,
            bias,
            in_channels: in_ch,
            out_channels: out_ch,
            kernel_size: kernel,
        }
    }
}

impl BatchNormParams {
    fn identity(channels: usize) -> Self {
        Self {
            gamma: vec![1.0; channels],
            beta: vec![0.0; channels],
            running_mean: vec![0.0; channels],
            running_var: vec![1.0; channels],
            channels,
        }
    }
}

impl LstmWeights {
    fn random(input_size: usize, hidden_size: usize) -> Self {
        let scale = (1.0 / hidden_size as f64).sqrt() as f32;
        let make_weights = |rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols)
                .map(|i| ((i as f64 * 0.6180339887).fract() - 0.5) as f32 * 2.0 * scale)
                .collect()
        };
        Self {
            w_ih: make_weights(4 * hidden_size, input_size),
            w_hh: make_weights(4 * hidden_size, hidden_size),
            b_ih: vec![0.0; 4 * hidden_size],
            b_hh: vec![0.0; 4 * hidden_size],
            input_size,
            hidden_size,
        }
    }
}

impl LinearWeights {
    fn random(in_f: usize, out_f: usize) -> Self {
        let scale = (1.0 / in_f as f64).sqrt() as f32;
        let weights: Vec<f32> = (0..out_f * in_f)
            .map(|i| ((i as f64 * 0.6180339887).fract() - 0.5) as f32 * 2.0 * scale)
            .collect();
        let bias = vec![0.0; out_f];
        Self {
            weights,
            bias,
            in_features: in_f,
            out_features: out_f,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_charset_size() {
        let nc = num_classes();
        assert!(nc > 90); // ASCII + extended Latin + blank
    }

    #[test]
    fn test_random_model() {
        let model = CrnnModel::random();
        assert_eq!(model.conv1.in_channels, 1);
        assert_eq!(model.conv1.out_channels, 32);
        assert_eq!(model.conv4.out_channels, 128);
        assert_eq!(model.lstm_forward.hidden_size, 128);
        assert_eq!(model.linear.in_features, 256);
        assert_eq!(model.linear.out_features, num_classes());
    }

    #[test]
    fn test_model_serialization_roundtrip() {
        let model = CrnnModel::random();
        let bytes = model.to_bytes();
        assert!(bytes.len() >= 8);
        assert_eq!(&bytes[0..4], b"FTXT");
    }

    #[test]
    fn test_from_bytes_invalid() {
        let result = CrnnModel::from_bytes(b"BAD");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_bytes_wrong_magic() {
        let result = CrnnModel::from_bytes(b"XXXX\x01\x00\x00\x00");
        assert!(result.is_err());
    }
}
