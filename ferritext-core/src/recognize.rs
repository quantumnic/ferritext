//! CRNN inference engine — pure Rust implementation.
//! Conv2D → BatchNorm → ReLU → MaxPool (×4) → BiLSTM → Linear

use crate::model::{BatchNormParams, ConvWeights, CrnnModel, LinearWeights, LstmWeights};
use image::GrayImage;

/// Run the CRNN model on a normalized grayscale image (height=32).
/// Returns logits of shape [time_steps, num_classes].
pub fn run_crnn(model: &CrnnModel, img: &GrayImage) -> Vec<Vec<f32>> {
    let (w, h) = img.dimensions();

    // Convert image to float tensor [1, h, w]
    let mut input = vec![vec![vec![0.0f32; w as usize]; h as usize]; 1];
    for y in 0..h {
        for x in 0..w {
            input[0][y as usize][x as usize] = img.get_pixel(x, y).0[0] as f32 / 255.0;
        }
    }

    // Conv block 1
    let x = conv2d(&input, &model.conv1);
    let x = batch_norm(&x, &model.bn1);
    let x = relu_3d(&x);
    let x = max_pool_2d(&x);

    // Conv block 2
    let x = conv2d(&x, &model.conv2);
    let x = batch_norm(&x, &model.bn2);
    let x = relu_3d(&x);
    let x = max_pool_2d(&x);

    // Conv block 3
    let x = conv2d(&x, &model.conv3);
    let x = batch_norm(&x, &model.bn3);
    let x = relu_3d(&x);
    let x = max_pool_2d(&x);

    // Conv block 4
    let x = conv2d(&x, &model.conv4);
    let x = batch_norm(&x, &model.bn4);
    let x = relu_3d(&x);
    let x = max_pool_2d(&x);

    // Reshape: [channels, h', w'] → sequence [w', channels*h']
    let channels = x.len();
    let height = if x.is_empty() { 0 } else { x[0].len() };
    let width = if height == 0 || x.is_empty() {
        0
    } else {
        x[0][0].len()
    };

    if width == 0 || channels == 0 {
        return vec![vec![0.0; crate::model::num_classes()]; 1];
    }

    let feature_size = channels * height.max(1);
    let mut sequence: Vec<Vec<f32>> = Vec::with_capacity(width.max(1));
    for t in 0..width {
        let mut features = Vec::with_capacity(feature_size);
        for c in 0..channels {
            for h_idx in 0..height {
                features.push(x[c][h_idx][t]);
            }
        }
        // Pad or truncate to lstm input_size
        features.resize(model.lstm_forward.input_size, 0.0);
        sequence.push(features);
    }

    // BiLSTM
    let forward_out = lstm_forward(&sequence, &model.lstm_forward);
    let backward_out = lstm_backward(&sequence, &model.lstm_backward);

    // Concatenate forward + backward
    let mut bilstm_out: Vec<Vec<f32>> = Vec::with_capacity(sequence.len());
    for t in 0..sequence.len() {
        let mut combined = forward_out[t].clone();
        combined.extend_from_slice(&backward_out[t]);
        bilstm_out.push(combined);
    }

    // Linear projection to num_classes
    let mut logits: Vec<Vec<f32>> = Vec::with_capacity(bilstm_out.len());
    for t in 0..bilstm_out.len() {
        logits.push(linear(&bilstm_out[t], &model.linear));
    }

    logits
}

/// 2D convolution with padding=1, stride=1.
fn conv2d(input: &[Vec<Vec<f32>>], weights: &ConvWeights) -> Vec<Vec<Vec<f32>>> {
    let in_ch = input.len();
    let in_h = if in_ch > 0 { input[0].len() } else { 0 };
    let in_w = if in_h > 0 { input[0][0].len() } else { 0 };

    if in_h == 0 || in_w == 0 {
        return vec![vec![vec![]; 0]; weights.out_channels];
    }

    let k = weights.kernel_size;
    let pad = k / 2;
    let out_h = in_h;
    let out_w = in_w;

    let mut output = vec![vec![vec![0.0f32; out_w]; out_h]; weights.out_channels];

    for oc in 0..weights.out_channels {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut sum = weights.bias[oc];
                for ic in 0..weights.in_channels.min(in_ch) {
                    for kh in 0..k {
                        for kw in 0..k {
                            let ih = oh as isize + kh as isize - pad as isize;
                            let iw = ow as isize + kw as isize - pad as isize;
                            if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                let w_idx = ((oc * weights.in_channels + ic) * k + kh) * k + kw;
                                if w_idx < weights.weights.len() {
                                    sum += input[ic][ih as usize][iw as usize]
                                        * weights.weights[w_idx];
                                }
                            }
                        }
                    }
                }
                output[oc][oh][ow] = sum;
            }
        }
    }

    output
}

/// Batch normalization (inference mode).
fn batch_norm(input: &[Vec<Vec<f32>>], params: &BatchNormParams) -> Vec<Vec<Vec<f32>>> {
    let eps = 1e-5f32;
    let mut output = input.to_vec();
    for (c, channel) in output.iter_mut().enumerate() {
        if c >= params.channels {
            break;
        }
        let mean = params.running_mean[c];
        let var = params.running_var[c];
        let gamma = params.gamma[c];
        let beta = params.beta[c];
        let inv_std = 1.0 / (var + eps).sqrt();
        for row in channel.iter_mut() {
            for val in row.iter_mut() {
                *val = gamma * (*val - mean) * inv_std + beta;
            }
        }
    }
    output
}

/// ReLU activation on 3D tensor.
fn relu_3d(input: &[Vec<Vec<f32>>]) -> Vec<Vec<Vec<f32>>> {
    input
        .iter()
        .map(|ch| {
            ch.iter()
                .map(|row| row.iter().map(|&v| v.max(0.0)).collect())
                .collect()
        })
        .collect()
}

/// 2x2 max pooling with stride 2.
fn max_pool_2d(input: &[Vec<Vec<f32>>]) -> Vec<Vec<Vec<f32>>> {
    input
        .iter()
        .map(|ch| {
            let h = ch.len();
            let w = if h > 0 { ch[0].len() } else { 0 };
            let out_h = h / 2;
            let out_w = w / 2;
            let mut pooled = vec![vec![0.0f32; out_w]; out_h];
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let v00 = ch[oh * 2][ow * 2];
                    let v01 = ch[oh * 2]
                        .get(ow * 2 + 1)
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY);
                    let v10 = ch
                        .get(oh * 2 + 1)
                        .map(|r| r[ow * 2])
                        .unwrap_or(f32::NEG_INFINITY);
                    let v11 = ch
                        .get(oh * 2 + 1)
                        .and_then(|r| r.get(ow * 2 + 1))
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY);
                    pooled[oh][ow] = v00.max(v01).max(v10).max(v11);
                }
            }
            pooled
        })
        .collect()
}

/// LSTM forward pass (left to right).
fn lstm_forward(sequence: &[Vec<f32>], weights: &LstmWeights) -> Vec<Vec<f32>> {
    lstm_pass(sequence, weights, false)
}

/// LSTM backward pass (right to left).
fn lstm_backward(sequence: &[Vec<f32>], weights: &LstmWeights) -> Vec<Vec<f32>> {
    lstm_pass(sequence, weights, true)
}

fn lstm_pass(sequence: &[Vec<f32>], weights: &LstmWeights, reverse: bool) -> Vec<Vec<f32>> {
    let hidden_size = weights.hidden_size;
    let mut h = vec![0.0f32; hidden_size];
    let mut c = vec![0.0f32; hidden_size];
    let mut outputs = vec![vec![0.0f32; hidden_size]; sequence.len()];

    let indices: Vec<usize> = if reverse {
        (0..sequence.len()).rev().collect()
    } else {
        (0..sequence.len()).collect()
    };

    for (out_idx, &t) in indices.iter().enumerate() {
        let x = &sequence[t];

        // gates = W_ih @ x + b_ih + W_hh @ h + b_hh
        let mut gates = vec![0.0f32; 4 * hidden_size];
        for i in 0..4 * hidden_size {
            let mut sum = weights.b_ih[i] + weights.b_hh[i];
            for j in 0..weights.input_size.min(x.len()) {
                let idx = i * weights.input_size + j;
                if idx < weights.w_ih.len() {
                    sum += weights.w_ih[idx] * x[j];
                }
            }
            for j in 0..hidden_size {
                let idx = i * hidden_size + j;
                if idx < weights.w_hh.len() {
                    sum += weights.w_hh[idx] * h[j];
                }
            }
            gates[i] = sum;
        }

        // Split into i, f, g, o gates
        for i in 0..hidden_size {
            let ig = sigmoid(gates[i]);
            let fg = sigmoid(gates[hidden_size + i]);
            let gg = gates[2 * hidden_size + i].tanh();
            let og = sigmoid(gates[3 * hidden_size + i]);

            c[i] = fg * c[i] + ig * gg;
            h[i] = og * c[i].tanh();
        }

        let actual_out_idx = if reverse {
            sequence.len() - 1 - out_idx
        } else {
            out_idx
        };
        outputs[actual_out_idx] = h.clone();
    }

    outputs
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Linear layer: y = Wx + b
fn linear(input: &[f32], weights: &LinearWeights) -> Vec<f32> {
    let mut output = weights.bias.clone();
    for i in 0..weights.out_features {
        for j in 0..weights.in_features.min(input.len()) {
            let idx = i * weights.in_features + j;
            if idx < weights.weights.len() {
                output[i] += weights.weights[idx] * input[j];
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CrnnModel;
    use image::GrayImage;

    #[test]
    fn test_conv2d_identity_shape() {
        let input = vec![vec![vec![1.0f32; 8]; 8]; 1]; // 1 channel, 8x8
        let weights = crate::model::ConvWeights {
            weights: vec![0.0; 32 * 1 * 3 * 3],
            bias: vec![0.0; 32],
            in_channels: 1,
            out_channels: 32,
            kernel_size: 3,
        };
        let output = conv2d(&input, &weights);
        assert_eq!(output.len(), 32);
        assert_eq!(output[0].len(), 8);
        assert_eq!(output[0][0].len(), 8);
    }

    #[test]
    fn test_relu() {
        let input = vec![vec![vec![-1.0, 0.0, 1.0]]];
        let output = relu_3d(&input);
        assert_eq!(output[0][0], vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_max_pool() {
        let input = vec![vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]];
        let output = max_pool_2d(&input);
        assert_eq!(output[0].len(), 1); // 2/2
        assert_eq!(output[0][0].len(), 2); // 4/2
        assert_eq!(output[0][0][0], 6.0);
        assert_eq!(output[0][0][1], 8.0);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_linear_layer() {
        let weights = LinearWeights {
            weights: vec![1.0, 0.0, 0.0, 1.0],
            bias: vec![0.5, -0.5],
            in_features: 2,
            out_features: 2,
        };
        let input = vec![1.0, 2.0];
        let output = linear(&input, &weights);
        assert!((output[0] - 1.5).abs() < 1e-6); // 1*1 + 0*2 + 0.5
        assert!((output[1] - 1.5).abs() < 1e-6); // 0*1 + 1*2 + (-0.5)
    }

    #[test]
    fn test_run_crnn_small() {
        let model = CrnnModel::random();
        let img = GrayImage::new(64, 32);
        let logits = run_crnn(&model, &img);
        // After 4 max pools, width 64 → 32 → 16 → 8 → 4
        assert!(!logits.is_empty());
        assert_eq!(logits[0].len(), crate::model::num_classes());
    }

    #[test]
    fn test_batch_norm_identity() {
        let input = vec![vec![vec![2.0, 3.0]; 2]; 3];
        let params = BatchNormParams {
            gamma: vec![1.0; 3],
            beta: vec![0.0; 3],
            running_mean: vec![0.0; 3],
            running_var: vec![1.0; 3],
            channels: 3,
        };
        let output = batch_norm(&input, &params);
        // With mean=0, var=1, gamma=1, beta=0: output ≈ input
        assert!((output[0][0][0] - 2.0).abs() < 1e-3);
    }
}
