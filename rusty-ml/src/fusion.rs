use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ndarray::Array2;

/// Trait for different fusion strategies
pub trait FusionStrategy<B: Backend> {
    fn fuse(&self, predictions: Vec<Array2<f32>>) -> Result<Array2<f32>>;
}

/// Late fusion with weighted averaging
pub struct LateFusion<B: Backend> {
    weights: Vec<f32>,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> LateFusion<B> {
    pub fn new(weights: Vec<f32>) -> Self {
        // Normalize weights to sum to 1
        let sum: f32 = weights.iter().sum();
        let normalized_weights = weights.iter().map(|w| w / sum).collect();

        Self {
            weights: normalized_weights,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> FusionStrategy<B> for LateFusion<B> {
    fn fuse(&self, predictions: Vec<Array2<f32>>) -> Result<Array2<f32>> {
        if predictions.is_empty() {
            return Err(anyhow::anyhow!("No predictions to fuse"));
        }

        let mut weighted_sum = predictions[0].clone() * self.weights[0];

        for (i, pred) in predictions.iter().skip(1).enumerate() {
            let weight_idx = std::cmp::min(i + 1, self.weights.len() - 1);
            weighted_sum = weighted_sum + pred * self.weights[weight_idx];
        }

        Ok(weighted_sum)
    }
}

/// Attention-based fusion for dynamic weighting
pub struct AttentionFusion<B: Backend> {
    attention_network: burn::nn::Linear<B>,
}

impl<B: Backend> AttentionFusion<B> {
    pub fn new(input_dim: usize) -> Self {
        Self {
            attention_network: burn::nn::Linear::new(input_dim, 1),
        }
    }
}

impl<B: Backend> FusionStrategy<B> for AttentionFusion<B> {
    fn fuse(&self, predictions: Vec<Array2<f32>>) -> Result<Array2<f32>> {
        // Convert to tensors and compute attention weights
        let tensors: Vec<Tensor<B, 2>> = predictions
            .iter()
            .map(|pred| Tensor::from_data(pred.clone().into()))
            .collect();

        // Compute attention weights for each modality
        let attention_weights: Vec<Tensor<B, 2>> = tensors
            .iter()
            .map(|t| self.attention_network.forward(t.clone()).softmax(1))
            .collect();

        // Weighted sum using attention
        let mut result = tensors[0].clone() * attention_weights[0].clone();
        for (tensor, weight) in tensors.iter().zip(attention_weights.iter()).skip(1) {
            result = result + tensor.clone() * weight.clone();
        }

        Ok(result.into_data().convert::<Vec<f32>>().into())
    }
}
