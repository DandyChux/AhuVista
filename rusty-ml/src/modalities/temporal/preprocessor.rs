use crate::utils::preprocessing::Preprocessor;
use anyhow::Result;
use ndarray::{s, Array1, Array2, Array3, Axis};
use std::collections::VecDeque;

/// Temporal data preprocessor for time-series clinical data
#[derive(Clone)]
pub struct TemporalPreprocessor {
    window_size: usize,
    stride: usize,
    overlap: f32,
    normalization_params: Option<(Array1<f32>, Array1<f32>)>, // (mean, std) per feature
    padding_strategy: PaddingStrategy,
    resampling_rate: Option<f32>,
    feature_names: Vec<String>,
    fitted: bool,
}

#[derive(Clone)]
pub enum PaddingStrategy {
    Zero,
    Forward,  // Repeat last value
    Backward, // Repeat first value
    Mean,     // Fill with mean of sequence
    Linear,   // Linear interpolation
}

impl TemporalPreprocessor {
    pub fn new(window_size: usize, stride: usize) -> Self {
        Self {
            window_size,
            stride,
            overlap: 0.5,
            normalization_params: None,
            padding_strategy: PaddingStrategy::Forward,
            resampling_rate: None,
            feature_names: Vec::new(),
            fitted: false,
        }
    }

    pub fn with_padding_strategy(mut self, strategy: PaddingStrategy) -> Self {
        self.padding_strategy = strategy;
        self
    }

    pub fn with_resampling(mut self, rate: f32) -> Self {
        self.resampling_rate = Some(rate);
        self
    }

    /// Create sliding windows from time series data
    pub fn create_windows(&self, data: &Array2<f32>) -> Result<Array3<f32>> {
        let (seq_len, n_features) = data.dim();

        if seq_len < self.window_size {
            // Handle sequences shorter than window size
            return self.handle_short_sequences(data);
        }

        let n_windows = (seq_len - self.window_size) / self.stride + 1;
        let mut windows = Array3::zeros((n_windows, self.window_size, n_features));

        for (window_idx, start_idx) in (0..=seq_len - self.window_size)
            .step_by(self.stride)
            .enumerate()
        {
            let end_idx = start_idx + self.window_size;
            let window_data = data.slice(s![start_idx..end_idx, ..]);
            windows
                .slice_mut(s![window_idx, .., ..])
                .assign(&window_data);
        }

        Ok(windows)
    }

    /// Handle sequences shorter than window size with padding
    fn handle_short_sequences(&self, data: &Array2<f32>) -> Result<Array3<f32>> {
        let (seq_len, n_features) = data.dim();
        let mut padded_data = Array2::zeros((self.window_size, n_features));

        // Copy existing data
        padded_data.slice_mut(s![..seq_len, ..]).assign(data);

        // Apply padding strategy for remaining positions
        match self.padding_strategy {
            PaddingStrategy::Zero => {
                // Already zeros from initialization
            }
            PaddingStrategy::Forward => {
                let last_row = data.row(seq_len - 1);
                for i in seq_len..self.window_size {
                    padded_data.row_mut(i).assign(&last_row);
                }
            }
            PaddingStrategy::Backward => {
                let first_row = data.row(0);
                // Shift data and prepend
                for i in 0..(self.window_size - seq_len) {
                    padded_data.row_mut(i).assign(&first_row);
                }
                padded_data
                    .slice_mut(s![self.window_size - seq_len.., ..])
                    .assign(data);
            }
            PaddingStrategy::Mean => {
                let mean_values = data.mean_axis(Axis(0)).unwrap();
                for i in seq_len..self.window_size {
                    padded_data.row_mut(i).assign(&mean_values);
                }
            }
            PaddingStrategy::Linear => {
                self.apply_linear_interpolation(&mut padded_data, seq_len)?;
            }
        }

        // Return as 3D array with single window
        Ok(padded_data.insert_axis(Axis(0)))
    }

    /// Apply linear interpolation for padding
    fn apply_linear_interpolation(
        &self,
        padded_data: &mut Array2<f32>,
        original_len: usize,
    ) -> Result<()> {
        let (_, n_features) = padded_data.dim();

        if original_len < 2 {
            return Ok(()); // Can't interpolate with less than 2 points
        }

        let first_val = padded_data.row(0).to_owned();
        let last_val = padded_data.row(original_len - 1).to_owned();

        for i in original_len..self.window_size {
            let alpha =
                (i - original_len + 1) as f32 / (self.window_size - original_len + 1) as f32;
            for j in 0..n_features {
                let interpolated = last_val[j] + alpha * (last_val[j] - first_val[j]);
                padded_data[[i, j]] = interpolated;
            }
        }

        Ok(())
    }

    /// Resample time series to different frequency
    pub fn resample(&self, data: Array2<f32>, target_length: usize) -> Result<Array2<f32>> {
        let (current_length, n_features) = data.dim();

        if current_length == target_length {
            return Ok(data);
        }

        let mut resampled = Array2::zeros((target_length, n_features));
        let ratio = current_length as f32 / target_length as f32;

        for i in 0..target_length {
            let source_idx = i as f32 * ratio;
            let lower_idx = source_idx.floor() as usize;
            let upper_idx = (source_idx.ceil() as usize).min(current_length - 1);
            let alpha = source_idx - source_idx.floor();

            if lower_idx == upper_idx {
                // No interpolation needed
                resampled.row_mut(i).assign(&data.row(lower_idx));
            } else {
                // Linear interpolation
                let lower_row = data.row(lower_idx);
                let upper_row = data.row(upper_idx);

                for j in 0..n_features {
                    let interpolated = lower_row[j] * (1.0 - alpha) + upper_row[j] * alpha;
                    resampled[[i, j]] = interpolated;
                }
            }
        }

        Ok(resampled)
    }

    /// Apply temporal smoothing to reduce noise
    pub fn smooth_temporal_data(
        &self,
        data: &Array2<f32>,
        window_size: usize,
    ) -> Result<Array2<f32>> {
        let (seq_len, n_features) = data.dim();
        let mut smoothed = data.clone();

        let half_window = window_size / 2;

        for i in half_window..(seq_len - half_window) {
            for j in 0..n_features {
                let start_idx = i - half_window;
                let end_idx = i + half_window + 1;
                let window_mean = data.slice(s![start_idx..end_idx, j]).mean().unwrap();
                smoothed[[i, j]] = window_mean;
            }
        }

        Ok(smoothed)
    }

    /// Detect and handle temporal outliers
    pub fn handle_temporal_outliers(
        &self,
        data: &Array2<f32>,
        z_threshold: f32,
    ) -> Result<Array2<f32>> {
        let (seq_len, n_features) = data.dim();
        let mut cleaned_data = data.clone();

        for j in 0..n_features {
            let column = data.column(j);
            let mean = column.mean().unwrap();
            let std = column.std(1.0);

            for i in 0..seq_len {
                let z_score = (column[i] - mean).abs() / std;
                if z_score > z_threshold {
                    // Replace outlier with interpolated value
                    if i == 0 {
                        cleaned_data[[i, j]] = column[1];
                    } else if i == seq_len - 1 {
                        cleaned_data[[i, j]] = column[seq_len - 2];
                    } else {
                        cleaned_data[[i, j]] = (column[i - 1] + column[i + 1]) / 2.0;
                    }
                }
            }
        }

        Ok(cleaned_data)
    }

    /// Extract temporal features like trends, seasonality
    pub fn extract_temporal_features(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
        let (seq_len, n_features) = data.dim();
        let mut features = Array2::zeros((seq_len, n_features * 4)); // Original + 3 derived features

        // Copy original features
        features.slice_mut(s![.., ..n_features]).assign(data);

        for j in 0..n_features {
            let column = data.column(j);

            // Moving average (trend)
            let window_size = std::cmp::min(seq_len / 4, 10);
            for i in window_size..seq_len {
                let start_idx = i.saturating_sub(window_size);
                let moving_avg = column.slice(s![start_idx..=i]).mean().unwrap();
                features[[i, n_features + j]] = moving_avg;
            }

            // First difference (velocity)
            for i in 1..seq_len {
                features[[i, 2 * n_features + j]] = column[i] - column[i - 1];
            }

            // Second difference (acceleration)
            for i in 2..seq_len {
                let first_diff_prev = column[i - 1] - column[i - 2];
                let first_diff_curr = column[i] - column[i - 1];
                features[[i, 3 * n_features + j]] = first_diff_curr - first_diff_prev;
            }
        }

        Ok(features)
    }

    /// Apply temporal augmentation for data augmentation
    pub fn augment_temporal_data(&self, data: &Array2<f32>) -> Result<Vec<Array2<f32>>> {
        let mut augmented_data = vec![data.clone()];

        // Time warping
        let warped = self.apply_time_warping(data, 0.1)?;
        augmented_data.push(warped);

        // Add noise
        let noisy = self.add_temporal_noise(data, 0.05)?;
        augmented_data.push(noisy);

        // Scaling
        let scaled = self.apply_temporal_scaling(data, 0.9, 1.1)?;
        augmented_data.push(scaled);

        Ok(augmented_data)
    }

    fn apply_time_warping(&self, data: &Array2<f32>, sigma: f32) -> Result<Array2<f32>> {
        let (seq_len, n_features) = data.dim();
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Create warping path
        let mut warp_path = Vec::new();
        for i in 0..seq_len {
            let noise = rng.gen_range(-sigma..sigma);
            let warped_idx = ((i as f32 + noise) * seq_len as f32 / seq_len as f32) as usize;
            warp_path.push(warped_idx.min(seq_len - 1));
        }

        let mut warped_data = Array2::zeros((seq_len, n_features));
        for (i, &src_idx) in warp_path.iter().enumerate() {
            warped_data.row_mut(i).assign(&data.row(src_idx));
        }

        Ok(warped_data)
    }

    fn add_temporal_noise(&self, data: &Array2<f32>, noise_level: f32) -> Result<Array2<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut noisy_data = data.clone();

        for value in noisy_data.iter_mut() {
            let noise = rng.gen_range(-noise_level..noise_level);
            *value += *value * noise;
        }

        Ok(noisy_data)
    }

    fn apply_temporal_scaling(
        &self,
        data: &Array2<f32>,
        min_scale: f32,
        max_scale: f32,
    ) -> Result<Array2<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = rng.gen_range(min_scale..max_scale);

        Ok(data * scale)
    }
}

impl Preprocessor for TemporalPreprocessor {
    type Input = Array2<f32>;
    type Output = Array3<f32>;

    fn fit(&mut self, data: &Self::Input) {
        let (_, n_features) = data.dim();

        // Compute normalization parameters
        let means = data.mean_axis(Axis(0)).unwrap();
        let stds = data.std_axis(Axis(0), 1.0);

        self.normalization_params = Some((means, stds));
        self.fitted = true;
    }

    fn transform(&self, data: Self::Input) -> Self::Output {
        if !self.fitted {
            panic!("TemporalPreprocessor must be fitted before transform");
        }

        // Apply preprocessing steps
        let mut processed_data = data;

        // Handle outliers
        processed_data = self
            .handle_temporal_outliers(&processed_data, 3.0)
            .expect("Failed to handle outliers");

        // Smooth data
        processed_data = self
            .smooth_temporal_data(&processed_data, 3)
            .expect("Failed to smooth data");

        // Normalize
        if let Some((means, stds)) = &self.normalization_params {
            for i in 0..means.len() {
                if stds[i] > 1e-8 {
                    for mut col in processed_data.column_mut(i).iter_mut() {
                        *col = (*col - means[i]) / stds[i];
                    }
                }
            }
        }

        // Create windows
        self.create_windows(&processed_data)
            .expect("Failed to create windows")
    }
}

/// Specialized preprocessor for different types of temporal clinical data
pub struct ClinicalTemporalPreprocessor {
    vital_signs_processor: TemporalPreprocessor,
    continuous_monitoring_processor: TemporalPreprocessor,
    medication_dosage_processor: TemporalPreprocessor,
}

impl ClinicalTemporalPreprocessor {
    pub fn new() -> Self {
        Self {
            // High-frequency vital signs (every minute)
            vital_signs_processor: TemporalPreprocessor::new(60, 30)
                .with_padding_strategy(PaddingStrategy::Forward),

            // Continuous monitoring (every second)
            continuous_monitoring_processor: TemporalPreprocessor::new(300, 150)
                .with_padding_strategy(PaddingStrategy::Linear),

            // Medication dosage (irregular intervals)
            medication_dosage_processor: TemporalPreprocessor::new(24, 12)
                .with_padding_strategy(PaddingStrategy::Zero),
        }
    }

    pub fn process_vital_signs(&mut self, data: Array2<f32>) -> Result<Array3<f32>> {
        if !self.vital_signs_processor.fitted {
            self.vital_signs_processor.fit(&data);
        }
        Ok(self.vital_signs_processor.transform(data))
    }

    pub fn process_continuous_monitoring(&mut self, data: Array2<f32>) -> Result<Array3<f32>> {
        if !self.continuous_monitoring_processor.fitted {
            self.continuous_monitoring_processor.fit(&data);
        }
        Ok(self.continuous_monitoring_processor.transform(data))
    }

    pub fn process_medication_dosage(&mut self, data: Array2<f32>) -> Result<Array3<f32>> {
        if !self.medication_dosage_processor.fitted {
            self.medication_dosage_processor.fit(&data);
        }
        Ok(self.medication_dosage_processor.transform(data))
    }
}
