use crate::utils::preprocessing::Preprocessor;
use anyhow::Result;
use ndarray::{s, Array1, Array2, Axis};
use std::collections::HashMap;

/// Comprehensive tabular data preprocessor for clinical data
#[derive(Clone)]
pub struct TabularPreprocessor {
    means: Option<Array1<f32>>,
    stds: Option<Array1<f32>>,
    mins: Option<Array1<f32>>,
    maxs: Option<Array1<f32>>,
    feature_names: Vec<String>,
    categorical_encodings: HashMap<String, HashMap<String, f32>>,
    missing_strategies: HashMap<usize, ImputationStrategy>,
    normalization_method: NormalizationMethod,
    fitted: bool,
}

#[derive(Clone)]
pub enum ImputationStrategy {
    Mean,
    Median,
    Mode,
    Forward,
    Constant(f32),
}

#[derive(Clone)]
pub enum NormalizationMethod {
    StandardScaling, // (x - mean) / std
    MinMaxScaling,   // (x - min) / (max - min)
    RobustScaling,   // Using median and IQR
    None,
}

impl TabularPreprocessor {
    pub fn new(feature_names: Vec<String>) -> Self {
        Self {
            means: None,
            stds: None,
            mins: None,
            maxs: None,
            feature_names,
            categorical_encodings: HashMap::new(),
            missing_strategies: HashMap::new(),
            normalization_method: NormalizationMethod::StandardScaling,
            fitted: false,
        }
    }

    pub fn with_normalization(mut self, method: NormalizationMethod) -> Self {
        self.normalization_method = method;
        self
    }

    pub fn with_imputation_strategy(
        mut self,
        feature_idx: usize,
        strategy: ImputationStrategy,
    ) -> Self {
        self.missing_strategies.insert(feature_idx, strategy);
        self
    }

    /// Handle missing values using specified strategies
    pub fn impute_missing(&self, mut data: Array2<f32>) -> Result<Array2<f32>> {
        let (n_rows, n_cols) = data.dim();

        for col in 0..n_cols {
            let strategy = self
                .missing_strategies
                .get(&col)
                .unwrap_or(&ImputationStrategy::Mean);

            let mut column = data.column_mut(col);
            let non_nan_values: Vec<f32> =
                column.iter().filter(|&&x| !x.is_nan()).cloned().collect();

            if non_nan_values.is_empty() {
                continue; // Skip if all values are NaN
            }

            let fill_value = match strategy {
                ImputationStrategy::Mean => {
                    non_nan_values.iter().sum::<f32>() / non_nan_values.len() as f32
                }
                ImputationStrategy::Median => {
                    let mut sorted = non_nan_values.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mid = sorted.len() / 2;
                    if sorted.len() % 2 == 0 {
                        (sorted[mid - 1] + sorted[mid]) / 2.0
                    } else {
                        sorted[mid]
                    }
                }
                ImputationStrategy::Mode => {
                    // Find most frequent value
                    let mut counts = HashMap::new();
                    for &value in &non_nan_values {
                        *counts.entry((value * 100.0) as i32).or_insert(0) += 1;
                    }
                    let mode = counts
                        .iter()
                        .max_by_key(|(_, &count)| count)
                        .map(|(&value, _)| value as f32 / 100.0)
                        .unwrap_or(0.0);
                    mode
                }
                ImputationStrategy::Forward => {
                    // Forward fill - use last known value
                    let mut last_valid = non_nan_values[0];
                    for i in 0..column.len() {
                        if !column[i].is_nan() {
                            last_valid = column[i];
                        } else {
                            column[i] = last_valid;
                        }
                    }
                    continue; // Skip the general fill below
                }
                ImputationStrategy::Constant(value) => *value,
            };

            // Fill NaN values
            for value in column.iter_mut() {
                if value.is_nan() {
                    *value = fill_value;
                }
            }
        }

        Ok(data)
    }

    /// Apply clinical domain knowledge for feature engineering
    pub fn engineer_clinical_features(&self, mut data: Array2<f32>) -> Result<Array2<f32>> {
        let (n_rows, n_cols) = data.dim();
        let mut engineered_data = Array2::zeros((n_rows, n_cols + 10)); // Add space for new features

        // Copy original features
        engineered_data.slice_mut(s![.., ..n_cols]).assign(&data);

        // Example feature engineering for pregnancy data
        for i in 0..n_rows {
            let row = data.row(i);

            // Assuming specific column indices for common clinical features
            if n_cols > 5 {
                let age = row[0];
                let weight = row[1];
                let height = row[2];
                let systolic_bp = row[3];
                let diastolic_bp = row[4];
                let glucose = row[5];

                // BMI calculation
                if height > 0.0 {
                    let bmi = weight / (height * height);
                    engineered_data[[i, n_cols]] = bmi;
                }

                // Blood pressure categories
                let bp_category = if systolic_bp >= 140.0 || diastolic_bp >= 90.0 {
                    2.0 // Hypertensive
                } else if systolic_bp >= 120.0 || diastolic_bp >= 80.0 {
                    1.0 // Elevated
                } else {
                    0.0 // Normal
                };
                engineered_data[[i, n_cols + 1]] = bp_category;

                // Age-based risk categories
                let age_risk = if age >= 35.0 {
                    2.0 // Advanced maternal age
                } else if age < 20.0 {
                    1.0 // Teen pregnancy
                } else {
                    0.0 // Normal age range
                };
                engineered_data[[i, n_cols + 2]] = age_risk;

                // Glucose categories (gestational diabetes risk)
                let glucose_category = if glucose >= 126.0 {
                    2.0 // Diabetic range
                } else if glucose >= 100.0 {
                    1.0 // Pre-diabetic
                } else {
                    0.0 // Normal
                };
                engineered_data[[i, n_cols + 3]] = glucose_category;
            }
        }

        Ok(engineered_data)
    }

    /// Detect and handle outliers using IQR method
    pub fn handle_outliers(
        &self,
        mut data: Array2<f32>,
        iqr_multiplier: f32,
    ) -> Result<Array2<f32>> {
        let (_, n_cols) = data.dim();

        for col in 0..n_cols {
            let column_data: Vec<f32> = data
                .column(col)
                .iter()
                .filter(|&&x| !x.is_nan())
                .cloned()
                .collect();

            if column_data.len() < 4 {
                continue; // Need at least 4 points for quartiles
            }

            let mut sorted_data = column_data.clone();
            sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let q1_idx = sorted_data.len() / 4;
            let q3_idx = 3 * sorted_data.len() / 4;
            let q1 = sorted_data[q1_idx];
            let q3 = sorted_data[q3_idx];
            let iqr = q3 - q1;

            let lower_bound = q1 - iqr_multiplier * iqr;
            let upper_bound = q3 + iqr_multiplier * iqr;

            // Cap outliers to bounds
            for value in data.column_mut(col).iter_mut() {
                if !value.is_nan() {
                    if *value < lower_bound {
                        *value = lower_bound;
                    } else if *value > upper_bound {
                        *value = upper_bound;
                    }
                }
            }
        }

        Ok(data)
    }
}

impl Preprocessor for TabularPreprocessor {
    type Input = Array2<f32>;
    type Output = Array2<f32>;

    fn fit(&mut self, data: &Self::Input) {
        let (_, n_cols) = data.dim();

        match self.normalization_method {
            NormalizationMethod::StandardScaling => {
                self.means = Some(data.mean_axis(Axis(0)).unwrap());
                self.stds = Some(data.std_axis(Axis(0), 1.0));
            }
            NormalizationMethod::MinMaxScaling => {
                let mut mins = Array1::zeros(n_cols);
                let mut maxs = Array1::zeros(n_cols);

                for col in 0..n_cols {
                    let column_data: Vec<f32> = data
                        .column(col)
                        .iter()
                        .filter(|&&x| !x.is_nan())
                        .cloned()
                        .collect();

                    if !column_data.is_empty() {
                        mins[col] = column_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                        maxs[col] = column_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    }
                }

                self.mins = Some(mins);
                self.maxs = Some(maxs);
            }
            _ => {}
        }

        self.fitted = true;
    }

    fn transform(&self, data: Self::Input) -> Self::Output {
        if !self.fitted {
            panic!("Preprocessor must be fitted before transform");
        }

        // Handle missing values
        let mut processed_data = self.impute_missing(data).unwrap();

        // Handle outliers
        processed_data = self.handle_outliers(processed_data, 1.5).unwrap();

        // Engineer clinical features
        processed_data = self.engineer_clinical_features(processed_data).unwrap();

        // Apply normalization
        match &self.normalization_method {
            NormalizationMethod::StandardScaling => {
                let means = self.means.as_ref().unwrap();
                let stds = self.stds.as_ref().unwrap();
                let n_original_cols = means.len();

                // Only normalize original columns, not engineered ones
                for col in 0..n_original_cols {
                    if stds[col] > 1e-8 {
                        // Avoid division by zero
                        for mut column in processed_data.column_mut(col).iter_mut() {
                            *column = (*column - means[col]) / stds[col];
                        }
                    }
                }
            }
            NormalizationMethod::MinMaxScaling => {
                let mins = self.mins.as_ref().unwrap();
                let maxs = self.maxs.as_ref().unwrap();
                let n_original_cols = mins.len();

                for col in 0..n_original_cols {
                    let range = maxs[col] - mins[col];
                    if range > 1e-8 {
                        // Avoid division by zero
                        for mut column in processed_data.column_mut(col).iter_mut() {
                            *column = (*column - mins[col]) / range;
                        }
                    }
                }
            }
            _ => {}
        }

        processed_data
    }
}

/// Specialized preprocessor for specific clinical data types
pub struct ClinicalDataPreprocessor {
    vital_signs_processor: TabularPreprocessor,
    lab_results_processor: TabularPreprocessor,
    demographics_processor: TabularPreprocessor,
}

impl ClinicalDataPreprocessor {
    pub fn new() -> Self {
        Self {
            vital_signs_processor: TabularPreprocessor::new(vec![
                "systolic_bp".to_string(),
                "diastolic_bp".to_string(),
                "heart_rate".to_string(),
                "temperature".to_string(),
            ])
            .with_normalization(NormalizationMethod::StandardScaling),

            lab_results_processor: TabularPreprocessor::new(vec![
                "glucose".to_string(),
                "protein".to_string(),
                "hemoglobin".to_string(),
                "hematocrit".to_string(),
            ])
            .with_normalization(NormalizationMethod::MinMaxScaling),

            demographics_processor: TabularPreprocessor::new(vec![
                "age".to_string(),
                "weight".to_string(),
                "height".to_string(),
                "gestational_age".to_string(),
            ])
            .with_normalization(NormalizationMethod::StandardScaling),
        }
    }

    pub fn process_vital_signs(&mut self, data: Array2<f32>) -> Result<Array2<f32>> {
        if !self.vital_signs_processor.fitted {
            self.vital_signs_processor.fit(&data);
        }
        Ok(self.vital_signs_processor.transform(data))
    }

    pub fn process_lab_results(&mut self, data: Array2<f32>) -> Result<Array2<f32>> {
        if !self.lab_results_processor.fitted {
            self.lab_results_processor.fit(&data);
        }
        Ok(self.lab_results_processor.transform(data))
    }

    pub fn process_demographics(&mut self, data: Array2<f32>) -> Result<Array2<f32>> {
        if !self.demographics_processor.fitted {
            self.demographics_processor.fit(&data);
        }
        Ok(self.demographics_processor.transform(data))
    }
}
