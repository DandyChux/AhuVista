// src/modalities/text/model.rs
use anyhow::Result;
use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};

/// TF-IDF based text classification model (lightweight alternative to transformers)
pub struct TfIdfModel {
    vocabulary: HashMap<String, usize>,
    idf_scores: Vec<f32>,
    max_features: usize,
    classifier_weights: Array2<f32>,
    classifier_bias: Array1<f32>,
    trained: bool,
    // Clinical keywords for pregnancy-related text
    clinical_keywords: HashSet<String>,
}

impl TfIdfModel {
    pub fn new(max_features: usize) -> Self {
        let clinical_keywords = [
            "pregnancy",
            "prenatal",
            "gestational",
            "fetal",
            "maternal",
            "ultrasound",
            "trimester",
            "delivery",
            "labor",
            "contractions",
            "hypertension",
            "diabetes",
            "preeclampsia",
            "bleeding",
            "pain",
            "weight",
            "growth",
            "heartbeat",
            "movement",
            "position",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Self {
            vocabulary: HashMap::new(),
            idf_scores: Vec::new(),
            max_features,
            classifier_weights: Array2::zeros((max_features, 2)), // Binary classification
            classifier_bias: Array1::zeros(2),
            trained: false,
            clinical_keywords,
        }
    }

    /// Build vocabulary from training corpus
    pub fn fit_vocabulary(&mut self, documents: &[String]) -> Result<()> {
        let mut word_doc_counts = HashMap::new();
        let total_docs = documents.len() as f32;

        // Count document frequency for each word
        for doc in documents {
            let words = self.tokenize(doc);
            let unique_words: HashSet<_> = words.into_iter().collect();

            for word in unique_words {
                *word_doc_counts.entry(word).or_insert(0) += 1;
            }
        }

        // Select top words and build vocabulary (prioritize clinical keywords)
        let mut word_freq: Vec<_> = word_doc_counts.into_iter().collect();

        // Sort by frequency, but boost clinical keywords
        word_freq.sort_by(|a, b| {
            let a_is_clinical = self.clinical_keywords.contains(&a.0);
            let b_is_clinical = self.clinical_keywords.contains(&b.0);

            match (a_is_clinical, b_is_clinical) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => b.1.cmp(&a.1), // Sort by frequency
            }
        });

        self.vocabulary = word_freq
            .into_iter()
            .take(self.max_features)
            .enumerate()
            .map(|(idx, (word, _))| (word, idx))
            .collect();

        // Compute IDF scores
        self.idf_scores = vec![0.0; self.vocabulary.len()];
        for doc in documents {
            let words = self.tokenize(doc);
            let unique_words: HashSet<_> = words.into_iter().collect();

            for word in unique_words {
                if let Some(&idx) = self.vocabulary.get(&word) {
                    self.idf_scores[idx] += 1.0;
                }
            }
        }

        // Convert to IDF: log(total_docs / doc_freq)
        for score in &mut self.idf_scores {
            *score = (total_docs / *score).ln();
        }

        Ok(())
    }

    /// Train simple linear classifier on TF-IDF features
    pub fn train(
        &mut self,
        documents: &[String],
        labels: &[i32],
        learning_rate: f32,
        epochs: usize,
    ) -> Result<()> {
        if !self.trained {
            self.fit_vocabulary(documents)?;
        }

        let features = self.vectorize_documents(documents)?;
        let n_samples = features.nrows();
        let n_features = features.ncols();

        // Initialize weights randomly
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for i in 0..n_features {
            for j in 0..2 {
                self.classifier_weights[[i, j]] = rng.gen_range(-0.1..0.1);
            }
        }

        // Simple gradient descent training
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (i, &label) in labels.iter().enumerate() {
                let x = features.row(i);
                let prediction = self.predict_probabilities(&x.to_owned())?;

                // One-hot encode true label
                let mut y_true = Array1::zeros(2);
                y_true[label as usize] = 1.0;

                // Compute gradients (cross-entropy loss)
                let error = &prediction - &y_true;
                total_loss += error.mapv(|x| x * x).sum();

                // Update weights
                for j in 0..n_features {
                    for k in 0..2 {
                        self.classifier_weights[[j, k]] -= learning_rate * error[k] * x[j];
                    }
                }

                // Update bias
                for k in 0..2 {
                    self.classifier_bias[k] -= learning_rate * error[k];
                }
            }

            if epoch % 10 == 0 {
                println!(
                    "Epoch {}: Loss = {:.4}",
                    epoch,
                    total_loss / n_samples as f32
                );
            }
        }

        self.trained = true;
        Ok(())
    }

    /// Convert documents to TF-IDF feature vectors
    pub fn vectorize_documents(&self, documents: &[String]) -> Result<Array2<f32>> {
        let mut features = Array2::zeros((documents.len(), self.vocabulary.len()));

        for (doc_idx, doc) in documents.iter().enumerate() {
            let tf_vector = self.compute_tf(doc);

            for (word, &vocab_idx) in &self.vocabulary {
                if let Some(&tf_score) = tf_vector.get(word) {
                    let tfidf_score = tf_score * self.idf_scores[vocab_idx];
                    features[[doc_idx, vocab_idx]] = tfidf_score;
                }
            }
        }

        Ok(features)
    }

    /// Predict class probabilities for a single document
    pub fn predict_probabilities(&self, tfidf_vector: &Array1<f32>) -> Result<Array1<f32>> {
        let mut logits = self.classifier_bias.clone();

        for (i, &weight) in tfidf_vector.iter().enumerate() {
            for j in 0..2 {
                logits[j] += weight * self.classifier_weights[[i, j]];
            }
        }

        // Apply softmax
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Array1<f32> = logits.mapv(|x| (x - max_logit).exp());
        let sum_exp = exp_logits.sum();

        Ok(exp_logits / sum_exp)
    }

    /// Public prediction interface
    pub fn predict(&self, document: &str) -> Result<Array2<f32>> {
        if !self.trained {
            return Err(anyhow::anyhow!("Model must be trained before prediction"));
        }

        let features = self.vectorize_documents(&[document.to_string()])?;
        let tfidf_vector = features.row(0).to_owned();
        let probabilities = self.predict_probabilities(&tfidf_vector)?;

        // Convert to 2D array for consistency with other models
        let mut result = Array2::zeros((1, 2));
        result.row_mut(0).assign(&probabilities);

        Ok(result)
    }

    /// Compute term frequency for a document
    fn compute_tf(&self, document: &str) -> HashMap<String, f32> {
        let words = self.tokenize(document);
        let total_words = words.len() as f32;
        let mut tf_map = HashMap::new();

        for word in words {
            *tf_map.entry(word).or_insert(0.0) += 1.0;
        }

        // Normalize by document length
        for (_, tf) in tf_map.iter_mut() {
            *tf /= total_words;
        }

        tf_map
    }

    /// Tokenize and clean text
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|word| {
                // Remove punctuation and keep only alphanumeric
                word.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
            })
            .filter(|word| {
                word.len() > 2 && // Remove very short words
                !self.is_stopword(word) // Remove stopwords
            })
            .collect()
    }

    /// Simple stopword filtering
    fn is_stopword(&self, word: &str) -> bool {
        let stopwords = [
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is",
            "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "must", "a", "an", "this", "that",
            "these", "those", "i", "you", "he", "she", "it", "we", "they",
        ];
        stopwords.contains(&word)
    }

    /// Extract clinical keywords and their frequencies
    pub fn extract_clinical_features(&self, document: &str) -> HashMap<String, f32> {
        let words = self.tokenize(document);
        let mut clinical_features = HashMap::new();

        for word in words {
            if self.clinical_keywords.contains(&word) {
                *clinical_features.entry(word).or_insert(0.0) += 1.0;
            }
        }

        clinical_features
    }

    pub fn is_trained(&self) -> bool {
        self.trained
    }

    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }
}

/// Bag-of-Words model as an alternative lightweight approach
pub struct BagOfWordsModel {
    vocabulary: HashMap<String, usize>,
    classifier_weights: Array2<f32>,
    max_features: usize,
    trained: bool,
}

impl BagOfWordsModel {
    pub fn new(max_features: usize) -> Self {
        Self {
            vocabulary: HashMap::new(),
            classifier_weights: Array2::zeros((max_features, 2)),
            max_features,
            trained: false,
        }
    }

    pub fn fit(&mut self, documents: &[String], labels: &[i32]) -> Result<()> {
        // Build vocabulary
        let mut word_counts = HashMap::new();
        for doc in documents {
            let words = self.tokenize(doc);
            for word in words {
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }

        // Select top words
        let mut word_freq: Vec<_> = word_counts.into_iter().collect();
        word_freq.sort_by(|a, b| b.1.cmp(&a.1));

        self.vocabulary = word_freq
            .into_iter()
            .take(self.max_features)
            .enumerate()
            .map(|(idx, (word, _))| (word, idx))
            .collect();

        // Train classifier (simplified logistic regression)
        let features = self.vectorize_documents(documents)?;
        // ... training logic similar to TF-IDF model

        self.trained = true;
        Ok(())
    }

    pub fn predict(&self, document: &str) -> Result<Array2<f32>> {
        if !self.trained {
            return Err(anyhow::anyhow!("Model must be trained before prediction"));
        }

        let features = self.vectorize_documents(&[document.to_string()])?;
        // ... prediction logic

        Ok(Array2::zeros((1, 2))) // Placeholder
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }

    fn vectorize_documents(&self, documents: &[String]) -> Result<Array2<f32>> {
        let mut features = Array2::zeros((documents.len(), self.vocabulary.len()));

        for (doc_idx, doc) in documents.iter().enumerate() {
            let words = self.tokenize(doc);
            let mut word_counts = HashMap::new();

            for word in words {
                *word_counts.entry(word).or_insert(0) += 1;
            }

            for (word, count) in word_counts {
                if let Some(&vocab_idx) = self.vocabulary.get(&word) {
                    features[[doc_idx, vocab_idx]] = count as f32;
                }
            }
        }

        Ok(features)
    }
}
