use crate::utils::preprocessing::Preprocessor;
use anyhow::Result;
use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};

/// Text preprocessor for clinical notes and medical text data
#[derive(Clone)]
pub struct TextPreprocessor {
    vocabulary: HashMap<String, usize>,
    max_features: usize,
    max_sequence_length: usize,
    min_word_frequency: usize,
    stop_words: HashSet<String>,
    clinical_abbreviations: HashMap<String, String>,
    text_cleaning_config: TextCleaningConfig,
    fitted: bool,
}

#[derive(Clone)]
pub struct TextCleaningConfig {
    pub lowercase: bool,
    pub remove_punctuation: bool,
    pub remove_numbers: bool,
    pub remove_stop_words: bool,
    pub expand_abbreviations: bool,
    pub remove_special_chars: bool,
    pub normalize_whitespace: bool,
}

impl Default for TextCleaningConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: true,
            remove_numbers: false, // Keep numbers for medical data
            remove_stop_words: true,
            expand_abbreviations: true,
            remove_special_chars: true,
            normalize_whitespace: true,
        }
    }
}

impl TextPreprocessor {
    pub fn new(max_features: usize, max_sequence_length: usize) -> Self {
        let stop_words = Self::get_medical_stop_words();
        let clinical_abbreviations = Self::get_clinical_abbreviations();

        Self {
            vocabulary: HashMap::new(),
            max_features,
            max_sequence_length,
            min_word_frequency: 2,
            stop_words,
            clinical_abbreviations,
            text_cleaning_config: TextCleaningConfig::default(),
            fitted: false,
        }
    }

    pub fn with_cleaning_config(mut self, config: TextCleaningConfig) -> Self {
        self.text_cleaning_config = config;
        self
    }

    pub fn with_min_frequency(mut self, min_freq: usize) -> Self {
        self.min_word_frequency = min_freq;
        self
    }

    /// Get medical-specific stop words
    fn get_medical_stop_words() -> HashSet<String> {
        let medical_stop_words = [
            // Common English stop words
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "must",
            "a",
            "an",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            // Medical-specific stop words
            "patient",
            "report",
            "noted",
            "showed",
            "appears",
            "seems",
            "suggests",
            "indicates",
            "consistent",
            "compatible",
            "unremarkable",
            "within",
            "normal",
            "limits",
            "stable",
            "unchanged",
            "compared",
            "prior",
            "study",
            "exam",
            "examination",
            "findings",
            "impression",
            "conclusion",
            "recommendation",
            "discussed",
            "reviewed",
            "evaluated",
            "assessed",
            "documented",
            // Time-related terms (often not clinically significant)
            "today",
            "yesterday",
            "tomorrow",
            "now",
            "then",
            "currently",
            "recently",
            "previously",
            "following",
            "during",
            "after",
            "before",
        ];

        medical_stop_words.iter().map(|s| s.to_string()).collect()
    }

    /// Get clinical abbreviations and their expansions
    fn get_clinical_abbreviations() -> HashMap<String, String> {
        let abbreviations = [
            // Pregnancy-related abbreviations
            ("g", "gravida"),
            ("p", "para"),
            ("ab", "abortion"),
            ("lmp", "last menstrual period"),
            ("edd", "expected delivery date"),
            ("ga", "gestational age"),
            ("fhr", "fetal heart rate"),
            ("bpp", "biophysical profile"),
            ("nst", "non stress test"),
            ("cst", "contraction stress test"),
            ("gtt", "glucose tolerance test"),
            ("gbs", "group b streptococcus"),
            // General medical abbreviations
            ("bp", "blood pressure"),
            ("hr", "heart rate"),
            ("rr", "respiratory rate"),
            ("temp", "temperature"),
            ("wt", "weight"),
            ("ht", "height"),
            ("bmi", "body mass index"),
            ("hx", "history"),
            ("px", "physical exam"),
            ("dx", "diagnosis"),
            ("tx", "treatment"),
            ("rx", "prescription"),
            ("sx", "symptoms"),
            ("fx", "fracture"),
            ("ca", "cancer"),
            // Laboratory values
            ("hgb", "hemoglobin"),
            ("hct", "hematocrit"),
            ("wbc", "white blood cells"),
            ("rbc", "red blood cells"),
            ("plt", "platelets"),
            ("bun", "blood urea nitrogen"),
            ("cr", "creatinine"),
            ("gfr", "glomerular filtration rate"),
            ("alt", "alanine transaminase"),
            ("ast", "aspartate transaminase"),
            ("ldh", "lactate dehydrogenase"),
            // Units and measurements
            ("mg", "milligrams"),
            ("mcg", "micrograms"),
            ("ml", "milliliters"),
            ("cc", "cubic centimeters"),
            ("mm", "millimeters"),
            ("cm", "centimeters"),
            ("kg", "kilograms"),
            ("lbs", "pounds"),
            ("bpm", "beats per minute"),
            // Common medical terms
            ("sob", "shortness of breath"),
            ("doe", "dyspnea on exertion"),
            ("cp", "chest pain"),
            ("n/v", "nausea and vomiting"),
            ("urti", "upper respiratory tract infection"),
            ("uti", "urinary tract infection"),
            ("dvt", "deep vein thrombosis"),
            ("pe", "pulmonary embolism"),
            ("mi", "myocardial infarction"),
            ("cva", "cerebrovascular accident"),
        ];

        abbreviations
            .iter()
            .map(|(abbr, expansion)| (abbr.to_string(), expansion.to_string()))
            .collect()
    }

    /// Clean and normalize text
    pub fn clean_text(&self, text: &str) -> String {
        let mut cleaned = text.to_string();

        // Lowercase
        if self.text_cleaning_config.lowercase {
            cleaned = cleaned.to_lowercase();
        }

        // Expand clinical abbreviations
        if self.text_cleaning_config.expand_abbreviations {
            cleaned = self.expand_abbreviations(&cleaned);
        }

        // Remove special characters but keep medical symbols
        if self.text_cleaning_config.remove_special_chars {
            cleaned = self.remove_special_characters(&cleaned);
        }

        // Remove punctuation (but keep some medical punctuation)
        if self.text_cleaning_config.remove_punctuation {
            cleaned = self.remove_punctuation(&cleaned);
        }

        // Remove numbers (optional for medical text)
        if self.text_cleaning_config.remove_numbers {
            cleaned = cleaned.chars().filter(|c| !c.is_ascii_digit()).collect();
        }

        // Normalize whitespace
        if self.text_cleaning_config.normalize_whitespace {
            cleaned = self.normalize_whitespace(&cleaned);
        }

        cleaned
    }

    /// Expand clinical abbreviations
    fn expand_abbreviations(&self, text: &str) -> String {
        let mut expanded = text.to_string();

        for (abbr, expansion) in &self.clinical_abbreviations {
            // Use word boundaries to avoid partial matches
            let pattern = format!(r"\b{}\b", regex::escape(abbr));
            if let Ok(re) = regex::Regex::new(&pattern) {
                expanded = re.replace_all(&expanded, expansion).to_string();
            }
        }

        expanded
    }

    /// Remove special characters while preserving medical notation
    fn remove_special_characters(&self, text: &str) -> String {
        text.chars()
            .filter(|c| {
                c.is_alphanumeric() ||
                c.is_whitespace() ||
                *c == '.' || *c == ',' || *c == ';' || *c == ':' || // Keep some punctuation
                *c == '/' || *c == '-' || *c == '+' || // Medical notation
                *c == '%' || *c == '>' || *c == '<' || *c == '=' // Comparison operators
            })
            .collect()
    }

    /// Remove punctuation while keeping medical-relevant punctuation
    fn remove_punctuation(&self, text: &str) -> String {
        text.chars()
            .filter(|c| {
                !c.is_ascii_punctuation() ||
                *c == '/' || *c == '-' || *c == '+' || // Medical notation
                *c == '%' || *c == '>' || *c == '<' || *c == '=' // Comparison operators
            })
            .collect()
    }

    /// Normalize whitespace
    fn normalize_whitespace(&self, text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Tokenize text into words
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let cleaned = self.clean_text(text);
        let mut tokens: Vec<String> = cleaned.split_whitespace().map(|s| s.to_string()).collect();

        // Remove stop words if configured
        if self.text_cleaning_config.remove_stop_words {
            tokens = tokens
                .into_iter()
                .filter(|token| !self.stop_words.contains(token))
                .collect();
        }

        // Filter by minimum length
        tokens = tokens.into_iter().filter(|token| token.len() > 1).collect();

        tokens
    }

    /// Build vocabulary from corpus
    pub fn build_vocabulary(&mut self, documents: &[String]) -> Result<()> {
        let mut word_counts: HashMap<String, usize> = HashMap::new();

        // Count word frequencies
        for doc in documents {
            let tokens = self.tokenize(doc);
            for token in tokens {
                *word_counts.entry(token).or_insert(0) += 1;
            }
        }

        // Filter by minimum frequency and select top words
        let mut word_freq: Vec<_> = word_counts
            .into_iter()
            .filter(|(_, count)| *count >= self.min_word_frequency)
            .collect();

        // Sort by frequency (descending)
        word_freq.sort_by(|a, b| b.1.cmp(&a.1));

        // Build vocabulary (reserve 0 for padding, 1 for unknown)
        self.vocabulary.insert("<PAD>".to_string(), 0);
        self.vocabulary.insert("<UNK>".to_string(), 1);

        for (i, (word, _)) in word_freq
            .into_iter()
            .take(self.max_features - 2)
            .enumerate()
        {
            self.vocabulary.insert(word, i + 2);
        }

        Ok(())
    }

    /// Convert text to sequence of token IDs
    pub fn text_to_sequence(&self, text: &str) -> Vec<usize> {
        let tokens = self.tokenize(text);
        let mut sequence = Vec::new();

        for token in tokens {
            let token_id = self.vocabulary.get(&token).cloned().unwrap_or(1); // 1 = <UNK>
            sequence.push(token_id);
        }

        // Truncate or pad to max_sequence_length
        if sequence.len() > self.max_sequence_length {
            sequence.truncate(self.max_sequence_length);
        } else {
            while sequence.len() < self.max_sequence_length {
                sequence.push(0); // 0 = <PAD>
            }
        }

        sequence
    }

    /// Convert multiple texts to padded sequences
    pub fn texts_to_sequences(&self, texts: &[String]) -> Array2<usize> {
        let mut sequences = Array2::zeros((texts.len(), self.max_sequence_length));

        for (i, text) in texts.iter().enumerate() {
            let sequence = self.text_to_sequence(text);
            for (j, &token_id) in sequence.iter().enumerate() {
                sequences[[i, j]] = token_id;
            }
        }

        sequences
    }

    /// Extract clinical keywords and their frequencies
    pub fn extract_clinical_keywords(&self, text: &str) -> HashMap<String, usize> {
        let clinical_keywords = [
            // Pregnancy conditions
            "preeclampsia",
            "gestational_diabetes",
            "placenta_previa",
            "placental_abruption",
            "hyperemesis_gravidarum",
            "oligohydramnios",
            "polyhydramnios",
            "intrauterine_growth_restriction",
            "preterm_labor",
            "cervical_insufficiency",
            "ectopic_pregnancy",
            "miscarriage",
            // Symptoms
            "bleeding",
            "cramping",
            "contractions",
            "nausea",
            "vomiting",
            "headache",
            "blurred_vision",
            "swelling",
            "weight_gain",
            "decreased_fetal_movement",
            // Measurements
            "blood_pressure",
            "fundal_height",
            "fetal_heart_rate",
            "cervical_length",
            "amniotic_fluid",
            "estimated_fetal_weight",
            "biophysical_profile",
            // Treatments
            "bed_rest",
            "medication",
            "monitoring",
            "delivery",
            "cesarean",
            "induction",
            "epidural",
            "magnesium_sulfate",
            "corticosteroids",
            "antibiotics",
        ];

        let tokens = self.tokenize(text);
        let mut keyword_counts = HashMap::new();

        for token in tokens {
            if clinical_keywords.contains(&token.as_str()) {
                *keyword_counts.entry(token).or_insert(0) += 1;
            }
        }

        keyword_counts
    }

    /// Extract n-grams from text
    pub fn extract_ngrams(&self, text: &str, n: usize) -> Vec<String> {
        let tokens = self.tokenize(text);
        let mut ngrams = Vec::new();

        for i in 0..=tokens.len().saturating_sub(n) {
            let ngram = tokens[i..i + n].join(" ");
            ngrams.push(ngram);
        }

        ngrams
    }

    /// Calculate TF-IDF features for documents
    pub fn calculate_tfidf(&self, documents: &[String]) -> Result<Array2<f32>> {
        let vocab_size = self.vocabulary.len();
        let num_docs = documents.len();

        // Calculate term frequencies
        let mut tf_matrix = Array2::zeros((num_docs, vocab_size));
        for (doc_idx, doc) in documents.iter().enumerate() {
            let tokens = self.tokenize(doc);
            let total_tokens = tokens.len() as f32;

            for token in tokens {
                if let Some(&token_id) = self.vocabulary.get(&token) {
                    tf_matrix[[doc_idx, token_id]] += 1.0 / total_tokens;
                }
            }
        }

        // Calculate document frequencies
        let mut df_vector = Array1::zeros(vocab_size);
        for doc in documents {
            let tokens: HashSet<String> = self.tokenize(doc).into_iter().collect();
            for token in tokens {
                if let Some(&token_id) = self.vocabulary.get(&token) {
                    df_vector[token_id] += 1.0;
                }
            }
        }

        // Calculate TF-IDF
        let mut tfidf_matrix = Array2::zeros((num_docs, vocab_size));
        for i in 0..num_docs {
            for j in 0..vocab_size {
                let tf = tf_matrix[[i, j]];
                let df = df_vector[j];
                let idf = if df > 0.0 {
                    (num_docs as f32 / df).ln() as f32
                } else {
                    0.0_f32
                };
                tfidf_matrix[[i, j]] = tf * idf;
            }
        }

        Ok(tfidf_matrix)
    }

    /// Detect medical entities (simplified NER)
    pub fn extract_medical_entities(&self, text: &str) -> HashMap<String, Vec<String>> {
        let mut entities = HashMap::new();
        let tokens = self.tokenize(text);

        // Simple pattern-based entity extraction
        let mut i = 0;
        while i < tokens.len() {
            let token = &tokens[i];

            // Medication patterns
            if self.is_medication_indicator(token) && i + 1 < tokens.len() {
                let medication = tokens[i + 1].clone();
                entities
                    .entry("medications".to_string())
                    .or_insert_with(Vec::new)
                    .push(medication);
                i += 2;
                continue;
            }

            // Symptom patterns
            if self.is_symptom_word(token) {
                entities
                    .entry("symptoms".to_string())
                    .or_insert_with(Vec::new)
                    .push(token.clone());
            }

            // Vital signs patterns
            if self.is_vital_sign_pattern(&tokens, i) {
                let vital_sign = self.extract_vital_sign(&tokens, i);
                if let Some(vs) = vital_sign {
                    entities
                        .entry("vital_signs".to_string())
                        .or_insert_with(Vec::new)
                        .push(vs);
                }
            }

            i += 1;
        }

        entities
    }

    fn is_medication_indicator(&self, token: &str) -> bool {
        ["prescribed", "taking", "medication", "drug", "rx"].contains(&token)
    }

    fn is_symptom_word(&self, token: &str) -> bool {
        [
            "pain",
            "bleeding",
            "nausea",
            "vomiting",
            "headache",
            "cramping",
            "swelling",
            "dizziness",
            "fatigue",
            "shortness",
            "breath",
        ]
        .contains(&token)
    }

    fn is_vital_sign_pattern(&self, tokens: &[String], index: usize) -> bool {
        if index + 1 >= tokens.len() {
            return false;
        }

        let current = &tokens[index];
        let next = &tokens[index + 1];

        // Check for "bp 120/80" or "blood pressure 120/80" patterns
        (current == "bp" || current == "blood")
            && (next.contains('/') || next.parse::<f32>().is_ok())
    }

    fn extract_vital_sign(&self, tokens: &[String], index: usize) -> Option<String> {
        if index + 1 >= tokens.len() {
            return None;
        }

        let indicator = &tokens[index];
        let value = &tokens[index + 1];

        Some(format!("{}: {}", indicator, value))
    }

    /// Generate text summary statistics
    pub fn text_statistics(&self, text: &str) -> HashMap<String, f32> {
        let tokens = self.tokenize(text);
        let sentences: Vec<&str> = text.split('.').collect();
        let words: Vec<&str> = text.split_whitespace().collect();

        let mut stats = HashMap::new();
        stats.insert("token_count".to_string(), tokens.len() as f32);
        stats.insert("sentence_count".to_string(), sentences.len() as f32);
        stats.insert("word_count".to_string(), words.len() as f32);
        stats.insert(
            "avg_word_length".to_string(),
            words.iter().map(|w| w.len()).sum::<usize>() as f32 / words.len() as f32,
        );
        stats.insert(
            "avg_sentence_length".to_string(),
            words.len() as f32 / sentences.len() as f32,
        );

        // Calculate clinical keyword density
        let clinical_keywords = self.extract_clinical_keywords(text);
        let clinical_density =
            clinical_keywords.values().sum::<usize>() as f32 / tokens.len() as f32;
        stats.insert("clinical_keyword_density".to_string(), clinical_density);

        stats
    }

    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }

    pub fn get_vocabulary(&self) -> &HashMap<String, usize> {
        &self.vocabulary
    }
}

impl Preprocessor for TextPreprocessor {
    type Input = Vec<String>;
    type Output = Array2<f32>;

    fn fit(&mut self, data: &Self::Input) {
        self.build_vocabulary(data)
            .expect("Failed to build vocabulary");
        self.fitted = true;
    }

    fn transform(&self, data: Self::Input) -> Self::Output {
        if !self.fitted {
            panic!("TextPreprocessor must be fitted before transform");
        }

        // Convert to TF-IDF features
        let tfidf_features = self
            .calculate_tfidf(&data)
            .expect("Failed to calculate TF-IDF features");

        tfidf_features
    }
}

/// Specialized preprocessor for different types of clinical text
pub struct ClinicalTextPreprocessor {
    note_processor: TextPreprocessor,
    discharge_summary_processor: TextPreprocessor,
    lab_report_processor: TextPreprocessor,
}

impl ClinicalTextPreprocessor {
    pub fn new() -> Self {
        Self {
            // Clinical notes - shorter, informal
            note_processor: TextPreprocessor::new(5000, 256)
                .with_cleaning_config(TextCleaningConfig {
                    lowercase: true,
                    remove_punctuation: true,
                    remove_numbers: false,
                    remove_stop_words: true,
                    expand_abbreviations: true,
                    remove_special_chars: false,
                    normalize_whitespace: true,
                })
                .with_min_frequency(2),

            // Discharge summaries - longer, formal
            discharge_summary_processor: TextPreprocessor::new(10000, 512)
                .with_cleaning_config(TextCleaningConfig {
                    lowercase: true,
                    remove_punctuation: false, // Keep structure
                    remove_numbers: false,
                    remove_stop_words: false, // Keep context
                    expand_abbreviations: true,
                    remove_special_chars: false,
                    normalize_whitespace: true,
                })
                .with_min_frequency(3),

            // Lab reports - structured, numeric
            lab_report_processor: TextPreprocessor::new(3000, 128)
                .with_cleaning_config(TextCleaningConfig {
                    lowercase: true,
                    remove_punctuation: false,
                    remove_numbers: false, // Important for lab values
                    remove_stop_words: true,
                    expand_abbreviations: true,
                    remove_special_chars: false, // Keep units and operators
                    normalize_whitespace: true,
                })
                .with_min_frequency(1),
        }
    }

    pub fn process_clinical_notes(&mut self, texts: Vec<String>) -> Result<Array2<f32>> {
        if !self.note_processor.fitted {
            self.note_processor.fit(&texts);
        }
        Ok(self.note_processor.transform(texts))
    }

    pub fn process_discharge_summaries(&mut self, texts: Vec<String>) -> Result<Array2<f32>> {
        if !self.discharge_summary_processor.fitted {
            self.discharge_summary_processor.fit(&texts);
        }
        Ok(self.discharge_summary_processor.transform(texts))
    }

    pub fn process_lab_reports(&mut self, texts: Vec<String>) -> Result<Array2<f32>> {
        if !self.lab_report_processor.fitted {
            self.lab_report_processor.fit(&texts);
        }
        Ok(self.lab_report_processor.transform(texts))
    }
}
