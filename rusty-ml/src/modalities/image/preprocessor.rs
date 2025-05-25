use crate::utils::preprocessing::Preprocessor;
use anyhow::Result;
use image::{imageops::FilterType, DynamicImage, ImageBuffer, Luma, Rgb};
use ndarray::{Array3, Array4, Axis};

/// Image preprocessor for ultrasound and medical images
#[derive(Clone)]
pub struct ImagePreprocessor {
    target_size: (u32, u32),
    normalization_method: ImageNormalization,
    augmentation_config: AugmentationConfig,
    contrast_enhancement: bool,
    noise_reduction: bool,
    normalization_params: Option<(f32, f32)>, // (mean, std)
    fitted: bool,
}

#[derive(Clone)]
pub enum ImageNormalization {
    MinMax,    // Scale to [0, 1]
    ZScore,    // (x - mean) / std
    Histogram, // Histogram equalization
    CLAHE,     // Contrast Limited Adaptive Histogram Equalization
}

#[derive(Clone)]
pub struct AugmentationConfig {
    pub rotation_range: f32,
    pub zoom_range: (f32, f32),
    pub brightness_range: (f32, f32),
    pub contrast_range: (f32, f32),
    pub horizontal_flip: bool,
    pub vertical_flip: bool,
    pub noise_level: f32,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            rotation_range: 15.0,
            zoom_range: (0.9, 1.1),
            brightness_range: (0.8, 1.2),
            contrast_range: (0.8, 1.2),
            horizontal_flip: true,
            vertical_flip: false,
            noise_level: 0.05,
        }
    }
}

impl ImagePreprocessor {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            target_size: (width, height),
            normalization_method: ImageNormalization::ZScore,
            augmentation_config: AugmentationConfig::default(),
            contrast_enhancement: true,
            noise_reduction: true,
            normalization_params: None,
            fitted: false,
        }
    }

    pub fn with_normalization(mut self, method: ImageNormalization) -> Self {
        self.normalization_method = method;
        self
    }

    pub fn with_augmentation(mut self, config: AugmentationConfig) -> Self {
        self.augmentation_config = config;
        self
    }

    /// Process a single image from DynamicImage to normalized Array3
    pub fn process_image(&self, img: DynamicImage) -> Result<Array3<f32>> {
        // Resize image
        let resized =
            img.resize_exact(self.target_size.0, self.target_size.1, FilterType::Triangle);

        // Convert to grayscale for ultrasound images (medical images often grayscale)
        let gray_img = resized.to_luma8();

        // Apply preprocessing steps
        let enhanced_img = if self.contrast_enhancement {
            self.enhance_contrast(&gray_img)?
        } else {
            gray_img
        };

        let denoised_img = if self.noise_reduction {
            self.reduce_noise(&enhanced_img)?
        } else {
            enhanced_img
        };

        // Convert to array
        let mut img_array = self.image_to_array(&denoised_img)?;

        // Apply normalization
        img_array = self.normalize_image(img_array)?;

        Ok(img_array)
    }

    /// Convert ImageBuffer to Array3<f32>
    fn image_to_array(&self, img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Result<Array3<f32>> {
        let (width, height) = img.dimensions();
        let mut array = Array3::zeros((1, height as usize, width as usize)); // CHW format

        for (x, y, pixel) in img.enumerate_pixels() {
            array[[0, y as usize, x as usize]] = pixel.0[0] as f32;
        }

        Ok(array)
    }

    /// Convert RGB ImageBuffer to Array3<f32>
    fn rgb_image_to_array(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Array3<f32>> {
        let (width, height) = img.dimensions();
        let mut array = Array3::zeros((3, height as usize, width as usize)); // CHW format

        for (x, y, pixel) in img.enumerate_pixels() {
            array[[0, y as usize, x as usize]] = pixel.0[0] as f32; // R
            array[[1, y as usize, x as usize]] = pixel.0[1] as f32; // G
            array[[2, y as usize, x as usize]] = pixel.0[2] as f32; // B
        }

        Ok(array)
    }

    /// Enhance contrast using histogram equalization
    fn enhance_contrast(
        &self,
        img: &ImageBuffer<Luma<u8>, Vec<u8>>,
    ) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>> {
        let (width, height) = img.dimensions();
        let mut histogram = vec![0u32; 256];

        // Compute histogram
        for pixel in img.pixels() {
            histogram[pixel.0[0] as usize] += 1;
        }

        // Compute cumulative distribution
        let total_pixels = (width * height) as f32;
        let mut cdf = vec![0.0f32; 256];
        cdf[0] = histogram[0] as f32 / total_pixels;

        for i in 1..256 {
            cdf[i] = cdf[i - 1] + histogram[i] as f32 / total_pixels;
        }

        // Apply histogram equalization
        let mut enhanced = ImageBuffer::new(width, height);
        for (x, y, pixel) in img.enumerate_pixels() {
            let old_value = pixel.0[0] as usize;
            let new_value = (cdf[old_value] * 255.0) as u8;
            enhanced.put_pixel(x, y, Luma([new_value]));
        }

        Ok(enhanced)
    }

    /// Reduce noise using Gaussian blur
    fn reduce_noise(
        &self,
        img: &ImageBuffer<Luma<u8>, Vec<u8>>,
    ) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>> {
        let (width, height) = img.dimensions();
        let mut denoised = ImageBuffer::new(width, height);

        // Simple 3x3 Gaussian kernel
        let kernel = [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]];
        let kernel_sum = 16.0;

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let mut sum = 0.0;

                for ky in 0..3 {
                    for kx in 0..3 {
                        let px = (x as i32 + kx as i32 - 1) as u32;
                        let py = (y as i32 + ky as i32 - 1) as u32;
                        let pixel_value = img.get_pixel(px, py).0[0] as f32;
                        sum += pixel_value * kernel[ky][kx];
                    }
                }

                let filtered_value = (sum / kernel_sum) as u8;
                denoised.put_pixel(x, y, Luma([filtered_value]));
            }
        }

        // Copy border pixels
        for y in 0..height {
            denoised.put_pixel(0, y, *img.get_pixel(0, y));
            denoised.put_pixel(width - 1, y, *img.get_pixel(width - 1, y));
        }
        for x in 0..width {
            denoised.put_pixel(x, 0, *img.get_pixel(x, 0));
            denoised.put_pixel(x, height - 1, *img.get_pixel(x, height - 1));
        }

        Ok(denoised)
    }

    /// Normalize image array
    fn normalize_image(&self, mut img_array: Array3<f32>) -> Result<Array3<f32>> {
        match self.normalization_method {
            ImageNormalization::MinMax => {
                let min_val = img_array.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = img_array.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                if max_val > min_val {
                    img_array = (img_array - min_val) / (max_val - min_val);
                }
            }
            ImageNormalization::ZScore => {
                if let Some((mean, std)) = self.normalization_params {
                    img_array = (img_array - mean) / std;
                } else {
                    let mean = img_array.mean().unwrap();
                    let std = img_array.std(1.0);
                    if std > 1e-8 {
                        img_array = (img_array - mean) / std;
                    }
                }
            }
            _ => {} // Other methods would be implemented here
        }

        Ok(img_array)
    }

    /// Apply data augmentation
    pub fn augment_image(&self, img_array: Array3<f32>) -> Result<Vec<Array3<f32>>> {
        let mut augmented_images = vec![img_array.clone()];

        // Horizontal flip
        if self.augmentation_config.horizontal_flip {
            let flipped = self.flip_horizontal(&img_array)?;
            augmented_images.push(flipped);
        }

        // Rotation
        if self.augmentation_config.rotation_range > 0.0 {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let angle = rng.gen_range(
                -self.augmentation_config.rotation_range..self.augmentation_config.rotation_range,
            );
            let rotated = self.rotate_image(&img_array, angle)?;
            augmented_images.push(rotated);
        }

        // Brightness adjustment
        if self.augmentation_config.brightness_range.0 != 1.0
            || self.augmentation_config.brightness_range.1 != 1.0
        {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let brightness_factor = rng.gen_range(
                self.augmentation_config.brightness_range.0
                    ..self.augmentation_config.brightness_range.1,
            );
            let brightened = self.adjust_brightness(&img_array, brightness_factor)?;
            augmented_images.push(brightened);
        }

        // Contrast adjustment
        if self.augmentation_config.contrast_range.0 != 1.0
            || self.augmentation_config.contrast_range.1 != 1.0
        {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let contrast_factor = rng.gen_range(
                self.augmentation_config.contrast_range.0
                    ..self.augmentation_config.contrast_range.1,
            );
            let contrasted = self.adjust_contrast(&img_array, contrast_factor)?;
            augmented_images.push(contrasted);
        }

        // Add noise
        if self.augmentation_config.noise_level > 0.0 {
            let noisy = self.add_noise(&img_array, self.augmentation_config.noise_level)?;
            augmented_images.push(noisy);
        }

        // Zoom
        if self.augmentation_config.zoom_range.0 != 1.0
            || self.augmentation_config.zoom_range.1 != 1.0
        {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let zoom_factor = rng.gen_range(
                self.augmentation_config.zoom_range.0..self.augmentation_config.zoom_range.1,
            );
            let zoomed = self.zoom_image(&img_array, zoom_factor)?;
            augmented_images.push(zoomed);
        }

        Ok(augmented_images)
    }

    /// Flip image horizontally
    fn flip_horizontal(&self, img_array: &Array3<f32>) -> Result<Array3<f32>> {
        let (channels, height, width) = img_array.dim();
        let mut flipped = Array3::zeros((channels, height, width));

        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    flipped[[c, h, w]] = img_array[[c, h, width - 1 - w]];
                }
            }
        }

        Ok(flipped)
    }

    /// Rotate image by given angle (degrees)
    fn rotate_image(&self, img_array: &Array3<f32>, angle: f32) -> Result<Array3<f32>> {
        let (channels, height, width) = img_array.dim();
        let mut rotated = Array3::zeros((channels, height, width));

        let angle_rad = angle.to_radians();
        let cos_angle = angle_rad.cos();
        let sin_angle = angle_rad.sin();

        let center_x = width as f32 / 2.0;
        let center_y = height as f32 / 2.0;

        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    // Translate to origin
                    let x = w as f32 - center_x;
                    let y = h as f32 - center_y;

                    // Rotate
                    let rotated_x = x * cos_angle - y * sin_angle;
                    let rotated_y = x * sin_angle + y * cos_angle;

                    // Translate back
                    let source_x = rotated_x + center_x;
                    let source_y = rotated_y + center_y;

                    // Bilinear interpolation
                    if source_x >= 0.0
                        && source_x < width as f32
                        && source_y >= 0.0
                        && source_y < height as f32
                    {
                        let x1 = source_x.floor() as usize;
                        let y1 = source_y.floor() as usize;
                        let x2 = (x1 + 1).min(width - 1);
                        let y2 = (y1 + 1).min(height - 1);

                        let fx = source_x - x1 as f32;
                        let fy = source_y - y1 as f32;

                        let interpolated = img_array[[c, y1, x1]] * (1.0 - fx) * (1.0 - fy)
                            + img_array[[c, y1, x2]] * fx * (1.0 - fy)
                            + img_array[[c, y2, x1]] * (1.0 - fx) * fy
                            + img_array[[c, y2, x2]] * fx * fy;

                        rotated[[c, h, w]] = interpolated;
                    }
                }
            }
        }

        Ok(rotated)
    }

    /// Adjust brightness
    fn adjust_brightness(&self, img_array: &Array3<f32>, factor: f32) -> Result<Array3<f32>> {
        let mut brightened = img_array.clone();
        brightened *= factor;

        // Clamp values to valid range
        for value in brightened.iter_mut() {
            *value = value.max(0.0).min(1.0);
        }

        Ok(brightened)
    }

    /// Adjust contrast
    fn adjust_contrast(&self, img_array: &Array3<f32>, factor: f32) -> Result<Array3<f32>> {
        let mean = img_array.mean().unwrap();
        let mut contrasted = img_array.clone();

        // Apply contrast: new_value = mean + factor * (old_value - mean)
        for value in contrasted.iter_mut() {
            *value = mean + factor * (*value - mean);
            *value = value.max(0.0).min(1.0); // Clamp to valid range
        }

        Ok(contrasted)
    }

    /// Add Gaussian noise
    fn add_noise(&self, img_array: &Array3<f32>, noise_level: f32) -> Result<Array3<f32>> {
        use rand::Rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, noise_level).unwrap();
        let mut noisy = img_array.clone();

        for value in noisy.iter_mut() {
            let noise = normal.sample(&mut rng);
            *value += noise;
            *value = value.max(0.0).min(1.0); // Clamp to valid range
        }

        Ok(noisy)
    }

    /// Zoom image (crop and resize)
    fn zoom_image(&self, img_array: &Array3<f32>, zoom_factor: f32) -> Result<Array3<f32>> {
        let (channels, height, width) = img_array.dim();
        let mut zoomed = Array3::zeros((channels, height, width));

        let new_height = (height as f32 / zoom_factor) as usize;
        let new_width = (width as f32 / zoom_factor) as usize;

        let start_h = (height - new_height) / 2;
        let start_w = (width - new_width) / 2;

        // Simple nearest neighbor resampling
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let source_h = start_h + (h * new_height / height);
                    let source_w = start_w + (w * new_width / width);

                    if source_h < height && source_w < width {
                        zoomed[[c, h, w]] = img_array[[c, source_h, source_w]];
                    }
                }
            }
        }

        Ok(zoomed)
    }

    /// Extract medical image features (texture, edges, etc.)
    pub fn extract_medical_features(&self, img_array: &Array3<f32>) -> Result<Array3<f32>> {
        let (channels, height, width) = img_array.dim();
        let mut features = Array3::zeros((channels * 4, height, width)); // Original + 3 feature maps

        // Copy original image
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    features[[c, h, w]] = img_array[[c, h, w]];
                }
            }
        }

        // Extract edges using Sobel operators
        let sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
        let sobel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

        for c in 0..channels {
            for h in 1..(height - 1) {
                for w in 1..(width - 1) {
                    let mut grad_x = 0.0;
                    let mut grad_y = 0.0;

                    for ky in 0..3 {
                        for kx in 0..3 {
                            let pixel = img_array[[c, h + ky - 1, w + kx - 1]];
                            grad_x += pixel * sobel_x[ky][kx];
                            grad_y += pixel * sobel_y[ky][kx];
                        }
                    }

                    // Edge magnitude
                    let edge_magnitude = (grad_x * grad_x + grad_y * grad_y).sqrt();
                    features[[channels + c, h, w]] = edge_magnitude;

                    // Edge direction
                    let edge_direction = grad_y.atan2(grad_x);
                    features[[2 * channels + c, h, w]] = edge_direction;
                }
            }
        }

        // Local Binary Pattern (simplified)
        for c in 0..channels {
            for h in 1..(height - 1) {
                for w in 1..(width - 1) {
                    let center = img_array[[c, h, w]];
                    let mut lbp = 0.0;

                    let neighbors = [
                        img_array[[c, h - 1, w - 1]],
                        img_array[[c, h - 1, w]],
                        img_array[[c, h - 1, w + 1]],
                        img_array[[c, h, w + 1]],
                        img_array[[c, h + 1, w + 1]],
                        img_array[[c, h + 1, w]],
                        img_array[[c, h + 1, w - 1]],
                        img_array[[c, h, w - 1]],
                    ];

                    for (i, &neighbor) in neighbors.iter().enumerate() {
                        if neighbor >= center {
                            lbp += 2.0_f32.powi(i as i32);
                        }
                    }

                    features[[3 * channels + c, h, w]] = lbp / 255.0; // Normalize
                }
            }
        }

        Ok(features)
    }
}

impl Preprocessor for ImagePreprocessor {
    type Input = Vec<DynamicImage>;
    type Output = Array4<f32>;

    fn fit(&mut self, data: &Self::Input) {
        if data.is_empty() {
            return;
        }

        // Compute normalization parameters across all images
        let mut all_values = Vec::new();

        for img in data.iter().take(100) {
            // Sample first 100 images for efficiency
            if let Ok(processed) = self.process_image(img.clone()) {
                all_values.extend(processed.iter().cloned());
            }
        }

        if !all_values.is_empty() {
            let mean = all_values.iter().sum::<f32>() / all_values.len() as f32;
            let variance = all_values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
                / all_values.len() as f32;
            let std = variance.sqrt();

            self.normalization_params = Some((mean, std));
        }

        self.fitted = true;
    }

    fn transform(&self, data: Self::Input) -> Self::Output {
        if !self.fitted {
            panic!("ImagePreprocessor must be fitted before transform");
        }

        let batch_size = data.len();
        if batch_size == 0 {
            return Array4::zeros((
                0,
                1,
                self.target_size.1 as usize,
                self.target_size.0 as usize,
            ));
        }

        // Process first image to get dimensions
        let first_processed = self
            .process_image(data[0].clone())
            .expect("Failed to process first image");
        let (channels, height, width) = first_processed.dim();

        let mut batch_array = Array4::zeros((batch_size, channels, height, width));

        for (i, img) in data.iter().enumerate() {
            if let Ok(processed) = self.process_image(img.clone()) {
                batch_array
                    .slice_mut(ndarray::s![i, .., .., ..])
                    .assign(&processed);
            }
        }

        batch_array
    }
}

/// Specialized preprocessor for different types of medical images
pub struct MedicalImagePreprocessor {
    ultrasound_processor: ImagePreprocessor,
    xray_processor: ImagePreprocessor,
    ct_processor: ImagePreprocessor,
}

impl MedicalImagePreprocessor {
    pub fn new() -> Self {
        Self {
            // Ultrasound images - typically lower resolution, more noise
            ultrasound_processor: ImagePreprocessor::new(224, 224)
                .with_normalization(ImageNormalization::CLAHE)
                .with_augmentation(AugmentationConfig {
                    rotation_range: 10.0,
                    zoom_range: (0.95, 1.05),
                    brightness_range: (0.9, 1.1),
                    contrast_range: (0.9, 1.1),
                    horizontal_flip: true,
                    vertical_flip: false,
                    noise_level: 0.02,
                }),

            // X-ray images - high contrast, need careful preprocessing
            xray_processor: ImagePreprocessor::new(512, 512)
                .with_normalization(ImageNormalization::Histogram)
                .with_augmentation(AugmentationConfig {
                    rotation_range: 5.0,
                    zoom_range: (0.98, 1.02),
                    brightness_range: (0.95, 1.05),
                    contrast_range: (0.95, 1.05),
                    horizontal_flip: false,
                    vertical_flip: false,
                    noise_level: 0.01,
                }),

            // CT images - already normalized, minimal preprocessing
            ct_processor: ImagePreprocessor::new(256, 256)
                .with_normalization(ImageNormalization::ZScore)
                .with_augmentation(AugmentationConfig {
                    rotation_range: 15.0,
                    zoom_range: (0.9, 1.1),
                    brightness_range: (0.8, 1.2),
                    contrast_range: (0.8, 1.2),
                    horizontal_flip: true,
                    vertical_flip: true,
                    noise_level: 0.03,
                }),
        }
    }

    pub fn process_ultrasound(&mut self, images: Vec<DynamicImage>) -> Result<Array4<f32>> {
        if !self.ultrasound_processor.fitted {
            self.ultrasound_processor.fit(&images);
        }
        Ok(self.ultrasound_processor.transform(images))
    }

    pub fn process_xray(&mut self, images: Vec<DynamicImage>) -> Result<Array4<f32>> {
        if !self.xray_processor.fitted {
            self.xray_processor.fit(&images);
        }
        Ok(self.xray_processor.transform(images))
    }

    pub fn process_ct(&mut self, images: Vec<DynamicImage>) -> Result<Array4<f32>> {
        if !self.ct_processor.fitted {
            self.ct_processor.fit(&images);
        }
        Ok(self.ct_processor.transform(images))
    }
}
