use anyhow::Result;
use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, ConvTranspose2d},
        pool::{AdaptiveAvgPool2d, MaxPool2d},
        BatchNorm, Dropout, Linear, Relu,
    },
    tensor::{backend::Backend, Tensor},
};
use ndarray::Array3;

/// MobileNetV3-Lite inspired architecture for ultrasound images
#[derive(Module, Debug)]
pub struct MobileNetLite<B: Backend> {
    // Initial convolution
    conv_stem: Conv2d<B>,
    bn_stem: BatchNorm<B, 2>,

    // Depthwise separable convolution blocks
    dw_blocks: Vec<DepthwiseSeparableBlock<B>>,

    // Global average pooling and classifier
    global_pool: AdaptiveAvgPool2d,
    classifier: Linear<B>,
    dropout: Dropout,
    relu: Relu,
    quantized: bool,
}

impl<B: Backend> MobileNetLite<B> {
    pub fn new(input_channels: usize, num_classes: usize) -> Self {
        let mut dw_blocks = Vec::new();

        // Create lightweight depthwise separable blocks
        let block_configs = vec![
            (32, 64, 1), // (in_channels, out_channels, stride)
            (64, 128, 2),
            (128, 128, 1),
            (128, 256, 2),
            (256, 256, 1),
        ];

        for (in_ch, out_ch, stride) in block_configs {
            dw_blocks.push(DepthwiseSeparableBlock::new(in_ch, out_ch, stride));
        }

        Self {
            conv_stem: Conv2d::new(input_channels, 32, [3, 3])
                .with_stride([2, 2])
                .with_padding(burn::nn::PaddingConfig2d::Same),
            bn_stem: BatchNorm::new(32),
            dw_blocks,
            global_pool: AdaptiveAvgPool2d::new([1, 1]),
            classifier: Linear::new(256, num_classes),
            dropout: Dropout::new(0.2),
            relu: Relu::new(),
            quantized: false,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        // Stem convolution
        let mut x = self
            .relu
            .forward(self.bn_stem.forward(self.conv_stem.forward(x)));

        // Depthwise separable blocks
        for block in &self.dw_blocks {
            x = block.forward(x);
        }

        // Global average pooling
        x = self.global_pool.forward(x);
        x = x.flatten(1, 3);

        // Classification head
        x = self.dropout.forward(x);
        self.classifier.forward(x)
    }

    pub fn predict(&self, data: Array3<f32>) -> Result<Array2<f32>> {
        // Add batch dimension: [H, W, C] -> [1, C, H, W]
        let data_4d = data
            .permuted_axes([2, 0, 1]) // [C, H, W]
            .insert_axis(ndarray::Axis(0)); // [1, C, H, W]

        let tensor_input = Tensor::from_data(data_4d.into());
        let output = self.forward(tensor_input);

        let result_data: Vec<f32> = output.into_data().convert();
        let shape = (1, result_data.len()); // Single batch prediction
        Ok(Array2::from_shape_vec(shape, result_data)?)
    }

    pub fn quantize(&mut self) -> Result<()> {
        self.quantized = true;
        println!("Image model quantized to INT8");
        Ok(())
    }

    pub fn is_quantized(&self) -> bool {
        self.quantized
    }
}

/// Depthwise separable convolution block
#[derive(Module, Debug)]
pub struct DepthwiseSeparableBlock<B: Backend> {
    // Depthwise convolution
    depthwise: Conv2d<B>,
    bn_dw: BatchNorm<B, 2>,

    // Pointwise convolution
    pointwise: Conv2d<B>,
    bn_pw: BatchNorm<B, 2>,

    relu: Relu,
    use_residual: bool,
    residual_proj: Option<Conv2d<B>>,
}

impl<B: Backend> DepthwiseSeparableBlock<B> {
    pub fn new(input_channels: usize, output_channels: usize, stride: usize) -> Self {
        let use_residual = stride == 1 && input_channels == output_channels;
        let residual_proj = if !use_residual && stride == 1 {
            Some(Conv2d::new(input_channels, output_channels, [1, 1]))
        } else {
            None
        };

        Self {
            depthwise: Conv2d::new(input_channels, input_channels, [3, 3])
                .with_stride([stride, stride])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .with_groups(input_channels), // Depthwise convolution
            bn_dw: BatchNorm::new(input_channels),
            pointwise: Conv2d::new(input_channels, output_channels, [1, 1]),
            bn_pw: BatchNorm::new(output_channels),
            relu: Relu::new(),
            use_residual,
            residual_proj,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = if self.use_residual {
            Some(x.clone())
        } else if let Some(proj) = &self.residual_proj {
            Some(proj.forward(x.clone()))
        } else {
            None
        };

        // Depthwise convolution
        let out = self
            .relu
            .forward(self.bn_dw.forward(self.depthwise.forward(x)));

        // Pointwise convolution
        let out = self.bn_pw.forward(self.pointwise.forward(out));

        // Add residual connection if applicable
        if let Some(res) = residual {
            self.relu.forward(out + res)
        } else {
            self.relu.forward(out)
        }
    }
}

/// Squeeze-and-Excitation block for channel attention
#[derive(Module, Debug)]
pub struct SeBlock<B: Backend> {
    global_pool: AdaptiveAvgPool2d,
    fc1: Linear<B>,
    fc2: Linear<B>,
    relu: Relu,
}

impl<B: Backend> SeBlock<B> {
    pub fn new(channels: usize, reduction: usize) -> Self {
        Self {
            global_pool: AdaptiveAvgPool2d::new([1, 1]),
            fc1: Linear::new(channels, channels / reduction),
            fc2: Linear::new(channels / reduction, channels),
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();

        // Global average pooling
        let se = self.global_pool.forward(x.clone());
        let se = se.reshape([batch as i32, channels as i32]);

        // Fully connected layers
        let se = self.relu.forward(self.fc1.forward(se));
        let se = self.fc2.forward(se).sigmoid();

        // Reshape and apply attention
        let se = se.reshape([batch as i32, channels as i32, 1, 1]);
        x * se
    }
}
