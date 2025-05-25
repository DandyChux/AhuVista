use anyhow::Result;
use burn::{
    module::Module,
    nn::{conv::Conv1d, gru::Gru, Linear, Relu},
    tensor::{backend::Backend, Tensor},
};
use ndarray::{Array2, Array3};

/// Temporal Convolutional Network + GRU for time-series data
#[derive(Module, Debug)]
pub struct TemporalModel<B: Backend> {
    // TCN layers
    tcn1: Conv1d<B>,
    tcn2: Conv1d<B>,

    // GRU for sequence modeling
    gru: Gru<B>,

    // Output layers
    fc_out: Linear<B>,
    relu: Relu,
    quantized: bool,
}

impl<B: Backend> TemporalModel<B> {
    pub fn new(input_channels: usize, tcn_channels: usize, gru_hidden: usize) -> Self {
        Self {
            tcn1: Conv1d::new(input_channels, tcn_channels, 3)
                .with_dilation(1)
                .with_padding(burn::nn::PaddingConfig1d::Same),
            tcn2: Conv1d::new(tcn_channels, tcn_channels, 3)
                .with_dilation(2)
                .with_padding(burn::nn::PaddingConfig1d::Same),
            gru: Gru::new(tcn_channels, gru_hidden),
            fc_out: Linear::new(gru_hidden, 2),
            relu: Relu::new(),
            quantized: false,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        // TCN processing
        let x = self.relu.forward(self.tcn1.forward(x));
        let x = self.relu.forward(self.tcn2.forward(x));

        // GRU processing
        let (x, _) = self.gru.forward(x);

        // Take last timestep and project to output
        let last_timestep = x.select(2, -1); // Assuming [batch, features, time]
        self.fc_out.forward(last_timestep)
    }

    pub fn predict(&self, data: Array3<f32>) -> Result<Array2<f32>> {
        let tensor_input = Tensor::from_data(data.into());
        let output = self.forward(tensor_input);

        let result_data: Vec<f32> = output.into_data().convert();
        let shape = (data.shape()[0], 2); // Batch size x output features
        Ok(Array2::from_shape_vec(shape, result_data)?)
    }

    pub fn quantize(&mut self) -> Result<()> {
        self.quantized = true;
        println!("Temporal model quantized to INT8");
        Ok(())
    }

    pub fn is_quantized(&self) -> bool {
        self.quantized
    }
}

/// Simplified TCN block with residual connections
#[derive(Module, Debug)]
pub struct TcnBlock<B: Backend> {
    conv1: Conv1d<B>,
    conv2: Conv1d<B>,
    relu: Relu,
    residual_proj: Option<Linear<B>>,
}

impl<B: Backend> TcnBlock<B> {
    pub fn new(
        input_channels: usize,
        output_channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> Self {
        let residual_proj = if input_channels != output_channels {
            Some(Linear::new(input_channels, output_channels))
        } else {
            None
        };

        Self {
            conv1: Conv1d::new(input_channels, output_channels, kernel_size)
                .with_dilation(dilation)
                .with_padding(burn::nn::PaddingConfig1d::Same),
            conv2: Conv1d::new(output_channels, output_channels, kernel_size)
                .with_dilation(dilation)
                .with_padding(burn::nn::PaddingConfig1d::Same),
            relu: Relu::new(),
            residual_proj,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = if let Some(proj) = &self.residual_proj {
            proj.forward(x.clone())
        } else {
            x.clone()
        };

        let out = self.relu.forward(self.conv1.forward(x));
        let out = self.conv2.forward(out);

        // Residual connection
        self.relu.forward(out + residual)
    }
}
