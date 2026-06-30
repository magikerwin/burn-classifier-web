// TODO: Remove this warning suppression once the model is integrated into the training loop.
#![allow(dead_code)]

use burn::{
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::activation::relu,
};

const IMAGE_WIDTH: usize = 28;
const IMAGE_HEIGHT: usize = 28;
const NUM_CLASSES: usize = 10;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        let fc1 = LinearConfig::new(IMAGE_WIDTH * IMAGE_HEIGHT, 128).init(device);
        let fc2 = LinearConfig::new(128, NUM_CLASSES).init(device);
        Self { fc1, fc2 }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        let x = relu(x);
        self.fc2.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_model_forward() {
        type TestBackend = NdArray;
        let device = Default::default();

        let model = Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 2>::zeros([4, IMAGE_WIDTH * IMAGE_HEIGHT], &device);
        let output = model.forward(input);

        let shape = output.shape();
        assert_eq!(shape.dims, [4, NUM_CLASSES]);
    }
}
