use std::error::Error;

use crate::Matrix;

#[derive(Debug)]
pub enum Activation {
    Sigmoid,
    None,
}

#[derive(Debug)]
pub struct Layer {
    w: Matrix,
    b: Matrix,
    act_func: Activation
}

#[derive(Debug)]
pub struct NN {
    layers: Vec<Layer>,
    learning_rate: f32
}

impl Activation {
    pub fn apply(&self, mat: Matrix) -> Matrix {
        match self {
            Self::Sigmoid => mat.sigmoid(),
            Self::None => mat
        }
    }
}

impl NN {
    pub fn new(config: Vec<usize>, learning_rate: f32) -> Self {
        let mut layers = vec![];
        for (inp, outp) in config
                                        .windows(2)
                                        .map(|x| (x[0], x[1])) 
        {
            let w = Matrix::randn(0., 1., (outp, inp));
            let b = Matrix::randn(0., 1., (outp, 1));
            layers.push(Layer { w, b, act_func: Activation::Sigmoid});
        }
        NN { layers, learning_rate }
    }

    pub fn forward(&mut self, xs: Matrix) -> Result<Matrix, Box<dyn Error>>{
        let mut ys = xs;
        for layer in self.layers.iter() {
            ys = layer.act_func.apply(layer.w.matmul(&ys)?.add(&layer.b)?);
        }

        Ok(ys)
    }

    pub fn train(&mut self, x_train: &Vec<Vec<f32>>, y_train: &Vec<f32>) -> Result<f32, Box<dyn Error>> {
        
        let mut losses = vec![];
        for (idx, data) in x_train.iter().enumerate() {
            let xs: Matrix = data.into();
            let ys = Matrix::from_vec(vec![y_train[idx]], (1, 1));
            
            let ys_pred = self.forward(xs)?;
            let loss = ys_pred.sub(&ys)?.powf(2.);
            
            let grads = loss.backward()?;

            // loss.print_comp_tree();
            
            for layer in self.layers.iter_mut() {
                let dw = grads.get(layer.w.id()).unwrap();
                let db = grads.get(layer.b.id()).unwrap();

                layer.w = layer.w.sub(&dw.mul_scalar(self.learning_rate))?.no_history();
                layer.b = layer.b.sub(&db.mul_scalar(self.learning_rate))?.no_history();

                // layer.w = layer.w.sub(&grads.get(layer.w.id()).unwrap().mul_scalar(self.learning_rate))?;
                // layer.b = layer.b.sub(&grads.get(layer.b.id()).unwrap().mul_scalar(self.learning_rate))?;
            }

            losses.push(loss.get(0, 0));
        }
        let res: f32 = losses.into_iter().sum();

        Ok(res/y_train.len() as f32)
    }
}