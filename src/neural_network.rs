use std::error::Error;

use rand::seq::SliceRandom;

use crate::{Matrix, error::NNError};

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
            let w = Matrix::randn(0., 1., (outp, inp), true);
            let b = Matrix::randn(0., 1., (outp, 1), true);
            layers.push(Layer { w, b, act_func: Activation::Sigmoid});
        }
        NN { layers, learning_rate }
    }

    pub fn forward(&self, xs: Matrix) -> Result<Matrix, Box<dyn Error>>{

        let mut ys = xs;
        for layer in self.layers.iter() {
            ys = layer.act_func.apply(layer.w.matmul(&ys)?.add(&layer.b)?);
            
        }

        Ok(ys)
    }

    pub fn train(&mut self, 
        x_train: &Vec<Vec<f32>>, 
        y_train: &Vec<f32>, 
        batch_size: usize, 
        epochs: usize) 
        -> Result<Vec<f32>, Box<dyn Error>> {
        
        if x_train.len() != y_train.len() {
            return Err(Box::new(NNError::TrainDataMismatch { 
                x_size: x_train.len(),
                y_size: y_train.len()
            }));
        }

        let training_size = y_train.len();

        if batch_size > training_size {
            return Err(Box::new(NNError::BatchSizeExceedsTrainingSet { 
                batch_size, 
                train_size: training_size  
            }));
        }

        let mut rng = rand::thread_rng();
        let mut history = vec![];

        let mut counter: usize = 0;

        for _ in 0..epochs {

            let (batch_x, batch_y): (Vec<Vec<f32>>, Vec<f32>) = if batch_size == 1 {
                let x = vec![x_train[counter].clone()];
                let y = vec![y_train[counter]];
                counter = (counter+1) % training_size;
                (x, y)
            } else {
                (0..training_size)
                    .collect::<Vec<usize>>()
                    .choose_multiple(&mut rng, batch_size)
                    .into_iter()
                    .map(|i| (x_train[*i].clone(), y_train[*i]))
                    .unzip()
            };

            let batch_x: Matrix = batch_x.into();
            let batch_y: Matrix = batch_y.into();
            let ys_pred = self.forward(batch_x.t())?.t();

            let loss = ys_pred.sub(&batch_y)?.powf(2.);
    
            let grads = loss.backward()?;
    
            for layer in self.layers.iter_mut() {
                let dw = grads.get(layer.w.id()).unwrap();
                let db = grads.get(layer.b.id()).unwrap();
    
                layer.w = layer.w.sub(&dw.mul_scalar(self.learning_rate))?.no_history();
                layer.b = layer.b.sub(&db.mul_scalar(self.learning_rate))?.no_history();
            }
    
            let loss = loss.sum(1)?.get(0, 0);
            history.push(loss/batch_size as f32);

        }

        Ok(history)
    }
}