use std::{error::Error, fmt::Display};

#[derive(Debug)]
pub enum MatrixError {
    ShapeMismatchError {
        a_shape: (usize, usize),
        b_shape: (usize, usize),
        op: String,
    },
    BroadCastError {
        got_shape: (usize, usize),
        expected_shape: (usize, usize),
    }
}

impl Error for MatrixError {}

impl Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatrixError::ShapeMismatchError { a_shape, b_shape, op } => 
                writeln!(f, "Shape mismatch error during [{}] operation: shape {:?} of self does not match shape {:?} of other",
                    op, a_shape, b_shape    
                ),
            MatrixError::BroadCastError { got_shape, expected_shape } =>
                writeln!(f, "Broadcast error: could not broadcast shape {:?} into shape {:?}",
                    got_shape, expected_shape    
                )
        }
    }
}

#[derive(Debug)]
pub enum NNError {
    TrainDataMismatch {
        x_size: usize,
        y_size: usize
    },
    BatchSizeExceedsTrainingSet {
        batch_size: usize,
        train_size: usize,
    }
}

impl Error for NNError {}

impl Display for NNError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NNError::TrainDataMismatch { x_size, y_size } => 
                writeln!(f, "Size mismatch error: size of x_train is {}, whereas size of y_train is {}",
                    x_size, y_size    
                ),
            NNError::BatchSizeExceedsTrainingSet { batch_size, train_size } =>
                writeln!(f, "Batch size mismatch error: batch size must be less than training data records, batch size is {} and training data size is {}",
                    batch_size, train_size    
                )
        }
    }
}