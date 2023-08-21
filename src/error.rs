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
        op: String
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
            MatrixError::BroadCastError { got_shape, expected_shape, op } =>
                writeln!(f, "Broadcast error during [{}] operation: could not broadcast shape {:?} into shape {:?}",
                    op, got_shape, expected_shape    
                )
        }
    }
}