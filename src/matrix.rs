use std::rc::Rc;
use rand::prelude::*;
use crate::{
    Operator, 
    BinaryOpType::{
        Add,
        Sub,
        Mul,
        Div,
        MatMul,
    },
    UnaryOpType::{
        Sigmoid,
        Transpose
    }
};

// #[derive(Debug, Clone)]
// enum OpType {
//     Binary(Tensor, Tensor, Operator),
//     Unary(Tensor, Operator)
// }

use std::sync::atomic::{AtomicUsize, Ordering};

fn get_id() -> usize {
    static COUNTER:AtomicUsize = AtomicUsize::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug)]
struct Matrix_ {
    id: usize,
    data: Rc<Vec<f32>>,
    shape: (usize, usize), // (rows, cols)
    is_var: bool,
    optype: Option<Operator>
}

#[derive(Debug, Clone)]
pub struct Matrix(Rc<Matrix_>);

impl Matrix_ {
    fn randn(from: f32, to: f32, (m, n): (usize, usize)) -> Self {  
        
        let size = m*n;

        let mut rng = rand::thread_rng();
        let data = (0..size)
            .map(|_| rng.gen_range(from..to))
            .collect::<Vec<f32>>();
        
        Self { 
            id: get_id(), 
            data: Rc::new(data),
            shape: (m, n),
            is_var: false, 
            optype: None 
        }
    }

    fn new(data: Vec<f32>, (m, n): (usize, usize), op: Option<Operator>) -> Self {  
        
        Self { 
            id: get_id(), 
            data: Rc::new(data),
            shape: (m, n),
            is_var: true, 
            optype: op 
        }
    }
}

macro_rules! binary_operator {
    ($name: ident, $op: tt, $op_type: expr) => {
        
        pub fn $name(&self, other: &Self) -> Result<Self, String> {

            if self.shape() != other.shape() {
                return Err("Shape mismatch error: columns or rows of self do not match columns or rows of other".to_string());
            }
    
            let (rows, cols) = self.shape();
    
            let mut data = vec![0.; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    data[i * cols + j] = self.get(i, j) $op other.get(i, j);
                }   
            }
    
            let op = Some(Operator::Binary(self.clone(), other.clone(), $op_type));
    
            Ok(Self(Rc::new(Matrix_::new(data, self.shape(), op))))
        }
        
    };
}


impl Matrix {

    pub fn ones(shape: (usize, usize)) -> Self {
        let data = vec![1.; shape.0 * shape.1];
        Self(Rc::new(Matrix_::new(data, shape, None)))
    }

    pub fn zeros(shape: (usize, usize)) -> Self {
        let data = vec![0.; shape.0 * shape.1];
        Self(Rc::new(Matrix_::new(data, shape, None)))
    }

    pub fn randn(from: f32, to: f32, shape: (usize, usize)) -> Self {
        Self(Rc::new(Matrix_::randn(from, to, shape)))
    }

    pub fn from_vec(data: Vec<f32>, shape: (usize, usize)) -> Self {
        Self(Rc::new(Matrix_::new(data, shape, None)))
    }

    pub fn print(&self) {
        
        let (_, cols) = self.shape();
        println!("[");
        for x in self.0.data.chunks(cols) {
            println!("    {:?}", x);
        }
        println!("]");
    }

    pub fn shape(&self) -> (usize, usize) {
        self.0.shape
    }

    pub fn id(&self) -> usize {
        self.0.id
    }

    pub fn op(&self) -> &Option<Operator> {
        &self.0.optype
    }

    pub fn is_variable(&self) -> bool {
        self.0.is_var
    }

    pub fn data(&self) -> &Vec<f32> {
        &self.0.data
    }

    pub fn get(&self, row_i: usize, col_j: usize) -> f32 {
        let (_, cols) = self.shape();
        self.0.data[row_i * cols + col_j]
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, String> {

        let (a_rows, a_cols) = self.shape();
        let (b_rows, b_cols) = other.shape();

        if a_cols != b_rows {
            return Err("Shape mismatch error: columns of self do not match rows of other".to_string());
        }
        let shape = (a_rows, b_cols);
        let mut data = vec![0.; a_rows * b_cols];
        for i in 0..a_rows {
            for j in 0..b_cols {
                for k in 0..a_cols {
                    data[i * b_cols + j] += self.get(i, k) * other.get(k, j);
                }
            }   
        }

        let op = Some(Operator::Binary(self.clone(), other.clone(), MatMul));

        Ok(Self(Rc::new(Matrix_::new(data, shape, op))))
    }

    binary_operator!(add, +, Add);
    binary_operator!(mul, *, Mul);
    binary_operator!(sub, -, Sub);
    binary_operator!(div, /, Div);
    

    pub fn t(&self) -> Matrix {
        let (rows, cols) = self.shape();
        let mut data = vec![0.; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] += self.get(i, j);
            }   
        }
        let op = Some(Operator::Unary(self.clone(), Transpose));

        Self(Rc::new(Matrix_::new(data, (cols, rows), op)))
    }

}