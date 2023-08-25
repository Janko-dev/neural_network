use std::{rc::Rc, usize};
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
        Transpose,
        Broadcast,
        Sum
    },
    BinaryScalarOpType::{
        MulScalar,
        Powf32
    }, 
    error::MatrixError::{
        BroadCastError,
        ShapeMismatchError, 
        self
    }
};
use std::sync::atomic::{AtomicUsize, Ordering};

// get unique id 
fn get_id() -> usize {
    static COUNTER:AtomicUsize = AtomicUsize::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug)]
struct Matrix_ {
    id: usize,
    data: Rc<Vec<f32>>,
    shape: (usize, usize), // (rows, cols)
    with_grad: bool,
    optype: Option<Operator>
}

#[derive(Debug, Clone)]
pub struct Matrix(Rc<Matrix_>);

type MatrixResult = Result<Matrix, MatrixError>;

impl Matrix_ {
    fn randn(
        from: f32, 
        to: f32, 
        (m, n): (usize, usize), 
        with_grad: bool
    ) -> Self {  
        
        let size = m*n;

        let mut rng = rand::thread_rng();
        let data = (0..size)
            .map(|_| rng.gen_range(from..to))
            .collect::<Vec<f32>>();
        
        Self { 
            id: get_id(), 
            data: Rc::new(data),
            shape: (m, n),
            with_grad,
            optype: None 
        }
    }

    fn new(
        data: Vec<f32>, 
        (m, n): (usize, usize), 
        op: Option<Operator>, 
        with_grad: bool
    ) -> Self {  
        
        Self { 
            id: get_id(), 
            data: Rc::new(data),
            shape: (m, n),
            with_grad,
            optype: op 
        }
    }

}

macro_rules! binary_operator {
    ($name: ident, $op: tt, $op_type: expr) => {
        
        pub fn $name(&self, other: &Self) -> MatrixResult {

            let shape = Matrix::broadcast_shape(self.shape(), other.shape());

            // println!("{:?}, {:?} => {:?}", self.shape(), other.shape(), shape);

            let l_broadcast = shape != self.shape();
            let r_broadcast = shape != other.shape();

            // println!("{}, {}, {:?}", l_broadcast, r_broadcast, shape);

            let (lhs, rhs) = match (l_broadcast, r_broadcast) {
                (true, true) =>    (self.broadcast_as(shape)?, other.broadcast_as(shape)?),
                (false, true) =>   (self.clone(), other.broadcast_as(shape)?),
                (true, false) =>   (self.broadcast_as(shape)?, other.clone()),
                (false, false) =>  (self.clone(), other.clone()),
            };

            // if self.shape() != other.shape() {
            //     return Err(ShapeMismatchError{
            //         a_shape: self.shape(),
            //         b_shape: other.shape(),
            //         op: stringify!($name).to_string()
            //     });
            // }
    
            let (rows, cols) = shape;
    
            let mut data = vec![0.; rows * cols];
            // println!("lhs: {:?}", lhs.shape());
            // lhs.print();
            // println!("rhs: {:?}", rhs.shape());
            // rhs.print();
            for i in 0..rows {
                for j in 0..cols {
                    data[i * cols + j] = lhs.get(i, j) $op rhs.get(i, j);
                }
            }

            let req_grad = lhs.requires_grad() || rhs.requires_grad();
            let op = Some(Operator::Binary(lhs, rhs, $op_type));
    
            Ok(Self(Rc::new(Matrix_::new(data, shape, op, req_grad))))
        }
        
    };
}


impl Matrix {

    pub fn ones(shape: (usize, usize), with_grad: bool) -> Self {
        let data = vec![1.; shape.0 * shape.1];
        Self(Rc::new(Matrix_::new(data, shape, None, with_grad)))
    }

    pub fn zeros(shape: (usize, usize), with_grad: bool) -> Self {
        let data = vec![0.; shape.0 * shape.1];
        Self(Rc::new(Matrix_::new(data, shape, None, with_grad)))
    }

    pub fn fill(shape: (usize, usize), value: f32, with_grad: bool) -> Self {
        let data = vec![value; shape.0 * shape.1];
        Self(Rc::new(Matrix_::new(data, shape, None, with_grad)))
    }

    pub fn randn(from: f32, to: f32, shape: (usize, usize), with_grad: bool) -> Self {
        Self(Rc::new(Matrix_::randn(from, to, shape, with_grad)))
    }

    pub fn from_vec(data: Vec<f32>, shape: (usize, usize), with_grad: bool) -> Self {
        Self(Rc::new(Matrix_::new(data, shape, None, with_grad)))
    }

    pub fn print(&self) {
        
        let (rows, cols) = self.shape();
        println!("[");
        for x in self.0.data.chunks(cols) {
            println!("    {:?}", x);
        }
        println!("]");
        println!("Shape: ({}, {})", rows, cols);
    }

    fn _print_comp_tree(&self, indent: usize) {
        let mat_info = format!("matrix id: {}, use grad: {}, shape: {:?}", self.id(), self.requires_grad(), self.shape());
        if let Some(op) = self.op() {
            println!("{:indent$}{} with op: {}", " ", mat_info, op.to_string(), indent=indent);
            match op {
                Operator::Binary(lhs, rhs, _) => {
                    lhs._print_comp_tree(indent+4);
                    rhs._print_comp_tree(indent+4);
                },
                Operator::Unary(val, _) | 
                Operator::BinaryScalar(val, _, _) => {
                    val._print_comp_tree(indent+4);
                }
            }
        } else {
            println!("{:indent$}{}", " ", mat_info, indent=indent);
        }
    }

    pub fn print_comp_tree(&self) {
        
        self._print_comp_tree(0);
    }

    pub fn no_history(&self) -> Self{
        Self(Rc::new(Matrix_::new(self.data().clone(), self.shape(), None, self.requires_grad())))
    }

    pub fn shape(&self) -> (usize, usize) {
        self.0.shape
    }

    pub fn id(&self) -> usize {
        self.0.id
    }

    pub fn op(&self) -> &Option<Operator>{
        &self.0.optype
    }

    pub fn requires_grad(&self) -> bool {
        self.0.with_grad
    }

    pub fn data(&self) -> &Vec<f32> {
        &self.0.data
    }

    pub fn get(&self, row_i: usize, col_j: usize) -> f32 {
        let (_, cols) = self.shape();
        self.0.data[row_i * cols + col_j]
    }

    pub fn matmul(&self, other: &Self) -> MatrixResult {

        let (a_rows, a_cols) = self.shape();
        let (b_rows, b_cols) = other.shape();

        if a_cols != b_rows {
            return Err(ShapeMismatchError { 
                a_shape: (a_rows, a_cols), 
                b_shape: (b_rows, b_cols), 
                op: "matmul".to_string() 
            });
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

        Ok(Self(Rc::new(Matrix_::new(data, shape, op, self.requires_grad() || other.requires_grad()))))
    }

    pub fn mul_scalar(&self, other: f32) -> Self {

        let (rows, cols) = self.shape();
    
        let mut data = vec![0.; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = self.get(i, j) * other;
            }   
        }

        let op = Some(Operator::BinaryScalar(self.clone(), other, MulScalar));

        Self(Rc::new(Matrix_::new(data, self.shape(), op, self.requires_grad())))
    }

    pub fn powf(&self, other: f32) -> Self {

        let (rows, cols) = self.shape();
    
        let mut data = vec![0.; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = self.get(i, j).powf(other);
            }   
        }

        let op = Some(Operator::BinaryScalar(self.clone(), other, Powf32));

        Self(Rc::new(Matrix_::new(data, self.shape(), op, self.requires_grad())))
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

        // exchange cols and rows to get transpose matrix
        Self(Rc::new(Matrix_::new(data, (cols, rows), op, self.requires_grad())))
    }

    pub fn sigmoid(&self) -> Matrix {
        let (rows, cols) = self.shape();
        let mut data = vec![0.; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] += _sigmoid(self.get(i, j));
            }   
        }
        let op = Some(Operator::Unary(self.clone(), Sigmoid));

        Self(Rc::new(Matrix_::new(data, (rows, cols), op, self.requires_grad())))
    }

    pub fn broadcast_as(&self, (rows, cols): (usize, usize)) -> MatrixResult {
    
        // (1, 2) => (3, 2)

        // [1, 2] => [1, 2]
        //           [1, 2]
        //           [1, 2]

        // (2, 1) => (2, 3)
        
        // [4] => [4, 4, 4]
        // [5]    [5, 5, 5]


        // (1, 1) => (1, 4)

        // [2] => 

        let data = match (self.shape(), (rows, cols)) {
            ((1, 1), (2..=usize::MAX, 1)) | 
            ((1, 2..=usize::MAX), _) => {
                let mut data = vec![];
                for _ in 0..rows {
                    data.extend(self.data().iter());
                }
                data
            },
            ((1, 1), (1, 2..=usize::MAX)) |
            ((2..=usize::MAX, 1), _) => {
                let mut data = vec![0.; rows * cols];
                for i in 0..rows {
                    for j in 0..cols {
                        data[i * cols + j] = self.get(i, 0);
                    }
                }
                data
            },
            _ => {
                return Err(BroadCastError { 
                    got_shape: self.shape(), 
                    expected_shape: (rows, cols) 
                });
            }
        };
        
        let op = Some(Operator::Unary(self.clone(), Broadcast));

        Ok(Self(Rc::new(Matrix_::new(data, (rows, cols), op, self.requires_grad()))))
    }

    pub fn sum(&self, axis: usize) -> MatrixResult {
        
        let (rows, cols) = self.shape();

        let (data, new_shape) = match axis {
            0 => {
                let shape = (rows, 1 as usize);

                let mut data = vec![0.; rows];
                for i in 0..rows {
                    for j in 0..cols {
                        data[i] += self.get(i, j);
                    }
                }
                (data, shape)
            },
            1 => {
                let shape = (1 as usize, cols);

                let mut data = vec![0.; cols];
                for j in 0..cols {
                    for i in 0..rows {
                        data[j] += self.get(i, j);
                    }
                }
                (data, shape)
            }
            _ => (self.data().clone(), (rows, cols))
        };
        
        let op = Some(Operator::Unary(self.clone(), Sum));

        Ok(Self(Rc::new(Matrix_::new(data, new_shape, op, self.requires_grad()))))
    }

    pub fn broadcast_shape(lhs: (usize, usize), rhs: (usize, usize)) -> (usize, usize) {
        
        let rhs = match rhs {
            (1, 1) => lhs,
            (1, x) => (lhs.0, x),
            (x, 1) => (x, lhs.1),
            x => x
        };
    
        let lhs = match lhs {
            (1, 1) => rhs,
            (1, x) => (rhs.0, x),
            (x, 1) => (x, rhs.1),
            x => x
        };
    
        if rhs == lhs {
            rhs
        } else {
            (0, 0)
        }
    }

}

fn _sigmoid(x: f32) -> f32 {
    1./(1. + (-x).exp())
}

impl Into<Matrix> for Vec<f32> {
    fn into(self) -> Matrix {
        let len = self.len();
        Matrix::from_vec(self, (len, 1), false)
    }
}

impl Into<Matrix> for &Vec<f32> {
    fn into(self) -> Matrix {
        let len = self.len();
        Matrix::from_vec(self.clone(), (len, 1), false)
    }
}

impl Into<Matrix> for Vec<Vec<f32>> {
    fn into(self) -> Matrix {
        let rows = self.len();
        let cols = if let Some(cols) = self.get(0) {
            cols.len()
        } else {
            0
        };

        for v in self.iter() {
            if v.len() == cols {
                continue;
            } else {
                return Matrix::from_vec(vec![], (0, 0), false);
            }
        }
        
        let data = self
            .into_iter()
            .flat_map(|v| v.into_iter())
            .collect::<Vec<f32>>();

        Matrix::from_vec(data, (rows, cols), false)
    }
}

impl Into<Matrix> for &Vec<Vec<f32>> {
    fn into(self) -> Matrix {
        let rows = self.len();
        let cols = if let Some(cols) = self.get(0) {
            cols.len()
        } else {
            0
        };

        for v in self.iter() {
            if v.len() == cols {
                continue;
            } else {
                return Matrix::from_vec(vec![], (0, 0), false);
            }
        }
        
        let data = self.clone()
            .into_iter()
            .flat_map(|v| v.into_iter())
            .collect::<Vec<f32>>();

        Matrix::from_vec(data, (rows, cols), false)
    }
}