use std::{rc::Rc, collections::HashMap, hash::Hash};
use rand::prelude::*;

#[derive(Debug, Clone)]
pub enum Operator {
    Add(Matrix, Matrix),
    Mul(Matrix, Matrix), 
    MatMul(Matrix, Matrix),
    Transpose(Matrix)
}

// #[derive(Debug, Clone)]
// enum OpType {
//     Binary(Tensor, Tensor, Operator),
//     Unary(Tensor, Operator)
// }

// #[derive(Debug)]
// pub struct Storage(Vec<f32>);

// impl Storage {
//     fn new_between(from: f32, to: f32, size: usize) -> Self {
//         let mut rng = rand::thread_rng();
//         let data = (0..size)
//             .map(|x| rng.gen_range(from..to))
//             .collect::<Vec<f32>>();
//         Storage(data)
//     }

//     // pub fn matmul 
// }

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
        // a_rows = 1, a_cols = 1
        // b_rows = 1, b_cols = 2
        // 
        let shape = (a_rows, b_cols);
        let mut data = vec![0.; a_rows * b_cols];
        for i in 0..a_rows {
            for j in 0..b_cols {
                for k in 0..a_cols {
                    data[i * b_cols + j] += self.get(i, k) * other.get(k, j);
                }
            }   
        }

        let op = Some(Operator::MatMul(self.clone(), other.clone()));

        Ok(Self(Rc::new(Matrix_::new(data, shape, op))))
    }

    pub fn add(&self, other: &Self) -> Result<Self, String> {

        if self.shape() != other.shape() {
            return Err("Shape mismatch error: columns or rows of self do not match columns or rows of other".to_string());
        }

        let (rows, cols) = self.shape();

        let mut data = vec![0.; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = self.get(i, j) + other.get(i, j);
            }   
        }

        let op = Some(Operator::Add(self.clone(), other.clone()));

        Ok(Self(Rc::new(Matrix_::new(data, self.shape(), op))))
    }

    pub fn mul(&self, other: &Self) -> Result<Self, String> {

        if self.shape() != other.shape() {
            return Err("Shape mismatch error: columns or rows of self do not match columns or rows of other".to_string());
        }

        let (rows, cols) = self.shape();

        let mut data = vec![0.; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = self.get(i, j) * other.get(i, j);
            }   
        }

        let op = Some(Operator::Mul(self.clone(), other.clone()));

        Ok(Self(Rc::new(Matrix_::new(data, self.shape(), op))))
    }

    pub fn t(&self) -> Matrix {
        let (rows, cols) = self.shape();
        let mut data = vec![0.; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] += self.get(i, j);
            }   
        }
        let op = Some(Operator::Transpose(self.clone()));

        Self(Rc::new(Matrix_::new(data, (cols, rows), op)))
    }

    pub fn visit<'a>(
        node: &'a Matrix, 
        nodes: Vec<&'a Matrix>, 
        already_seen: &mut HashMap<usize, bool>
    ) -> (bool, Vec<&'a Matrix>) {

        if let Some(&visited) = already_seen.get(&node.id()) {
            return (visited, nodes);
        }
        let mut visited = false;
        let mut nodes = if node.is_variable() {
            visited = true;
            nodes
        } else if let Some(op) = node.op() {
            match op {
                Operator::Add(lhs, rhs) | 
                Operator::Mul(lhs, rhs) | 
                Operator::MatMul(lhs, rhs) => {
                    let (vis, nodes) = Matrix::visit(lhs, nodes, already_seen);
                    visited |= vis;
                    
                    let (vis, nodes) = Matrix::visit(rhs, nodes, already_seen);
                    visited |= vis;
                    nodes
                },
                Operator::Transpose(mat) => {
                    let (vis, nodes) = Matrix::visit(mat, nodes, already_seen);
                    visited |= vis;
                    nodes
                }
            }
        } else {
            nodes
        };

        already_seen.insert(node.id(), visited);
        if visited {
            nodes.push(node);
        }
        (visited, nodes)
    }

    pub fn topological_sort(&self) -> Vec<&Matrix> {
        let (_, mut sorted_nodes) = Matrix::visit(self, vec![], &mut HashMap::new()); 
        sorted_nodes.reverse();
        sorted_nodes
    }

    pub fn backward(&self) -> Result<GradStore, String> {
        
        let sorted_nodes = self.topological_sort(); 
        let mut grads = GradStore::new();
        grads.insert(self, Matrix::ones(self.shape()));

        for node in sorted_nodes.iter() {
            // if node.is_variable() {
            //     continue;
            // }
            let grad = grads.remove(node).unwrap();
            if let Some(op) = node.op() {
                match op {
                    Operator::MatMul(lhs, rhs) => {
                        // C = A @ B
                        // 
                        // A' += [1] @ B
                        // B' += A.T @ [1]
                        let lhs_grad = grad.matmul(&rhs.t())?;
                        let lhs_sum_grad = grads.or_insert(lhs);
                        *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;

                        let rhs_grad = lhs.t().matmul(&grad)?;
                        let rhs_sum_grad = grads.or_insert(rhs);
                        *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                    },
                    Operator::Add(lhs, rhs) => {

                    },
                    Operator::Mul(lhs, rhs) => {

                    },
                    Operator::Transpose(mat) => {

                    }
                }
            }
        }
        println!("{:?}", sorted_nodes.len());

        Ok(grads)
    }

}

use std::sync::atomic::{AtomicUsize, Ordering};
fn get_id() -> usize {
    static COUNTER:AtomicUsize = AtomicUsize::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug)]
pub struct GradStore(HashMap<usize, Matrix>);

impl GradStore {
    pub fn new() -> Self {
        GradStore(HashMap::new())
    }

    pub fn get(&self, id: usize) -> Option<&Matrix> {
        self.0.get(&id)
    }

    pub fn insert(&mut self, mat: &Matrix, grad: Matrix) -> Option<Matrix> {
        self.0.insert(mat.id(), grad)
    }

    pub fn remove(&mut self, mat: &Matrix) -> Option<Matrix> {
        self.0.remove(&mat.id())
    }

    fn or_insert(&mut self, mat: &Matrix) -> &mut Matrix {
        use std::collections::hash_map::Entry;
        let grad = match self.0.entry(mat.id()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let grad = Matrix::zeros(mat.shape());
                entry.insert(grad)
            }
        };
        grad
    }
}