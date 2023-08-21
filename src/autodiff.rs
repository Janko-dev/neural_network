use std::collections::HashMap;

use crate::Matrix;

#[derive(Debug, Clone)]
pub enum BinaryOpType {
    Add,
    Mul,
    Sub,
    Div,
    MatMul,
}

#[derive(Debug, Clone)]
pub enum UnaryOpType {
    Transpose,
    Sigmoid
}

#[derive(Debug, Clone)]
pub enum BinaryScalarOpType {
    MulScalar,
    Powf32
}

#[derive(Debug, Clone)]
pub enum Operator {
    Binary(Matrix, Matrix, BinaryOpType),
    BinaryScalar(Matrix, f32, BinaryScalarOpType),
    Unary(Matrix, UnaryOpType)
}

impl Operator {
    pub fn to_string(&self) -> String {
        match self {
            Self::Binary(_, _, BinaryOpType::Add) => "Add".to_string(),
            Self::Binary(_, _, BinaryOpType::Sub) => "Sub".to_string(),
            Self::Binary(_, _, BinaryOpType::Mul) => "Mul".to_string(),
            Self::Binary(_, _, BinaryOpType::Div) => "Div".to_string(),
            Self::Binary(_, _, BinaryOpType::MatMul) => "MatMul".to_string(),
            
            Self::BinaryScalar(_, _, BinaryScalarOpType::MulScalar) => "MulScalar".to_string(),
            Self::BinaryScalar(_, _, BinaryScalarOpType::Powf32) => "MulScalar".to_string(),
            
            Self::Unary(_, UnaryOpType::Sigmoid) => "Sigmoid".to_string(),
            Self::Unary(_, UnaryOpType::Transpose) => "Transpose".to_string(),
            
        }
    } 
}

#[derive(Debug)]
pub struct GradMap(HashMap<usize, Matrix>);

impl GradMap {
    pub fn new() -> Self {
        Self(HashMap::new())
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

    pub fn or_insert(&mut self, mat: &Matrix) -> &mut Matrix {
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

fn visit<'a>(
    node: &'a Matrix, 
    nodes: Vec<&'a Matrix>, 
    already_seen: &mut HashMap<usize, bool>
) -> Vec<&'a Matrix> {

    already_seen.insert(node.id(), true);

    let mut nodes = if let Some(op) = node.op() {
        match op {
            Operator::Binary(lhs, rhs, _) => {
                let nodes = visit(lhs, nodes, already_seen);
                let nodes = visit(rhs, nodes, already_seen);
                nodes
            },
            Operator::Unary(mat, _) |
            Operator::BinaryScalar(mat, _, _) => {
                let nodes = visit(mat, nodes, already_seen);
                nodes
            }
        }
    } else {
        return nodes;
    };
    nodes.push(node);
    nodes
}


impl Matrix {

    pub fn topological_sort(&self) -> Vec<&Matrix> {
        let mut sorted_nodes = visit(self, vec![], &mut HashMap::new()); 
        sorted_nodes.reverse();
        sorted_nodes
    }

    pub fn backward(&self) -> Result<GradMap, String> {
        
        let sorted_nodes = self.topological_sort();
        // println!("{:?}", sorted_nodes.clone().into_iter().map(|x| x.id()).collect::<Vec<usize>>());
        let mut grads = GradMap::new();
        grads.insert(self, Matrix::ones(self.shape()));

        for node in sorted_nodes.iter() {
            if !node.is_variable() {
                continue;
            }
            let grad = grads.remove(node).unwrap();
            if let Some(op) = node.op() {
                match op {
                    Operator::Binary(lhs, rhs, BinaryOpType::MatMul) => {
                        // C = A @ B
                        // 
                        // A' += C' @ B
                        // B' += A.T @ C'
                        let lhs_grad = grad.matmul(&rhs.t())?;
                        let lhs_sum_grad = grads.or_insert(lhs);
                        *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;

                        let rhs_grad = lhs.t().matmul(&grad)?;
                        let rhs_sum_grad = grads.or_insert(rhs);
                        *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                    },
                    Operator::Binary(lhs, rhs, BinaryOpType::Add) => {
                        let lhs_sum_grad = grads.or_insert(lhs);
                        *lhs_sum_grad = lhs_sum_grad.add(&grad)?;

                        let rhs_sum_grad = grads.or_insert(rhs);
                        *rhs_sum_grad = rhs_sum_grad.add(&grad)?;
                    },
                    Operator::Binary(lhs, rhs, BinaryOpType::Sub) => {
                        let lhs_sum_grad = grads.or_insert(lhs);
                        *lhs_sum_grad = lhs_sum_grad.add(&grad)?;

                        let rhs_grad = grad.mul(&Matrix::fill(rhs.shape(), -1.))?;
                        let rhs_sum_grad = grads.or_insert(rhs);
                        *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                    },
                    Operator::Binary(lhs, rhs, BinaryOpType::Mul) => {
                        let lhs_grad = grad.mul(&rhs)?;
                        let lhs_sum_grad = grads.or_insert(lhs);
                        *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;

                        let rhs_grad = grad.mul(&lhs)?;
                        let rhs_sum_grad = grads.or_insert(rhs);
                        *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                    },
                    Operator::Binary(lhs, rhs, BinaryOpType::Div) => {
                        // c = a/b
                        // a/b => a' = 1/b
                        // a/b => b' = -a * b^(-2) => -a/b^2
                        // 
                        // x/5 = 1/5 * x = 1/5
                        // 5/x = 5 * 1/x = 5 * x^(-2) = 5 * -1 * x^(-2) = -5 * x^(-2) => -5/x^2
                        let lhs_grad = grad.mul(&Matrix::ones(rhs.shape()).div(&rhs)?)?;
                        let lhs_sum_grad = grads.or_insert(lhs);
                        *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;

                        let negative_lhs = lhs.mul(&Matrix::fill(lhs.shape(), -1.))?;
                        let rhs_squared = rhs.mul(&rhs)?;
                        let rhs_grad = grad.mul(&negative_lhs.div(&rhs_squared)?)?;
                        let rhs_sum_grad = grads.or_insert(rhs);
                        *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                    },
                    Operator::Unary(mat, UnaryOpType::Transpose) => {
                        let mat_grad = grad.t();
                        let mat_sum_grad = grads.or_insert(mat);
                        *mat_sum_grad = mat_sum_grad.add(&mat_grad)?;
                    },
                    Operator::Unary(mat, UnaryOpType::Sigmoid) => {
                        let sigmoid_der = node.mul(&Matrix::ones(node.shape()).sub(&node)?)?;                        
                        let mat_grad = grad.mul(&sigmoid_der)?;
                        let mat_sum_grad = grads.or_insert(mat);
                        *mat_sum_grad = mat_sum_grad.add(&mat_grad)?;
                    },
                    Operator::BinaryScalar(lhs, rhs, BinaryScalarOpType::MulScalar) => {
                        let lhs_grad = grad.mul_scalar(*rhs);
                        let lhs_sum_grad = grads.or_insert(lhs);
                        *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
                    },
                    Operator::BinaryScalar(lhs, rhs, BinaryScalarOpType::Powf32) => {
                        // x^2 => 
                        let lhs_grad = grad.mul_scalar(*rhs).mul(&lhs.powf((*rhs)-1.))?;
                        let lhs_sum_grad = grads.or_insert(lhs);
                        *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
                    },
                }
            }
            grads.insert(node, grad);
        }

        Ok(grads)
    }
}