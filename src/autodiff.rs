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
pub enum Operator {
    Binary(Matrix, Matrix, BinaryOpType),
    Unary(Matrix, UnaryOpType)
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
            Operator::Binary(lhs, rhs, _) => {
                let (vis, nodes) = visit(lhs, nodes, already_seen);
                visited |= vis;
                
                let (vis, nodes) = visit(rhs, nodes, already_seen);
                visited |= vis;
                nodes
            },
            Operator::Unary(mat, _) => {
                let (vis, nodes) = visit(mat, nodes, already_seen);
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


impl Matrix {

    pub fn topological_sort(&self) -> Vec<&Matrix> {
        let (_, mut sorted_nodes) = visit(self, vec![], &mut HashMap::new()); 
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
                    _ => {}
                }
            }
        }

        Ok(grads)
    }
}