use std::error::Error;

use crate::matrix::Matrix;

mod matrix;

// fn main(){
//     let mut stride: Vec<_> = vec![2, 3]
//         .iter()
//         .rev()
//         .scan(1, |prod, u| {
//             let prod_pre_mult = *prod;
//             *prod *= u;
//             Some(prod_pre_mult)
//         })
//         .collect();
//     stride.reverse();
//     dbg!(stride);
//     // 1 2 3 4 5 6
//     // shape (2, 3)
//     // 1 2 3
//     // 4 5 6

//     // stride (3, 1)
//     // 
// }

fn main() -> Result<(), Box<dyn Error>>{

    // let a = Matrix::from_vec(vec![2.], (1, 1));    
    // let b = Matrix::from_vec(vec![4.], (1, 1));
    
    // a.print();
    // b.print();

    // let c = a.matmul(&b)?;
    // c.print();



    // let a = Matrix::randn(0., 1., (3, 5));
    // let b = Matrix::randn(0., 1., (5, 2));
    let a = Matrix::from_vec(vec![1., 2.], (1, 2));    
    let b = Matrix::from_vec(vec![2., 3.], (2, 1));
    
    a.print();
    b.print();

    let c = a.matmul(&b)?;
    c.print();

    // 1 * 2 + 2 * 3 = 8
    // 1
    // 
    let grads = c.backward()?;
    grads.get(a.id()).unwrap().print();
    grads.get(b.id()).unwrap().print();

    // dbg!(&c);
    // dbg!(&grads);
    Ok(())
    
}
