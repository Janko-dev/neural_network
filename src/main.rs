use std::error::Error;

use neural_network::{Matrix, GradStore};

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

    
    // let a = Matrix::from_vec(vec![1., 2.], (1, 2));    
    // let b = Matrix::from_vec(vec![2., 3.], (2, 1));
    
    // a.print();
    // b.print();

    // let c = a.matmul(&b)?;
    // c.print();

    // let grads = c.backward()?;
    // grads.get(a.id()).unwrap().print();
    // grads.get(b.id()).unwrap().print();

    println!("{}", 0. / 0.);

    Ok(())
    
}
