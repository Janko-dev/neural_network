use std::error::Error;

use neural_network::Matrix;

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

    
    // let a = Matrix::from_vec(vec![0.4], (1, 1));
    
    // a.print();

    // let c = a.sigmoid();

    // c.print();

    // c.print_comp_tree();

    // println!("\nGrads");
    // let grads = c.backward()?;

    // grads.get(a.id()).unwrap().print();

    let w = Matrix::randn(0., 1., (2, 6));
    let b = Matrix::randn(0., 1., (2, 1));

    let x = Matrix::from_vec(vec![2., 1., -3., 4., -1., 0.5], (6, 1));

    let y = w.matmul(&x)?.add(&b)?.sigmoid();

    y.print();

    y.print_comp_tree();

    let grads = y.backward()?;
    grads.get(w.id()).unwrap().print();
    w.print();

    let w = w.sub(grads.get(w.id()).unwrap())?;

    Ok(())
    
}
