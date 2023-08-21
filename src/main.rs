use std::error::Error;

use neural_network::{Matrix, NN};

fn main() -> Result<(), Box<dyn Error>>{

    
    // let a = Matrix::from_vec(vec![0.4], (1, 1));
    
    // a.print();

    // let c = a.sigmoid();

    // c.print();

    // c.print_comp_tree();

    // println!("\nGrads");
    // let grads = c.backward()?;

    // grads.get(a.id()).unwrap().print();

    let x_train = vec![
        vec![0., 0.],
        vec![0., 1.],
        vec![1., 0.],
        vec![1., 1.],
    ];

    let y_train = vec![0., 1., 1., 0.];

    let mut nn = NN::new(vec![2, 3, 1], 0.3);
    
    for _ in 0..10 {
        let loss = nn.train(&x_train, &y_train)?;
        println!("loss: {}", loss);
    }

    Ok(())
}
