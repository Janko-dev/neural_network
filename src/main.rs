use std::error::Error;

use neural_network::NN;

const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 20;

fn main() -> Result<(), Box<dyn Error>>{

    let x_train = vec![
        vec![0., 0.],
        vec![0., 1.],
        vec![1., 0.],
        vec![1., 1.],
    ];

    let y_train = vec![0., 1., 1., 0.];

    let mut nn = NN::new(vec![2, 4, 1], 1.5);
    
    for _ in 0..EPOCHS {
        let loss = nn.train(&x_train, &y_train, BATCH_SIZE)?;
        loss.print();
    }

    nn.forward((&x_train[0]).into())?.print();
    nn.forward((&x_train[1]).into())?.print();
    nn.forward((&x_train[2]).into())?.print();
    nn.forward((&x_train[3]).into())?.print();

    Ok(())
}
