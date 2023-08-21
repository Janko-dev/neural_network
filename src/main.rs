use std::error::Error;

use neural_network::{NN, Matrix};

const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 20;

fn broadcast_shape(lhs: (usize, usize), rhs: (usize, usize)) -> (usize, usize) {
    
    // let lhs_bcast = (lhs.0 < rhs.0, lhs.1 < rhs.1);
    // let rhs_bcast = (rhs.0 < lhs.0, rhs.1 < lhs.1);
    // (lhs_bcast, rhs_bcast)
    let rhs = match rhs {
        (1, x) => (lhs.0, x),
        (x, 1) => (x, lhs.1),
        x => x
    };

    let lhs = match lhs {
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

fn main() -> Result<(), Box<dyn Error>>{

    // let lhs_shape = (3, 2);
    // let rhs_shape = (1, 2);
    
    // let shape = broadcast_shape((3, 2), (1, 2));
    
    // dbg!(shape);
    // let l_broadcast = shape != lhs_shape;
    // let r_broadcast = shape != rhs_shape;

    // match (l_broadcast, r_broadcast) => {
    //     (true, true) =>    (lhs.broadcast_as(shape)?, rhs.broadcast_as(shape)?),
    //     (false, true) =>   (lhs, rhs.broadcast_as(shape)?),
    //     (true, false) =>   (lhs.broadcast_as(shape)?, rhs),
    //     (false, false) =>  (lhs, rhs), 
    // }

    let a = Matrix::from_vec(vec![1., 2., 3., 4., 5., 6.], (3, 2), true);
    let b = Matrix::from_vec(vec![1., 2., 3.], (3, 1), true);

    let c = b.broadcast_as(a.shape());
    
    a.print();
    b.print();
    c.print();
    c.print_comp_tree();

    // let grads = c.backward()?;

    // grads.get(b.id()).unwrap().print();

    // let x_train = vec![
    //     vec![0., 0.],
    //     vec![0., 1.],
    //     vec![1., 0.],
    //     vec![1., 1.],
    // ];

    // let y_train = vec![0., 1., 1., 0.];

    // let mut nn = NN::new(vec![2, 4, 1], 1.5);
    
    // for _ in 0..EPOCHS {
    //     let loss = nn.train(&x_train, &y_train, BATCH_SIZE)?;
    //     loss.print();
    // }

    // nn.forward((&x_train[0]).into())?.print();
    // nn.forward((&x_train[1]).into())?.print();
    // nn.forward((&x_train[2]).into())?.print();
    // nn.forward((&x_train[3]).into())?.print();

    Ok(())
}
