use std::error::Error;

use neural_network::{NN, Matrix};
use plotlib::{repr::Plot, view::ContinuousView, page::Page, style::LineStyle};

use std::env;
fn main() -> Result<(), Box<dyn Error>>{
    env::set_var("RUST_BACKTRACE", "1");
    // let x = Matrix::broadcast_shape((1, 4), (1, 1));
    // println!("{:?}", x);

    // dbg!(Matrix::broadcast_shape((1, 4), (1, 1)));
    // dbg!(Matrix::broadcast_shape((4, 1), (1, 1)));
    // dbg!(Matrix::broadcast_shape((1, 1), (1, 4)));
    // dbg!(Matrix::broadcast_shape((1, 1), (4, 1)));

    // let a = Matrix::from_vec(vec![1., 2., 3., 4.], (1, 4), true);
    // let b = Matrix::from_vec(vec![2.], (1, 1), true);

    // let c = b.broadcast_as((4, 1))?;
    // c.print();

    // let a = Matrix::from_vec(vec![1., 2., 3., 4., 5., 6.], (3, 2), true);
    // let b = Matrix::from_vec(vec![1., 2.], (1, 2), true);

    // let c = b.add(&a)?;
    
    // a.print();
    // b.print();
    // c.print();

    // c.print_comp_tree();

    // let grads = c.backward()?;

    // println!("-------------------------------");
    // grads.get(b.id()).unwrap().print();
    // grads.get(a.id()).unwrap().print();

    const BATCH_SIZE: usize = 4;
    const EPOCHS: usize = 300;

    let x_train = vec![
        vec![0., 0.],
        vec![0., 1.],
        vec![1., 0.],
        vec![1., 1.],
    ];

    let y_train = vec![0., 1., 1., 0.];

    let mut nn = NN::new(vec![2, 4, 1], 0.00005);
    
    // let xs: Matrix = x_train.into();
    // let ys = nn.forward(xs.t())?;

    // ys.print();

    // for _ in 0..EPOCHS {
    //     let loss = nn.train(&x_train, &y_train, BATCH_SIZE)?;
    //     println!("loss: {}", loss);
    // }
    let losses = nn.train(&x_train, &y_train, BATCH_SIZE, EPOCHS)?;

    let p = Plot::new(
        losses
            .iter()
            .enumerate()
            .map(|(i, x)| (i as f64, *x as f64))
            .collect::<Vec<(f64, f64)>>())
        .line_style(
            LineStyle::new() // uses the default marker
                .colour("#35C788")
        );
    
    let v = ContinuousView::new()
        .add(p)
        .x_label("Epochs")
        .y_label("Loss");

    Page::single(&v).save("test.svg").unwrap();
        
    nn.forward((&x_train[0]).into())?.print();
    nn.forward((&x_train[1]).into())?.print();
    nn.forward((&x_train[2]).into())?.print();
    nn.forward((&x_train[3]).into())?.print();

    Ok(())
}
