use std::error::Error;

use neural_network::NN;
use plotlib::{repr::Plot, view::ContinuousView, page::Page, style::LineStyle};

use std::env;

const BATCH_SIZE: usize = 1;
const EPOCHS: usize = 1000;
const SVG_PATH: &'static str = "examples/xor/xor_nn_loss.svg"; 

fn main() -> Result<(), Box<dyn Error>>{
    env::set_var("RUST_BACKTRACE", "1");

    // XOR inputs
    let x_train = vec![
        vec![0., 0.],
        vec![0., 1.],
        vec![1., 0.],
        vec![1., 1.],
    ];

    // XOR outputs
    let y_train = vec![0., 1., 1., 0.];

    // 2 input NN with 1 hidden layer of 4 nodes and a single output node
    // learning rate is set very high to find the point of overfitting
    let mut nn = NN::new(vec![2, 4, 1], 4.5);
    
    // train NN with batch size of 1 for 1000 epochs
    let losses = nn.train(&x_train, &y_train, BATCH_SIZE, EPOCHS)?;

    // create line graph of the loss
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
    
    // save graph as SVG
    Page::single(&v).save(SVG_PATH).unwrap();
    
    // test the model against the inputs
    println!("prediction for input (0, 0) = {}", nn.forward((&x_train[0]).into())?.get(0, 0));
    println!("prediction for input (0, 1) = {}", nn.forward((&x_train[1]).into())?.get(0, 0));
    println!("prediction for input (1, 0) = {}", nn.forward((&x_train[2]).into())?.get(0, 0));
    println!("prediction for input (1, 1) = {}", nn.forward((&x_train[3]).into())?.get(0, 0));
    
    Ok(())
}
