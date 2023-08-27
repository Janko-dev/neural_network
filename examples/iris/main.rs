use std::{error::Error, collections::HashMap};

use neural_network::{NN, UtilityError};
use plotlib::{repr::Plot, view::ContinuousView, page::Page, style::LineStyle};

use std::env;

const BATCH_SIZE: usize = 64;
const EPOCHS: usize = 300;
const NN_CONFIG: [usize; 5] = [4, 2, 4, 5, 3];
const LEARNING_RATE: f32 = 0.2;

const SVG_PATH: &'static str = "examples/iris/iris_nn_loss.svg"; 

fn read_csv(file_name: &str, drop: Vec<usize>) -> Result<Vec<Vec<f32>>, Box<dyn Error>>{
    let content = std::fs::read_to_string(file_name)?;
    let mut rdr = csv::Reader::from_reader(content.as_bytes());

    let mut table = vec![];

    let mut counter = 0;
    let mut label_encoder: HashMap<String, usize> = HashMap::new();

    for result in rdr.records() {

        let record = result?;

        table.push((0..record.len())
            .map(|x| drop
                .iter()
                .map(move |&y| (x, y))
            )
            .flatten()
            .filter(|(i, j)| *i != *j)
            .map(|(x, _)| {
                let x = record[x].trim();
                match x.parse::<f32>() {
                    Ok(n) => n,
                    Err(_) => {
                        if let Some(n) = label_encoder.get(x) {
                            *n as f32
                        } else {
                            label_encoder.insert(x.to_string(), counter);
                            counter += 1;
                            (counter - 1) as f32
                        }
                    }
                }
            })
            .collect::<Vec<f32>>()
        );
    }
    Ok(table)
}

fn partition_labels(table: Vec<Vec<f32>>, y_index: usize) -> Result<(Vec<Vec<f32>>, Vec<f32>), Box<dyn Error>> {

    let csv_column_len = table[0].len() - 1;
    if y_index > csv_column_len {
        return Err(Box::new(UtilityError::LabelIndexOutOfBound { y_index, csv_column_len }));
    }

    let mut xs = vec![];
    let mut ys = vec![];

    for row in table {
        let filtered_row = (0..row.len())
            .filter(|&i| i != y_index)
            .map(|i| row[i])
            .collect::<Vec<f32>>();
        
        xs.push(filtered_row);
        ys.push(row[y_index]);
    }

    Ok((xs, ys))
}

fn main() -> Result<(), Box<dyn Error>>{
    env::set_var("RUST_BACKTRACE", "1");

    // drop column 0 (id column)
    let input_dataset = read_csv("examples/iris/iris.csv", vec![0])?;
    
    // print 10 records
    println!("Records 45-55:");
    for i in 45..55 {
        println!("{}: {:?}", i, input_dataset[i]);
    }

    // partition columns where the labels are index 4
    let (x_train, y_train) = partition_labels(input_dataset, 4)?;
    
    println!("Records 45-55 divided into inputs and labels:");
    for i in 45..55 {
        println!("{}: {:?}\t[{:?}]", i, x_train[i], y_train[i]);
    }

    // 4 input NN with 2 hidden layer of 4 nodes and 3 output nodes for the labels
    let mut nn = NN::new(NN_CONFIG.to_vec(), LEARNING_RATE);
    
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
    // println!("prediction for input (0, 0) = {}", nn.forward((&x_train[0]).into())?.get(0, 0));
    // println!("prediction for input (0, 1) = {}", nn.forward((&x_train[1]).into())?.get(0, 0));
    // println!("prediction for input (1, 0) = {}", nn.forward((&x_train[2]).into())?.get(0, 0));
    // println!("prediction for input (1, 1) = {}", nn.forward((&x_train[3]).into())?.get(0, 0));
    
    Ok(())
}
