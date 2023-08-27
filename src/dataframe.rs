use std::{error::Error, mem, io::Read, collections::HashMap};

use csv::Reader;

use crate::error::UtilityError;

pub struct DataFrame {
    headers: Vec<String>,

}



// pub struct LabelEncoding {
//     counter: usize,
//     has_seen: HashMap<String, bool>
// }

// pub struct OneHotEncoding {
//     has_seen: HashMap<String, bool>
// }
// x    1 0 0
// x    1 0 0
// y    0 1 0
// z    0 0 1

// type Dataset = (Vec<Vec<f32>>, Vec<f32>);

pub enum EncodingScheme {
    Label,
    OneHot
}

// impl EncodingScheme {
//     pub fn encode(&self, reader: Reader<&[u8]>, y_index: usize) -> Result<Dataset, Box<dyn Error>> {
//         match self {
//             EncodingScheme::Label => EncodingScheme::label_encoding(reader, y_index),
//             EncodingScheme::OneHot => EncodingScheme::one_hot_encoding(reader, y_index),
//         }
//     }

//     fn label_encoding(reader: Reader<&[u8]>, y_index: usize) -> Result<Dataset, Box<dyn Error>>{

//         let mut counter: usize = 0;
//         let mut has_seen = HashMap::<String, usize>::new();
        
//         let mut xs = vec![];
//         let mut ys = vec![];

//         for result in reader.records() {

//             let record = result?;
            
//             let y_element = record.get(y_index).unwrap();
//             let y = match y_element.parse::<f32>() {
//                 Ok(n) => n,
//                 Err(_) => {
//                     if let Some(n) = has_seen.get(y_element) {
//                         *n as f32
//                     } else {
//                         counter += 1;
//                         has_seen.insert(y_element.to_string(), counter);
//                         (counter - 1) as f32
//                     }
//                 }
//             };

//             ys.push(y);

//             let x = (0..csv_column_len)
//                 .filter(|i| *i != y_index)
//                 .map(|i| encoding.encode(record[i].to_string()) )
//                 .collect::<Vec<f32>>();
//             xs.push(x);
//         }

//         Ok((xs, ys))
//     }

//     fn one_hot_encoding(reader: Reader<&[u8]>, y_index: usize) -> Result<Dataset, Box<dyn Error>>{

//         let counter: usize = 0;
//         let has_seen = HashMap::<String, bool>::new();
//     }
// }

// #[derive(Debug)]
// pub enum TableDT {
//     String(String),
//     Number(f32)
// }

// pub type Table = Vec<Vec<TableDT>>;



// pub fn encode(mut table: Table, columns: Vec<usize>, encoding: EncodingScheme) -> Result<Table, Box<dyn Error>> {

//     match encoding {
//         EncodingScheme::Label => {
            
//             for col in columns.into_iter() {

//                 let mut counter: usize = 0;
//                 let mut has_seen = HashMap::<String, usize>::new();
                
//                 table = table
//                     .iter()
//                     .map(|row| 
//                         row
//                         .iter()
//                         .enumerate()
//                         .filter(|(i, _)| *i == col)
//                         .map(|(_, f)| {
//                             match f {
//                                 TableDT::Number(n) => TableDT::Number(*n),
//                                 TableDT::String(s) => match s.parse::<f32>() {
//                                     Ok(n) => TableDT::Number(n),
//                                     Err(_) => if let Some(n) = has_seen.get(s) {
//                                         TableDT::Number(*n as f32)
//                                     } else {
//                                         counter += 1;
//                                         has_seen.insert(s.clone(), counter);
//                                         TableDT::Number((counter - 1) as f32)
//                                     }
//                                 }
//                             }
//                         })
//                         .collect()
//                     )
//                     .collect();
//             }
//             Ok(table)
//         },
//         EncodingScheme::OneHot => {
            
//             Ok(table)
//         }
//     }

//     // Ok(())
// }

// pub fn partition_labels(table: Table, y_index: usize) -> Result<(Table, Vec<TableDT>), Box<dyn Error>> {

//     let csv_column_len = rdr.headers()?.len() - 1;
//     if y_index > csv_column_len {
//         return Err(Box::new(UtilityError::LabelIndexOutOfBound { y_index, csv_column_len }));
//     }
//     Ok((table, vec![]))
// }