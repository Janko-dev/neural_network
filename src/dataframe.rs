use std::{error::Error, ops::{IndexMut, Index}};

#[derive(Debug, PartialEq)]
pub enum DFType {
    F32(f32),
    F64(f64),
    STR(String)
}

#[derive(Debug)]
pub struct DataFrame {
    headers: Vec<String>,
    data: Vec<DFType>,
    shape: (usize, usize)
}

pub enum EncodingScheme {
    Label,
    OneHot
}

impl DataFrame {
    pub fn from_csv<S: Into<String>>(file_path: S) -> Result<Self, Box<dyn Error>> {
        
        let content = std::fs::read_to_string(file_path.into())?;
        let mut rdr = csv::Reader::from_reader(content.as_bytes());

        let mut data = vec![];
        let headers = rdr
            .headers()?
            .to_owned()
            .iter()
            .map(|x| x.trim().to_string())
            .collect::<Vec<String>>();

        let cols = headers.len();
        let mut rows = 0;

        for result in rdr.records() {

            let record = result?;
            
            data.extend(record
                .iter()
                .map(|x| {
                    let x = x.trim();
                    match x.parse::<f32>() {
                        Ok(n) => DFType::F32(n),
                        Err(_) => DFType::STR(x.to_string())
                    }
                })
                .collect::<Vec<DFType>>()
            );

            rows += 1;
        }

        Ok(Self { headers, data, shape: (rows, cols) })
    }

    pub fn encode(&mut self, encoding: EncodingScheme) {
        
    }

}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn df_csv_test() {
        let file_path = "examples/iris/test_input.csv";
        let df = DataFrame::from_csv(file_path);

        assert!(!df.is_err());
        let df = df.unwrap();
        assert_eq!(df.headers, vec!["id", "test", "label"]);
        assert_eq!(df.shape, (3, 3));
        assert_eq!(df.data[0..3], vec![DFType::F32(1.), DFType::STR("x".to_string()), DFType::F32(1.)]);
        
        
    }
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