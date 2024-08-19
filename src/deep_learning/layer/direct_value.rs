
#[cfg (not (target_family = "wasm"))]
use std::fs::File;
use std::io::{self, Read, Write, BufReader, BufRead, Lines};
use ndarray::prelude::{
    Axis,
    Array2,
};

use crate::deep_learning::*;
use crate::deep_learning::layer::*;

pub struct DirectValue {
    value: Array2<f64>,
}
impl DirectValue {
    pub fn new(value: Array2<f64>) -> DirectValue {
        DirectValue {
            value: value,
        }
    }
    pub fn new_from_len(row_len: usize, col_len: usize) -> DirectValue {
        return DirectValue::new(Array2::<f64>::zeros((row_len, col_len)))
    }
    pub fn layer_label() -> &'static str {
        "direct"
    }
    pub fn import<'a, T>(lines: &mut T) -> Self
        where T: Iterator<Item = &'a str>
    {
        println!("import {}", Self::layer_label());
        // value shape
        let shape_line = lines.next().unwrap();
        let mut shape_line_split = shape_line.split(',');
        let dim: (usize, usize) = (shape_line_split.next().unwrap().parse::<usize>().unwrap(), shape_line_split.next().unwrap().parse::<usize>().unwrap());
        // value
        let mut value = Array2::<f64>::zeros(dim);
        for row_i in 0..dim.0 {
            let line = lines.next().unwrap();
            let mut line_split = line.split(',');
            for col_i in 0..dim.1 {
                value[(row_i, col_i)] = line_split.next().unwrap().parse::<f64>().unwrap();
            }
        }

        DirectValue {
            value: value,
        }
    }
}
impl NetworkLayer for DirectValue {
    fn forward(&mut self, _is_learning: bool) -> Array2<f64> {
        self.value.clone()
    }
    fn backward(&mut self, _dout: Array2<f64>) {
        // Nothinf to do
    }
    fn set_value(&mut self, value: &Array2<f64>) {
        if self.value.shape() != value.shape() {
            panic!("Different shape. self.value: {:?} value:{:?}", self.value.shape(), value.shape());
        }
        self.value.assign(value);
    }
    fn set_lbl(&mut self, _value: &Array2<f64>) {
        // Nothing to do
    }
    fn clean(&mut self) {
        // Nothing to do
    }
    fn plot(&self){
        // Nothing to do
    }
    fn weight_squared_sum(&self) -> f64 {
        return 0f64;
    }
    fn weight_sum(&self) -> f64 {
        return 0f64;
    }
    #[cfg (not (target_family = "wasm"))]
    fn export(&self, file: &mut File) -> Result<(), Box<std::error::Error>> {
        writeln!(file, "{}", Self::layer_label())?;

        writeln!(file, "{},{}", self.value.shape()[0], self.value.shape()[1])?;
        for row in self.value.axis_iter(Axis(0)) {
            for v in row {
                write!(file, "{},", v)?;
            }
            writeln!(file, "")?;
        }

        file.flush()?;
        Ok(())
    }
}