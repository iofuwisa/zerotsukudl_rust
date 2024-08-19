
use std::fs::File;
use std::io::{self, Read, Write, BufReader, BufRead, Lines};
use ndarray::prelude::{
    Array2,
};

use crate::deep_learning::*;
use crate::deep_learning::layer::*;

// Relu
// y = x (x > 0)
// y = 0 (x <= 0)

pub struct Relu {
    x: Box<dyn NetworkLayer>,
    y: Option<Array2<f64>>, 
}
impl Relu {
    pub fn new<TX>(x: TX) -> Relu
        where TX: NetworkLayer + 'static {
            Relu {
            x: Box::new(x),
            y: None,
        }
    }
    pub fn get_x(&self) -> &Box<dyn NetworkLayer> {&self.x}
    pub fn layer_label() -> &'static str {
        "relu"
    }
    pub fn import<'a, T>(lines: &mut T) -> Self
        where T: Iterator<Item = &'a str>
    {
        println!("import {}", Self::layer_label());
        let x = neural_network::import_network_layer(lines);

        Relu {
            x: x,
            y: None,
        }
    }
}
impl NetworkLayer for Relu {
    fn forward(&mut self, is_learning: bool) -> Array2<f64> {
        if self.y.is_none() {
            let x = self.x.forward(is_learning);
            self.y = Some(x.mapv(|n: f64| -> f64{if n > 0.0 {n} else {0.0}}));
        }
        return self.y.clone().unwrap();
    }
    fn backward(&mut self, dout: Array2<f64>) {
        // println!("relu backward");
        let x = self.x.forward(true);
        if dout.shape() != x.shape() {
            panic!("Different shape. dout: {:?} x: {:?}", dout.shape(), x.shape());
        }

        let mut iter_dout = dout.iter();
        let dx = x.mapv(|n: f64| -> f64 {
            let d = iter_dout.next().unwrap();
            if n > 0.0 {    // z = x (x > 0)
                *d
            } else {        // z = 0 (x <= 0)
                0.0
            }
        });

        self.x.backward(dx);
    }
    fn set_value(&mut self, value: &Array2<f64>) {
        self.x.set_value(value);
        self.clean();
    }
    fn set_lbl(&mut self, value: &Array2<f64>) {
        self.x.set_lbl(value);
        self.clean();
    }
    fn clean(&mut self) {
        self.y = None;
    }
    fn plot(&self){
        self.x.plot();
    }
    fn weight_squared_sum(&self) -> f64 {
        return self.x.weight_squared_sum();
    }
    fn weight_sum(&self) -> f64 {
        return self.x.weight_sum();
    }
    #[cfg (not (target_family = "wasm"))]
    fn export(&self, file: &mut File) -> Result<(), Box<std::error::Error>> {
        writeln!(file, "{}", Self::layer_label())?;
        file.flush()?;
        self.x.export(file)?;
        Ok(())
    }
}

#[cfg(test)]
mod test_mod {
    use super::*;

    use ndarray::prelude::{
        arr2,
    };

    #[test]
    fn test_forward() {
        let value = DirectValue::new(arr2(&
            [
                [2.0, -3.0, 0.0],
                [1.0, 0.1, -0.12345],
            ]
        ));
        let mut relu = Relu::new(value);

        let relu_res = relu.forward(false);
        
        assert_eq!(relu_res, arr2(&
            [
                [2.0, 0.0, 0.0],
                [1.0, 0.1, 0.0],
            ]
        ));
    }

    #[test]
    fn test_backward() {
        let value = DirectValue::new(arr2(&
            [
                [2.0, -3.0, 0.0],
                [1.0, 0.1, -0.12345],
            ]
        ));
        let mut relu = Relu::new(value);
        let dout = arr2(&
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        );

        relu.backward(dout);
    }
}