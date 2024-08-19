use std::fs::File;
use std::io::{self, Read, Write, BufReader, BufRead, Lines};
use ndarray::prelude::{
    Array2,
};
use std::f64::consts::E;

use crate::deep_learning::*;
use crate::deep_learning::layer::*;

// Sigmoid
// y = 1 / (1 + exp(-x))
pub struct Sigmoid {
    x: Box<dyn NetworkLayer>,
    y: Option<Array2<f64>>,
}
impl Sigmoid {
    pub fn new<TX>(x: TX) -> Sigmoid
    where TX: NetworkLayer + 'static {
        Sigmoid {
            x: Box::new(x),
            y: None,
        }
    }
    pub fn get_x(&self) -> &Box<dyn NetworkLayer> {&self.x}
    pub fn layer_label() -> &'static str {
        "sigmoid"
    }
    pub fn import<'a, T>(lines: &mut T) -> Self
        where T: Iterator<Item = &'a str>
    {
        println!("import {}", Self::layer_label());
        let x = neural_network::import_network_layer(lines);

        Sigmoid {
            x: x,
            y: None,
        }
    }
}
impl NetworkLayer for Sigmoid {
    // f(x) =  1 / (1 + exp(-x))
    fn forward(&mut self, is_learning: bool) -> Array2<f64> {
        if self.y.is_none() {
            // -x
            let x = self.x.forward(is_learning) * -1.0;
            // exp(-x)
            let x = x.mapv(|n: f64| -> f64 {E.powf(n)});
            // 1 + exp(-x)
            let x = x + 1.0;
            // 1 / (1 + exp(-x))
            let y = 1.0 / x;

            self.y = Some(y);
        }
        return self.y.clone().unwrap();
    }
    // f(x)' = (1 - f(x)) f(x)
    fn backward(&mut self, dout: Array2<f64>) {
        let fx = self.forward(true);
        if dout.shape() != fx.shape() {
            panic!("Different shape. dout: {:?} fx:{:?}", dout.shape(), fx.shape());
        }

        let mut iter_dout = dout.iter();
        let dx = fx.mapv(|n: f64| -> f64 {
            let d = iter_dout.next().unwrap();
            // (1 - f(x)) f(x)
            return d * (1.0 - n) * n
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
mod test_sigmoid_mod {
    use super::*;

    use ndarray::prelude::{
        arr2,
    };

    use crate::deep_learning::common::*;
    use crate::deep_learning::optimizer::*;

    #[test]
    fn test_forward() {
        let value = DirectValue::new(arr2(&
            [
                [2.0, -3.0, 0.0],
                [1.0, 0.1, -0.1],
            ]
        ));
        let mut sigmoid = Sigmoid::new(value);

        let sigmoid_res = sigmoid.forward(true);
        
        assert_eq!(round_digit_arr2(&sigmoid_res, -4), round_digit_arr2(&arr2(&
            [
                [0.88077077, 0.04742587317, 0.5],
                [0.73105857863, 0.52497918747, 0.4750201252],
            ]
        ), -4));
    }

    #[test]
    fn test_backward() {
        let arr2_value = arr2(&
            [
                [2.0, -3.0, 0.0],
                [1.0, 0.1, -0.1],
            ]
        );
        // Use NetworkBatchAffineValueLayer to check side effects
        let value = AffineDirectValue::new(
            arr2_value.clone(),
            Sgd::new(0.01)
        );
        let mut sigmoid = Sigmoid::new(value);
        let dout = arr2(&
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        );

        sigmoid.backward(dout.clone());

        assert_eq!(
            round_digit_arr2(&sigmoid.x.forward(true), -4),
            // (1 - f(x)) f(x)
            round_digit_arr2(&(arr2_value.clone()-(((1.0-sigmoid.forward(true))*sigmoid.forward(true))*dout*0.01)), -4)
        );
    } 
}