use std::fs::File;
use std::io::{self, Read, Write, BufReader, BufRead, Lines};
use ndarray::prelude::{
    Array2,
    Axis,
};

use crate::deep_learning::*;
use crate::deep_learning::layer::*;
use crate::deep_learning::optimizer::*;
use crate::deep_learning::common::*;

pub struct NetworkBatchNormValueLayer {
    value: Array2<f64>,
    optimizer: Box<dyn Optimizer>,
}
impl NetworkBatchNormValueLayer {
    pub fn new<TO>(value: Array2<f64>, optimizer: TO) -> Self
        where TO: Optimizer + 'static
    {
        NetworkBatchNormValueLayer {
            value: value,
            optimizer: Box::new(optimizer),
        }
    }
    pub fn new_from_len<TO>(row_len: usize, col_len: usize, optimizer: TO) -> AffineDirectValue
        where TO: Optimizer + 'static
    {
        return AffineDirectValue::new(Array2::<f64>::zeros((row_len, col_len)), optimizer);
    }
}
impl NetworkLayer for NetworkBatchNormValueLayer {
    fn forward(&mut self, _is_learning: bool) -> Array2<f64> {
        self.value.clone()
    }
    fn backward(&mut self, dout: Array2<f64>) {
        let updated_value = self.optimizer.update(&self.value, &dout);
        self.value.assign(&updated_value);
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
    fn plot(&self) {
        // Nothing to do
    }
    fn weight_squared_sum(&self) -> f64 {
        return 0f64;
    }
    fn weight_sum(&self) -> f64 {
        return 0f64;
    }
}

// Batch normalization
pub struct BatchNorm {
    x: Box<dyn NetworkLayer>,
    y: Option<Array2<f64>>, 
    w: Box<dyn NetworkLayer>,
    b: Box<dyn NetworkLayer>,
    normalized: Option<Array2<f64>>,
    distribute: Option<Array2<f64>>,
    average: Option<Array2<f64>>,
}
impl BatchNorm {
    pub fn new<TX, TW, TB>(x: TX, w: TW, b: TB) -> BatchNorm
        where   TX: NetworkLayer + 'static,
                TW: NetworkLayer + 'static,
                TB: NetworkLayer + 'static,
    {
        BatchNorm {
            x: Box::new(x),
            y: None,
            w: Box::new(w),
            b: Box::new(b),
            normalized: None,
            distribute: None,
            average: None,
        }
    }
    pub fn get_x(&self) -> &Box<dyn NetworkLayer> {&self.x}
    pub fn layer_label() -> &'static str {
        "batchnorm"
    }
    pub fn import<'a, T>(lines: &mut T) -> Self
        where T: Iterator<Item = &'a str>
    {
        println!("import {}", Self::layer_label());
        let x = neural_network::import_network_layer(lines);
        let w = neural_network::import_network_layer(lines);
        let b = neural_network::import_network_layer(lines);

        BatchNorm {
            x: x,
            y: None, 
            w: w,
            b: b,
            normalized: None,
            distribute: None,
            average: None,
        }
    }
}
impl NetworkLayer for BatchNorm {
    fn forward(&mut self, is_learning: bool) -> Array2<f64> {
        if self.y.is_none() {
            let x = self.x.forward(is_learning);
            let w = self.w.forward(is_learning);
            let b = self.b.forward(is_learning);

            let (average, distribute) = calc_distribute_and_broadcast(&x);

            // Calc normalize
            let normalized = (&x - &average) / sqrt_arr2(&(&distribute + 10f64.powi(-6)));

            // Apply weight and bias
            let y = normalized.clone() * w + b;

            self.y = Some(y);
            self.normalized = Some(normalized);
            self.distribute = Some(distribute);
            self.average = Some(average);

        }
        return self.y.clone().unwrap();
    }

    // refference: https://qiita.com/t-tkd3a/items/14950dbf55f7a3095600
    fn backward(&mut self, dout: Array2<f64>) {
        let x = self.x.forward(true);
        let w = self.w.forward(true);
        let normalized = self.normalized.as_ref().unwrap();
        let distribute = self.distribute.as_ref().unwrap();
        let average = self.average.as_ref().unwrap();

        let dw = backword_broadcast(&(normalized * &dout));

        let db = dout.sum_axis(Axis(0)).to_shared().reshape((1, w.shape()[1])).to_owned();


        // // d15 +
        // let d15 = dout.clone();
        
        // // d14 Broadcast
        // let d14 = backword_broadcast(&d15);

        // // 13 db
        // self.b.backward(d14.clone());
        
        // // d12b *
        // let d12b = normalized * &dout;

        // // d11 Broadcast
        // let d11 = backword_broadcast(&d12b);

        // // 10
        // self.w.backward(d11.clone());



        // d12a *
        let d12a = w * &dout;

        // d9a *
        let d9a =  &d12a /  sqrt_arr2(&(distribute + 10f64.powi(-6)));

        // d9b *
        let d9b = &d12a * (x.clone() - average);

        // d8 Broadcast
        let d8 = backword_broadcast(&d9b);

        // d7 1/x
        let d7 = d8 * (-1.0 / (distribute + 10f64.powi(-6)));

        // d6 sqrt(x)
        let d6 = d7 / (2.0 * sqrt_arr2(&(distribute + 10f64.powi(-6).sqrt())));

        // d5 avg
        let d5 = backword_average(&d6, x.shape()[0]);

        // d4 ^2
        let d4 = d5 * (x.clone() - average);
        
        //d3a -
        let d3a = d9a + d4;

        //d3b -
        let d3b = -&d3a;

        // d2 Broadcast
        let d2 = backword_broadcast(&d3b);

        // d1
        let d1 = backword_average(&d2, x.shape()[0]);
        
        let dx = &d3a + d1;

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
        return
            self.x.weight_squared_sum() +
            self.w.weight_squared_sum() +
            self.b.weight_squared_sum();
    }
    fn weight_sum(&self) -> f64 {
        return
            self.x.weight_sum() +
            self.w.weight_sum() +
            self.b.weight_sum();
    }
    #[cfg (not (target_family = "wasm"))]
    fn export(&self, file: &mut File) -> Result<(), Box<std::error::Error>> {
        writeln!(file, "{}", Self::layer_label())?;
        file.flush()?;
        self.x.export(file)?;
        self.w.export(file)?;
        self.b.export(file)?;
        Ok(())
    }
}

fn calc_average_and_broadcast(x: &Array2<f64>) -> Array2<f64> {
    // Calc average for each column
    let mut average = Array2::<f64>::zeros(x.dim());
    for col in 0..x.shape()[1] {
        let col_x = x.index_axis(Axis(1), col);
        let mut col_average = average.index_axis_mut(Axis(1), col);

        let mut sum = 0f64;
        for n in &col_x {
            sum += *n;
        }
        let batch_average = sum / col_x.len() as f64;

        col_average.fill(batch_average);
    }
    return average;
}

fn calc_distribute_and_broadcast(x: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let average = calc_average_and_broadcast(x);

    // Calc distibute for each column
    let avg_diff_squared = (x.clone() - &average) * (x.clone() - &average);
    let mut distribute = Array2::<f64>::zeros(x.dim());
    for col in 0..x.shape()[1] {
        let col_avg_diff_squared = avg_diff_squared.index_axis(Axis(1), col);
        let mut col_distribute = distribute.index_axis_mut(Axis(1), col);

        let mut sum = 0f64;
        for n in &col_avg_diff_squared {
            sum += *n;
        }
        let batch_distribute = sum / col_avg_diff_squared.len() as f64;

        col_distribute.fill(batch_distribute);
    }
    return (average, distribute);
}

fn backword_average(x: &Array2<f64>, row_len: usize) -> Array2<f64> {
    let mut y = Array2::<f64>::zeros((row_len, x.shape()[1]));
    for col_i in 0..x.shape()[1] {
        let mut col_y = y.index_axis_mut(Axis(1), col_i);
        col_y.fill(x[(0, col_i)] / row_len as f64);
    }
    return y;
}

fn backword_broadcast(dout: &Array2<f64>) -> Array2<f64> {
    let d = dout.sum_axis(Axis(0)).to_shared().reshape((1, dout.shape()[1])).to_owned();
    return d;
}

#[cfg(test)]
mod test {
    use super::*;

    use ndarray::prelude::{
        arr2,
    };
    use rand::{Rng};

    #[test]
    fn test_batch_norm_forward() {
        let mut batch_norm = BatchNorm::new(
            AffineDirectValue::new_from_len(100, 100, Sgd::new(0.01)),
            NetworkBatchNormValueLayer::new(
                Array2::<f64>::ones((100, 100)),
                Sgd::new(0.01)
            ),
            NetworkBatchNormValueLayer::new(
                Array2::<f64>::zeros((100, 100)),
                Sgd::new(0.01)
            ),
        );

        let mut rng = rand::thread_rng();
        let batch_data = Array2::from_shape_fn((100, 100), 
            |_| -> f64 {
                return rng.gen::<f64>() * 2f64 + 1f64;
            }
        );
        batch_norm.set_value(&batch_data);

        let y = batch_norm.forward(false);

        let (average, distribute) = calc_distribute_and_broadcast(&y);
        assert_eq!(
            round_digit_arr2(&average, -5),
            Array2::<f64>::zeros(average.dim())
        );
        assert_eq!(
            round_digit_arr2(&distribute, -5),
            Array2::<f64>::ones(distribute.dim())
        );

    }

    #[test]
    fn test_batch_norm_calc_average_and_broadcast() {
        let x = arr2(&
            [
                [1f64, 2f64, 3f64],
                [4f64, 5f64, 6f64],
            ]
        );

        let acerage = calc_average_and_broadcast(&x);

        assert_eq!(acerage, arr2(&
            [
                [2.5f64, 3.5f64, 4.5f64],
                [2.5f64, 3.5f64, 4.5f64],
            ])
        );
    }

    #[test]
    fn test_batch_norm_calc_distribute_and_broadcast() {
        let x = arr2(&
            [
                [1f64, 2f64, 3f64],
                [5f64, 6f64, 7f64],
            ]
        );

        let (acerage, distibute) = calc_distribute_and_broadcast(&x);

        assert_eq!(acerage, arr2(&
            [
                [3f64, 4f64, 5f64],
                [3f64, 4f64, 5f64],
            ])
        );
        assert_eq!(distibute, arr2(&
            [
                [4f64, 4f64, 4f64],
                [4f64, 4f64, 4f64],
            ])
        )
    }

    #[test]
    fn test_batch_norm_backword_average() {
        let x = arr2(&
            [
                [3f64, 6f64, 9f64]
            ]
        );
        let row_len: usize = 3;

        let y = backword_average(&x, row_len);

        assert_eq!(y, arr2(&
            [
                [1f64, 2f64, 3f64],
                [1f64, 2f64, 3f64],
                [1f64, 2f64, 3f64],
            ]
        ));

    }

    #[test]
    fn test_batch_norm_backword_broadcast() {
        let x = arr2(&
            [
                [1f64, 2f64, 3f64],
                [4f64, 5f64, 6f64],
                [7f64, 8f64, 9f64],
            ]
        );

        let y = backword_broadcast(&x);

        assert_eq!(y, arr2(&
            [
                [12f64, 15f64, 18f64],
            ]
        ));
    }
}