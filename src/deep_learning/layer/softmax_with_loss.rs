use std::f64::consts::E;
use std::fs::File;
use std::io::{self, Read, Write, BufReader, BufRead, Lines};
use ndarray::prelude::{
    Array2,
};
use ndarray::Axis;

use crate::deep_learning::*;
use crate::deep_learning::layer::*;
use crate::deep_learning::common::*;


// Softmax with loss
pub struct SoftmaxWithLoss {
    x: Box<dyn NetworkLayer>,
    t: Array2<f64>,
    z: Option<Array2<f64>>, 
}
impl SoftmaxWithLoss {
    pub fn new<TX>(x: TX, t: Array2<f64>) -> SoftmaxWithLoss
    where TX: NetworkLayer + 'static
    {
        SoftmaxWithLoss {
            x: Box::new(x),
            t: t,
            z: None,
        }
    }
    pub fn get_x(&self) -> &Box<dyn NetworkLayer> {&self.x}
    pub fn get_t(&self) -> &Array2<f64> {&self.t}
    pub fn layer_label() -> &'static str {
        "softmax"
    }
    pub fn import<'a, T>(lines: &mut T) -> Self
        where T: Iterator<Item = &'a str>
    {
        println!("import {}", Self::layer_label());
        // t shape
        let shape_line = lines.next().unwrap();
        let mut shape_line_split = shape_line.split(',');
        let dim: (usize, usize) = (shape_line_split.next().unwrap().parse::<usize>().unwrap(), shape_line_split.next().unwrap().parse::<usize>().unwrap());
        // t
        let mut t = Array2::<f64>::zeros(dim);
        for row_i in 0..dim.0 {
            let line = lines.next().unwrap();
            let mut line_split = line.split(',');
            for col_i in 0..dim.1 {
                t[(row_i, col_i)] = line_split.next().unwrap().parse::<f64>().unwrap();
            }
        }

        let x = neural_network::import_network_layer(lines);

        SoftmaxWithLoss {
            x: x,
            t: t,
            z: None,
        }
    }
}
impl NetworkLayer for SoftmaxWithLoss {
    fn forward(&mut self, is_learning: bool) -> Array2<f64> {
        if self.z.is_none() {
            
            let x = self.x.forward(is_learning);

            let softmax_res = softmax(&x);

            let z = crosss_entropy_error(&softmax_res, &self.t);

            // // Weight decay
            // let decay = self.weight_squared_sum().sqrt() * 0.1f64 / 2f64;
            // let z = z + decay;

            self.z = Some(z);
        }
        // println!("soft for:\n{:?}", self.z.as_ref().unwrap());
        self.z.clone().unwrap()
    }
    fn forward_skip_loss(&mut self, is_learning: bool) -> Array2<f64> {
        self.x.forward(is_learning)
    }
    fn backward(&mut self, dout: Array2<f64>) {
        // println!("softmax backward");
        let x = self.x.forward(true);
        let softmax_res = softmax(&x);

        let dx = dout * (softmax_res - &self.t);

        // // Weight decay
        // let d_decay = self.weight_sum().sqrt() * 0.1f64;
        // let dx = dx + d_decay;

        self.x.backward(dx);
    }
    fn set_value(&mut self, value: &Array2<f64>) {
        self.x.set_value(value);
        self.clean();
    }
    fn set_lbl(&mut self, value: &Array2<f64>) {
        if self.t.shape() != value.shape() {
            panic!("Different shape. self.t: {:?} value:{:?}", self.t.shape(), value.shape());
        }
        self.t.assign(value);
        self.x.set_lbl(value);
        self.clean();
    }
    fn clean(&mut self) {
        self.z = None;
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

        writeln!(file, "{},{}", self.t.shape()[0], self.t.shape()[1])?;
        for row in self.t.axis_iter(Axis(0)) {
            for v in row {
                write!(file, "{},", v)?;
            }
            writeln!(file, "")?;
        }

        file.flush()?;
        self.x.export(file)?;
        Ok(())
    }
}

fn softmax(x: &Array2<f64>) -> Array2<f64> {
    // Create Array same shape from x
    let mut z = Array2::<f64>::zeros(x.dim());
    // let mut z = x.clone();

    for row_i in 0..x.shape()[0] {
        // Get row
        let mut r = x.index_axis(Axis(0), row_i).to_owned();

        // INFINITY and NaN measure( 0 <= r < 100)
        // MIN
        let min_index = max_index_in_arr1(&(&r*-1.0));
        if r[min_index] < 0.0 {
            r += r[min_index] * -1.0;
        }
        // MAX
        let max_index = max_index_in_arr1(&r);
        let max = r[max_index];
        if max > 100.0 {
            r *= 100.0 / max;
        }

        // Find max
        let max_index = max_index_in_arr1(&r);
        let max = r[max_index];

        // Σ(exp(ai + c)
        let mut sum_exp_a = 0.0;
        for ai in &r {
            sum_exp_a += E.powf(*ai + max);
            if E.powf(*ai + max).is_infinite() {
                println!("INFINITY soft exp_a. ai:{}, max:{}", *ai, max);
            }
            if E.powf(*ai + max).is_nan() {
                println!("Nan soft exp_a. ai:{}, max:{}", *ai, max);
            }
        }

        // exp(ak + c)/Σ(exp(ai + c))
        for col_i in 0..x.shape()[1] {
            z[(row_i, col_i)] = E.powf(r[col_i] + max) / sum_exp_a;
            if z[(row_i, col_i)].is_infinite() {
                println!("INFINITY sof exp(ak + c)/Σ(exp(ai + c)). r[col_i]: {} max:{}, sum:{}", r[col_i], max, sum_exp_a);
            }
            if z[(row_i, col_i)].is_nan() {
                println!("NaN sof exp(ak + c)/Σ(exp(ai + c)). r[col_i]: {} max:{}, sum:{}", r[col_i], max, sum_exp_a);
                println!("row: {:?}", r);
            }
        }
    }
    return z;
}

fn crosss_entropy_error(x: &Array2<f64>, t: &Array2<f64>) -> Array2<f64> {
    if x.len() != t.len() {
        panic!("Different shape. x:{:?} t:{:?}", x.shape(), t.shape());
    }

    // Create Array same len with row len x has
    let mut z = Array2::<f64>::zeros([x.shape()[0], 1]);

    for row_i in 0..x.shape()[0] {
        // Get row
        // INFINITY measure( 0 < x_row)
        let x_row = x.index_axis(Axis(0), row_i).to_owned() + 0.00000001;
        let t_row = t.index_axis(Axis(0), row_i).to_owned();

        // Find correct label index
        let correct_index = max_index_in_arr1(&t_row);

        z[(row_i, 0)] = x_row[correct_index].log(E) * -1.0;
        if x_row[correct_index].log(E).is_infinite() {
            println!("INFINITY cross. x_row[correct_index]:{}", x_row[correct_index]);
        }
        if x_row[correct_index].log(E).is_nan() {
            println!("NAN cross. x_row[correct_index]:{}", x_row[correct_index]);
        }
    }
    return z;
}

#[cfg(test)]
mod test_softmax_with_loss_mod {
    use super::*;

    use ndarray::prelude::{
        arr2,
    };

    use crate::deep_learning::optimizer::*;

    #[test]
    fn test_softmax() {
        let x = arr2(&
            [
                [2.0,   5.0, 3.0,  3.0],
                [0.0, -10.0, 7.0, 12.0],
            ]
        );

        let softmax_res = softmax(&x);

        // Row0 max index
        assert_eq!(max_index_in_arr1(&softmax_res.index_axis(Axis(0), 0).to_owned()), 1);
        // Row0 sum
        assert_eq!(round_digit(sum_arr1(&softmax_res.index_axis(Axis(0), 0).to_owned()), -4), 1.0);
        // Row1 max index
        assert_eq!(max_index_in_arr1(&softmax_res.index_axis(Axis(0), 1).to_owned()), 3);
        // Row0 sum
        assert_eq!(round_digit(sum_arr1(&softmax_res.index_axis(Axis(0), 1).to_owned()), -4), 1.0);

    }

    #[test]
    fn test_crosss_entropy_error() {
        let x = arr2(&
            [
                [0.3, 0.1, 0.5, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ]
        );

        let t = arr2(&
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        );

        let cee_res = crosss_entropy_error(&x, &t);

        assert_eq!(cee_res.shape(), [2, 1]);
        assert_eq!(round_digit(cee_res[(0, 0)], -4) , round_digit((0.5 as f64).log(E) * -1.0, -4));
        assert_eq!(round_digit(cee_res[(1, 0)], -4), round_digit((1.0 as f64).log(E) * -1.0, -4));
    }
    #[test]
    fn test_backward() {
        let arr2_x = arr2(&
            [
                [2.0,   5.0, 3.0,  3.0],
                [0.0, -10.0, 7.0, 12.0],
            ]
        );
        // Use NetworkBatchAffineValueLayer to check side effects
        let x = AffineDirectValue::new(
            arr2_x.clone(),
            Sgd::new(0.01)
        );
        let t = arr2(&
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        );
        let mut softmaxLoss= SoftmaxWithLoss::new(x, t.clone());

        let dout = arr2(&
            [
                [1.0, -1.0, 2.0, -2.0],
                [0.0,  1.0, 2.0,  0.0],
            ]
        );

        softmaxLoss.backward(dout.clone());

        assert_eq!(
            round_digit_arr2(&softmaxLoss.x.forward(true), -4),
            round_digit_arr2(&(arr2_x.clone()-((softmax(&arr2_x)-t)*dout*0.01)), -4)
        );
    }
}