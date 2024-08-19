use std::fs::File;
use std::io::{self, Read, Write, BufReader, BufRead, Lines};
use ndarray::prelude::{
    Array2,
    Axis,
};

use crate::deep_learning::layer::*;
use crate::deep_learning::optimizer::*;
use crate::deep_learning::common::*;
use crate::deep_learning::*;

// Affine
pub struct Affine {
    x: Box<dyn NetworkLayer>,
    w: Box<dyn NetworkLayer>,
    b: Box<dyn NetworkLayer>,
    z: Option<Array2<f64>>, 
}
impl Affine {
    pub fn new<TX, TW, TB>(x: TX, w: TW, b: TB)
        -> Affine
        where   TX : NetworkLayer + 'static,
                TW : NetworkLayer + 'static,
                TB : NetworkLayer + 'static,
    {
        Affine {
            x: Box::new(x),
            w: Box::new(w),
            b: Box::new(b),
            z: None,
        }
    }
    pub fn new_random<TX, TWO, TBO>(x: TX, input_len: usize, neuron_len: usize, optimizer_w: TWO, optimizer_b: TBO)
        -> Affine
        where   TX : NetworkLayer + 'static,
                TWO: Optimizer + 'static,
                TBO: Optimizer + 'static
    {
        // Generate initialize weight and biasn by normal distibution
        let affine_weight = AffineDirectValue::new(
            Array2::from_shape_vec(
                (input_len as usize, neuron_len as usize),
                norm_random_vec(input_len * neuron_len)
            ).ok().unwrap(),
            optimizer_w    
        );
        let affine_bias = AffineDirectValue::new(
            Array2::from_shape_vec(
                (1, neuron_len as usize),
                norm_random_vec(neuron_len)
                    .into_iter()
                    .map(|x: f64| {x / 100.0})
                    .collect()
            ).ok().unwrap(),
            optimizer_b
        );

       return Affine::new(x, affine_weight, affine_bias);
    }
    pub fn new_random_with_name<TX, TWO, TBO>(x: TX, input_len: usize, neuron_len: usize, optimizer_w: TWO, optimizer_b: TBO, name: String)
    -> Affine
    where   TX : NetworkLayer + 'static,
            TWO: Optimizer + 'static,
            TBO: Optimizer + 'static
    {
        // Generate initialize weight and biasn by normal distibution
        let affine_weight = AffineDirectValue::new_with_name(
            Array2::from_shape_vec(
                (input_len as usize, neuron_len as usize),
                norm_random_vec(input_len * neuron_len)
            ).ok().unwrap(),
            optimizer_w,
            name.clone() + "_weight",
        );
        let affine_bias = AffineDirectValue::new_with_name(
            Array2::from_shape_vec(
                (1, neuron_len as usize),
                norm_random_vec(neuron_len)
                    .into_iter()
                    .map(|x: f64| {x / 100.0})
                    .collect()
            ).ok().unwrap(),
            optimizer_b,
            name.clone() + "_bias",
        );

        return Affine::new(x, affine_weight, affine_bias);
    }
    pub fn get_x(&self) -> &Box<dyn NetworkLayer> {&self.x}
    pub fn get_w(&self) -> &Box<dyn NetworkLayer> {&self.w}
    pub fn get_b(&self) -> &Box<dyn NetworkLayer> {&self.b}
    pub fn layer_label() -> &'static str {
        "affine"
    }
    pub fn import<'a, T>(lines: &mut T) -> Self
        where T: Iterator<Item = &'a str>
    {
        println!("import {}", Self::layer_label());
        let x = neural_network::import_network_layer(lines);
        let w = neural_network::import_network_layer(lines);
        let b = neural_network::import_network_layer(lines);

        Affine {
            x: x,
            w: w,
            b: b,
            z: None,
        }
    }
}
impl NetworkLayer for Affine {
    fn forward(&mut self, is_learning: bool) -> Array2<f64> {
        if self.z.is_none() {
            let x = self.x.forward(is_learning);
            let w = self.w.forward(is_learning);
            let b = self.b.forward(is_learning);
            self.z = Some(x.dot(&w) + b);
        }
        self.z.clone().unwrap()
    }
    fn backward(&mut self, dout: Array2<f64>) {
        let w = self.w.forward(true);
        let w_t = w.t();
        let dx = dout.dot(&w_t);
        self.x.backward(dx,);

        let x = self.x.forward(true);
        let x_t = x.t();
        let dw = x_t.dot(&dout);
        self.w.backward(dw);

        let db = dout.sum_axis(Axis(0)).to_shared().reshape((1, w.shape()[1])).to_owned();
        // println!("db: {:?}", db);
        self.b.backward(db);
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
        self.z = None;
    }
    fn plot(&self){
        self.x.plot();
        self.w.plot();
        self.b.plot();
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


#[cfg(test)]
mod test {
    use super::*;

    use ndarray::{
        arr2,
    };
    use mockall::predicate::*;

    #[test]
    fn test_affine_new_random() {
        let x = DirectValue::new(arr2(&
            [
                [1.0,  2.0],
                [1.0, -2.0]
            ]
        ));

        let mut affine = Affine::new_random(
            x,
            2,
            10,
            Sgd::new(0.01),
            Sgd::new(0.01)
        );

        assert_eq!(affine.x.forward(true).shape(), [2, 2]);
        assert_eq!(affine.w.forward(true).shape(), [2, 10]);
        assert_eq!(affine.b.forward(true).shape(), [1, 10]);
        // println!("x:\n{}", affine.x.value);
        // println!("w:\n{}", affine.w.value);
        // println!("b:\n{}", affine.b.value);
    }

    #[test]
    fn test_affine_forward() {
        let x = DirectValue::new(arr2(&
            [
                [1.0,  2.0],
                [1.0, -2.0]
            ]
        ));
        let w = DirectValue::new(arr2(&
            [
                [ 0.5,  0.2, 1.5],
                [-1.0, -0.5, 2.0]
            ]
        ));
        let b = DirectValue::new(arr2(&
            [
                [1.0, 2.0, 1.0]
            ]
        ));

        let mut affine = Affine::new(x, w, b);

        let y = affine.forward(false);
        assert_eq!(y, arr2(&
            [
                [-0.5, 1.2,  6.5],
                [ 3.5, 3.2, -1.5]
            ]
        ));
    }

    #[test]
    fn test_affine_backward() {
        println!("affine backward");
        // X
        let mut mock_x = MockNetworkLayer::new();
        let mut x_value = arr2(&
            [
                [01f64, 02f64, 03f64],
                [11f64, 12f64, 13f64],
            ]
        );
        mock_x.expect_forward()
            .returning(|_| -> Array2<f64> {
                arr2(&
                    [
                        [01f64, 02f64, 03f64],
                        [11f64, 12f64, 13f64],
                    ]
                )
            })
        ;

        // W
        let mut mock_w = MockNetworkLayer::new();
        let mut w_value = arr2(&
            [
                [01f64, 02f64, 03f64, 04f64],
                [11f64, 12f64, 13f64, 14f64],
                [21f64, 22f64, 23f64, 24f64],
                [31f64, 32f64, 33f64, 34f64],
            ]
        );
        mock_w.expect_forward()
            .returning(|_| -> Array2<f64> {
                arr2(&
                    [
                        [01f64, 02f64, 03f64, 04f64],
                        [11f64, 12f64, 13f64, 14f64],
                        [21f64, 22f64, 23f64, 24f64],
                        [31f64, 32f64, 33f64, 34f64],
                    ]
                )
            })
        ;

        // B
        let mut mock_b = MockNetworkLayer::new();
        let mut _b_value = arr2(&
            [
                [1f64, 2f64, 3f64, 4f64]
            ]
        );
        mock_b.expect_forward()
            .returning(|_| -> Array2<f64> {
                arr2(&
                    [
                        [1f64, 2f64, 3f64, 4f64]
                    ]
                )
            })
        ;

        let dout = arr2(&
            [
                [10f64, 20f64, 30f64, 40f64],
                [20f64, 40f64, 60f64, 80f64],
            ]
        );

        // expect X
        let expect_dx = dout.dot(&w_value.t());
        mock_x.expect_backward()
            .times(1)
            .with(eq(expect_dx))
            .returning(|_| {})
        ;
        
        // expect W
        let expect_dw = x_value.t().dot(&dout);
        mock_w.expect_backward()
            .times(1)
            .with(eq(expect_dw))
            .returning(|_| {})
        ;
        
        // expect B
        let expect_db = arr2(&
            [
                [30f64, 60f64, 90f64, 120f64],
            ]
        );
        // println!("expect_db: {:?}", expect_db);
        mock_b.expect_backward()
            .times(1)
            .with(eq(expect_db))
            .returning(|_| {})
        ;

        let mut affine = Affine::new(mock_x, mock_w, mock_b);
        affine.backward(dout);
    }
}
