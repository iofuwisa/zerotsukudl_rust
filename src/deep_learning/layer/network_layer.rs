
#[cfg (not (target_family = "wasm"))]
use std::fs::File;
use ndarray::prelude::{
    Array2,
};

#[cfg_attr(test, mockall::automock)]
pub trait NetworkLayer {
    fn forward(&mut self, is_learning: bool) -> Array2<f64>;
    fn forward_skip_loss(&mut self, is_learning: bool) -> Array2<f64> {self.forward(is_learning)}
    fn backward(&mut self, _dout: Array2<f64>);
    fn set_value(&mut self, value: &Array2<f64>);
    fn set_lbl(&mut self, value: &Array2<f64>);
    fn clean(&mut self);
    fn is_loss_layer(&self) -> bool {false}
    fn plot(&self);
    fn weight_squared_sum(&self) -> f64;
    fn weight_sum(&self) -> f64;
    #[cfg (not (target_family = "wasm"))]
    fn export(&self, file: &mut File) -> Result<(), Box<std::error::Error>> {
        File::open("aaaaaaa")?;
        Ok(())
    }
}