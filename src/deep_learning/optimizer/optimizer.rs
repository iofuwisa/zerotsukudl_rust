use ndarray::prelude::{
    Array2,
};

#[cfg_attr(test, mockall::automock)]
pub trait Optimizer {
    fn update(&mut self, target: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64>;
}

// Reference
// https://data-science.gr.jp/theory/tml_optimizer_of_gradient_descent.html
