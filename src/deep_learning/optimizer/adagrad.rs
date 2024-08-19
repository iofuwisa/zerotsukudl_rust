use ndarray::prelude::{
    Array2,
};

use crate::deep_learning::optimizer::*;
use crate::deep_learning::common::*;

pub struct AdaGrad {
    learning_rate: f64,
    grad_squared_sum: Option<Array2<f64>>
}
impl AdaGrad {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate: learning_rate,
            grad_squared_sum: None,
        }
    }
}
impl Optimizer for AdaGrad {
    fn update(&mut self, target: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        if self.grad_squared_sum.is_none() {
            self.grad_squared_sum = Some(Array2::<f64>::zeros(target.dim()));
        }
        let mut grad_squared_sum = self.grad_squared_sum.as_mut().unwrap();

        grad_squared_sum.assign(&(grad_squared_sum.clone() + gradient * gradient));

        return
            target -
                self.learning_rate /
                sqrt_arr2(&(grad_squared_sum.clone() + (10.0 as f64).powi(-6))) *
                gradient;
    }
}

#[cfg(test)]
mod test_adagrad_mod {
    use super::*;

    use ndarray::prelude::{
        arr2,
    };

    #[test]
    fn update() {
        let mut adagrad = AdaGrad::new(0.1);
        let target = arr2(&
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        );
        let gradient = arr2(&
            [
                [1.0, 3.0, 5.0],
                [2.0, 4.0, 6.0],
            ]
        );

        let updated = adagrad.update(&target, &gradient);

        let grad_suared_sum = gradient.clone() * gradient.clone();
        let expect_updated = 
            target.clone() -
                0.1 /
                sqrt_arr2(&(grad_suared_sum.clone() + (10.0 as f64).powi(-6))) *
                gradient.clone();
        assert_eq!(updated, expect_updated);

        let updated = adagrad.update(&target, &gradient);

        let grad_suared_sum = grad_suared_sum.clone() + gradient.clone() * gradient.clone();
        let expect_updated = 
            target.clone() -
                0.1 /
                sqrt_arr2(&(grad_suared_sum.clone() + (10.0 as f64).powi(-6))) *
                gradient.clone();
        assert_eq!(updated, expect_updated);
    }
}