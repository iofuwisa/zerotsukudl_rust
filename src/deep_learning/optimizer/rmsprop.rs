use ndarray::prelude::{
    Array2,
};

use crate::deep_learning::optimizer::*;
use crate::deep_learning::common::*;

pub struct Rmsprop {
    learning_rate: f64,
    velocity: Option<Array2<f64>>,
    friction: f64,
}
impl Rmsprop {
    pub fn new(learning_rate: f64, friction: f64) -> Self {
        Self {
            learning_rate: learning_rate,
            velocity: None,
            friction: friction,
        }
    }
}
impl Optimizer for Rmsprop {
    fn update(&mut self, target: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        if self.velocity.is_none() {
            self.velocity = Some(Array2::<f64>::zeros(target.dim()));
        }
        let mut velocity = self.velocity.as_mut().unwrap();

        velocity.assign(&(velocity.clone() * self.friction + gradient * gradient * (1.0 - self.friction)));
        return
            target - 
                self.learning_rate /
                sqrt_arr2(&(velocity.clone() + (10.0 as f64).powi(-6)))
                * gradient;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use ndarray::prelude::{
        arr2,
    };

    #[test]
    fn update() {
        let mut rmsprop = Rmsprop::new(0.1, 0.9);
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

        let updated = rmsprop.update(&target, &gradient);

        let velocity = gradient.clone() * gradient.clone() * 0.1;
        let expect_updated =
            target.clone() -
                0.1 /
                sqrt_arr2(&(velocity.clone() + (10.0 as f64).powi(-6))) *
                gradient.clone();
        assert_eq!(
            round_digit_arr2(&updated, -6),
            round_digit_arr2(&expect_updated, -6)
        );

        let updated = rmsprop.update(&target, &gradient);

        let velocity = velocity.clone() * 0.9 + gradient.clone() * gradient.clone() * 0.1;
        let expect_updated =
        target.clone() -
            0.1 /
            sqrt_arr2(&(velocity.clone() + (10.0 as f64).powi(-6))) *
            gradient.clone();
        assert_eq!(
            round_digit_arr2(&updated, -6),
            round_digit_arr2(&expect_updated, -6)
        );
    }
}