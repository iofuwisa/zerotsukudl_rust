use ndarray::prelude::{
    Array2,
};

use crate::deep_learning::optimizer::*;

pub struct Momentum {
    learning_rate: f64,
    velocity: Option<Array2<f64>>,
    friction: f64,
}
impl Momentum {
    pub fn new(learning_rate: f64, friction: f64) -> Self {
        Self {
            learning_rate: learning_rate,
            velocity: None,
            friction: friction,
        }
    }
}
impl Optimizer for Momentum {
    fn update(&mut self, target: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        if self.velocity.is_none() {
            self.velocity = Some(Array2::<f64>::zeros(target.dim()));
        }
        let mut velocity = self.velocity.as_mut().unwrap();

        velocity.assign(&(velocity.clone() * self.friction - gradient * self.learning_rate));

        return target + velocity.clone();
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
        let mut momentum = Momentum::new(0.1, 0.9);
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

        let updated = momentum.update(&target, &gradient);

        let velocity = gradient * momentum.learning_rate * -1.0;
        let expect_updated = target + velocity;
        assert_eq!(updated, expect_updated);
    }
}