use ndarray::prelude::{
    Array2,
};

use crate::deep_learning::optimizer::*;

pub struct Sgd {
    learning_rate: f64,
}
impl Sgd {
    pub fn new(learning_rate: f64) -> Self {
        Sgd {
            learning_rate: learning_rate,
        }
    }
}
impl Optimizer for Sgd {
    fn update(&mut self, target: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        return target - gradient * self.learning_rate;
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
        let mut sgd = Sgd::new(0.1);
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

        let updated = sgd.update(&target, &gradient);

        assert_eq!(updated, arr2(&
            [
                [0.9, 1.7, 2.5],
                [3.8, 4.6, 5.4]
            ]
        ));
    }
}