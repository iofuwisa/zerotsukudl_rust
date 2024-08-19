use ndarray::prelude::{
    Array2,
};

use crate::deep_learning::optimizer::*;
use crate::deep_learning::common::*;

pub struct Adam {
    learning_rate: f64,
    m: Option<Array2<f64>>,
    v: Option<Array2<f64>>,
    friction_m: f64,
    friction_v: f64,
    update_count: u32,
}
impl Adam {
    pub fn new(learning_rate: f64, friction_m: f64, friction_v: f64) -> Self {
        Self {
            learning_rate: learning_rate,
            m: None,
            v: None,
            friction_m: friction_m,
            friction_v: friction_v,
            update_count: 0,
        }
    }
}
impl Optimizer for Adam {
    fn update(&mut self, target: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        self.update_count += 1;

        if self.m.is_none() {
            self.m = Some(Array2::<f64>::zeros(target.dim()));
        }
        let mut m = self.m.as_mut().unwrap();

        if self.v.is_none() {
            self.v = Some(Array2::<f64>::zeros(target.dim()));
        }
        let mut v = self.v.as_mut().unwrap();

        m.assign(&(self.friction_m * m.clone() + (1.0 - self.friction_m) * gradient));

        v.assign(&(self.friction_v * v.clone() + (1.0 - self.friction_v) * gradient * gradient));

        let m_d = m.clone() / (1.0 - self.friction_m.powi(self.update_count as i32));

        let v_d = v.clone() / (1.0 - self.friction_v.powi(self.update_count as i32));

        return 
            target - 
                self.learning_rate *
                m_d / 
                sqrt_arr2(&(v_d + (10.0 as f64).powi(-6)))
        ;
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
        let mut adam = Adam::new(0.1, 0.9, 0.99);
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

        let updated = adam.update(&target, &gradient);

        let m = 0.1 * gradient.clone();
        let v = 0.01 * gradient.clone() * gradient.clone();
        let m_d = m.clone() / (1.0 - (0.9 as f64).powf(1.0));
        let v_d = v.clone() / (1.0 - (0.99 as f64).powf(1.0));
        
        let expect_updated = 
            target.clone() - 
                0.1 / 
                (v_d + (10.0 as f64).powi(-6)).mapv(|v: f64| -> f64 {v.sqrt()})
                    * m_d;
        assert_eq!(round_digit_arr2(&updated, -6), round_digit_arr2(&expect_updated, -6));


        let updated = adam.update(&target, &gradient);

        let m = m * 0.9 + 0.1 * gradient.clone();
        let v = v * 0.99 + 0.01 * gradient.clone() * gradient.clone();
        let m_d = m.clone() / (1.0 - (0.9 as f64).powf(2.0));
        let v_d = v.clone() / (1.0 - (0.99 as f64).powf(2.0));

        let expect_updated = 
            target.clone() - 
                0.1 / 
                (v_d + (10.0 as f64).powi(-6)).mapv(|v: f64| -> f64 {v.sqrt()})
                    * m_d;
        assert_eq!(round_digit_arr2(&updated, -6), round_digit_arr2(&expect_updated, -6));
    }
}
