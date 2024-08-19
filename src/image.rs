use ndarray::prelude::{
    Array2,
    Array3,
    Axis,
    s,
};

pub struct TransImage {
    pixels: Array3<f64>,
    matrix: Array2<f64>,
    screen_pos: ((usize, usize), (usize, usize)),
}

impl TransImage {
    pub fn new_zero(shape: (usize, usize, usize)) -> Self {
        Self {
            pixels: Array3::<f64>::zeros(shape),
            matrix: Array2::<f64>::eye(2),
            screen_pos: ((0, 0), (shape.1, shape.2)),
        }
    }

    pub fn from_pixels(pixels: Array3<f64>) -> Self {
        Self {
            screen_pos: ((0, 0), (pixels.dim().1, pixels.dim().2)),
            pixels: pixels,
            matrix: Array2::<f64>::eye(2),
        }
    }

    pub fn resize_to(&mut self, x_size: usize, y_size: usize) {
        let (_, or_y_size, or_x_size) = self.pixels.dim();

        let x_rate = (x_size as f64) / (or_x_size as f64);
        let y_rate = (y_size as f64) / (or_y_size as f64);

        self.matrix[(0,0)] *= x_rate;
        self.matrix[(1,1)] *= y_rate;
    }

    pub fn set_screen_pos(&mut self, screen_pos: ((usize, usize), (usize, usize))) {
        self.screen_pos = screen_pos;
    }

    pub fn set_screen_pos_ignore_zero(&mut self) {
        let img_shape = self.pixels.dim();
        let mut head_col = 0;
        let mut head_row = 0;
        let mut tail_col = img_shape.2 - 1;
        let mut tail_row = img_shape.1 - 1;

        for col_i in 0..img_shape.2 {
            let col = self.pixels.index_axis(Axis(2), col_i);
            if col.sum() > 0f64 {
                head_col = col_i;
                break;
            }
        }
        for row_i in 0..img_shape.1 {
            let row = self.pixels.index_axis(Axis(1), row_i);
            if row.sum() > 0f64 {
                head_row = row_i;
                break;
            }
        }
        for col_i in (0..img_shape.2).rev() {
            let col = self.pixels.index_axis(Axis(2), col_i);
            if col.sum() > 0f64 {
                tail_col = col_i;
                break;
            }
        }
        for row_i in (0..img_shape.1).rev() {
            let row = self.pixels.index_axis(Axis(1), row_i);
            if row.sum() > 0f64 {
                tail_row = row_i;
                break;
            }
        }

        self.set_screen_pos(((head_row, head_col), (tail_row + 1, tail_col + 1)));
    
    }

    pub fn sampling(&self) -> Array3<f64> {
        // Calc new image size
        let (channel_size, _, _) = self.pixels.dim();
        let y_offeset = self.screen_pos.0.0;
        let y_size = self.screen_pos.1.0 - self.screen_pos.0.0;
        let x_offeset = self.screen_pos.0.1;
        let x_size = self.screen_pos.1.1 - self.screen_pos.0.1;
        
        let mut new_pixels = Array3::<f64>::zeros((channel_size, y_size, x_size));

        let inv_matrix = inverse_matrix(self.matrix.clone());
        
        for channel_i in 0..channel_size {
            for y_i in 0..y_size {
                for x_i in 0..x_size {
                    let to_pos = Array2::<f64>::from_shape_vec((2,1), vec!(x_i as f64, y_i as f64)).unwrap();
                    let from_pos = inv_matrix.dot(&to_pos);
                    // nearest neighbor
                    let from_pos = (
                        channel_i,
                        y_offeset + (from_pos[(1,0)] + 0.5f64) as usize,
                        x_offeset + (from_pos[(0,0)] + 0.5f64) as usize
                    );
                    if contain_index(self.pixels.dim(), from_pos) {
                        new_pixels[(channel_i, y_i, x_i)] = self.pixels[from_pos];
                    } else {
                        new_pixels[(channel_i, y_i, x_i)] = 0f64;
                    }
                }   
            }
        }

        return new_pixels;
    }
}

fn inverse_matrix(mut matrix: Array2<f64>) -> Array2<f64> {
    let mut diag = matrix.diag_mut();
    diag.swap(0, 1);
    let ad = diag.product();
    let bc = matrix.product() / ad;
    let tmp = matrix * Array2::from_shape_vec((2,2), vec![1f64, -1f64, -1f64, 1f64]).unwrap();
    return tmp / (ad - bc);
}

fn contain_index(shape :(usize, usize, usize), index: (usize, usize, usize)) -> bool {
    return shape.0 > index.0 && shape.1 > index.1 && shape.2 > index.2
}

#[cfg(test)]
mod test_trans_image {
    use super::*;

    use crate::mnist_images::*;

    #[test]
    fn test_resize() {
        // Load MNIST
        let mnist = MnistImages::new(5, 1, 1);

        for i in 0..5 {
            let img = mnist.get_trn_img().index_axis(Axis(0), i).clone().to_owned().into_raw_vec();

            let width = 28;
            let height = 28;
            for j in 0..width+2 {
                print!("-");
            }
            println!("");
            for i in 0..height {
                print!("|");
                for j in 0..width {
                    print!("{}", if img[(i*width+j)] > 0.0 {"*"} else {" "});
                }
                println!("|");
            }
            for j in 0..width+2 {
                print!("-");
            }
            println!("");

            let mut trans = TransImage::from_pixels(Array3::<f64>::from_shape_vec((1, 28, 28), img).unwrap());
            trans.set_screen_pos_ignore_zero();
            let img2 = trans.sampling();

            let mut trans = TransImage::from_pixels(img2);
            trans.resize_to(28, 28);
            trans.set_screen_pos(((0, 0), (28, 28)));
            let img2 = trans.sampling();

            let height = img2.dim().1;
            let width = img2.dim().2;
            for j in 0..width+2 {
                print!("-");
            }
            println!("");
            for i in 0..height {
                print!("|");
                for j in 0..width {
                    print!("{}", if img2[(0, i, j)] > 0.0 {"*"} else {" "});
                }
                println!("|");
            }
            for j in 0..width+2 {
                print!("-");
            }
            println!("");
        }
    }

    #[test]
    fn test_new() {
        let image_shape = (1usize, 2usize, 3usize);
        let trans = TransImage::new_zero(image_shape);
        let expect_matrix = Array2::<f64>::from_shape_vec((2,2), vec![
            1f64, 0f64,
            0f64, 1f64,
        ]).unwrap();

        assert_eq!(trans.matrix, expect_matrix);
    }

    #[test]
    fn test_inverse_matrix() {
        let matrix = Array2::<f64>::from_shape_vec((2,2), vec![
            1f64, 2f64,
            3f64, 4f64,
        ]).unwrap();

        let expect = Array2::<f64>::from_shape_vec((2,2), vec![
            -2f64, 1f64,
            1.5f64, -0.5f64,
        ]).unwrap();

        let inv = inverse_matrix(matrix);

        assert_eq!(inv, expect);
    }
}