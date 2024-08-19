use std::fs::File;
use std::io::{self, Read, Write, BufReader, BufRead, Lines};
use ndarray::{
    Array1,
    Array2,
    ArrayView1,
    Axis,
};

use crate::deep_learning::*;
use crate::deep_learning::layer::*;
// use crate::deep_learning::optimizer::*;
// use crate::deep_learning::common::*;


pub struct Pooling {
    x: Box<dyn NetworkLayer>,
    y: Option<Array2<f64>>,
    x_shape: (usize, usize, usize, usize),
    filter_h: usize,
    filter_w: usize,
    stride: usize,
    padding: usize,
    col_max_index: Option<Array1<usize>>,
}
impl Pooling {
    pub fn new<TX>(x: TX, x_shape: (usize, usize, usize, usize), filter_h: usize, filter_w: usize, stride: usize, padding: usize)
        -> Pooling
        where   TX : NetworkLayer + 'static,
    {
        Pooling {
            x: Box::new(x),
            y: None,
            x_shape: x_shape,
            filter_h: filter_h,
            filter_w: filter_w,
            stride: stride,
            padding: padding,
            col_max_index: None,
        }
    }
    pub fn layer_label() -> &'static str {
        "pooling"
    }
    pub fn import<'a, T>(lines: &mut T) -> Self
        where T: Iterator<Item = &'a str>
    {
        println!("import {}", Self::layer_label());
        // x_shape
        let shape_line = lines.next().unwrap();
        let mut shape_line_split = shape_line.split(',');
        let x_shape: (usize, usize, usize, usize) = (
            shape_line_split.next().unwrap().parse::<usize>().unwrap(),
            shape_line_split.next().unwrap().parse::<usize>().unwrap(),
            shape_line_split.next().unwrap().parse::<usize>().unwrap(),
            shape_line_split.next().unwrap().parse::<usize>().unwrap(),
        );

        // filter_h
        let value_line = lines.next().unwrap();
        let filter_h = value_line.parse::<usize>().unwrap();

        // filter_w
        let value_line = lines.next().unwrap();
        let filter_w = value_line.parse::<usize>().unwrap();

        // stride
        let value_line = lines.next().unwrap();
        let stride = value_line.parse::<usize>().unwrap();

        // padding
        let value_line = lines.next().unwrap();
        let padding = value_line.parse::<usize>().unwrap();

        let x = neural_network::import_network_layer(lines);

        Pooling {
            x: x,
            y: None,
            x_shape: x_shape,
            filter_h: filter_h,
            filter_w: filter_w,
            stride: stride,
            padding: padding,
            col_max_index: None,
        }
    }
}
impl NetworkLayer for Pooling {
    fn forward(&mut self, is_learning: bool) -> Array2<f64> {
        if self.y.is_none() {
            let x = self.x.forward(is_learning);
            let (batch_num, channel_num, x_h, x_w) = self.x_shape;
            let step_h = (x_h + 2 * self.padding - self.filter_h) / self.stride + 1;
            let step_w = (x_w + 2 * self.padding - self.filter_w) / self.stride + 1;

            let x_4d = x.to_shared().reshape(self.x_shape).to_owned();
            let col = im2col(&x_4d, self.filter_h, self.filter_w, self.stride, self.padding);
            let shaped_col = col.to_shared().reshape((batch_num*channel_num*step_h*step_w , self.filter_h*self.filter_w));

            let mut col_max = Array1::<f64>::zeros(shaped_col.shape()[0]);
            let mut col_max_index = Array1::<usize>::zeros(shaped_col.shape()[0]);
            for col_i in 0..shaped_col.shape()[0] {
                let indexed_col = shaped_col.index_axis(Axis(0), col_i);

                let mut max_index = 0;
                for row_i in 1..indexed_col.len() {
                    if indexed_col[max_index] < indexed_col[row_i] {
                        max_index = row_i;
                    }
                }
                col_max[col_i] = indexed_col[max_index];
                col_max_index[col_i] = max_index;
            }
            let mut col_max_3d = col_max.to_shared().reshape((batch_num, step_h*step_w, channel_num)).to_owned();
            col_max_3d.swap_axes(1, 2);

            let y = col_max_3d.to_shared().reshape((batch_num, channel_num*step_h*step_w)).to_owned();
            self.y = Some(y);
            self.col_max_index = Some(col_max_index);
        }
        self.y.clone().unwrap()
    }
    fn backward(&mut self, dout: Array2<f64>) {
        self.forward(true);
        let col_max_index = self.col_max_index.as_ref().unwrap();

        let (batch_num, channel_num, x_h, x_w) = self.x_shape;
        let step_h = (x_h + 2 * self.padding - self.filter_h) / self.stride + 1;
        let step_w = (x_w + 2 * self.padding - self.filter_w) / self.stride + 1;

        let mut dout_3d = dout.to_shared().reshape((batch_num, channel_num, step_h*step_w)).to_owned();
        dout_3d.swap_axes(1, 2);

        let col_d_1d = dout_3d.to_shared().reshape(batch_num*channel_num*step_h*step_w);

        let mut col_dx = Array2::<f64>::zeros((batch_num*channel_num*step_h*step_w, self.filter_h*self.filter_w));
        for col_i in 0..batch_num*channel_num*step_h*step_w {
            col_dx[(col_i, col_max_index[col_i])] = col_d_1d[col_i];
        }
        let col_dx = col_dx;

        let dx_4d = col2im(&col_dx, self.x_shape, (0, 0, self.filter_h, self.filter_w), self.stride, self.padding);

        let dx = dx_4d.to_shared().reshape((batch_num, channel_num*x_h*x_w)).to_owned();

        // println!("max_idx: {:?}", col_max_index);
        // println!("dx: {:?}", dx);
        self.x.backward(dx);
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
        self.y = None;
        self.x.clean();
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

        writeln!(file, "{},{},{},{}", self.x_shape.0, self.x_shape.1, self.x_shape.2, self.x_shape.3)?;
        writeln!(file, "{}", self.filter_h)?;
        writeln!(file, "{}", self.filter_w)?;
        writeln!(file, "{}", self.stride)?;
        writeln!(file, "{}", self.padding)?;

        file.flush()?;
        self.x.export(file)?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use mockall::predicate::*;
    use ndarray::{
        Array,
        arr2,
    };

    #[test]
    fn test_pooling_forward() {
        // B:2, C:2 H:6 W:6
        let value = DirectValue::new(
            Array::from_shape_vec(
                (2,72),
                vec![
                    01f64, 02f64, 03f64, 04f64, 05f64, 06f64,
                    11f64, 12f64, 13f64, 14f64, 15f64, 16f64,
                    21f64, 22f64, 23f64, 24f64, 25f64, 26f64,
                    31f64, 32f64, 33f64, 34f64, 35f64, 36f64,
                    41f64, 42f64, 43f64, 44f64, 45f64, 46f64,
                    51f64, 52f64, 53f64, 54f64, 55f64, 56f64,

                    101f64, 102f64, 103f64, 104f64, 105f64, 106f64,
                    111f64, 112f64, 113f64, 114f64, 115f64, 116f64,
                    121f64, 122f64, 123f64, 124f64, 125f64, 126f64,
                    131f64, 132f64, 133f64, 134f64, 135f64, 136f64,
                    141f64, 142f64, 143f64, 144f64, 145f64, 146f64,
                    151f64, 152f64, 153f64, 154f64, 155f64, 156f64,

                    201f64, 202f64, 203f64, 204f64, 205f64, 306f64,
                    211f64, 212f64, 213f64, 214f64, 215f64, 316f64,
                    221f64, 222f64, 223f64, 224f64, 225f64, 326f64,
                    231f64, 232f64, 233f64, 234f64, 235f64, 336f64,
                    241f64, 242f64, 243f64, 244f64, 245f64, 346f64,
                    251f64, 252f64, 253f64, 254f64, 255f64, 356f64,

                    401f64, 402f64, 403f64, 404f64, 405f64, 406f64,
                    411f64, 412f64, 413f64, 414f64, 415f64, 416f64,
                    421f64, 422f64, 423f64, 424f64, 425f64, 426f64,
                    431f64, 432f64, 433f64, 434f64, 435f64, 436f64,
                    441f64, 442f64, 443f64, 444f64, 445f64, 446f64,
                    451f64, 452f64, 453f64, 454f64, 455f64, 456f64,
                ]
            ).ok().unwrap()
        );

        let expect = Array::from_shape_vec(
            (2,18),
            vec![
                12f64, 14f64, 16f64,
                32f64, 34f64, 36f64,
                52f64, 54f64, 56f64,

                112f64, 114f64, 116f64,
                132f64, 134f64, 136f64,
                152f64, 154f64, 156f64,

                212f64, 214f64, 316f64,
                232f64, 234f64, 336f64,
                252f64, 254f64, 356f64,

                412f64, 414f64, 416f64,
                432f64, 434f64, 436f64,
                452f64, 454f64, 456f64,
            ]
        ).ok().unwrap();

        let filter_h = 2;
        let filter_w = 2;
        let stride = 2;
        let pad = 0;
        let mut pool = Pooling::new(value, (2, 2, 6, 6), filter_h, filter_w, stride, pad);

        let y = pool.forward(false);

        assert_eq!(y, expect);
    }

    #[test]
    fn test_pooling_backward() {
        // B:2, C:2 H:2 W:2
        let dout = Array::from_shape_vec(
            (2,8),
            vec![
                001f64, 002f64,
                011f64, 012f64,
                
                101f64, 102f64,
                111f64, 112f64,

                201f64, 202f64,
                211f64, 212f64,
                
                301f64, 302f64,
                311f64, 312f64,
            ]
        ).ok().unwrap();

        // X
        let mut x = MockNetworkLayer::new();
        x.expect_forward()
            .returning(|_| -> Array2<f64> {
                // B:2, C:2 H:6 W:6
                Array::from_shape_vec(
                    (2,72),
                    vec![
                        101f64, 002f64, 003f64, 004f64, 005f64, 106f64,
                        011f64, 012f64, 113f64, 014f64, 015f64, 016f64,
                        021f64, 022f64, 023f64, 024f64, 025f64, 026f64,
                        031f64, 132f64, 033f64, 134f64, 035f64, 136f64,
                        041f64, 042f64, 143f64, 044f64, 045f64, 046f64,
                        051f64, 052f64, 053f64, 054f64, 055f64, 056f64,
        
                        001f64, 002f64, 003f64, 004f64, 005f64, 006f64,
                        111f64, 012f64, 013f64, 014f64, 015f64, 016f64,
                        021f64, 022f64, 023f64, 124f64, 025f64, 026f64,
                        031f64, 132f64, 033f64, 034f64, 035f64, 036f64,
                        041f64, 042f64, 043f64, 044f64, 145f64, 046f64,
                        051f64, 052f64, 053f64, 054f64, 055f64, 056f64,
        
                        001f64, 002f64, 003f64, 004f64, 005f64, 006f64,
                        011f64, 012f64, 013f64, 014f64, 015f64, 016f64,
                        121f64, 022f64, 023f64, 024f64, 125f64, 026f64,
                        131f64, 032f64, 033f64, 034f64, 135f64, 036f64,
                        041f64, 042f64, 043f64, 044f64, 045f64, 046f64,
                        051f64, 052f64, 053f64, 054f64, 055f64, 056f64,
        
                        001f64, 102f64, 003f64, 004f64, 005f64, 006f64,
                        011f64, 012f64, 013f64, 114f64, 015f64, 016f64,
                        021f64, 022f64, 023f64, 024f64, 025f64, 026f64,
                        031f64, 032f64, 033f64, 034f64, 035f64, 136f64,
                        041f64, 142f64, 043f64, 044f64, 045f64, 046f64,
                        051f64, 052f64, 053f64, 054f64, 055f64, 056f64,
                    ]
                ).ok().unwrap()
            })
        ;

        // expect X
        let dx_expect = Array::from_shape_vec(
            (2,72),
            vec![
                000f64, 000f64, 000f64, 000f64, 000f64, 002f64,
                000f64, 000f64, 001f64, 000f64, 000f64, 000f64,
                000f64, 000f64, 000f64, 000f64, 000f64, 000f64,
                000f64, 000f64, 000f64, 000f64, 000f64, 012f64,
                000f64, 000f64, 011f64, 000f64, 000f64, 000f64,
                000f64, 000f64, 000f64, 000f64, 000f64, 000f64,
    
                000f64, 000f64, 000f64, 000f64, 000f64, 000f64,
                101f64, 000f64, 000f64, 000f64, 000f64, 000f64,
                000f64, 000f64, 000f64, 102f64, 000f64, 000f64,
                000f64, 111f64, 000f64, 000f64, 000f64, 000f64,
                000f64, 000f64, 000f64, 000f64, 112f64, 000f64,
                000f64, 000f64, 000f64, 000f64, 000f64, 000f64,
    
                000f64, 000f64, 000f64, 000f64, 000f64, 000f64,
                000f64, 000f64, 000f64, 000f64, 000f64, 000f64,
                201f64, 000f64, 000f64, 000f64, 202f64, 000f64,
                211f64, 000f64, 000f64, 000f64, 212f64, 000f64,
                000f64, 000f64, 000f64, 000f64, 000f64, 000f64,
                000f64, 000f64, 000f64, 000f64, 000f64, 000f64,
    
                000f64, 301f64, 000f64, 000f64, 000f64, 000f64,
                000f64, 000f64, 000f64, 302f64, 000f64, 000f64,
                000f64, 000f64, 000f64, 000f64, 000f64, 000f64,
                000f64, 000f64, 000f64, 000f64, 000f64, 312f64,
                000f64, 311f64, 000f64, 000f64, 000f64, 000f64,
                000f64, 000f64, 000f64, 000f64, 000f64, 000f64,
            ]
        ).ok().unwrap();
        // println!("dx_expect: {:?}", dx_expect);
        x.expect_backward()
            .times(1)
            .with(eq(dx_expect))
            .returning(|_| {})
        ;

        let mut pooling = Pooling::new(
            x,
            (2, 2, 6, 6),
            3,
            3,
            3,
            0,
        );

        pooling.backward(dout);
    }

    #[test]
    fn aa() {
        let a = arr2(&
            [
                [1f64,2f64,3f64],
                [6f64,2f64,3f64],
                [6f64,9f64,3f64],
                [6f64,9f64,12f64],
            ]
        );

        let aa = a.map_axis(
            Axis(1),
            |nums: ArrayView1<f64>| -> f64 {
                let mut max = nums[0];
                for n in nums {
                    if max < *n {
                        max = *n;
                    }
                }
                return max;
            }
        );
        println!("{:?}", aa);
    }
}