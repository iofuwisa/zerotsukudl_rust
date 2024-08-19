use plotters::prelude::*;
use chrono::Local;
use std::path::Path;

pub fn plot_rate(rates: Vec<f64>, caption: &str) -> Result<(), Box<dyn std::error::Error>> {
    // image size
    let image_width = 1080;
    let image_height = 720;

    // BitMapBackend for generate file
    let date_str = Local::now().format("%m_%d_%H_%M_%S%.f").to_string();
    let file_path_str = "./graph/".to_string() + caption + &date_str + ".png";
    let root = BitMapBackend::new(
        Path::new(&file_path_str),
        (image_width, image_height))
    .into_drawing_area();

    // Background is white
    root.fill(&WHITE)?;
    
    let caption = "";
    let font = ("sans-serif", 20);

    let x_range = 0u32..(rates.len() as u32);
    let y_range = 0f64..1f64;

    // Graph setting
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, font.into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            x_range.clone(),
            y_range.clone(),
        )?;
    
    // Draw grid
    chart.configure_mesh().draw()?;

    // Add x data to rate
    let rates_with_x = rates.clone().into_iter()
        .zip(x_range.clone().collect::<Vec<u32>>().into_iter())
        .map(|(y, x)| (x, y));

    // Draw rate
    chart
        .draw_series(LineSeries::new(
            rates_with_x,
            &RED
        ))?
        .label("rate")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Line setting
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

pub fn plot_loss(rates: Vec<f64>, caption: &str) -> Result<(), Box<dyn std::error::Error>> {
    // image size
    let image_width = 1080;
    let image_height = 720;

    // BitMapBackend for generate file
    let date_str = Local::now().format("%m_%d_%H_%M_%S%.f").to_string();
    let file_path_str = "./graph/".to_string() + caption + &date_str + ".png";
    let root = BitMapBackend::new(
        Path::new(&file_path_str),
        (image_width, image_height))
    .into_drawing_area();

    // Background is white
    root.fill(&WHITE)?;
    
    let caption = "";
    let font = ("sans-serif", 20);

    let x_range = 0u32..(rates.len() as u32);
    let y_range = 0f64..3f64;

    // Graph setting
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, font.into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            x_range.clone(),
            y_range.clone(),
        )?;
    
    // Draw grid
    chart.configure_mesh().draw()?;

    // Add x data to rate
    let rates_with_x = rates.clone().into_iter()
        .zip(x_range.clone().collect::<Vec<u32>>().into_iter())
        .map(|(y, x)| (x, y));

    // Draw rate
    chart
        .draw_series(LineSeries::new(
            rates_with_x,
            &RED
        ))?
        .label("rate")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Line setting
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

pub fn plot_histogram(data: Vec<f64>, caption: &str) -> Result<(), Box<dyn std::error::Error>> {
    // image size
    let image_width = 1080;
    let image_height = 720;

    // BitMapBackend for generate file
    let date_str = Local::now().format("%m_%d_%H_%M_%S%.f").to_string();
    let file_path_str = "./graph/".to_string() + caption + &date_str + ".png";
    let root = BitMapBackend::new(
        Path::new(&file_path_str),
        (image_width, image_height))
    .into_drawing_area();

    // Background is white
    root.fill(&WHITE)?;
    
    let caption = "";
    let font = ("sans-serif", 20);

    // Count max
    let mut max_count: u32 = 0;
    for r in (-40..40).step_by(1) {
        let range_low = r as f64 / 10f64;
        let range_high = (r + 1) as f64 / 10f64;
        let mut count = 0;
        for d in &data {
            if *d >= range_low && *d < range_high {
                count += 1;
            }
        }
        max_count = if max_count<count {count} else {max_count};
    }

    // Graph setting
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, font.into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            (-4f64..4f64).step(0.1).use_round().into_segmented(),
            0u32..max_count
        )?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(&WHITE.mix(0.3))
        .axis_desc_style(font)
        .draw()?;

    chart.draw_series(
        Histogram::vertical(&chart)
            .style(RED.mix(0.5).filled())
            .data(data.iter().map(|x: &f64| (*x, 1))),
    )?;

    Ok(())
}

pub fn plot_bias_histogram(data: Vec<f64>, caption: &str) -> Result<(), Box<dyn std::error::Error>> {
    // image size
    let image_width = 1080;
    let image_height = 720;

    // BitMapBackend for generate file
    let date_str = Local::now().format("%m_%d_%H_%M_%S%.f").to_string();
    let file_path_str = "./graph/".to_string() + caption + &date_str + ".png";
    let root = BitMapBackend::new(
        Path::new(&file_path_str),
        (image_width, image_height))
    .into_drawing_area();

    // Background is white
    root.fill(&WHITE)?;
    
    let caption = "";
    let font = ("sans-serif", 20);

    // Count max
    let mut max_count: u32 = 0;
    for r in (-40..40).step_by(1) {
        let range_low = r as f64 / 10f64;
        let range_high = (r + 1) as f64 / 10f64;
        let mut count = 0;
        for d in &data {
            if *d >= range_low && *d < range_high {
                count += 1;
            }
        }
        max_count = if max_count<count {count} else {max_count};
    }

    // Graph setting
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, font.into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            (-0.1f64..0.1f64).step(0.001).use_round().into_segmented(),
            0u32..max_count
        )?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(&WHITE.mix(0.3))
        .axis_desc_style(font)
        .draw()?;

    chart.draw_series(
        Histogram::vertical(&chart)
            .style(RED.mix(0.5).filled())
            .data(data.iter().map(|x: &f64| (*x, 1))),
    )?;

    Ok(())
}


#[cfg(test)]
mod graph_plotters_test {
    use super::*;

    use crate::deep_learning::common::*;
    
    #[test]
    fn test_plot_rate() {
        let data = vec![0.1, 0.5, 0.8, 0.5, 0.0, 0.5];
        plot_rate(data, "test_rate");
    }

    #[test]
    fn test_plot_histogram() {
        let data = norm_random_vec(30000);
        plot_histogram(data, "test_histo");
    }

    #[test]
    fn test_aa() {
        // let v: Vec<f64> = (-4f64..4f64).collect();
        for n in (-4..4).step_by(2) {
            println!("{}", n);
        }
    }
    
}