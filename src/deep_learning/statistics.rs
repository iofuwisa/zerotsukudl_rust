pub fn average(data: &Vec<f64>) -> f64 {
    let mut sum = 0f64;
    for d in data {
        sum += d;
    }
    return sum / data.len() as f64;
}

pub fn distribute(data: &Vec<f64>) -> (f64, f64) {
    let avg = average(data);
    let mut diff_squared_sum = 0f64;
    for d in data {
        diff_squared_sum += (d - avg).powi(2);
    }
    let dist = diff_squared_sum / data.len() as f64;
    return (dist, avg);
}

pub fn standard_devication(data: &Vec<f64>) -> (f64, f64, f64) {
    let (dist, avg) = distribute(data);
    let std_dev = dist.sqrt();
    return (std_dev, dist, avg);
}