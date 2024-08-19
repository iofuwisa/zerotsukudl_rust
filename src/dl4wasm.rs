use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use js_sys::*;
use web_sys::{Request, RequestInit, RequestMode, Response, Element, CanvasRenderingContext2d, HtmlCanvasElement, ImageData};
use ndarray::{
    s,
    Array2
};
use serde::{Deserialize, Serialize};

use crate::deep_learning::neural_network::*;

fn console_log(s: &str) {
    web_sys::console::log_1(&JsValue::from(s));
}

#[wasm_bindgen(start)]
pub fn run() {
    console_log("start!!!!!!!!!!!!!!!!!");
}

#[wasm_bindgen]
pub async fn guess() -> i32 {
    console_log("guess!!!!!!!!!!!!!!");
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas: Element = document.get_element_by_id("mainCanvas").unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| ())
        .unwrap();

    let context = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .unwrap();

    let image_width = canvas.width();
    let image_height = canvas.height();

    let data: ImageData = context
        .get_image_data(0.0, 0.0, image_width as f64, image_height as f64)
        .unwrap();
    
    // sRGB
    let image_data_srgb = data.data().to_vec();

    let mut image_data_gray = Vec::<u8>::with_capacity((image_width*image_height) as usize);
    for i in 0..image_data_srgb.len()/4 {
        let mut max = image_data_srgb[i*4];
        for j in 1..4 {
            if max < image_data_srgb[i*4+j] {
                max = image_data_srgb[i*4+j]
            }
        }
        image_data_gray.push(max);
    }

    // console_log(&format!("image_data_srgb: {:?}", image_data_srgb.len()));
    // console_log(&format!("image_data_srgb: {:?}", image_data_srgb));
    // console_log(&format!("image_data_gray: {:?}", image_data_gray.len()));
    // console_log(&format!("image_data_gray: {:?}", image_data_gray));

    let img = Array2::from_shape_vec((image_width as usize, image_height as usize), image_data_gray).unwrap();
    let mut converted_img = Array2::<f64>::zeros((28, 28));

    // convert 28*28
    for hi in 0..28-1 {
        let hi_range_head = (image_height as f64 *  hi as f64         / 28f64).round() as usize;
        let hi_range_tail = (image_height as f64 * (hi as f64 + 1f64) / 28f64).round() as usize;
        let hi_range = hi_range_head..hi_range_tail;
        for wi in 0..28-1 {
            let wi_range_head = (image_width as f64 *  wi as f64          / 28f64).round() as usize;
            let wi_range_tail = (image_width as f64 * (wi as f64 + 1f64 ) / 28f64).round() as usize;
            let wi_range = wi_range_head..wi_range_tail;

            let l = img.slice(s![hi_range.clone(), wi_range]);
            let mut max = l[(0, 0)];
            for n in l {
                if max < *n {
                    max = *n;
                }
            }
            // converted_img[(hi, wi)] = max as f64 / 255f64;
            converted_img[(hi, wi)] = if max > 0u8 {1f64} else {0f64};
        }
    }

    let dl_data_string = fetch().await;
    // console_log(dl_data_string.as_str());

    let mut nn = NeuralNetwork::import(dl_data_string.as_str());
    
    // gen stub batch
    let converted_img = converted_img.to_shared().reshape((1,28*28));
    let MINIBATCH_SIZE: usize = 100;
    let mut batch = Array2::<f64>::zeros((MINIBATCH_SIZE, 28*28));
    batch.assign(&converted_img);

    for i in 0..28 {
        let mut line = "".to_string();
        for j in 0..28 {
            if batch[(0,i*28+j)] > 0f64 {
                line.push_str("x");
            } else {
                line.push_str(" ");
            }
        }
        console_log(&line);
    }

    let res = nn.guess(&batch);

    let mut max_index = 0;
    for i in 1..10 {
        if res[(0, max_index)] < res[(0, i)] {
            max_index = i;
        }
    }

    console_log(&format!("{:?}", res));
    console_log(&format!("result: {}", max_index));

    console_log("finish");

    return max_index as i32;
}

#[wasm_bindgen]
pub async fn fetch() -> String {
    let mut opts = RequestInit::new();
    opts.method("GET");
    opts.mode(RequestMode::SameOrigin);
    let url = "/nn.csv";
    let request_res = Request::new_with_str_and_init(&url, &opts);
    if let Ok(request) = request_res {
        let window = web_sys::window().unwrap();
        let resp_value_res = JsFuture::from(window.fetch_with_request(&request)).await;

        if let Ok(resp_value) = resp_value_res {
            let resp: Response = resp_value.dyn_into().unwrap();
            console_log("Success");
            console_log(&format!("url: {}", resp.url()));
            console_log(&format!("status: {}", resp.status_text()));

            let text_res = resp.text();
            if let Ok(text_future) = text_res {
                let text_future_res = JsFuture::from(text_future).await;
                
                if let Ok(text) = text_future_res {
                    // console_log(&format!("text: {}", text.as_string().unwrap()));
                    return text.as_string().unwrap();
                } else if let Err(e) = text_future_res {
                    console_log("Futue error");
                    return "".to_string();
                }
            
            } else if let Err(e) = text_res {
                console_log("Text error");
                return "".to_string();
            }
            
        } else if let Err(e) = resp_value_res {
            console_log("Request error");
            return "".to_string();
        }

    } else if let Err(e) = request_res {
        console_log("Init error");
        return "".to_string();
    }
    return "".to_string();
}
