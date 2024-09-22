mod model;

use model::NNModel;
use serde_json::json;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref MODEL: Mutex<NNModel> = Mutex::new(NNModel::new(2, 10, 1));
}

#[no_mangle]
pub extern "C" fn generate_prediction(input: *const c_char) -> *const c_char {
    let c_str = unsafe {
        assert!(!input.is_null());
        CStr::from_ptr(input)
    };
    let input_str = c_str.to_str().unwrap();

    // Parse input JSON
    let input_data: Vec<f64> = serde_json::from_str(input_str).unwrap();

    // Generate prediction
    let model = MODEL.lock().unwrap();
    let predictions = model.predict(input_data, 2).unwrap();

    // Convert predictions to JSON
    let output_data = serde_json::to_string(&predictions).unwrap();
    let c_string = CString::new(output_data).unwrap();
    c_string.into_raw()
}
