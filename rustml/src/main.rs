mod processor;

use std::env;

fn main() {
    println!("Starting the data processor...");
    processor::process_data();
}
