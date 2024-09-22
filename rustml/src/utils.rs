use std::fs::File;
use std::io;

pub fn load_file(path: &String) -> std::result::Result<String, io::Error> {
    let mut file = File::open(path)?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    Ok(contents)
}