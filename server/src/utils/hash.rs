use bcrypt;
use super::api_error::ErrorType;

// Hashes a string value, used for hashing passwords
pub fn hash(s: &String) -> Result<String, ErrorType>{
    let hashed_password = bcrypt::hash(s, 4)
        .map_err(|e| ErrorType::InternalServerError(e.to_string()))?;

    Ok(hashed_password)
}

// Verifies a string value against a hashed value
pub fn verify(password: &str, hash: &str) -> bool {
    let parshed_hash = bcrypt::verify(password, hash).unwrap();

    parshed_hash
}