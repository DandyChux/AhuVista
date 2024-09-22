// This file contains schema definitions for forms of data that will be used in the application.
use serde::{Deserialize, Serialize};
use chrono::prelude::*;
use validator::Validate;

use crate::models::enums::UserType;

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct UserSignupSchema { 
    #[validate(
        email(message = "Invalid email address"),
        length(min = 1, message = "Email is required")
    )]
    pub email: String,
    #[validate(
        length(min = 8, message = "Password must be at least 8 characters")
    )]
    pub password: String,
    pub user_type: UserType
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UserLoginSchema {
    pub email: String,
    pub password: String
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UserProfileUpdateSchema {
    pub full_name: Option<String>,
    pub date_of_birth: Option<NaiveDate>,
    pub gender: Option<String>,
    pub address: Option<String>,
    pub phone_number: Option<String>,
    pub credentials: Option<String>, // Applicable for professionals
}