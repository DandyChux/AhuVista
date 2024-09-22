use std::fmt;

use serde::{Deserialize, Serialize};
use actix_web::{http::StatusCode, HttpResponse, error::ResponseError};
use validator::{ValidationError, ValidationErrors};

use crate::models::message;

// #[derive(Debug)]
// pub enum ErrorType {
//     ValidationError,
//     InternalServerError,
//     NotFound
// }

// #[derive(Serialize, Deserialize, Debug)]
// pub struct ErrorType {
//     status_code: u16,
//     message: String,
//     error: String
// }

// #[derive(Debug, Display, Error)]
// pub enum CustomError {
//     #[display(fmt = "An internal server error occurred. Please try again later.")]
//     InternalServerError,
//     #[display(fmt = "Not found!")]
//     NotFound,
//     #[display(fmt = "Validation Error: {}", field)]
//     ValidationError { field: String },
//     #[display(fmt = "Bad request")]
//     BadClientData
// }

// impl ErrorType {
//     // pub fn new(message: &str, status: StatusCode) -> Self {
//     //     Self {
//     //         status: status,
//     //         message: message.to_string(),
//     //     }
//     // }
//     pub fn new(err_type: StatusCode, message: Option<String>, waiting_time: Option<String>, status: Option<String>) -> Self {
//         Self {
//             err_type,
//             message,
//             waiting_time,
//             status
//         }
//     }
// }

// impl From<sqlx::Error> for ErrorType {
//     fn from(err: sqlx::Error) -> Self {
//         match err {
//             sqlx::Error::RowNotFound => Self::new("Resource not found", StatusCode::NOT_FOUND),
//             _ => Self::new("Internal server error", StatusCode::INTERNAL_SERVER_ERROR),
//         }
//     }
// }

// impl std::fmt::Display for ErrorType {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(
//             f, 
//             "status: {}, message: {}",
//             self.status, self.message
//         )
//     }
// }

#[derive(Debug)]
pub enum ErrorType {
    ValidationError(ValidationErrors),
    InternalServerError(String),
    DatabaseError(sqlx::Error),
    NotFound
}

impl ErrorType {
    fn status_code(&self) -> StatusCode {
        match self {
            ErrorType::ValidationError(_) => StatusCode::BAD_REQUEST,
            ErrorType::InternalServerError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorType::DatabaseError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorType::NotFound => StatusCode::NOT_FOUND,
            _ => StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

impl fmt::Display for ErrorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status_code = self.status_code();
        let status_text = status_code.as_str();

        match self {
            ErrorType::ValidationError(msg) => {
                write!(f, "{}: {}", status_text, msg)
            }
            ErrorType::InternalServerError(msg) => {
                write!(f, "{}: {}", status_text, msg)
            }
            ErrorType::DatabaseError(err) => {
                write!(f, "{}: {}", status_text, err)
            }
            ErrorType::NotFound => {
                write!(f, "{}: Not found", status_text)
            }
        }
    }
}

impl std::error::Error for ErrorType {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ErrorType::DatabaseError(err) => Some(err),
            _ => None
        }
    }
}

impl ResponseError for ErrorType {
    fn status_code(&self) -> StatusCode {
        self.status_code()
    }

    fn error_response(&self) -> HttpResponse {
        let status_code = self.status_code();
        let message = self.to_string();

        HttpResponse::build(status_code).body(message)
    }
}