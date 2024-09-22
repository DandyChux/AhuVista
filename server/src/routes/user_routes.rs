use actix_web::http::StatusCode;
use actix_web::{web, HttpResponse, Responder};
use actix_web::web::{Json, Data};
use log::{info, error};
use crate::utils::api_error::ErrorType;
use crate::schema::{UserSignupSchema, UserLoginSchema};
use crate::models::{User, UserType};

use crate::AppState;

pub fn user_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/auth")
            .route("/signup", web::post().to(signup))
            .route("/signin", web::post().to(signin))
    );
}

async fn signup(state: Data<AppState>, payload: Json<UserSignupSchema>) -> Result<impl Responder, ErrorType> {
    let user = payload.into_inner();
    let email = user.email;
    let password = user.password;
    let user_type = user.user_type;

    let user = sqlx::query_as!(
        User,
        r#"
        INSERT INTO users (email, password_hash, user_type)
        VALUES ($1, $2, $3)
        "#,
        email,
        password,
        user_type as UserType
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        error!("Failed to create user: {:?}", e);
        ErrorType::DatabaseError(sqlx::Error::from(e)).into()
    })?;

    info!("User created successfully");

    Ok(HttpResponse::Created().body("User signuped successfully"))
}

async fn signin(state: Data<AppState>, payload : Json<UserLoginSchema>) -> Result<impl Responder, ErrorType> {
    let email = &payload .email; 

    let user = sqlx::query_as!(
        User,
        r#"
        SELECT 
            id,
            email,
            password_hash,
            user_type as "user_type: UserType",
            created_at,
            updated_at 
        FROM users 
        WHERE email = $1
        "#,
        email
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        error!("Failed to fetch user: {:?}", e);
        ErrorType::DatabaseError(sqlx::Error::from(e)).into()
    })?;

    if let Some(user) = user {
        Ok(HttpResponse::Ok().body("User logged in successfully"))
    } else {
        Err(ErrorType::NotFound.into()) // Return a not found error if the user is not found
    }
}