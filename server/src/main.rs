pub mod routes;
pub mod db;
pub mod utils;
pub mod schema;
pub mod ml;
pub mod config;
pub mod models;
pub mod middleware;

use actix_web::middleware::Logger;
use actix_cors::Cors;
use actix_web::{web, App, HttpServer, http::header, web::Data};
use log::{info, error, debug};
use std::{env, sync::{Arc, RwLock}};
use config::{Config, ENV_CONFIG};
use sqlx::PgPool;
use crate::utils::api_error::ErrorType;

pub struct AppState {
    db: PgPool,
    env: Config
    // model: ml::Model
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let pool = db::init_pool().await.expect("Failed to initailze database");

    let mut _migrations = match sqlx::migrate!("./migrations").run(&pool).await {
        Ok(migrations) => {
            info!("Migrations run successfully");
            migrations
        }
        Err(err) => {
            error!("Failed to run migrations: {:?}", err);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "Failed to run migrations"));
        }
    };

    let app_state = Arc::new(RwLock::new(AppState {
        db: pool.clone(),
        env: ENV_CONFIG.clone()
    }));

    HttpServer::new(move || {
        let cors = Cors::default()
        .allowed_methods(vec!["GET", "POST", "PATCH", "DELETE"])
        .allowed_headers(vec![
            header::CONTENT_TYPE,
            header::AUTHORIZATION,
            header::ACCEPT,
        ])
        .supports_credentials();
        App::new()
            .app_data(web::Data::new(app_state.clone()))
            .service(
                web::scope("/api")
                    .configure(routes::init)
            )
            .wrap(cors)
            .wrap(Logger::default())
    })
    .bind((ENV_CONFIG.server_host.clone(), ENV_CONFIG.server_port))?
    .run()
    .await

}
