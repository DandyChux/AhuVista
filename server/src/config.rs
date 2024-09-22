use log::{info, error, debug};
use std::env;
use dotenvy::{dotenv, from_filename};
use once_cell::sync::Lazy;

#[derive(Debug, Clone)]
pub struct Config {
    pub app_env: String,
    pub database_url: String,
    pub server_host: String,
    pub server_port: u16,
    pub jwt_secret: String,
    pub jwt_expiry: i64,
    pub jwt_maxage: i64,
}

impl Config {
    pub fn init() -> Self {
        if std::env::var_os("RUST_LOG").is_none() {
            // std::env::set_var("RUST_LOG", "actix_web=debug,sqlx=debug,info");
            std::env::set_var("RUST_LOG", "actix_web=info,actix_server=info,sqlx=info,debug");
        }

        // let environment;
        // if cfg!(debug_assertions) {
        //     // Load from `.env.local` in development
        //     from_filename(".env.local").ok().expect("Failed to load .env.local file");
        //     environment = "development";
        // } else {
        //     // Load from `.env` in production
        //     dotenv().ok().expect("Failed to load .env file");
        //     environment = "production";
        // }
        dotenv().map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err)).unwrap();
        let environment = env::var("APP_ENV").unwrap_or_else(|_| "dev".to_string());

        env_logger::init();
        info!("🌍 Starting server in {} environment...", environment);

        let database_url = match env::var("DATABASE_URL") {
            Ok(url) => url.to_string(),
            Err(_) => {
                error!("DATABASE_URL is not set");
                panic!("DATABASE_URL is not set");
            }
        };

        let app_env = env::var("APP_ENV").unwrap_or_else(|_| "dev".to_string());
        let server_host = env::var("SERVER_HOST").unwrap_or_else(|_| "http://localhost".to_string());
        let server_port = env::var("SERVER_PORT").unwrap_or_else(|_| "8000".to_string()).parse::<u16>().unwrap();
        let jwt_secret = env::var("JWT_SECRET").expect("JWT_SECRET must be set");
        let jwt_expiry = env::var("JWT_EXPIRY").unwrap_or_else(|_| "3600".to_string()).parse::<i64>().unwrap();
        let jwt_maxage = env::var("JWT_MAXAGE").unwrap_or_else(|_| "86400".to_string()).parse::<i64>().unwrap();

        Self {
            app_env,
            database_url,
            server_host,
            server_port,
            jwt_secret,
            jwt_expiry,
            jwt_maxage,
        }
    }
}

pub static ENV_CONFIG: Lazy<Config> = Lazy::new(|| Config::init());