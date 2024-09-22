use sqlx::PgPool;
use crate::config::ENV_CONFIG;
use log::info;

pub async fn init_pool() -> Result<PgPool, sqlx::Error> {
    let pool = PgPool::connect(&ENV_CONFIG.database_url).await?;
    info!("Connected to database");
    Ok(pool)
}