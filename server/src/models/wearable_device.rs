use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct WearableDevice {
    pub id: Uuid,
    pub user_id: Uuid,
    pub device_type: String,
    pub device_id: String,
    pub created_at: DateTime<Utc>,
}