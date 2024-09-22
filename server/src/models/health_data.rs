use serde::{Deserialize, Serialize};
use sqlx::{FromRow, types::Json};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::models::enums::SourceType;

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct HealthData {
    pub id: Uuid,
    pub user_id: Uuid,
    pub data_type: String,
    pub data: serde_json::Value, // Use serde_json::Value for JSONB fields
    pub source: SourceType,
    pub created_at: DateTime<Utc>,
}