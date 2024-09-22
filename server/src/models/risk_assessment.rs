use serde::{Deserialize, Serialize};
use sqlx::{FromRow, types::Json};
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct RiskAssessment {
    pub id: Uuid,
    pub user_id: Uuid,
    pub assessment: serde_json::Value, // Use serde_json::Value for JSONB fields
    pub created_at: DateTime<Utc>,
}