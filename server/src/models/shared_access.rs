use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::models::enums::AccessLevel;

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct SharedAccess {
    pub id: Uuid,
    pub patient_id: Uuid,
    pub professional_id: Uuid,
    pub access_level: AccessLevel,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}