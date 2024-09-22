use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;
use chrono::{DateTime, Utc, NaiveDate};

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct UserProfile {
    pub user_id: Uuid,
    pub full_name: Option<String>,
    pub date_of_birth: Option<NaiveDate>,
    pub gender: Option<String>,
    pub address: Option<String>,
    pub phone_number: Option<String>,
    pub credentials: Option<String>, // Applicable for professionals
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}