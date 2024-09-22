use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::models::enums::UserType;

#[derive(Serialize, Deserialize, Debug, FromRow)]
pub struct User {
    pub id: Uuid,
    pub email: String,
    pub password_hash: String,
    pub user_type: UserType,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}