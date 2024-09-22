use serde::{Deserialize, Serialize};
use sqlx::Type;

#[derive(Debug, Serialize, Deserialize, Type)]
#[sqlx(type_name = "user_type_enum")] // Matches the PostgreSQL enum type
#[sqlx(rename_all = "lowercase")]     // Matches the case of the PostgreSQL enum values
pub enum UserType {
    Patient,
    Professional,
}

#[derive(Debug, Serialize, Deserialize, Type)]
#[sqlx(type_name = "source_enum")]
#[sqlx(rename_all = "lowercase")]
pub enum SourceType {
    Manual,
    Imported,
    Wearable,
}

#[derive(Debug, Serialize, Deserialize, Type)]
#[sqlx(type_name = "access_level_enum")]
#[sqlx(rename_all = "lowercase")]
pub enum AccessLevel {
    Read,
    Write,
}

#[derive(Debug, Serialize, Deserialize, Type)]
#[sqlx(type_name = "notification_type_enum")]
#[sqlx(rename_all = "lowercase")]
pub enum NotificationType {
    Appointment,
    Alert,
    Reminder,
}
