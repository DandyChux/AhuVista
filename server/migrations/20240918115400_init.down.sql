-- Add down migration script here
-- Disable the uuid-ossp extension
DROP EXTENSION IF EXISTS "uuid-ossp";

-- Drop Notifications Table
DROP TABLE IF EXISTS notifications;

-- Drop Wearable Devices Table
DROP TABLE IF EXISTS wearable_devices;

-- Drop Shared Access Table
DROP TABLE IF EXISTS shared_access;

-- Drop Messages Table
DROP TABLE IF EXISTS messages;

-- Drop Risk Assessments Table
DROP TABLE IF EXISTS risk_assessments;

-- Drop Health Data Table
DROP TABLE IF EXISTS health_data;

-- Drop User Profiles Table
DROP TABLE IF EXISTS user_profiles;

-- Drop Users Table
DROP TABLE IF EXISTS users;

-- Drop Enum Type Definitions
DROP TYPE IF EXISTS notification_type_enum;
DROP TYPE IF EXISTS access_level_enum;
DROP TYPE IF EXISTS source_enum;
DROP TYPE IF EXISTS user_type_enum;
