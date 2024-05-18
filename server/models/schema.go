package models

import (
	"time"

	"gorm.io/gorm"
)

// User represents a user in the system.
type User struct {
	UserID       string         `gorm:"primaryKey;not null" json:"userID,omitempty"`
	Username     string         `gorm:"uniqueIndex;unique;not null" json:"username,omitempty"`
	PasswordHash string         `gorm:"not null" json:"passwordHash,omitempty"`
	Email        string         `gorm:"uniqueIndex" json:"email,omitempty"`
	Role         string         `gorm:"not null" json:"role,omitempty"`
	CreatedAt    time.Time      `gorm:"autoCreateTime" json:"createdAt,omitempty"`
	UpdatedAt    time.Time      `gorm:"autoUpdateTime" json:"updatedAt,omitempty"`
	DeletedAt    gorm.DeletedAt `gorm:"index" json:"deletedAt,omitempty"`
}

// Patient represents a patient's profile.
type Patient struct {
	PatientID   string         `gorm:"primaryKey;not null" json:"patientID,omitempty"`
	UserID      string         `gorm:"foreignKey:UserID;not null;constraint:OnUpdate:CASCADE,OnDelete:CASCADE" json:"userID,omitempty"`
	FirstName   string         `gorm:"not null" json:"firstName,omitempty"`
	LastName    string         `gorm:"not null" json:"lastName,omitempty"`
	DateOfBirth time.Time      `gorm:"not null" json:"dateOfBirth,omitempty"`
	Gender      string         `gorm:"not null" json:"gender,omitempty"`
	BloodType   *string        `gorm:"type:varchar(3)" json:"bloodType,omitempty"`
	CreatedAt   time.Time      `gorm:"autoCreateTime" json:"createdAt,omitempty"`
	UpdatedAt   time.Time      `gorm:"autoUpdateTime" json:"updatedAt,omitempty"`
	DeletedAt   gorm.DeletedAt `gorm:"index" json:"deletedAt,omitempty"`
}

// HealthRecord represents a health record for a patient.
type HealthRecord struct {
	RecordID          string         `gorm:"primaryKey;not null" json:"recordID,omitempty"`
	PatientID         string         `gorm:"foreignKey:PatientID;not null;constraint:OnUpdate:CASCADE,OnDelete:CASCADE" json:"patientID,omitempty"`
	RecordDate        time.Time      `gorm:"not null" json:"recordDate,omitempty"`
	Height            float64        `gorm:"not null" json:"height,omitempty"`
	Weight            float64        `gorm:"not null" json:"weight,omitempty"`
	BloodPressure     *string        `gorm:"type:varchar(7)" json:"bloodPressure,omitempty"`
	HeartRate         *int           `json:"heartRate,omitempty"`
	Symptoms          string         `gorm:"type:json" json:"symptoms,omitempty"`
	OtherMeasurements *string        `gorm:"type:json" json:"otherMeasurements,omitempty"`
	CreatedAt         time.Time      `gorm:"autoCreateTime" json:"createdAt,omitempty"`
	UpdatedAt         time.Time      `gorm:"autoUpdateTime" json:"updatedAt,omitempty"`
	DeletedAt         gorm.DeletedAt `gorm:"index" json:"deletedAt,omitempty"`
}

// Prediction represents a prediction made for a patient.
type Prediction struct {
	PredictionID      string         `gorm:"primaryKey;not null" json:"predictionID,omitempty"`
	PatientID         string         `gorm:"foreignKey:PatientID;not null;constraint:OnUpdate:CASCADE,OnDelete:CASCADE" json:"patientID,omitempty"`
	PredictionDate    time.Time      `gorm:"not null" json:"predictionDate,omitempty"`
	PredictionDetails string         `gorm:"type:json; not null" json:"predictionDetails,omitempty"`
	ModelUsed         string         `gorm:"not null" json:"modelUsed,omitempty"`
	ConfidenceLevel   float64        `gorm:"not null" json:"confidenceLevel,omitempty"`
	CreatedAt         time.Time      `gorm:"autoCreateTime" json:"createdAt,omitempty"`
	UpdatedAt         time.Time      `gorm:"autoUpdateTime" json:"updatedAt,omitempty"`
	DeletedAt         gorm.DeletedAt `gorm:"index" json:"deletedAt,omitempty"`
}

// AuthenticationLog represents an entry in the authentication logs.
type AuthenticationLog struct {
	LogID        string         `gorm:"primaryKey;not null" json:"logID,omitempty"`
	UserID       string         `gorm:"foreignKey:UserID;not null;constraint:OnUpdate:CASCADE,OnDelete:CASCADE" json:"userID,omitempty"`
	LoginTime    time.Time      `json:"loginTime,omitempty"`
	LogoutTime   time.Time      `json:"logoutTime,omitempty"`
	SessionToken string         `gorm:"uniqueIndex;unique;not null" json:"sessionToken,omitempty"`
	IPAddress    string         `json:"ipAddress,omitempty"`
	CreatedAt    time.Time      `gorm:"autoCreateTime" json:"createdAt,omitempty"`
	UpdatedAt    time.Time      `gorm:"autoUpdateTime" json:"updatedAt,omitempty"`
	DeletedAt    gorm.DeletedAt `gorm:"index" json:"deletedAt,omitempty"`
}

// Session represents a user session.
type Session struct {
	SessionID    string         `gorm:"primaryKey;not null" json:"sessionID,omitempty"`
	UserID       string         `gorm:"foreignKey:UserID;not null;constraint:OnUpdate:CASCADE,OnDelete:CASCADE" json:"userID,omitempty"`
	SessionToken string         `gorm:"uniqueIndex;unique;not null" json:"sessionToken,omitempty"`
	ExpiresAt    time.Time      `json:"expiresAt,omitempty"`
	CreatedAt    time.Time      `gorm:"autoCreateTime" json:"createdAt,omitempty"`
	UpdatedAt    time.Time      `gorm:"autoUpdateTime" json:"updatedAt,omitempty"`
	DeletedAt    gorm.DeletedAt `gorm:"index" json:"deletedAt,omitempty"`
	IsValid      bool           `json:"isValid,omitempty"`
}
