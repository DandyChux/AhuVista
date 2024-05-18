package api

import (
	"encoding/json"
	"net/http"
	"os"
	"time"

	"github.com/dandychux/predict-rgr/server/models"
	"github.com/dandychux/predict-rgr/server/utils"
	"github.com/lucsky/cuid"
)

// PredictionRequest represents the expected format of the JSON request body
type PredictionRequest struct {
	PatientData string `json:"patientData"`
}

// PredictionResponse represents the structure of the JSON response body
type PredictionResponse struct {
	Prediction string `json:"prediction"`
	Error      string `json:"error,omitempty"`
}

func PredictionHandler(w http.ResponseWriter, r *http.Request) {
	// Decode the request body into the struct
	var req PredictionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	prediction, err := models.PredictOutcome(req.PatientData)
	if err != nil {
		http.Error(w, "Error predicting outcome: "+err.Error(), http.StatusInternalServerError)
		return
	}

	resp := PredictionResponse{Prediction: prediction}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

type RegisterRequest struct {
	Username    string    `json:"username"`
	Password    string    `json:"password"`
	Email       string    `json:"email"`
	FirstName   string    `json:"firstName"`
	LastName    string    `json:"lastName"`
	Role        string    `json:"role"`
	Gender      string    `json:"gender"`
	BloodType   string    `json:"bloodType"`
	DateOfBirth time.Time `json:"dateOfBirth"`
}

type AuthResponse struct {
	JWT     string `json:"jwt"`
	Session string `json:"session"`
}

func RegisterHandler(w http.ResponseWriter, r *http.Request) {
	// Decode the request body into the struct
	var req RegisterRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Validate the username and password
	username := req.Username
	password := req.Password

	// Check if username is already taken
	var user models.User
	result := utils.DB.Where("username = ?", username).First(&user)
	if result.Error == nil {
		http.Error(w, "Username is already taken", http.StatusConflict)
		return
	}

	// Hash the password before storing it in the database
	hashedPassword, err := utils.HashPassword(password)
	if err != nil {
		http.Error(w, "Error hashing password", http.StatusInternalServerError)
		return
	}

	// Create a new user in the database
	user = models.User{
		Username:     username,
		PasswordHash: hashedPassword,
		UserID:       cuid.New(),
		Email:        req.Email,
		Role:         req.Role,
	}
	if err := utils.DB.Create(&user).Error; err != nil {
		http.Error(w, "Error creating user", http.StatusInternalServerError)
		return
	}

	// Create a new patient profile for the user
	patient := models.Patient{
		UserID:      user.UserID,
		PatientID:   cuid.New(),
		FirstName:   req.FirstName,
		LastName:    req.LastName,
		DateOfBirth: req.DateOfBirth,
		Gender:      req.Gender,
		BloodType:   &req.BloodType,
	}
	if err := utils.DB.Create(&patient).Error; err != nil {
		http.Error(w, "Error creating patient profile", http.StatusInternalServerError)
		return
	}

	// Generate JWT token
	jwtKey := []byte(os.Getenv("JWT_KEY"))
	tokenString, err := utils.GenerateJWT(req.Username, jwtKey)
	if err != nil {
		http.Error(w, "Error generating JWT token", http.StatusInternalServerError)
		return
	}

	// Create the session token
	sessionToken := utils.GenerateSessionToken()

	// Return the tokens
	resp := AuthResponse{JWT: tokenString, Session: sessionToken}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

type LoginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

func LoginHandler(w http.ResponseWriter, r *http.Request) {
	// Decode the request body into the struct
	var req LoginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Validate the username and password
	// If they are valid, create a JWT token and a session token
	username := req.Username
	password := req.Password

	// Check if the user exists
	var user models.User
	result := utils.DB.Where("username = ?", username).First(&user)
	if result.Error != nil {
		http.Error(w, "Invalid username. User does not exist in the database.", http.StatusUnauthorized)
		return
	}

	// Compare the password with the hashed password
	if !utils.ComparePasswords(user.PasswordHash, password) {
		http.Error(w, "Invalid password. Please check and try again.", http.StatusUnauthorized)
		return
	}

	// Generate JWT token
	jwtKey := []byte(os.Getenv("JWT_KEY"))
	tokenString, err := utils.GenerateJWT(req.Username, jwtKey)
	if err != nil {
		http.Error(w, "Error generating JWT token", http.StatusInternalServerError)
		return
	}

	// Create the session token
	sessionToken := utils.GenerateSessionToken()

	// Return the tokens
	resp := AuthResponse{JWT: tokenString, Session: sessionToken}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

type LogoutRequest struct {
	Session string `json:"session"`
}

func LogoutHandler(w http.ResponseWriter, r *http.Request) {
	// Decode the request body into the struct
	var req LogoutRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Invalidate the session token
	err := utils.InvalidateSessionToken(req.Session)
	if err != nil {
		http.Error(w, "Error invalidating session token", http.StatusInternalServerError)
		return
	}

	// Return a success message
	json.NewEncoder(w).Encode(map[string]string{"message": "Successfully logged out"})
}

type RefreshRequest struct {
	JWT     string `json:"jwt"`
	Session string `json:"session"`
}

func RefreshHandler(w http.ResponseWriter, r *http.Request) {
	// Decode the request body into the struct
	var req RefreshRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Check if the JWT and session tokens are provided
	if req.JWT == "" || req.Session == "" {
		http.Error(w, "Invalid JWT or session token", http.StatusUnauthorized)
		return
	}

	// Validate the JWT token
	jwtKey := []byte(os.Getenv("JWT_KEY"))
	claims, err := utils.VerifyJWT(req.JWT, jwtKey)
	if err != nil {
		http.Error(w, "Invalid JWT token", http.StatusUnauthorized)
		return
	}

	// Validate the session token
	if !utils.VerifySessionToken(req.Session) {
		http.Error(w, "Invalid session token", http.StatusUnauthorized)
		return
	}

	// Assume we extract the username from the validated JWT claims
	username, ok := claims["username"].(string)
	if !ok {
		http.Error(w, "Invalid JWT claims", http.StatusUnauthorized)
		return
	}

	// Generate a new JWT token
	newTokenString, err := utils.GenerateJWT(username, jwtKey)
	if err != nil {
		http.Error(w, "Error generating JWT token", http.StatusInternalServerError)
		return
	}

	// Create the new session token
	newSessionToken := utils.GenerateSessionToken()

	// Return the new tokens
	resp := AuthResponse{JWT: newTokenString, Session: newSessionToken}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
