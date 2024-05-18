package utils

import (
	"crypto/rand"
	"encoding/base64"
	"errors"
	"time"

	"github.com/dandychux/predict-rgr/server/models"
	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/crypto/bcrypt"
)

func GenerateSessionToken() string {
	// Create a byte slice of size 32
	b := make([]byte, 32)

	// Generate random bytes
	_, err := rand.Read(b)
	if err != nil {
		panic(err)
	}

	// Conver the byte slice to a base64 string
	token := base64.StdEncoding.EncodeToString(b)

	return token
}

func VerifySessionToken(token string) bool {
	// Check if the token is a valid base64 string
	_, err := base64.StdEncoding.DecodeString(token)
	if err != nil {
		return false
	}

	// Check if the token exists in the database and is still valid
	var session models.Session
	result := DB.Where("session_token = ? AND expires_at > ? AND is_valid = ?", token, time.Now(), true).First(&session)
	if result.Error != nil {
		return false
	}

	return true
}

func InvalidateSessionToken(token string) error {
	// This should include logic to remove the token from the database or cache
	result := DB.Model(&models.Session{}).Where("session_token = ?", token).Update("is_valid", false)
	if result.Error != nil {
		return result.Error
	}

	return nil
}

// VerifyJWT verifies the JWT token and returns the claims if the token is valid
func VerifyJWT(tokenString string, jwtKey []byte) (jwt.MapClaims, error) {
	// Parse the JWT token
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		// Check the signing method
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, errors.New("Invalid signing method")
		}

		return jwtKey, nil
	})

	if err != nil {
		return nil, err
	}

	// Check if the token is valid
	if !token.Valid {
		return nil, errors.New("Invalid token")
	}

	// Get the claims
	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok {
		return nil, errors.New("Invalid claims")
	}

	return claims, nil
}

// GenerateJWT generates a new JWT token with the given claims
func GenerateJWT(username string, jwtKey []byte) (string, error) {
	// Create a new token with the HS256 signing method
	token := jwt.New(jwt.SigningMethodHS256)
	claims := token.Claims.(jwt.MapClaims)

	// Set the claims for the token
	claims["username"] = username
	claims["exp"] = time.Now().Add(time.Hour * 24).Unix()

	// Sign and get the complete encoded token as a string
	tokenString, err := token.SignedString(jwtKey)
	if err != nil {
		return "", err
	}

	return tokenString, nil
}

// HashPassword hashes the given password using bcrypt
func HashPassword(password string) (string, error) {
	// Check if the password is empty
	if password == "" {
		return "", errors.New("Password cannot be empty")
	}

	// Convert the password string to a byte slice
	var passwordBytes = []byte(password)

	// Hash the password with the min cost
	hashedBytes, err := bcrypt.GenerateFromPassword(passwordBytes, bcrypt.MinCost)

	return string(hashedBytes), err
}

// ComparePasswords compares the given password with the hashed password
func ComparePasswords(hashedPassword, password string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(hashedPassword), []byte(password))

	return err == nil
}
