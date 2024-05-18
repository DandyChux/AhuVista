package main

import (
	"log"
	"net/http"

	"github.com/dandychux/predict-rgr/server/api"
	"github.com/dandychux/predict-rgr/server/middleware"
	"github.com/dandychux/predict-rgr/server/models"
	"github.com/dandychux/predict-rgr/server/utils"
	"github.com/rs/cors"
)

func main() {
	// Initialize the database
	err := utils.InitDB()
	if err != nil {
		log.Fatalf("Error initializing database: %v", err)
	}

	// Perform AutoMigrate
	utils.DB.AutoMigrate(&models.User{}, &models.Patient{}, &models.HealthRecord{}, &models.Prediction{}, &models.AuthenticationLog{}, &models.Session{})

	// Create a new router
	router := api.NewRouter()

	// Custom CORS configuration
	corsConfig := cors.New(cors.Options{
		AllowedHeaders: []string{"Authorization", "Origin", "Content-Type", "Accept"},
		AllowedMethods: []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		// AllowedOrigins: []string{"*"},
		AllowedOrigins:   []string{"http://localhost:3000"},
		AllowCredentials: true,
		// Enable Debugging for testing, consider disabling in production
		Debug: true,
	})

	// Add CORS middleware
	router.Use(corsConfig.Handler)

	// Add middleware
	router.Use(middleware.LoggingMiddleware)

	// Start the server
	server := http.Server{
		Addr:    ":8080",
		Handler: router,
	}

	log.Println("Starting server on :8080")
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("Error starting server: %v", err)
	}

	// Close the database connection when main function returns
	defer func() {
		if err := utils.CloseDB(); err != nil {
			log.Printf("Error closing database: %v", err)
		}
	}()
}
