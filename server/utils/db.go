package utils

import (
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/joho/godotenv"
	"gorm.io/driver/postgres"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

var DB *gorm.DB // Exported variable to be used in other packages

// InitDB initializes a connection to the database
func InitDB() error {
	// Load environment variables from .env file
	var envFilePath string
	if os.Getenv("APP_ENV") == "production" {
		envFilePath = filepath.Join("..", ".env")
	} else {
		envFilePath = filepath.Join("..", ".env.local")
	}
	fmt.Println("Loading environment variables from: ", envFilePath)

	// Load environment variables from the chosen file
	if err := godotenv.Load(envFilePath); err != nil {
		log.Fatal("Error loading .env file: ", err)
	}

	// Connect to the database
	err := initializeDB()
	if err != nil {
		return err
	}

	return nil
}

func initializeDB() error {
	dbType := os.Getenv("DB_TYPE")
	connectionString := os.Getenv("DB_CONNECTION_STRING")

	if dbType == "" || connectionString == "" {
		return errors.New("DB_TYPE or DB_CONNECTION_STRING environment variable is not set")
	}

	var dialector gorm.Dialector
	switch dbType {
	case "sqlite":
		dialector = sqlite.Open(connectionString)
	case "postgres":
		dialector = postgres.Open(connectionString)
	default:
		return errors.New("Unsupported database type: " + dbType)
	}

	var err error
	DB, err = gorm.Open(dialector, &gorm.Config{})
	if err != nil {
		return err
	}

	return nil
}

// CloseDB closes the database connection
func CloseDB() error {
	sqlDB, err := DB.DB()
	if err != nil {
		return err
	}
	return sqlDB.Close()
}
