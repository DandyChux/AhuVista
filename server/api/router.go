package api

import (
	"github.com/gorilla/mux"
)

func NewRouter() *mux.Router {
	router := mux.NewRouter()
	// router.PathPrefix("/api")
	router.HandleFunc("/predict", PredictionHandler).Methods("POST")
	router.HandleFunc("/signup", RegisterHandler).Methods("POST")
	router.HandleFunc("/login", LoginHandler).Methods("POST")
	router.HandleFunc("/logout", LogoutHandler).Methods("POST")
	router.HandleFunc("/refresh", RefreshHandler).Methods("POST")
	return router
}
