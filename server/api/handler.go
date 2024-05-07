package api

import (
	"fmt"
	"net/http"
	"predict-rgr/backend/model"
)

func PredictionHandler(w http.ResponseWriter, r *http.Request) {
	// Handle prediction
	prediction := model.PredictOutcome()
	w.Write([]byte(fmt.Sprintf("Prediction: %s", prediction)))
}
