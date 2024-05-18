package middleware

import (
	"log"
	"net/http"
)

func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Create a response writer that captures the status code
		rw := &responseWriter{ResponseWriter: w}
		next.ServeHTTP(rw, r)
		log.Println(r.Method, "-", r.RequestURI, "(", rw.getStatus(), ")")
	})
}

type responseWriter struct {
	http.ResponseWriter
	status int
}

func (rw *responseWriter) WriteHeader(status int) {
	rw.status = status
	rw.ResponseWriter.WriteHeader(status)
}

func (rw *responseWriter) getStatus() int {
	if rw.status == 0 {
		return http.StatusOK
	}
	return rw.status
}
