package main

import (
	"log"
	"net/http"
	"os"
)

func main() {
	// Validasi environment variables
	requiredEnvVars := []string{"HF_TOKEN", "GROQ_API_KEY", "OPENROUTER_API_KEY"}
	for _, envVar := range requiredEnvVars {
		if os.Getenv(envVar) == "" {
			log.Printf("WARNING: Environment variable %s is not set", envVar)
		}
	}
	
	// Setup routes
	http.HandleFunc("/api/ai", handleAIRequest)
	http.HandleFunc("/health", handleHealthCheck)
	http.HandleFunc("/", handleRoot)
	
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	
	log.Printf("Server running on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}
