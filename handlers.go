package main

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
)

func handleRoot(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":    "online",
		"service":   "jinxie-ai-aggregator",
		"version":   "1.0.0",
		"endpoint":  "POST /api/ai",
		"models":    "deepseek, llama4, groq-llama, moon-ai, qwen-coder, sonoma-ai",
	})
}

func handleHealthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok", "service": "jinxie-ai-aggregator"})
}

func handleAIRequest(w http.ResponseWriter, r *http.Request) {
	// Set CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}
	
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	// Parse request body
	var req AIRequest
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		sendErrorResponse(w, "Failed to read request body", err)
		return
	}
	
	if err := json.Unmarshal(body, &req); err != nil {
		sendErrorResponse(w, "Failed to parse JSON", err)
		return
	}
	
	// Validasi request
	if req.Message == "" {
		sendErrorResponse(w, "Message is required", nil)
		return
	}
	
	if req.Model == "" {
		sendErrorResponse(w, "Model is required", nil)
		return
	}
	
	// Proses berdasarkan model yang diminta
	var response string
	var processErr error
	
	switch req.Model {
	case "deepseek":
		response, processErr = callDeepSeek(req)
	case "llama4":
		response, processErr = callLlama4(req)
	case "groq-llama":
		response, processErr = callGroqLlama(req)
	case "moon-ai":
		response, processErr = callMoonAI(req)
	case "qwen-coder":
		response, processErr = callQwenCoder(req)
	case "sonoma-ai":
		response, processErr = callSonomaAI(req)
	default:
		sendErrorResponse(w, "Unknown model: "+req.Model, nil)
		return
	}
	
	if processErr != nil {
		sendErrorResponse(w, "AI processing error", processErr)
		return
	}
	
	// Kirim response sukses
	resp := AIResponse{
		Success: true,
		Message: response,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func sendErrorResponse(w http.ResponseWriter, message string, err error) {
	resp := AIResponse{
		Success: false,
		Error:   message,
	}
	
	if err != nil {
		resp.Error += ": " + err.Error()
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusInternalServerError)
	json.NewEncoder(w).Encode(resp)
}
