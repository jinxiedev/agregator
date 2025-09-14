package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// Struct untuk request
type AIRequest struct {
	Message    string    `json:"message"`
	Model      string    `json:"model"`
	ImageURL   string    `json:"image_url,omitempty"`
	ChatID     string    `json:"chat_id,omitempty"`
	SenderID   string    `json:"sender_id,omitempty"`
	History    []Message `json:"history,omitempty"`
}

// Struct untuk response
type AIResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
	Error   string `json:"error,omitempty"`
}

// Struct untuk pesan dalam history
type Message struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

func main() {
	// Validasi environment variables
	requiredEnvVars := []string{"HF_TOKEN", "GROQ_API_KEY", "OPENROUTER_API_KEY"}
	for _, envVar := range requiredEnvVars {
		if os.Getenv(envVar) == "" {
			log.Fatalf("Environment variable %s is required", envVar)
		}
	}
	
	http.HandleFunc("/api/ai", handleAIRequest)
	http.HandleFunc("/health", handleHealthCheck)
	
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	
	log.Printf("Server running on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
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

func prepareMessages(req AIRequest) []map[string]interface{} {
	var messages []map[string]interface{}
	
	// Tambahkan history jika ada
	for _, msg := range req.History {
		message := map[string]interface{}{
			"role":    msg.Role,
			"content": msg.Content,
		}
		messages = append(messages, message)
	}
	
	// Siapkan konten untuk user message
	var content interface{}
	if req.ImageURL != "" {
		// Format multimodal (teks + gambar)
		content = []map[string]interface{}{
			{
				"type": "text",
				"text": req.Message,
			},
			{
				"type": "image_url",
				"image_url": map[string]interface{}{
					"url":    req.ImageURL,
					"detail": "high",
				},
			},
		}
	} else {
		// Hanya teks
		content = req.Message
	}
	
	// Tambahkan user message
	userMessage := map[string]interface{}{
		"role":    "user",
		"content": content,
	}
	messages = append(messages, userMessage)
	
	return messages
}

// Implementasi fungsi-fungsi pemanggilan API AI
func callDeepSeek(req AIRequest) (string, error) {
	payload := map[string]interface{}{
		"model":       "deepseek-ai/DeepSeek-V3.1:fireworks-ai",
		"messages":    prepareMessages(req),
		"max_tokens":  4000,
		"temperature": 0.7,
		"stream":      false,
	}
	
	return callHuggingFaceAPI(payload)
}

func callLlama4(req AIRequest) (string, error) {
	payload := map[string]interface{}{
		"model":       "meta-llama/Llama-4-Maverick-17B-128E-Instruct:groq",
		"messages":    prepareMessages(req),
		"max_tokens":  4000,
		"temperature": 0.7,
	}
	
	return callHuggingFaceAPI(payload)
}

func callGroqLlama(req AIRequest) (string, error) {
	// Siapkan messages dengan system prompt khusus
	messages := prepareMessages(req)
	if len(messages) == 1 && messages[0]["role"] == "user" {
		// Tambahkan system prompt jika tidak ada history
		systemPrompt := map[string]interface{}{
			"role": "system",
			"content": "Kamu adalah Jinxie AI, asisten pintar yang ahli dalam coding, debugging, dan analisis teknis dengan model LLaMA 4. Berikan jawaban yang talkative, informatif, dan jelas, sertakan tips, penjelasan, dan contoh ketika perlu. Gunakan bahasa Indonesia untuk penjelasan umum, tetapi biarkan semua kode tetap dalam bahasa aslinya.",
		}
		messages = append([]map[string]interface{}{systemPrompt}, messages...)
	}
	
	payload := map[string]interface{}{
		"model":       "llama-3.3-70b-versatile",
		"messages":    messages,
		"max_tokens":  4000,
		"temperature": 0.3,
		"top_p":       0.9,
	}
	
	client := &http.Client{Timeout: 30 * time.Second}
	payloadBytes, _ := json.Marshal(payload)
	
	httpReq, err := http.NewRequest("POST", "https://api.groq.com/openai/v1/chat/completions", 
		strings.NewReader(string(payloadBytes)))
	if err != nil {
		return "", err
	}
	
	httpReq.Header.Set("Authorization", "Bearer "+os.Getenv("GROQ_API_KEY"))
	httpReq.Header.Set("Content-Type", "application/json")
	
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := ioutil.ReadAll(resp.Body)
		return "", fmt.Errorf("API error: %s - %s", resp.Status, string(body))
	}
	
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	
	choices, ok := result["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return "", fmt.Errorf("invalid response format")
	}
	
	choice := choices[0].(map[string]interface{})
	message := choice["message"].(map[string]interface{})
	
	return message["content"].(string), nil
}

func callMoonAI(req AIRequest) (string, error) {
	// Siapkan messages dengan system prompt khusus
	messages := prepareMessages(req)
	if len(messages) == 1 && messages[0]["role"] == "user" {
		// Tambahkan system prompt jika tidak ada history
		systemPrompt := map[string]interface{}{
			"role": "system",
			"content": "Kamu adalah Jinxie AI, asisten pintar yang ahli dalam memberikan informasi, analisis, dan penjelasan teknis. Gunakan bahasa Indonesia untuk semua penjelasan agar mudah dimengerti. Fokus pada jawaban yang jelas, ringkas, dan akurat, sertakan contoh atau tabel jika perlu. Jangan menulis kode pemrograman.",
		}
		messages = append([]map[string]interface{}{systemPrompt}, messages...)
	}
	
	payload := map[string]interface{}{
		"model":       "moonshotai/kimi-k2-instruct-0905",
		"messages":    messages,
		"max_tokens":  4000,
		"temperature": 0.3,
		"top_p":       0.9,
	}
	
	client := &http.Client{Timeout: 30 * time.Second}
	payloadBytes, _ := json.Marshal(payload)
	
	httpReq, err := http.NewRequest("POST", "https://api.groq.com/openai/v1/chat/completions", 
		strings.NewReader(string(payloadBytes)))
	if err != nil {
		return "", err
	}
	
	httpReq.Header.Set("Authorization", "Bearer "+os.Getenv("GROQ_API_KEY"))
	httpReq.Header.Set("Content-Type", "application/json")
	
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := ioutil.ReadAll(resp.Body)
		return "", fmt.Errorf("API error: %s - %s", resp.Status, string(body))
	}
	
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	
	choices, ok := result["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return "", fmt.Errorf("invalid response format")
	}
	
	choice := choices[0].(map[string]interface{})
	message := choice["message"].(map[string]interface{})
	
	return message["content"].(string), nil
}

func callQwenCoder(req AIRequest) (string, error) {
	// Siapkan messages dengan system prompt khusus
	messages := prepareMessages(req)
	if len(messages) == 1 && messages[0]["role"] == "user" {
		// Tambahkan system prompt jika tidak ada history
		systemPrompt := map[string]interface{}{
			"role": "system",
			"content": "Kamu adalah AI programmer yang ahli dalam coding. Bantu dengan kode, debugging, dan penjelasan teknis. Gunakan bahasa Indonesia untuk penjelasan umum, tapi pertahankan kode dalam bahasa aslinya.",
		}
		messages = append([]map[string]interface{}{systemPrompt}, messages...)
	}
	
	payload := map[string]interface{}{
		"model":       "qwen/qwen3-coder",
		"messages":    messages,
		"max_tokens":  2000,
		"temperature": 0.3,
	}
	
	client := &http.Client{Timeout: 30 * time.Second}
	payloadBytes, _ := json.Marshal(payload)
	
	httpReq, err := http.NewRequest("POST", "https://openrouter.ai/api/v1/chat/completions", 
		strings.NewReader(string(payloadBytes)))
	if err != nil {
		return "", err
	}
	
	httpReq.Header.Set("Authorization", "Bearer "+os.Getenv("OPENROUTER_API_KEY"))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("HTTP-Referer", "https://github.com/jinxie-bot")
	httpReq.Header.Set("X-Title", "Jinxie WhatsApp Bot")
	
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := ioutil.ReadAll(resp.Body)
		return "", fmt.Errorf("API error: %s - %s", resp.Status, string(body))
	}
	
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	
	choices, ok := result["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return "", fmt.Errorf("invalid response format")
	}
	
	choice := choices[0].(map[string]interface{})
	message := choice["message"].(map[string]interface{})
	
	return message["content"].(string), nil
}

func callSonomaAI(req AIRequest) (string, error) {
	payload := map[string]interface{}{
		"model":       "openrouter/sonoma-sky-alpha",
		"messages":    prepareMessages(req),
		"max_tokens":  5000,
		"temperature": 0.7,
	}
	
	client := &http.Client{Timeout: 30 * time.Second}
	payloadBytes, _ := json.Marshal(payload)
	
	httpReq, err := http.NewRequest("POST", "https://openrouter.ai/api/v1/chat/completions", 
		strings.NewReader(string(payloadBytes)))
	if err != nil {
		return "", err
	}
	
	httpReq.Header.Set("Authorization", "Bearer "+os.Getenv("OPENROUTER_API_KEY"))
	httpReq.Header.Set("Content-Type", "application/json")
	
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := ioutil.ReadAll(resp.Body)
		return "", fmt.Errorf("API error: %s - %s", resp.Status, string(body))
	}
	
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	
	choices, ok := result["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return "", fmt.Errorf("invalid response format")
	}
	
	choice := choices[0].(map[string]interface{})
	message := choice["message"].(map[string]interface{})
	
	return message["content"].(string), nil
}

func callHuggingFaceAPI(payload map[string]interface{}) (string, error) {
	client := &http.Client{Timeout: 30 * time.Second}
	payloadBytes, _ := json.Marshal(payload)
	
	httpReq, err := http.NewRequest("POST", "https://router.huggingface.co/v1/chat/completions", 
		strings.NewReader(string(payloadBytes)))
	if err != nil {
		return "", err
	}
	
	httpReq.Header.Set("Authorization", "Bearer "+os.Getenv("HF_TOKEN"))
	httpReq.Header.Set("Content-Type", "application/json")
	
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := ioutil.ReadAll(resp.Body)
		return "", fmt.Errorf("API error: %s - %s", resp.Status, string(body))
	}
	
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	
	choices, ok := result["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return "", fmt.Errorf("invalid response format")
	}
	
	choice := choices[0].(map[string]interface{})
	message := choice["message"].(map[string]interface{})
	
	return message["content"].(string), nil
}
