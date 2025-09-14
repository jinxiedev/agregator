package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// ==================== MODELS ====================
type AIRequest struct {
	Message     string                 `json:"message"`
	Model       string                 `json:"model"`
	ImageURL    string                 `json:"imageUrl,omitempty"`
	ChatID      string                 `json:"chatId,omitempty"`
	SenderID    string                 `json:"senderId,omitempty"`
	MaxTokens   int                    `json:"maxTokens,omitempty"`
	Temperature float64                `json:"temperature,omitempty"`
	Options     map[string]interface{} `json:"options,omitempty"`
}

type AIResponse struct {
	Success   bool        `json:"success"`
	Model     string      `json:"model"`
	Response  string      `json:"response"`
	Timestamp string      `json:"timestamp"`
	Error     string      `json:"error,omitempty"`
}

type Message struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	Timestamp int64  `json:"timestamp"`
}

type ConversationMemory struct {
	Messages []Message `json:"messages"`
}

// ==================== CONFIG ====================
type Config struct {
	Port              string
	APIKeys           []string
	HuggingFaceToken  string
	GroqAPIKey        string
	OpenRouterAPIKey  string
}

func loadConfig() *Config {
	return &Config{
		Port:              getEnv("PORT", "8080"),
		APIKeys:           strings.Split(getEnv("API_KEYS", "default_api_key_1,default_api_key_2"), ","),
		HuggingFaceToken:  getEnv("HF_TOKEN", ""),
		GroqAPIKey:        getEnv("GROQ_API_KEY", ""),
		OpenRouterAPIKey:  getEnv("OPENROUTER_API_KEY", ""),
	}
}

func getEnv(key, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return value
}

// ==================== MEMORY MANAGEMENT ====================
var (
	conversationMemory = make(map[string]*ConversationMemory)
	memoryMutex        = &sync.RWMutex{}
)

func getConversationHistory(chatId, senderId string, maxMessages int) []Message {
	key := chatId + ":" + senderId
	
	memoryMutex.RLock()
	defer memoryMutex.RUnlock()
	
	if memory, exists := conversationMemory[key]; exists {
		if len(memory.Messages) > maxMessages {
			return memory.Messages[len(memory.Messages)-maxMessages:]
		}
		return memory.Messages
	}
	
	return []Message{}
}

func addToConversationHistory(chatId, senderId, role, content string) {
	key := chatId + ":" + senderId
	
	memoryMutex.Lock()
	defer memoryMutex.Unlock()
	
	if _, exists := conversationMemory[key]; !exists {
		conversationMemory[key] = &ConversationMemory{
			Messages: []Message{},
		}
	}
	
	message := Message{
		Role:      role,
		Content:   content,
		Timestamp: time.Now().UnixMilli(),
	}
	
	conversationMemory[key].Messages = append(conversationMemory[key].Messages, message)
	
	// Auto cleanup
	if len(conversationMemory) > 1000 {
		for k := range conversationMemory {
			if k != key {
				delete(conversationMemory, k)
				break
			}
		}
	}
}

func clearConversationHistory(chatId, senderId string) {
	key := chatId + ":" + senderId
	
	memoryMutex.Lock()
	defer memoryMutex.Unlock()
	
	delete(conversationMemory, key)
}

// ==================== AI SERVICES ====================
func callDeepSeek(message, imageURL, chatID, senderID string, cfg *Config) (string, error) {
	messages := []map[string]interface{}{}
	
	if chatID != "" && senderID != "" {
		history := getConversationHistory(chatID, senderID, 10)
		for _, msg := range history {
			messages = append(messages, map[string]interface{}{
				"role":    msg.Role,
				"content": msg.Content,
			})
		}
	}
	
	userContent := []map[string]interface{}{}
	
	if message != "" {
		userContent = append(userContent, map[string]interface{}{
			"type": "text",
			"text": message,
		})
	}
	
	if imageURL != "" {
		userContent = append(userContent, map[string]interface{}{
			"type": "image_url",
			"image_url": map[string]interface{}{
				"url":    imageURL,
				"detail": "high",
			},
		})
	}
	
	userMessage := map[string]interface{}{
		"role":    "user",
		"content": userContent,
	}
	messages = append(messages, userMessage)
	
	payload := map[string]interface{}{
		"model":       "deepseek-ai/DeepSeek-V3.1:fireworks-ai",
		"messages":    messages,
		"max_tokens":  4000,
		"temperature": 0.7,
		"stream":      false,
	}
	
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal payload: %v", err)
	}
	
	req, err := http.NewRequest("POST", "https://router.huggingface.co/v1/chat/completions", 
		bytes.NewReader(payloadBytes))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}
	
	req.Header.Set("Authorization", "Bearer "+cfg.HuggingFaceToken)
	req.Header.Set("Content-Type", "application/json")
	
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("API request failed: %v", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API request failed with status: %s", resp.Status)
	}
	
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to parse response: %v", err)
	}
	
	choices, ok := result["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return "", fmt.Errorf("invalid response format: no choices")
	}
	
	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("invalid choice format")
	}
	
	messageObj, ok := choice["message"].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("invalid message format")
	}
	
	response, ok := messageObj["content"].(string)
	if !ok {
		return "", fmt.Errorf("invalid content format")
	}
	
	if chatID != "" && senderID != "" {
		addToConversationHistory(chatID, senderID, "user", message)
		addToConversationHistory(chatID, senderID, "assistant", response)
	}
	
	return response, nil
}

func callGroqLlama(message, chatID, senderID string, cfg *Config) (string, error) {
	messages := []map[string]interface{}{}
	
	if chatID != "" && senderID != "" {
		history := getConversationHistory(chatID, senderID, 10)
		for _, msg := range history {
			messages = append(messages, map[string]interface{}{
				"role":    msg.Role,
				"content": msg.Content,
			})
		}
	}
	
	if len(messages) == 0 {
		messages = append(messages, map[string]interface{}{
			"role": "system",
			"content": "Kamu adalah Jinxie AI, asisten pintar yang ahli dalam coding, debugging, dan analisis teknis. Berikan jawaban yang informatif dan jelas, sertakan tips dan contoh ketika perlu. Gunakan bahasa Indonesia untuk penjelasan umum, tetapi biarkan semua kode tetap dalam bahasa aslinya.",
		})
	}
	
	messages = append(messages, map[string]interface{}{
		"role":    "user",
		"content": message,
	})
	
	payload := map[string]interface{}{
		"model":       "llama-3.3-70b-versatile",
		"messages":    messages,
		"max_tokens":  4000,
		"temperature": 0.3,
		"top_p":       0.9,
	}
	
	return callGroqAPI(payload, cfg, chatID, senderID, message)
}

func callMoonAI(message, chatID, senderID string, cfg *Config) (string, error) {
	messages := []map[string]interface{}{}
	
	if chatID != "" && senderID != "" {
		history := getConversationHistory(chatID, senderID, 10)
		for _, msg := range history {
			messages = append(messages, map[string]interface{}{
				"role":    msg.Role,
				"content": msg.Content,
			})
		}
	}
	
	if len(messages) == 0 {
		messages = append(messages, map[string]interface{}{
			"role": "system",
			"content": "Kamu adalah Jinxie AI, asisten pintar yang ahli dalam memberikan informasi, analisis, dan penjelasan teknis. Gunakan bahasa Indonesia untuk semua penjelasan agar mudah dimengerti. Fokus pada jawaban yang jelas, ringkas, dan akurat, sertakan contoh atau tabel jika perlu. Jangan menulis kode pemrograman.",
		})
	}
	
	messages = append(messages, map[string]interface{}{
		"role":    "user",
		"content": message,
	})
	
	payload := map[string]interface{}{
		"model":       "moonshotai/kimi-k2-instruct-0905",
		"messages":    messages,
		"max_tokens":  4000,
		"temperature": 0.3,
		"top_p":       0.9,
	}
	
	return callGroqAPI(payload, cfg, chatID, senderID, message)
}

func callGroqAPI(payload map[string]interface{}, cfg *Config, chatID, senderID, userMessage string) (string, error) {
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal payload: %v", err)
	}
	
	req, err := http.NewRequest("POST", "https://api.groq.com/openai/v1/chat/completions", 
		bytes.NewReader(payloadBytes))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}
	
	req.Header.Set("Authorization", "Bearer "+cfg.GroqAPIKey)
	req.Header.Set("Content-Type", "application/json")
	
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("API request failed: %v", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API request failed with status: %s", resp.Status)
	}
	
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to parse response: %v", err)
	}
	
	choices, ok := result["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return "", fmt.Errorf("invalid response format: no choices")
	}
	
	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("invalid choice format")
	}
	
	messageObj, ok := choice["message"].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("invalid message format")
	}
	
	response, ok := messageObj["content"].(string)
	if !ok {
		return "", fmt.Errorf("invalid content format")
	}
	
	if chatID != "" && senderID != "" {
		addToConversationHistory(chatID, senderID, "user", userMessage)
		addToConversationHistory(chatID, senderID, "assistant", response)
	}
	
	return response, nil
}

// ==================== HTTP HANDLERS ====================
func authMiddleware(next http.HandlerFunc, cfg *Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" {
			sendErrorResponse(w, "missing_auth", "Authorization header required", http.StatusUnauthorized)
			return
		}
		
		parts := strings.Split(authHeader, " ")
		if len(parts) != 2 || parts[0] != "Bearer" {
			sendErrorResponse(w, "invalid_auth_format", "Invalid authorization format", http.StatusUnauthorized)
			return
		}
		
		apiKey := parts[1]
		valid := false
		for _, validKey := range cfg.APIKeys {
			if apiKey == validKey {
				valid = true
				break
			}
		}
		
		if !valid {
			sendErrorResponse(w, "invalid_api_key", "Invalid API key", http.StatusUnauthorized)
			return
		}
		
		next(w, r)
	}
}

func handleAIRequest(cfg *Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req AIRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			sendErrorResponse(w, "invalid_json", "Invalid JSON format", http.StatusBadRequest)
			return
		}

		if req.Message == "" {
			sendErrorResponse(w, "missing_message", "Message is required", http.StatusBadRequest)
			return
		}

		var response string
		var err error

		switch req.Model {
		case "deepseek":
			response, err = callDeepSeek(req.Message, req.ImageURL, req.ChatID, req.SenderID, cfg)
		case "groq-llama":
			response, err = callGroqLlama(req.Message, req.ChatID, req.SenderID, cfg)
		case "moon":
			response, err = callMoonAI(req.Message, req.ChatID, req.SenderID, cfg)
		default:
			sendErrorResponse(w, "unsupported_model", 
				fmt.Sprintf("Unsupported model: %s", req.Model), http.StatusBadRequest)
			return
		}

		if err != nil {
			sendErrorResponse(w, "api_error", err.Error(), http.StatusInternalServerError)
			return
		}

		sendSuccessResponse(w, req.Model, response)
	}
}

func handleClearHistory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	chatID := r.URL.Query().Get("chatId")
	senderID := r.URL.Query().Get("senderId")

	if chatID == "" || senderID == "" {
		sendErrorResponse(w, "missing_params", "chatId and senderId are required", http.StatusBadRequest)
		return
	}

	clearConversationHistory(chatID, senderID)
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"message": "Conversation history cleared",
	})
}

func sendSuccessResponse(w http.ResponseWriter, model, response string) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(AIResponse{
		Success:   true,
		Model:     model,
		Response:  response,
		Timestamp: time.Now().Format(time.RFC3339),
	})
}

func sendErrorResponse(w http.ResponseWriter, errorCode, errorMsg string, statusCode int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(AIResponse{
		Success:   false,
		Error:     fmt.Sprintf("%s: %s", errorCode, errorMsg),
		Timestamp: time.Now().Format(time.RFC3339),
	})
}

func healthCheck(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

// ==================== MAIN FUNCTION ====================
func main() {
	cfg := loadConfig()
	
	// Setup routes
	http.HandleFunc("/api/ai", authMiddleware(handleAIRequest(cfg), cfg))
	http.HandleFunc("/api/ai/clear", authMiddleware(handleClearHistory, cfg))
	http.HandleFunc("/health", healthCheck)
	
	fmt.Printf("Server starting on port %s\n", cfg.Port)
	if err := http.ListenAndServe(":"+cfg.Port, nil); err != nil {
		fmt.Printf("Error starting server: %v\n", err)
		os.Exit(1)
	}
}