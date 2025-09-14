package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

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
	
	return callOpenRouterAPI(payload)
}

func callSonomaAI(req AIRequest) (string, error) {
	payload := map[string]interface{}{
		"model":       "openrouter/sonoma-sky-alpha",
		"messages":    prepareMessages(req),
		"max_tokens":  5000,
		"temperature": 0.7,
	}
	
	return callOpenRouterAPI(payload)
}

func callOpenRouterAPI(payload map[string]interface{}) (string, error) {
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
		body, _ := io.ReadAll(resp.Body)
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
