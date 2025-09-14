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
	
	return callGroqAPI(payload)
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
	
	return callGroqAPI(payload)
}

func callGroqAPI(payload map[string]interface{}) (string, error) {
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
