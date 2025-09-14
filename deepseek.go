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
