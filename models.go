package main

// Struct untuk request
type AIRequest struct {
	Message    string        `json:"message"`
	Model      string        `json:"model"`
	ImageURL   string        `json:"image_url,omitempty"`
	ChatID     string        `json:"chat_id,omitempty"`
	SenderID   string        `json:"sender_id,omitempty"`
	History    []HistoryItem `json:"history,omitempty"`
}

// Struct untuk history item
type HistoryItem struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

// Struct untuk response
type AIResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
	Error   string `json:"error,omitempty"`
}
