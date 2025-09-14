package main

import "encoding/json"

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
