# Open WebUI Chat Interface Guide

Open WebUI provides a ChatGPT-like interface for interacting with local Ollama models. Perfect for testing RAG responses and direct model interaction.

## Quick Access

| Component | URL | Setup |
|-----------|-----|-------|
| **Chat Interface** | `http://your-host:8085` | Create account on first visit |
| **Admin Panel** | Settings → Admin Panel | Admin account required |

## Why Open WebUI for RAG?

### Benefits
- **Familiar Interface**: ChatGPT-like experience
- **Multi-Model Support**: Switch between Ollama models easily
- **Conversation History**: Persistent chat sessions
- **Document Upload**: Test document-based queries
- **API Integration**: Can integrate with your RAG system

### RAG Use Cases
- **Manual Testing**: Test RAG responses interactively
- **Model Comparison**: Compare outputs from different models
- **Prompt Engineering**: Refine prompts for better results
- **User Interface**: Provide end-users with chat interface
- **Document Q&A**: Upload and query documents directly

## Initial Setup

### 1. First Visit Setup
```bash
# Access Open WebUI
open http://your-host:8085

# First user automatically becomes admin
# Create account:
# Name: Your Name
# Email: your-email@domain.com  
# Password: your-secure-password
```

### 2. Configure Ollama Connection
The interface should automatically connect to Ollama. If not:

1. **Check Settings** → **Connections** → **Ollama**
2. **Verify URL**: Should show `http://ollama:11434` (internal Docker networking)
3. **Test Connection**: Click "Test Connection"

### 3. Verify Available Models
```bash
# In chat interface, click model dropdown
# Should show available Ollama models:
# - llama3.2:3b
# - nomic-embed-text
# - Any other models you've pulled
```

## Basic Usage

### 1. Simple Chat
```
User: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence (AI) that focuses on developing algorithms and statistical models that enable computer systems to improve their performance on tasks through experience, without being explicitly programmed for each specific task...
```

### 2. Model Selection
- Click the model dropdown in the top bar
- Select from available models (llama3.2:3b, mistral:7b, etc.)
- Each conversation can use different models
- Model choice affects response quality, speed, and style

### 3. Chat Features
- **New Chat**: Start fresh conversations
- **Chat History**: Access previous conversations
- **Export Chat**: Download conversation history
- **Regenerate**: Get alternative responses
- **Edit Messages**: Modify previous messages

## RAG Integration

### 1. Document Upload
```bash
# Upload documents directly in chat
1. Click the paperclip icon
2. Select PDF, TXT, or other supported files
3. Documents are processed and embedded automatically
4. Ask questions about uploaded content
```

### 2. RAG Chat Example
```
User: [Uploads company_handbook.pdf]
User: What is our vacation policy?