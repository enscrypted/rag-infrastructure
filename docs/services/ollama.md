# Ollama Local LLM Guide

Ollama enables you to run large language models locally. Perfect for RAG applications requiring privacy, low latency, or offline capabilities.

## Quick Access

| Component | URL | Purpose |
|-----------|-----|---------|
| **Ollama API** | `http://your-host:11434` | REST API for model interaction |
| **Model Management** | `docker exec rag-ollama ollama` | CLI commands |

## Why Ollama for RAG?

### Benefits
- **Privacy**: Models run locally, data never leaves your infrastructure
- **Cost Control**: No per-token charges, unlimited usage
- **Customization**: Fine-tune models for your specific domain
- **Offline Capability**: Works without internet connection
- **Low Latency**: Direct access without API round-trips

### RAG Use Cases
- **Text Generation**: Answer questions based on retrieved context
- **Embeddings**: Generate vector representations for similarity search
- **Summarization**: Condense retrieved documents
- **Classification**: Categorize documents and queries

## Initial Setup

### 1. Check Available Models
```bash
# List installed models
docker exec rag-ollama ollama list

# Should show something like:
# NAME                ID              SIZE    MODIFIED
# llama3.2:3b         a80c4f17acd5    2.0 GB  2 weeks ago
# nomic-embed-text    0a109f422b47    274 MB  2 weeks ago
```

### 2. Install Additional Models
```bash
# Popular models for RAG
docker exec rag-ollama ollama pull llama3.2:7b      # Better quality, slower
docker exec rag-ollama ollama pull mistral:7b       # Good balance
docker exec rag-ollama ollama pull codellama:7b     # For code generation
docker exec rag-ollama ollama pull llava:7b         # For multimodal RAG

# Specialized embedding models
docker exec rag-ollama ollama pull nomic-embed-text  # General purpose
docker exec rag-ollama ollama pull all-minilm       # Lightweight
```

### 3. Test Installation
```bash
# Test text generation
curl http://your-host:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "What is RAG?",
  "stream": false
}'

# Test embeddings
curl http://your-host:11434/api/embeddings -d '{
  "model": "nomic-embed-text", 
  "prompt": "Hello world"
}'
```

## API Usage for RAG

### 1. Text Generation

```python
import requests
import json

class OllamaRAG:
    def __init__(self, base_url="http://your-host:11434"):
        self.base_url = base_url
    
    def generate_response(self, prompt, model="llama3.2:3b", **kwargs):
        """Generate text response"""
        response = requests.post(f"{self.base_url}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        })
        return response.json()["response"]
    
    def rag_generate(self, question, context, model="llama3.2:3b"):
        """Generate RAG response with context"""
        prompt = f"""Based on the following context, answer the question accurately.
        If the context doesn't contain the answer, say so.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        return self.generate_response(prompt, model)

# Usage
ollama = OllamaRAG()
context = "Python is a programming language used for AI development."
answer = ollama.rag_generate("What is Python?", context)
print(answer)
```

### 2. Embeddings Generation

```python
def generate_embeddings(self, texts, model="nomic-embed-text"):
    """Generate embeddings for texts

    Note: Ollama has two embedding endpoints:
    - /api/embed (newer, recommended for 2025+)
    - /api/embeddings (legacy, still works)

    Both return the same results, /api/embed supports batch input.
    """
    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    for text in texts:
        # Using /api/embeddings (legacy but widely compatible)
        response = requests.post(f"{self.base_url}/api/embeddings", json={
            "model": model,
            "prompt": text
        })
        embeddings.append(response.json()["embedding"])

    return embeddings

def generate_embeddings_batch(self, texts, model="nomic-embed-text"):
    """Generate embeddings using newer /api/embed endpoint (2025+)

    Supports batch input for better performance.
    """
    if isinstance(texts, str):
        texts = [texts]

    # /api/embed supports 'input' as list for batch processing
    response = requests.post(f"{self.base_url}/api/embed", json={
        "model": model,
        "input": texts
    })
    return response.json()["embeddings"]

# Generate embeddings for documents
docs = ["Machine learning is a subset of AI", "Python is great for data science"]
embeddings = ollama.generate_embeddings(docs)
print(f"Generated {len(embeddings)} embeddings, each with {len(embeddings[0])} dimensions")
```

### 3. Streaming Responses

```python
def stream_response(self, prompt, model="llama3.2:3b"):
    """Stream response for real-time output"""
    response = requests.post(
        f"{self.base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": True},
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            if not chunk.get("done", False):
                yield chunk["response"]
            else:
                break

# Usage - stream response in real-time
for token in ollama.stream_response("Explain machine learning"):
    print(token, end="", flush=True)
```

## Model Selection Guide

### Text Generation Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **llama3.2:3b** | 2.0GB | Fast | Good | Quick responses, development |
| **llama3.2:7b** | 4.1GB | Medium | Better | Production RAG, complex queries |
| **llama3.2:70b** | 40GB+ | Slow | Best | High-quality responses, analysis |
| **mistral:7b** | 4.1GB | Medium | Good | Balanced performance |
| **codellama:7b** | 3.8GB | Medium | Code | Code generation and analysis |

### Embedding Models

| Model | Dimensions | Best For |
|-------|------------|----------|
| **nomic-embed-text** | 768 | General purpose, good performance |
| **all-minilm** | 384 | Lightweight, fast processing |
| **e5-large** | 1024 | High quality semantic search |

## Advanced RAG Patterns

### 1. Multi-Model RAG

```python
class MultiModelRAG:
    def __init__(self):
        self.embedding_model = "nomic-embed-text"
        self.generation_model = "llama3.2:7b"
        self.code_model = "codellama:7b"
    
    def route_query(self, query):
        """Route query to appropriate model"""
        if any(keyword in query.lower() for keyword in ['code', 'python', 'function', 'script']):
            return self.code_model
        else:
            return self.generation_model
    
    def smart_generate(self, question, context):
        """Use appropriate model based on query type"""
        model = self.route_query(question)
        return self.rag_generate(question, context, model)
```

### 2. Prompt Engineering for RAG

```python
def create_rag_prompt(self, question, context, task_type="qa"):
    """Create optimized prompts for different RAG tasks"""
    
    prompts = {
        "qa": f"""You are a helpful assistant. Answer the question based on the provided context.
        
Context: {context}

Question: {question}

Provide a clear, accurate answer. If the context doesn't contain enough information, say so.

Answer:""",
        
        "summarize": f"""Summarize the following text concisely, focusing on key points:

Text: {context}

Summary:""",
        
        "analyze": f"""Analyze the following context and answer the analytical question:

Context: {context}

Question: {question}

Provide a detailed analysis with reasoning:

Analysis:"""
    }
    
    return prompts.get(task_type, prompts["qa"])
```

### 3. Model Fine-tuning Preparation

```python
def prepare_training_data(self, qa_pairs):
    """Prepare data for potential model fine-tuning"""
    training_data = []
    
    for question, context, answer in qa_pairs:
        prompt = self.create_rag_prompt(question, context)
        training_data.append({
            "prompt": prompt,
            "completion": answer
        })
    
    return training_data
```

## Performance Optimization

### 1. Model Management

```bash
# Check model resource usage
docker stats rag-ollama

# Remove unused models to save space
docker exec rag-ollama ollama rm unused-model:tag

# Pull specific model variants
docker exec rag-ollama ollama pull llama3.2:3b-q4_0  # Quantized version
docker exec rag-ollama ollama pull llama3.2:3b-q8_0  # Higher quality quantization
```

### 2. Connection Pooling

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class OptimizedOllama:
    def __init__(self, base_url="http://your-host:11434"):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        
        # Configure connection pooling
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            retry=retry_strategy
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def generate(self, prompt, model="llama3.2:3b"):
        """Optimized generation with connection reuse"""
        return self.session.post(f"{self.base_url}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }).json()["response"]
```

### 3. Batch Processing

```python
def batch_embeddings(self, texts, model="nomic-embed-text", batch_size=10):
    """Process embeddings in batches for efficiency"""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = []
        
        for text in batch:
            response = requests.post(f"{self.base_url}/api/embeddings", json={
                "model": model,
                "prompt": text
            })
            batch_embeddings.append(response.json()["embedding"])
        
        embeddings.extend(batch_embeddings)
        
        # Optional: Add delay to prevent overload
        time.sleep(0.1)
    
    return embeddings
```

## Model Configuration

### 1. Custom Model Parameters

```python
def generate_with_config(self, prompt, model="llama3.2:3b", **config):
    """Generate with custom parameters"""
    default_config = {
        "temperature": 0.7,      # Randomness (0.0-2.0)
        "top_p": 0.9,           # Nucleus sampling
        "top_k": 40,            # Top-k sampling
        "repeat_penalty": 1.1,   # Repetition penalty
        "num_ctx": 2048,        # Context window
    }
    
    merged_config = {**default_config, **config}
    
    return requests.post(f"{self.base_url}/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": merged_config
    }).json()["response"]

# Usage for different scenarios
# Conservative/factual responses
factual_response = ollama.generate_with_config(
    prompt, temperature=0.1, top_p=0.8
)

# Creative/diverse responses  
creative_response = ollama.generate_with_config(
    prompt, temperature=1.2, top_p=0.95
)
```

### 2. Model Information

```python
def get_model_info(self, model_name):
    """Get detailed model information"""
    response = requests.post(f"{self.base_url}/api/show", json={
        "name": model_name
    })
    
    info = response.json()
    return {
        "parameters": info.get("parameters", {}),
        "template": info.get("template", ""),
        "details": info.get("details", {}),
        "model_size": info.get("size", 0)
    }
```

## Monitoring and Debugging

### 1. Health Checks

```python
def health_check(self):
    """Check Ollama service health"""
    try:
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        return {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "models": len(response.json().get("models", [])),
            "response_time": response.elapsed.total_seconds()
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### 2. Performance Monitoring

```python
import time
from functools import wraps

def monitor_performance(func):
    """Decorator to monitor model performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            print(f"{func.__name__}: {duration:.2f}s")
            return result
        except Exception as e:
            print(f"{func.__name__} failed after {time.time() - start_time:.2f}s: {e}")
            raise
    return wrapper

@monitor_performance
def monitored_generate(prompt, model="llama3.2:3b"):
    return ollama.generate_response(prompt, model)
```

## Troubleshooting

### Common Issues

**"Model not found"**
```bash
# Check available models
docker exec rag-ollama ollama list

# Pull missing model
docker exec rag-ollama ollama pull llama3.2:3b
```

**"Connection refused"**
```bash
# Check Ollama service status
docker compose ps ollama

# Check logs
docker compose logs ollama

# Restart if needed
docker compose restart ollama
```

**"Out of memory"**
```bash
# Check available memory
docker stats rag-ollama

# Use smaller model or increase Docker memory limit
docker exec rag-ollama ollama pull llama3.2:1b  # Smaller model
```

**Slow responses**
```bash
# Check system resources
htop

# Use quantized models for better performance
docker exec rag-ollama ollama pull llama3.2:3b-q4_0
```

## Best Practices

### Performance
1. **Choose appropriate model size** for your hardware
2. **Use quantized models** (q4_0, q8_0) for better performance
3. **Keep frequently used models** loaded
4. **Monitor resource usage** and scale accordingly

### Security
1. **Restrict API access** to trusted networks
2. **Validate all inputs** to prevent prompt injection
3. **Monitor model usage** for abuse detection
4. **Regular updates** of Ollama and models

### Cost Optimization
1. **Remove unused models** to save storage
2. **Use appropriate context windows** for your use case
3. **Batch similar requests** for efficiency
4. **Cache frequent responses** when possible

## Learning Resources

- [Ollama Documentation](https://ollama.ai/docs) - Official documentation
- [Model Library](https://ollama.ai/library) - Available models and variants
- [Ollama GitHub](https://github.com/ollama/ollama) - Source code and issues
- [Model Cards](https://huggingface.co/models) - Detailed model information