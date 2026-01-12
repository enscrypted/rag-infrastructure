# Basic RAG Example

This example demonstrates a simple RAG (Retrieval-Augmented Generation) implementation using the RAG Infrastructure Stack.

## Features

- Document ingestion with automatic embedding generation
- Vector similarity search using MongoDB
- Response generation using local Ollama models
- Interactive Q&A interface
- Simple error handling and logging

## Prerequisites

1. RAG Infrastructure Stack deployed and running
2. Python 3.8+
3. Required Python packages

## Setup

1. **Install dependencies:**
   ```bash
   pip install pymongo requests
   ```

2. **Set environment variables (optional):**
   ```bash
   export MONGO_URL="mongodb://your-host:27017/"
   export OLLAMA_BASE_URL="http://your-host:11434"
   ```

3. **Ensure required models are available:**
   ```bash
   # Check if models are installed
   curl http://your-host:11434/api/tags
   
   # Install if needed
   docker exec rag-ollama ollama pull nomic-embed-text
   docker exec rag-ollama ollama pull llama3.2:3b
   ```

## Usage

### Basic Usage

```bash
python simple_rag.py
```

The script will:
1. Connect to MongoDB and Ollama
2. Create vector search index if needed
3. Load sample documents (if collection is empty)
4. Start interactive Q&A session

### Interactive Commands

- Ask any question about the loaded documents
- `stats` - View collection statistics
- `clear` - Clear all documents
- `reload` - Clear and reload sample data
- `quit` - Exit the application

### Example Session

```
🚀 Simple RAG Demo Starting...
MongoDB: mongodb://localhost:27017/
Ollama: http://localhost:11434
📊 Connected to MongoDB: rag_demo.documents
🤖 Connected to Ollama: http://localhost:11434
✅ Vector index created

📚 Loading sample documents...
✅ Document added: 507f1f77bcf86cd799439011
✅ Document added: 507f1f77bcf86cd799439012
...
✅ Loaded 5 sample documents

============================================================
🎯 RAG DEMO - Interactive Mode
============================================================
Ask questions about the loaded documents!
Commands: 'stats', 'clear', 'reload', 'quit'
============================================================

❓ Your question: What is Python used for?

❓ Question: What is Python used for?
🔍 Found 1 similar documents

🤖 Answer: Based on the provided context, Python is used for:

1. **Rapid Application Development** - Due to its high-level built-in data structures and dynamic features
2. **Scripting** - As a scripting language to automate tasks
3. **Glue language** - To connect and integrate existing software components together

Python's dynamic typing, dynamic binding, and interpreted nature make it particularly attractive for these applications.

📚 Sources (1):
  1. Relevance: 0.856 | Topic: programming
```

## Code Structure

```python
class SimpleRAG:
    def __init__(self, mongo_url, ollama_url, database, collection)
        # Initialize connections and setup vector index
    
    def add_document(self, content, metadata)
        # Add document with automatic embedding generation
    
    def search_similar(self, query, limit=5)
        # Vector similarity search
    
    def query(self, question, num_docs=3)
        # Complete RAG pipeline: retrieve + generate
    
    def get_stats(self)
        # Collection statistics
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGO_URL` | `mongodb://localhost:27017/` | MongoDB connection string |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |

### Models Used

- **Embedding**: `nomic-embed-text` (768 dimensions)
- **Generation**: `llama3.2:3b`

## Sample Data

The example includes 5 sample documents covering:
- Python programming
- Machine learning
- Docker containers
- Vector databases
- RAG architecture

## Customization

### Adding Your Own Documents

```python
rag = SimpleRAG()

# Add individual document
rag.add_document(
    content="Your document content here",
    metadata={"source": "my_doc", "category": "examples"}
)

# Bulk load from files
import os
docs_dir = "path/to/your/documents"
for filename in os.listdir(docs_dir):
    with open(os.path.join(docs_dir, filename), 'r') as f:
        content = f.read()
        rag.add_document(content, {"source": filename})
```

### Changing Models

```python
# In _generate_embedding method
"model": "your-preferred-embedding-model"

# In _generate_response method  
"model": "your-preferred-llm-model"
```

### Search Parameters

```python
# Adjust retrieval parameters
similar_docs = self.search_similar(
    question, 
    limit=5  # Number of documents to retrieve
)

# Modify vector search pipeline
"numCandidates": limit * 20  # Increase for better quality
```

## Troubleshooting

### Common Issues

**"Connection refused"**
- Check if MongoDB and Ollama services are running
- Verify the connection URLs are correct
- Test connectivity: `curl http://your-host:11434/api/tags`

**"Vector index not found"**
- Wait a few minutes for index creation
- Check MongoDB logs for errors
- Try recreating the collection

**"No embedding returned"**
- Ensure `nomic-embed-text` model is installed
- Check Ollama logs: `docker compose logs ollama`
- Verify model list: `docker exec rag-ollama ollama list`

**"Empty search results"**
- Check if documents have embeddings
- Verify vector index exists: `collection.list_search_indexes()`
- Try different search terms

### Performance Tips

1. **Batch document insertion** for large datasets
2. **Adjust `numCandidates`** based on collection size
3. **Use metadata filtering** for focused search
4. **Monitor memory usage** with large embedding collections

## Next Steps

1. **Enhance retrieval**: Add hybrid search (vector + keyword)
2. **Improve generation**: Use better prompts or models
3. **Add evaluation**: Implement quality metrics
4. **Web interface**: Create REST API or web UI
5. **Production features**: Add authentication, caching, monitoring

## Related Examples

- [Multi-modal RAG](../multimodal-rag/) - Handle images and text
- [Graph RAG](../graph-rag/) - Use Neo4j for enhanced retrieval
- [Evaluation](../evaluation/) - Measure RAG performance