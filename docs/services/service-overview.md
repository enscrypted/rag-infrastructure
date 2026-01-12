# RAG Infrastructure Services Overview

Complete guide to all 18 services in the RAG infrastructure stack, with quick access information and RAG-specific usage.

## 🗄️ Vector & Document Databases

### MongoDB (Primary Vector DB)
- **Access**: `http://your-host:27017` | `mongodb://your-host:27017/`
- **Purpose**: Native vector search, document storage
- **RAG Use**: Primary vector database, document retrieval, hybrid search
- **Key Features**: Vector search indexes, aggregation pipelines, geospatial
- **Getting Started**: 
  ```javascript
  // Create vector index
  db.documents.createSearchIndex({
    definition: { vectorSearchType: "knn", fields: [{ type: "vector", path: "embedding", numDimensions: 1536, similarity: "cosine" }] },
    name: "vector_index"
  })
  ```

### ChromaDB
- **Access**: `http://your-host:8000` | Token: `your-password`
- **Purpose**: Dedicated vector database with simple API
- **RAG Use**: Alternative vector store, metadata filtering, collections
- **Key Features**: Auto-embeddings, metadata search, persistence
- **Getting Started**:
  ```python
  import chromadb
  client = chromadb.HttpClient(host="your-host", port=8000)
  collection = client.create_collection("documents")
  ```

### Qdrant
- **Access**: `http://your-host:6333`
- **Purpose**: High-performance vector search
- **RAG Use**: Fast similarity search, payload filtering
- **Key Features**: HNSW algorithm, quantization, clustering
- **Getting Started**:
  ```python
  from qdrant_client import QdrantClient
  client = QdrantClient(host="your-host", port=6333)
  ```

### Weaviate
- **Access**: `http://your-host:8080`
- **Purpose**: GraphQL vector database
- **RAG Use**: Semantic search, auto-schema, multi-modal
- **Key Features**: GraphQL API, auto-vectorization, modules
- **Getting Started**:
  ```python
  import weaviate
  client = weaviate.Client("http://your-host:8080")
  ```

## 🕸️ Graph & Search

### Neo4j
- **Access**: `http://your-host:7474` | `bolt://your-host:7687`
- **Credentials**: neo4j / your-password
- **Purpose**: Graph database for relationships and knowledge graphs
- **RAG Use**: Entity relationships, graph RAG, knowledge traversal
- **Key Features**: Cypher queries, APOC procedures, Graph Data Science
- **Getting Started**:
  ```cypher
  CREATE (doc:Document {content: "text"})
  CREATE (entity:Entity {name: "concept"})
  CREATE (doc)-[:MENTIONS]->(entity)
  ```

### Elasticsearch
- **Access**: `http://your-host:9200`
- **Purpose**: Full-text search and analytics
- **RAG Use**: Keyword search, hybrid retrieval, text analysis
- **Key Features**: BM25 scoring, analyzers, aggregations
- **Getting Started**:
  ```bash
  curl -X POST "your-host:9200/documents/_doc" -H 'Content-Type: application/json' -d '{"content": "document text"}'
  ```

### Kibana
- **Access**: `http://your-host:5601`
- **Purpose**: Elasticsearch visualization and management
- **RAG Use**: Search analytics, document exploration, dashboards
- **Key Features**: Visualizations, dev tools, index management

## 🤖 LLM & AI Tools

### Ollama
- **Access**: `http://your-host:11434`
- **Purpose**: Local LLM serving (Llama, Mistral, etc.)
- **RAG Use**: Text generation, embeddings, response synthesis
- **Key Features**: Multiple models, streaming, embeddings API
- **Getting Started**:
  ```bash
  # Pull models
  docker exec rag-ollama ollama pull llama3.2:3b
  # Generate text
  curl -X POST http://your-host:11434/api/generate -d '{"model": "llama3.2:3b", "prompt": "Hello"}'
  ```

### Open WebUI
- **Access**: `http://your-host:8085`
- **Setup**: Create account on first visit
- **Purpose**: ChatGPT-like interface for Ollama models
- **RAG Use**: Manual testing, user interface, conversation history
- **Key Features**: Multi-model chat, document upload, API integration

## 📊 Observability

### Langfuse
- **Access**: `http://your-host:3000`
- **Setup**: Create account, get API keys from project settings
- **Purpose**: LLM observability and tracing
- **RAG Use**: Pipeline tracing, performance monitoring, quality scoring
- **Key Features**: Traces, spans, scores, cost tracking
- **Getting Started**:
  ```python
  from langfuse import Langfuse
  langfuse = Langfuse(public_key="pk-lf-...", secret_key="sk-lf-...", host="http://your-host:3000")
  trace = langfuse.trace(name="rag_query", input="question")
  ```

## 💾 Storage & Cache

### MinIO (S3 Storage)
- **Access**: `http://your-host:9001` (Console), `http://your-host:9000` (API)
- **Credentials**: admin / your-password
- **Purpose**: S3-compatible object storage
- **RAG Use**: Document storage, model artifacts, backups
- **Key Features**: S3 API, buckets, versioning
- **Getting Started**:
  ```python
  from minio import Minio
  client = Minio("your-host:9000", access_key="admin", secret_key="your-password", secure=False)
  ```

### Redis
- **Access**: `redis://your-host:6379`
- **Purpose**: In-memory caching and pub/sub
- **RAG Use**: Response caching, session storage, embeddings cache
- **Key Features**: Caching, pub/sub, data structures
- **Getting Started**:
  ```python
  import redis
  r = redis.Redis(host="your-host", port=6379, decode_responses=True)
  ```

### RedisInsight
- **Access**: `http://your-host:8001`
- **Purpose**: Redis management UI
- **RAG Use**: Monitor cache performance, debug Redis operations
- **Key Features**: Key browser, profiler, CLI

## 🛠️ Development Tools

### Jupyter Lab
- **Access**: `http://your-host:8888`
- **Token**: your-password
- **Purpose**: Interactive data science environment
- **RAG Use**: RAG development, data analysis, experimentation
- **Key Features**: Notebooks, kernels, extensions
- **Getting Started**: Upload notebook or create new Python notebook

### n8n
- **Access**: `http://your-host:5678`
- **Credentials**: admin / your-password
- **Purpose**: Workflow automation
- **RAG Use**: Document ingestion pipelines, API integrations
- **Key Features**: Visual workflows, triggers, connectors

## 🎨 MongoDB Management UIs

### Mongoku (Modern)
- **Access**: `http://your-host:3100`
- **Purpose**: Modern MongoDB management interface
- **RAG Use**: Document exploration, query building, schema analysis
- **Key Features**: Modern UI, query builder, aggregation pipeline builder

### Vector Search UI (Custom)
- **Access**: `http://your-host:8090`
- **Purpose**: Specialized MongoDB vector search interface
- **RAG Use**: Vector index management, similarity search testing
- **Key Features**: Vector search, index creation, document insertion

### Mongo Express (Traditional)
- **Access**: `http://your-host:8081`
- **Credentials**: admin / your-password
- **Purpose**: Basic MongoDB administration
- **RAG Use**: Simple document CRUD, database management
- **Key Features**: Document editing, index management, import/export

## 🚀 Quick Start Workflow

### 1. Set Up Your First RAG Pipeline

```python
# 1. Connect to MongoDB
from pymongo import MongoClient
client = MongoClient('mongodb://your-host:27017/')
collection = client.rag_db.documents

# 2. Create vector index
collection.create_search_index({
    "definition": {
        "vectorSearchType": "knn",
        "fields": [{
            "type": "vector",
            "path": "embedding", 
            "numDimensions": 768,
            "similarity": "cosine"
        }]
    },
    "name": "vector_index"
})

# 3. Generate embeddings with Ollama
import requests
def get_embedding(text):
    response = requests.post('http://your-host:11434/api/embeddings', json={
        "model": "nomic-embed-text", 
        "prompt": text
    })
    return response.json()["embedding"]

# 4. Add documents
docs = [
    {"content": "Python is great for AI", "embedding": get_embedding("Python is great for AI")},
    {"content": "RAG combines retrieval and generation", "embedding": get_embedding("RAG combines retrieval and generation")}
]
collection.insert_many(docs)

# 5. Search similar documents
query_vector = get_embedding("What is Python?")
pipeline = [
    {"$vectorSearch": {
        "index": "vector_index",
        "queryVector": query_vector,
        "path": "embedding",
        "limit": 3
    }}
]
results = list(collection.aggregate(pipeline))

# 6. Generate response with Ollama
context = "\n".join([doc["content"] for doc in results])
response = requests.post('http://your-host:11434/api/generate', json={
    "model": "llama3.2:3b",
    "prompt": f"Context: {context}\n\nQuestion: What is Python?\n\nAnswer:",
    "stream": false
})
print(response.json()["response"])
```

### 2. Monitor with Langfuse

```python
from langfuse import Langfuse

langfuse = Langfuse(
    public_key="your-public-key",
    secret_key="your-secret-key", 
    host="http://your-host:3000"
)

trace = langfuse.trace(name="rag_pipeline", input="What is Python?")
retrieval_span = trace.span(name="retrieval", output=f"Found {len(results)} docs")
generation_span = trace.span(name="generation", output=response.json()["response"])
trace.update(output=response.json()["response"])
```

## 🔧 Service Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Access Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  Open WebUI │ Vector UI │ Mongoku │ Jupyter │ n8n │ Langfuse   │
├─────────────────────────────────────────────────────────────────┤
│                     Application Layer                           │
├─────────────────────────────────────────────────────────────────┤
│     Your RAG Applications & Python Scripts                      │
├─────────────────────────────────────────────────────────────────┤
│                      Service Layer                              │
├─────────────────────────────────────────────────────────────────┤
│ MongoDB │ Neo4j │ Ollama │ ChromaDB │ Elasticsearch │ Redis     │
├─────────────────────────────────────────────────────────────────┤
│                    Infrastructure                               │
├─────────────────────────────────────────────────────────────────┤
│              Docker Containers & Networking                     │
└─────────────────────────────────────────────────────────────────┘
```

## 📚 Learn More

- **[MongoDB Guide](./mongodb.md)** - Complete vector search documentation
- **[Neo4j Guide](./neo4j.md)** - Graph database and relationships
- **[Ollama Guide](./ollama.md)** - Local LLM management
- **[Langfuse Guide](./langfuse.md)** - Observability and monitoring
- **[ChromaDB Guide](./chromadb.md)** - Vector database operations

## ⚡ Pro Tips

1. **Start Simple**: Begin with MongoDB + Ollama + Langfuse
2. **Use Vector UI**: Great for testing vector searches visually
3. **Monitor Everything**: Set up Langfuse tracing from day one
4. **Hybrid Search**: Combine vector similarity with metadata filtering
5. **Cache Responses**: Use Redis for frequently asked questions
6. **Backup Data**: Regular exports from MinIO and databases
7. **Scale Gradually**: Add more vector DBs as you need them