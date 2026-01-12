# RAG Infrastructure Documentation

Complete documentation for the RAG Infrastructure Stack - your one-stop resource for building production RAG applications.

## 📖 Documentation Structure

### 🚀 Quick Start
- **[Getting Started Tutorial](./tutorials/getting-started.md)** - Build your first RAG application step-by-step
- **[Service Overview](./services/service-overview.md)** - Quick reference for all 18 services

### 🛠️ Service Guides
Detailed documentation for each service in the stack:

#### Vector & Document Databases
- **[MongoDB Vector Search](./services/mongodb.md)** - Native vector search, aggregation pipelines, hybrid queries
- **[ChromaDB](./services/chromadb.md)** - Simple vector database with Python client
- **[Service Overview](./services/service-overview.md#vector--document-databases)** - Qdrant, Weaviate quick reference

#### Graph & Search  
- **[Neo4j Graph Database](./services/neo4j.md)** - Knowledge graphs, entity relationships, Cypher queries
- **[Service Overview](./services/service-overview.md#graph--search)** - Elasticsearch, Kibana quick reference

#### LLM & AI Tools
- **[Ollama Local LLMs](./services/ollama.md)** - Model management, embeddings, text generation
- **[Service Overview](./services/service-overview.md#llm--ai-tools)** - Open WebUI quick reference

#### Observability
- **[Langfuse](./services/langfuse.md)** - LLM tracing, monitoring, quality assessment

#### Storage & Development
- **[Service Overview](./services/service-overview.md#storage--cache)** - MinIO, Redis, Jupyter, n8n quick reference

### 💡 Concepts & Architecture
- **[RAG Architecture Guide](./concepts/rag-architecture.md)** - Understanding RAG patterns, implementation strategies
- **[Vector Embeddings](./concepts/vector-embeddings.md)** - Working with embeddings *(coming soon)*
- **[Graph RAG](./concepts/graph-rag.md)** - Combining knowledge graphs with RAG *(coming soon)*

### 🎯 Tutorials & Examples
- **[Getting Started](./tutorials/getting-started.md)** - Complete beginner tutorial with working code
- **[Basic RAG Example](../examples/basic-rag/)** - Simple document Q&A system
- **[Advanced Patterns](./tutorials/advanced-rag.md)** - Multi-modal, hybrid search *(coming soon)*

## 🔗 Quick Navigation

### By Use Case

**🆕 New to RAG?**
1. Read [RAG Architecture Guide](./concepts/rag-architecture.md)
2. Follow [Getting Started Tutorial](./tutorials/getting-started.md)  
3. Try [Basic RAG Example](../examples/basic-rag/)

**🏗️ Building Production RAG?**
1. Study [MongoDB Vector Search](./services/mongodb.md)
2. Set up [Langfuse Monitoring](./services/langfuse.md)
3. Implement [Neo4j Graph RAG](./services/neo4j.md)

**🔧 Need Service Reference?**
1. Check [Service Overview](./services/service-overview.md) 
2. Find specific service documentation
3. Review connection examples and troubleshooting

### By Service Type

| Service Type | Quick Access | Detailed Guides |
|-------------|--------------|----------------|
| **Vector Databases** | [Overview](./services/service-overview.md#vector--document-databases) | [MongoDB](./services/mongodb.md), [ChromaDB](./services/chromadb.md) |
| **Graph Database** | [Overview](./services/service-overview.md#graph--search) | [Neo4j](./services/neo4j.md) |
| **Local LLMs** | [Overview](./services/service-overview.md#llm--ai-tools) | [Ollama](./services/ollama.md) |
| **Observability** | [Overview](./services/service-overview.md#observability) | [Langfuse](./services/langfuse.md) |
| **Storage & Dev Tools** | [Overview](./services/service-overview.md#storage--cache) | Service Overview |

## 💻 Code Examples

### Quick RAG Pipeline
```python
# Complete RAG example using the stack
from pymongo import MongoClient
import requests

# 1. Connect to services
mongo = MongoClient('mongodb://your-host:27017/')
collection = mongo.rag_db.documents

# 2. Generate embedding
def get_embedding(text):
    response = requests.post('http://your-host:11434/api/embeddings', 
                           json={"model": "nomic-embed-text", "prompt": text})
    return response.json()["embedding"]

# 3. Search similar documents  
query_vector = get_embedding("What is machine learning?")
results = collection.aggregate([{
    "$vectorSearch": {
        "index": "vector_index",
        "queryVector": query_vector, 
        "path": "embedding",
        "limit": 5
    }
}])

# 4. Generate response
context = "\n".join([doc["content"] for doc in results])
response = requests.post('http://your-host:11434/api/generate', json={
    "model": "llama3.2:3b",
    "prompt": f"Context: {context}\n\nQuestion: What is machine learning?\n\nAnswer:",
    "stream": False
})

print(response.json()["response"])
```

### Service Connection Examples
```python
# MongoDB Vector Search
from pymongo import MongoClient
client = MongoClient('mongodb://your-host:27017/')

# Neo4j Graph Database  
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://your-host:7687', auth=('neo4j', 'password'))

# ChromaDB Vector Store
import chromadb
chroma = chromadb.HttpClient(host="your-host", port=8000)

# Langfuse Observability
from langfuse import Langfuse
langfuse = Langfuse(host="http://your-host:3000", public_key="pk-...", secret_key="sk-...")
```

## 🎯 Common Tasks

### Setting Up Vector Search
1. **[Create MongoDB Vector Index](./services/mongodb.md#vector-search-setup)**
2. **[Generate Embeddings with Ollama](./services/ollama.md#embeddings-generation)**
3. **[Perform Similarity Search](./services/mongodb.md#performing-vector-search)**

### Monitoring RAG Performance  
1. **[Set Up Langfuse Tracing](./services/langfuse.md#rag-tracing-implementation)**
2. **[Add Quality Scores](./services/langfuse.md#advanced-tracing-with-scores)**
3. **[Monitor Performance](./services/langfuse.md#production-monitoring)**

### Building Knowledge Graphs
1. **[Create Graph Schema](./services/neo4j.md#initial-setup)**
2. **[Extract Entities](./services/neo4j.md#entity-extraction-and-storage)**
3. **[Graph-Enhanced Retrieval](./services/neo4j.md#graph-enhanced-retrieval)**

## 🔍 Troubleshooting

### Quick Diagnostics

**Services Not Starting?**
```bash
# Check all service status
docker compose ps

# Check specific service logs
docker compose logs servicename

# Test connectivity
curl http://your-host:8090/api/collections  # Vector UI
curl http://your-host:11434/api/tags        # Ollama
curl http://your-host:3000/api/public/health # Langfuse
```

**RAG Pipeline Issues?**
- **No search results**: Check vector index creation
- **Poor quality responses**: Review retrieval relevance
- **Slow performance**: Monitor with Langfuse, check model sizes
- **Memory errors**: Reduce batch sizes, use smaller models

**Common Solutions**:
- [MongoDB Troubleshooting](./services/mongodb.md#troubleshooting)
- [Ollama Troubleshooting](./services/ollama.md#troubleshooting)  
- [Langfuse Troubleshooting](./services/langfuse.md#troubleshooting)

## 📚 Learning Path

### Beginner (Week 1)
1. **Understand RAG**: Read [RAG Architecture Guide](./concepts/rag-architecture.md)
2. **Try the Tutorial**: Complete [Getting Started](./tutorials/getting-started.md)
3. **Explore Services**: Use Vector UI, Open WebUI, and Langfuse dashboards

### Intermediate (Week 2-3)  
1. **Deep Dive Services**: Study [MongoDB](./services/mongodb.md) and [Ollama](./services/ollama.md) guides
2. **Build Custom RAG**: Extend the [Basic RAG Example](../examples/basic-rag/)
3. **Add Monitoring**: Implement [Langfuse Tracing](./services/langfuse.md)

### Advanced (Week 4+)
1. **Graph RAG**: Learn [Neo4j integration](./services/neo4j.md) 
2. **Multi-Modal**: Explore image and document processing
3. **Production**: Implement caching, scaling, security

## 🤝 Contributing

Found an issue or want to improve the documentation?

- **Quick Fixes**: Edit the files directly and submit a PR
- **New Guides**: Follow the existing structure and style
- **Examples**: Add your RAG implementations to `examples/`

See **[Contributing Guide](../CONTRIBUTING.md)** for detailed instructions.

## 📞 Getting Help

- **Documentation Issues**: Check troubleshooting sections first
- **Service Problems**: Review individual service guides  
- **General Questions**: Open a GitHub discussion
- **Bug Reports**: Create a detailed issue

## 🔗 External Resources

### Official Documentation
- [MongoDB Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Langfuse Documentation](https://langfuse.com/docs)

### Community Resources
- [RAG Papers and Research](https://arxiv.org/abs/2005.11401)
- [Vector Database Comparison](https://www.pinecone.io/learn/vector-database/)
- [LLM Evaluation Frameworks](https://docs.ragas.io/)
- [Embedding Best Practices](https://platform.openai.com/docs/guides/embeddings)

---

**Ready to build amazing RAG applications? Start with the [Getting Started Tutorial](./tutorials/getting-started.md)!** 🚀