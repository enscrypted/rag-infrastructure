# 🚀 RAG Infrastructure Stack

**A complete, production-ready AI/ML development environment for Retrieval-Augmented Generation (RAG) experimentation and deployment.**

[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MongoDB](https://img.shields.io/badge/MongoDB-8.2-green.svg)](https://mongodb.com)
[![Neo4j](https://img.shields.io/badge/Neo4j-5-red.svg)](https://neo4j.com)

## 🎯 What This Provides

This repository contains everything needed to deploy a comprehensive RAG/AI development stack on any server, homelab, or cloud instance. Perfect for:

- **AI/ML Research & Development**
- **RAG Application Prototyping**
- **Local LLM Experimentation**
- **Vector Database Testing**
- **Graph-based AI Applications**

## ✨ Features

### 🗄️ **Vector & Document Databases**
- **MongoDB 8.2** - Native vector search capabilities (no Atlas required)
- **ChromaDB** - Popular vector database for embeddings
- **Qdrant** - High-performance vector similarity search
- **Weaviate** - GraphQL-based vector database

### 🕸️ **Graph & Search**
- **Neo4j 5** - Property graph database with GDS plugins
- **Elasticsearch** - Full-text search and analytics

### 🤖 **LLM & AI Tools**
- **Ollama** - Local LLM serving (Llama, Mistral, etc.)
- **Open WebUI** - ChatGPT-like interface for local models
- **Langfuse** - LLM observability and tracing (open-source alternative to LangSmith)

### 💾 **Storage & Caching**
- **MinIO** - S3-compatible object storage
- **Redis** - In-memory caching and pub/sub

### 🛠️ **Development Tools**
- **Jupyter Lab** - Data science notebooks
- **n8n** - Workflow automation and AI pipelines
- **Custom Vector Search UI** - Modern interface for MongoDB vector operations

### 🎨 **Multiple MongoDB UIs**
- **Vector Search UI** - Custom-built for RAG workflows
- **Mongoku** - Modern, HuggingFace-maintained interface
- **Mongo Express** - Traditional admin interface

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vector DBs    │    │   Graph & Search │    │   LLM Tools     │
│                 │    │                 │    │                 │
│ • MongoDB       │    │ • Neo4j         │    │ • Ollama        │
│ • ChromaDB      │    │ • Elasticsearch │    │ • Open WebUI    │
│ • Qdrant        │    │                 │    │ • Langfuse      │
│ • Weaviate      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
         │   Storage       │    │   Docker Host   │    │   Dev Tools     │
         │                 │    │                 │    │                 │
         │ • MinIO         │────│ • 18 Services  │────│ • Jupyter       │
         │ • Redis         │    │ • Docker Net    │    │ • n8n           │
         │                 │    │ • Volumes       │    │ • Multiple UIs  │
         └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+, Debian 11+, CentOS 8+) or macOS
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: Minimum 50GB free space, Recommended 100GB+
- **CPU**: 4+ cores recommended for full stack

### Software Requirements
- [Docker](https://docs.docker.com/get-docker/) 20.10+
- [Docker Compose](https://docs.docker.com/compose/install/) 2.0+
- `bash` shell
- `curl` and basic command-line tools

### Network Requirements
- Open ports: 3000, 3100, 5601, 6333, 6379, 7474, 7687, 8000, 8001, 8080-8090, 8888, 9000-9001, 11434
- Internet access for Docker image pulls
- (Optional) DNS server access for hostname configuration

## 🚀 Quick Start

### Option A: Clone from GitHub
```bash
git clone https://github.com/enscrypted/rag-infrastructure.git
cd rag-infrastructure
```

### Option B: Copy to Server via SCP
```bash
# From your local machine, copy to your server
scp -r /path/to/rag-infrastructure root@YOUR_SERVER_IP:/tmp/

# SSH into your server
ssh root@YOUR_SERVER_IP
cd /tmp/rag-infrastructure
```

### Run the Interactive Deployment
```bash
chmod +x scripts/deploy.sh
sudo scripts/deploy.sh
```

The script will prompt you for:
- **Host IP Address** (e.g., 192.168.1.100)
- **DNS/Proxy Preference** (hostnames vs direct IP access)
- **Service Profile** (essential vs full stack)
- **Default Password** for all services

### 3. Access Your Services

After deployment, check `CREDENTIALS.md` for all access URLs and credentials.

**Quick Access Examples:**
- Vector Search UI: `http://your-ip:8090`
- Chat Interface: `http://your-ip:8085`
- Neo4j Browser: `http://your-ip:7474`
- Jupyter Lab: `http://your-ip:8888`

## 🔧 Configuration Options

### Service Profiles

**Essential Stack** (5 core services):
- MongoDB + Vector Search UI
- Neo4j
- Langfuse
- Ollama + Open WebUI

**Full Stack** (18 services):
- All essential services plus:
- ChromaDB, Qdrant, Weaviate
- Elasticsearch + Kibana
- MinIO + Redis + RedisInsight
- Jupyter Lab + n8n
- Additional MongoDB UIs

### DNS & Hostname Setup

The deployment supports three access methods:

1. **Direct IP Access** - Services accessible via IP:PORT
2. **Manual DNS Setup** - Configure your own Pi-hole/DNS server
3. **Automated Setup** - Uses included script (may need adjustments)

## 📚 Quick Example

```python
# Complete RAG pipeline using the stack
from pymongo import MongoClient
import requests

# Connect to services
mongo = MongoClient('mongodb://your-ip:27017/')
collection = mongo.rag_db.documents

# Generate embedding
response = requests.post('http://your-ip:11434/api/embeddings', 
                        json={"model": "nomic-embed-text", "prompt": "What is RAG?"})
query_vector = response.json()["embedding"]

# Search similar documents
results = collection.aggregate([{
    "$vectorSearch": {
        "index": "vector_index",
        "queryVector": query_vector,
        "path": "embedding",
        "limit": 5
    }
}])

# Generate response with context
context = "\n".join([doc["content"] for doc in results])
response = requests.post('http://your-ip:11434/api/generate', json={
    "model": "llama3.2:3b",
    "prompt": f"Context: {context}\n\nQuestion: What is RAG?\n\nAnswer:"
})

print(response.json()["response"])
```

> **For detailed code examples and integration patterns, see the [service documentation](docs/services/) and [tutorials](docs/tutorials/).**

## 📖 Documentation

### 📋 Service Documentation

**Overview & Quick Reference:**
- **[Service Overview](docs/services/service-overview.md)** - Quick reference for all 18 services with access URLs and basic usage

**Core RAG Services:**
- **[MongoDB Vector Search](docs/services/mongodb.md)** - Native vector search, aggregations, and hybrid RAG patterns  
- **[Neo4j Graph Database](docs/services/neo4j.md)** - Graph modeling, Cypher queries, and Graph RAG implementation
- **[Ollama Local LLMs](docs/services/ollama.md)** - Model management, API usage, and embedding generation
- **[Langfuse Observability](docs/services/langfuse.md)** - Complete tracing, monitoring, and quality assessment

**Vector Databases:**
- **[ChromaDB](docs/services/chromadb.md)** - Simple vector database with Python client
- **[Qdrant](docs/services/qdrant.md)** - High-performance vector search with web UI
- **[Weaviate](docs/services/weaviate.md)** - GraphQL vector database with auto-vectorization

**Search & Analytics:**
- **[Elasticsearch & Kibana](docs/services/elasticsearch.md)** - Full-text search, analytics, and visualization

**Storage & Caching:**
- **[MinIO Object Storage](docs/services/minio.md)** - S3-compatible storage for documents and models
- **[Redis & RedisInsight](docs/services/redis.md)** - High-performance caching and session management

**Development Tools:**
- **[Jupyter Lab](docs/services/jupyter.md)** - Interactive RAG development and data analysis
- **[Open WebUI](docs/services/open-webui.md)** - ChatGPT-like interface for local models
- **[n8n Workflow Automation](docs/services/n8n.md)** - Visual workflow designer for RAG pipelines

### Getting Started
- **[RAG Architecture Guide](docs/concepts/rag-architecture.md)** - Understanding retrieval-augmented generation
- **[Getting Started Tutorial](docs/tutorials/getting-started.md)** - Build your first RAG application
- **[Basic RAG Example](examples/basic-rag/)** - Simple document Q&A system

## 🛠️ Management

### Service Management
```bash
# Check service status
docker compose ps

# View logs
docker compose logs -f mongodb

# Restart a service
docker compose restart neo4j

# Update all images
docker compose pull && docker compose up -d

# Scale services
docker compose up -d --scale ollama=2
```

### Backup & Restore
```bash
# Backup MongoDB
docker exec rag-mongodb mongodump --archive=/data/backup.gz --gzip

# Backup Neo4j
docker exec rag-neo4j neo4j-admin database dump neo4j

# Backup all volumes
docker run --rm -v /var/lib/docker/volumes:/backup alpine \
  tar czf /backup/rag-stack-backup-$(date +%Y%m%d).tar.gz rag-infrastructure*
```

## 🐛 Troubleshooting

### Common Issues

**MongoDB Vector Search Not Working**
```bash
# Check replica set status
docker exec rag-mongodb mongosh --eval "rs.status()"

# Reinitialize if needed
docker exec rag-mongodb mongosh --eval "rs.initiate()"
```

**Services Not Accessible**
1. Check firewall settings: `ufw status`
2. Verify containers are running: `docker compose ps`
3. Check port conflicts: `netstat -tlnp | grep :8080`

**DNS Resolution Issues**
1. Test direct IP access first
2. Verify DNS server configuration
3. Check reverse proxy settings

### Performance Tuning

**For Limited RAM (8GB)**
- Use essential stack only
- Reduce Neo4j memory: `NEO4J_dbms_memory_heap_max__size=1G`
- Limit Elasticsearch: `ES_JAVA_OPTS=-Xms256m -Xmx512m`

**For High Performance**
- Allocate more memory to services
- Use SSD storage for Docker volumes
- Consider service-specific optimization

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Areas for Contribution
- Additional vector databases
- New example applications
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MongoDB](https://mongodb.com) for native vector search
- [Neo4j](https://neo4j.com) for graph database excellence
- [Ollama](https://ollama.ai) for local LLM serving
- [Langfuse](https://langfuse.com) for open-source LLM observability
- [Hugging Face](https://huggingface.co) for Mongoku UI
- All the open-source projects that make this stack possible

## 🔗 Related Projects

- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
- [LlamaIndex](https://github.com/run-llama/llama_index) - Data framework for LLMs
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database

---

**Ready to build the future of AI applications? Deploy your RAG infrastructure today!** 🚀
