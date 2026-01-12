# RAG Infrastructure Stack - Default Access Guide

> **Note**: This file is generated automatically during deployment with your specific configuration.
> The URLs and credentials shown below are examples for the default deployment.

## Default Service Credentials

**Default Username/Password**: `admin` / `your-configured-password`

Most services use the password you configured during deployment. Some services require account creation on first visit.

## Service Access URLs

### Core Vector & Document Databases
- **MongoDB**: `mongodb://your-host:27017/` (no auth required)
- **Vector Search UI** (Custom): `http://your-host:8090` 
- **ChromaDB**: `http://your-host:8000` (Token: your-password)
- **Qdrant**: `http://your-host:6333` (no auth required)
- **Weaviate**: `http://your-host:8080` (no auth required)

### MongoDB Management UIs
- **Mongoku** (Modern): `http://your-host:3100`
- **Mongo Express** (Basic): `http://your-host:8081` (admin/your-password)

### Graph Database & Search
- **Neo4j Browser**: `http://your-host:7474` (neo4j/your-password)
- **Elasticsearch**: `http://your-host:9200` (no auth required)
- **Kibana**: `http://your-host:5601` (no auth required)

### LLM & AI Tools
- **Ollama API**: `http://your-host:11434` (no auth required)
- **Open WebUI**: `http://your-host:8085` (create account on first visit)

### Observability
- **Langfuse**: `http://your-host:3000` (create account on first visit)

### Storage & Cache
- **MinIO Console**: `http://your-host:9001` (admin/your-password)
- **Redis**: `redis://your-host:6379` (no auth required)
- **RedisInsight**: `http://your-host:8001` (no auth required)

### Development Tools
- **Jupyter Lab**: `http://your-host:8888` (Token: your-password)
- **n8n**: `http://n8n.home.local:5678` (admin/your-password)

## Connection Examples

### Python Applications
```python
from pymongo import MongoClient
from neo4j import GraphDatabase
import requests

# MongoDB connection (replace with your host/password)
client = MongoClient('mongodb://your-host:27017/')

# Neo4j connection
driver = GraphDatabase.driver('bolt://your-host:7687', auth=('neo4j', 'your-password'))

# Ollama API
response = requests.post(
    'http://your-host:11434/api/generate',
    json={'model': 'llama3.2:3b', 'prompt': 'Hello, world!'}
)
```

### Environment Variables
```bash
# Set these in your application environment
export MONGO_URL="mongodb://your-host:27017/"
export NEO4J_URI="bolt://your-host:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"
export OLLAMA_BASE_URL="http://your-host:11434"
export LANGFUSE_HOST="http://your-host:3000"
```

### REST API Examples
```bash
# Vector Search UI API
curl http://your-host:8090/api/collections

# ChromaDB with auth
curl -H "X-Chroma-Token: your-password" http://your-host:8000/api/v1/collections

# Elasticsearch query
curl http://your-host:9200/_search

# Ollama model list
curl http://your-host:11434/api/tags
```

### Docker Internal Networking
```python
# When connecting from within Docker containers, use service names:
client = MongoClient('mongodb://mongodb:27017/')
driver = GraphDatabase.driver('bolt://neo4j:7687', auth=('neo4j', 'password'))
ollama_url = 'http://ollama:11434'
```

## Notes

- Replace `your-host` with your actual server IP address
- Replace `your-password` with the password you configured during deployment
- Some services (Langfuse, Open WebUI) require account creation on first visit
- The Vector Search UI provides a modern interface for MongoDB vector operations
- Services are accessible both by hostname (if DNS configured) and direct IP
- For production deployments, consider using stronger passwords and enabling SSL

## Troubleshooting

### Can't Access Services
1. Check if services are running: `docker compose ps`
2. Verify ports are not blocked by firewall
3. Try direct IP access if using hostnames: `http://YOUR-IP:PORT`
4. Check DNS configuration if using custom domain names

### Authentication Issues
1. Verify you're using the correct password from deployment
2. Some services require account creation on first visit
3. Check service-specific logs: `docker compose logs service-name`