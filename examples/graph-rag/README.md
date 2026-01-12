# Graph RAG Example

This example demonstrates Graph RAG (Retrieval-Augmented Generation with Knowledge Graphs) using Neo4j and MongoDB.

## What is Graph RAG?

Graph RAG enhances traditional vector-based RAG by:
- **Entity Extraction**: Automatically identifying entities in documents
- **Relationship Mapping**: Building a knowledge graph of connections
- **Hybrid Retrieval**: Combining vector similarity with graph traversal
- **Context Enrichment**: Using entity relationships to improve answers

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Documents  │────▶│   Ollama    │────▶│  Entities   │
└─────────────┘     │ (Extract)   │     └──────┬──────┘
                    └─────────────┘            │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   MongoDB   │◀────│  Embeddings │     │    Neo4j    │
│  (Vectors)  │     └─────────────┘     │   (Graph)   │
└──────┬──────┘                         └──────┬──────┘
       │                                       │
       └───────────────┬───────────────────────┘
                       ▼
               ┌─────────────┐
               │   Hybrid    │
               │  Retrieval  │
               └─────────────┘
```

## Prerequisites

1. RAG Infrastructure Stack deployed
2. Python 3.8+
3. Required packages: `pymongo`, `neo4j`, `requests`

## Setup

```bash
pip install pymongo neo4j requests
```

## Usage

### Basic Usage

```bash
python graph_rag.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `MONGO_URL` | `mongodb://localhost:27017/` | MongoDB connection |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |

### Interactive Commands

- Ask any question - uses hybrid (vector + graph) retrieval
- `graph` - View knowledge graph statistics
- `entity <name>` - Get details about a specific entity
- `quit` - Exit

### Example Session

```
🚀 Graph RAG Demo Starting...
📊 Connected to Neo4j: bolt://localhost:7687
📊 Connected to MongoDB: graph_rag_demo
🤖 Connected to Ollama: http://localhost:11434

📚 Loading sample documents...
📄 Document stored: 507f1f77bcf86cd799439011
🔍 Extracted 4 entities
🔗 Extracted 3 relationships
✅ Knowledge graph updated
...

❓ Your question: What frameworks are used for machine learning?

🔍 Searching with Graph RAG...

🤖 Answer (hybrid (vector + graph)):
Based on the context, the main machine learning frameworks are:
1. TensorFlow - developed by Google, supports CPU and GPU
2. PyTorch - developed by Meta, known for dynamic computation graphs

📚 Sources (3):
   1. [hybrid] TensorFlow is an open-source machine learning framework...
   2. [vector] Machine learning is a subset of artificial intelligence...
```

## How It Works

### 1. Document Ingestion

```python
rag = GraphRAG()

# Add document - automatically extracts entities and builds graph
rag.add_document(
    content="TensorFlow is developed by Google for machine learning.",
    metadata={"source": "tech_doc"}
)
```

### 2. Entity Extraction

The system uses Ollama to extract entities:
- **PERSON**: People mentioned (e.g., "Guido van Rossum")
- **ORG**: Organizations (e.g., "Google", "OpenAI")
- **TECH**: Technologies (e.g., "Python", "TensorFlow")
- **CONCEPT**: Abstract concepts (e.g., "machine learning")

### 3. Knowledge Graph

Entities are stored in Neo4j with relationships:
```cypher
(TensorFlow:Entity:TECH)-[:DEVELOPED_BY]->(Google:Entity:ORG)
(Document)-[:MENTIONS]->(TensorFlow)
```

### 4. Hybrid Retrieval

Queries use both:
- **Vector Search**: Find semantically similar documents
- **Graph Search**: Find documents mentioning related entities

### 5. Response Generation

Context from both sources is combined for better answers.

## Customization

### Adding Custom Entity Types

```python
def _extract_entities(self, text: str) -> List[Dict]:
    prompt = f"""Extract entities with types:
    PERSON, ORG, TECH, CONCEPT, PRODUCT, LOCATION
    ..."""
```

### Adjusting Hybrid Weights

```python
# In hybrid_search method
combined.sort(
    key=lambda x: x["vector_score"] + x["graph_score"] * 0.5,  # Adjust weight
    reverse=True
)
```

### Custom Relationships

```python
# Modify _extract_relationships for domain-specific relations
relationships = ["WORKS_FOR", "CREATED", "USES", "COMPETES_WITH", "PART_OF"]
```

## Benefits Over Standard RAG

| Feature | Standard RAG | Graph RAG |
|---------|-------------|-----------|
| Retrieval | Vector similarity only | Vector + entity relationships |
| Context | Isolated documents | Connected knowledge |
| Multi-hop | Limited | Follows relationship chains |
| Entity queries | Keyword matching | Structured entity lookup |

## Troubleshooting

**"Neo4j connection refused"**
```bash
docker compose logs neo4j
# Check if service is running
```

**"No entities extracted"**
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Try with more descriptive text

**"Empty graph"**
- Verify documents were added successfully
- Check Neo4j browser: `http://localhost:7474`

## Learn More

- [Neo4j Documentation](../../docs/services/neo4j.md)
- [MongoDB Vector Search](../../docs/services/mongodb.md)
- [RAG Architecture](../../docs/concepts/rag-architecture.md)
