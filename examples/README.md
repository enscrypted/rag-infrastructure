# RAG Infrastructure Examples

Working examples demonstrating different RAG implementations using the stack.

## Available Examples

### Basic RAG
**Location:** [`basic-rag/`](./basic-rag/)

Simple RAG implementation using MongoDB vector search and Ollama:
- Document ingestion with automatic embeddings
- Vector similarity search
- Interactive Q&A interface

```bash
cd basic-rag
python simple_rag.py
```

### Graph RAG
**Location:** [`graph-rag/`](./graph-rag/)

Enhanced RAG using Neo4j knowledge graphs for hybrid retrieval:
- Entity extraction from documents
- Relationship mapping in Neo4j
- Combined vector + graph search
- Multi-hop reasoning support

```bash
cd graph-rag
python graph_rag.py
```

### RAG Evaluation
**Location:** [`evaluation/`](./evaluation/)

Measure and monitor RAG system performance:
- Automated relevance scoring
- Answer quality assessment
- Latency tracking
- Langfuse integration for observability

```bash
cd evaluation
python eval_rag.py
```

### Multimodal RAG
**Location:** [`multimodal-rag/`](./multimodal-rag/)

Process both text and images in a unified RAG pipeline:
- Image analysis using LLaVA vision model
- Combined text + image retrieval
- Visual question answering
- Automatic image description generation

```bash
cd multimodal-rag
python multimodal_rag.py
```

## Prerequisites

All examples require:
1. RAG Infrastructure Stack deployed and running
2. Python 3.8+
3. Required models pulled in Ollama:
   ```bash
   docker exec rag-ollama ollama pull nomic-embed-text
   docker exec rag-ollama ollama pull llama3.2:3b
   ```

## Quick Start

1. **Deploy the stack:**
   ```bash
   cd .. && ./scripts/deploy.sh
   ```

2. **Install Python dependencies:**
   ```bash
   pip install pymongo neo4j requests langfuse
   ```

3. **Set environment variables (optional):**
   ```bash
   export MONGO_URL="mongodb://localhost:27017/"
   export OLLAMA_BASE_URL="http://localhost:11434"
   export NEO4J_URI="bolt://localhost:7687"
   ```

4. **Run an example:**
   ```bash
   cd basic-rag
   python simple_rag.py
   ```

## Example Comparison

| Feature | Basic RAG | Graph RAG | Evaluation | Multimodal |
|---------|-----------|-----------|------------|------------|
| Vector Search | Yes | Yes | Yes | Yes |
| Knowledge Graph | No | Yes | No | No |
| Entity Extraction | No | Yes | No | No |
| Image Processing | No | No | No | Yes |
| Performance Metrics | No | No | Yes | No |
| Langfuse Tracing | No | No | Yes | No |
| Best For | Learning | Complex queries | Quality assurance | Visual content |

## Learn More

- [RAG Architecture Guide](../docs/concepts/rag-architecture.md)
- [MongoDB Vector Search](../docs/services/mongodb.md)
- [Neo4j Documentation](../docs/services/neo4j.md)
- [Langfuse Observability](../docs/services/langfuse.md)
