# ChromaDB Vector Database Guide

ChromaDB is a popular open-source vector database designed for AI applications. It's perfect for RAG systems requiring fast similarity search and easy integration.

## Important: v2 API (2025+)

**The v1 API is deprecated** and returns HTTP 410. Use the v2 API for all operations.

| API Version | Status | Endpoint Pattern |
|-------------|--------|------------------|
| v1 | ❌ Deprecated (HTTP 410) | `/api/v1/...` |
| v2 | ✅ Current | `/api/v2/tenants/default_tenant/databases/default_database/...` |

## Quick Access

| Component | URL | Credentials |
|-----------|-----|-------------|
| **ChromaDB API** | `http://your-host:8000` | Token: your-password |
| **Health Check (v2)** | `http://your-host:8000/api/v2/heartbeat` | - |

## Why ChromaDB for RAG?

### Key Benefits
- **Simple API**: Easy to use REST API and Python client
- **Fast Queries**: Optimized for similarity search
- **Metadata Filtering**: Rich filtering capabilities
- **Embeddings Support**: Multiple embedding functions
- **Persistence**: Automatic data persistence

### RAG Use Cases
- **Document Search**: Find relevant documents by similarity
- **Hybrid Search**: Combine vector and metadata filtering
- **Multi-Collection**: Separate different document types
- **Real-time Updates**: Add/update documents on the fly

## Initial Setup

### 1. Verify Installation (v2 API)
```bash
# Check ChromaDB status (v2 API)
curl http://your-host:8000/api/v2/heartbeat

# Should return: {"nanosecond heartbeat": timestamp}
```

### 2. Test API Access (v2 API with Bearer Token)
```bash
# v2 API uses Bearer token authentication
curl -H "Authorization: Bearer your-password" \
     "http://your-host:8000/api/v2/tenants/default_tenant/databases/default_database/collections"

# Should return: [] (empty list initially)
```

### 3. Python Client Setup
```python
import chromadb
from chromadb.config import Settings

# Connect to ChromaDB with token authentication (2025+ syntax)
client = chromadb.HttpClient(
    host="your-host",
    port=8000,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        chroma_client_auth_credentials="your-token",
        chroma_auth_token_transport_header="X-Chroma-Token"  # For Python client
    )
)

# Test connection
print(f"Heartbeat: {client.heartbeat()}")
print(f"Collections: {client.list_collections()}")
```

### 4. Direct REST API (v2) Example
```python
import requests

# v2 API base URL
base_url = "http://your-host:8000/api/v2/tenants/default_tenant/databases/default_database"
headers = {"Authorization": "Bearer your-token"}

# Create collection
response = requests.post(f"{base_url}/collections",
    headers=headers,
    json={"name": "my_collection"}
)
collection_id = response.json()["id"]  # v2 returns UUID

# Add documents (use collection UUID)
requests.post(f"{base_url}/collections/{collection_id}/add",
    headers=headers,
    json={
        "ids": ["doc1", "doc2"],
        "documents": ["Hello world", "Test document"],
        "embeddings": [[0.1] * 768, [0.2] * 768]  # Required if no auto-embed
    }
)

# Query (use collection UUID)
response = requests.post(f"{base_url}/collections/{collection_id}/query",
    headers=headers,
    json={
        "query_embeddings": [[0.15] * 768],
        "n_results": 2
    }
)
print(response.json())
```

## Basic Operations

### 1. Creating Collections

```python
class ChromaRAG:
    def __init__(self, host="your-host", port=8000, token="your-password"):
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=chromadb.Settings(
                chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                chroma_client_auth_credentials=token,
                chroma_auth_token_transport_header="X-Chroma-Token"
            )
        )
    
    def create_collection(self, name, metadata=None, embedding_function=None):
        """Create a new collection with optional embedding function"""
        # ChromaDB 2025+ uses get_or_create_collection as best practice
        collection = self.client.get_or_create_collection(
            name=name,
            metadata=metadata or {"description": "RAG documents"},
            embedding_function=embedding_function  # Optional custom embedder
        )
        print(f"✅ Created/retrieved collection: {name}")
        return collection
    
    def setup_embedding_function(self, model="text-embedding-ada-002"):
        """Setup custom embedding function (optional)"""
        from chromadb.utils import embedding_functions
        
        # Example with OpenAI (requires API key)
        # return embedding_functions.OpenAIEmbeddingFunction(
        #     model_name=model
        # )
        
        # Default: Uses built-in all-MiniLM-L6-v2 via ONNX
        return None  # Let ChromaDB use default embedder

# Usage
chroma_rag = ChromaRAG()
collection = chroma_rag.get_or_create_collection("documents")
```

### 2. Adding Documents

```python
def add_documents(self, collection, documents, embeddings=None, metadatas=None):
    """Add documents to collection"""
    
    # Generate IDs if not provided
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Add documents
    collection.add(
        documents=documents,
        embeddings=embeddings,  # Optional: ChromaDB can generate embeddings
        metadatas=metadatas or [{}] * len(documents),
        ids=ids
    )
    
    print(f"✅ Added {len(documents)} documents to collection")
    return ids

# Example usage
documents = [
    "Python is a programming language used for AI development",
    "Machine learning is a subset of artificial intelligence",
    "Vector databases store high-dimensional embeddings"
]

metadatas = [
    {"source": "python_guide", "category": "programming"},
    {"source": "ml_intro", "category": "ai"},
    {"source": "vector_guide", "category": "databases"}
]

doc_ids = chroma_rag.add_documents(collection, documents, metadatas=metadatas)
```

### 3. Querying Collections

```python
def search_similar(self, collection, query_text, n_results=5, where=None):
    """Search for similar documents"""
    
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where  # Metadata filtering
    )
    
    formatted_results = []
    for i in range(len(results['ids'][0])):
        formatted_results.append({
            'id': results['ids'][0][i],
            'document': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i]
        })
    
    return formatted_results

# Search examples
# Basic similarity search
results = chroma_rag.search_similar(collection, "What is Python?")

# Filtered search
filtered_results = chroma_rag.search_similar(
    collection, 
    "programming languages",
    where={"category": "programming"}
)

for result in results:
    print(f"Distance: {result['distance']:.3f}")
    print(f"Document: {result['document'][:100]}...")
    print(f"Metadata: {result['metadata']}\n")
```

## Advanced RAG Patterns

### 1. Multi-Collection RAG

```python
class MultiCollectionRAG:
    def __init__(self, client):
        self.client = client
        self.collections = {}
    
    def setup_collections(self):
        """Setup different collections for different document types"""
        
        collection_configs = {
            "technical_docs": {"description": "Technical documentation"},
            "research_papers": {"description": "Academic research papers"},
            "company_knowledge": {"description": "Company-specific knowledge"},
            "code_snippets": {"description": "Code examples and snippets"}
        }
        
        for name, metadata in collection_configs.items():
            self.collections[name] = self.client.get_or_create_collection(
                name=name, 
                metadata=metadata
            )
    
    def route_query(self, query):
        """Route query to appropriate collection(s)"""
        
        # Simple routing logic (could be more sophisticated)
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['code', 'function', 'syntax']):
            return ['code_snippets']
        elif any(term in query_lower for term in ['research', 'paper', 'study']):
            return ['research_papers']
        elif any(term in query_lower for term in ['company', 'internal', 'policy']):
            return ['company_knowledge']
        else:
            return ['technical_docs', 'research_papers']  # Search multiple
    
    def multi_collection_search(self, query, n_results=5):
        """Search across multiple relevant collections"""
        
        target_collections = self.route_query(query)
        all_results = []
        
        for collection_name in target_collections:
            if collection_name in self.collections:
                collection = self.collections[collection_name]
                results = collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
                
                # Add collection info to results
                for i in range(len(results['ids'][0])):
                    all_results.append({
                        'collection': collection_name,
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    })
        
        # Sort by distance and return top results
        all_results.sort(key=lambda x: x['distance'])
        return all_results[:n_results]
```

### 2. Hybrid Search with Metadata

```python
def hybrid_search(self, collection, query, metadata_filters=None, date_range=None):
    """Combine vector similarity with metadata filtering"""
    
    # Build where clause
    where_clause = {}
    
    if metadata_filters:
        where_clause.update(metadata_filters)
    
    if date_range:
        where_clause["$and"] = [
            {"created_date": {"$gte": date_range["start"]}},
            {"created_date": {"$lte": date_range["end"]}}
        ]
    
    # Perform search
    results = collection.query(
        query_texts=[query],
        n_results=10,
        where=where_clause if where_clause else None
    )
    
    return results

# Example: Search for recent Python documents
recent_python_docs = chroma_rag.hybrid_search(
    collection,
    "Python programming",
    metadata_filters={"category": "programming"},
    date_range={"start": "2024-01-01", "end": "2024-12-31"}
)
```

### 3. Embedding Management

```python
def add_documents_with_custom_embeddings(self, collection, documents, embedding_function):
    """Add documents with custom embeddings"""
    
    # Generate embeddings using your preferred model
    embeddings = []
    for doc in documents:
        # Use OpenAI, Ollama, or other embedding models
        embedding = embedding_function(doc)
        embeddings.append(embedding)
    
    # Add to ChromaDB
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )

def update_embeddings(self, collection, document_ids, new_embeddings):
    """Update embeddings for existing documents"""
    
    collection.update(
        ids=document_ids,
        embeddings=new_embeddings
    )
    
    print(f"✅ Updated embeddings for {len(document_ids)} documents")
```

## Integration with RAG Pipeline

### 1. Complete RAG Implementation

```python
class ChromaDBRAG:
    def __init__(self, client, ollama_url="http://your-host:11434"):
        self.client = client
        self.collection = None
        self.ollama_url = ollama_url
    
    def setup_knowledge_base(self, collection_name="rag_documents"):
        """Setup ChromaDB collection for RAG"""
        self.collection = self.client.get_or_create_collection(collection_name)
    
    def ingest_documents(self, documents, metadatas=None):
        """Ingest documents into knowledge base"""
        
        # Let ChromaDB generate embeddings automatically
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas or [{}] * len(documents),
            ids=ids
        )
        
        return ids
    
    def retrieve_context(self, query, n_results=5):
        """Retrieve relevant context for query"""
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format context
        context_docs = []
        for i in range(len(results['ids'][0])):
            context_docs.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'relevance_score': 1.0 - results['distances'][0][i]  # Convert distance to score
            })
        
        return context_docs
    
    def generate_response(self, query, context_docs):
        """Generate response using Ollama"""
        
        # Combine context
        context_text = "\n\n".join([doc['content'] for doc in context_docs])
        
        prompt = f"""Based on the following context, answer the question accurately.

Context:
{context_text}

Question: {query}

Answer:"""
        
        # Call Ollama
        import requests
        response = requests.post(f"{self.ollama_url}/api/generate", json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        })
        
        return response.json()["response"]
    
    def query(self, question):
        """Complete RAG pipeline"""
        
        # Retrieve relevant documents
        context_docs = self.retrieve_context(question)
        
        if not context_docs:
            return "I don't have enough information to answer this question."
        
        # Generate response
        answer = self.generate_response(question, context_docs)
        
        return {
            "question": question,
            "answer": answer,
            "sources": context_docs
        }

# Usage
rag = ChromaDBRAG(client)
rag.setup_knowledge_base()

# Add documents
documents = [
    "ChromaDB is an open-source vector database for AI applications.",
    "Vector databases enable similarity search using embeddings.",
    "RAG combines retrieval and generation for better AI responses."
]

rag.ingest_documents(documents)

# Query
result = rag.query("What is ChromaDB?")
print(f"Answer: {result['answer']}")
```

### 2. Real-time Document Updates

```python
def update_knowledge_base(self, document_id, new_content, new_metadata=None):
    """Update existing document in knowledge base"""
    
    self.collection.update(
        ids=[document_id],
        documents=[new_content],
        metadatas=[new_metadata] if new_metadata else None
    )

def delete_documents(self, document_ids):
    """Remove documents from knowledge base"""
    
    self.collection.delete(ids=document_ids)
    print(f"🗑️ Deleted {len(document_ids)} documents")

def get_collection_stats(self):
    """Get collection statistics"""
    
    count = self.collection.count()
    return {
        "total_documents": count,
        "collection_name": self.collection.name
    }
```

## Performance Optimization

### 1. Batch Operations

```python
def batch_ingest(self, documents, batch_size=100):
    """Ingest documents in batches for better performance"""
    
    total_docs = len(documents)
    processed = 0
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        batch_ids = [f"doc_{i + j}" for j in range(len(batch))]
        
        self.collection.add(
            documents=batch,
            ids=batch_ids
        )
        
        processed += len(batch)
        print(f"📈 Processed {processed}/{total_docs} documents")
    
    print(f"✅ Batch ingest complete: {total_docs} documents")
```

### 2. Query Optimization

```python
def optimized_search(self, query, n_results=5, include_embeddings=False):
    """Optimized search with minimal data transfer"""
    
    include_params = ["documents", "metadatas", "distances"]
    if include_embeddings:
        include_params.append("embeddings")
    
    results = self.collection.query(
        query_texts=[query],
        n_results=n_results,
        include=include_params
    )
    
    return results
```

## Monitoring and Maintenance

### 1. Collection Management

```python
def list_all_collections(self):
    """List all collections with statistics"""
    
    collections = self.client.list_collections()
    
    collection_info = []
    for collection in collections:
        coll_obj = self.client.get_collection(collection.name)
        info = {
            "name": collection.name,
            "count": coll_obj.count(),
            "metadata": collection.metadata
        }
        collection_info.append(info)
    
    return collection_info

def backup_collection(self, collection_name, output_file):
    """Export collection data for backup"""
    
    collection = self.client.get_collection(collection_name)
    
    # Get all documents
    all_data = collection.get(
        include=["documents", "metadatas", "embeddings"]
    )
    
    # Save to file
    import json
    with open(output_file, 'w') as f:
        json.dump({
            "collection_name": collection_name,
            "data": all_data
        }, f)
    
    print(f"💾 Backed up collection to {output_file}")
```

### 2. Health Monitoring

```python
def health_check(self):
    """Check ChromaDB health and performance"""
    
    try:
        # Test connection
        version = self.client.get_version()
        
        # Test basic operations
        test_collection = self.client.get_or_create_collection("health_test")
        test_collection.add(
            documents=["test document"],
            ids=["test_id"]
        )
        
        # Test search
        results = test_collection.query(
            query_texts=["test"],
            n_results=1
        )
        
        # Cleanup
        self.client.delete_collection("health_test")
        
        return {
            "status": "healthy",
            "version": version,
            "search_working": len(results['ids'][0]) > 0
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

## Troubleshooting

### Common Issues

**"Connection refused"**
```bash
# Check ChromaDB service status
docker compose ps chromadb

# Check logs
docker compose logs chromadb

# Test basic connectivity (v2 API)
curl http://your-host:8000/api/v2/heartbeat
```

**"HTTP 410 Gone" (v1 API deprecated)**
```python
# The v1 API is deprecated! Use v2 API instead:
# OLD (returns 410): /api/v1/collections
# NEW (correct):     /api/v2/tenants/default_tenant/databases/default_database/collections
```

**"Authentication failed"**
```python
# Verify token configuration (v2 API uses Bearer auth)
import requests
response = requests.get(
    "http://your-host:8000/api/v2/tenants/default_tenant/databases/default_database/collections",
    headers={"Authorization": "Bearer your-password"}
)
print(f"Status: {response.status_code}")
```

**"Collection not found"**
```python
# List available collections
collections = client.list_collections()
print("Available collections:", [c.name for c in collections])

# Create collection if needed
collection = client.get_or_create_collection("your_collection_name")
```

**Poor search quality**
```python
# Check embedding quality
results = collection.query(
    query_texts=["test query"],
    n_results=5,
    include=["documents", "distances", "embeddings"]
)

# Analyze distances (lower = more similar)
for i, distance in enumerate(results['distances'][0]):
    print(f"Result {i}: distance = {distance:.3f}")
```

## Best Practices

### Data Management
1. **Use meaningful IDs**: Include timestamps or source info
2. **Rich metadata**: Add searchable metadata fields
3. **Regular cleanup**: Remove outdated documents
4. **Backup strategy**: Regular exports of important collections

### Performance
1. **Batch operations**: Use batch adds/updates for large datasets
2. **Appropriate n_results**: Don't retrieve more than needed
3. **Metadata filtering**: Use filters to reduce search space
4. **Monitor collection size**: Large collections may need optimization

### Security
1. **Token authentication**: Always use authentication in production
2. **Network security**: Restrict access to trusted networks
3. **Data validation**: Validate inputs before adding to collections
4. **Access logging**: Monitor who accesses what data

## Learning Resources

- [ChromaDB Documentation](https://docs.trychroma.com/) - Official documentation
- [ChromaDB Cookbook](https://docs.trychroma.com/cookbook) - Examples and patterns
- [Vector Database Guide](https://www.pinecone.io/learn/vector-database/) - General vector DB concepts
- [Embedding Best Practices](https://platform.openai.com/docs/guides/embeddings) - Embedding techniques