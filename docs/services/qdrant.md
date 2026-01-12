# Qdrant Vector Database Guide

Qdrant is a high-performance vector similarity search engine optimized for large-scale ML applications. Perfect for fast vector retrieval in RAG systems.

## Quick Access

| Component | URL | Purpose |
|-----------|-----|---------|
| **Qdrant Web UI** | `http://your-host:6333/dashboard` | Vector database management interface |
| **REST API** | `http://your-host:6333` | HTTP API for vector operations |
| **gRPC API** | `your-host:6334` | High-performance gRPC interface |

## Why Qdrant for RAG?

### Performance Benefits
- **HNSW Algorithm**: Hierarchical Navigable Small World for fast approximate search
- **Memory Efficiency**: Optimized memory usage with payload quantization  
- **High Throughput**: Designed for production workloads
- **Filtering**: Combine vector search with metadata filtering
- **Clustering**: Distributed deployment support

### RAG Use Cases
- **Fast Similarity Search**: Sub-millisecond vector queries
- **Hybrid Filtering**: Combine vector similarity with metadata filters
- **Multi-tenant RAG**: Isolate vectors by user or organization
- **Real-time Indexing**: Add vectors without rebuilding indexes
- **Approximate Search**: Balance speed vs accuracy for large datasets

## Initial Setup

### 1. Access Qdrant Dashboard
```bash
# Navigate to Qdrant Web UI
open http://your-host:6333/dashboard

# No authentication required by default
# Dashboard provides collection management and query interface
```

### 2. Verify Installation with Python
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Connect to Qdrant
client = QdrantClient(host="your-host", port=6333)

# Check cluster info
info = client.get_collections()
print(f"Qdrant collections: {info}")

# Health check
health = client.http.get("/")
print(f"Qdrant status: {health.status_code}")
```

### 3. Create Your First Collection
```python
from qdrant_client.models import Distance, VectorParams, PointStruct

# Create a collection for document embeddings
collection_name = "documents"

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=768,  # Dimension of your embeddings (e.g., nomic-embed-text)
        distance=Distance.COSINE  # Distance metric
    )
)

print(f"Created collection: {collection_name}")

# Verify collection creation
collections = client.get_collections()
print(f"Available collections: {[col.name for col in collections.collections]}")
```

## Vector Operations for RAG

### 1. Document Ingestion Pipeline
```python
import requests
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
import uuid
from datetime import datetime

class QdrantRAGManager:
    def __init__(self, host="your-host", port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.ollama_url = "http://your-host:11434"
    
    def setup_rag_collection(self, collection_name="rag_documents"):
        """Create optimized collection for RAG documents"""
        
        # Delete if exists (for development)
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        # Create collection with optimized settings
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=768,  # nomic-embed-text dimensions
                distance=Distance.COSINE
            ),
            # Optimization settings
            optimizers_config={
                "deleted_threshold": 0.2,
                "vacuum_min_vector_number": 1000,
                "default_segment_number": 0,
                "max_segment_size_kb": None,
                "memmap_threshold_kb": None,
                "indexing_threshold": 20000,
                "flush_interval_sec": 5,
                "max_optimization_threads": 1
            }
        )
        
        return collection_name
    
    def generate_embedding(self, text):
        """Generate embedding using Ollama"""
        response = requests.post(f"{self.ollama_url}/api/embeddings", json={
            "model": "nomic-embed-text",
            "prompt": text
        })
        
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            raise Exception(f"Embedding generation failed: {response.text}")
    
    def add_document(self, collection_name, document_text, metadata=None):
        """Add a document with embedding and metadata"""
        
        # Generate embedding
        embedding = self.generate_embedding(document_text)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "text": document_text,
            "added_at": datetime.now().isoformat(),
            "text_length": len(document_text),
            "word_count": len(document_text.split())
        })
        
        # Generate unique ID
        point_id = str(uuid.uuid4())
        
        # Create point
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=metadata
        )
        
        # Upsert to collection
        operation_result = self.client.upsert(
            collection_name=collection_name,
            points=[point]
        )
        
        print(f"Added document {point_id}: {operation_result}")
        return point_id
    
    def batch_add_documents(self, collection_name, documents):
        """Efficiently add multiple documents"""
        
        points = []
        
        for i, doc in enumerate(documents):
            text = doc["text"]
            metadata = doc.get("metadata", {})
            
            # Generate embedding
            embedding = self.generate_embedding(text)
            
            # Prepare metadata
            metadata.update({
                "text": text,
                "batch_index": i,
                "added_at": datetime.now().isoformat()
            })
            
            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=metadata
            )
            
            points.append(point)
        
        # Batch upsert
        operation_result = self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        print(f"Batch added {len(points)} documents: {operation_result}")
        return [p.id for p in points]

# Usage example
rag_manager = QdrantRAGManager()

# Setup collection
collection_name = rag_manager.setup_rag_collection()

# Add sample documents
sample_docs = [
    {
        "text": "Qdrant is a vector database optimized for similarity search",
        "metadata": {"category": "technology", "source": "documentation"}
    },
    {
        "text": "RAG combines retrieval and generation for better AI responses",
        "metadata": {"category": "ai", "source": "research"}
    },
    {
        "text": "Vector embeddings represent text as high-dimensional vectors",
        "metadata": {"category": "ml", "source": "tutorial"}
    }
]

document_ids = rag_manager.batch_add_documents(collection_name, sample_docs)
print(f"Added documents with IDs: {document_ids}")
```

### 2. Advanced Vector Search
```python
def search_similar_documents(self, collection_name, query_text, limit=10, 
                           score_threshold=0.7, filters=None):
    """Perform similarity search with optional filtering"""
    
    # Generate query embedding
    query_vector = self.generate_embedding(query_text)
    
    # Build filter if provided
    search_filter = None
    if filters:
        conditions = []
        for field, value in filters.items():
            conditions.append(
                FieldCondition(key=field, match=MatchValue(value=value))
            )
        search_filter = Filter(must=conditions)
    
    # Perform search
    search_results = self.client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=search_filter,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
        with_vectors=False  # Don't return vectors to save bandwidth
    )
    
    # Format results
    results = []
    for hit in search_results:
        results.append({
            "id": hit.id,
            "score": hit.score,
            "text": hit.payload.get("text", ""),
            "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
        })
    
    return results

# Example searches
query = "What is vector similarity search?"

# Basic similarity search
results = rag_manager.search_similar_documents(collection_name, query, limit=5)

print(f"Search results for: '{query}'")
for i, result in enumerate(results, 1):
    print(f"{i}. Score: {result['score']:.3f}")
    print(f"   Text: {result['text'][:100]}...")
    print(f"   Metadata: {result['metadata']}")

# Filtered search (only technology category)
filtered_results = rag_manager.search_similar_documents(
    collection_name, 
    query, 
    limit=5,
    filters={"category": "technology"}
)
```

### 3. Hybrid Search with Multiple Vectors
```python
def setup_multi_vector_collection(self, collection_name="multi_vector_docs"):
    """Create collection with multiple vector types"""
    
    from qdrant_client.models import VectorParams
    
    # Create collection with named vectors
    self.client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "content": VectorParams(size=768, distance=Distance.COSINE),  # Content embeddings
            "title": VectorParams(size=768, distance=Distance.COSINE),    # Title embeddings
            "summary": VectorParams(size=384, distance=Distance.COSINE)   # Summary embeddings (different model)
        }
    )
    
    return collection_name

def multi_vector_search(self, collection_name, query_text, vector_weights=None):
    """Search across multiple vector types with weighting"""
    
    if vector_weights is None:
        vector_weights = {"content": 0.7, "title": 0.2, "summary": 0.1}
    
    query_embedding = self.generate_embedding(query_text)
    
    # Search each vector type
    all_results = {}
    
    for vector_name, weight in vector_weights.items():
        if weight > 0:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=(vector_name, query_embedding),
                limit=20,
                with_payload=True
            )
            
            # Weight the scores
            for hit in results:
                hit.score *= weight
                
                if hit.id in all_results:
                    all_results[hit.id]["score"] += hit.score
                else:
                    all_results[hit.id] = {
                        "score": hit.score,
                        "payload": hit.payload
                    }
    
    # Sort by combined score
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )[:10]
    
    return [(doc_id, data["score"], data["payload"]) for doc_id, data in sorted_results]
```

## Web Dashboard Usage

### 1. Dashboard Features
```bash
# Access dashboard at http://your-host:6333/dashboard

Features available:
1. Collections Management
   - View all collections
   - Create new collections
   - Configure vector parameters
   - Delete collections

2. Data Browser
   - Browse points in collections
   - View vector data and payloads
   - Search and filter points
   - Manual point insertion

3. Search Interface
   - Test vector queries
   - Configure search parameters
   - Apply filters
   - View search results

4. Cluster Information
   - Node status
   - Memory usage
   - Performance metrics
   - Configuration details
```

### 2. Collection Management via UI
```bash
# Create Collection via UI:
1. Click "Create Collection"
2. Enter collection name
3. Set vector size (e.g., 768 for nomic-embed-text)
4. Choose distance metric (Cosine for most RAG use cases)
5. Configure advanced options if needed
6. Click "Create"

# Browse Data:
1. Select collection from dropdown
2. Use pagination to browse points
3. Click on points to view details
4. Use search filters to find specific points
```

### 3. Testing Queries via Dashboard
```bash
# Query Interface:
1. Select collection
2. Enter vector values manually or upload JSON
3. Set limit and score threshold
4. Apply filters if needed
5. Execute search
6. View results with scores and payloads
```

## Performance Optimization

### 1. Index Configuration
```python
def optimize_collection_for_production(self, collection_name):
    """Configure collection for production performance"""
    
    from qdrant_client.models import (
        OptimizersConfigDiff, HnswConfigDiff, WalConfigDiff
    )
    
    # Update collection with optimized settings
    self.client.update_collection(
        collection_name=collection_name,
        optimizer_config=OptimizersConfigDiff(
            deleted_threshold=0.2,          # Clean up deleted vectors
            vacuum_min_vector_number=1000,  # Minimum vectors before cleanup
            default_segment_number=0,       # Auto-determine segments
            indexing_threshold=20000,       # Start indexing after 20k vectors
            flush_interval_sec=5            # Flush to disk interval
        ),
        hnsw_config=HnswConfigDiff(
            m=16,               # Number of connections per layer
            ef_construct=100,   # Size of candidate set during construction
            full_scan_threshold=10000,  # Use exact search for small collections
            max_indexing_threads=0,     # Use all available threads
            on_disk=False              # Keep index in memory for speed
        ),
        wal_config=WalConfigDiff(
            wal_capacity_mb=32,     # Write-ahead log size
            wal_segments_ahead=0    # Pre-allocate WAL segments
        )
    )
    
    print(f"Optimized collection {collection_name} for production")
```

### 2. Batch Operations
```python
def efficient_bulk_operations(self, collection_name, operations):
    """Perform bulk operations efficiently"""
    
    from qdrant_client.models import UpdateOperation, UpsertOperation
    
    # Batch size for optimal performance
    batch_size = 100
    
    for i in range(0, len(operations), batch_size):
        batch = operations[i:i + batch_size]
        
        try:
            self.client.batch_update_points(
                collection_name=collection_name,
                update_operations=batch
            )
            print(f"Processed batch {i//batch_size + 1}")
            
        except Exception as e:
            print(f"Batch {i//batch_size + 1} failed: {e}")
```

### 3. Memory Management
```python
def monitor_collection_stats(self, collection_name):
    """Monitor collection performance and memory usage"""
    
    # Get collection info
    info = self.client.get_collection(collection_name)
    
    stats = {
        "vectors_count": info.vectors_count,
        "indexed_vectors_count": info.indexed_vectors_count,
        "points_count": info.points_count,
        "segments_count": len(info.segments) if info.segments else 0,
        "config": {
            "vector_size": info.config.vector_size,
            "distance": info.config.distance
        },
        "status": info.status
    }
    
    print(f"Collection {collection_name} stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return stats
```

## Production Deployment

### 1. Clustering Setup
```python
def setup_qdrant_cluster():
    """Configuration for Qdrant cluster deployment"""
    
    cluster_config = {
        "storage": {
            "storage_path": "./storage",
            "snapshots_path": "./snapshots",
            "temp_path": "./temp"
        },
        "service": {
            "host": "0.0.0.0",
            "port": 6333,
            "grpc_port": 6334,
            "enable_cors": True,
            "max_request_size_mb": 32
        },
        "cluster": {
            "enabled": True,
            "p2p": {
                "port": 6335
            },
            "consensus": {
                "max_message_queue_size": 16384,
                "tick_period_ms": 100
            }
        },
        "log_level": "info"
    }
    
    return cluster_config
```

### 2. Backup and Recovery
```python
def backup_collection(self, collection_name, backup_path):
    """Create collection backup"""
    
    # Create snapshot
    snapshot_result = self.client.create_snapshot(collection_name)
    snapshot_name = snapshot_result.name
    
    print(f"Created snapshot: {snapshot_name}")
    
    # Download snapshot (in production, use proper file handling)
    snapshot_data = self.client.get_snapshot(collection_name, snapshot_name)
    
    with open(backup_path, 'wb') as f:
        f.write(snapshot_data)
    
    print(f"Snapshot saved to: {backup_path}")
    
    return snapshot_name

def restore_collection(self, collection_name, backup_path):
    """Restore collection from backup"""
    
    # Upload and restore snapshot
    with open(backup_path, 'rb') as f:
        snapshot_data = f.read()
    
    # Implementation depends on Qdrant version
    # Typically involves uploading snapshot and triggering restore
    print(f"Restored collection {collection_name} from {backup_path}")
```

## Integration Patterns

### 1. FastAPI Integration
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None
    score_threshold: float = 0.0

class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]

@app.post("/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    """Search endpoint using Qdrant"""
    
    try:
        results = rag_manager.search_similar_documents(
            collection_name="rag_documents",
            query_text=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold,
            filters=request.filters
        )
        
        return [SearchResult(**result) for result in results]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_document")
async def add_document(text: str, metadata: Optional[Dict[str, Any]] = None):
    """Add document endpoint"""
    
    try:
        doc_id = rag_manager.add_document(
            collection_name="rag_documents",
            document_text=text,
            metadata=metadata
        )
        
        return {"document_id": doc_id, "status": "added"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 2. Streaming Search for Large Results
```python
async def streaming_search(self, collection_name, query_text, batch_size=100):
    """Stream search results for large result sets"""
    
    query_vector = self.generate_embedding(query_text)
    offset = 0
    
    while True:
        batch_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=batch_size,
            offset=offset,
            with_payload=True
        )
        
        if not batch_results:
            break
        
        for hit in batch_results:
            yield {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
        
        offset += batch_size
        
        # Break if fewer results than batch_size (reached end)
        if len(batch_results) < batch_size:
            break
```

## Troubleshooting

### Common Issues

**"Collection not found"**
```python
# Check available collections
collections = client.get_collections()
print([col.name for col in collections.collections])

# Create collection if missing
if "your_collection" not in [col.name for col in collections.collections]:
    client.create_collection(
        collection_name="your_collection",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
```

**"Vector dimension mismatch"**
```python
# Check collection vector configuration
collection_info = client.get_collection("your_collection")
print(f"Expected vector size: {collection_info.config.vector_size}")

# Ensure embedding dimension matches
embedding = generate_embedding("test")
print(f"Actual embedding size: {len(embedding)}")
```

**"Slow search performance"**
```python
# Check if collection is indexed
collection_info = client.get_collection("your_collection")
print(f"Indexed vectors: {collection_info.indexed_vectors_count}")
print(f"Total vectors: {collection_info.vectors_count}")

# Wait for indexing to complete or trigger optimization
if collection_info.indexed_vectors_count < collection_info.vectors_count:
    print("Indexing in progress...")
```

**"Memory usage too high"**
```bash
# Monitor Qdrant memory usage
docker stats rag-qdrant

# Configure memory limits in docker-compose.yml
mem_limit: 4g
memswap_limit: 4g
```

## Learning Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/) - Official documentation
- [Python Client Guide](https://qdrant.tech/documentation/python-client/) - Complete SDK reference
- [Vector Database Tutorial](https://qdrant.tech/articles/vector-database/) - Concepts and best practices
- [Performance Benchmarks](https://qdrant.tech/benchmarks/) - Performance comparisons