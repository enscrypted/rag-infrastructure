# Weaviate Vector Database Guide

Weaviate is a GraphQL-based vector database with built-in machine learning capabilities. Excellent for semantic search and AI-powered applications.

## Quick Access

| Component | URL | Purpose |
|-----------|-----|---------|
| **Weaviate Console** | `http://your-host:8080/v1/` | API endpoint and health check |
| **GraphQL Playground** | `http://your-host:8080/v1/graphql` | Interactive query interface |
| **Schema Endpoint** | `http://your-host:8080/v1/schema` | Schema management API |

## Why Weaviate for RAG?

### Unique Features
- **GraphQL Interface**: Intuitive query language for complex searches
- **Auto-Schema**: Automatic schema inference from data
- **Built-in ML Models**: Text vectorization without external services
- **Hybrid Search**: Combine keyword and vector search natively
- **Multi-modal Support**: Text, images, and custom vectors

### RAG Use Cases
- **Semantic Document Search**: Natural language querying
- **Multi-modal RAG**: Combine text and image retrieval
- **Auto-vectorization**: Automatic embedding generation
- **Knowledge Graph Integration**: Rich relationship modeling
- **Cross-reference Search**: Find related concepts across collections

## Initial Setup

### 1. Verify Weaviate Installation
```bash
# Check Weaviate health
curl http://your-host:8080/v1/

# Should return server information
# Access GraphQL playground
open http://your-host:8080/v1/graphql
```

### 2. Connect with Python Client (v4 - 2025+)
```python
import weaviate
from weaviate.classes.init import Auth

# Initialize v4 client (recommended for Weaviate >= 1.23.7)
# For local/self-hosted Weaviate without authentication:
client = weaviate.connect_to_custom(
    http_host="your-host",
    http_port=8080,
    http_secure=False,
    grpc_host="your-host",
    grpc_port=50051,
    grpc_secure=False,
)

# Check if Weaviate is ready
print(f"Weaviate ready: {client.is_ready()}")

# Get cluster information
meta = client.get_meta()
print(f"Weaviate version: {meta}")

# Always close when done
# client.close()
```

### 3. Create Collection (v4 syntax - replaces schema)
```python
from weaviate.classes.config import Configure, Property, DataType

# Create collection with your own vectors (no auto-vectorization)
client.collections.create(
    name="Document",
    description="A document for RAG retrieval",
    # For self-provided vectors (bring your own embeddings):
    vectorizer_config=Configure.Vectorizer.none(),
    properties=[
        Property(name="title", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT),
        Property(name="category", data_type=DataType.TEXT),
        Property(name="source", data_type=DataType.TEXT),
        Property(name="chunk_index", data_type=DataType.INT),
    ],
)
print("Created Document collection")

# Verify collection exists
collections = client.collections.list_all()
print(f"Available collections: {list(collections.keys())}")
```

## Document Management for RAG

### 1. RAG Document Manager
```python
from datetime import datetime
import uuid

class WeaviateRAGManager:
    def __init__(self, weaviate_url="http://your-host:8080"):
        self.client = weaviate.Client(weaviate_url)
    
    def setup_rag_schema(self):
        """Setup optimized schema for RAG documents"""
        
        # Delete existing schema if needed (development only)
        try:
            existing_schema = self.client.schema.get()
            for class_obj in existing_schema.get('classes', []):
                if class_obj['class'] == 'Document':
                    self.client.schema.delete_class('Document')
        except:
            pass
        
        # Create document schema with auto-vectorization
        document_schema = {
            "class": "Document",
            "description": "RAG document chunks with auto-vectorization",
            "properties": [
                {
                    "name": "title",
                    "dataType": ["text"],
                    "description": "Document title"
                },
                {
                    "name": "content",
                    "dataType": ["text"], 
                    "description": "Document content (chunk)",
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "vectorizePropertyName": False
                        }
                    }
                },
                {
                    "name": "category",
                    "dataType": ["text"],
                    "description": "Document category"
                },
                {
                    "name": "source_url",
                    "dataType": ["text"],
                    "description": "Original source URL"
                },
                {
                    "name": "document_id",
                    "dataType": ["text"],
                    "description": "Original document identifier"
                },
                {
                    "name": "chunk_index",
                    "dataType": ["int"],
                    "description": "Chunk position in document"
                },
                {
                    "name": "word_count",
                    "dataType": ["int"],
                    "description": "Number of words in chunk"
                },
                {
                    "name": "created_at",
                    "dataType": ["date"],
                    "description": "Creation timestamp"
                }
            ],
            "vectorizer": "text2vec-transformers",
            "moduleConfig": {
                "text2vec-transformers": {
                    "poolingStrategy": "masked_mean",
                    "vectorizeClassName": True
                }
            }
        }
        
        self.client.schema.create_class(document_schema)
        return "Document"
    
    def add_document(self, title, content, category=None, source_url=None, 
                    document_id=None, chunk_index=0):
        """Add a document chunk to Weaviate"""
        
        if document_id is None:
            document_id = str(uuid.uuid4())
        
        # Prepare document data
        document_data = {
            "title": title,
            "content": content,
            "category": category or "uncategorized",
            "source_url": source_url or "",
            "document_id": document_id,
            "chunk_index": chunk_index,
            "word_count": len(content.split()),
            "created_at": datetime.now().isoformat()
        }
        
        # Add to Weaviate (auto-vectorization will occur)
        result = self.client.data_object.create(
            data_object=document_data,
            class_name="Document"
        )
        
        print(f"Added document chunk: {result}")
        return result
    
    def batch_add_documents(self, documents):
        """Efficiently add multiple documents using batch operations"""
        
        # Configure batch
        self.client.batch.configure(
            batch_size=100,
            dynamic=True,
            timeout_retries=3,
            callback=self.batch_callback
        )
        
        # Add documents to batch
        with self.client.batch as batch:
            for i, doc in enumerate(documents):
                document_data = {
                    "title": doc.get("title", f"Document {i}"),
                    "content": doc["content"],
                    "category": doc.get("category", "uncategorized"),
                    "source_url": doc.get("source_url", ""),
                    "document_id": doc.get("document_id", str(uuid.uuid4())),
                    "chunk_index": doc.get("chunk_index", 0),
                    "word_count": len(doc["content"].split()),
                    "created_at": datetime.now().isoformat()
                }
                
                batch.add_data_object(document_data, "Document")
        
        print(f"Batch added {len(documents)} documents")
    
    def batch_callback(self, results):
        """Callback for batch operations"""
        if results is not None:
            for result in results:
                if 'result' in result and 'errors' in result['result']:
                    if result['result']['errors']:
                        print(f"Batch error: {result['result']['errors']}")

# Initialize and setup
weaviate_manager = WeaviateRAGManager()
collection_name = weaviate_manager.setup_rag_schema()

# Add sample documents
sample_docs = [
    {
        "title": "Introduction to RAG",
        "content": "Retrieval-Augmented Generation combines information retrieval with text generation to produce more accurate and informative responses.",
        "category": "ai",
        "source_url": "https://example.com/rag-intro"
    },
    {
        "title": "Vector Databases Explained",
        "content": "Vector databases store high-dimensional vectors and enable similarity search for machine learning applications.",
        "category": "database",
        "source_url": "https://example.com/vector-db"
    },
    {
        "title": "Weaviate Features",
        "content": "Weaviate provides GraphQL interfaces, auto-vectorization, and hybrid search capabilities for modern AI applications.",
        "category": "technology",
        "source_url": "https://weaviate.io/features"
    }
]

weaviate_manager.batch_add_documents(sample_docs)
```

### 2. Semantic Search with GraphQL
```python
def semantic_search(self, query, limit=10, category_filter=None, min_certainty=0.7):
    """Perform semantic search using GraphQL"""
    
    # Build GraphQL query
    where_filter = {
        "operator": "And",
        "operands": []
    }
    
    # Add category filter if specified
    if category_filter:
        where_filter["operands"].append({
            "path": ["category"],
            "operator": "Equal",
            "valueText": category_filter
        })
    
    # Construct nearText query
    near_text = {
        "concepts": [query],
        "certainty": min_certainty
    }
    
    # Execute search
    result = (
        self.client.query
        .get("Document", ["title", "content", "category", "source_url", "document_id", "chunk_index"])
        .with_near_text(near_text)
        .with_limit(limit)
        .with_additional(["certainty", "distance"])
        .with_where(where_filter if where_filter["operands"] else None)
        .do()
    )
    
    # Format results
    documents = result.get("data", {}).get("Get", {}).get("Document", [])
    
    formatted_results = []
    for doc in documents:
        formatted_results.append({
            "title": doc["title"],
            "content": doc["content"],
            "category": doc["category"],
            "source_url": doc["source_url"],
            "document_id": doc["document_id"],
            "chunk_index": doc["chunk_index"],
            "certainty": doc["_additional"]["certainty"],
            "distance": doc["_additional"]["distance"]
        })
    
    return formatted_results

# Example searches
query = "What is vector similarity search?"

# Basic semantic search
results = weaviate_manager.semantic_search(query, limit=5)

print(f"Search results for: '{query}'")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['title']} (certainty: {result['certainty']:.3f})")
    print(f"   {result['content'][:100]}...")
    print(f"   Category: {result['category']}")

# Filtered search
tech_results = weaviate_manager.semantic_search(
    query, 
    limit=3, 
    category_filter="technology"
)
```

### 3. Hybrid Search (Vector + Keyword)
```python
def hybrid_search(self, query, alpha=0.7, limit=10):
    """Combine vector similarity with keyword search"""
    
    # Hybrid search with alpha parameter
    # alpha = 1.0: pure vector search
    # alpha = 0.0: pure keyword search  
    # alpha = 0.7: balanced hybrid (recommended)
    
    result = (
        self.client.query
        .get("Document", ["title", "content", "category", "source_url"])
        .with_hybrid(
            query=query,
            alpha=alpha  # Balance between vector (1.0) and keyword (0.0) search
        )
        .with_limit(limit)
        .with_additional(["score", "explainScore"])
        .do()
    )
    
    documents = result.get("data", {}).get("Get", {}).get("Document", [])
    
    formatted_results = []
    for doc in documents:
        formatted_results.append({
            "title": doc["title"],
            "content": doc["content"],
            "category": doc["category"],
            "source_url": doc["source_url"],
            "score": doc["_additional"]["score"],
            "explanation": doc["_additional"].get("explainScore", "")
        })
    
    return formatted_results

# Example hybrid search
hybrid_results = weaviate_manager.hybrid_search(
    "machine learning vector database", 
    alpha=0.7, 
    limit=5
)

print("Hybrid search results:")
for result in hybrid_results:
    print(f"- {result['title']} (score: {result['score']:.3f})")
    print(f"  {result['content'][:80]}...")
```

## GraphQL Playground Usage

### 1. Interactive Queries
```graphql
# Access GraphQL Playground at http://your-host:8080/v1/graphql

# Basic semantic search query
{
  Get {
    Document(
      nearText: {
        concepts: ["artificial intelligence"]
        certainty: 0.7
      }
      limit: 5
    ) {
      title
      content
      category
      _additional {
        certainty
        distance
      }
    }
  }
}

# Filtered search with conditions
{
  Get {
    Document(
      where: {
        path: ["category"]
        operator: Equal
        valueText: "ai"
      }
      nearText: {
        concepts: ["machine learning"]
      }
      limit: 3
    ) {
      title
      content
      source_url
    }
  }
}

# Aggregation query
{
  Aggregate {
    Document {
      meta {
        count
      }
      category {
        count
        topOccurrences {
          value
          occurs
        }
      }
    }
  }
}
```

### 2. Complex Relationship Queries
```graphql
# Multi-step semantic search
{
  Get {
    Document(
      nearText: {
        concepts: ["database performance"]
        certainty: 0.6
      }
    ) {
      title
      content
      _additional {
        certainty
      }
    }
  }
}

# Hybrid search via GraphQL
{
  Get {
    Document(
      hybrid: {
        query: "vector search optimization"
        alpha: 0.7
      }
      limit: 10
    ) {
      title
      content
      category
      _additional {
        score
        explainScore
      }
    }
  }
}
```

## Advanced Features

### 1. Multi-modal RAG with Images
```python
def setup_multimodal_schema(self):
    """Create schema for text + image documents"""
    
    multimodal_schema = {
        "class": "MultimodalDocument",
        "description": "Documents with text and images",
        "properties": [
            {
                "name": "title",
                "dataType": ["text"],
                "description": "Document title"
            },
            {
                "name": "content",
                "dataType": ["text"],
                "description": "Text content"
            },
            {
                "name": "image",
                "dataType": ["blob"],
                "description": "Associated image"
            },
            {
                "name": "image_description",
                "dataType": ["text"],
                "description": "Description of the image"
            }
        ],
        "vectorizer": "multi2vec-clip",  # CLIP model for text+image
        "moduleConfig": {
            "multi2vec-clip": {
                "textFields": ["title", "content", "image_description"],
                "imageFields": ["image"]
            }
        }
    }
    
    self.client.schema.create_class(multimodal_schema)
    return "MultimodalDocument"

def multimodal_search(self, text_query, image_query=None):
    """Search using both text and image"""
    
    near_media = {"concepts": [text_query]}
    if image_query:
        near_media["mediaObjects"] = [{"image": image_query}]
    
    result = (
        self.client.query
        .get("MultimodalDocument", ["title", "content", "image_description"])
        .with_near_text(near_media)
        .with_limit(10)
        .with_additional(["certainty"])
        .do()
    )
    
    return result.get("data", {}).get("Get", {}).get("MultimodalDocument", [])
```

### 2. Custom Vectorization
```python
def setup_custom_vector_schema(self):
    """Create schema with custom vectors (no auto-vectorization)"""
    
    custom_schema = {
        "class": "CustomDocument",
        "description": "Documents with custom embeddings",
        "properties": [
            {
                "name": "title",
                "dataType": ["text"],
                "description": "Document title"
            },
            {
                "name": "content",
                "dataType": ["text"],
                "description": "Document content"
            }
        ],
        "vectorizer": "none"  # No auto-vectorization
    }
    
    self.client.schema.create_class(custom_schema)
    
def add_document_with_custom_vector(self, title, content, custom_vector):
    """Add document with pre-computed embedding"""
    
    document_data = {
        "title": title,
        "content": content
    }
    
    result = self.client.data_object.create(
        data_object=document_data,
        class_name="CustomDocument",
        vector=custom_vector  # Provide custom embedding
    )
    
    return result
```

### 3. Real-time Updates and Deletion
```python
def update_document(self, document_uuid, updated_data):
    """Update existing document"""
    
    result = self.client.data_object.update(
        uuid=document_uuid,
        class_name="Document",
        data_object=updated_data
    )
    
    print(f"Updated document {document_uuid}")
    return result

def delete_document(self, document_uuid):
    """Delete document by UUID"""
    
    self.client.data_object.delete(
        uuid=document_uuid,
        class_name="Document"
    )
    
    print(f"Deleted document {document_uuid}")

def find_and_update_documents(self, category, update_data):
    """Find documents by criteria and update them"""
    
    # Find documents
    result = (
        self.client.query
        .get("Document", ["title"])
        .with_where({
            "path": ["category"],
            "operator": "Equal", 
            "valueText": category
        })
        .with_additional(["id"])
        .do()
    )
    
    documents = result.get("data", {}).get("Get", {}).get("Document", [])
    
    # Update each document
    for doc in documents:
        doc_uuid = doc["_additional"]["id"]
        self.update_document(doc_uuid, update_data)
    
    return len(documents)
```

## Monitoring and Analytics

### 1. Collection Statistics
```python
def get_collection_stats(self):
    """Get comprehensive collection statistics"""
    
    # Get aggregated stats
    result = (
        self.client.query
        .aggregate("Document")
        .with_meta_count()
        .with_fields("category { count topOccurrences { value occurs } }")
        .with_fields("word_count { count sum mean maximum minimum }")
        .do()
    )
    
    stats = result.get("data", {}).get("Aggregate", {}).get("Document", [])[0]
    
    formatted_stats = {
        "total_documents": stats["meta"]["count"],
        "categories": {
            "total_categories": stats["category"]["count"],
            "distribution": stats["category"]["topOccurrences"]
        },
        "content_stats": {
            "total_words": stats["word_count"]["sum"],
            "avg_words": stats["word_count"]["mean"],
            "max_words": stats["word_count"]["maximum"],
            "min_words": stats["word_count"]["minimum"]
        }
    }
    
    return formatted_stats

# Get and display stats
stats = weaviate_manager.get_collection_stats()
print(f"Total documents: {stats['total_documents']}")
print(f"Categories: {stats['categories']['total_categories']}")
for cat in stats['categories']['distribution']:
    print(f"  {cat['value']}: {cat['occurs']} documents")
```

### 2. Performance Monitoring
```python
def monitor_query_performance(self, query_func, *args, **kwargs):
    """Monitor query execution time"""
    
    import time
    
    start_time = time.time()
    result = query_func(*args, **kwargs)
    execution_time = time.time() - start_time
    
    print(f"Query executed in {execution_time:.3f} seconds")
    print(f"Returned {len(result) if isinstance(result, list) else 'N/A'} results")
    
    return result, execution_time

# Example usage
results, exec_time = weaviate_manager.monitor_query_performance(
    weaviate_manager.semantic_search,
    "machine learning",
    limit=10
)
```

## Production Configuration

### 1. Authentication Setup
```python
# For production, enable authentication
client = weaviate.Client(
    url="http://your-host:8080",
    auth_client_secret=weaviate.AuthApiKey("your-api-key")
)
```

### 2. Backup and Restore
```python
def backup_schema_and_data(self):
    """Backup Weaviate schema and data"""
    
    import json
    
    # Backup schema
    schema = self.client.schema.get()
    with open("weaviate_schema_backup.json", "w") as f:
        json.dump(schema, f, indent=2)
    
    # Backup data (for small datasets)
    result = (
        self.client.query
        .get("Document")
        .with_additional(["id", "vector"])
        .do()
    )
    
    with open("weaviate_data_backup.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("Backup completed")

def restore_from_backup(self, schema_file, data_file):
    """Restore schema and data from backup"""
    
    import json
    
    # Restore schema
    with open(schema_file, "r") as f:
        schema = json.load(f)
    
    for class_def in schema["classes"]:
        try:
            self.client.schema.create_class(class_def)
            print(f"Restored class: {class_def['class']}")
        except Exception as e:
            print(f"Failed to restore class {class_def['class']}: {e}")
    
    # Restore data
    with open(data_file, "r") as f:
        data = json.load(f)
    
    documents = data.get("data", {}).get("Get", {}).get("Document", [])
    
    for doc in documents:
        vector = doc["_additional"].pop("vector")
        doc_id = doc["_additional"].pop("id")
        
        try:
            self.client.data_object.create(
                data_object=doc,
                class_name="Document",
                uuid=doc_id,
                vector=vector
            )
        except Exception as e:
            print(f"Failed to restore document {doc_id}: {e}")
    
    print(f"Restored {len(documents)} documents")
```

## Troubleshooting

### Common Issues

**"Schema class not found"**
```python
# Check existing schema
schema = client.schema.get()
classes = [cls['class'] for cls in schema.get('classes', [])]
print(f"Available classes: {classes}")

# Create missing class
if "Document" not in classes:
    weaviate_manager.setup_rag_schema()
```

**"Vectorization fails"**
```python
# Check vectorizer configuration
schema = client.schema.get("Document")
vectorizer = schema.get('vectorizer')
print(f"Configured vectorizer: {vectorizer}")

# Test vectorization
test_result = client.data_object.create(
    data_object={"content": "test document"},
    class_name="Document"
)
print(f"Test document created: {test_result}")
```

**"GraphQL syntax errors"**
```bash
# Use GraphQL playground to test queries
# Check query syntax at http://your-host:8080/v1/graphql
# Enable query debugging
```

**"Poor search results"**
```python
# Adjust certainty threshold
results = client.query.get("Document").with_near_text({
    "concepts": ["your query"],
    "certainty": 0.5  # Lower threshold for more results
}).do()

# Try hybrid search for better relevance
hybrid_results = client.query.get("Document").with_hybrid(
    query="your query",
    alpha=0.7
).do()
```

## Learning Resources

- [Weaviate Documentation](https://weaviate.io/developers/weaviate/) - Official documentation
- [GraphQL Tutorial](https://weaviate.io/developers/weaviate/api/graphql) - Complete GraphQL reference
- [Vector Search Guide](https://weaviate.io/developers/weaviate/concepts/vector-index) - Understanding vector operations
- [Hybrid Search Explained](https://weaviate.io/developers/weaviate/api/graphql/search-operators#hybrid) - Combining keyword and vector search