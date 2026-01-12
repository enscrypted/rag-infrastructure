# MongoDB Vector Search Guide

MongoDB 8.2+ includes native vector search capabilities, making it an excellent choice for RAG applications without requiring MongoDB Atlas.

## Overview

This deployment uses MongoDB Community Edition 8.2 with:
- Replica set for document operations
- No authentication (simplified for development)
- Multiple UI options for management

### Important: Vector Search Requirements

**MongoDB Community Edition 8.2** supports `$vectorSearch` but requires **mongot** (MongoDB's search binary) running alongside `mongod`.

| Setup | Vector Search Support |
|-------|----------------------|
| MongoDB Atlas | ✅ Built-in |
| MongoDB Community + mongot | ✅ Requires `mongodb-community-search` container |
| MongoDB Community (basic) | ❌ Use cosine similarity aggregation instead |

**For this deployment**: If you need `$vectorSearch`, add the `mongodb/mongodb-community-search` container to your docker-compose. Otherwise, use the manual cosine similarity approach shown below, which works without mongot.

## Vector Search Setup

### Creating a Vector Index

```python
# Using PyMongo driver (recommended for 2025+)
from pymongo.operations import SearchIndexModel

# Define the vector search index
index_definition = {
    "fields": [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 1536,  # Match your embedding model dimensions
            "similarity": "cosine"  # Options: "cosine", "euclidean", "dotProduct"
        }
    ]
}

# Create the index model
model = SearchIndexModel(
    definition=index_definition,
    name="vector_index",
    type="vectorSearch"
)

# Create the search index
collection.create_search_indexes([model])

# For MongoDB shell (alternative)
# db.documents.createSearchIndex({
#   "type": "vectorSearch",
#   "definition": {
#     "fields": [{
#       "type": "vector",
#       "path": "embedding",
#       "numDimensions": 1536,
#       "similarity": "cosine"
#     }]
#   },
#   "name": "vector_index"
# })
```

### Performing Vector Search

```python
from pymongo import MongoClient

client = MongoClient('mongodb://your-host:27017/')
db = client.rag_database
collection = db.documents

# Vector search aggregation pipeline (MongoDB 8.2+ syntax)
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": your_query_embedding,  # Your embedding vector
            "path": "embedding",  # Field containing document embeddings
            "numCandidates": 100,  # Documents to consider (10-20x limit)
            "limit": 10,  # Final results to return
            "exact": False  # False for ANN (fast), True for ENN (exact)
        }
    },
    {
        "$addFields": {
            "score": {"$meta": "vectorSearchScore"}  # Add similarity score
        }
    },
    {
        "$project": {
            "content": 1,
            "metadata": 1,
            "score": 1,
            "_id": 1  # Include or exclude as needed
        }
    }
]

results = list(collection.aggregate(pipeline))

# For filtered vector search (MongoDB 8.2+)
pipeline_with_filter = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": your_query_embedding,
            "path": "embedding",
            "numCandidates": 100,
            "limit": 10,
            "filter": {  # Pre-filter documents
                "metadata.category": "technology",
                "metadata.date": {"$gte": "2024-01-01"}
            }
        }
    }
]
```

## Document Structure

Recommended document structure for RAG:

```python
document = {
    "_id": ObjectId(),
    "content": "The actual text content",
    "metadata": {
        "source": "document.pdf",
        "page": 1,
        "section": "Introduction"
    },
    "embedding": [0.1, -0.2, 0.3, ...],  # 1536-dimensional vector
    "created_at": datetime.utcnow()
}
```

## Available UIs

### 1. Vector Search UI (Custom)
- URL: `http://your-host:8090`
- Purpose: Specialized for vector operations
- Features: Vector search, index management, document insertion

### 2. Mongoku (Modern)
- URL: `http://your-host:3100`
- Purpose: General MongoDB management
- Features: Modern UI, query builder, data visualization

### 3. Mongo Express (Traditional)
- URL: `http://your-host:8081`
- Purpose: Basic admin interface
- Credentials: admin/your-password

## Common Operations

### Inserting Documents with Embeddings

```python
import openai
from pymongo import MongoClient

client = MongoClient('mongodb://your-host:27017/')
collection = client.rag_database.documents

# Generate embedding
def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Insert document
doc = {
    "content": "MongoDB supports vector search natively",
    "metadata": {"source": "docs"},
    "embedding": get_embedding("MongoDB supports vector search natively")
}

collection.insert_one(doc)
```

### Hybrid Search (Text + Vector)

```python
# Hybrid search combining vector similarity and text matching (MongoDB 8.2+)
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 200,  # Increase candidates for hybrid search
            "limit": 50  # Get more initial results for filtering
        }
    },
    {
        "$addFields": {
            "vector_score": {"$meta": "vectorSearchScore"}
        }
    },
    {
        "$match": {
            "content": {"$regex": keyword, "$options": "i"}  # Case-insensitive text match
        }
    },
    {
        "$addFields": {
            "text_match_boost": {
                "$cond": [
                    {"$regexMatch": {
                        "input": "$content",
                        "regex": keyword,
                        "options": "i"
                    }},
                    0.2,  # Boost score if text matches
                    0
                ]
            }
        }
    },
    {
        "$addFields": {
            "combined_score": {
                "$add": ["$vector_score", "$text_match_boost"]
            }
        }
    },
    {
        "$sort": {"combined_score": -1}
    },
    {
        "$limit": 10
    }
]

results = list(collection.aggregate(pipeline))
```

## Best Practices

### Indexing Strategy
- Create vector indexes on collections with substantial data
- Use appropriate `numCandidates` (typically 10-20x your limit)
- Consider compound indexes for filtered vector search

### Performance Tips
- Use projection to limit returned fields
- Implement pagination for large result sets
- Monitor query performance with `explain()`

### Data Management
- Implement proper error handling for index creation
- Use bulk operations for inserting many documents
- Consider TTL indexes for temporary data

## Alternative: Manual Cosine Similarity (No mongot Required)

If you don't have mongot configured, use this aggregation pipeline for vector search:

```python
import math
from pymongo import MongoClient

client = MongoClient('mongodb://your-host:27017/', directConnection=True)
db = client.rag_database
collection = db.documents

def cosine_similarity_search(query_vector, collection, limit=10):
    """
    Manual cosine similarity search using MongoDB aggregation.
    Works without mongot/Atlas Search.
    """
    # Calculate magnitude of query vector
    query_magnitude = math.sqrt(sum(x**2 for x in query_vector))

    pipeline = [
        {
            "$addFields": {
                # Dot product: sum of element-wise multiplication
                "dot_product": {
                    "$reduce": {
                        "input": {"$range": [0, {"$size": "$embedding"}]},
                        "initialValue": 0,
                        "in": {
                            "$add": [
                                "$$value",
                                {"$multiply": [
                                    {"$arrayElemAt": ["$embedding", "$$this"]},
                                    {"$arrayElemAt": [query_vector, "$$this"]}
                                ]}
                            ]
                        }
                    }
                },
                # Magnitude of document embedding
                "doc_magnitude": {
                    "$sqrt": {
                        "$reduce": {
                            "input": "$embedding",
                            "initialValue": 0,
                            "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
                        }
                    }
                }
            }
        },
        {
            "$addFields": {
                "cosine_similarity": {
                    "$cond": [
                        {"$eq": ["$doc_magnitude", 0]},
                        0,
                        {"$divide": ["$dot_product", {"$multiply": ["$doc_magnitude", query_magnitude]}]}
                    ]
                }
            }
        },
        {"$sort": {"cosine_similarity": -1}},
        {"$limit": limit},
        {"$project": {"content": 1, "metadata": 1, "cosine_similarity": 1}}
    ]

    return list(collection.aggregate(pipeline))

# Usage
results = cosine_similarity_search(query_embedding, collection, limit=5)
for doc in results:
    print(f"Score: {doc['cosine_similarity']:.4f} - {doc['content'][:50]}...")
```

> **Performance Note**: This approach works for small to medium collections (up to ~100K documents). For larger collections, consider adding mongot or using a dedicated vector database like ChromaDB or Qdrant.

## Troubleshooting

### Vector Search Not Working
```bash
# Check replica set status
docker exec rag-mongodb mongosh --eval "rs.status()"

# Reinitialize if needed
docker exec rag-mongodb mongosh --eval "rs.initiate()"
```

### $vectorSearch Returns Error
If you see errors like "requires additional configuration" or "mongot not running":
- **Option 1**: Add the `mongodb/mongodb-community-search` container
- **Option 2**: Use the manual cosine similarity approach above

### Index Issues
```javascript
// List all indexes
db.documents.getIndexes()

// Drop and recreate vector index
db.documents.dropSearchIndex("vector_index")
// Then recreate with createSearchIndex
```

### Performance Problems
```javascript
// Check query execution
db.documents.aggregate(pipeline).explain("executionStats")

// Monitor current operations
db.currentOp()
```

## Resources

- [MongoDB Vector Search Documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [Vector Search Tutorial](https://www.mongodb.com/developer/products/atlas/vector-search-tutorial/)
- [RAG with MongoDB Blog](https://www.mongodb.com/developer/products/atlas/rag-with-mongodb-and-openai/)