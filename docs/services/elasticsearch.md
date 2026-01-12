# Elasticsearch & Kibana Guide

Elasticsearch provides powerful full-text search and analytics, while Kibana offers visualization and management. Essential for hybrid RAG systems combining keyword and vector search.

## Quick Access

| Component | URL | Purpose |
|-----------|-----|---------|
| **Elasticsearch API** | `http://your-host:9200` | REST API for search operations |
| **Kibana Dashboard** | `http://your-host:5601` | Web interface for data exploration |
| **Cluster Health** | `http://your-host:9200/_cluster/health` | Cluster status endpoint |

## Why Elasticsearch for RAG?

### Search Capabilities
- **Full-Text Search**: Advanced text analysis and keyword matching
- **Hybrid Search**: Combine with vector databases for best results
- **Faceted Search**: Filter by multiple criteria simultaneously
- **Analytics**: Aggregations and statistical analysis of search patterns
- **Real-time**: Near real-time search and indexing

### RAG Use Cases
- **Keyword Retrieval**: Traditional text search for exact matches
- **Metadata Filtering**: Filter documents by categories, dates, sources
- **Search Analytics**: Track user queries and result quality
- **Document Classification**: Automatic categorization of content
- **Hybrid RAG**: Combine with vector search for comprehensive retrieval

## Initial Setup

### 1. Verify Elasticsearch
```bash
# Check Elasticsearch health
curl http://your-host:9200/_cluster/health

# Get cluster information
curl http://your-host:9200/

# Access Kibana
open http://your-host:5601
```

### 2. Connect with Python
```python
from elasticsearch import Elasticsearch
import json

# Initialize Elasticsearch client
es = Elasticsearch([{'host': 'your-host', 'port': 9200, 'scheme': 'http'}])

# Test connection
if es.ping():
    print("Connected to Elasticsearch")
else:
    print("Could not connect to Elasticsearch")

# Get cluster info
info = es.info()
print(f"Elasticsearch version: {info['version']['number']}")
```

### 3. Create Your First Index
```python
# Define mapping for RAG documents
rag_mapping = {
    "mappings": {
        "properties": {
            "title": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            },
            "content": {
                "type": "text",
                "analyzer": "standard"
            },
            "category": {
                "type": "keyword"
            },
            "source": {
                "type": "keyword"
            },
            "tags": {
                "type": "keyword"
            },
            "created_at": {
                "type": "date",
                "format": "strict_date_optional_time||epoch_millis"
            },
            "document_id": {
                "type": "keyword"
            },
            "chunk_index": {
                "type": "integer"
            },
            "word_count": {
                "type": "integer"
            },
            "vector_score": {
                "type": "float"
            }
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "custom_analyzer": {
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "stop",
                        "snowball"
                    ]
                }
            }
        }
    }
}

# Create index
index_name = "rag_documents"
es.indices.create(index=index_name, body=rag_mapping, ignore=400)
print(f"Created index: {index_name}")
```

## Document Management for RAG

### 1. RAG Document Manager
```python
from datetime import datetime
import uuid

class ElasticsearchRAGManager:
    def __init__(self, host="your-host", port=9200):
        self.es = Elasticsearch([{'host': host, 'port': port, 'scheme': 'http'}])
        self.index_name = "rag_documents"
    
    def setup_rag_index(self):
        """Create optimized index for RAG documents"""
        
        mapping = {
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "standard",
                        "term_vector": "with_positions_offsets"
                    },
                    "category": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "document_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "word_count": {"type": "integer"},
                    "embedding_model": {"type": "keyword"},
                    "vector_score": {"type": "float"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "max_result_window": 50000,
                "analysis": {
                    "analyzer": {
                        "content_analyzer": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"]
                        }
                    }
                }
            }
        }
        
        # Delete and recreate index (development only)
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
        
        self.es.indices.create(index=self.index_name, body=mapping)
        return self.index_name
    
    def add_document(self, title, content, category=None, source=None, 
                    tags=None, document_id=None, chunk_index=0, vector_score=None):
        """Add a document to Elasticsearch"""
        
        if document_id is None:
            document_id = str(uuid.uuid4())
        
        doc = {
            "title": title,
            "content": content,
            "category": category or "uncategorized",
            "source": source or "unknown",
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "document_id": document_id,
            "chunk_index": chunk_index,
            "word_count": len(content.split()),
            "vector_score": vector_score
        }
        
        # Index document
        result = self.es.index(
            index=self.index_name,
            body=doc,
            refresh='wait_for'  # Make immediately searchable
        )
        
        print(f"Indexed document: {result['_id']}")
        return result['_id']
    
    def bulk_add_documents(self, documents):
        """Efficiently add multiple documents using bulk API"""
        
        from elasticsearch.helpers import bulk
        
        actions = []
        for doc in documents:
            action = {
                "_index": self.index_name,
                "_source": {
                    "title": doc.get("title", ""),
                    "content": doc["content"],
                    "category": doc.get("category", "uncategorized"),
                    "source": doc.get("source", "unknown"),
                    "tags": doc.get("tags", []),
                    "created_at": datetime.now().isoformat(),
                    "document_id": doc.get("document_id", str(uuid.uuid4())),
                    "chunk_index": doc.get("chunk_index", 0),
                    "word_count": len(doc["content"].split())
                }
            }
            actions.append(action)
        
        # Bulk index
        success, failed = bulk(self.es, actions)
        print(f"Bulk indexed {success} documents, {len(failed)} failed")
        
        # Refresh index
        self.es.indices.refresh(index=self.index_name)
        return success, failed

# Initialize manager
es_manager = ElasticsearchRAGManager()
index_name = es_manager.setup_rag_index()

# Add sample documents
sample_docs = [
    {
        "title": "Introduction to Elasticsearch",
        "content": "Elasticsearch is a distributed, RESTful search and analytics engine capable of solving a growing number of use cases.",
        "category": "search",
        "source": "documentation",
        "tags": ["elasticsearch", "search", "analytics"]
    },
    {
        "title": "Full-Text Search Basics",
        "content": "Full-text search involves searching through the complete text of documents to find relevant matches based on keywords and phrases.",
        "category": "search",
        "source": "tutorial",
        "tags": ["full-text", "search", "keywords"]
    },
    {
        "title": "RAG with Hybrid Search",
        "content": "Combining vector similarity search with traditional keyword search provides the best of both semantic and exact matching.",
        "category": "ai",
        "source": "research",
        "tags": ["rag", "hybrid", "vector", "keyword"]
    }
]

es_manager.bulk_add_documents(sample_docs)
```

### 2. Advanced Search Operations
```python
def keyword_search(self, query, filters=None, limit=10, highlight=True):
    """Perform keyword-based search with optional filters"""
    
    # Build query
    search_body = {
        "size": limit,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^2", "content", "tags"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    }
                ]
            }
        },
        "sort": [
            "_score",
            {"created_at": {"order": "desc"}}
        ]
    }
    
    # Add filters
    if filters:
        filter_clauses = []
        for field, value in filters.items():
            if isinstance(value, list):
                filter_clauses.append({"terms": {field: value}})
            else:
                filter_clauses.append({"term": {field: value}})
        
        search_body["query"]["bool"]["filter"] = filter_clauses
    
    # Add highlighting
    if highlight:
        search_body["highlight"] = {
            "fields": {
                "title": {},
                "content": {
                    "fragment_size": 150,
                    "number_of_fragments": 3
                }
            },
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"]
        }
    
    # Execute search
    result = self.es.search(index=self.index_name, body=search_body)
    
    # Format results
    hits = []
    for hit in result['hits']['hits']:
        formatted_hit = {
            "id": hit['_id'],
            "score": hit['_score'],
            "title": hit['_source']['title'],
            "content": hit['_source']['content'],
            "category": hit['_source']['category'],
            "source": hit['_source']['source'],
            "tags": hit['_source']['tags'],
            "created_at": hit['_source']['created_at']
        }
        
        # Add highlights if available
        if 'highlight' in hit:
            formatted_hit['highlights'] = {
                "title": hit['highlight'].get('title', []),
                "content": hit['highlight'].get('content', [])
            }
        
        hits.append(formatted_hit)
    
    return {
        "total": result['hits']['total']['value'],
        "max_score": result['hits']['max_score'],
        "hits": hits
    }

def faceted_search(self, query=None, facet_fields=None, limit=10):
    """Search with faceted navigation"""
    
    if facet_fields is None:
        facet_fields = ["category", "source", "tags"]
    
    # Build search body
    search_body = {
        "size": limit,
        "query": {
            "match_all": {}
        } if not query else {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "content"],
                "fuzziness": "AUTO"
            }
        },
        "aggs": {}
    }
    
    # Add aggregations for facets
    for field in facet_fields:
        search_body["aggs"][f"{field}_facet"] = {
            "terms": {
                "field": field,
                "size": 20
            }
        }
    
    # Execute search
    result = self.es.search(index=self.index_name, body=search_body)
    
    # Format facets
    facets = {}
    for facet_name in facet_fields:
        facet_key = f"{facet_name}_facet"
        if facet_key in result['aggregations']:
            facets[facet_name] = [
                {"value": bucket['key'], "count": bucket['doc_count']}
                for bucket in result['aggregations'][facet_key]['buckets']
            ]
    
    return {
        "hits": [hit['_source'] for hit in result['hits']['hits']],
        "facets": facets,
        "total": result['hits']['total']['value']
    }

# Example searches
query = "vector search database"

# Basic keyword search
results = es_manager.keyword_search(query, limit=5)
print(f"Found {results['total']} documents")
for hit in results['hits']:
    print(f"- {hit['title']} (score: {hit['score']:.2f})")
    if 'highlights' in hit and hit['highlights']['content']:
        print(f"  ...{hit['highlights']['content'][0]}...")

# Filtered search
filtered_results = es_manager.keyword_search(
    query,
    filters={"category": "ai", "tags": ["vector"]},
    limit=3
)

# Faceted search
faceted_results = es_manager.faceted_search(query)
print("\nFacets:")
for facet_name, facet_values in faceted_results['facets'].items():
    print(f"{facet_name}: {facet_values}")
```

### 3. Hybrid RAG Implementation
```python
def hybrid_rag_search(self, query, vector_results=None, 
                     keyword_weight=0.3, vector_weight=0.7, limit=10):
    """Combine Elasticsearch keyword search with vector search results"""
    
    # Perform keyword search
    keyword_results = self.keyword_search(query, limit=limit*2)
    
    # If vector results provided, combine scores
    if vector_results:
        # Create lookup for vector scores
        vector_scores = {doc.get('id', doc.get('_id', str(i))): doc.get('score', 0) 
                        for i, doc in enumerate(vector_results)}
        
        # Normalize scores
        max_keyword_score = max([hit['score'] for hit in keyword_results['hits']], default=1)
        max_vector_score = max(vector_scores.values(), default=1)
        
        combined_results = []
        seen_ids = set()
        
        # Score keyword results
        for hit in keyword_results['hits']:
            hit_id = hit['id']
            if hit_id in seen_ids:
                continue
            
            keyword_score = hit['score'] / max_keyword_score
            vector_score = vector_scores.get(hit_id, 0) / max_vector_score
            
            combined_score = (keyword_weight * keyword_score + 
                            vector_weight * vector_score)
            
            hit['combined_score'] = combined_score
            hit['keyword_score'] = keyword_score
            hit['vector_score'] = vector_score
            
            combined_results.append(hit)
            seen_ids.add(hit_id)
        
        # Add vector-only results
        for doc in vector_results:
            doc_id = doc.get('id', doc.get('_id'))
            if doc_id not in seen_ids:
                vector_score = doc.get('score', 0) / max_vector_score
                combined_score = vector_weight * vector_score
                
                # Create minimal hit structure
                hit = {
                    'id': doc_id,
                    'combined_score': combined_score,
                    'keyword_score': 0,
                    'vector_score': vector_score,
                    'title': doc.get('title', ''),
                    'content': doc.get('content', ''),
                    'category': doc.get('category', ''),
                    'source': doc.get('source', '')
                }
                combined_results.append(hit)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return combined_results[:limit]
    
    else:
        return keyword_results['hits']
```

## Kibana Dashboard Usage

### 1. Index Management
```bash
# Access Kibana at http://your-host:5601

Navigation:
1. Stack Management → Index Management
   - View all indices
   - Monitor index health
   - Manage index settings

2. Stack Management → Index Patterns
   - Create index patterns for visualization
   - Configure field mappings
   - Set up time fields

3. Discover
   - Explore your data
   - Create saved searches
   - Filter and query documents
```

### 2. Creating Visualizations
```bash
# Kibana Visualizations for RAG Analytics:

1. Document Count Over Time
   - Visualization Type: Line Chart
   - X-axis: Date Histogram on 'created_at'
   - Y-axis: Count of documents

2. Category Distribution
   - Visualization Type: Pie Chart
   - Buckets: Terms aggregation on 'category'

3. Search Query Analysis
   - Create index for search logs
   - Track popular queries
   - Monitor search performance

4. Content Analytics
   - Word count distribution
   - Document length analysis
   - Tag frequency
```

### 3. Dashboard Creation
```json
// Example dashboard configuration
{
  "title": "RAG Analytics Dashboard",
  "panels": [
    {
      "title": "Document Count Over Time",
      "type": "line",
      "query": {
        "index": "rag_documents",
        "aggregations": {
          "docs_over_time": {
            "date_histogram": {
              "field": "created_at",
              "calendar_interval": "day"
            }
          }
        }
      }
    },
    {
      "title": "Top Categories",
      "type": "pie",
      "query": {
        "index": "rag_documents",
        "aggregations": {
          "categories": {
            "terms": {
              "field": "category",
              "size": 10
            }
          }
        }
      }
    }
  ]
}
```

## Analytics and Monitoring

### 1. Search Analytics
```python
def track_search_query(self, user_query, results_count, user_id=None):
    """Track search queries for analytics"""
    
    search_log = {
        "query": user_query,
        "results_count": results_count,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "query_length": len(user_query),
        "query_words": len(user_query.split())
    }
    
    # Index to search analytics
    self.es.index(
        index="search_analytics",
        body=search_log,
        refresh='wait_for'
    )

def get_search_analytics(self, days=7):
    """Get search analytics for the last N days"""
    
    search_body = {
        "size": 0,
        "query": {
            "range": {
                "timestamp": {
                    "gte": f"now-{days}d/d",
                    "lte": "now/d"
                }
            }
        },
        "aggs": {
            "popular_queries": {
                "terms": {
                    "field": "query.keyword",
                    "size": 10
                }
            },
            "searches_over_time": {
                "date_histogram": {
                    "field": "timestamp",
                    "calendar_interval": "day"
                }
            },
            "avg_results_count": {
                "avg": {
                    "field": "results_count"
                }
            }
        }
    }
    
    result = self.es.search(index="search_analytics", body=search_body)
    
    return {
        "total_searches": result['hits']['total']['value'],
        "popular_queries": result['aggregations']['popular_queries']['buckets'],
        "searches_per_day": result['aggregations']['searches_over_time']['buckets'],
        "avg_results": result['aggregations']['avg_results_count']['value']
    }
```

### 2. Performance Monitoring
```python
def monitor_index_performance(self):
    """Monitor index performance metrics"""
    
    # Get index stats
    stats = self.es.indices.stats(index=self.index_name)
    
    index_stats = {
        "total_docs": stats['_all']['total']['docs']['count'],
        "index_size_bytes": stats['_all']['total']['store']['size_in_bytes'],
        "search_time_ms": stats['_all']['total']['search']['time_in_millis'],
        "search_count": stats['_all']['total']['search']['query_total'],
        "indexing_time_ms": stats['_all']['total']['indexing']['time_in_millis'],
        "indexing_count": stats['_all']['total']['indexing']['index_total']
    }
    
    return index_stats

def optimize_index(self):
    """Optimize index performance"""
    
    # Force merge segments
    self.es.indices.forcemerge(index=self.index_name, max_num_segments=1)
    
    # Refresh index
    self.es.indices.refresh(index=self.index_name)
    
    # Clear cache
    self.es.indices.clear_cache(index=self.index_name)
    
    print(f"Optimized index: {self.index_name}")
```

## Production Configuration

### 1. Security Configuration
```yaml
# elasticsearch.yml configuration
xpack.security.enabled: true
xpack.security.transport.ssl.enabled: true
xpack.security.http.ssl.enabled: true

# Create users and roles
elasticsearch-setup-passwords auto
```

### 2. Performance Tuning
```yaml
# Elasticsearch settings for production
cluster.name: "rag-cluster"
node.name: "rag-node-1"
path.data: /usr/share/elasticsearch/data
path.logs: /usr/share/elasticsearch/logs
network.host: 0.0.0.0
http.port: 9200
discovery.type: single-node

# Memory settings
-Xms2g
-Xmx2g

# Index settings
index.number_of_shards: 1
index.number_of_replicas: 0
index.refresh_interval: 5s
```

### 3. Backup and Restore
```python
def create_snapshot_repository(self):
    """Setup snapshot repository for backups"""
    
    repository_settings = {
        "type": "fs",
        "settings": {
            "location": "/usr/share/elasticsearch/snapshots",
            "compress": True
        }
    }
    
    self.es.snapshot.create_repository(
        repository="backup_repo",
        body=repository_settings
    )

def create_backup(self, snapshot_name=None):
    """Create index snapshot"""
    
    if not snapshot_name:
        snapshot_name = f"rag_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    snapshot_body = {
        "indices": self.index_name,
        "ignore_unavailable": True,
        "include_global_state": False
    }
    
    result = self.es.snapshot.create(
        repository="backup_repo",
        snapshot=snapshot_name,
        body=snapshot_body
    )
    
    return snapshot_name
```

## Troubleshooting

### Common Issues

**"Index not found"**
```python
# Check if index exists
if not es.indices.exists(index="rag_documents"):
    print("Index does not exist")
    es_manager.setup_rag_index()
```

**"Mapping conflicts"**
```python
# Check current mapping
mapping = es.indices.get_mapping(index="rag_documents")
print(json.dumps(mapping, indent=2))

# Update mapping for new fields
new_field_mapping = {
    "properties": {
        "new_field": {
            "type": "text"
        }
    }
}
es.indices.put_mapping(index="rag_documents", body=new_field_mapping)
```

**"Poor search performance"**
```python
# Check index stats
stats = es.indices.stats(index="rag_documents")
print(f"Segments: {stats['_all']['total']['segments']['count']}")

# Optimize if too many segments
if stats['_all']['total']['segments']['count'] > 5:
    es.indices.forcemerge(index="rag_documents", max_num_segments=1)
```

**"Kibana connection issues"**
```bash
# Check Elasticsearch connectivity from Kibana
curl -X GET "kibana:5601/api/status"

# Verify Kibana configuration
cat /usr/share/kibana/config/kibana.yml
```

## Learning Resources

- [Elasticsearch Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html) - Official documentation
- [Kibana User Guide](https://www.elastic.co/guide/en/kibana/current/index.html) - Complete Kibana reference
- [Search API Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-search.html) - Query DSL reference
- [Performance Tuning](https://www.elastic.co/guide/en/elasticsearch/reference/current/tune-for-search-speed.html) - Optimization guide