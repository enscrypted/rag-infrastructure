# Neo4j Graph Database Guide

Neo4j is a leading graph database that excels at storing and querying highly connected data. In RAG applications, it's perfect for knowledge graphs, entity relationships, and graph-enhanced retrieval.

## Quick Access

| Component | URL | Credentials |
|-----------|-----|-------------|
| **Neo4j Browser** | `http://your-host:7474` | neo4j / your-password |
| **Bolt Protocol** | `bolt://your-host:7687` | neo4j / your-password |

## Why Neo4j for RAG?

### Graph-Enhanced RAG
- **Entity Relationships**: Model how concepts, people, and ideas connect
- **Path Queries**: Find indirect relationships between entities
- **Context Expansion**: Traverse graphs to find related information
- **Hybrid Retrieval**: Combine vector similarity with graph traversal

### Use Cases
- **Knowledge Base**: Company knowledge with relationships
- **Research Papers**: Authors, citations, topics, institutions
- **Product Catalogs**: Categories, features, recommendations
- **Legal Documents**: Cases, statutes, precedents

## Initial Setup

### 1. First Login
```bash
# Access Neo4j Browser
open http://your-host:7474

# Connect using:
# Connect URL: bolt://localhost:7687
# Username: neo4j
# Password: your-configured-password
```

### 2. Verify Installation
```cypher
-- Check Neo4j version and installed procedures
CALL dbms.procedures() YIELD name 
WHERE name STARTS WITH 'apoc' OR name STARTS WITH 'gds'
RETURN count(name) as procedure_count;

-- Should return procedures if APOC and GDS are installed
```

### 3. Basic Graph Creation
```cypher
-- Create sample knowledge graph
CREATE (ai:Topic {name: "Artificial Intelligence", category: "Technology"})
CREATE (ml:Topic {name: "Machine Learning", category: "Technology"})  
CREATE (rag:Topic {name: "RAG", category: "Technology"})
CREATE (python:Topic {name: "Python", category: "Programming"})

CREATE (ai)-[:INCLUDES]->(ml)
CREATE (ml)-[:ENABLES]->(rag)
CREATE (rag)-[:IMPLEMENTED_IN]->(python)

RETURN ai, ml, rag, python;
```

## RAG Integration Patterns

### 1. Entity Extraction and Storage

```python
from neo4j import GraphDatabase
import spacy

class GraphRAG:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.nlp = spacy.load("en_core_web_sm")
    
    def extract_and_store_entities(self, document_text, doc_id):
        """Extract entities and relationships from text"""
        doc = self.nlp(document_text)
        
        # Create document node using execute_query (recommended in 2025+)
        self.driver.execute_query(
            """
            MERGE (d:Document {id: $doc_id})
            SET d.content = $content
            """, 
            doc_id=doc_id, 
            content=document_text,
            database_="neo4j"  # Always specify database for efficiency
        )
        
        # Extract and create entity nodes
        for ent in doc.ents:
            self.driver.execute_query(
                """
                MERGE (e:Entity {name: $name, type: $type})
                MERGE (d:Document {id: $doc_id})
                MERGE (d)-[:MENTIONS]->(e)
                """, 
                name=ent.text, 
                type=ent.label_, 
                doc_id=doc_id,
                database_="neo4j"
            )
    
    def close(self):
        """Close the driver connection"""
        self.driver.close()

# Usage with context manager for proper cleanup
with GraphDatabase.driver("bolt://your-host:7687", auth=("neo4j", "your-password")) as driver:
    graph_rag = GraphRAG("bolt://your-host:7687", "neo4j", "your-password")
    graph_rag.extract_and_store_entities("Python is used for AI development", "doc_1")
    graph_rag.close()
```

### 2. Graph-Enhanced Retrieval

```python
def graph_enhanced_search(self, query_entities, max_depth=2):
    """Find documents connected to query entities"""
    # Using execute_query for better transaction management
    records, summary, keys = self.driver.execute_query(
        """
        MATCH (e:Entity)
        WHERE e.name IN $entities
        MATCH (e)<-[:MENTIONS*1..$max_depth]-(d:Document)
        RETURN DISTINCT d.id as doc_id, d.content as content,
               count(*) as relationship_strength
        ORDER BY relationship_strength DESC
        LIMIT 10
        """, 
        entities=query_entities, 
        max_depth=max_depth,
        database_="neo4j"
    )
    
    return [{"doc_id": record["doc_id"], 
            "content": record["content"],
            "graph_score": record["relationship_strength"]} 
           for record in records]

# Find documents related to "Python" and "AI"
related_docs = graph_rag.graph_enhanced_search(["Python", "AI"], max_depth=2)

# Access query statistics
print(f"Query executed in {summary.result_available_after}ms")
```

### 3. Knowledge Graph Reasoning

```cypher
-- Find indirect relationships
MATCH (a:Entity {name: "Python"})-[r*1..3]-(b:Entity)
WHERE b.name CONTAINS "Machine Learning"
RETURN a.name, b.name, [rel in r | type(rel)] as relationship_path
LIMIT 5;

-- Find most connected entities (influence ranking)
MATCH (e:Entity)-[r]-()
RETURN e.name, count(r) as connections
ORDER BY connections DESC
LIMIT 10;

-- Shortest path between concepts
MATCH path = shortestPath((a:Entity {name: "Python"})-[*]-(b:Entity {name: "Neural Networks"}))
RETURN path;
```

## Common Cypher Queries for RAG

### Data Exploration

```cypher
-- Count nodes and relationships
MATCH (n) RETURN labels(n), count(n);
MATCH ()-[r]->() RETURN type(r), count(r);

-- Sample data structure
MATCH (n)-[r]->(m) 
RETURN n, r, m 
LIMIT 25;

-- Find orphaned nodes (no relationships)
MATCH (n) 
WHERE NOT (n)--() 
RETURN n;
```

### Entity Management

```cypher
-- Find entities by type
MATCH (e:Entity) 
WHERE e.type = "PERSON" 
RETURN e.name, size((e)--()) as connections
ORDER BY connections DESC;

-- Merge duplicate entities
MATCH (e1:Entity), (e2:Entity)
WHERE e1.name = e2.name AND id(e1) < id(e2)
CALL apoc.refactor.mergeNodes([e1, e2]) 
YIELD node
RETURN node;

-- Find similar entities by name
MATCH (e:Entity)
WHERE e.name =~ "(?i).*python.*"
RETURN e.name, e.type;
```

### Document Analysis

```cypher
-- Documents with most entity mentions
MATCH (d:Document)-[:MENTIONS]->(e:Entity)
RETURN d.id, count(e) as entity_count
ORDER BY entity_count DESC
LIMIT 10;

-- Find documents mentioning specific entity types
MATCH (d:Document)-[:MENTIONS]->(e:Entity)
WHERE e.type IN ["PERSON", "ORG"]
RETURN d.id, collect(e.name) as mentioned_entities;

-- Co-occurrence analysis
MATCH (d:Document)-[:MENTIONS]->(e1:Entity),
      (d)-[:MENTIONS]->(e2:Entity)
WHERE e1.name < e2.name
RETURN e1.name, e2.name, count(d) as co_occurrences
ORDER BY co_occurrences DESC
LIMIT 20;
```

## Performance Optimization

### Indexing Strategy

```cypher
-- Create indexes for faster queries
CREATE INDEX entity_name_index FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type_index FOR (e:Entity) ON (e.type);
CREATE INDEX document_id_index FOR (d:Document) ON (d.id);

-- Composite index for entity lookup
CREATE INDEX entity_name_type_index FOR (e:Entity) ON (e.name, e.type);

-- Full-text search index
CALL db.index.fulltext.createNodeIndex("entityFulltext", ["Entity"], ["name", "description"]);

-- Query using full-text index
CALL db.index.fulltext.queryNodes("entityFulltext", "machine learning") 
YIELD node, score
RETURN node.name, score
ORDER BY score DESC;
```

### Query Optimization

```cypher
-- Use EXPLAIN to analyze query performance
EXPLAIN MATCH (e:Entity {name: "Python"})-[*1..2]-(related)
RETURN related.name, labels(related);

-- Use PROFILE for detailed execution statistics  
PROFILE MATCH (d:Document)-[:MENTIONS]->(e:Entity {type: "TECHNOLOGY"})
RETURN d.id, count(e)
ORDER BY count(e) DESC;
```

## Graph Data Science (GDS)

### Centrality Analysis

```cypher
-- Create graph projection
CALL gds.graph.project(
    'entityGraph',
    'Entity', 
    'RELATED_TO'
);

-- PageRank to find important entities
CALL gds.pageRank.stream('entityGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name as entity, score
ORDER BY score DESC
LIMIT 10;

-- Betweenness centrality (bridge entities)
CALL gds.betweenness.stream('entityGraph')
YIELD nodeId, score  
RETURN gds.util.asNode(nodeId).name as entity, score
ORDER BY score DESC;
```

### Community Detection

```cypher
-- Louvain community detection
CALL gds.louvain.stream('entityGraph')
YIELD nodeId, communityId
RETURN communityId, collect(gds.util.asNode(nodeId).name) as entities
ORDER BY size(entities) DESC;

-- Label propagation
CALL gds.labelPropagation.stream('entityGraph')
YIELD nodeId, communityId
RETURN communityId, count(*) as community_size
ORDER BY community_size DESC;
```

## Integration with Vector Search

### Hybrid RAG Approach

```python
class HybridRAG:
    def __init__(self, mongo_client, neo4j_driver):
        self.mongo = mongo_client
        self.neo4j = neo4j_driver
    
    def hybrid_search(self, query, vector_results):
        """Enhance vector search with graph relationships"""
        
        # Extract entities from top vector results
        entities = self.extract_entities_from_results(vector_results)
        
        # Find related entities in graph using execute_query
        records, summary, keys = self.neo4j.execute_query(
            """
            MATCH (e:Entity)-[*1..2]-(related:Entity)
            WHERE e.name IN $entities
            RETURN DISTINCT related.name as entity
            LIMIT 50
            """, 
            entities=entities,
            database_="neo4j"
        )
        
        # Extract entity names from results
        graph_entities = [record["entity"] for record in records]
        
        # Expand search with graph-discovered entities
        expanded_query = query + " " + " ".join(graph_entities)
        
        # Perform new vector search with expanded context
        return self.vector_search(expanded_query)
```

## Monitoring and Maintenance

### Health Checks

```cypher
-- Database health
CALL dbms.components() YIELD name, versions, edition;

-- Memory usage
CALL dbms.queryJmx("org.neo4j:instance=kernel#0,name=Memory Pools") 
YIELD attributes 
RETURN attributes;

-- Active connections
CALL dbms.listConnections() YIELD username, connectionCount;
```

### Backup and Restore

```bash
# Backup database
docker exec rag-neo4j neo4j-admin database dump neo4j --to-path=/backups

# Restore database
docker exec rag-neo4j neo4j-admin database load neo4j --from-path=/backups
```

## Troubleshooting

### Common Issues

**"Database unavailable"**
```bash
# Check container status
docker compose logs neo4j

# Check if database is online
docker exec rag-neo4j cypher-shell -u neo4j -p password "RETURN 'OK' as status;"
```

**Slow queries**
```cypher
-- Find long-running queries
CALL dbms.listQueries() 
YIELD query, elapsedTimeMillis, status
WHERE elapsedTimeMillis > 1000
RETURN query, elapsedTimeMillis, status;

-- Kill slow query
CALL dbms.killQuery('query-id');
```

**Memory issues**
```bash
# Increase heap size in docker-compose.yml
NEO4J_dbms_memory_heap_max__size: 4G
NEO4J_dbms_memory_pagecache_size: 2G
```

## Best Practices

### Data Modeling
1. **Keep it simple**: Start with basic nodes and relationships
2. **Use meaningful labels**: Entity, Document, Topic, etc.
3. **Property consistency**: Standardize property names and types
4. **Avoid deep hierarchies**: Limit relationship traversal depth

### Query Performance  
1. **Index frequently queried properties**
2. **Use parameterized queries** to avoid query plan cache pollution
3. **Limit result sets** with appropriate LIMIT clauses
4. **Profile expensive queries** and optimize

### Security
1. **Change default password** immediately
2. **Use least privilege** principle for application users
3. **Enable authentication** for production deployments
4. **Regular backups** of critical graph data

## Learning Resources

- [Neo4j Graph Academy](https://graphacademy.neo4j.com/) - Free online courses
- [Cypher Manual](https://neo4j.com/docs/cypher-manual/current/) - Complete query language reference
- [Graph Data Science Library](https://neo4j.com/docs/graph-data-science/) - Advanced analytics
- [APOC Procedures](https://neo4j.com/labs/apoc/) - Utility procedures library