#!/usr/bin/env python3
"""
Graph RAG Example - Knowledge Graph Enhanced Retrieval

This example demonstrates Graph RAG using:
- Neo4j for knowledge graph storage
- MongoDB for document/vector storage
- Ollama for embeddings and generation
- Entity extraction and relationship mapping

Usage:
    python graph_rag.py
"""

import os
import re
from typing import List, Dict, Optional, Tuple
from neo4j import GraphDatabase
from pymongo import MongoClient
import requests
import json


class GraphRAG:
    """Graph-enhanced RAG combining Neo4j knowledge graphs with vector search."""

    def __init__(self,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 mongo_url: str = "mongodb://localhost:27017/",
                 ollama_url: str = "http://localhost:11434",
                 database: str = "graph_rag_demo"):

        # Neo4j connection
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print(f"📊 Connected to Neo4j: {neo4j_uri}")

        # MongoDB connection
        self.mongo_client = MongoClient(mongo_url)
        self.db = self.mongo_client[database]
        self.documents = self.db.documents
        print(f"📊 Connected to MongoDB: {database}")

        # Ollama
        self.ollama_url = ollama_url
        print(f"🤖 Connected to Ollama: {ollama_url}")

        # Setup indexes
        self._setup_neo4j_indexes()
        self._setup_vector_index()

    def _setup_neo4j_indexes(self):
        """Create Neo4j indexes for efficient lookups."""
        with self.driver.session() as session:
            # Create indexes for entity lookup
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)")
        print("✅ Neo4j indexes ready")

    def _setup_vector_index(self):
        """Create MongoDB vector search index."""
        try:
            indexes = list(self.documents.list_search_indexes())
            if not any(idx.get('name') == 'vector_index' for idx in indexes):
                self.documents.create_search_index({
                    "definition": {
                        "mappings": {
                            "dynamic": True,
                            "fields": {
                                "embedding": {
                                    "type": "knnVector",
                                    "dimensions": 768,
                                    "similarity": "cosine"
                                }
                            }
                        }
                    },
                    "name": "vector_index"
                })
                print("✅ MongoDB vector index created")
            else:
                print("✅ MongoDB vector index exists")
        except Exception as e:
            print(f"⚠️ Vector index setup: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama."""
        response = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text}
        )
        return response.json()["embedding"]

    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract entities from text using LLM."""
        prompt = f"""Extract named entities from the following text.
Return as JSON array with objects containing "name", "type" (PERSON, ORG, TECH, CONCEPT), and "description".

Text: {text}

Return only valid JSON array, no other text:"""

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}
        )

        result = response.json()["response"]

        # Try to parse JSON from response
        try:
            # Find JSON array in response
            match = re.search(r'\[.*\]', result, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass

        return []

    def _extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Extract relationships between entities."""
        if len(entities) < 2:
            return []

        entity_names = [e["name"] for e in entities]
        prompt = f"""Given these entities: {entity_names}

And this text: {text}

Extract relationships between entities. Return as JSON array with objects containing:
"source", "target", "relationship" (e.g., "WORKS_FOR", "CREATED", "USES", "RELATED_TO")

Return only valid JSON array:"""

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}
        )

        result = response.json()["response"]

        try:
            match = re.search(r'\[.*\]', result, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass

        return []

    def add_document(self, content: str, metadata: Dict = None) -> str:
        """Add document with entity extraction and graph building."""
        metadata = metadata or {}

        # Generate embedding
        embedding = self._get_embedding(content)

        # Store in MongoDB
        doc = {
            "content": content,
            "metadata": metadata,
            "embedding": embedding
        }
        result = self.documents.insert_one(doc)
        doc_id = str(result.inserted_id)

        print(f"📄 Document stored: {doc_id}")

        # Extract entities and relationships
        entities = self._extract_entities(content)
        print(f"🔍 Extracted {len(entities)} entities")

        relationships = self._extract_relationships(content, entities)
        print(f"🔗 Extracted {len(relationships)} relationships")

        # Build knowledge graph
        with self.driver.session() as session:
            # Create document node
            session.run(
                "MERGE (d:Document {id: $doc_id}) SET d.preview = $preview",
                doc_id=doc_id,
                preview=content[:100]
            )

            # Create entity nodes and link to document
            for entity in entities:
                session.run("""
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type, e.description = $description
                    WITH e
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:MENTIONS]->(e)
                """,
                    name=entity.get("name", "Unknown"),
                    type=entity.get("type", "UNKNOWN"),
                    description=entity.get("description", ""),
                    doc_id=doc_id
                )

            # Create relationships between entities
            for rel in relationships:
                session.run("""
                    MATCH (s:Entity {name: $source})
                    MATCH (t:Entity {name: $target})
                    MERGE (s)-[r:RELATED_TO {type: $rel_type}]->(t)
                """,
                    source=rel.get("source"),
                    target=rel.get("target"),
                    rel_type=rel.get("relationship", "RELATED_TO")
                )

        print(f"✅ Knowledge graph updated")
        return doc_id

    def vector_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Standard vector similarity search."""
        query_embedding = self._get_embedding(query)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": limit * 10,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        return list(self.documents.aggregate(pipeline))

    def graph_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search using knowledge graph relationships."""
        # Extract entities from query
        query_entities = self._extract_entities(query)
        entity_names = [e["name"] for e in query_entities]

        if not entity_names:
            return []

        # Find related documents through graph
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.name IN $names
                MATCH (d:Document)-[:MENTIONS]->(e)
                WITH d, count(e) as relevance
                ORDER BY relevance DESC
                LIMIT $limit
                RETURN d.id as doc_id, d.preview as preview, relevance
            """, names=entity_names, limit=limit)

            graph_results = []
            for record in result:
                graph_results.append({
                    "doc_id": record["doc_id"],
                    "preview": record["preview"],
                    "relevance": record["relevance"]
                })

            return graph_results

    def get_entity_context(self, entity_name: str) -> Dict:
        """Get full context for an entity from the knowledge graph."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {name: $name})
                OPTIONAL MATCH (e)-[r]-(related:Entity)
                OPTIONAL MATCH (d:Document)-[:MENTIONS]->(e)
                RETURN e.name as name,
                       e.type as type,
                       e.description as description,
                       collect(DISTINCT {name: related.name, type: related.type, rel: type(r)}) as related,
                       collect(DISTINCT d.id) as documents
            """, name=entity_name)

            record = result.single()
            if record:
                return {
                    "name": record["name"],
                    "type": record["type"],
                    "description": record["description"],
                    "related_entities": [r for r in record["related"] if r["name"]],
                    "document_ids": record["documents"]
                }
            return None

    def hybrid_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Combine vector search with graph-based retrieval."""
        # Get results from both methods
        vector_results = self.vector_search(query, limit)
        graph_results = self.graph_search(query, limit)

        # Merge and deduplicate
        seen_ids = set()
        combined = []

        # Add vector results with scores
        for doc in vector_results:
            doc_id = str(doc.get("_id"))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined.append({
                    "doc_id": doc_id,
                    "content": doc.get("content"),
                    "vector_score": doc.get("score", 0),
                    "graph_score": 0,
                    "source": "vector"
                })

        # Add graph results
        for doc in graph_results:
            doc_id = doc.get("doc_id")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                # Fetch full document
                full_doc = self.documents.find_one({"_id": doc_id})
                combined.append({
                    "doc_id": doc_id,
                    "content": full_doc.get("content") if full_doc else doc.get("preview"),
                    "vector_score": 0,
                    "graph_score": doc.get("relevance", 0),
                    "source": "graph"
                })
            else:
                # Update existing with graph score
                for c in combined:
                    if c["doc_id"] == doc_id:
                        c["graph_score"] = doc.get("relevance", 0)
                        c["source"] = "hybrid"

        # Sort by combined score
        combined.sort(key=lambda x: x["vector_score"] + x["graph_score"] * 0.5, reverse=True)

        return combined[:limit]

    def query(self, question: str, use_graph: bool = True) -> Dict:
        """Complete Graph RAG pipeline."""
        # Retrieve documents
        if use_graph:
            results = self.hybrid_search(question, limit=5)
            method = "hybrid (vector + graph)"
        else:
            results = self.vector_search(question, limit=5)
            method = "vector only"

        if not results:
            return {
                "question": question,
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
                "method": method
            }

        # Build context
        context_parts = []
        for doc in results:
            content = doc.get("content", "")
            if content:
                context_parts.append(content)

        context = "\n\n---\n\n".join(context_parts)

        # Generate response
        prompt = f"""Based on the following context, answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer:"""

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}
        )

        answer = response.json()["response"]

        return {
            "question": question,
            "answer": answer,
            "sources": results,
            "method": method
        }

    def visualize_graph(self, limit: int = 50):
        """Get graph data for visualization."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r]-(other:Entity)
                RETURN e.name as source, e.type as source_type,
                       type(r) as relationship,
                       other.name as target, other.type as target_type
                LIMIT $limit
            """, limit=limit)

            nodes = {}
            edges = []

            for record in result:
                source = record["source"]
                if source and source not in nodes:
                    nodes[source] = {"type": record["source_type"]}

                target = record["target"]
                if target and target not in nodes:
                    nodes[target] = {"type": record["target_type"]}

                if source and target and record["relationship"]:
                    edges.append({
                        "source": source,
                        "target": target,
                        "relationship": record["relationship"]
                    })

            return {"nodes": nodes, "edges": edges}

    def close(self):
        """Close database connections."""
        self.driver.close()
        self.mongo_client.close()


# Sample documents about AI/ML topics
SAMPLE_DOCUMENTS = [
    {
        "content": """Python is a high-level programming language created by Guido van Rossum.
        It is widely used in machine learning and data science. Popular Python libraries include
        NumPy for numerical computing, Pandas for data manipulation, and TensorFlow for deep learning.
        Python's simplicity makes it ideal for beginners and experts alike.""",
        "metadata": {"topic": "programming", "source": "tech_overview"}
    },
    {
        "content": """Machine learning is a subset of artificial intelligence that enables computers
        to learn from data. Key techniques include supervised learning, unsupervised learning, and
        reinforcement learning. Companies like Google, OpenAI, and Anthropic are leading ML research.
        Neural networks form the backbone of modern deep learning systems.""",
        "metadata": {"topic": "ai", "source": "ml_intro"}
    },
    {
        "content": """TensorFlow is an open-source machine learning framework developed by Google.
        It supports both CPU and GPU computation. PyTorch, developed by Meta, is another popular
        framework known for its dynamic computation graphs. Both frameworks are used extensively
        in research and production environments.""",
        "metadata": {"topic": "frameworks", "source": "ml_tools"}
    },
    {
        "content": """Large Language Models (LLMs) like GPT and Claude are trained on vast amounts
        of text data. These models use transformer architecture invented by researchers at Google.
        Applications include chatbots, code generation, and content creation. Fine-tuning allows
        customization for specific tasks.""",
        "metadata": {"topic": "llm", "source": "ai_overview"}
    },
    {
        "content": """Neo4j is a graph database that stores data as nodes and relationships.
        It uses the Cypher query language for data manipulation. Graph databases excel at
        representing connected data like social networks, knowledge graphs, and recommendation
        systems. Neo4j integrates well with Python through the official driver.""",
        "metadata": {"topic": "databases", "source": "graph_db"}
    }
]


def main():
    """Demo the Graph RAG system."""
    print("🚀 Graph RAG Demo Starting...")
    print("=" * 60)

    # Initialize
    rag = GraphRAG(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        mongo_url=os.getenv("MONGO_URL", "mongodb://localhost:27017/"),
        ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # Check if we need to load sample data
    if rag.documents.count_documents({}) == 0:
        print("\n📚 Loading sample documents...")
        for doc in SAMPLE_DOCUMENTS:
            rag.add_document(doc["content"], doc["metadata"])
        print(f"✅ Loaded {len(SAMPLE_DOCUMENTS)} documents")
    else:
        print(f"📊 Found {rag.documents.count_documents({})} existing documents")

    # Interactive mode
    print("\n" + "=" * 60)
    print("🎯 GRAPH RAG DEMO - Interactive Mode")
    print("=" * 60)
    print("Commands: 'graph' (show graph), 'entity <name>', 'quit'")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n❓ Your question: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                break

            if user_input.lower() == 'graph':
                graph_data = rag.visualize_graph()
                print(f"\n📊 Knowledge Graph:")
                print(f"   Nodes: {len(graph_data['nodes'])}")
                print(f"   Edges: {len(graph_data['edges'])}")
                for node, data in list(graph_data['nodes'].items())[:10]:
                    print(f"   - {node} ({data['type']})")
                continue

            if user_input.lower().startswith('entity '):
                entity_name = user_input[7:].strip()
                context = rag.get_entity_context(entity_name)
                if context:
                    print(f"\n📌 Entity: {context['name']} ({context['type']})")
                    print(f"   Description: {context['description']}")
                    print(f"   Related: {[r['name'] for r in context['related_entities'][:5]]}")
                    print(f"   Documents: {len(context['document_ids'])}")
                else:
                    print(f"   Entity '{entity_name}' not found")
                continue

            # Query with Graph RAG
            print("\n🔍 Searching with Graph RAG...")
            result = rag.query(user_input, use_graph=True)

            print(f"\n🤖 Answer ({result['method']}):")
            print(result['answer'])

            print(f"\n📚 Sources ({len(result['sources'])}):")
            for i, src in enumerate(result['sources'][:3], 1):
                source_type = src.get('source', 'unknown')
                print(f"   {i}. [{source_type}] {src.get('content', '')[:80]}...")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    rag.close()
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
