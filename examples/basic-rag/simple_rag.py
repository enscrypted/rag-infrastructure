#!/usr/bin/env python3
"""
Simple RAG Example - Basic Document Q&A System

This example demonstrates a minimal RAG implementation using:
- MongoDB for vector storage
- Ollama for embeddings and generation
- Basic retrieval and generation pipeline

Usage:
    python simple_rag.py
"""

import os
import sys
from typing import List, Dict, Optional
from pymongo import MongoClient
import requests
import json

class SimpleRAG:
    """A basic RAG implementation for document question-answering."""
    
    def __init__(self, 
                 mongo_url: str = "mongodb://localhost:27017/",
                 ollama_url: str = "http://localhost:11434",
                 database: str = "rag_demo",
                 collection: str = "documents"):
        
        self.mongo_client = MongoClient(mongo_url)
        self.db = self.mongo_client[database]
        self.collection = self.db[collection]
        self.ollama_url = ollama_url
        
        print(f"📊 Connected to MongoDB: {database}.{collection}")
        print(f"🤖 Connected to Ollama: {ollama_url}")
        
        # Setup vector index
        self._setup_vector_index()
    
    def _setup_vector_index(self):
        """Create vector search index if it doesn't exist."""
        try:
            # Check if index exists
            indexes = list(self.collection.list_search_indexes())
            if any(idx.get('name') == 'vector_index' for idx in indexes):
                print("✅ Vector index already exists")
                return
            
            # Create new index using SearchIndexModel (MongoDB 8.2+ syntax)
            from pymongo.operations import SearchIndexModel
            
            model = SearchIndexModel(
                definition={
                    "fields": [{
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 768,  # nomic-embed-text dimensions
                        "similarity": "cosine"
                    }]
                },
                name="vector_index",
                type="vectorSearch"
            )
            self.collection.create_search_indexes([model])
            print("✅ Vector index created")
        except Exception as e:
            print(f"⚠️  Vector index setup: {e}")
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using Ollama."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            return None
    
    def add_document(self, content: str, metadata: Dict = None) -> Optional[str]:
        """Add a document to the knowledge base."""
        if metadata is None:
            metadata = {}
        
        # Generate embedding
        embedding = self._generate_embedding(content)
        if embedding is None:
            return None
        
        # Create document
        document = {
            "content": content,
            "metadata": metadata,
            "embedding": embedding
        }
        
        try:
            result = self.collection.insert_one(document)
            doc_id = str(result.inserted_id)
            print(f"✅ Document added: {doc_id}")
            return doc_id
        except Exception as e:
            print(f"❌ Failed to add document: {e}")
            return None
    
    def search_similar(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for similar documents using vector similarity."""
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            return []
        
        # Vector search pipeline
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
                "$addFields": {
                    "score": {"$meta": "vectorSearchScore"}
                }
            },
            {
                "$project": {
                    "content": 1,
                    "metadata": 1,
                    "score": 1,
                    "_id": 0
                }
            }
        ]
        
        try:
            results = list(self.collection.aggregate(pipeline))
            print(f"🔍 Found {len(results)} similar documents")
            return results
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def _generate_response(self, query: str, context: str) -> Optional[str]:
        """Generate response using Ollama."""
        prompt = f"""Based on the provided context, answer the question accurately and concisely.
If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"❌ Response generation failed: {e}")
            return None
    
    def query(self, question: str, num_docs: int = 3) -> Dict:
        """Complete RAG pipeline: retrieve and generate."""
        print(f"\n❓ Question: {question}")
        
        # Retrieve similar documents
        similar_docs = self.search_similar(question, limit=num_docs)
        
        if not similar_docs:
            return {
                "question": question,
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
                "context_used": ""
            }
        
        # Combine context from retrieved documents
        context_parts = []
        for i, doc in enumerate(similar_docs, 1):
            score = doc.get('score', 0)
            content = doc.get('content', '')
            context_parts.append(f"Document {i} (relevance: {score:.3f}):\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = self._generate_response(question, context)
        if answer is None:
            answer = "Sorry, I encountered an error generating the response."
        
        return {
            "question": question,
            "answer": answer,
            "sources": similar_docs,
            "context_used": context
        }
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        result = self.collection.delete_many({})
        print(f"🗑️  Cleared {result.deleted_count} documents")
    
    def get_stats(self) -> Dict:
        """Get collection statistics."""
        count = self.collection.count_documents({})
        return {
            "total_documents": count,
            "database": self.db.name,
            "collection": self.collection.name
        }

def load_sample_data(rag: SimpleRAG):
    """Load sample documents for testing."""
    print("\n📚 Loading sample documents...")
    
    sample_docs = [
        {
            "content": "Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together.",
            "metadata": {"topic": "programming", "language": "python", "source": "python_guide"}
        },
        {
            "content": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            "metadata": {"topic": "ai", "subtopic": "machine_learning", "source": "ml_intro"}
        },
        {
            "content": "Docker is a set of platform as a service products that use OS-level virtualization to deliver software in packages called containers. Containers are isolated from one another and bundle their own software, libraries and configuration files.",
            "metadata": {"topic": "devops", "technology": "docker", "source": "container_guide"}
        },
        {
            "content": "Vector databases are specialized databases designed to store and query high-dimensional vector data efficiently. They are essential for applications like semantic search, recommendation systems, and similarity matching in AI applications.",
            "metadata": {"topic": "databases", "type": "vector", "source": "vector_db_guide"}
        },
        {
            "content": "Retrieval-Augmented Generation (RAG) is a natural language processing technique that combines the capabilities of large language models with external knowledge retrieval. This approach helps overcome the limitations of pre-trained models by providing relevant, up-to-date information during text generation.",
            "metadata": {"topic": "ai", "subtopic": "rag", "source": "rag_overview"}
        }
    ]
    
    for doc in sample_docs:
        rag.add_document(doc["content"], doc["metadata"])
    
    print(f"✅ Loaded {len(sample_docs)} sample documents")

def interactive_mode(rag: SimpleRAG):
    """Run interactive Q&A session."""
    print("\n" + "="*60)
    print("🎯 RAG DEMO - Interactive Mode")
    print("="*60)
    print("Ask questions about the loaded documents!")
    print("Commands: 'stats', 'clear', 'reload', 'quit'")
    print("="*60)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            elif question.lower() == 'stats':
                stats = rag.get_stats()
                print(f"📊 Stats: {stats}")
                continue
            
            elif question.lower() == 'clear':
                rag.clear_collection()
                continue
            
            elif question.lower() == 'reload':
                rag.clear_collection()
                load_sample_data(rag)
                continue
            
            # Process RAG query
            result = rag.query(question)
            
            print(f"\n🤖 Answer: {result['answer']}")
            
            if result['sources']:
                print(f"\n📚 Sources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'], 1):
                    score = source.get('score', 0)
                    metadata = source.get('metadata', {})
                    topic = metadata.get('topic', 'N/A')
                    print(f"  {i}. Relevance: {score:.3f} | Topic: {topic}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main application entry point."""
    # Configuration
    mongo_url = os.getenv('MONGO_URL', 'mongodb://localhost:27017/')
    ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    print("🚀 Simple RAG Demo Starting...")
    print(f"MongoDB: {mongo_url}")
    print(f"Ollama: {ollama_url}")
    
    try:
        # Initialize RAG system
        rag = SimpleRAG(mongo_url=mongo_url, ollama_url=ollama_url)
        
        # Check if we have documents
        stats = rag.get_stats()
        if stats['total_documents'] == 0:
            print("\n📝 No documents found. Loading sample data...")
            load_sample_data(rag)
        else:
            print(f"\n📊 Found {stats['total_documents']} existing documents")
        
        # Start interactive mode
        interactive_mode(rag)
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure MongoDB is running and accessible")
        print("2. Ensure Ollama is running with nomic-embed-text and llama3.2:3b models")
        print("3. Check network connectivity")
        sys.exit(1)

if __name__ == "__main__":
    main()