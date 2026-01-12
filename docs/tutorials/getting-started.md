# Getting Started Tutorial

This tutorial will walk you through building your first RAG application using the deployed infrastructure stack.

## Prerequisites

- RAG Infrastructure Stack deployed and running
- Python 3.8+ installed
- Basic understanding of Python and APIs

## Step 1: Setup Python Environment

```bash
# Create virtual environment
python -m venv rag-tutorial
source rag-tutorial/bin/activate  # Linux/Mac
# or
rag-tutorial\Scripts\activate     # Windows

# Install required packages
pip install pymongo requests openai python-dotenv langfuse
```

## Step 2: Configuration

Create a `.env` file with your stack configuration:

```bash
# .env file
MONGO_URL=mongodb://your-host:27017/
OLLAMA_BASE_URL=http://your-host:11434
LANGFUSE_HOST=http://your-host:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...  # Get from Langfuse UI
LANGFUSE_SECRET_KEY=sk-lf-...  # Get from Langfuse UI

# Optional: OpenAI for better embeddings
OPENAI_API_KEY=sk-...
```

## Step 3: Basic RAG Implementation

Create `simple_rag.py`:

```python
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import requests
import json
from typing import List, Dict

load_dotenv()

class SimpleRAG:
    def __init__(self):
        self.mongo_client = MongoClient(os.getenv('MONGO_URL'))
        self.db = self.mongo_client.tutorial_db
        self.collection = self.db.documents
        self.ollama_url = os.getenv('OLLAMA_BASE_URL')
        
        # Ensure collection has vector index
        self.setup_vector_index()
    
    def setup_vector_index(self):
        """Create vector search index if it doesn't exist"""
        try:
            self.collection.create_search_index({
                "definition": {
                    "vectorSearchType": "knn",
                    "fields": [{
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 768,  # nomic-embed-text dimensions
                        "similarity": "cosine"
                    }]
                },
                "name": "vector_index"
            })
            print("✅ Vector index created")
        except Exception as e:
            print(f"Vector index might already exist: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama"""
        response = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            }
        )
        response.raise_for_status()
        return response.json()["embedding"]
    
    def add_document(self, content: str, metadata: Dict = None):
        """Add a document to the knowledge base"""
        if metadata is None:
            metadata = {}
        
        document = {
            "content": content,
            "metadata": metadata,
            "embedding": self.generate_embedding(content)
        }
        
        result = self.collection.insert_one(document)
        print(f"✅ Document added with ID: {result.inserted_id}")
        return result.inserted_id
    
    def search_similar(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for similar documents"""
        query_embedding = self.generate_embedding(query)
        
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
                    "score": 1
                }
            }
        ]
        
        return list(self.collection.aggregate(pipeline))
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using Ollama"""
        prompt = f"""Based on the following context, please answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    
    def query(self, question: str) -> Dict:
        """Complete RAG pipeline"""
        # Retrieve similar documents
        similar_docs = self.search_similar(question, limit=3)
        
        if not similar_docs:
            return {
                "question": question,
                "answer": "I don't have enough information to answer this question.",
                "sources": []
            }
        
        # Combine context
        context = "\n\n".join([doc["content"] for doc in similar_docs])
        
        # Generate answer
        answer = self.generate_response(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "sources": similar_docs
        }

# Example usage
if __name__ == "__main__":
    rag = SimpleRAG()
    
    # Add some sample documents
    print("Adding sample documents...")
    
    documents = [
        {
            "content": "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, and AI.",
            "metadata": {"topic": "programming", "source": "python_intro"}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn and improve from data without being explicitly programmed.",
            "metadata": {"topic": "ai", "source": "ml_basics"}
        },
        {
            "content": "Docker is a containerization platform that allows developers to package applications and their dependencies into lightweight containers.",
            "metadata": {"topic": "devops", "source": "docker_guide"}
        }
    ]
    
    for doc in documents:
        rag.add_document(doc["content"], doc["metadata"])
    
    print("\n" + "="*50)
    print("RAG System Ready! Ask questions...")
    print("="*50)
    
    # Interactive query loop
    while True:
        question = input("\nYour question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        result = rag.query(question)
        
        print(f"\n🤖 Answer: {result['answer']}")
        print(f"\n📚 Sources ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'], 1):
            score = source.get('score', 0)
            metadata = source.get('metadata', {})
            print(f"  {i}. Score: {score:.3f} | Topic: {metadata.get('topic', 'N/A')}")
```

## Step 4: Run Your RAG System

```bash
python simple_rag.py
```

Example interaction:
```
Adding sample documents...
✅ Vector index created
✅ Document added with ID: 507f1f77bcf86cd799439011

==================================================
RAG System Ready! Ask questions...
==================================================

Your question (or 'quit' to exit): What is Python used for?

🤖 Answer: Based on the context provided, Python is used for several key applications:

1. Web development
2. Data science
3. Artificial intelligence (AI)

Python is particularly valued for its simplicity and readability, which makes it an excellent choice for these diverse applications.

📚 Sources (1):
  1. Score: 0.856 | Topic: programming
```

## Step 5: Monitoring with Langfuse

Add observability to track your RAG performance:

```python
from langfuse import Langfuse

class ObservableRAG(SimpleRAG):
    def __init__(self):
        super().__init__()
        self.langfuse = Langfuse(
            host=os.getenv('LANGFUSE_HOST'),
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
            secret_key=os.getenv('LANGFUSE_SECRET_KEY')
        )
    
    def query(self, question: str) -> Dict:
        # Create trace for this query
        trace = self.langfuse.trace(
            name="rag_query",
            input=question
        )
        
        # Retrieval step
        retrieval_span = trace.span(name="retrieval")
        similar_docs = self.search_similar(question, limit=3)
        retrieval_span.end(output={"num_docs": len(similar_docs)})
        
        if not similar_docs:
            trace.update(output="No relevant documents found")
            return {
                "question": question,
                "answer": "I don't have enough information to answer this question.",
                "sources": []
            }
        
        # Generation step
        generation_span = trace.span(name="generation")
        context = "\n\n".join([doc["content"] for doc in similar_docs])
        answer = self.generate_response(question, context)
        generation_span.end(output=answer)
        
        result = {
            "question": question,
            "answer": answer,
            "sources": similar_docs
        }
        
        trace.update(output=result)
        return result
```

## Step 6: Web Interface (Optional)

Create a simple web interface using the Vector Search UI:

1. Open `http://your-host:8090` in your browser
2. Navigate to your collection (`tutorial_db.documents`)
3. Use the search interface to test vector similarity
4. View and manage your documents

## Next Steps

### Enhance Your RAG System
1. **Better Embeddings**: Use OpenAI or fine-tuned models
2. **Chunk Optimization**: Split larger documents intelligently
3. **Hybrid Search**: Combine vector and keyword search
4. **Response Evaluation**: Add quality metrics

### Advanced Features
1. **Graph Integration**: Use Neo4j for relationship-aware retrieval
2. **Multi-modal RAG**: Include images and other media
3. **Real-time Updates**: Stream new documents automatically
4. **API Development**: Build REST API for your RAG system

### Production Considerations
1. **Security**: Add authentication and authorization
2. **Scaling**: Implement caching and load balancing
3. **Monitoring**: Set up alerts and performance tracking
4. **Backup**: Regular data backup strategies

## Troubleshooting

### Common Issues

**"Vector index not found"**
- Wait a few minutes for index creation to complete
- Check MongoDB logs: `docker compose logs mongodb`

**"Connection refused to Ollama"**
- Ensure Ollama service is running: `docker compose ps`
- Check if models are pulled: `docker exec rag-ollama ollama list`

**"Empty search results"**
- Verify documents were added successfully
- Check embedding generation is working
- Try different search queries

### Getting Help

1. Check service status: `docker compose ps`
2. View logs: `docker compose logs <service>`
3. Test connectivity: `./scripts/test-connectivity.sh`
4. Visit the Vector Search UI for visual debugging

## Resources

- [MongoDB Vector Search Docs](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Langfuse Tracing Guide](https://langfuse.com/docs/tracing)
- [RAG Evaluation Techniques](https://docs.ragas.io/)