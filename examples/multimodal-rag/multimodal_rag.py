#!/usr/bin/env python3
"""
Multimodal RAG Example - Image and Text Processing

This example demonstrates multimodal RAG using:
- Ollama LLaVA for image understanding
- MongoDB for vector storage
- Combined image + text retrieval

Usage:
    python multimodal_rag.py
"""

import os
import base64
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from pymongo import MongoClient
import requests
import json


@dataclass
class Document:
    """Represents a document (text or image) in the RAG system."""
    content: str  # Text content or image description
    doc_type: str  # "text" or "image"
    source: str  # File path or identifier
    metadata: Dict
    embedding: Optional[List[float]] = None


class MultimodalRAG:
    """RAG system supporting both text and image inputs."""

    def __init__(self,
                 mongo_url: str = "mongodb://localhost:27017/",
                 ollama_url: str = "http://localhost:11434",
                 database: str = "multimodal_rag"):

        self.mongo_client = MongoClient(mongo_url)
        self.db = self.mongo_client[database]
        self.documents = self.db.documents
        self.ollama_url = ollama_url

        print(f"Connected to MongoDB: {database}")
        print(f"Connected to Ollama: {ollama_url}")

        self._setup_vector_index()
        self._check_models()

    def _setup_vector_index(self):
        """Create vector search index if needed."""
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
                print("Vector index created")
        except Exception as e:
            print(f"Vector index setup: {e}")

    def _check_models(self):
        """Check required Ollama models are available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            models = [m["name"] for m in response.json().get("models", [])]

            # Check for vision model
            vision_models = ["llava", "llava:7b", "llava:13b", "llama3.2-vision"]
            has_vision = any(vm in m for m in models for vm in vision_models)

            if has_vision:
                print("Vision model available")
            else:
                print("Warning: No vision model found. Run: ollama pull llava")

            # Check for embedding model
            if any("nomic" in m or "embed" in m for m in models):
                print("Embedding model available")
            else:
                print("Warning: No embedding model. Run: ollama pull nomic-embed-text")

        except Exception as e:
            print(f"Model check failed: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama."""
        response = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text}
        )
        return response.json()["embedding"]

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _analyze_image(self, image_path: str, prompt: str = None) -> str:
        """Analyze image using LLaVA vision model."""
        if prompt is None:
            prompt = """Describe this image in detail. Include:
1. Main subjects and objects
2. Colors and visual elements
3. Text visible in the image (if any)
4. Overall context and meaning

Provide a comprehensive description:"""

        image_b64 = self._encode_image(image_path)

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": "llava",
                "prompt": prompt,
                "images": [image_b64],
                "stream": False
            }
        )

        return response.json()["response"]

    def _get_image_hash(self, image_path: str) -> str:
        """Generate hash for image deduplication."""
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def add_text(self, content: str, metadata: Dict = None) -> str:
        """Add text document to the knowledge base."""
        metadata = metadata or {}

        embedding = self._get_embedding(content)

        doc = {
            "content": content,
            "doc_type": "text",
            "source": metadata.get("source", "manual"),
            "metadata": metadata,
            "embedding": embedding
        }

        result = self.documents.insert_one(doc)
        doc_id = str(result.inserted_id)
        print(f"Text document added: {doc_id}")
        return doc_id

    def add_image(self, image_path: str, metadata: Dict = None) -> str:
        """Add image to knowledge base with automatic description."""
        metadata = metadata or {}
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Check for duplicates
        image_hash = self._get_image_hash(image_path)
        existing = self.documents.find_one({"metadata.image_hash": image_hash})
        if existing:
            print(f"Image already exists: {existing['_id']}")
            return str(existing["_id"])

        print(f"Analyzing image: {path.name}...")

        # Generate description using vision model
        description = self._analyze_image(image_path)

        # Generate embedding from description
        embedding = self._get_embedding(description)

        doc = {
            "content": description,
            "doc_type": "image",
            "source": str(path.absolute()),
            "metadata": {
                **metadata,
                "filename": path.name,
                "image_hash": image_hash,
                "file_size": path.stat().st_size
            },
            "embedding": embedding
        }

        result = self.documents.insert_one(doc)
        doc_id = str(result.inserted_id)
        print(f"Image added: {doc_id}")
        print(f"  Description: {description[:100]}...")
        return doc_id

    def add_directory(self, dir_path: str, extensions: List[str] = None) -> int:
        """Add all supported files from a directory."""
        if extensions is None:
            extensions = [".txt", ".md", ".png", ".jpg", ".jpeg", ".gif", ".webp"]

        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        count = 0
        for file_path in path.rglob("*"):
            if file_path.suffix.lower() in extensions:
                try:
                    if file_path.suffix.lower() in [".txt", ".md"]:
                        content = file_path.read_text()
                        self.add_text(content, {"source": str(file_path)})
                    else:
                        self.add_image(str(file_path))
                    count += 1
                except Exception as e:
                    print(f"Failed to add {file_path}: {e}")

        print(f"Added {count} files from {dir_path}")
        return count

    def search(self, query: str, limit: int = 5, doc_type: str = None) -> List[Dict]:
        """Search for similar documents using vector similarity."""
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
                    "doc_type": 1,
                    "source": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        # Add type filter if specified
        if doc_type:
            pipeline.insert(1, {"$match": {"doc_type": doc_type}})

        results = list(self.documents.aggregate(pipeline))
        return results

    def query(self, question: str, include_images: bool = True) -> Dict:
        """Complete multimodal RAG pipeline."""
        print(f"\nQuestion: {question}")

        # Search for relevant documents
        results = self.search(question, limit=5)

        if not results:
            return {
                "question": question,
                "answer": "I don't have enough information to answer this question.",
                "sources": []
            }

        # Separate text and image results
        text_docs = [r for r in results if r["doc_type"] == "text"]
        image_docs = [r for r in results if r["doc_type"] == "image"]

        print(f"Found {len(text_docs)} text docs, {len(image_docs)} image docs")

        # Build context
        context_parts = []

        for doc in text_docs:
            context_parts.append(f"[Text Document]\n{doc['content']}")

        for doc in image_docs:
            context_parts.append(f"[Image: {doc['metadata'].get('filename', 'unknown')}]\n{doc['content']}")

        context = "\n\n---\n\n".join(context_parts)

        # Generate response
        prompt = f"""Based on the following context (which includes both text documents and image descriptions), answer the question accurately.

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
            "sources": [
                {
                    "type": r["doc_type"],
                    "source": r.get("source", "unknown"),
                    "score": r.get("score", 0),
                    "preview": r["content"][:200]
                }
                for r in results
            ]
        }

    def query_with_image(self, question: str, image_path: str) -> Dict:
        """Query using both text question and an image as input."""
        print(f"\nQuestion: {question}")
        print(f"With image: {image_path}")

        # Analyze the query image
        image_context = self._analyze_image(
            image_path,
            f"Analyze this image in the context of the following question: {question}"
        )

        # Search for related documents
        combined_query = f"{question} {image_context[:500]}"
        results = self.search(combined_query, limit=5)

        # Build context
        context_parts = [f"[Query Image Analysis]\n{image_context}"]

        for doc in results:
            if doc["doc_type"] == "text":
                context_parts.append(f"[Text Document]\n{doc['content']}")
            else:
                context_parts.append(f"[Related Image]\n{doc['content']}")

        context = "\n\n---\n\n".join(context_parts)

        # Generate response
        prompt = f"""Based on the following context (including analysis of a user-provided image), answer the question.

Context:
{context}

Question: {question}

Answer:"""

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}
        )

        return {
            "question": question,
            "image_analysis": image_context,
            "answer": response.json()["response"],
            "sources": [
                {
                    "type": r["doc_type"],
                    "source": r.get("source", "unknown"),
                    "score": r.get("score", 0)
                }
                for r in results
            ]
        }

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        total = self.documents.count_documents({})
        text_count = self.documents.count_documents({"doc_type": "text"})
        image_count = self.documents.count_documents({"doc_type": "image"})

        return {
            "total_documents": total,
            "text_documents": text_count,
            "image_documents": image_count
        }

    def clear(self):
        """Clear all documents."""
        result = self.documents.delete_many({})
        print(f"Deleted {result.deleted_count} documents")


# Sample data for demonstration
SAMPLE_TEXTS = [
    {
        "content": """Machine learning models can be categorized into supervised, unsupervised, and reinforcement learning.
        Supervised learning uses labeled data to train models for classification and regression tasks.
        Popular algorithms include linear regression, decision trees, and neural networks.""",
        "metadata": {"topic": "machine_learning", "source": "ml_guide"}
    },
    {
        "content": """Computer vision enables machines to interpret and understand visual information from the world.
        Key tasks include image classification, object detection, and semantic segmentation.
        Deep learning models like CNNs and Vision Transformers have revolutionized this field.""",
        "metadata": {"topic": "computer_vision", "source": "cv_intro"}
    },
    {
        "content": """Natural language processing (NLP) allows computers to understand and generate human language.
        Modern NLP uses transformer architectures like BERT and GPT.
        Applications include chatbots, translation, and sentiment analysis.""",
        "metadata": {"topic": "nlp", "source": "nlp_overview"}
    },
    {
        "content": """Data visualization helps communicate complex information through charts and graphs.
        Common chart types include bar charts, line graphs, scatter plots, and heatmaps.
        Good visualizations tell a story and highlight key insights in the data.""",
        "metadata": {"topic": "visualization", "source": "dataviz_guide"}
    },
    {
        "content": """Docker containers package applications with their dependencies for consistent deployment.
        Images are built from Dockerfiles and can be shared via registries.
        Kubernetes orchestrates containers at scale across multiple nodes.""",
        "metadata": {"topic": "devops", "source": "container_guide"}
    }
]


def create_sample_image(output_path: str) -> str:
    """Create a simple test image using PIL if available."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Create a simple diagram
        img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(img)

        # Draw boxes
        draw.rectangle([50, 50, 150, 100], outline='blue', width=2)
        draw.rectangle([250, 50, 350, 100], outline='green', width=2)
        draw.rectangle([150, 180, 250, 230], outline='red', width=2)

        # Draw arrows
        draw.line([150, 75, 250, 75], fill='black', width=2)
        draw.line([100, 100, 200, 180], fill='black', width=2)
        draw.line([300, 100, 200, 180], fill='black', width=2)

        # Add labels
        draw.text((70, 60), "Input", fill='blue')
        draw.text((265, 60), "Process", fill='green')
        draw.text((170, 190), "Output", fill='red')
        draw.text((100, 260), "Sample Architecture Diagram", fill='black')

        img.save(output_path)
        return output_path

    except ImportError:
        print("PIL not installed. Skipping sample image creation.")
        print("Install with: pip install Pillow")
        return None


def main():
    """Demo the multimodal RAG system."""
    print("=" * 60)
    print("Multimodal RAG Demo")
    print("=" * 60)

    # Initialize
    rag = MultimodalRAG(
        mongo_url=os.getenv("MONGO_URL", "mongodb://localhost:27017/"),
        ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # Check current stats
    stats = rag.get_stats()
    print(f"\nCurrent stats: {stats}")

    # Load sample data if empty
    if stats["total_documents"] == 0:
        print("\nLoading sample text documents...")
        for doc in SAMPLE_TEXTS:
            rag.add_text(doc["content"], doc["metadata"])

        # Try to create and add a sample image
        sample_img = "sample_diagram.png"
        if create_sample_image(sample_img):
            print("\nAdding sample image...")
            rag.add_image(sample_img, {"topic": "architecture"})

        stats = rag.get_stats()
        print(f"\nUpdated stats: {stats}")

    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  <question>     - Ask a question")
    print("  image <path>   - Add an image")
    print("  stats          - Show statistics")
    print("  clear          - Clear all documents")
    print("  quit           - Exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYour input: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                break

            if user_input.lower() == "stats":
                print(rag.get_stats())
                continue

            if user_input.lower() == "clear":
                rag.clear()
                continue

            if user_input.lower().startswith("image "):
                image_path = user_input[6:].strip()
                try:
                    rag.add_image(image_path)
                except Exception as e:
                    print(f"Error adding image: {e}")
                continue

            # Regular query
            result = rag.query(user_input)

            print(f"\nAnswer: {result['answer']}")
            print(f"\nSources ({len(result['sources'])}):")
            for i, src in enumerate(result['sources'], 1):
                print(f"  {i}. [{src['type']}] {src['source']} (score: {src['score']:.3f})")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
