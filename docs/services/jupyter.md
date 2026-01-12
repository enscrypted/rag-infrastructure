# Jupyter Lab Development Environment

Jupyter Lab provides an interactive development environment perfect for RAG experimentation, data analysis, and prototyping.

## Quick Access

| Component | URL | Credentials |
|-----------|-----|-------------|
| **Jupyter Lab** | `http://your-host:8888` | Token: your-password |
| **File Browser** | Built into interface | - |

## Why Jupyter for RAG?

### Benefits
- **Interactive Development**: Test code snippets and see results immediately
- **Data Visualization**: Analyze retrieval quality and model performance
- **Notebook Documentation**: Document your RAG experiments
- **Package Management**: Install and test new libraries easily
- **Git Integration**: Version control for your experiments

### RAG Use Cases
- **Prototype Development**: Build and test RAG pipelines interactively
- **Data Analysis**: Analyze document embeddings and retrieval performance
- **Model Evaluation**: Compare different models and approaches
- **Research Notebooks**: Document experiments and findings
- **Tutorial Creation**: Create step-by-step RAG tutorials

## Initial Setup

### 1. Access Jupyter Lab
```bash
# Navigate to Jupyter Lab
open http://your-host:8888

# Enter token when prompted: your-password
# Or use the full URL with token:
open http://your-host:8888/?token=your-password
```

### 2. Verify Environment
```python
# Create new notebook and test basic functionality
import sys
print(f"Python version: {sys.version}")

# Check available packages
import importlib
packages = ['numpy', 'pandas', 'requests', 'pymongo', 'matplotlib']
for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"✅ {pkg} available")
    except ImportError:
        print(f"❌ {pkg} not available")
```

### 3. Install RAG Dependencies
```python
# Install additional packages for RAG development
!pip install langfuse chromadb neo4j sentence-transformers plotly seaborn

# Restart kernel after installation
```

## RAG Development Workflow

### 1. Notebook Structure
```python
# Recommended notebook structure for RAG experiments

# Cell 1: Imports and Configuration
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from langfuse import Langfuse
import json
from datetime import datetime

# Configuration
MONGO_URL = "mongodb://mongodb:27017/"
OLLAMA_URL = "http://ollama:11434"
LANGFUSE_HOST = "http://langfuse:3000"

# Cell 2: Helper Functions
def get_embedding(text, model="nomic-embed-text"):
    """Generate embedding using Ollama"""
    response = requests.post(f"{OLLAMA_URL}/api/embeddings", json={
        "model": model,
        "prompt": text
    })
    return response.json()["embedding"]

def vector_search(query, collection, k=5):
    """Perform vector similarity search"""
    query_vector = get_embedding(query)
    pipeline = [{
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_vector,
            "path": "embedding",
            "limit": k
        }
    }]
    return list(collection.aggregate(pipeline))

# Cell 3: Data Exploration
# Your exploration and experimentation code here...
```

### 2. Interactive RAG Development
```python
# Example: Building and testing a RAG system interactively

# Connect to services
client = MongoClient(MONGO_URL)
collection = client.rag_experiments.documents

# Add sample documents
sample_docs = [
    "Jupyter Lab is an interactive development environment for data science",
    "RAG combines retrieval and generation for better AI responses",
    "Vector databases enable semantic search using embeddings"
]

for i, doc in enumerate(sample_docs):
    collection.insert_one({
        "id": f"doc_{i}",
        "content": doc,
        "embedding": get_embedding(doc),
        "timestamp": datetime.now()
    })

print(f"✅ Added {len(sample_docs)} documents")

# Test retrieval
query = "What is Jupyter Lab?"
results = vector_search(query, collection, k=3)

print(f"\n🔍 Query: {query}")
print(f"📄 Found {len(results)} results:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['content']}")
```

### 3. Visualization and Analysis
```python
# Analyze embedding distributions
import matplotlib.pyplot as plt
import numpy as np

# Get all embeddings
docs = list(collection.find({}, {"embedding": 1, "content": 1}))
embeddings = [doc["embedding"] for doc in docs]
texts = [doc["content"] for doc in docs]

# Visualize embedding dimensions
embedding_matrix = np.array(embeddings)

plt.figure(figsize=(12, 6))

# Plot 1: Embedding distribution
plt.subplot(1, 2, 1)
plt.hist(embedding_matrix.flatten(), bins=50, alpha=0.7)
plt.title("Embedding Value Distribution")
plt.xlabel("Embedding Values")
plt.ylabel("Frequency")

# Plot 2: Dimensionality analysis
plt.subplot(1, 2, 2)
mean_values = np.mean(embedding_matrix, axis=0)
plt.plot(mean_values[:100])  # First 100 dimensions
plt.title("Mean Embedding Values (First 100 Dims)")
plt.xlabel("Dimension")
plt.ylabel("Mean Value")

plt.tight_layout()
plt.show()

# Calculate similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(embedding_matrix)

# Visualize similarity heatmap
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(similarity_matrix, 
            xticklabels=[f"Doc {i}" for i in range(len(texts))],
            yticklabels=[f"Doc {i}" for i in range(len(texts))],
            annot=True, cmap='viridis')
plt.title("Document Similarity Matrix")
plt.show()
```

## Advanced RAG Experiments

### 1. Model Comparison
```python
# Compare different embedding models
models = ["nomic-embed-text", "all-minilm"]
query = "machine learning algorithms"

results_by_model = {}

for model in models:
    print(f"\n🧪 Testing model: {model}")
    
    # Generate embeddings with this model
    query_embedding = get_embedding(query, model)
    
    # Update documents with new embeddings (for comparison)
    # In practice, you'd have separate collections per model
    
    # Perform search and store results
    # results_by_model[model] = search_results
    
# Compare and visualize results
```

### 2. Retrieval Quality Analysis
```python
# Analyze retrieval quality with ground truth data
import pandas as pd

def evaluate_retrieval(queries_and_expected, collection, k=5):
    """Evaluate retrieval quality"""
    results = []
    
    for query, expected_docs in queries_and_expected:
        retrieved = vector_search(query, collection, k)
        retrieved_ids = [doc.get("id", "") for doc in retrieved]
        
        # Calculate metrics
        precision_at_k = len(set(retrieved_ids) & set(expected_docs)) / k
        recall_at_k = len(set(retrieved_ids) & set(expected_docs)) / len(expected_docs)
        
        results.append({
            "query": query,
            "precision@k": precision_at_k,
            "recall@k": recall_at_k,
            "retrieved": retrieved_ids
        })
    
    return pd.DataFrame(results)

# Example evaluation
test_queries = [
    ("What is Jupyter?", ["doc_0"]),
    ("How does RAG work?", ["doc_1"]),
    # Add more test cases...
]

eval_results = evaluate_retrieval(test_queries, collection)
print(eval_results)

# Visualize results
eval_results[["precision@k", "recall@k"]].plot(kind="bar", figsize=(10, 6))
plt.title("Retrieval Quality Metrics")
plt.ylabel("Score")
plt.show()
```

### 3. Interactive Parameter Tuning
```python
# Interactive widgets for parameter tuning
from ipywidgets import interact, IntSlider, FloatSlider

@interact(
    k=IntSlider(min=1, max=10, value=5, description="Top K:"),
    temperature=FloatSlider(min=0.1, max=2.0, value=0.7, step=0.1, description="Temperature:")
)
def interactive_rag(k, temperature):
    """Interactive RAG parameter tuning"""
    
    query = "Tell me about vector databases"
    
    # Retrieve documents
    docs = vector_search(query, collection, k)
    context = "\n".join([doc["content"] for doc in docs])
    
    # Generate response with specified temperature
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    response = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    })
    
    print(f"🔍 Retrieved {len(docs)} documents")
    print(f"🤖 Response (temp={temperature}):")
    print(response.json()["response"])
    print(f"\n📄 Sources:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc['content'][:100]}...")
```

## Data Science Integration

### 1. Document Analysis
```python
# Analyze your document corpus
docs = list(collection.find({}))
df = pd.DataFrame(docs)

# Basic statistics
print(f"📊 Corpus Statistics:")
print(f"Total documents: {len(df)}")
print(f"Average content length: {df['content'].str.len().mean():.1f} chars")
print(f"Vocabulary size: {len(set(' '.join(df['content']).split()))}")

# Content length distribution
plt.figure(figsize=(10, 4))
plt.hist(df['content'].str.len(), bins=20, alpha=0.7)
plt.title("Document Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Count")
plt.show()
```

### 2. Embedding Space Exploration
```python
# Use dimensionality reduction to visualize embeddings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Get embeddings matrix
embeddings = np.array([doc["embedding"] for doc in docs])

# PCA reduction
pca = PCA(n_components=2)
embeddings_2d_pca = pca.fit_transform(embeddings)

# t-SNE reduction
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d_tsne = tsne.fit_transform(embeddings)

# Plot both
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# PCA plot
ax1.scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], alpha=0.7)
ax1.set_title("PCA Visualization")
ax1.set_xlabel("First Principal Component")
ax1.set_ylabel("Second Principal Component")

# t-SNE plot
ax2.scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1], alpha=0.7)
ax2.set_title("t-SNE Visualization")
ax2.set_xlabel("t-SNE 1")
ax2.set_ylabel("t-SNE 2")

plt.tight_layout()
plt.show()
```

## Collaboration and Sharing

### 1. Export Notebooks
```python
# Save notebook with outputs for sharing
# File → Export Notebook As... → HTML/PDF

# Or programmatically export
!jupyter nbconvert --to html your_notebook.ipynb
!jupyter nbconvert --to pdf your_notebook.ipynb  # Requires LaTeX
```

### 2. Git Integration
```bash
# Initialize git repository in Jupyter terminal
!git init
!git add *.ipynb
!git commit -m "Initial RAG experiments"

# Push to remote repository
!git remote add origin your-repo-url
!git push -u origin main
```

### 3. Notebook Templates
Create reusable templates for common RAG tasks:

```python
# Template: RAG Evaluation Notebook
# Save as: templates/rag_evaluation_template.ipynb

"""
RAG System Evaluation Template

This notebook provides a standard framework for evaluating RAG systems.
Fill in the sections below with your specific implementation.
"""

# 1. Configuration
# TODO: Set your service URLs and credentials

# 2. Data Loading
# TODO: Load your test dataset

# 3. Model Setup
# TODO: Initialize your RAG system

# 4. Evaluation Metrics
# TODO: Define your evaluation criteria

# 5. Run Evaluation
# TODO: Execute evaluation pipeline

# 6. Results Analysis
# TODO: Analyze and visualize results
```

## Best Practices

### Notebook Organization
1. **Clear Structure**: Use consistent cell organization
2. **Documentation**: Add markdown cells explaining each step
3. **Modular Code**: Define reusable functions
4. **Version Control**: Regular commits of working notebooks

### Performance Tips
1. **Memory Management**: Clear large variables when done
2. **Chunked Processing**: Process large datasets in chunks
3. **Async Operations**: Use async for I/O operations when possible
4. **Resource Monitoring**: Monitor memory and CPU usage

### Reproducibility
1. **Fixed Seeds**: Set random seeds for reproducible results
2. **Environment Documentation**: Document package versions
3. **Data Versioning**: Track dataset versions
4. **Configuration Management**: Use config files for parameters

## Troubleshooting

### Common Issues

**"Kernel not responding"**
```python
# Restart kernel: Kernel → Restart Kernel
# Or restart from notebook:
import os
os._exit(0)
```

**"Memory error"**
```python
# Monitor memory usage
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")

# Clear variables
del large_variable
import gc
gc.collect()
```

**"Package not found"**
```bash
# Install from notebook
!pip install package-name

# Or use conda
!conda install package-name

# Restart kernel after installation
```

## Learning Resources

- [Jupyter Documentation](https://jupyter.org/documentation) - Official documentation
- [Jupyter Book](https://jupyterbook.org/) - Create online books from notebooks
- [nbviewer](https://nbviewer.org/) - Share notebooks online
- [Jupyter Tips & Tricks](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/) - Power user guide