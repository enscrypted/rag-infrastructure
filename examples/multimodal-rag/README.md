# Multimodal RAG Example

This example demonstrates multimodal RAG (Retrieval-Augmented Generation) that processes both text documents and images using Ollama's vision models.

## What is Multimodal RAG?

Multimodal RAG extends traditional text-based RAG to handle multiple data types:
- **Text Documents**: Standard text processing with embeddings
- **Images**: Automatic image analysis and description using vision models
- **Combined Retrieval**: Search across both text and image content
- **Visual Q&A**: Answer questions about images in your knowledge base

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Text     │────▶│   Ollama    │────▶│  Embedding  │
│  Documents  │     │  (embed)    │     │   Vector    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
┌─────────────┐     ┌─────────────┐            │
│   Images    │────▶│   LLaVA     │            │
│             │     │  (vision)   │            │
└─────────────┘     └──────┬──────┘            │
                           │                    │
                    ┌──────▼──────┐            │
                    │ Description │            │
                    └──────┬──────┘            │
                           │                    │
                    ┌──────▼──────┐            │
                    │   Ollama    │────────────┤
                    │   (embed)   │            │
                    └──────┬──────┘            │
                           │                    │
                    ┌──────▼────────────────────▼──────┐
                    │         MongoDB Vector           │
                    │         (unified store)          │
                    └─────────────────────────────────┘
```

## Prerequisites

1. RAG Infrastructure Stack deployed
2. Python 3.8+
3. Required packages: `pymongo`, `requests`
4. Optional: `Pillow` for sample image generation
5. Ollama models:
   - `nomic-embed-text` (embeddings)
   - `llava` (vision model)
   - `llama3.2:3b` (text generation)

## Setup

```bash
# Install dependencies
pip install pymongo requests Pillow

# Pull required Ollama models
docker exec rag-ollama ollama pull nomic-embed-text
docker exec rag-ollama ollama pull llava
docker exec rag-ollama ollama pull llama3.2:3b
```

## Usage

### Basic Usage

```bash
python multimodal_rag.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGO_URL` | `mongodb://localhost:27017/` | MongoDB connection |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |

### Interactive Commands

- `<question>` - Ask a question about documents and images
- `image <path>` - Add an image to the knowledge base
- `stats` - Show document statistics
- `clear` - Clear all documents
- `quit` - Exit

### Example Session

```
============================================================
Multimodal RAG Demo
============================================================
Connected to MongoDB: multimodal_rag
Connected to Ollama: http://localhost:11434
Vision model available
Embedding model available

Loading sample text documents...
Text document added: 507f1f77bcf86cd799439011
Text document added: 507f1f77bcf86cd799439012
...

Adding sample image...
Analyzing image: sample_diagram.png...
Image added: 507f1f77bcf86cd799439016
  Description: The image shows a flowchart with three boxes...

============================================================
Interactive Mode
============================================================

Your input: What is computer vision used for?

Question: What is computer vision used for?
Found 2 text docs, 1 image docs

Answer: Computer vision enables machines to interpret visual information.
Key tasks include image classification, object detection, and semantic
segmentation. Deep learning models like CNNs and Vision Transformers
have revolutionized this field.

Sources (3):
  1. [text] cv_intro (score: 0.892)
  2. [text] ml_guide (score: 0.756)
  3. [image] sample_diagram.png (score: 0.634)

Your input: image /path/to/my/photo.jpg
Analyzing image: photo.jpg...
Image added: 507f1f77bcf86cd799439017
  Description: The image shows a sunset over mountains...
```

## How It Works

### 1. Text Document Processing

```python
rag = MultimodalRAG()

# Add text document
rag.add_text(
    content="Machine learning uses algorithms to learn from data.",
    metadata={"topic": "ml", "source": "tutorial"}
)
```

Text documents are:
1. Embedded using `nomic-embed-text`
2. Stored in MongoDB with the embedding vector
3. Searchable via vector similarity

### 2. Image Processing

```python
# Add image - automatically analyzed and embedded
rag.add_image(
    "/path/to/diagram.png",
    metadata={"topic": "architecture"}
)
```

Images are processed in multiple steps:
1. **Analysis**: LLaVA generates a detailed text description
2. **Embedding**: The description is embedded using `nomic-embed-text`
3. **Storage**: Both description and embedding stored in MongoDB
4. **Deduplication**: Images are hashed to prevent duplicates

### 3. Unified Search

```python
# Search across all document types
results = rag.search("neural networks", limit=5)

# Search specific type only
text_only = rag.search("neural networks", doc_type="text")
images_only = rag.search("neural networks", doc_type="image")
```

### 4. Multimodal Queries

```python
# Standard query (searches both text and images)
result = rag.query("What techniques are used in computer vision?")

# Query with an image input
result = rag.query_with_image(
    "What does this diagram show?",
    "/path/to/query_image.png"
)
```

## Supported Image Formats

- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)
- GIF (`.gif`)
- WebP (`.webp`)

## Adding Documents in Bulk

```python
# Add all files from a directory
rag.add_directory(
    "/path/to/documents",
    extensions=[".txt", ".md", ".png", ".jpg"]
)
```

## Customization

### Using Different Vision Models

```python
# In _analyze_image method, change the model
response = requests.post(
    f"{self.ollama_url}/api/generate",
    json={
        "model": "llava:13b",  # Use larger model
        "prompt": prompt,
        "images": [image_b64],
        "stream": False
    }
)
```

Available vision models in Ollama:
- `llava` (7B, default)
- `llava:13b` (better quality)
- `llava:34b` (best quality, requires more RAM)
- `llama3.2-vision` (Meta's vision model)

### Custom Image Analysis Prompts

```python
# Customize what information to extract from images
description = rag._analyze_image(
    image_path,
    prompt="""Analyze this technical diagram and extract:
    1. All labeled components
    2. Connections between components
    3. Data flow direction
    4. Any text or annotations"""
)
```

### Adjusting Search Weights

For hybrid text+image results, you can filter by type:

```python
# Get mixed results
all_results = rag.search(query, limit=10)

# Post-process to balance types
text_results = [r for r in all_results if r["doc_type"] == "text"][:3]
image_results = [r for r in all_results if r["doc_type"] == "image"][:2]
balanced = text_results + image_results
```

## Use Cases

### 1. Technical Documentation with Diagrams
- Index documentation PDFs and their diagrams
- Answer questions referencing both text and visuals

### 2. Product Catalogs
- Store product descriptions and images
- Enable visual product search

### 3. Educational Content
- Index textbooks with illustrations
- Answer questions about visual concepts

### 4. Research Papers
- Process papers with figures and charts
- Query about experimental results shown in graphs

## Performance Considerations

### Image Processing Time
- LLaVA analysis takes 5-30 seconds per image depending on model size
- Consider batch processing for large image collections

### Memory Usage
- `llava:7b` requires ~8GB RAM
- `llava:13b` requires ~16GB RAM
- `llava:34b` requires ~32GB RAM

### Storage
- Only image descriptions are stored (not raw images)
- Original images referenced by file path

## Troubleshooting

**"No vision model found"**
```bash
# Install LLaVA
docker exec rag-ollama ollama pull llava
```

**"Image analysis timeout"**
- Large images take longer to process
- Resize images before adding (recommended: max 1024px)
- Use a smaller model (`llava:7b`)

**"Empty image description"**
- Check image file is valid and readable
- Verify LLaVA model is loaded: `ollama list`
- Check Ollama logs: `docker compose logs ollama`

**"Poor search results for images"**
- Image descriptions may not capture all relevant details
- Try custom analysis prompts for your domain
- Consider adding manual tags in metadata

## Learn More

- [Ollama Vision Models](https://ollama.com/blog/vision-models)
- [MongoDB Vector Search](../../docs/services/mongodb.md)
- [LLaVA Model](https://ollama.com/library/llava)
- [CLIP Embeddings (OpenAI)](https://cookbook.openai.com/examples/custom_image_embedding_search)
- [Multimodal RAG Guide](https://medium.com/kx-systems/guide-to-multimodal-rag-for-images-and-text-10dab36e3117)
