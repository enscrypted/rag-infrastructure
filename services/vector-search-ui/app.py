"""
Modern Vector Search UI for MongoDB
A clean, developer-friendly interface for RAG experimentation
"""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import json
import os

app = FastAPI(title="MongoDB Vector Search UI", version="1.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# MongoDB connection
MONGO_URL = os.getenv("MONGO_URL", "mongodb://mongodb:27017")
client = AsyncIOMotorClient(MONGO_URL)
db = client.rag_database

class Collection(BaseModel):
    name: str
    count: int
    indexes: List[str]

class Document(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    similarity_score: Optional[float] = None

class VectorSearchRequest(BaseModel):
    collection: str
    query_text: str
    query_vector: Optional[List[float]] = None
    k: int = 10
    filters: Optional[Dict[str, Any]] = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main dashboard"""
    collections = await db.list_collection_names()
    
    collection_stats = []
    for coll_name in collections:
        coll = db[coll_name]
        count = await coll.count_documents({})
        indexes = await coll.list_indexes().to_list(None)
        
        # Check for vector indexes
        has_vector = any('vector' in str(idx) for idx in indexes)
        
        collection_stats.append({
            'name': coll_name,
            'count': count,
            'has_vector': has_vector,
            'indexes': [idx['name'] for idx in indexes]
        })
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "collections": collection_stats,
        "mongo_url": MONGO_URL
    })

@app.get("/collection/{collection_name}", response_class=HTMLResponse)
async def view_collection(request: Request, collection_name: str):
    """View collection details and search interface"""
    collection = db[collection_name]
    
    # Get sample documents
    sample_docs = await collection.find().limit(10).to_list(10)
    
    # Get collection stats
    count = await collection.count_documents({})
    indexes = await collection.list_indexes().to_list(None)
    
    # Check if collection has vector search capability
    has_vector = any('vector' in str(idx) for idx in indexes)
    
    return templates.TemplateResponse("collection.html", {
        "request": request,
        "collection_name": collection_name,
        "document_count": count,
        "sample_docs": sample_docs,
        "indexes": indexes,
        "has_vector": has_vector
    })

@app.post("/api/vector-search")
async def vector_search(search_request: VectorSearchRequest):
    """Perform vector similarity search"""
    collection = db[search_request.collection]
    
    # Build the aggregation pipeline
    pipeline = []
    
    # If query vector provided, use it directly
    if search_request.query_vector:
        pipeline.append({
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": search_request.query_vector,
                "path": "embedding",
                "numCandidates": search_request.k * 10,
                "limit": search_request.k
            }
        })
    else:
        # For text queries, you can integrate with embedding models like:
        # - OpenAI text-embedding-ada-002
        # - Sentence Transformers 
        # - Ollama embeddings
        return JSONResponse({
            "error": "Text queries require embedding model integration. Please provide a query vector as JSON array, or integrate an embedding model in the backend.",
            "example": "Use: [0.1, -0.2, 0.3, ...] with your vector embeddings",
            "suggested_models": ["text-embedding-ada-002", "sentence-transformers/all-MiniLM-L6-v2", "ollama embeddings"]
        }, status_code=400)
    
    # Add filters if provided
    if search_request.filters:
        pipeline.append({"$match": search_request.filters})
    
    # Add similarity score
    pipeline.append({
        "$addFields": {
            "similarity_score": {"$meta": "vectorSearchScore"}
        }
    })
    
    # Execute search
    try:
        results = await collection.aggregate(pipeline).to_list(search_request.k)
        
        # Format results
        documents = []
        for doc in results:
            documents.append({
                "id": str(doc.get("_id", "")),
                "content": doc.get("content", doc.get("text", "")),
                "metadata": {k: v for k, v in doc.items() 
                           if k not in ["_id", "embedding", "content", "text", "similarity_score"]},
                "similarity_score": doc.get("similarity_score", 0)
            })
        
        return JSONResponse({
            "success": True,
            "results": documents,
            "count": len(documents)
        })
        
    except Exception as e:
        return JSONResponse({
            "error": f"Search failed: {str(e)}"
        }, status_code=500)

@app.post("/api/create-index")
async def create_vector_index(collection_name: str = Form(...), 
                              dimension: int = Form(1536),
                              similarity: str = Form("cosine")):
    """Create a vector search index"""
    collection = db[collection_name]
    
    try:
        # Create vector search index
        index_spec = {
            "definition": {
                "vectorSearchType": "knn",
                "fields": [{
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": dimension,
                    "similarity": similarity
                }]
            },
            "name": "vector_index"
        }
        
        await collection.create_search_index(index_spec)
        
        return JSONResponse({
            "success": True,
            "message": f"Vector index created for {collection_name}"
        })
        
    except Exception as e:
        return JSONResponse({
            "error": f"Failed to create index: {str(e)}"
        }, status_code=500)

@app.post("/api/insert-document")
async def insert_document(
    collection_name: str = Form(...),
    content: str = Form(...),
    metadata: str = Form("{}"),
    embedding: str = Form(None)
):
    """Insert a document with optional embedding"""
    collection = db[collection_name]
    
    try:
        # Safely parse metadata JSON
        try:
            parsed_metadata = json.loads(metadata) if metadata else {}
            if not isinstance(parsed_metadata, dict):
                raise ValueError("Metadata must be a valid JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse({
                "error": f"Invalid metadata JSON: {str(e)}"
            }, status_code=400)
        
        doc = {
            "content": content,
            "metadata": parsed_metadata,
            "created_at": datetime.utcnow()
        }
        
        # Safely parse embedding JSON
        if embedding:
            try:
                parsed_embedding = json.loads(embedding)
                if not isinstance(parsed_embedding, list) or not all(isinstance(x, (int, float)) for x in parsed_embedding):
                    raise ValueError("Embedding must be a list of numbers")
                doc["embedding"] = parsed_embedding
            except (json.JSONDecodeError, ValueError) as e:
                return JSONResponse({
                    "error": f"Invalid embedding JSON: {str(e)}"
                }, status_code=400)
        
        result = await collection.insert_one(doc)
        
        return JSONResponse({
            "success": True,
            "id": str(result.inserted_id)
        })
        
    except Exception as e:
        return JSONResponse({
            "error": f"Failed to insert document: {str(e)}"
        }, status_code=500)

@app.get("/api/collections")
async def list_collections():
    """List all collections with stats"""
    collections = await db.list_collection_names()
    
    stats = []
    for name in collections:
        coll = db[name]
        count = await coll.count_documents({})
        stats.append({"name": name, "count": count})
    
    return JSONResponse({"collections": stats})

@app.delete("/api/collection/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection"""
    try:
        await db.drop_collection(collection_name)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({
            "error": f"Failed to delete collection: {str(e)}"
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)