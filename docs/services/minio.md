# MinIO Object Storage Guide

MinIO provides S3-compatible object storage for the RAG infrastructure stack. Essential for storing documents, model artifacts, backups, and large datasets.

## Quick Access

| Component | URL | Credentials |
|-----------|-----|-------------|
| **MinIO Console** | `http://your-host:9001` | admin / your-password |
| **S3 API Endpoint** | `http://your-host:9000` | admin / your-password |

## Why MinIO for RAG?

### Benefits
- **S3-Compatible**: Drop-in replacement for AWS S3
- **Local Storage**: No cloud dependencies or costs
- **High Performance**: Optimized for large file operations
- **Versioning**: Track document and model changes
- **Backup Storage**: Reliable storage for database backups

### RAG Use Cases
- **Document Storage**: Store original documents before processing
- **Model Artifacts**: Save fine-tuned models and embeddings
- **Data Lake**: Organize datasets by project or experiment
- **Backup Repository**: Regular backups of databases and configurations
- **Static Assets**: Store images, PDFs, and other media files

## Initial Setup

### 1. Access MinIO Console
```bash
# Navigate to MinIO Console
open http://your-host:9001

# Login with credentials:
# Username: admin
# Password: your-password
```

### 2. Create Your First Bucket
```bash
# In the MinIO console:
1. Click "Create Bucket"
2. Enter bucket name: "rag-documents"
3. Set region: "us-east-1" (default)
4. Click "Create Bucket"
```

### 3. Test with Python Client
```python
from minio import Minio

# Initialize MinIO client
minio_client = Minio(
    "your-host:9000",
    access_key="admin",
    secret_key="your-password",
    secure=False  # Use True for HTTPS
)

# Test connection
buckets = minio_client.list_buckets()
for bucket in buckets:
    print(f"Bucket: {bucket.name}, Created: {bucket.creation_date}")
```

## Document Management for RAG

### 1. Organizing RAG Data Structure
```python
# Recommended bucket structure for RAG workflows
bucket_structure = {
    "rag-documents": "Original documents (PDFs, text files)",
    "rag-processed": "Processed and chunked documents", 
    "rag-embeddings": "Pre-computed embedding vectors",
    "rag-models": "Fine-tuned models and artifacts",
    "rag-backups": "Database backups and exports"
}

# Create buckets
for bucket_name, description in bucket_structure.items():
    try:
        minio_client.make_bucket(bucket_name)
        print(f"Created bucket: {bucket_name} - {description}")
    except Exception as e:
        print(f"Bucket {bucket_name} already exists or error: {e}")
```

### 2. Document Upload and Processing Pipeline
```python
import os
from datetime import datetime

class RAGDocumentManager:
    def __init__(self, minio_client):
        self.minio = minio_client
        self.documents_bucket = "rag-documents"
        self.processed_bucket = "rag-processed"
    
    def upload_document(self, file_path, document_id=None):
        """Upload document with metadata"""
        
        if not document_id:
            document_id = os.path.basename(file_path)
        
        # Add timestamp prefix for versioning
        object_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{document_id}"
        
        # Upload with metadata
        metadata = {
            "Content-Type": self.get_content_type(file_path),
            "X-Original-Name": document_id,
            "X-Upload-Time": datetime.now().isoformat(),
            "X-File-Size": str(os.path.getsize(file_path))
        }
        
        self.minio.fput_object(
            bucket_name=self.documents_bucket,
            object_name=object_name,
            file_path=file_path,
            metadata=metadata
        )
        
        print(f"Uploaded: {file_path} -> {object_name}")
        return object_name
    
    def get_content_type(self, file_path):
        """Determine content type from file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        content_types = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.html': 'text/html'
        }
        return content_types.get(ext, 'application/octet-stream')
    
    def list_documents(self, prefix=""):
        """List all documents with metadata"""
        objects = self.minio.list_objects(
            bucket_name=self.documents_bucket,
            prefix=prefix,
            recursive=True
        )
        
        documents = []
        for obj in objects:
            # Get object metadata
            stat = self.minio.stat_object(self.documents_bucket, obj.object_name)
            
            documents.append({
                "name": obj.object_name,
                "size": obj.size,
                "last_modified": obj.last_modified,
                "etag": obj.etag,
                "metadata": stat.metadata
            })
        
        return documents
    
    def download_document(self, object_name, local_path):
        """Download document to local path"""
        self.minio.fget_object(
            bucket_name=self.documents_bucket,
            object_name=object_name,
            file_path=local_path
        )
        return local_path

# Usage example
doc_manager = RAGDocumentManager(minio_client)

# Upload a document
uploaded_name = doc_manager.upload_document("/path/to/document.pdf", "research_paper_001")

# List all documents
documents = doc_manager.list_documents()
for doc in documents:
    print(f"{doc['name']} - {doc['size']} bytes - {doc['last_modified']}")
```

### 3. Processed Data Storage
```python
def store_processed_chunks(self, document_id, chunks, embeddings=None):
    """Store processed document chunks with optional embeddings"""
    
    import json
    
    # Prepare processed data
    processed_data = {
        "document_id": document_id,
        "processing_time": datetime.now().isoformat(),
        "chunk_count": len(chunks),
        "chunks": []
    }
    
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "chunk_id": f"{document_id}_chunk_{i}",
            "content": chunk,
            "char_count": len(chunk),
            "embedding": embeddings[i] if embeddings else None
        }
        processed_data["chunks"].append(chunk_data)
    
    # Store as JSON
    json_data = json.dumps(processed_data, indent=2)
    object_name = f"{document_id}_processed.json"
    
    # Upload processed data
    from io import BytesIO
    data_stream = BytesIO(json_data.encode('utf-8'))
    
    self.minio.put_object(
        bucket_name=self.processed_bucket,
        object_name=object_name,
        data=data_stream,
        length=len(json_data),
        content_type="application/json"
    )
    
    print(f"Stored processed chunks: {object_name}")
    return object_name
```

## Backup and Restore Operations

### 1. Database Backup Storage
```python
def backup_mongodb_to_minio(self):
    """Backup MongoDB and store in MinIO"""
    
    import subprocess
    import tempfile
    
    # Create temporary backup file
    with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as tmp_file:
        backup_path = tmp_file.name
    
    # Run mongodump
    backup_cmd = [
        "docker", "exec", "rag-mongodb", 
        "mongodump", "--archive", "--gzip"
    ]
    
    with open(backup_path, 'wb') as backup_file:
        subprocess.run(backup_cmd, stdout=backup_file, check=True)
    
    # Upload to MinIO
    backup_name = f"mongodb_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gz"
    
    minio_client.fput_object(
        bucket_name="rag-backups",
        object_name=backup_name,
        file_path=backup_path,
        metadata={
            "X-Backup-Type": "mongodb",
            "X-Backup-Time": datetime.now().isoformat()
        }
    )
    
    # Clean up
    os.unlink(backup_path)
    print(f"MongoDB backup stored: {backup_name}")
    return backup_name

def backup_neo4j_to_minio(self):
    """Backup Neo4j database and store in MinIO"""
    
    # Similar pattern for Neo4j backup
    backup_cmd = [
        "docker", "exec", "rag-neo4j",
        "neo4j-admin", "database", "dump", "neo4j", "--to-path=/tmp"
    ]
    
    subprocess.run(backup_cmd, check=True)
    
    # Copy from container and upload to MinIO
    # Implementation details...
```

### 2. Model Artifact Management
```python
def store_model_artifacts(self, model_name, model_path, metadata=None):
    """Store trained models and artifacts"""
    
    import tarfile
    import tempfile
    
    # Create tar archive of model directory
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
        archive_path = tmp_file.name
    
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(model_path, arcname=model_name)
    
    # Upload model archive
    object_name = f"models/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
    
    upload_metadata = {
        "X-Model-Name": model_name,
        "X-Model-Version": metadata.get("version", "1.0") if metadata else "1.0",
        "X-Training-Date": datetime.now().isoformat(),
        "X-Model-Type": metadata.get("type", "unknown") if metadata else "unknown"
    }
    
    minio_client.fput_object(
        bucket_name="rag-models",
        object_name=object_name,
        file_path=archive_path,
        metadata=upload_metadata
    )
    
    # Clean up
    os.unlink(archive_path)
    print(f"Model archived: {object_name}")
    return object_name
```

## Web Console Usage

### 1. Console Navigation
- **Buckets**: View and manage all buckets
- **Browser**: Navigate files within buckets
- **Access Keys**: Manage API credentials
- **Settings**: Configure server settings
- **Monitoring**: View usage statistics

### 2. Bucket Management
```bash
# Console operations:
1. Create Bucket: Click "+" → Enter name → Set policy
2. Upload Files: Drag & drop or click "Upload" 
3. Set Permissions: Bucket → Access → Policy settings
4. Enable Versioning: Bucket → Settings → Versioning
```

### 3. Access Policies
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"AWS": "*"},
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::rag-documents/*"
    }
  ]
}
```

## Integration with RAG Pipeline

### 1. Document Processing Workflow
```python
async def process_document_workflow(file_path):
    """Complete document processing workflow with MinIO storage"""
    
    # 1. Upload original document
    doc_manager = RAGDocumentManager(minio_client)
    uploaded_name = doc_manager.upload_document(file_path)
    
    # 2. Download for processing
    with tempfile.NamedTemporaryFile() as tmp_file:
        doc_manager.download_document(uploaded_name, tmp_file.name)
        
        # 3. Process document (extract text, chunk, embed)
        chunks = extract_text_and_chunk(tmp_file.name)
        embeddings = generate_embeddings(chunks)
        
        # 4. Store processed data
        processed_name = doc_manager.store_processed_chunks(
            uploaded_name, chunks, embeddings
        )
        
        # 5. Store in vector database
        store_in_mongodb(chunks, embeddings, uploaded_name)
        
    return {
        "original": uploaded_name,
        "processed": processed_name,
        "chunks": len(chunks)
    }
```

### 2. Model Management
```python
def deploy_model_from_minio(model_object_name):
    """Download and deploy model from MinIO storage"""
    
    import tempfile
    import tarfile
    
    # Download model archive
    with tempfile.NamedTemporaryFile(suffix='.tar.gz') as tmp_file:
        minio_client.fget_object(
            bucket_name="rag-models",
            object_name=model_object_name,
            file_path=tmp_file.name
        )
        
        # Extract model
        with tarfile.open(tmp_file.name, 'r:gz') as tar:
            tar.extractall("/tmp/models/")
        
        # Load and deploy model
        # Implementation specific to your model framework
        
    return f"Model {model_object_name} deployed successfully"
```

## Monitoring and Analytics

### 1. Storage Usage Monitoring
```python
def get_storage_analytics():
    """Get storage usage analytics"""
    
    analytics = {}
    
    # Get bucket sizes
    for bucket_name in ["rag-documents", "rag-processed", "rag-models", "rag-backups"]:
        total_size = 0
        object_count = 0
        
        objects = minio_client.list_objects(bucket_name, recursive=True)
        for obj in objects:
            total_size += obj.size
            object_count += 1
        
        analytics[bucket_name] = {
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024*1024), 2),
            "object_count": object_count
        }
    
    return analytics

# Usage
storage_stats = get_storage_analytics()
for bucket, stats in storage_stats.items():
    print(f"{bucket}: {stats['size_mb']} MB, {stats['object_count']} objects")
```

### 2. Access Logging
```python
# Enable audit logging in MinIO configuration
# Set environment variables:
# MINIO_AUDIT_WEBHOOK_ENABLE=on
# MINIO_AUDIT_WEBHOOK_ENDPOINT=http://your-logging-service

def analyze_access_patterns():
    """Analyze document access patterns"""
    
    # Parse MinIO audit logs to understand:
    # - Most accessed documents
    # - Access frequency
    # - User patterns
    # - Error rates
    
    pass  # Implementation depends on logging setup
```

## Performance Optimization

### 1. Configuration Tuning
```bash
# Environment variables for performance
MINIO_CACHE_DRIVES=/mnt/cache1,/mnt/cache2
MINIO_CACHE_EXCLUDE="*.tmp,*.log"
MINIO_COMPRESS_ENABLED=on
MINIO_COMPRESS_EXTENSIONS=.txt,.log,.csv,.json
```

### 2. Large File Handling
```python
def upload_large_file(file_path, bucket_name, object_name):
    """Upload large files with multipart upload"""
    
    # MinIO Python client automatically uses multipart for files > 64MB
    # Configure part size for optimal performance
    
    from minio.commonconfig import REPLACE, CopySource
    
    # For very large files, use streaming upload
    with open(file_path, 'rb') as file_data:
        file_stat = os.stat(file_path)
        
        minio_client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=file_data,
            length=file_stat.st_size,
            content_type='application/octet-stream',
            part_size=10*1024*1024  # 10MB parts
        )
```

## Troubleshooting

### Common Issues

**"Connection refused"**
```bash
# Check MinIO service status
docker compose ps minio

# Check MinIO logs
docker compose logs minio

# Test connectivity
curl http://your-host:9000/minio/health/live
```

**"Access denied"**
```python
# Verify credentials
try:
    minio_client.list_buckets()
    print("Credentials valid")
except Exception as e:
    print(f"Authentication failed: {e}")
```

**"Disk space issues"**
```bash
# Check Docker volume usage
docker system df

# Check MinIO data directory
du -sh /var/lib/docker/volumes/rag-infrastructure_minio_data
```

**"Slow upload/download"**
```python
# Use appropriate part size for large files
# Increase concurrent connections
minio_client = Minio(
    "your-host:9000",
    access_key="admin",
    secret_key="your-password",
    secure=False,
    # Increase timeout for large operations
    timeout=300
)
```

## Security Best Practices

### 1. Access Control
```bash
# Create read-only user for applications
mc admin user add minio-alias readonly-user readonly-password
mc admin policy attach minio-alias readonly readonly-user
```

### 2. Encryption
```bash
# Enable server-side encryption
# Add to docker-compose.yml environment:
MINIO_KMS_AUTO_ENCRYPTION: "on"
MINIO_KMS_SECRET_KEY: "your-encryption-key"
```

### 3. Backup Strategy
```bash
# Regular backup automation
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
mc mirror minio-alias/rag-documents backup-location/documents_$DATE
mc mirror minio-alias/rag-models backup-location/models_$DATE
```

## Learning Resources

- [MinIO Documentation](https://docs.min.io/) - Official documentation
- [Python Client API](https://docs.min.io/docs/python-client-api-reference.html) - Complete SDK reference
- [S3 Compatibility](https://docs.min.io/docs/aws-cli-with-minio.html) - Using AWS tools with MinIO
- [Performance Tuning](https://docs.min.io/docs/minio-server-configuration-guide.html) - Optimization guide