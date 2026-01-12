# Redis & RedisInsight Guide

Redis provides high-performance in-memory caching and data structures, while RedisInsight offers a powerful web-based management interface. Essential for RAG response caching and session management.

## Quick Access

| Component | URL | Purpose |
|-----------|-----|---------|
| **RedisInsight UI** | `http://your-host:8001` | Web interface for Redis management |
| **Redis Server** | `redis://your-host:6379` | Redis database connection |
| **Redis CLI** | `docker exec -it rag-redis redis-cli` | Command line interface |

## Why Redis for RAG?

### Performance Benefits
- **In-Memory Storage**: Sub-millisecond response times
- **Caching**: Reduce expensive LLM and embedding calls
- **Session Management**: Track user conversations and context
- **Real-time Operations**: Pub/sub for live updates
- **Data Structures**: Lists, sets, hashes for complex RAG workflows

### RAG Use Cases
- **Response Caching**: Cache generated responses for repeated queries
- **Embedding Cache**: Store computed embeddings to avoid recomputation
- **Session Storage**: Maintain conversation history and context
- **Rate Limiting**: Control API usage and user requests
- **Real-time Analytics**: Track queries, performance metrics
- **Background Jobs**: Queue document processing tasks

## Initial Setup

### 1. Access RedisInsight
```bash
# Navigate to RedisInsight web interface
open http://your-host:8001

# First time setup:
1. Click "Add Redis Database"
2. Host: redis (Docker internal name) or your-host
3. Port: 6379
4. Name: RAG Redis
5. Click "Add Redis Database"
```

### 2. Connect with Python
```python
import redis
import json
from datetime import datetime, timedelta
import hashlib

# Initialize Redis client
r = redis.Redis(host='your-host', port=6379, decode_responses=True, db=0)

# Test connection
try:
    r.ping()
    print("Connected to Redis")
except redis.ConnectionError:
    print("Could not connect to Redis")

# Get server info
info = r.info()
print(f"Redis version: {info['redis_version']}")
print(f"Used memory: {info['used_memory_human']}")
```

### 3. Basic Operations
```python
# String operations
r.set("test_key", "Hello Redis")
value = r.get("test_key")
print(f"Retrieved: {value}")

# Set with expiration (TTL)
r.setex("temp_key", 60, "Expires in 60 seconds")

# Hash operations
r.hset("user:123", mapping={
    "name": "John Doe",
    "email": "john@example.com",
    "last_login": datetime.now().isoformat()
})

user_data = r.hgetall("user:123")
print(f"User data: {user_data}")

# List operations
r.lpush("recent_queries", "What is machine learning?")
r.lpush("recent_queries", "How does RAG work?")
recent = r.lrange("recent_queries", 0, 4)  # Get last 5 queries
print(f"Recent queries: {recent}")
```

## RAG Caching Implementation

### 1. Response Caching System
```python
class RAGCacheManager:
    def __init__(self, redis_host='your-host', redis_port=6379, default_ttl=3600):
        self.redis = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            decode_responses=True,
            db=0
        )
        self.default_ttl = default_ttl
        
    def _generate_cache_key(self, query, context_hash=None):
        """Generate consistent cache key for query"""
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Create hash
        key_data = f"{normalized_query}:{context_hash or ''}"
        cache_key = hashlib.md5(key_data.encode()).hexdigest()
        
        return f"rag:response:{cache_key}"
    
    def cache_response(self, query, response, context_documents=None, 
                      user_id=None, ttl=None):
        """Cache RAG response with metadata"""
        
        # Generate context hash from source documents
        context_hash = None
        if context_documents:
            context_str = ''.join(sorted([doc.get('id', '') for doc in context_documents]))
            context_hash = hashlib.md5(context_str.encode()).hexdigest()
        
        cache_key = self._generate_cache_key(query, context_hash)
        
        # Prepare cached data
        cached_data = {
            "response": response,
            "query": query,
            "context_hash": context_hash,
            "source_count": len(context_documents) if context_documents else 0,
            "cached_at": datetime.now().isoformat(),
            "user_id": user_id,
            "hit_count": 1
        }
        
        # Store in Redis with TTL
        self.redis.setex(
            cache_key, 
            ttl or self.default_ttl,
            json.dumps(cached_data)
        )
        
        # Track cache statistics
        self._update_cache_stats("cache_write")
        
        print(f"Cached response for query: {query[:50]}...")
        return cache_key
    
    def get_cached_response(self, query, context_documents=None):
        """Retrieve cached response if available"""
        
        # Generate same cache key
        context_hash = None
        if context_documents:
            context_str = ''.join(sorted([doc.get('id', '') for doc in context_documents]))
            context_hash = hashlib.md5(context_str.encode()).hexdigest()
        
        cache_key = self._generate_cache_key(query, context_hash)
        
        # Try to get from cache
        cached_json = self.redis.get(cache_key)
        
        if cached_json:
            cached_data = json.loads(cached_json)
            
            # Update hit count
            cached_data["hit_count"] += 1
            cached_data["last_accessed"] = datetime.now().isoformat()
            
            # Update cache with new hit count
            ttl = self.redis.ttl(cache_key)
            if ttl > 0:
                self.redis.setex(cache_key, ttl, json.dumps(cached_data))
            
            # Track cache hit
            self._update_cache_stats("cache_hit")
            
            print(f"Cache HIT for query: {query[:50]}...")
            return cached_data["response"], cached_data
        
        # Track cache miss
        self._update_cache_stats("cache_miss")
        print(f"Cache MISS for query: {query[:50]}...")
        return None, None
    
    def _update_cache_stats(self, stat_type):
        """Update cache statistics"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Daily stats
        self.redis.hincrby(f"cache:stats:{today}", stat_type, 1)
        
        # Overall stats
        self.redis.hincrby("cache:stats:total", stat_type, 1)
        
        # Set expiration for daily stats (keep for 30 days)
        self.redis.expire(f"cache:stats:{today}", 30 * 24 * 3600)
    
    def get_cache_stats(self, days=7):
        """Get cache performance statistics"""
        
        stats = {"daily": {}, "total": {}}
        
        # Get daily stats
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            daily_stats = self.redis.hgetall(f"cache:stats:{date}")
            if daily_stats:
                stats["daily"][date] = {
                    "hits": int(daily_stats.get("cache_hit", 0)),
                    "misses": int(daily_stats.get("cache_miss", 0)),
                    "writes": int(daily_stats.get("cache_write", 0))
                }
                
                # Calculate hit rate
                total_requests = stats["daily"][date]["hits"] + stats["daily"][date]["misses"]
                if total_requests > 0:
                    stats["daily"][date]["hit_rate"] = stats["daily"][date]["hits"] / total_requests
        
        # Get total stats
        total_stats = self.redis.hgetall("cache:stats:total")
        if total_stats:
            stats["total"] = {
                "hits": int(total_stats.get("cache_hit", 0)),
                "misses": int(total_stats.get("cache_miss", 0)),
                "writes": int(total_stats.get("cache_write", 0))
            }
            
            total_requests = stats["total"]["hits"] + stats["total"]["misses"]
            if total_requests > 0:
                stats["total"]["hit_rate"] = stats["total"]["hits"] / total_requests
        
        return stats

# Initialize cache manager
cache_manager = RAGCacheManager()

# Example usage
query = "What is machine learning?"
response = "Machine learning is a method of data analysis..."

# Cache the response
cache_key = cache_manager.cache_response(
    query=query,
    response=response,
    context_documents=[{"id": "doc1"}, {"id": "doc2"}],
    user_id="user123"
)

# Try to retrieve from cache
cached_response, metadata = cache_manager.get_cached_response(
    query=query,
    context_documents=[{"id": "doc1"}, {"id": "doc2"}]
)

if cached_response:
    print(f"Retrieved from cache: {cached_response}")
    print(f"Cache metadata: {metadata}")
```

### 2. Embedding Cache
```python
def cache_embedding(self, text, model_name, embedding_vector, ttl=86400):
    """Cache embedding vectors to avoid recomputation"""
    
    # Generate cache key
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cache_key = f"embedding:{model_name}:{text_hash}"
    
    # Prepare embedding data
    embedding_data = {
        "text": text,
        "model": model_name,
        "embedding": embedding_vector,
        "dimensions": len(embedding_vector),
        "cached_at": datetime.now().isoformat(),
        "access_count": 1
    }
    
    # Cache embedding with 1 day TTL
    self.redis.setex(
        cache_key,
        ttl,
        json.dumps(embedding_data)
    )
    
    return cache_key

def get_cached_embedding(self, text, model_name):
    """Retrieve cached embedding if available"""
    
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cache_key = f"embedding:{model_name}:{text_hash}"
    
    cached_json = self.redis.get(cache_key)
    
    if cached_json:
        embedding_data = json.loads(cached_json)
        
        # Update access count
        embedding_data["access_count"] += 1
        embedding_data["last_accessed"] = datetime.now().isoformat()
        
        # Update cache
        ttl = self.redis.ttl(cache_key)
        if ttl > 0:
            self.redis.setex(cache_key, ttl, json.dumps(embedding_data))
        
        return embedding_data["embedding"], embedding_data
    
    return None, None

# Example embedding cache usage
text = "What is artificial intelligence?"
model = "nomic-embed-text"

# First check cache
cached_embedding, metadata = cache_manager.get_cached_embedding(text, model)

if cached_embedding:
    print(f"Using cached embedding: {len(cached_embedding)} dimensions")
else:
    # Generate new embedding (simulate with random data)
    import random
    new_embedding = [random.random() for _ in range(768)]
    
    # Cache the new embedding
    cache_manager.cache_embedding(text, model, new_embedding)
    print("Generated and cached new embedding")
```

### 3. Session Management
```python
def create_session(self, user_id, session_data=None, ttl=3600):
    """Create user session with conversation context"""
    
    import uuid
    session_id = str(uuid.uuid4())
    
    session_info = {
        "session_id": session_id,
        "user_id": user_id,
        "created_at": datetime.now().isoformat(),
        "last_activity": datetime.now().isoformat(),
        "conversation_history": [],
        "context_documents": [],
        "preferences": session_data or {}
    }
    
    # Store session
    session_key = f"session:{session_id}"
    self.redis.setex(session_key, ttl, json.dumps(session_info))
    
    # Add to user's active sessions
    user_sessions_key = f"user:{user_id}:sessions"
    self.redis.sadd(user_sessions_key, session_id)
    self.redis.expire(user_sessions_key, ttl)
    
    return session_id

def add_to_conversation(self, session_id, query, response, context_docs=None):
    """Add query-response pair to conversation history"""
    
    session_key = f"session:{session_id}"
    session_json = self.redis.get(session_key)
    
    if session_json:
        session_data = json.loads(session_json)
        
        # Add conversation turn
        conversation_turn = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "context_docs": context_docs or [],
            "turn_number": len(session_data["conversation_history"]) + 1
        }
        
        session_data["conversation_history"].append(conversation_turn)
        session_data["last_activity"] = datetime.now().isoformat()
        
        # Keep only last 20 turns to manage memory
        if len(session_data["conversation_history"]) > 20:
            session_data["conversation_history"] = session_data["conversation_history"][-20:]
        
        # Update session
        ttl = self.redis.ttl(session_key)
        if ttl > 0:
            self.redis.setex(session_key, ttl, json.dumps(session_data))
        
        return True
    
    return False

def get_conversation_context(self, session_id, last_n_turns=5):
    """Get recent conversation context for RAG"""
    
    session_key = f"session:{session_id}"
    session_json = self.redis.get(session_key)
    
    if session_json:
        session_data = json.loads(session_json)
        history = session_data["conversation_history"]
        
        # Get last N conversation turns
        recent_turns = history[-last_n_turns:] if history else []
        
        # Format as context
        context = {
            "session_id": session_id,
            "user_id": session_data["user_id"],
            "turns": recent_turns,
            "total_turns": len(history)
        }
        
        return context
    
    return None

# Session usage example
session_id = cache_manager.create_session(
    user_id="user123",
    session_data={"preferred_language": "en", "domain": "ai"}
)

# Add conversation turns
cache_manager.add_to_conversation(
    session_id=session_id,
    query="What is machine learning?",
    response="Machine learning is a subset of AI...",
    context_docs=["doc1", "doc2"]
)

cache_manager.add_to_conversation(
    session_id=session_id,
    query="Can you explain neural networks?",
    response="Neural networks are computing systems...",
    context_docs=["doc3", "doc4"]
)

# Get conversation context
context = cache_manager.get_conversation_context(session_id)
print(f"Conversation context: {len(context['turns'])} turns")
```

## RedisInsight Dashboard Usage

### 1. Database Overview
```bash
# RedisInsight Features at http://your-host:8001

1. Database Browser
   - View all keys by pattern
   - Filter by data type (String, Hash, List, Set, Sorted Set)
   - Search keys with wildcards
   - View key details and values

2. Memory Analysis
   - Memory usage by key patterns
   - Top memory consuming keys
   - Memory distribution by data type
   - Memory optimization suggestions

3. Performance Monitor
   - Real-time command statistics
   - Slowlog analysis
   - Client connections
   - Memory usage trends
```

### 2. Key Management via UI
```bash
# Using RedisInsight Browser:

1. Browse Keys:
   - Pattern search: "rag:*", "session:*", "embedding:*"
   - Filter by type: Hash, String, List
   - Sort by memory usage or TTL

2. View/Edit Data:
   - Click on key to view details
   - Edit values directly in UI
   - Set/modify TTL values
   - Delete keys individually or in bulk

3. Monitor Performance:
   - View command statistics
   - Track slow operations
   - Monitor memory usage
   - Analyze key distribution
```

### 3. CLI Operations
```bash
# Useful Redis CLI commands for RAG operations

# Connect to Redis
docker exec -it rag-redis redis-cli

# View all RAG cache keys
KEYS rag:*

# Get cache statistics
HGETALL cache:stats:total

# Monitor real-time commands
MONITOR

# Get memory usage
MEMORY USAGE cache_key

# View slow queries
SLOWLOG GET 10

# Check key expiration
TTL session:abc123

# Clear cache by pattern
EVAL "return redis.call('DEL', unpack(redis.call('KEYS', ARGV[1])))" 0 "rag:response:*"
```

## Advanced Features

### 1. Rate Limiting
```python
def rate_limit_user(self, user_id, action, max_requests=100, window_seconds=3600):
    """Implement rate limiting for users"""
    
    window_start = int(datetime.now().timestamp()) // window_seconds
    rate_limit_key = f"rate_limit:{action}:{user_id}:{window_start}"
    
    # Get current count
    current_count = self.redis.get(rate_limit_key)
    
    if current_count is None:
        # First request in this window
        self.redis.setex(rate_limit_key, window_seconds, 1)
        return True, max_requests - 1
    
    current_count = int(current_count)
    
    if current_count >= max_requests:
        # Rate limit exceeded
        return False, 0
    
    # Increment counter
    self.redis.incr(rate_limit_key)
    return True, max_requests - current_count - 1

# Usage example
allowed, remaining = cache_manager.rate_limit_user(
    user_id="user123",
    action="rag_query",
    max_requests=50,
    window_seconds=3600  # 1 hour window
)

if not allowed:
    print("Rate limit exceeded")
else:
    print(f"Request allowed, {remaining} requests remaining")
```

### 2. Real-time Analytics
```python
def track_query_analytics(self, query, user_id=None, response_time=None):
    """Track real-time query analytics"""
    
    timestamp = datetime.now()
    hour_key = timestamp.strftime("%Y-%m-%d-%H")
    
    # Track queries per hour
    self.redis.hincrby(f"analytics:hourly:{hour_key}", "query_count", 1)
    
    # Track response times
    if response_time:
        self.redis.hincrby(f"analytics:hourly:{hour_key}", "total_response_time", int(response_time * 1000))
    
    # Track unique users
    if user_id:
        self.redis.sadd(f"analytics:users:{hour_key}", user_id)
    
    # Popular query terms (simplified)
    words = query.lower().split()[:5]  # First 5 words
    for word in words:
        if len(word) > 3:  # Skip short words
            self.redis.zincrby("analytics:popular_terms", 1, word)
    
    # Set expiration for analytics data
    self.redis.expire(f"analytics:hourly:{hour_key}", 7 * 24 * 3600)  # 7 days
    self.redis.expire(f"analytics:users:{hour_key}", 7 * 24 * 3600)

def get_realtime_analytics(self):
    """Get real-time analytics dashboard data"""
    
    current_hour = datetime.now().strftime("%Y-%m-%d-%H")
    
    # Current hour stats
    hour_stats = self.redis.hgetall(f"analytics:hourly:{current_hour}")
    
    # Active users this hour
    active_users = self.redis.scard(f"analytics:users:{current_hour}")
    
    # Popular terms (top 10)
    popular_terms = self.redis.zrevrange("analytics:popular_terms", 0, 9, withscores=True)
    
    # Cache hit rate
    cache_stats = self.get_cache_stats(days=1)
    
    analytics = {
        "current_hour": {
            "queries": int(hour_stats.get("query_count", 0)),
            "active_users": active_users,
            "avg_response_time": 0
        },
        "popular_terms": [{"term": term.decode() if isinstance(term, bytes) else term, 
                          "count": int(score)} for term, score in popular_terms],
        "cache_performance": cache_stats.get("total", {})
    }
    
    # Calculate average response time
    if hour_stats.get("query_count", 0) and hour_stats.get("total_response_time", 0):
        analytics["current_hour"]["avg_response_time"] = (
            int(hour_stats["total_response_time"]) / int(hour_stats["query_count"])
        )
    
    return analytics
```

### 3. Background Job Queue
```python
def queue_document_processing(self, document_id, document_path, priority=0):
    """Queue document for background processing"""
    
    job_data = {
        "job_id": str(uuid.uuid4()),
        "document_id": document_id,
        "document_path": document_path,
        "queued_at": datetime.now().isoformat(),
        "status": "queued",
        "priority": priority
    }
    
    # Add to priority queue (lower score = higher priority)
    self.redis.zadd("job_queue:document_processing", {json.dumps(job_data): priority})
    
    # Track job status
    self.redis.setex(f"job:{job_data['job_id']}", 3600, json.dumps(job_data))
    
    return job_data["job_id"]

def get_next_job(self, queue_name="document_processing"):
    """Get next job from priority queue"""
    
    # Get highest priority job (lowest score)
    jobs = self.redis.zrange(f"job_queue:{queue_name}", 0, 0, withscores=True)
    
    if jobs:
        job_json, priority = jobs[0]
        job_data = json.loads(job_json)
        
        # Remove from queue
        self.redis.zrem(f"job_queue:{queue_name}", job_json)
        
        # Update status
        job_data["status"] = "processing"
        job_data["started_at"] = datetime.now().isoformat()
        self.redis.setex(f"job:{job_data['job_id']}", 3600, json.dumps(job_data))
        
        return job_data
    
    return None

def complete_job(self, job_id, result=None, error=None):
    """Mark job as completed"""
    
    job_json = self.redis.get(f"job:{job_id}")
    
    if job_json:
        job_data = json.loads(job_json)
        job_data["status"] = "completed" if not error else "failed"
        job_data["completed_at"] = datetime.now().isoformat()
        
        if result:
            job_data["result"] = result
        if error:
            job_data["error"] = str(error)
        
        # Store final result for 24 hours
        self.redis.setex(f"job:{job_id}", 24 * 3600, json.dumps(job_data))
        
        return True
    
    return False
```

## Performance Optimization

### 1. Memory Management
```python
def optimize_memory_usage(self):
    """Optimize Redis memory usage"""
    
    # Get memory info
    memory_info = self.redis.info('memory')
    used_memory = memory_info['used_memory']
    max_memory = memory_info.get('maxmemory', 0)
    
    print(f"Memory usage: {memory_info['used_memory_human']} / {memory_info.get('maxmemory_human', 'unlimited')}")
    
    # Find large keys
    large_keys = []
    for key in self.redis.scan_iter(count=1000):
        memory_usage = self.redis.memory_usage(key)
        if memory_usage and memory_usage > 1024 * 1024:  # > 1MB
            large_keys.append((key, memory_usage))
    
    # Sort by size
    large_keys.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Found {len(large_keys)} keys > 1MB")
    for key, size in large_keys[:10]:
        print(f"  {key}: {size / 1024 / 1024:.2f} MB")
    
    return large_keys

def cleanup_expired_data(self):
    """Clean up expired or old data"""
    
    # Remove old analytics data (older than 30 days)
    cutoff_date = datetime.now() - timedelta(days=30)
    
    for key in self.redis.scan_iter(match="analytics:*"):
        # Extract date from key if possible
        try:
            if "hourly" in key:
                date_part = key.split(":")[-1]  # Get YYYY-MM-DD-HH
                key_date = datetime.strptime(date_part[:10], "%Y-%m-%d")
                
                if key_date < cutoff_date:
                    self.redis.delete(key)
                    print(f"Deleted old analytics key: {key}")
        except:
            continue
    
    # Trim popular terms to top 1000
    self.redis.zremrangebyrank("analytics:popular_terms", 0, -1001)
```

### 2. Connection Management
```python
def setup_connection_pool(self, max_connections=20):
    """Setup Redis connection pool for better performance"""
    
    import redis
    
    pool = redis.ConnectionPool(
        host='your-host',
        port=6379,
        decode_responses=True,
        max_connections=max_connections,
        socket_keepalive=True,
        socket_keepalive_options={},
        health_check_interval=30
    )
    
    self.redis = redis.Redis(connection_pool=pool)
    return pool
```

## Monitoring and Alerts

### 1. Health Monitoring
```python
def check_redis_health(self):
    """Comprehensive Redis health check"""
    
    health_status = {
        "status": "healthy",
        "issues": [],
        "metrics": {}
    }
    
    try:
        # Basic connectivity
        self.redis.ping()
        
        # Memory usage
        memory_info = self.redis.info('memory')
        memory_usage_percent = (memory_info['used_memory'] / memory_info.get('maxmemory', memory_info['used_memory'] * 2)) * 100
        
        health_status["metrics"]["memory_usage_percent"] = memory_usage_percent
        
        if memory_usage_percent > 90:
            health_status["status"] = "warning"
            health_status["issues"].append("High memory usage")
        
        # Connection count
        clients_info = self.redis.info('clients')
        connected_clients = clients_info['connected_clients']
        
        health_status["metrics"]["connected_clients"] = connected_clients
        
        if connected_clients > 100:
            health_status["issues"].append("High client connection count")
        
        # Keyspace info
        keyspace_info = self.redis.info('keyspace')
        total_keys = sum([int(db['keys']) for db in keyspace_info.values() if isinstance(db, dict)])
        
        health_status["metrics"]["total_keys"] = total_keys
        
        # Response time test
        import time
        start = time.time()
        self.redis.get("health_check_key")
        response_time = (time.time() - start) * 1000
        
        health_status["metrics"]["response_time_ms"] = response_time
        
        if response_time > 100:
            health_status["issues"].append("Slow response time")
        
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["issues"].append(f"Connection error: {str(e)}")
    
    return health_status
```

## Troubleshooting

### Common Issues

**"Connection refused"**
```python
# Check Redis service status
import redis

try:
    r = redis.Redis(host='your-host', port=6379, socket_timeout=5)
    r.ping()
    print("Redis is accessible")
except redis.ConnectionError as e:
    print(f"Cannot connect to Redis: {e}")
    # Check: docker compose ps redis
    # Check: docker compose logs redis
```

**"Memory usage too high"**
```python
# Check memory usage
info = r.info('memory')
print(f"Used memory: {info['used_memory_human']}")
print(f"Peak memory: {info['used_memory_peak_human']}")

# Find memory-heavy keys
for key in r.scan_iter(count=100):
    size = r.memory_usage(key)
    if size > 1024 * 1024:  # > 1MB
        print(f"Large key: {key} - {size/1024/1024:.2f}MB")
```

**"RedisInsight not accessible"**
```bash
# Check RedisInsight service
docker compose ps redisinsight

# Check logs
docker compose logs redisinsight

# Verify port mapping
docker port rag-redisinsight 8001
```

**"Slow queries"**
```python
# Check slow log
slowlog = r.slowlog_get(10)
for entry in slowlog:
    print(f"Slow query: {entry['command']} - {entry['duration']}μs")

# Monitor commands in real-time
# redis-cli MONITOR
```

## Learning Resources

- [Redis Documentation](https://redis.io/docs/) - Official documentation
- [RedisInsight Guide](https://redis.com/redis-enterprise/redis-insight/) - UI management tool
- [Redis Python Client](https://redis-py.readthedocs.io/) - Python library documentation
- [Redis Performance](https://redis.io/docs/management/optimization/) - Performance optimization guide