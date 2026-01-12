# Langfuse Observability Guide

Langfuse provides comprehensive observability for LLM applications. Essential for monitoring RAG performance, debugging issues, and improving system quality.

## Quick Access

| Component | URL | Purpose |
|-----------|-----|---------|
| **Langfuse UI** | `http://your-host:3000` | Web interface for traces and analytics |
| **API Endpoint** | `http://your-host:3000/api` | REST API for integrations |

## Why Langfuse for RAG?

### Observability Benefits
- **Trace Complete RAG Pipeline**: From query to response
- **Performance Monitoring**: Latency, cost, quality metrics
- **Error Debugging**: Detailed error traces and context
- **Quality Assessment**: Human feedback and automated scoring
- **A/B Testing**: Compare different RAG configurations

### RAG-Specific Features
- **Multi-step Tracing**: Track retrieval → generation → response
- **Token Usage**: Monitor embedding and generation costs
- **Quality Scores**: Rate retrieval relevance and answer quality
- **User Feedback**: Collect thumbs up/down and detailed feedback

## Initial Setup

### 1. First Login
```bash
# Access Langfuse UI
open http://your-host:3000

# Create your first account (no existing users)
# Email: your-email@domain.com
# Password: your-secure-password
```

### 2. Create Project and Get API Keys
1. **Create Project**: Click "New Project" → Enter project name
2. **Get API Keys**: Project Settings → API Keys
   - Copy `Public Key` (pk-lf-...)
   - Copy `Secret Key` (sk-lf-...)
   - Copy `Host URL` (http://your-host:3000)

### 3. Test Connection
```python
from langfuse import Langfuse

langfuse = Langfuse(
    public_key="pk-lf-your-public-key",
    secret_key="sk-lf-your-secret-key", 
    host="http://your-host:3000"
)

# Test connection
health = langfuse.health_check()
print(f"Langfuse connection: {health}")
```

## RAG Tracing Implementation

### 1. Basic RAG Tracing

```python
from langfuse import Langfuse
import time

class TracedRAG:
    def __init__(self, langfuse_config):
        self.langfuse = Langfuse(**langfuse_config)
    
    def traced_rag_query(self, question, user_id=None):
        """Complete RAG pipeline with full tracing"""
        
        # Start main trace
        trace = self.langfuse.trace(
            name="rag_query",
            input=question,
            user_id=user_id,
            metadata={"source": "api", "version": "1.0"}
        )
        
        try:
            # Step 1: Query processing
            processing_span = trace.span(
                name="query_processing",
                input=question
            )
            processed_query = self.process_query(question)
            processing_span.end(output=processed_query)
            
            # Step 2: Retrieval
            retrieval_span = trace.span(
                name="document_retrieval", 
                input=processed_query
            )
            
            start_time = time.time()
            retrieved_docs = self.retrieve_documents(processed_query)
            retrieval_time = time.time() - start_time
            
            retrieval_span.end(
                output={
                    "num_documents": len(retrieved_docs),
                    "documents": retrieved_docs[:3]  # First 3 for brevity
                },
                metadata={"retrieval_time_ms": retrieval_time * 1000}
            )
            
            # Step 3: Generation
            generation_span = trace.span(
                name="response_generation",
                input={
                    "query": processed_query,
                    "context": retrieved_docs
                }
            )
            
            start_time = time.time()
            response = self.generate_response(processed_query, retrieved_docs)
            generation_time = time.time() - start_time
            
            generation_span.end(
                output=response,
                metadata={"generation_time_ms": generation_time * 1000}
            )
            
            # Complete trace
            trace.update(
                output=response,
                metadata={
                    "total_documents": len(retrieved_docs),
                    "response_length": len(response)
                }
            )
            
            return {
                "response": response,
                "sources": retrieved_docs,
                "trace_id": trace.id
            }
            
        except Exception as e:
            trace.update(
                output=f"Error: {str(e)}",
                level="ERROR"
            )
            raise

# Usage
rag = TracedRAG({
    "public_key": "pk-lf-...",
    "secret_key": "sk-lf-...",
    "host": "http://your-host:3000"
})

result = rag.traced_rag_query("What is machine learning?", user_id="user123")
```

### 2. Advanced Tracing with Scores

```python
def traced_rag_with_scoring(self, question, user_id=None):
    """RAG with automatic quality scoring"""
    
    trace = self.langfuse.trace(
        name="rag_query_scored",
        input=question,
        user_id=user_id
    )
    
    # ... RAG pipeline steps ...
    
    # Add quality scores
    trace.score(
        name="retrieval_relevance",
        value=self.score_retrieval_relevance(question, retrieved_docs),
        comment="Automated relevance scoring"
    )
    
    trace.score(
        name="response_quality", 
        value=self.score_response_quality(question, response, retrieved_docs),
        comment="Automated quality assessment"
    )
    
    trace.score(
        name="response_faithfulness",
        value=self.score_faithfulness(response, retrieved_docs),
        comment="How well response reflects source material"
    )
    
    return result

def score_retrieval_relevance(self, query, documents):
    """Score how relevant retrieved documents are to query"""
    # Implementation: semantic similarity, keyword overlap, etc.
    # Return score between 0.0 and 1.0
    return 0.85

def score_response_quality(self, query, response, context):
    """Score overall response quality"""
    # Implementation: coherence, completeness, accuracy
    return 0.92

def score_faithfulness(self, response, sources):
    """Score how faithful response is to source material"""
    # Implementation: fact checking against sources
    return 0.88
```

### 3. User Feedback Integration

```python
def collect_user_feedback(self, trace_id, feedback_type, value, comment=None):
    """Collect user feedback for traces"""
    
    self.langfuse.score(
        trace_id=trace_id,
        name=feedback_type,
        value=value,
        comment=comment,
        source="user_feedback"
    )

# Usage examples
# Thumbs up/down feedback
rag.collect_user_feedback(trace_id, "user_rating", 1.0, "Helpful response")
rag.collect_user_feedback(trace_id, "user_rating", 0.0, "Wrong information")

# Detailed ratings
rag.collect_user_feedback(trace_id, "helpfulness", 0.8, "Good but could be more detailed")
rag.collect_user_feedback(trace_id, "accuracy", 0.9, "Information seems correct")
```

## Monitoring and Analytics

### 1. Performance Dashboards

Access the Langfuse UI to view:

**Overview Dashboard**
- Total traces and spans
- Average response times
- Error rates
- Token usage

**Traces View**
- Individual request traces
- Performance breakdowns
- Error debugging
- User sessions

**Analytics**
- Quality score trends
- Performance over time
- User feedback analysis
- Cost tracking

### 2. Custom Metrics

```python
def track_custom_metrics(self, trace, rag_metrics):
    """Track custom RAG-specific metrics"""
    
    trace.update(metadata={
        "embedding_model": rag_metrics.get("embedding_model"),
        "generation_model": rag_metrics.get("generation_model"),
        "num_documents_retrieved": rag_metrics.get("num_docs"),
        "average_doc_relevance": rag_metrics.get("avg_relevance"),
        "context_length": rag_metrics.get("context_length"),
        "response_tokens": rag_metrics.get("response_tokens"),
        "total_cost": rag_metrics.get("total_cost")
    })

# Example usage
rag_metrics = {
    "embedding_model": "nomic-embed-text",
    "generation_model": "llama3.2:7b",
    "num_docs": 5,
    "avg_relevance": 0.78,
    "context_length": 1200,
    "response_tokens": 150,
    "total_cost": 0.002
}

trace.track_custom_metrics(trace, rag_metrics)
```

### 3. A/B Testing Framework

```python
class RAGExperiments:
    def __init__(self, langfuse):
        self.langfuse = langfuse
    
    def run_ab_test(self, question, user_id, experiment="default"):
        """Run A/B test between different RAG configurations"""
        
        trace = self.langfuse.trace(
            name="rag_ab_test",
            input=question,
            user_id=user_id,
            tags=[f"experiment:{experiment}"]
        )
        
        if experiment == "enhanced_retrieval":
            # Use different retrieval strategy
            result = self.enhanced_rag_pipeline(question, trace)
        elif experiment == "better_prompts":
            # Use improved prompts
            result = self.improved_prompt_pipeline(question, trace)
        else:
            # Default pipeline
            result = self.default_rag_pipeline(question, trace)
        
        trace.update(
            output=result,
            metadata={"experiment_variant": experiment}
        )
        
        return result

# Compare experiment performance in Langfuse UI
# Filter by tags to analyze each variant
```

## Production Monitoring

### 1. Error Tracking

```python
def trace_with_error_handling(self, question):
    """RAG pipeline with comprehensive error tracking"""
    
    trace = self.langfuse.trace(name="rag_production", input=question)
    
    try:
        # Retrieval with error handling
        retrieval_span = trace.span(name="retrieval")
        try:
            docs = self.retrieve_documents(question)
            retrieval_span.end(output={"num_docs": len(docs)})
        except Exception as e:
            retrieval_span.end(
                output=f"Retrieval error: {str(e)}",
                level="ERROR",
                status_message=str(e)
            )
            raise
        
        # Generation with error handling
        generation_span = trace.span(name="generation")
        try:
            response = self.generate_response(question, docs)
            generation_span.end(output=response)
        except Exception as e:
            generation_span.end(
                output=f"Generation error: {str(e)}",
                level="ERROR",
                status_message=str(e)
            )
            raise
        
        trace.update(output=response, level="DEFAULT")
        return response
        
    except Exception as e:
        trace.update(
            output=f"Pipeline failed: {str(e)}",
            level="ERROR",
            status_message=str(e)
        )
        # Log for external monitoring
        logger.error(f"RAG pipeline failed: {e}", extra={"trace_id": trace.id})
        raise
```

### 2. Performance Alerts

```python
def check_performance_thresholds(self, trace, response_time, quality_score):
    """Monitor performance and trigger alerts"""
    
    # Response time threshold
    if response_time > 5.0:  # 5 seconds
        trace.update(metadata={"performance_alert": "slow_response"})
        self.send_alert(f"Slow response: {response_time:.2f}s")
    
    # Quality threshold
    if quality_score < 0.7:
        trace.update(metadata={"quality_alert": "low_quality"})
        self.send_alert(f"Low quality response: {quality_score:.2f}")
    
    # Add performance scores
    trace.score(name="response_time", value=min(1.0, 5.0 / response_time))
    trace.score(name="quality", value=quality_score)
```

## Cost Tracking

### 1. Token Usage Monitoring

```python
def track_token_usage(self, trace, model_calls):
    """Track token usage across different models"""
    
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    
    for model_call in model_calls:
        input_tokens = model_call.get("input_tokens", 0)
        output_tokens = model_call.get("output_tokens", 0)
        cost = model_call.get("cost", 0.0)
        
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_cost += cost
        
        # Track per-model usage
        trace.span(
            name=f"{model_call['model']}_usage",
            input=f"Input tokens: {input_tokens}",
            output=f"Output tokens: {output_tokens}",
            metadata={
                "model": model_call["model"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost
            }
        )
    
    # Track total usage
    trace.update(metadata={
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_usd": total_cost
    })
```

## Data Export and Analysis

### 1. Export Traces for Analysis

```python
def export_traces_for_analysis(self, start_date, end_date):
    """Export trace data for external analysis"""
    
    # Use Langfuse API to fetch traces
    traces = self.langfuse.get_traces(
        from_timestamp=start_date,
        to_timestamp=end_date,
        limit=1000
    )
    
    # Convert to DataFrame for analysis
    import pandas as pd
    
    trace_data = []
    for trace in traces:
        trace_data.append({
            "id": trace.id,
            "timestamp": trace.timestamp,
            "input": trace.input,
            "output": trace.output,
            "user_id": trace.user_id,
            "session_id": trace.session_id,
            "duration_ms": trace.duration,
            "metadata": trace.metadata
        })
    
    df = pd.DataFrame(trace_data)
    return df

# Analyze performance trends
df = langfuse_client.export_traces_for_analysis("2024-01-01", "2024-01-31")
print(f"Average response time: {df['duration_ms'].mean():.2f}ms")
print(f"Total queries: {len(df)}")
```

## Integration Patterns

### 1. FastAPI Integration

```python
from fastapi import FastAPI
from langfuse.decorators import langfuse_context, observe

app = FastAPI()

@app.post("/query")
@observe()
def rag_endpoint(question: str, user_id: str = None):
    """RAG endpoint with automatic tracing"""
    
    # Langfuse automatically creates trace
    langfuse_context.update_current_trace(
        user_id=user_id,
        metadata={"endpoint": "/query"}
    )
    
    # Your RAG logic here
    result = process_rag_query(question)
    
    return result

@observe()
def process_rag_query(question: str):
    """RAG processing with automatic span creation"""
    
    # Each function call becomes a span
    docs = retrieve_documents(question)
    response = generate_response(question, docs)
    
    return {"response": response, "sources": docs}
```

### 2. LangChain Integration

```python
from langfuse.callback import CallbackHandler

# Initialize Langfuse handler
langfuse_handler = CallbackHandler(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="http://your-host:3000"
)

# Use with LangChain
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    callbacks=[langfuse_handler]
)

# Automatic tracing for LangChain operations
result = qa_chain.run("What is machine learning?")
```

## Best Practices

### Tracing Strategy
1. **Trace at the right level**: Not too granular, not too high-level
2. **Include relevant context**: User ID, session, experiment variant
3. **Use meaningful names**: Clear span and trace names
4. **Add metadata**: Model versions, parameters, timestamps

### Performance
1. **Async logging**: Don't block main thread for tracing
2. **Sampling**: Trace subset of requests in high-volume scenarios
3. **Batch uploads**: Optimize network calls
4. **Local buffering**: Handle network interruptions gracefully

### Privacy and Security
1. **Sanitize sensitive data**: Remove PII from traces
2. **Access controls**: Restrict Langfuse UI access
3. **Data retention**: Set appropriate retention policies
4. **Audit trails**: Track who accesses what data

## Troubleshooting

### Common Issues

**"Connection refused"**
```python
# Check service status
import requests
try:
    response = requests.get("http://your-host:3000/api/public/health")
    print(f"Langfuse status: {response.status_code}")
except Exception as e:
    print(f"Langfuse unreachable: {e}")
```

**"Invalid API key"**
```python
# Verify API keys
langfuse = Langfuse(public_key="pk-...", secret_key="sk-...", host="http://your-host:3000")
try:
    health = langfuse.health_check()
    print("API keys valid")
except Exception as e:
    print(f"API key issue: {e}")
```

**Missing traces**
- Check trace.end() calls
- Verify network connectivity
- Check Langfuse logs: `docker compose logs langfuse`

## Learning Resources

- [Langfuse Documentation](https://langfuse.com/docs) - Complete documentation
- [Langfuse Cookbook](https://langfuse.com/docs/cookbook) - Integration examples
- [Observability Best Practices](https://langfuse.com/guides/videos/observability-llm-systems) - Video tutorials
- [LLM Evaluation Guide](https://langfuse.com/docs/scores/model-based-evals) - Quality assessment