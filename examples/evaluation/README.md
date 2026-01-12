# RAG Evaluation Example

This example demonstrates how to evaluate RAG system performance using automated metrics and Langfuse observability.

## What Does This Evaluate?

- **Retrieval Quality**: How relevant are the retrieved documents to the query?
- **Answer Quality**: How well does the generated answer match expected answers?
- **Latency**: How fast is the end-to-end RAG pipeline?
- **Traceability**: Full observability through Langfuse tracing

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Test Cases │────▶│  RAG System │────▶│   Metrics   │
│  (Q&A pairs)│     │  (MongoDB + │     │  Scoring    │
└─────────────┘     │   Ollama)   │     └──────┬──────┘
                    └─────────────┘            │
                                               ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Langfuse   │◀────│   Results   │
                    │  (Traces)   │     │   Report    │
                    └─────────────┘     └─────────────┘
```

## Prerequisites

1. RAG Infrastructure Stack deployed
2. Python 3.8+
3. Required packages: `pymongo`, `requests`
4. Optional: `langfuse` for tracing

## Setup

```bash
pip install pymongo requests langfuse
```

## Usage

### Basic Usage

```bash
python eval_rag.py
```

### With Langfuse Tracing

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="http://localhost:3000"

python eval_rag.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGO_URL` | `mongodb://localhost:27017/` | MongoDB connection |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `LANGFUSE_HOST` | `http://localhost:3000` | Langfuse server URL |
| `LANGFUSE_PUBLIC_KEY` | None | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | None | Langfuse secret key |

## Example Output

```
🚀 RAG Evaluation Starting...
============================================================
📊 Connected to MongoDB: rag_eval
🤖 Connected to Ollama: http://localhost:11434
✅ Langfuse connected: http://localhost:3000

📚 Loading sample documents...
✅ Added 5 test documents

🧪 Running evaluation on 5 test cases...
============================================================

[1/5] What is Python and who created it?...
   Relevance: 0.85
   Quality:   0.90
   Latency:   1523ms

[2/5] What is machine learning?...
   Relevance: 0.80
   Quality:   0.85
   Latency:   1245ms
...

============================================================
📊 EVALUATION REPORT
============================================================

📈 Aggregate Metrics:
   Total Queries:      5
   Avg Relevance:      82.00%
   Avg Answer Quality: 78.00%
   Avg Latency:        1432ms

📋 Per-Query Results:
------------------------------------------------------------
✅ What is Python and who created it?...
   Relevance: 85.00% | Quality: 90.00% | Latency: 1523ms
✅ What is machine learning?...
   Relevance: 80.00% | Quality: 85.00% | Latency: 1245ms
...

------------------------------------------------------------

🎯 Overall Grade: B - Good (80.00%)

🔗 View detailed traces in Langfuse dashboard

💾 Results saved to eval_results.json
```

## Evaluation Metrics

### Relevance Score (0-1)

Measures how well retrieved documents relate to the query:
- Uses LLM-as-judge to rate relevance on 0-10 scale
- Normalized to 0-1 range
- Considers semantic similarity, not just keyword matching

### Answer Quality Score (0-1)

Compares generated answers to expected answers:
- Accuracy: Is the information correct?
- Completeness: Does it cover key points?
- Relevance: Does it address the question?

### Latency (ms)

End-to-end pipeline timing:
- Embedding generation
- Vector search
- LLM response generation

## How It Works

### 1. Test Case Definition

```python
TEST_CASES = [
    {
        "query": "What is Python?",
        "expected_answer": "Python is a high-level programming language..."
    },
    # More test cases...
]
```

### 2. RAG Pipeline Execution

For each test case:
1. Generate query embedding
2. Perform vector search
3. Generate answer with context
4. Record latency

### 3. Automated Scoring

```python
# Relevance scoring
prompt = f"""Rate how relevant these documents are to the query (0-10):
Query: {query}
Documents: {docs}
"""

# Quality scoring
prompt = f"""Rate how well the actual answer matches expected (0-10):
Expected: {expected}
Actual: {actual}
"""
```

### 4. Results Aggregation

- Per-query metrics
- Aggregate statistics
- Overall grade (A-F)

## Customization

### Adding Custom Test Cases

```python
CUSTOM_TEST_CASES = [
    {
        "query": "Your custom question?",
        "expected_answer": "The expected answer..."
    }
]

evaluator.run_evaluation(CUSTOM_TEST_CASES)
```

### Custom Scoring Functions

```python
def custom_relevance_scorer(query: str, documents: List[str]) -> float:
    # Your custom relevance logic
    # Return float between 0 and 1
    pass

# Override in RAGEvaluator
evaluator._score_relevance = custom_relevance_scorer
```

### Adjusting Grading Thresholds

```python
# In print_report function
if overall >= 0.9:
    grade = "A+ - Excellent"
elif overall >= 0.8:
    grade = "A - Great"
# ... customize thresholds
```

## Langfuse Integration

When Langfuse credentials are provided, each evaluation creates:

### Traces
- One trace per query evaluation
- Spans for retrieval, generation, and scoring

### Scores
- `relevance`: Document relevance score
- `answer_quality`: Answer accuracy score
- `latency_ms`: Pipeline latency

### Dashboard Views
- View traces in Langfuse UI
- Compare evaluations over time
- Identify problem queries

## Best Practices

### Creating Good Test Cases

1. **Diverse queries**: Cover different question types
2. **Clear expected answers**: Specific, verifiable answers
3. **Representative**: Match real user queries
4. **Edge cases**: Include challenging questions

### Interpreting Results

| Grade | Meaning | Action |
|-------|---------|--------|
| A (80%+) | Excellent | Ready for production |
| B (70-79%) | Good | Minor improvements needed |
| C (60-69%) | Acceptable | Review problem areas |
| D (50-59%) | Needs work | Significant improvements needed |
| F (<50%) | Poor | Major overhaul required |

### Improving Scores

**Low Relevance?**
- Add more documents to knowledge base
- Improve document chunking
- Use better embedding model

**Low Answer Quality?**
- Improve prompt engineering
- Use larger LLM model
- Add more context to prompts

**High Latency?**
- Reduce number of retrieved documents
- Use faster models
- Add caching layer

## Troubleshooting

**"No test documents found"**
- Run `evaluator.setup_test_data()` first
- Check MongoDB connection

**"Langfuse connection failed"**
- Verify Langfuse service is running
- Check API keys are correct
- Confirm host URL is accessible

**"Score parsing failed"**
- LLM may return non-numeric responses
- Default score of 0.5 is used
- Check Ollama is running correctly

**"Empty search results"**
- Ensure vector index exists
- Verify documents have embeddings
- Check MongoDB aggregation pipeline

## Output Files

### eval_results.json

```json
{
  "total_queries": 5,
  "avg_relevance": 0.82,
  "avg_answer_quality": 0.78,
  "avg_latency_ms": 1432.5,
  "results": [
    {
      "query": "What is Python?",
      "relevance": 0.85,
      "quality": 0.90,
      "latency_ms": 1523,
      "trace_id": "abc123..."
    }
  ]
}
```

## Learn More

- [Langfuse Documentation](../../docs/services/langfuse.md)
- [MongoDB Vector Search](../../docs/services/mongodb.md)
- [RAG Architecture](../../docs/concepts/rag-architecture.md)
