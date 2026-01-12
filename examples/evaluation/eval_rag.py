#!/usr/bin/env python3
"""
RAG Evaluation Example - Measure RAG Performance with Langfuse

This example demonstrates how to evaluate RAG systems using:
- Langfuse for tracing and scoring
- Custom evaluation metrics
- Automated quality assessment

Usage:
    python eval_rag.py
"""

import os
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from pymongo import MongoClient
import requests
import json


@dataclass
class EvalResult:
    """Evaluation result for a single query."""
    query: str
    expected_answer: str
    actual_answer: str
    retrieved_docs: List[str]
    relevance_score: float
    answer_quality: float
    latency_ms: float
    trace_id: Optional[str] = None


class RAGEvaluator:
    """Evaluate RAG system performance with multiple metrics."""

    def __init__(self,
                 mongo_url: str = "mongodb://localhost:27017/",
                 ollama_url: str = "http://localhost:11434",
                 langfuse_host: str = "http://localhost:3000",
                 langfuse_public_key: str = None,
                 langfuse_secret_key: str = None,
                 database: str = "rag_eval"):

        self.mongo_client = MongoClient(mongo_url)
        self.db = self.mongo_client[database]
        self.documents = self.db.documents
        self.ollama_url = ollama_url
        self.langfuse_host = langfuse_host

        # Langfuse setup (optional)
        self.langfuse = None
        if langfuse_public_key and langfuse_secret_key:
            try:
                from langfuse import Langfuse
                self.langfuse = Langfuse(
                    host=langfuse_host,
                    public_key=langfuse_public_key,
                    secret_key=langfuse_secret_key
                )
                print(f"✅ Langfuse connected: {langfuse_host}")
            except ImportError:
                print("⚠️ Langfuse not installed. Run: pip install langfuse")
            except Exception as e:
                print(f"⚠️ Langfuse connection failed: {e}")

        print(f"📊 Connected to MongoDB: {database}")
        print(f"🤖 Connected to Ollama: {ollama_url}")

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama."""
        response = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text}
        )
        return response.json()["embedding"]

    def _vector_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Perform vector similarity search."""
        query_embedding = self._get_embedding(query)

        # Try $vectorSearch first, fall back to basic search
        try:
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
                {"$project": {"content": 1, "metadata": 1, "score": {"$meta": "vectorSearchScore"}}}
            ]
            return list(self.documents.aggregate(pipeline))
        except:
            # Fallback: return all docs (for testing without vector index)
            return list(self.documents.find({}, {"content": 1, "metadata": 1}).limit(limit))

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Ollama."""
        prompt = f"""Based on the following context, answer the question concisely.

Context:
{context}

Question: {query}

Answer:"""

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}
        )
        return response.json()["response"]

    def _score_relevance(self, query: str, documents: List[str]) -> float:
        """Score document relevance using LLM."""
        if not documents:
            return 0.0

        docs_text = "\n---\n".join(documents[:3])
        prompt = f"""Rate how relevant these documents are to the query on a scale of 0-10.

Query: {query}

Documents:
{docs_text}

Return only a number between 0 and 10:"""

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}
        )

        try:
            score = float(response.json()["response"].strip().split()[0])
            return min(max(score / 10.0, 0.0), 1.0)  # Normalize to 0-1
        except:
            return 0.5  # Default if parsing fails

    def _score_answer_quality(self, query: str, expected: str, actual: str) -> float:
        """Score answer quality compared to expected answer."""
        prompt = f"""Rate how well the actual answer matches the expected answer on a scale of 0-10.
Consider: accuracy, completeness, and relevance to the question.

Question: {query}

Expected Answer: {expected}

Actual Answer: {actual}

Return only a number between 0 and 10:"""

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}
        )

        try:
            score = float(response.json()["response"].strip().split()[0])
            return min(max(score / 10.0, 0.0), 1.0)
        except:
            return 0.5

    def evaluate_query(self, query: str, expected_answer: str) -> EvalResult:
        """Evaluate a single query through the RAG pipeline."""
        trace = None
        trace_id = None

        # Start Langfuse trace if available
        if self.langfuse:
            trace = self.langfuse.trace(name="rag_evaluation", input=query)
            trace_id = trace.id

        start_time = time.time()

        # Retrieval
        if trace:
            retrieval_span = trace.span(name="retrieval", input=query)

        docs = self._vector_search(query, limit=5)
        doc_contents = [d.get("content", "") for d in docs]

        if trace:
            retrieval_span.end(output={"num_docs": len(docs)})

        # Generation
        context = "\n\n".join(doc_contents)

        if trace:
            generation_span = trace.span(name="generation", input={"query": query, "context_length": len(context)})

        actual_answer = self._generate_answer(query, context)

        if trace:
            generation_span.end(output=actual_answer)

        latency_ms = (time.time() - start_time) * 1000

        # Scoring
        if trace:
            scoring_span = trace.span(name="scoring")

        relevance_score = self._score_relevance(query, doc_contents)
        answer_quality = self._score_answer_quality(query, expected_answer, actual_answer)

        if trace:
            scoring_span.end(output={"relevance": relevance_score, "quality": answer_quality})

        # Log scores to Langfuse
        if trace:
            trace.score(name="relevance", value=relevance_score)
            trace.score(name="answer_quality", value=answer_quality)
            trace.score(name="latency_ms", value=latency_ms)

        return EvalResult(
            query=query,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            retrieved_docs=doc_contents,
            relevance_score=relevance_score,
            answer_quality=answer_quality,
            latency_ms=latency_ms,
            trace_id=trace_id
        )

    def run_evaluation(self, test_cases: List[Dict]) -> Dict:
        """Run evaluation on multiple test cases."""
        results = []

        print(f"\n🧪 Running evaluation on {len(test_cases)} test cases...")
        print("=" * 60)

        for i, case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] {case['query'][:50]}...")

            result = self.evaluate_query(case["query"], case["expected_answer"])
            results.append(result)

            print(f"   Relevance: {result.relevance_score:.2f}")
            print(f"   Quality:   {result.answer_quality:.2f}")
            print(f"   Latency:   {result.latency_ms:.0f}ms")

        # Calculate aggregate metrics
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        avg_quality = sum(r.answer_quality for r in results) / len(results)
        avg_latency = sum(r.latency_ms for r in results) / len(results)

        summary = {
            "total_queries": len(results),
            "avg_relevance": avg_relevance,
            "avg_answer_quality": avg_quality,
            "avg_latency_ms": avg_latency,
            "results": [
                {
                    "query": r.query,
                    "relevance": r.relevance_score,
                    "quality": r.answer_quality,
                    "latency_ms": r.latency_ms,
                    "trace_id": r.trace_id
                }
                for r in results
            ]
        }

        return summary

    def add_test_document(self, content: str, metadata: Dict = None):
        """Add a document to the test collection."""
        embedding = self._get_embedding(content)
        doc = {
            "content": content,
            "metadata": metadata or {},
            "embedding": embedding
        }
        self.documents.insert_one(doc)

    def setup_test_data(self):
        """Setup sample test documents."""
        if self.documents.count_documents({}) > 0:
            return

        test_docs = [
            "Python is a high-level programming language known for its readability and versatility. It was created by Guido van Rossum and first released in 1991.",
            "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "Docker is a platform for developing, shipping, and running applications in containers. Containers package code and dependencies together.",
            "MongoDB is a document-oriented NoSQL database used for high volume data storage. It stores data in flexible, JSON-like documents.",
            "Neural networks are computing systems inspired by biological neural networks. They are the foundation of deep learning algorithms."
        ]

        for doc in test_docs:
            self.add_test_document(doc, {"source": "test_data"})

        print(f"✅ Added {len(test_docs)} test documents")


# Sample test cases
TEST_CASES = [
    {
        "query": "What is Python and who created it?",
        "expected_answer": "Python is a high-level programming language created by Guido van Rossum, first released in 1991."
    },
    {
        "query": "What is machine learning?",
        "expected_answer": "Machine learning is a subset of AI that enables systems to learn from experience without explicit programming."
    },
    {
        "query": "What are Docker containers?",
        "expected_answer": "Docker containers package applications with their code and dependencies together for consistent deployment."
    },
    {
        "query": "What type of database is MongoDB?",
        "expected_answer": "MongoDB is a document-oriented NoSQL database that stores data in JSON-like documents."
    },
    {
        "query": "What are neural networks based on?",
        "expected_answer": "Neural networks are computing systems inspired by biological neural networks in the brain."
    }
]


def print_report(summary: Dict):
    """Print evaluation report."""
    print("\n" + "=" * 60)
    print("📊 EVALUATION REPORT")
    print("=" * 60)

    print(f"\n📈 Aggregate Metrics:")
    print(f"   Total Queries:      {summary['total_queries']}")
    print(f"   Avg Relevance:      {summary['avg_relevance']:.2%}")
    print(f"   Avg Answer Quality: {summary['avg_answer_quality']:.2%}")
    print(f"   Avg Latency:        {summary['avg_latency_ms']:.0f}ms")

    print(f"\n📋 Per-Query Results:")
    print("-" * 60)

    for r in summary["results"]:
        status = "✅" if r["quality"] >= 0.7 else "⚠️" if r["quality"] >= 0.5 else "❌"
        print(f"{status} {r['query'][:40]}...")
        print(f"   Relevance: {r['relevance']:.2%} | Quality: {r['quality']:.2%} | Latency: {r['latency_ms']:.0f}ms")

    # Summary grades
    print("\n" + "-" * 60)
    overall = (summary['avg_relevance'] + summary['avg_answer_quality']) / 2

    if overall >= 0.8:
        grade = "A - Excellent"
    elif overall >= 0.7:
        grade = "B - Good"
    elif overall >= 0.6:
        grade = "C - Acceptable"
    elif overall >= 0.5:
        grade = "D - Needs Improvement"
    else:
        grade = "F - Poor"

    print(f"\n🎯 Overall Grade: {grade} ({overall:.2%})")

    if summary.get("results") and summary["results"][0].get("trace_id"):
        print(f"\n🔗 View detailed traces in Langfuse dashboard")


def main():
    """Run RAG evaluation."""
    print("🚀 RAG Evaluation Starting...")
    print("=" * 60)

    # Initialize evaluator
    evaluator = RAGEvaluator(
        mongo_url=os.getenv("MONGO_URL", "mongodb://localhost:27017/"),
        ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        langfuse_host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY")
    )

    # Setup test data
    evaluator.setup_test_data()

    # Run evaluation
    summary = evaluator.run_evaluation(TEST_CASES)

    # Print report
    print_report(summary)

    # Save results
    with open("eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Results saved to eval_results.json")


if __name__ == "__main__":
    main()
