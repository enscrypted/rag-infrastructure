#!/usr/bin/env python3
"""
RAG Infrastructure Test Suite - Generic Version
================================================
Interactive test script that prompts for configuration.
Use this for new deployments or when testing different hosts.

For automated testing with pre-configured host, use:
    python test_infrastructure.py --host YOUR_IP --non-interactive

Resource-conscious design:
- Configurable delays between service tests
- Retry logic for transient network failures
- Generous timeouts to handle cold starts
- Sequential execution to minimize concurrent load
"""

import sys
import time
import json
import uuid
import requests
from datetime import datetime, timezone
from typing import List, Callable
from functools import wraps

# Try to import colorama, fallback to no-color if not available
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    class Fore:
        GREEN = RED = CYAN = YELLOW = ""
    class Style:
        RESET_ALL = ""
    HAS_COLOR = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    def tabulate(data, headers, tablefmt=None):
        result = " | ".join(headers) + "\n"
        result += "-" * 60 + "\n"
        for row in data:
            result += " | ".join(str(x) for x in row) + "\n"
        return result


# Default configuration - will be overridden by user input or CLI args
DEFAULT_CONFIG = {
    "host_ip": "",
    "test_prefix": "test_rag_",

    # Resource management
    "delay_between_services": 5,
    "delay_between_tests": 1,
    "max_retries": 2,
    "retry_delay": 3,

    # Timeouts (generous for cold starts and heavy load)
    "timeout_short": 15,
    "timeout_medium": 60,
    "timeout_long": 180,

    # Credentials (defaults - user can override)
    "chromadb_token": "admin",
    "neo4j_password": "password",
}


def get_user_config() -> dict:
    """Interactive configuration gathering"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"RAG Infrastructure Test Suite - Configuration")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    config = DEFAULT_CONFIG.copy()

    # Get host IP
    while not config["host_ip"]:
        host = input(f"{Fore.YELLOW}Enter host IP address: {Style.RESET_ALL}").strip()
        if host:
            config["host_ip"] = host

    # Test mode
    print(f"\n{Fore.YELLOW}Test modes:{Style.RESET_ALL}")
    print("  1. Quick (2s delays, basic tests)")
    print("  2. Normal (5s delays, full tests)")
    print("  3. Gentle (10s delays, extra retries)")

    mode = input(f"{Fore.YELLOW}Select mode [1-3, default=2]: {Style.RESET_ALL}").strip() or "2"

    if mode == "1":
        config["delay_between_services"] = 2
        config["delay_between_tests"] = 0.5
        config["max_retries"] = 1
    elif mode == "3":
        config["delay_between_services"] = 10
        config["delay_between_tests"] = 2
        config["max_retries"] = 3

    # Credentials
    print(f"\n{Fore.YELLOW}Credentials (press Enter for defaults):{Style.RESET_ALL}")

    neo4j_pass = input(f"  Neo4j password [{config['neo4j_password']}]: ").strip()
    if neo4j_pass:
        config["neo4j_password"] = neo4j_pass

    chroma_token = input(f"  ChromaDB token [{config['chromadb_token']}]: ").strip()
    if chroma_token:
        config["chromadb_token"] = chroma_token

    return config


class TestResult:
    def __init__(self, service: str, test: str, status: bool, message: str, duration: float):
        self.service = service
        self.test = test
        self.status = status
        self.message = message
        self.duration = duration


def retry_on_failure(max_retries: int = 2, delay: int = 3):
    """Decorator to retry failed tests"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    result, message = func(*args, **kwargs)
                    if result:
                        return result, message
                    if attempt < max_retries:
                        time.sleep(delay)
                        continue
                    return result, message
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay)
                        continue
                    return False, f"Exception after {max_retries + 1} attempts: {str(e)[:80]}"
            return False, str(last_exception)[:100] if last_exception else "Unknown error"
        return wrapper
    return decorator


class RagInfrastructureTester:
    def __init__(self, config: dict):
        self.config = config
        self.results: List[TestResult] = []
        self.host = config["host_ip"]
        self.test_id = str(uuid.uuid4())[:8]
        self.test_prefix = config["test_prefix"]

    def print_header(self, text: str):
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}{text}")
        print(f"{Fore.CYAN}{'='*60}\n")

    def run_test(self, service: str, test_name: str, test_func, *args, **kwargs) -> bool:
        time.sleep(self.config.get("delay_between_tests", 1))

        start_time = time.time()
        try:
            result, message = test_func(*args, **kwargs)
            duration = time.time() - start_time
            self.results.append(TestResult(service, test_name, result, message, duration))

            status_icon = f"{Fore.GREEN}✅" if result else f"{Fore.RED}❌"
            print(f"{status_icon} {service}: {test_name} - {message}")
            return result
        except Exception as e:
            duration = time.time() - start_time
            message = f"Exception: {str(e)[:100]}"
            self.results.append(TestResult(service, test_name, False, message, duration))
            print(f"{Fore.RED}❌ {service}: {test_name} - {message}")
            return False

    def service_pause(self):
        delay = self.config.get("delay_between_services", 5)
        print(f"{Fore.YELLOW}  [Pausing {delay}s...]{Style.RESET_ALL}")
        time.sleep(delay)

    # ═══════════════════════════════════════════════════════════════════════════
    # MONGODB
    # ═══════════════════════════════════════════════════════════════════════════

    def test_mongodb(self):
        self.print_header("Testing MongoDB - Full CRUD Operations")

        try:
            from pymongo import MongoClient
        except ImportError:
            self.results.append(TestResult("MongoDB", "Import", False, "pymongo not installed", 0))
            print(f"{Fore.RED}❌ MongoDB: pymongo not installed (pip install pymongo){Style.RESET_ALL}")
            return

        db_name = f"{self.test_prefix}db_{self.test_id}"
        client = None

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def connect():
            nonlocal client
            client = MongoClient(
                self.host, 27017,
                directConnection=True,
                serverSelectionTimeoutMS=self.config["timeout_short"] * 1000
            )
            client.server_info()
            return True, "Connected"

        if not self.run_test("MongoDB", "Connect", connect):
            return

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def create_docs():
            db = client[db_name]
            docs = [
                {"content": "Test document 1", "category": "test"},
                {"content": "Test document 2", "category": "test"},
            ]
            result = db.documents.insert_many(docs)
            return True, f"Inserted {len(result.inserted_ids)} documents"

        self.run_test("MongoDB", "CREATE", create_docs)

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def read_docs():
            db = client[db_name]
            count = db.documents.count_documents({})
            return True, f"Found {count} documents"

        self.run_test("MongoDB", "READ", read_docs)

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def update_docs():
            db = client[db_name]
            result = db.documents.update_one({"category": "test"}, {"$set": {"updated": True}})
            return True, f"Updated {result.modified_count} document(s)"

        self.run_test("MongoDB", "UPDATE", update_docs)

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def delete_docs():
            client.drop_database(db_name)
            return True, f"Dropped database '{db_name}'"

        self.run_test("MongoDB", "DELETE", delete_docs)

        if client:
            client.close()

    # ═══════════════════════════════════════════════════════════════════════════
    # NEO4J
    # ═══════════════════════════════════════════════════════════════════════════

    def test_neo4j(self):
        self.print_header("Testing Neo4j - Graph Operations")

        try:
            from neo4j import GraphDatabase
        except ImportError:
            self.results.append(TestResult("Neo4j", "Import", False, "neo4j not installed", 0))
            print(f"{Fore.RED}❌ Neo4j: neo4j driver not installed (pip install neo4j){Style.RESET_ALL}")
            return

        driver = None
        test_label = f"TestNode_{self.test_id}"

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def connect():
            nonlocal driver
            driver = GraphDatabase.driver(
                f"bolt://{self.host}:7687",
                auth=("neo4j", self.config["neo4j_password"])
            )
            driver.verify_connectivity()
            return True, "Connected via Bolt"

        if not self.run_test("Neo4j", "Connect", connect):
            return

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def create_nodes():
            driver.execute_query(
                f"CREATE (n:{test_label} {{name: 'Test', type: 'TestNode'}})",
                database_="neo4j"
            )
            return True, "Created test node"

        self.run_test("Neo4j", "CREATE", create_nodes)

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def read_nodes():
            records, _, _ = driver.execute_query(
                f"MATCH (n:{test_label}) RETURN n.name as name",
                database_="neo4j"
            )
            return True, f"Found {len(records)} node(s)"

        self.run_test("Neo4j", "READ", read_nodes)

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def delete_nodes():
            driver.execute_query(f"MATCH (n:{test_label}) DELETE n", database_="neo4j")
            return True, "Cleaned up test nodes"

        self.run_test("Neo4j", "DELETE", delete_nodes)

        if driver:
            driver.close()

    # ═══════════════════════════════════════════════════════════════════════════
    # OLLAMA
    # ═══════════════════════════════════════════════════════════════════════════

    def test_ollama(self):
        self.print_header("Testing Ollama - LLM Operations")

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def check_api():
            response = requests.get(
                f"http://{self.host}:11434/api/tags",
                timeout=self.config["timeout_short"]
            )
            models = response.json().get("models", [])
            return True, f"Found {len(models)} models"

        self.run_test("Ollama", "API Check", check_api)

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def generate_embedding():
            response = requests.post(
                f"http://{self.host}:11434/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": "test"},
                timeout=self.config["timeout_medium"]
            )
            if response.status_code == 200:
                emb = response.json().get("embedding", [])
                return True, f"Generated {len(emb)}-dim embedding"
            return False, f"Status {response.status_code}"

        self.run_test("Ollama", "Embeddings", generate_embedding)

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def generate_text():
            response = requests.post(
                f"http://{self.host}:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": "Say hello in 3 words.",
                    "stream": False
                },
                timeout=self.config["timeout_long"]
            )
            if response.status_code == 200:
                text = response.json().get("response", "")[:30]
                return True, f"Generated: '{text}...'"
            return False, f"Status {response.status_code}"

        self.run_test("Ollama", "Generation", generate_text)

    # ═══════════════════════════════════════════════════════════════════════════
    # CHROMADB (v2 API)
    # ═══════════════════════════════════════════════════════════════════════════

    def test_chromadb(self):
        self.print_header("Testing ChromaDB - Vector Database (v2 API)")

        collection_name = f"{self.test_prefix}coll_{self.test_id}"
        collection_id = None
        base_url = f"http://{self.host}:8000/api/v2/tenants/default_tenant/databases/default_database"
        headers = {"Authorization": f"Bearer {self.config['chromadb_token']}"}

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def heartbeat():
            response = requests.get(
                f"http://{self.host}:8000/api/v2/heartbeat",
                timeout=self.config["timeout_short"]
            )
            return response.status_code == 200, f"Status {response.status_code}"

        if not self.run_test("ChromaDB", "Heartbeat", heartbeat):
            return

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def create_collection():
            nonlocal collection_id
            response = requests.post(
                f"{base_url}/collections",
                headers=headers,
                json={"name": collection_name},
                timeout=self.config["timeout_short"]
            )
            if response.status_code in [200, 201]:
                collection_id = response.json().get("id")
                return True, f"Created '{collection_name}'"
            return False, f"Status {response.status_code}"

        self.run_test("ChromaDB", "CREATE", create_collection)

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def add_docs():
            if not collection_id:
                return False, "No collection ID"
            response = requests.post(
                f"{base_url}/collections/{collection_id}/add",
                headers=headers,
                json={
                    "ids": ["test1", "test2"],
                    "documents": ["Hello", "World"],
                    "embeddings": [[0.1]*768, [0.2]*768]
                },
                timeout=self.config["timeout_medium"]
            )
            return response.status_code in [200, 201], f"Status {response.status_code}"

        self.run_test("ChromaDB", "ADD", add_docs)

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def query():
            if not collection_id:
                return False, "No collection ID"
            response = requests.post(
                f"{base_url}/collections/{collection_id}/query",
                headers=headers,
                json={"query_embeddings": [[0.1]*768], "n_results": 2},
                timeout=self.config["timeout_medium"]
            )
            if response.status_code == 200:
                ids = response.json().get("ids", [[]])[0]
                return True, f"Found {len(ids)} results"
            return False, f"Status {response.status_code}"

        self.run_test("ChromaDB", "QUERY", query)

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def delete_collection():
            response = requests.delete(
                f"{base_url}/collections/{collection_name}",
                headers=headers,
                timeout=self.config["timeout_short"]
            )
            return response.status_code in [200, 204], f"Status {response.status_code}"

        self.run_test("ChromaDB", "DELETE", delete_collection)

    # ═══════════════════════════════════════════════════════════════════════════
    # ELASTICSEARCH
    # ═══════════════════════════════════════════════════════════════════════════

    def test_elasticsearch(self):
        self.print_header("Testing Elasticsearch - Search Operations")

        index_name = f"{self.test_prefix}idx_{self.test_id}"

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def health():
            response = requests.get(
                f"http://{self.host}:9200/_cluster/health",
                timeout=self.config["timeout_short"]
            )
            status = response.json().get("status", "unknown")
            return status in ["green", "yellow"], f"Cluster: {status}"

        if not self.run_test("Elasticsearch", "Health", health):
            return

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def create_index():
            response = requests.put(
                f"http://{self.host}:9200/{index_name}",
                json={"mappings": {"properties": {"content": {"type": "text"}}}},
                timeout=self.config["timeout_short"]
            )
            return response.status_code in [200, 201], f"Status {response.status_code}"

        self.run_test("Elasticsearch", "CREATE", create_index)

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def index_doc():
            requests.post(
                f"http://{self.host}:9200/{index_name}/_doc/1",
                json={"content": "Test document"},
                timeout=self.config["timeout_short"]
            )
            requests.post(f"http://{self.host}:9200/{index_name}/_refresh", timeout=5)
            return True, "Indexed document"

        self.run_test("Elasticsearch", "INDEX", index_doc)

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def search():
            response = requests.post(
                f"http://{self.host}:9200/{index_name}/_search",
                json={"query": {"match": {"content": "test"}}},
                timeout=self.config["timeout_short"]
            )
            if response.status_code == 200:
                hits = response.json().get("hits", {}).get("total", {}).get("value", 0)
                return True, f"Found {hits} hits"
            return False, f"Status {response.status_code}"

        self.run_test("Elasticsearch", "SEARCH", search)

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def delete_index():
            response = requests.delete(
                f"http://{self.host}:9200/{index_name}",
                timeout=self.config["timeout_short"]
            )
            return response.status_code in [200, 204], f"Status {response.status_code}"

        self.run_test("Elasticsearch", "DELETE", delete_index)

    # ═══════════════════════════════════════════════════════════════════════════
    # REDIS
    # ═══════════════════════════════════════════════════════════════════════════

    def test_redis(self):
        self.print_header("Testing Redis - Cache Operations")

        try:
            import redis
        except ImportError:
            self.results.append(TestResult("Redis", "Import", False, "redis not installed", 0))
            print(f"{Fore.RED}❌ Redis: redis-py not installed (pip install redis){Style.RESET_ALL}")
            return

        key = f"{self.test_prefix}{self.test_id}"
        r = None

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def connect():
            nonlocal r
            r = redis.Redis(host=self.host, port=6379, decode_responses=True,
                          socket_timeout=self.config["timeout_short"])
            r.ping()
            return True, "Connected"

        if not self.run_test("Redis", "Connect", connect):
            return

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def set_get():
            r.set(key, "test_value")
            val = r.get(key)
            return val == "test_value", f"Value: {val}"

        self.run_test("Redis", "SET/GET", set_get)

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def delete():
            r.delete(key)
            return True, "Deleted key"

        self.run_test("Redis", "DELETE", delete)

    # ═══════════════════════════════════════════════════════════════════════════
    # UI SERVICES
    # ═══════════════════════════════════════════════════════════════════════════

    def test_ui_services(self):
        self.print_header("Testing UI Services - Connectivity")

        services = [
            ("Langfuse", 3000),
            ("Vector Search UI", 8090),
            ("Mongoku", 3100),
            ("Mongo Express", 8081),
            ("Open WebUI", 8085),
            ("Kibana", 5601),
            ("RedisInsight", 8001),
            ("Jupyter Lab", 8888),
            ("n8n", 5678),
            ("MinIO Console", 9001),
        ]

        for name, port in services:
            @retry_on_failure(1, 2)
            def test_service(p=port):
                response = requests.get(
                    f"http://{self.host}:{p}",
                    timeout=self.config["timeout_short"],
                    allow_redirects=True
                )
                return True, f"HTTP {response.status_code}"

            self.run_test(name, "Web UI", test_service)

    # ═══════════════════════════════════════════════════════════════════════════
    # RAG PIPELINE
    # ═══════════════════════════════════════════════════════════════════════════

    def test_rag_pipeline(self):
        self.print_header("Testing Full RAG Pipeline")

        @retry_on_failure(self.config["max_retries"], self.config["retry_delay"])
        def pipeline():
            from pymongo import MongoClient

            # 1. Embed
            embed_resp = requests.post(
                f"http://{self.host}:11434/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": "What is AI?"},
                timeout=self.config["timeout_medium"]
            )
            embedding = embed_resp.json()["embedding"]

            # 2. Store
            client = MongoClient(self.host, 27017, directConnection=True)
            db = client[f"{self.test_prefix}pipeline"]
            db.docs.insert_one({"text": "AI is artificial intelligence", "embedding": embedding})

            # 3. Retrieve
            found = db.docs.find_one({})

            # 4. Generate
            gen_resp = requests.post(
                f"http://{self.host}:11434/api/generate",
                json={"model": "llama3.2:3b", "prompt": "Say 'ok'", "stream": False},
                timeout=self.config["timeout_long"]
            )

            # 5. Cleanup
            client.drop_database(f"{self.test_prefix}pipeline")
            client.close()

            return True, f"embed({len(embedding)}d) → store → retrieve → generate"

        self.run_test("RAG Pipeline", "End-to-End", pipeline)

    # ═══════════════════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════════════════

    def generate_report(self) -> bool:
        self.print_header("Test Report Summary")

        service_summary = {}
        for result in self.results:
            if result.service not in service_summary:
                service_summary[result.service] = {"passed": 0, "failed": 0}
            if result.status:
                service_summary[result.service]["passed"] += 1
            else:
                service_summary[result.service]["failed"] += 1

        table_data = []
        for service, data in service_summary.items():
            total = data["passed"] + data["failed"]
            status = "✅" if data["failed"] == 0 else "❌"
            table_data.append([status, service, f"{data['passed']}/{total}", f"{(data['passed']/total)*100:.0f}%"])

        print(tabulate(table_data, headers=["", "Service", "Passed", "Rate"], tablefmt="grid"))

        failed = [r for r in self.results if not r.status]
        if failed:
            print(f"\n{Fore.RED}Failed Tests:{Style.RESET_ALL}")
            for r in failed:
                print(f"  • {r.service} - {r.test}: {r.message}")

        total = len(self.results)
        passed = len([r for r in self.results if r.status])
        rate = (passed / total * 100) if total > 0 else 0

        print(f"\n{Fore.CYAN}Overall: {passed}/{total} ({rate:.1f}%){Style.RESET_ALL}")

        report = {
            "timestamp": datetime.now().isoformat(),
            "host": self.host,
            "test_id": self.test_id,
            "total": total,
            "passed": passed,
            "rate": rate,
            "results": [
                {"service": r.service, "test": r.test, "status": r.status, "message": r.message}
                for r in self.results
            ]
        }

        report_file = f"test_report_{self.test_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{Fore.CYAN}Report saved to: {report_file}{Style.RESET_ALL}")

        return rate >= 90

    def run_all_tests(self) -> bool:
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"RAG Infrastructure Test Suite")
        print(f"Host: {self.host}")
        print(f"Test ID: {self.test_id}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}{Style.RESET_ALL}\n")

        self.test_mongodb()
        self.service_pause()

        self.test_neo4j()
        self.service_pause()

        self.test_ollama()
        self.service_pause()

        self.test_chromadb()
        self.service_pause()

        self.test_elasticsearch()
        self.service_pause()

        self.test_redis()
        self.service_pause()

        self.test_ui_services()
        self.service_pause()

        self.test_rag_pipeline()

        return self.generate_report()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG Infrastructure Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_infrastructure.py                    # Interactive mode
  python test_infrastructure.py --host 192.168.1.100 --non-interactive
  python test_infrastructure.py --host 192.168.1.100 --gentle
        """
    )
    parser.add_argument("--host", help="Host IP address")
    parser.add_argument("--delay", type=int, default=5, help="Delay between services (seconds)")
    parser.add_argument("--gentle", action="store_true", help="Gentle mode (10s delays, more retries)")
    parser.add_argument("--non-interactive", action="store_true", help="Skip interactive prompts")
    parser.add_argument("--neo4j-password", default="password", help="Neo4j password")
    parser.add_argument("--chromadb-token", default="admin", help="ChromaDB token")
    args = parser.parse_args()

    # Get configuration
    if args.non_interactive and args.host:
        config = DEFAULT_CONFIG.copy()
        config["host_ip"] = args.host
        config["neo4j_password"] = args.neo4j_password
        config["chromadb_token"] = args.chromadb_token
        if args.gentle:
            config["delay_between_services"] = 10
            config["delay_between_tests"] = 2
            config["max_retries"] = 3
        else:
            config["delay_between_services"] = args.delay
    elif args.host:
        config = DEFAULT_CONFIG.copy()
        config["host_ip"] = args.host
        config["delay_between_services"] = args.delay
        if args.gentle:
            config["delay_between_services"] = 10
            config["delay_between_tests"] = 2
            config["max_retries"] = 3
    else:
        config = get_user_config()

    # Check packages
    required = ["pymongo", "neo4j", "redis", "requests"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"\n{Fore.YELLOW}Optional packages not installed: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}{Style.RESET_ALL}")
        print("Tests for missing packages will be skipped.\n")

    tester = RagInfrastructureTester(config)
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
