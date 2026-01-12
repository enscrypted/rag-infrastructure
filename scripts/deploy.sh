#!/bin/bash
set -e

# ═══════════════════════════════════════════════════════════════════════════
# RAG Infrastructure Stack - Generic Deployment
# Complete AI/ML/RAG development environment for any homelab or server setup
# ═══════════════════════════════════════════════════════════════════════════

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
DEFAULT_DEPLOY_DIR="/opt/rag-infrastructure"
DEFAULT_DOCKER_NETWORK="rag-net"

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                    RAG Infrastructure Stack                                ║"
echo "║                   Generic Deployment Script                               ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "This script will deploy a complete AI/ML/RAG development environment with:"
echo "• Vector Databases: MongoDB, ChromaDB, Qdrant, Weaviate"
echo "• Graph Database: Neo4j"
echo "• Search: Elasticsearch"
echo "• LLM Tools: Ollama, Open WebUI"
echo "• Observability: Langfuse"
echo "• Storage: MinIO, Redis"
echo "• Dev Tools: Jupyter Lab, n8n"
echo "• Multiple MongoDB UIs including custom Vector Search interface"
echo ""

# Function to get user input with default
get_input() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"
    
    echo -e "${CYAN}$prompt${NC}"
    if [ ! -z "$default" ]; then
        read -p "[$default]: " input
        input=${input:-$default}
    else
        read -p ": " input
    fi
    
    eval "$var_name='$input'"
}

# Function to validate IP address
validate_ip() {
    local ip=$1
    if [[ $ip =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
        IFS='.' read -r -a ip_array <<< "$ip"
        [[ ${ip_array[0]} -le 255 && ${ip_array[1]} -le 255 && \
           ${ip_array[2]} -le 255 && ${ip_array[3]} -le 255 ]]
        return $?
    else
        return 1
    fi
}

# Function to validate deployment directory path
validate_deploy_dir() {
    local dir="$1"
    
    # Check if path is absolute and safe
    if [[ ! "$dir" =~ ^/[a-zA-Z0-9][a-zA-Z0-9/_-]*$ ]]; then
        echo "❌ Invalid path format. Use absolute path with alphanumeric characters, hyphens, and underscores only."
        return 1
    fi
    
    # Prevent dangerous system directories
    local dangerous_dirs=("/" "/bin" "/boot" "/dev" "/etc" "/home" "/lib" "/lib64" "/proc" "/root" "/run" "/sbin" "/sys" "/tmp" "/usr" "/var")
    for dangerous in "${dangerous_dirs[@]}"; do
        if [[ "$dir" = "$dangerous" ]] || [[ "$dir" = "$dangerous"/* ]] && [[ ${#dir} -le $((${#dangerous} + 10)) ]]; then
            echo "❌ Cannot use system directory '$dir' for deployment. Choose a dedicated path like /opt/rag-infrastructure"
            return 1
        fi
    done
    
    # Warn about existing directories in sensitive locations
    if [[ "$dir" =~ ^/(home|Users)/ ]]; then
        echo "⚠️  Warning: Deploying to user directory. Ensure this is intentional."
    fi
    
    return 0
}

# Configuration Collection
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}                           CONFIGURATION                                     ${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Get deployment directory with validation
while true; do
    get_input "Enter deployment directory" "$DEFAULT_DEPLOY_DIR" "DEPLOY_DIR"
    if validate_deploy_dir "$DEPLOY_DIR"; then
        break
    else
        echo -e "${RED}Please enter a valid and safe deployment directory.${NC}"
    fi
done

# Get host IP for external access
while true; do
    get_input "Enter the host/container IP address for external access (e.g., 192.168.1.100)" "" "HOST_IP"
    if validate_ip "$HOST_IP"; then
        break
    else
        echo -e "${RED}❌ Invalid IP address. Please enter a valid IPv4 address.${NC}"
    fi
done

# DNS/Proxy setup preference
echo ""
echo -e "${CYAN}DNS/Proxy Setup Preference:${NC}"
echo "1) I will configure DNS (Pi-hole) and reverse proxy (Nginx Proxy Manager) manually"
echo "2) I want to use the automated script (untested - may need manual adjustments)"
echo "3) I will access services directly via IP:PORT (no DNS/proxy needed)"
read -p "Choose option [1/2/3]: " dns_option

case $dns_option in
    1)
        SETUP_DNS=false
        USE_HOSTNAMES=true
        echo -e "${GREEN}✅ Manual DNS/proxy setup selected${NC}"
        ;;
    2)
        SETUP_DNS=true
        USE_HOSTNAMES=true
        echo -e "${YELLOW}⚠️ Automated DNS setup selected (may require manual verification)${NC}"
        ;;
    3)
        SETUP_DNS=false
        USE_HOSTNAMES=false
        echo -e "${GREEN}✅ Direct IP access selected${NC}"
        ;;
    *)
        SETUP_DNS=false
        USE_HOSTNAMES=true
        echo -e "${GREEN}✅ Defaulting to manual DNS/proxy setup${NC}"
        ;;
esac

# Service selection
echo ""
echo -e "${CYAN}Select services to deploy:${NC}"
echo "1) Essential stack (MongoDB, Neo4j, Langfuse, Ollama, Vector UI)"
echo "2) Full stack (All 18 services including multiple UIs and tools)"
read -p "Choose option [1/2]: " service_option

case $service_option in
    1)
        DEPLOY_PROFILE="essential"
        echo -e "${GREEN}✅ Essential stack selected${NC}"
        ;;
    *)
        DEPLOY_PROFILE="full"
        echo -e "${GREEN}✅ Full stack selected${NC}"
        ;;
esac

# Credentials
echo ""
echo -e "${CYAN}Credential Configuration:${NC}"
echo "For simplicity, you can use the same password for all services or customize individual ones."
echo ""
get_input "Enter default password for all services" "admin123" "DEFAULT_PASSWORD"

# Confirm deployment
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}                        DEPLOYMENT SUMMARY                                  ${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Deployment Directory: $DEPLOY_DIR"
echo "Host IP: $HOST_IP"
echo "Service Profile: $DEPLOY_PROFILE"
echo "DNS/Hostname Setup: $([ "$USE_HOSTNAMES" = true ] && echo "Enabled" || echo "Disabled")"
echo "Default Password: $DEFAULT_PASSWORD"
echo ""

# Warning for destructive operations
if [ -d "$DEPLOY_DIR" ]; then
    echo -e "${RED}⚠️ WARNING: Directory $DEPLOY_DIR already exists!${NC}"
    echo "This deployment will:"
    echo "  • Stop all existing containers"
    echo "  • Remove all containers and volumes"
    echo "  • Delete all data (CANNOT BE UNDONE)"
    echo "  • Create a fresh installation"
    echo ""
fi

read -p "Do you want to proceed with deployment? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Deployment cancelled."
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SERVICES_DIR="$( dirname "$SCRIPT_DIR" )/services"

# Verify vector search UI files exist
echo ""
echo -e "${BLUE}🔍 Checking required service files...${NC}"
if [ ! -d "$SERVICES_DIR/vector-search-ui" ]; then
    echo -e "${RED}❌ Error: vector-search-ui directory not found!${NC}"
    echo "Expected location: $SERVICES_DIR/vector-search-ui"
    echo "Please ensure the services directory is properly structured."
    exit 1
fi

if [ ! -f "$SERVICES_DIR/vector-search-ui/app.py" ] || [ ! -f "$SERVICES_DIR/vector-search-ui/Dockerfile" ]; then
    echo -e "${RED}❌ Error: vector-search-ui files missing!${NC}"
    echo "Required files: app.py, Dockerfile, templates/, static/"
    exit 1
fi

echo -e "${GREEN}✅ All required service files found${NC}"

# Cleanup existing deployment
if [ -d "$DEPLOY_DIR" ]; then
    echo -e "${RED}🔥 Cleaning up existing deployment...${NC}"
    cd "$DEPLOY_DIR" 2>/dev/null || true
    docker compose down -v 2>/dev/null || true
    docker system prune -f 2>/dev/null || true
    rm -rf "$DEPLOY_DIR"/*
fi

# Create fresh deployment directory
echo -e "${GREEN}📁 Creating deployment directory: $DEPLOY_DIR${NC}"
mkdir -p "$DEPLOY_DIR"
cd "$DEPLOY_DIR"

# Copy service files
echo "📂 Copying service files..."
cp -r "$SERVICES_DIR"/* .

# Generate docker-compose.yml based on profile and configuration
echo "📝 Generating docker-compose configuration..."

# Function to generate hostname URLs
get_service_url() {
    local service=$1
    local port=$2
    if [ "$USE_HOSTNAMES" = true ]; then
        echo "http://$service.home.local"
    else
        echo "http://$HOST_IP:$port"
    fi
}

# Create docker-compose.yml
cat > docker-compose.yml << COMPOSE_EOF
services:
  # ═══════════════════════════════════════════════════════════════════════════
  # VECTOR DATABASES
  # ═══════════════════════════════════════════════════════════════════════════
  
  mongodb:
    image: mongodb/mongodb-community-server:8.2.3-ubi9
    container_name: rag-mongodb
    restart: unless-stopped
    ports:
      - "$HOST_IP:27017:27017"
    volumes:
      - mongodb_data:/data/db
    command: mongod --replSet rs0 --bind_ip_all --noauth
    networks:
      - $DEFAULT_DOCKER_NETWORK
    healthcheck:
      test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
      interval: 10s
      timeout: 5s
      retries: 5

COMPOSE_EOF

# Add full stack services if selected
if [ "$DEPLOY_PROFILE" = "full" ]; then
    cat >> docker-compose.yml << COMPOSE_EOF
  chromadb:
    image: chromadb/chroma:latest
    container_name: rag-chromadb
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      CHROMA_SERVER_AUTH_PROVIDER: chromadb.auth.token_authn.TokenAuthServerProvider
      CHROMA_SERVER_AUTH_CREDENTIALS: $DEFAULT_PASSWORD
      CHROMA_AUTH_TOKEN_TRANSPORT_HEADER: X-Chroma-Token
      PERSIST_DIRECTORY: /chroma/data
      ANONYMIZED_TELEMETRY: 'false'
    volumes:
      - chromadb_data:/chroma/data
    networks:
      - $DEFAULT_DOCKER_NETWORK

  qdrant:
    image: qdrant/qdrant:latest
    container_name: rag-qdrant
    restart: unless-stopped
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - $DEFAULT_DOCKER_NETWORK

  weaviate:
    image: semitechnologies/weaviate:latest
    container_name: rag-weaviate
    restart: unless-stopped
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      CLUSTER_HOSTNAME: 'weaviate'
    volumes:
      - weaviate_data:/var/lib/weaviate
    networks:
      - $DEFAULT_DOCKER_NETWORK

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.17.0
    container_name: rag-elasticsearch
    restart: unless-stopped
    ports:
      - "9200:9200"
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    networks:
      - $DEFAULT_DOCKER_NETWORK

COMPOSE_EOF
fi

# Add essential services
cat >> docker-compose.yml << COMPOSE_EOF
  # ═══════════════════════════════════════════════════════════════════════════
  # GRAPH DATABASE
  # ═══════════════════════════════════════════════════════════════════════════
  
  neo4j:
    image: neo4j:5-community
    container_name: rag-neo4j
    restart: unless-stopped
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/$DEFAULT_PASSWORD
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
      NEO4J_dbms_memory_heap_initial__size: 512m
      NEO4J_dbms_memory_heap_max__size: 2G
    volumes:
      - neo4j_data:/data
    networks:
      - $DEFAULT_DOCKER_NETWORK

  # ═══════════════════════════════════════════════════════════════════════════
  # OBSERVABILITY
  # ═══════════════════════════════════════════════════════════════════════════
  
  langfuse-db:
    image: postgres:16-alpine
    container_name: rag-langfuse-db
    restart: unless-stopped
    environment:
      POSTGRES_DB: langfuse
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: $DEFAULT_PASSWORD
    volumes:
      - langfuse_db:/var/lib/postgresql/data
    networks:
      - $DEFAULT_DOCKER_NETWORK

  langfuse:
    image: langfuse/langfuse:2
    container_name: rag-langfuse
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://langfuse:$DEFAULT_PASSWORD@langfuse-db:5432/langfuse
      NEXTAUTH_URL: $(get_service_url "langfuse" "3000")
      NEXTAUTH_SECRET: $(openssl rand -base64 32)
      SALT: $(openssl rand -base64 16)
      TELEMETRY_ENABLED: 'false'
    depends_on:
      - langfuse-db
    networks:
      - $DEFAULT_DOCKER_NETWORK

  # ═══════════════════════════════════════════════════════════════════════════
  # LOCAL LLM
  # ═══════════════════════════════════════════════════════════════════════════
  
  ollama:
    image: ollama/ollama:latest
    container_name: rag-ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    networks:
      - $DEFAULT_DOCKER_NETWORK

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: rag-webui
    restart: unless-stopped
    ports:
      - "8085:8080"
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
      WEBUI_SECRET_KEY: $DEFAULT_PASSWORD
      WEBUI_URL: $(get_service_url "chat" "8085")
    volumes:
      - webui_data:/app/backend/data
    networks:
      - $DEFAULT_DOCKER_NETWORK

  # ═══════════════════════════════════════════════════════════════════════════
  # MONGODB UIs
  # ═══════════════════════════════════════════════════════════════════════════
  
  # Custom Vector Search UI
  vector-search-ui:
    build: ./vector-search-ui
    container_name: rag-vector-ui
    restart: unless-stopped
    ports:
      - "8090:8090"
    environment:
      MONGO_URL: mongodb://mongodb:27017
    depends_on:
      - mongodb
    networks:
      - $DEFAULT_DOCKER_NETWORK

COMPOSE_EOF

# Add full stack UIs and tools if selected
if [ "$DEPLOY_PROFILE" = "full" ]; then
    cat >> docker-compose.yml << COMPOSE_EOF
  # Additional MongoDB UIs
  mongoku:
    image: huggingface/mongoku:latest
    container_name: rag-mongoku
    restart: unless-stopped
    ports:
      - "3100:3100"
    environment:
      MONGOKU_DEFAULT_HOST: mongodb://mongodb:27017
      MONGOKU_SERVER_ORIGIN: $(get_service_url "mongoku" "3100")
    depends_on:
      - mongodb
    networks:
      - $DEFAULT_DOCKER_NETWORK

  mongo-express:
    image: mongo-express:latest
    container_name: rag-mongo-ui
    restart: unless-stopped
    ports:
      - "8081:8081"
    environment:
      ME_CONFIG_MONGODB_URL: mongodb://mongodb:27017/
      ME_CONFIG_MONGODB_SERVER: mongodb
      ME_CONFIG_BASICAUTH_USERNAME: admin
      ME_CONFIG_BASICAUTH_PASSWORD: $DEFAULT_PASSWORD
      ME_CONFIG_MONGODB_ENABLE_ADMIN: 'true'
    depends_on:
      - mongodb
    networks:
      - $DEFAULT_DOCKER_NETWORK

  # Additional services
  kibana:
    image: docker.elastic.co/kibana/kibana:8.17.0
    container_name: rag-kibana
    restart: unless-stopped
    ports:
      - "5601:5601"
    environment:
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
      SERVER_BASEPATH: ""
    depends_on:
      - elasticsearch
    networks:
      - $DEFAULT_DOCKER_NETWORK

  minio:
    image: minio/minio:latest
    container_name: rag-minio
    restart: unless-stopped
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: $DEFAULT_PASSWORD
      MINIO_BROWSER_REDIRECT_URL: $(get_service_url "minio" "9001")
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data
    networks:
      - $DEFAULT_DOCKER_NETWORK

  redis:
    image: redis:7-alpine
    container_name: rag-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - $DEFAULT_DOCKER_NETWORK

  redisinsight:
    image: redislabs/redisinsight:latest
    container_name: rag-redis-ui
    restart: unless-stopped
    ports:
      - "8001:5540"
    volumes:
      - redisinsight_data:/db
    networks:
      - $DEFAULT_DOCKER_NETWORK

  jupyter:
    image: jupyter/datascience-notebook:latest
    container_name: rag-jupyter
    restart: unless-stopped
    ports:
      - "8888:8888"
    environment:
      JUPYTER_ENABLE_LAB: "yes"
      JUPYTER_TOKEN: $DEFAULT_PASSWORD
    volumes:
      - jupyter_data:/home/jovyan
      - ./notebooks:/home/jovyan/work
    networks:
      - $DEFAULT_DOCKER_NETWORK

  n8n:
    image: n8nio/n8n:latest
    container_name: rag-n8n
    restart: unless-stopped
    ports:
      - "5678:5678"
    environment:
      N8N_BASIC_AUTH_ACTIVE: "true"
      N8N_BASIC_AUTH_USER: admin
      N8N_BASIC_AUTH_PASSWORD: $DEFAULT_PASSWORD
      N8N_HOST: $(get_service_url "n8n" "5678" | sed 's/http:\/\///')
      WEBHOOK_URL: $(get_service_url "n8n" "5678")/
    volumes:
      - n8n_data:/home/node/.n8n
    networks:
      - $DEFAULT_DOCKER_NETWORK

COMPOSE_EOF
fi

# Add networks and volumes
cat >> docker-compose.yml << COMPOSE_EOF

networks:
  $DEFAULT_DOCKER_NETWORK:
    driver: bridge

volumes:
  mongodb_data:
  neo4j_data:
  langfuse_db:
  ollama_models:
  webui_data:
COMPOSE_EOF

if [ "$DEPLOY_PROFILE" = "full" ]; then
    cat >> docker-compose.yml << COMPOSE_EOF
  chromadb_data:
  qdrant_data:
  weaviate_data:
  elasticsearch_data:
  minio_data:
  redis_data:
  redisinsight_data:
  jupyter_data:
  n8n_data:
COMPOSE_EOF
fi

# Create notebooks directory
mkdir -p notebooks

# Start the deployment
echo ""
echo -e "${GREEN}🚀 Starting deployment...${NC}"
docker compose up -d

# Wait for services
echo "⏳ Waiting for services to initialize (30 seconds)..."
sleep 30

# Initialize MongoDB replica set
echo "🔧 Initializing MongoDB replica set for vector search..."
docker exec rag-mongodb mongosh --eval "rs.initiate({_id: 'rs0', members: [{_id: 0, host: 'mongodb:27017'}]})" 2>/dev/null || true

# Pull Ollama models
echo "📥 Pulling essential Ollama models..."
docker exec rag-ollama ollama pull llama3.2:3b || echo "⚠️ Ollama model pull failed - you can pull models manually later"
docker exec rag-ollama ollama pull nomic-embed-text || echo "⚠️ Embedding model pull failed - you can pull manually later"

# Generate credentials file
echo "📝 Generating credentials file..."
cat > CREDENTIALS.md << CREDS_EOF
# RAG Infrastructure Stack - Access Information

## Default Credentials
- **Username**: admin
- **Password**: $DEFAULT_PASSWORD

## Service Access

### Core AI/ML Services
CREDS_EOF

if [ "$USE_HOSTNAMES" = true ]; then
    cat >> CREDENTIALS.md << CREDS_EOF
- **Vector Search UI**: http://vector-ui.home.local:8090
- **MongoDB**: mongodb://$HOST_IP:27017
- **Neo4j Browser**: http://neo4j.home.local:7474 (neo4j/$DEFAULT_PASSWORD)
- **Langfuse**: http://langfuse.home.local:3000
- **Chat Interface**: http://chat.home.local:8085
- **Ollama API**: http://ollama.home.local:11434
CREDS_EOF
else
    cat >> CREDENTIALS.md << CREDS_EOF
- **Vector Search UI**: http://$HOST_IP:8090
- **MongoDB**: mongodb://$HOST_IP:27017
- **Neo4j Browser**: http://$HOST_IP:7474 (neo4j/$DEFAULT_PASSWORD)
- **Langfuse**: http://$HOST_IP:3000
- **Chat Interface**: http://$HOST_IP:8085
- **Ollama API**: http://$HOST_IP:11434
CREDS_EOF
fi

if [ "$DEPLOY_PROFILE" = "full" ]; then
    if [ "$USE_HOSTNAMES" = true ]; then
        cat >> CREDENTIALS.md << CREDS_EOF

### Additional Services (Full Stack)
- **Mongoku**: http://mongoku.home.local:3100
- **Mongo Express**: http://mongo-ui.home.local:8081
- **ChromaDB**: http://$HOST_IP:8000 (Token: $DEFAULT_PASSWORD)
- **Qdrant**: http://$HOST_IP:6333
- **Weaviate**: http://$HOST_IP:8080
- **Elasticsearch**: http://$HOST_IP:9200
- **Kibana**: http://kibana.home.local:5601
- **MinIO Console**: http://minio.home.local:9001
- **Redis**: redis://$HOST_IP:6379
- **RedisInsight**: http://redis-ui.home.local:8001
- **Jupyter Lab**: http://jupyter.home.local:8888 (Token: $DEFAULT_PASSWORD)
- **n8n**: http://n8n.home.local:5678
CREDS_EOF
    else
        cat >> CREDENTIALS.md << CREDS_EOF

### Additional Services (Full Stack)
- **Mongoku**: http://$HOST_IP:3100
- **Mongo Express**: http://$HOST_IP:8081
- **ChromaDB**: http://$HOST_IP:8000 (Token: $DEFAULT_PASSWORD)
- **Qdrant**: http://$HOST_IP:6333
- **Weaviate**: http://$HOST_IP:8080
- **Elasticsearch**: http://$HOST_IP:9200
- **Kibana**: http://$HOST_IP:5601
- **MinIO Console**: http://$HOST_IP:9001
- **Redis**: redis://$HOST_IP:6379
- **RedisInsight**: http://$HOST_IP:8001
- **Jupyter Lab**: http://$HOST_IP:8888 (Token: $DEFAULT_PASSWORD)
- **n8n**: http://$HOST_IP:5678
CREDS_EOF
    fi
fi

cat >> CREDENTIALS.md << CREDS_EOF

## Connection Examples

### Python
\`\`\`python
from pymongo import MongoClient
client = MongoClient('mongodb://$HOST_IP:27017/')

from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://$HOST_IP:7687', auth=('neo4j', '$DEFAULT_PASSWORD'))
\`\`\`

### Environment Variables
\`\`\`bash
export MONGO_URL="mongodb://$HOST_IP:27017/"
export NEO4J_URI="bolt://$HOST_IP:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="$DEFAULT_PASSWORD"
export LANGFUSE_HOST="http://$HOST_IP:3000"
\`\`\`
CREDS_EOF

# Run DNS setup if requested
if [ "$SETUP_DNS" = true ]; then
    echo ""
    echo -e "${YELLOW}🔧 Running DNS/Proxy setup script...${NC}"
    if [ -f "$SCRIPT_DIR/setup-dns-and-proxy.sh" ]; then
        bash "$SCRIPT_DIR/setup-dns-and-proxy.sh" "$HOST_IP"
    else
        echo -e "${RED}❌ DNS setup script not found. You'll need to configure DNS manually.${NC}"
    fi
fi

# Display completion summary
clear
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                        ✅ DEPLOYMENT COMPLETE!                             ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Installation Directory: $DEPLOY_DIR"
echo "🌐 Host IP: $HOST_IP"
echo "🔑 Default Password: $DEFAULT_PASSWORD"
echo "📋 Access Details: See CREDENTIALS.md"
echo ""

if [ "$USE_HOSTNAMES" = true ]; then
    echo -e "${YELLOW}📌 DNS/Proxy Setup Required:${NC}"
    echo "Add these entries to your Pi-hole DNS (pointing to your reverse proxy):"
    echo "  mongoku.home.local, vector-ui.home.local, neo4j.home.local"
    echo "  langfuse.home.local, chat.home.local, etc."
    echo ""
    echo "Configure your reverse proxy (Nginx Proxy Manager) to route:"
    echo "  *.home.local → $HOST_IP:<service_port>"
    echo ""
fi

echo -e "${GREEN}✨ Services Status:${NC}"
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo -e "${GREEN}🎉 Your RAG Infrastructure Stack is ready for AI/ML development!${NC}"
echo ""
echo "Next steps:"
echo "• Check CREDENTIALS.md for access details"
echo "• Configure DNS/proxy if using hostnames"
echo "• Start experimenting with vector search at the Vector Search UI"
echo "• Pull additional Ollama models as needed"
echo ""