#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════
# DNS and Proxy Configuration Helper Script
# Generic configuration generator for RAG Infrastructure Stack
# 
# ⚠️  DISCLAIMER: This script is UNTESTED with different setups!
# It was developed for a specific Proxmox/Pi-hole/NPM environment.
# Please review and adapt the generated configurations for your setup.
# ═══════════════════════════════════════════════════════════════════════════

# Get configuration parameters
HOST_IP="$1"
PROXY_IP="$2"
DNS_DOMAIN="${3:-home.local}"

if [ -z "$HOST_IP" ]; then
    echo "Usage: $0 <host_ip> [proxy_ip] [dns_domain]"
    echo ""
    echo "Parameters:"
    echo "  host_ip     - IP where RAG services are running (e.g., 192.168.1.100)"
    echo "  proxy_ip    - IP of reverse proxy server (defaults to same as host_ip)"
    echo "  dns_domain  - Domain for services (defaults to 'home.local')"
    echo ""
    echo "Example: $0 192.168.1.100 192.168.1.105 home.local"
    exit 1
fi

# Default proxy IP to host IP if not specified
PROXY_IP="${PROXY_IP:-$HOST_IP}"

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                DNS and Proxy Configuration Helper                          ║"
echo "║                                                                          ║"
echo "║  ⚠️  WARNING: This script generates configuration examples that may      ║"
echo "║     need adaptation for your specific DNS/proxy setup!                  ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration for:"
echo "  • RAG Services Host: $HOST_IP"
echo "  • Reverse Proxy: $PROXY_IP"
echo "  • DNS Domain: $DNS_DOMAIN"
echo ""

# Service definitions with their ports and descriptions
cat > service-mappings.txt << MAPPINGS_EOF
# Core Services
mongodb.$DNS_DOMAIN:27017:MongoDB Database (use mongodb:// protocol)
mongoku.$DNS_DOMAIN:3100:Mongoku - Modern MongoDB UI
vector-ui.$DNS_DOMAIN:8090:Vector Search UI - Custom MongoDB Vector Search
mongo-ui.$DNS_DOMAIN:8081:MongoDB Express UI (Basic)
langfuse.$DNS_DOMAIN:3000:LLM Observability Platform
neo4j.$DNS_DOMAIN:7474:Neo4j Browser
neo4j-bolt.$DNS_DOMAIN:7687:Neo4j Bolt Protocol
elasticsearch.$DNS_DOMAIN:9200:Elasticsearch API
kibana.$DNS_DOMAIN:5601:Kibana Dashboard

# Vector Databases
chromadb.$DNS_DOMAIN:8000:ChromaDB Vector Database
qdrant.$DNS_DOMAIN:6333:Qdrant Vector Database
weaviate.$DNS_DOMAIN:8080:Weaviate GraphQL API

# Storage & Cache
minio.$DNS_DOMAIN:9001:MinIO S3 Console
minio-api.$DNS_DOMAIN:9000:MinIO S3 API
redis.$DNS_DOMAIN:6379:Redis Cache (use redis:// protocol)
redis-ui.$DNS_DOMAIN:8001:RedisInsight UI

# Local LLM
ollama.$DNS_DOMAIN:11434:Ollama API
chat.$DNS_DOMAIN:8085:Open WebUI Chat Interface

# Development Tools
jupyter.$DNS_DOMAIN:8888:Jupyter Lab
n8n.$DNS_DOMAIN:5678:n8n Workflow Automation
MAPPINGS_EOF

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: DNS Server Configuration (e.g., Pi-hole, pfSense, etc.)
# ═══════════════════════════════════════════════════════════════════════════

echo "📌 STEP 1: Configure DNS Server"
echo "==============================="
echo ""
echo "⚠️  DISCLAIMER: These instructions are examples for Pi-hole."
echo "   Adapt for your DNS server (pfSense, OPNsense, router, etc.)"
echo ""
echo "If using Pi-hole, access admin interface and add these DNS records:"
echo "Go to: Local DNS → DNS Records (or similar in your DNS server)"
echo ""
echo "Add all these entries pointing to your reverse proxy ($PROXY_IP):"
echo "────────────────────────────────────────────────────────────────────"

# Generate DNS entries
while IFS=: read -r hostname port description; do
  # Skip comments and empty lines
  [[ "$hostname" =~ ^#.*$ || -z "$hostname" ]] && continue
  echo "$hostname → $PROXY_IP"
done < service-mappings.txt

echo ""
echo "Alternative: Use Pi-hole CLI to add records:"
echo "────────────────────────────────────────────────────────────────────"
echo ""

# Generate generic Pi-hole CLI commands
cat > pihole-add-dns.sh << PIHOLE_EOF
#!/bin/bash
# Run this on your Pi-hole server
# ⚠️  ADAPT TO YOUR SETUP - this is an example for Pi-hole

# Add custom DNS entries
cat >> /etc/pihole/custom.list << EOF
PIHOLE_EOF

# Generate DNS entries for Pi-hole
while IFS=: read -r hostname port description; do
  [[ "$hostname" =~ ^#.*$ || -z "$hostname" ]] && continue
  echo "$PROXY_IP $hostname" >> pihole-add-dns.sh
done < service-mappings.txt

cat >> pihole-add-dns.sh << 'PIHOLE_EOF'
EOF

# Restart Pi-hole DNS (adapt command for your setup)
pihole restartdns
echo "✅ DNS records added!"
PIHOLE_EOF

echo "Pi-hole CLI script saved to: pihole-add-dns.sh"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Reverse Proxy Configuration (Nginx Proxy Manager, etc.)
# ═══════════════════════════════════════════════════════════════════════════

echo "📌 STEP 2: Configure Reverse Proxy"
echo "=================================="
echo ""
echo "⚠️  DISCLAIMER: These instructions are for Nginx Proxy Manager."
echo "   Adapt for your reverse proxy (Apache, Traefik, Caddy, etc.)"
echo ""
echo "If using Nginx Proxy Manager, access at: http://$PROXY_IP:81"
echo "Default login: admin@example.com / changeme"
echo ""
echo "Add these Proxy Hosts (Hosts → Add Proxy Host):"
echo "────────────────────────────────────────────────────────────────────"
echo ""

# Generate NPM configuration table
echo "Domain Name                    │ Forward Host    │ Port  │ Websockets │ Notes"
echo "──────────────────────────────┼────────────────┼───────┼────────────┼──────────────"

while IFS=: read -r hostname port description; do
  [[ "$hostname" =~ ^#.*$ || -z "$hostname" ]] && continue
  
  # Determine if websockets needed
  ws="No"
  [[ "$hostname" =~ jupyter\.$DNS_DOMAIN || "$hostname" =~ n8n\.$DNS_DOMAIN ]] && ws="Yes"
  
  # Format output
  printf "%-30s │ %-15s │ %-5s │ %-10s │ %s\n" \
    "$hostname" "$HOST_IP" "$port" "$ws" "$description"
done < service-mappings.txt

echo ""
echo "💡 Reverse Proxy Configuration Tips:"
echo "  • Enable 'Websocket Support' for services like Jupyter and n8n"
echo "  • Enable 'Block Common Exploits' for security"
echo "  • Consider SSL certificates via Let's Encrypt for external access"
echo "  • Test each service after configuration"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Generate environment file for applications
# ═══════════════════════════════════════════════════════════════════════════

echo "📌 STEP 3: Application Configuration"
echo "===================================="
echo ""
echo "For your applications, use these connection strings:"
echo ""

cat > app-connections.env << APP_ENV_EOF
# RAG Stack Connection Configuration
# Use these in your Python/Node.js applications
# Replace PASSWORD with your actual service passwords

# MongoDB (Vector Search)
MONGO_URL=mongodb://admin:PASSWORD@mongodb.$DNS_DOMAIN:27017/

# Neo4j (Graph Database)  
NEO4J_URI=bolt://neo4j.$DNS_DOMAIN:7687
NEO4J_BROWSER=http://neo4j.$DNS_DOMAIN

# Elasticsearch
ELASTICSEARCH_URL=http://elasticsearch.$DNS_DOMAIN:9200

# Vector Databases
CHROMADB_URL=http://chromadb.$DNS_DOMAIN:8000
QDRANT_URL=http://qdrant.$DNS_DOMAIN:6333
WEAVIATE_URL=http://weaviate.$DNS_DOMAIN:8080

# Observability
LANGFUSE_URL=http://langfuse.$DNS_DOMAIN

# Storage
MINIO_ENDPOINT=minio-api.$DNS_DOMAIN:9000
MINIO_CONSOLE=http://minio.$DNS_DOMAIN
REDIS_URL=redis://default:PASSWORD@redis.$DNS_DOMAIN:6379

# Local LLM
OLLAMA_BASE_URL=http://ollama.$DNS_DOMAIN:11434

# Development
JUPYTER_URL=http://jupyter.$DNS_DOMAIN
N8N_URL=http://n8n.$DNS_DOMAIN

# Direct IP access (if DNS not configured)
MONGO_URL_DIRECT=mongodb://admin:PASSWORD@$HOST_IP:27017/
NEO4J_URI_DIRECT=bolt://$HOST_IP:7687
LANGFUSE_URL_DIRECT=http://$HOST_IP:3000
OLLAMA_BASE_URL_DIRECT=http://$HOST_IP:11434
APP_ENV_EOF

echo "Connection strings saved to: app-connections.env"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Generate connectivity test script
# ═══════════════════════════════════════════════════════════════════════════

cat > test-connectivity.sh << TEST_EOF
#!/bin/bash
# Test connectivity to RAG infrastructure services

echo "Testing service connectivity..."
echo "Using domain: $DNS_DOMAIN"
echo ""

# Core web services to test
services=(
  "mongo-ui.$DNS_DOMAIN"
  "vector-ui.$DNS_DOMAIN"
  "langfuse.$DNS_DOMAIN"
  "neo4j.$DNS_DOMAIN"
  "chat.$DNS_DOMAIN"
TEST_EOF

# Add additional services if using full stack
cat >> test-connectivity.sh << 'TEST_EOF'
  # Uncomment these if using full stack
  # "kibana.$DNS_DOMAIN"
  # "minio.$DNS_DOMAIN"
  # "redis-ui.$DNS_DOMAIN"
  # "jupyter.$DNS_DOMAIN"
  # "n8n.$DNS_DOMAIN"
)

echo "Testing core services..."
for service in "${services[@]}"; do
  # Skip commented lines
  [[ "$service" =~ ^#.*$ ]] && continue
  
  if curl -s -o /dev/null -w "%{http_code}" "http://$service" 2>/dev/null | grep -q "200\|301\|302\|401\|403"; then
    echo "✅ $service - OK"
  else
    echo "❌ $service - Failed"
    echo "   Try: http://HOST_IP:PORT if DNS not configured"
  fi
done

echo ""
echo "Testing direct IP access..."
HOST_IP="$HOST_IP"
if curl -s -o /dev/null -w "%{http_code}" "http://$HOST_IP:8090" 2>/dev/null | grep -q "200\|301\|302"; then
  echo "✅ Direct IP access works: http://$HOST_IP:8090"
else
  echo "❌ Direct IP access failed - check if services are running"
fi
TEST_EOF

chmod +x test-connectivity.sh

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                        Configuration Summary                               ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "⚠️  IMPORTANT: This script generates EXAMPLE configurations!"
echo "   Please review and adapt all generated files for your specific setup."
echo ""
echo "📁 Files created:"
echo "  • service-mappings.txt     - Complete service list"
echo "  • pihole-add-dns.sh       - DNS setup script (EXAMPLE)"
echo "  • app-connections.env     - Connection strings for applications"
echo "  • test-connectivity.sh    - Connectivity test script"
echo ""
echo "📋 Next steps:"
echo ""
echo "1️⃣  Configure your DNS server:"
echo "    • Pi-hole: Run pihole-add-dns.sh (after reviewing/adapting)"
echo "    • Other DNS: Manually add entries from service-mappings.txt"
echo ""
echo "2️⃣  Configure your reverse proxy:"
echo "    • Nginx Proxy Manager: Add proxy hosts from table above"
echo "    • Other proxies: Create similar configurations"
echo ""
echo "3️⃣  Test connectivity:"
echo "    ./test-connectivity.sh"
echo ""
echo "4️⃣  Access your services:"
echo "    • Vector Search UI: http://vector-ui.$DNS_DOMAIN"
echo "    • Chat Interface:   http://chat.$DNS_DOMAIN"
echo "    • Langfuse:         http://langfuse.$DNS_DOMAIN"
echo "    • MongoDB UI:       http://mongo-ui.$DNS_DOMAIN"
echo "    • Neo4j Browser:    http://neo4j.$DNS_DOMAIN"
echo ""
echo "📖 Documentation:"
echo "   See README.md for detailed setup instructions and troubleshooting"
echo ""
echo "🔧 Alternative: Direct IP access (if DNS not configured)"
echo "   Vector Search UI: http://$HOST_IP:8090"
echo "   All services available at http://$HOST_IP:<PORT>"
echo ""