# n8n Workflow Automation Guide

n8n is a powerful workflow automation tool that enables you to create sophisticated RAG data pipelines, document processing workflows, and AI integrations without coding.

## Quick Access

| Component | URL | Credentials |
|-----------|-----|-------------|
| **n8n Editor** | `http://your-host:5678` | admin / your-password |
| **Webhook URL** | `http://your-host:5678/webhook/` | For external integrations |
| **API Endpoint** | `http://your-host:5678/api/v1/` | REST API access |

## Why n8n for RAG?

### Automation Benefits
- **Visual Workflow Design**: Create complex processes with drag-and-drop
- **No-Code RAG Pipelines**: Build document ingestion and processing flows
- **400+ Integrations**: Connect to databases, APIs, storage systems
- **Scheduling**: Automated document processing and maintenance tasks
- **Error Handling**: Robust error recovery and notification systems

### RAG Use Cases
- **Document Ingestion Pipelines**: Automate PDF processing, chunking, and embedding
- **Data Synchronization**: Keep vector databases updated with new content
- **Quality Assurance**: Automated testing and validation of RAG responses
- **Content Monitoring**: Watch for new documents and process automatically
- **Analytics Workflows**: Generate reports on RAG performance and usage

## Initial Setup

### 1. Access n8n Editor
```bash
# Navigate to n8n interface
open http://your-host:5678

# First time setup:
# Email: admin@yourdomain.com
# Password: your-password
# First name: Admin
# Last name: User
```

### 2. Configure Environment Variables
```bash
# n8n configuration for RAG integration
N8N_HOST=your-host
N8N_PORT=5678
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=your-password

# Database connections
MONGODB_URL=mongodb://mongodb:27017/rag_db
REDIS_URL=redis://redis:6379
OLLAMA_URL=http://ollama:11434
LANGFUSE_URL=http://langfuse:3000
```

### 3. Test Basic Workflow
```json
{
  "name": "Test RAG Connection",
  "nodes": [
    {
      "parameters": {},
      "name": "When clicking 'Test workflow'",
      "type": "n8n-nodes-base.manualTrigger",
      "typeVersion": 1,
      "position": [250, 300]
    },
    {
      "parameters": {
        "url": "http://ollama:11434/api/tags",
        "options": {}
      },
      "name": "Test Ollama Connection",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [450, 300]
    }
  ],
  "connections": {
    "When clicking 'Test workflow'": {
      "main": [
        [
          {
            "node": "Test Ollama Connection",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

## RAG Workflow Templates

### 1. Document Processing Pipeline
```json
{
  "name": "RAG Document Processing Pipeline",
  "nodes": [
    {
      "parameters": {
        "path": "process-document",
        "options": {}
      },
      "name": "Webhook - New Document",
      "type": "n8n-nodes-base.webhook",
      "typeVersion": 1,
      "position": [250, 300]
    },
    {
      "parameters": {
        "mode": "runOnceForEachItem",
        "jsCode": "// Extract document information\nconst documentUrl = $input.item.json.document_url;\nconst documentType = $input.item.json.document_type;\nconst metadata = $input.item.json.metadata || {};\n\nreturn {\n  document_url: documentUrl,\n  document_type: documentType,\n  metadata: metadata,\n  processing_id: Date.now().toString(),\n  status: 'processing'\n};"
      },
      "name": "Extract Document Info",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [450, 300]
    },
    {
      "parameters": {
        "url": "={{$node['Extract Document Info'].json.document_url}}",
        "options": {
          "response": {
            "response": {
              "responseFormat": "file"
            }
          }
        }
      },
      "name": "Download Document",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [650, 300]
    },
    {
      "parameters": {
        "mode": "runOnceForEachItem",
        "jsCode": "// Extract text from document\n// This is a simplified example - in practice you'd use PDF parsers\nconst content = $input.item.binary.data;\nconst documentText = content.toString(); // Simplified\n\n// Chunk the document\nfunction chunkText(text, maxLength = 1000) {\n  const chunks = [];\n  let start = 0;\n  \n  while (start < text.length) {\n    let end = start + maxLength;\n    if (end < text.length) {\n      // Try to break at sentence end\n      const lastPeriod = text.lastIndexOf('.', end);\n      if (lastPeriod > start + 500) {\n        end = lastPeriod + 1;\n      }\n    }\n    \n    chunks.push(text.slice(start, end).trim());\n    start = end;\n  }\n  \n  return chunks;\n}\n\nconst chunks = chunkText(documentText);\n\nreturn chunks.map((chunk, index) => ({\n  chunk_text: chunk,\n  chunk_index: index,\n  document_id: $input.item.json.processing_id,\n  total_chunks: chunks.length\n}));"
      },
      "name": "Extract and Chunk Text",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [850, 300]
    },
    {
      "parameters": {
        "url": "http://ollama:11434/api/embeddings",
        "sendHeaders": true,
        "headerParameters": {
          "parameters": [
            {
              "name": "Content-Type",
              "value": "application/json"
            }
          ]
        },
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "model",
              "value": "nomic-embed-text"
            },
            {
              "name": "prompt",
              "value": "={{$json.chunk_text}}"
            }
          ]
        },
        "options": {}
      },
      "name": "Generate Embeddings",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [1050, 300]
    },
    {
      "parameters": {
        "operation": "insertMany",
        "collection": "documents",
        "fields": "document_id,chunk_text,chunk_index,total_chunks,embedding,created_at",
        "options": {}
      },
      "name": "Store in MongoDB",
      "type": "n8n-nodes-base.mongoDb",
      "typeVersion": 1,
      "position": [1250, 300]
    }
  ],
  "connections": {
    "Webhook - New Document": {
      "main": [
        [
          {
            "node": "Extract Document Info",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Extract Document Info": {
      "main": [
        [
          {
            "node": "Download Document",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Download Document": {
      "main": [
        [
          {
            "node": "Extract and Chunk Text",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Extract and Chunk Text": {
      "main": [
        [
          {
            "node": "Generate Embeddings",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Generate Embeddings": {
      "main": [
        [
          {
            "node": "Store in MongoDB",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

### 2. RAG Quality Assurance Workflow
```json
{
  "name": "RAG Quality Assurance",
  "nodes": [
    {
      "parameters": {
        "rule": {
          "interval": [
            {
              "field": "hours"
            }
          ]
        }
      },
      "name": "Schedule - Hourly QA",
      "type": "n8n-nodes-base.cron",
      "typeVersion": 1,
      "position": [250, 300]
    },
    {
      "parameters": {
        "operation": "find",
        "collection": "test_questions",
        "options": {}
      },
      "name": "Get Test Questions",
      "type": "n8n-nodes-base.mongoDb",
      "typeVersion": 1,
      "position": [450, 300]
    },
    {
      "parameters": {
        "mode": "runOnceForEachItem",
        "jsCode": "// Test RAG query\nconst question = $input.item.json.question;\nconst expectedAnswer = $input.item.json.expected_answer;\nconst questionId = $input.item.json._id;\n\nreturn {\n  question: question,\n  expected_answer: expectedAnswer,\n  question_id: questionId,\n  test_timestamp: new Date().toISOString()\n};"
      },
      "name": "Prepare Test Data",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [650, 300]
    },
    {
      "parameters": {
        "url": "http://your-rag-api:8000/query",
        "sendHeaders": true,
        "headerParameters": {
          "parameters": [
            {
              "name": "Content-Type",
              "value": "application/json"
            }
          ]
        },
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "query",
              "value": "={{$json.question}}"
            }
          ]
        },
        "options": {}
      },
      "name": "Query RAG System",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [850, 300]
    },
    {
      "parameters": {
        "mode": "runOnceForEachItem",
        "jsCode": "// Evaluate response quality\nconst actualAnswer = $input.item.json.response;\nconst expectedAnswer = $input.item.json.expected_answer;\nconst question = $input.item.json.question;\n\n// Simple similarity scoring (in practice, use more sophisticated methods)\nfunction calculateSimilarity(str1, str2) {\n  const words1 = str1.toLowerCase().split(' ');\n  const words2 = str2.toLowerCase().split(' ');\n  \n  const commonWords = words1.filter(word => words2.includes(word));\n  const similarity = commonWords.length / Math.max(words1.length, words2.length);\n  \n  return similarity;\n}\n\nconst similarityScore = calculateSimilarity(actualAnswer, expectedAnswer);\nconst qualityScore = similarityScore > 0.7 ? 'PASS' : 'FAIL';\n\nreturn {\n  question: question,\n  expected_answer: expectedAnswer,\n  actual_answer: actualAnswer,\n  similarity_score: similarityScore,\n  quality_score: qualityScore,\n  test_timestamp: $input.item.json.test_timestamp,\n  question_id: $input.item.json.question_id\n};"
      },
      "name": "Evaluate Quality",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [1050, 300]
    },
    {
      "parameters": {
        "operation": "insert",
        "collection": "quality_reports",
        "fields": "question,expected_answer,actual_answer,similarity_score,quality_score,test_timestamp,question_id",
        "options": {}
      },
      "name": "Store QA Results",
      "type": "n8n-nodes-base.mongoDb",
      "typeVersion": 1,
      "position": [1250, 300]
    },
    {
      "parameters": {
        "conditions": {
          "string": [
            {
              "value1": "={{$json.quality_score}}",
              "value2": "FAIL"
            }
          ]
        }
      },
      "name": "Check if Failed",
      "type": "n8n-nodes-base.if",
      "typeVersion": 1,
      "position": [1450, 300]
    },
    {
      "parameters": {
        "fromEmail": "alerts@yourdomain.com",
        "toEmail": "admin@yourdomain.com",
        "subject": "RAG Quality Alert",
        "text": "RAG quality test failed for question: {{$json.question}}\n\nExpected: {{$json.expected_answer}}\nActual: {{$json.actual_answer}}\nSimilarity Score: {{$json.similarity_score}}"
      },
      "name": "Send Alert Email",
      "type": "n8n-nodes-base.emailSend",
      "typeVersion": 2,
      "position": [1650, 200]
    }
  ]
}
```

### 3. Content Synchronization Workflow
```json
{
  "name": "Content Sync from External Sources",
  "nodes": [
    {
      "parameters": {
        "rule": {
          "interval": [
            {
              "field": "minutes",
              "minutesInterval": 30
            }
          ]
        }
      },
      "name": "Schedule - Every 30 min",
      "type": "n8n-nodes-base.cron",
      "typeVersion": 1,
      "position": [250, 300]
    },
    {
      "parameters": {
        "url": "https://api.example.com/documents",
        "authentication": "headerAuth",
        "headerAuth": {
          "name": "Authorization",
          "value": "Bearer your-api-token"
        },
        "sendQuery": true,
        "queryParameters": {
          "parameters": [
            {
              "name": "since",
              "value": "={{DateTime.now().minus({minutes: 30}).toISO()}}"
            }
          ]
        },
        "options": {}
      },
      "name": "Fetch New Documents",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [450, 300]
    },
    {
      "parameters": {
        "conditions": {
          "number": [
            {
              "value1": "={{$json.documents.length}}",
              "operation": "larger",
              "value2": 0
            }
          ]
        }
      },
      "name": "Check if New Documents",
      "type": "n8n-nodes-base.if",
      "typeVersion": 1,
      "position": [650, 300]
    },
    {
      "parameters": {
        "fieldName": "documents",
        "options": {}
      },
      "name": "Split Documents",
      "type": "n8n-nodes-base.itemLists",
      "typeVersion": 3,
      "position": [850, 300]
    },
    {
      "parameters": {
        "url": "http://n8n:5678/webhook/process-document",
        "sendHeaders": true,
        "headerParameters": {
          "parameters": [
            {
              "name": "Content-Type",
              "value": "application/json"
            }
          ]
        },
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "document_url",
              "value": "={{$json.url}}"
            },
            {
              "name": "document_type",
              "value": "={{$json.type}}"
            },
            {
              "name": "metadata",
              "value": "={{$json.metadata}}"
            }
          ]
        },
        "options": {}
      },
      "name": "Trigger Document Processing",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [1050, 300]
    }
  ]
}
```

## Advanced Workflow Patterns

### 1. Error Handling and Retry Logic
```json
{
  "name": "Robust Document Processing with Error Handling",
  "nodes": [
    {
      "parameters": {
        "path": "robust-process",
        "options": {}
      },
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "typeVersion": 1,
      "position": [250, 300]
    },
    {
      "parameters": {
        "mode": "runOnceForEachItem",
        "jsCode": "// Initialize retry counter\nreturn {\n  ...item.json,\n  retry_count: 0,\n  max_retries: 3\n};"
      },
      "name": "Initialize Retry Counter",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [450, 300]
    },
    {
      "parameters": {
        "resource": "document",
        "operation": "process",
        "additionalFields": {}
      },
      "name": "Process Document",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [650, 300],
      "onError": "continueErrorOutput"
    },
    {
      "parameters": {
        "conditions": {
          "boolean": [
            {
              "value1": "={{$json.error !== undefined}}"
            }
          ]
        }
      },
      "name": "Check for Error",
      "type": "n8n-nodes-base.if",
      "typeVersion": 1,
      "position": [850, 300]
    },
    {
      "parameters": {
        "conditions": {
          "number": [
            {
              "value1": "={{$json.retry_count}}",
              "operation": "smaller",
              "value2": "={{$json.max_retries}}"
            }
          ]
        }
      },
      "name": "Check Retry Count",
      "type": "n8n-nodes-base.if",
      "typeVersion": 1,
      "position": [1050, 200]
    },
    {
      "parameters": {
        "amount": 5,
        "unit": "seconds"
      },
      "name": "Wait Before Retry",
      "type": "n8n-nodes-base.wait",
      "typeVersion": 1,
      "position": [1250, 100]
    },
    {
      "parameters": {
        "mode": "runOnceForEachItem",
        "jsCode": "// Increment retry counter\nreturn {\n  ...item.json,\n  retry_count: item.json.retry_count + 1\n};"
      },
      "name": "Increment Retry",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [1450, 100]
    },
    {
      "parameters": {
        "operation": "insert",
        "collection": "failed_documents",
        "fields": "document_id,error_message,retry_count,failed_at",
        "options": {}
      },
      "name": "Log Failure",
      "type": "n8n-nodes-base.mongoDb",
      "typeVersion": 1,
      "position": [1250, 300]
    }
  ]
}
```

### 2. Data Validation and Quality Checks
```json
{
  "name": "Data Validation Pipeline",
  "nodes": [
    {
      "parameters": {
        "mode": "runOnceForEachItem",
        "jsCode": "// Validate document data\nconst doc = $input.item.json;\nconst errors = [];\n\n// Check required fields\nif (!doc.title || doc.title.trim().length === 0) {\n  errors.push('Title is required');\n}\n\nif (!doc.content || doc.content.trim().length < 10) {\n  errors.push('Content must be at least 10 characters');\n}\n\n// Check content quality\nconst wordCount = doc.content.split(' ').length;\nif (wordCount < 5) {\n  errors.push('Content too short');\n}\n\n// Check for suspicious content\nconst suspiciousPatterns = ['spam', 'click here', 'buy now'];\nconst hasSuspiciousContent = suspiciousPatterns.some(pattern => \n  doc.content.toLowerCase().includes(pattern)\n);\n\nif (hasSuspiciousContent) {\n  errors.push('Content contains suspicious patterns');\n}\n\nreturn {\n  ...doc,\n  validation_errors: errors,\n  is_valid: errors.length === 0,\n  validation_timestamp: new Date().toISOString()\n};"
      },
      "name": "Validate Document",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [450, 300]
    },
    {
      "parameters": {
        "conditions": {
          "boolean": [
            {
              "value1": "={{$json.is_valid}}"
            }
          ]
        }
      },
      "name": "Check Validation",
      "type": "n8n-nodes-base.if",
      "typeVersion": 1,
      "position": [650, 300]
    }
  ]
}
```

## Integration with RAG Services

### 1. MongoDB Integration
```javascript
// n8n MongoDB node configuration
{
  "parameters": {
    "operation": "find",
    "collection": "documents",
    "query": "{ \"category\": \"{{$json.category}}\" }",
    "options": {
      "sort": "{ \"created_at\": -1 }",
      "limit": 10
    }
  },
  "name": "Query MongoDB",
  "type": "n8n-nodes-base.mongoDb"
}

// Aggregation pipeline example
{
  "parameters": {
    "operation": "aggregate",
    "collection": "documents", 
    "query": "[{\"$match\": {\"category\": \"{{$json.category}}\"}}, {\"$group\": {\"_id\": \"$source\", \"count\": {\"$sum\": 1}}}]"
  },
  "name": "Aggregate Documents",
  "type": "n8n-nodes-base.mongoDb"
}
```

### 2. Redis Integration
```javascript
// Redis operations via HTTP requests
{
  "parameters": {
    "url": "http://redis:6379",
    "method": "POST",
    "sendHeaders": true,
    "headerParameters": {
      "parameters": [
        {
          "name": "Content-Type",
          "value": "application/json"
        }
      ]
    },
    "sendBody": true,
    "bodyParameters": {
      "parameters": [
        {
          "name": "command",
          "value": "SET"
        },
        {
          "name": "key",
          "value": "cache:{{$json.document_id}}"
        },
        {
          "name": "value",
          "value": "{{$json.content}}"
        }
      ]
    }
  },
  "name": "Cache in Redis",
  "type": "n8n-nodes-base.httpRequest"
}
```

### 3. Ollama Integration
```javascript
// Generate embeddings
{
  "parameters": {
    "url": "http://ollama:11434/api/embeddings",
    "sendBody": true,
    "bodyParameters": {
      "parameters": [
        {
          "name": "model",
          "value": "nomic-embed-text"
        },
        {
          "name": "prompt", 
          "value": "={{$json.text}}"
        }
      ]
    }
  },
  "name": "Generate Embedding",
  "type": "n8n-nodes-base.httpRequest"
}

// Generate text response
{
  "parameters": {
    "url": "http://ollama:11434/api/generate",
    "sendBody": true,
    "bodyParameters": {
      "parameters": [
        {
          "name": "model",
          "value": "llama3.2:3b"
        },
        {
          "name": "prompt",
          "value": "Context: {{$json.context}}\n\nQuestion: {{$json.question}}\n\nAnswer:"
        },
        {
          "name": "stream",
          "value": false
        }
      ]
    }
  },
  "name": "Generate Response",
  "type": "n8n-nodes-base.httpRequest"
}
```

### 4. Langfuse Tracing Integration
```javascript
// Send trace to Langfuse
{
  "parameters": {
    "url": "http://langfuse:3000/api/public/ingestion",
    "sendHeaders": true,
    "headerParameters": {
      "parameters": [
        {
          "name": "Authorization",
          "value": "Bearer {{$credentials.langfuse.secretKey}}"
        },
        {
          "name": "Content-Type",
          "value": "application/json"
        }
      ]
    },
    "sendBody": true,
    "bodyParameters": {
      "parameters": [
        {
          "name": "batch",
          "value": "[{\"id\": \"{{$json.trace_id}}\", \"type\": \"trace\", \"timestamp\": \"{{$json.timestamp}}\", \"body\": {\"name\": \"document_processing\", \"input\": \"{{$json.input}}\", \"output\": \"{{$json.output}}\"}}]"
        }
      ]
    }
  },
  "name": "Send to Langfuse",
  "type": "n8n-nodes-base.httpRequest"
}
```

## Custom Functions and Utilities

### 1. Text Processing Functions
```javascript
// Custom text processing node
{
  "parameters": {
    "mode": "runOnceForEachItem",
    "jsCode": "// Advanced text processing utilities\n\nclass TextProcessor {\n  static cleanText(text) {\n    return text\n      .replace(/\\s+/g, ' ')\n      .replace(/[^a-zA-Z0-9\\s.,!?-]/g, '')\n      .trim();\n  }\n  \n  static extractKeywords(text, minLength = 3, maxKeywords = 10) {\n    const words = text.toLowerCase().split(/\\s+/);\n    const stopWords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'];\n    \n    const keywords = words\n      .filter(word => word.length >= minLength)\n      .filter(word => !stopWords.includes(word))\n      .reduce((acc, word) => {\n        acc[word] = (acc[word] || 0) + 1;\n        return acc;\n      }, {});\n    \n    return Object.entries(keywords)\n      .sort(([,a], [,b]) => b - a)\n      .slice(0, maxKeywords)\n      .map(([word]) => word);\n  }\n  \n  static estimateReadingTime(text, wordsPerMinute = 200) {\n    const wordCount = text.split(/\\s+/).length;\n    return Math.ceil(wordCount / wordsPerMinute);\n  }\n  \n  static detectLanguage(text) {\n    // Simple language detection based on character patterns\n    const englishPattern = /^[a-zA-Z\\s.,!?-]+$/;\n    return englishPattern.test(text) ? 'en' : 'unknown';\n  }\n}\n\nconst inputText = $input.item.json.content;\n\nreturn {\n  original_text: inputText,\n  cleaned_text: TextProcessor.cleanText(inputText),\n  keywords: TextProcessor.extractKeywords(inputText),\n  reading_time_minutes: TextProcessor.estimateReadingTime(inputText),\n  detected_language: TextProcessor.detectLanguage(inputText),\n  word_count: inputText.split(/\\s+/).length,\n  character_count: inputText.length\n};"
  },
  "name": "Advanced Text Processing",
  "type": "n8n-nodes-base.code"
}
```

### 2. Quality Scoring Functions
```javascript
// Quality assessment node
{
  "parameters": {
    "mode": "runOnceForEachItem",
    "jsCode": "// Quality assessment utilities\n\nclass QualityAssessor {\n  static assessContentQuality(text) {\n    const scores = {};\n    \n    // Length score (0-1)\n    const wordCount = text.split(/\\s+/).length;\n    scores.length = Math.min(wordCount / 100, 1); // Optimal around 100 words\n    \n    // Readability score (simplified)\n    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);\n    const avgWordsPerSentence = wordCount / sentences.length;\n    scores.readability = avgWordsPerSentence < 20 ? 1 : Math.max(0, 1 - (avgWordsPerSentence - 20) / 20);\n    \n    // Diversity score (unique words ratio)\n    const words = text.toLowerCase().split(/\\s+/);\n    const uniqueWords = new Set(words);\n    scores.diversity = uniqueWords.size / words.length;\n    \n    // Overall score\n    scores.overall = (scores.length + scores.readability + scores.diversity) / 3;\n    \n    return scores;\n  }\n  \n  static assessRelevance(query, document) {\n    const queryWords = query.toLowerCase().split(/\\s+/);\n    const docWords = document.toLowerCase().split(/\\s+/);\n    \n    const matches = queryWords.filter(word => docWords.includes(word));\n    return matches.length / queryWords.length;\n  }\n}\n\nconst content = $input.item.json.content;\nconst query = $input.item.json.query || '';\n\nconst qualityScores = QualityAssessor.assessContentQuality(content);\nconst relevanceScore = query ? QualityAssessor.assessRelevance(query, content) : null;\n\nreturn {\n  ...item.json,\n  quality_assessment: {\n    scores: qualityScores,\n    relevance_score: relevanceScore,\n    assessment_timestamp: new Date().toISOString()\n  }\n};"
  },
  "name": "Quality Assessment",
  "type": "n8n-nodes-base.code"
}
```

## Monitoring and Analytics

### 1. Workflow Performance Monitoring
```javascript
// Performance monitoring node
{
  "parameters": {
    "mode": "runOnceForEachItem", 
    "jsCode": "// Track workflow performance\n\nclass PerformanceMonitor {\n  static trackWorkflowStep(stepName, startTime, endTime, data) {\n    const duration = endTime - startTime;\n    \n    return {\n      step_name: stepName,\n      duration_ms: duration,\n      start_time: new Date(startTime).toISOString(),\n      end_time: new Date(endTime).toISOString(),\n      data_size: JSON.stringify(data).length,\n      timestamp: new Date().toISOString()\n    };\n  }\n}\n\n// Record step performance\nconst stepStart = $input.item.json.step_start_time || Date.now();\nconst stepEnd = Date.now();\nconst stepName = 'document_processing';\n\nconst performanceData = PerformanceMonitor.trackWorkflowStep(\n  stepName,\n  stepStart,\n  stepEnd,\n  $input.item.json\n);\n\nreturn {\n  ...item.json,\n  performance: performanceData,\n  step_end_time: stepEnd\n};"
  },
  "name": "Track Performance",
  "type": "n8n-nodes-base.code"
}
```

### 2. Error Tracking and Alerting
```javascript
// Error tracking node
{
  "parameters": {
    "mode": "runOnceForEachItem",
    "jsCode": "// Enhanced error tracking\n\nclass ErrorTracker {\n  static categorizeError(error) {\n    const errorString = error.toString().toLowerCase();\n    \n    if (errorString.includes('timeout')) return 'TIMEOUT';\n    if (errorString.includes('connection')) return 'CONNECTION';\n    if (errorString.includes('authentication')) return 'AUTH';\n    if (errorString.includes('validation')) return 'VALIDATION';\n    if (errorString.includes('rate limit')) return 'RATE_LIMIT';\n    \n    return 'UNKNOWN';\n  }\n  \n  static shouldAlert(errorCategory, errorCount) {\n    const alertThresholds = {\n      'TIMEOUT': 3,\n      'CONNECTION': 2,\n      'AUTH': 1,\n      'VALIDATION': 5,\n      'RATE_LIMIT': 1,\n      'UNKNOWN': 3\n    };\n    \n    return errorCount >= (alertThresholds[errorCategory] || 3);\n  }\n}\n\nconst error = $input.item.json.error;\nconst errorCategory = ErrorTracker.categorizeError(error);\n\nreturn {\n  error_message: error.toString(),\n  error_category: errorCategory,\n  error_timestamp: new Date().toISOString(),\n  workflow_name: 'document_processing',\n  should_alert: ErrorTracker.shouldAlert(errorCategory, 1),\n  context: {\n    document_id: $input.item.json.document_id,\n    user_id: $input.item.json.user_id,\n    workflow_step: 'embedding_generation'\n  }\n};"
  },
  "name": "Track Error",
  "type": "n8n-nodes-base.code"
}
```

## Best Practices

### 1. Workflow Organization
- **Naming Conventions**: Use descriptive, consistent node names
- **Error Handling**: Always include error paths and recovery logic
- **Documentation**: Add notes to complex nodes explaining their purpose
- **Modularity**: Break complex workflows into smaller, reusable sub-workflows

### 2. Performance Optimization
- **Batch Processing**: Process multiple items together when possible
- **Caching**: Use Redis to cache expensive operations
- **Async Operations**: Use webhooks and queues for long-running tasks
- **Resource Management**: Monitor memory and CPU usage

### 3. Security Considerations
- **Credentials Management**: Use n8n's credential system, never hardcode secrets
- **Input Validation**: Always validate external inputs
- **Rate Limiting**: Implement rate limiting for external APIs
- **Audit Logging**: Track workflow executions and data access

## Troubleshooting

### Common Issues

**"Workflow execution failed"**
```javascript
// Debug node to inspect data flow
{
  "parameters": {
    "mode": "runOnceForEachItem",
    "jsCode": "// Debug data flow\nconsole.log('Current item:', $input.item.json);\nconsole.log('Previous nodes:', $input.all());\n\nreturn item.json;"
  },
  "name": "Debug Node",
  "type": "n8n-nodes-base.code"
}
```

**"HTTP request timeout"**
```javascript
// Add timeout handling
{
  "parameters": {
    "url": "http://your-api:8000/endpoint",
    "timeout": 30000,  // 30 seconds
    "options": {
      "redirect": {
        "followRedirect": true,
        "maxRedirect": 3
      },
      "response": {
        "response": {
          "fullResponse": true
        }
      }
    }
  },
  "name": "HTTP Request with Timeout",
  "type": "n8n-nodes-base.httpRequest"
}
```

**"Memory usage too high"**
```bash
# Monitor n8n container resources
docker stats rag-n8n

# Check workflow execution history
# In n8n UI: Executions → View detailed logs

# Optimize workflow by processing smaller batches
```

**"Database connection issues"**
```javascript
// Test database connectivity
{
  "parameters": {
    "mode": "runOnceForEachItem",
    "jsCode": "// Test MongoDB connection\ntry {\n  // This would be handled by the MongoDB node\n  return {\n    connection_test: 'success',\n    timestamp: new Date().toISOString()\n  };\n} catch (error) {\n  return {\n    connection_test: 'failed',\n    error: error.toString(),\n    timestamp: new Date().toISOString()\n  };\n}"
  },
  "name": "Test DB Connection",
  "type": "n8n-nodes-base.code"
}
```

## Learning Resources

- [n8n Documentation](https://docs.n8n.io/) - Official documentation
- [n8n Community](https://community.n8n.io/) - User community and support
- [Workflow Templates](https://n8n.io/workflows/) - Pre-built workflow examples
- [Custom Node Development](https://docs.n8n.io/integrations/creating-nodes/) - Building custom integrations