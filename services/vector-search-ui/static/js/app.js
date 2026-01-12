// MongoDB Vector Search UI - Frontend JavaScript

function vectorSearchApp() {
    return {
        selectedCollection: '',
        collections: [],
        searchResults: [],
        isSearching: false,
        showCreateIndex: false,
        showInsertDoc: false,
        
        searchQuery: {
            queryVector: '',
            k: 10,
            filters: ''
        },
        
        indexForm: {
            collection: '',
            dimension: 1536,
            similarity: 'cosine'
        },
        
        docForm: {
            collection: '',
            content: '',
            metadata: '{}',
            embedding: ''
        },

        init() {
            this.loadCollections();
        },

        async loadCollections() {
            try {
                const response = await fetch('/api/collections');
                const data = await response.json();
                this.collections = data.collections;
            } catch (error) {
                this.showNotification('Error loading collections: ' + error.message, 'error');
            }
        },

        selectCollection(collectionName) {
            this.selectedCollection = collectionName;
            this.searchResults = [];
            this.searchQuery.queryVector = '';
        },

        async performSearch() {
            if (!this.searchQuery.queryVector.trim()) {
                this.showNotification('Please enter a query vector', 'error');
                return;
            }

            this.isSearching = true;
            this.searchResults = [];

            try {
                // Parse the query vector
                const queryVector = JSON.parse(this.searchQuery.queryVector);
                
                // Parse filters if provided
                const filters = this.searchQuery.filters ? JSON.parse(this.searchQuery.filters) : null;

                const searchRequest = {
                    collection: this.selectedCollection,
                    query_vector: queryVector,
                    k: this.searchQuery.k,
                    filters: filters
                };

                const response = await fetch('/api/vector-search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(searchRequest)
                });

                const result = await response.json();

                if (result.success) {
                    this.searchResults = result.results;
                    this.showNotification(`Found ${result.count} results`, 'success');
                } else {
                    this.showNotification('Search failed: ' + result.error, 'error');
                }
            } catch (error) {
                this.showNotification('Search error: ' + error.message, 'error');
            } finally {
                this.isSearching = false;
            }
        },

        async createIndex() {
            try {
                const formData = new FormData();
                formData.append('collection_name', this.indexForm.collection);
                formData.append('dimension', this.indexForm.dimension);
                formData.append('similarity', this.indexForm.similarity);

                const response = await fetch('/api/create-index', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.success) {
                    this.showNotification('Vector index created successfully!', 'success');
                    this.showCreateIndex = false;
                    this.loadCollections();
                } else {
                    this.showNotification('Failed to create index: ' + result.error, 'error');
                }
            } catch (error) {
                this.showNotification('Error creating index: ' + error.message, 'error');
            }
        },

        async insertDocument() {
            try {
                const formData = new FormData();
                formData.append('collection_name', this.docForm.collection);
                formData.append('content', this.docForm.content);
                formData.append('metadata', this.docForm.metadata);
                
                if (this.docForm.embedding) {
                    formData.append('embedding', this.docForm.embedding);
                }

                const response = await fetch('/api/insert-document', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.success) {
                    this.showNotification('Document inserted successfully!', 'success');
                    this.showInsertDoc = false;
                    this.docForm = { collection: '', content: '', metadata: '{}', embedding: '' };
                    this.loadCollections();
                } else {
                    this.showNotification('Failed to insert document: ' + result.error, 'error');
                }
            } catch (error) {
                this.showNotification('Error inserting document: ' + error.message, 'error');
            }
        },

        generateSampleVector() {
            // Generate a sample vector for testing
            const dimensions = this.indexForm.dimension || 1536;
            const vector = Array.from({ length: dimensions }, () => Math.random() - 0.5);
            this.searchQuery.queryVector = JSON.stringify(vector, null, 2);
        },

        formatJSON(jsonString) {
            try {
                return JSON.stringify(JSON.parse(jsonString), null, 2);
            } catch {
                return jsonString;
            }
        },

        showNotification(message, type = 'success') {
            // Create notification element
            const notification = document.createElement('div');
            notification.className = `notification ${type}`;
            notification.textContent = message;
            
            document.body.appendChild(notification);
            
            // Trigger animation
            setTimeout(() => notification.classList.add('show'), 100);
            
            // Auto remove
            setTimeout(() => {
                notification.classList.remove('show');
                setTimeout(() => document.body.removeChild(notification), 300);
            }, 3000);
        }
    }
}

// Utility functions for the existing Tailwind template
function toggleDarkMode() {
    document.documentElement.classList.toggle('dark');
}

function showCreateCollection() {
    document.getElementById('createCollectionModal').classList.remove('hidden');
}

function hideCreateCollection() {
    document.getElementById('createCollectionModal').classList.add('hidden');
}

async function createCollection() {
    const name = document.getElementById('collectionName').value;
    if (!name) return;
    
    try {
        const response = await fetch('/api/collections', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name: name })
        });
        
        if (response.ok) {
            hideCreateCollection();
            location.reload();
        }
    } catch (error) {
        alert('Error creating collection: ' + error.message);
    }
}

async function deleteCollection(name) {
    if (!confirm(`Delete collection ${name}?`)) return;
    
    try {
        const response = await fetch(`/api/collection/${name}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            location.reload();
        }
    } catch (error) {
        alert('Error deleting collection: ' + error.message);
    }
}

function showCreateIndex() {
    alert('Create Vector Index - Use the main interface for full functionality!');
}

function showImportData() {
    alert('Import Data - Coming Soon! Use the Insert Document feature for now.');
}