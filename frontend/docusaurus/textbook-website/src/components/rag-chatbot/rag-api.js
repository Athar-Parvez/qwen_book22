// rag-api.js
// This file would handle all API interactions with the backend RAG system

const RAG_API_BASE_URL = typeof window !== 'undefined' 
  ? window.RAG_API_BASE_URL || '/api/rag'  // Allow override via window object or default to relative path
  : process.env.RAG_API_BASE_URL || 'http://localhost:8000/api/rag'; // Server-side fallback

export class RAGService {
  constructor(config = {}) {
    this.baseUrl = config.baseUrl || RAG_API_BASE_URL;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      ...config.headers,
    };
  }

  async query(text, options = {}) {
    try {
      const response = await fetch(`${this.baseUrl}/query`, {
        method: 'POST',
        headers: {
          ...this.defaultHeaders,
          ...options.headers,
        },
        body: JSON.stringify({
          query: text,
          top_k: options.topK || 3,
          threshold: options.threshold || 0.7,
          ...options.payload
        }),
      });

      if (!response.ok) {
        throw new Error(`RAG API request failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return {
        success: true,
        response: data.response,
        sources: data.sources || [],
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      console.error('RAG API error:', error);
      return {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString(),
      };
    }
  }

  async healthCheck() {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      return response.ok;
    } catch (error) {
      console.error('RAG API health check failed:', error);
      return false;
    }
  }

  // Additional methods for other RAG functionality could be added here:
  // - index management
  // - document upload
  // - conversation history
  // - etc.
}

// Create a default instance for convenience
export const defaultRAGService = new RAGService();

export default RAGService;