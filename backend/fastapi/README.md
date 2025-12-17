# Backend – FastAPI RAG Service

This service provides:
- RAG-based question answering
- Vector search (Qdrant)
- Metadata storage (Neon Postgres)

## Endpoints (planned):
- `/chat` - RAG-based question answering
- `/ingest` - Content ingestion for RAG system
- `/health` - Service health check

## Status: In progress

## Getting Started

To run the backend service locally:

1. Set up your virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Run the development server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

## Project Structure

```
backend/fastapi/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/          # API route definitions
│   ├── models/              # Pydantic models
│   ├── schemas/             # Request/response schemas
│   ├── database/            # Database connection and session management
│   ├── services/            # Business logic
│   └── utils/               # Utility functions
├── requirements.txt         # Python dependencies
├── .env.example            # Example environment variables
└── README.md               # This file
```

## Environment Variables

The service requires the following environment variables:

```env
QDRANT_URL=your_qdrant_instance_url
QDRANT_API_KEY=your_qdrant_api_key  # If using cloud instance
NEON_DATABASE_URL=your_neon_postgres_connection_string
OPENAI_API_KEY=your_openai_api_key  # For embeddings
DEBUG=true  # Set to false in production
```

## Dependencies

- FastAPI - Web framework
- Qdrant - Vector database
- Neon Postgres - Metadata storage
- OpenAI SDK - For embeddings and LLM interactions
- SQLAlchemy - Database ORM
- Pydantic - Data validation

## Planned Endpoints

- `POST /chat` - Submit questions and get RAG-enhanced answers
- `POST /ingest` - Ingest textbook content into the RAG system
- `GET /health` - Service health and readiness check

## Contributing

1. Create your feature branch
2. Add tests for new functionality
3. Follow the existing code structure and patterns
4. Update this README if adding new endpoints or changing functionality