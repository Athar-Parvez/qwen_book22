# Quickstart: Physical AI & Humanoid Robotics Textbook

## Overview
This quickstart guide provides instructions for setting up, running, and contributing to the Physical AI & Humanoid Robotics textbook project.

## Prerequisites
- Python 3.11+
- Node.js 18+ (for Docusaurus)
- Git
- Access to Qdrant Cloud (vector database)
- Access to Neon Serverless Postgres (metadata storage)
- Access to OpenAI API (for embeddings and chat features)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd physical-ai-textbook
```

### 2. Backend Setup (FastAPI)
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys and database URLs

# Run the backend server
python -m uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Setup (Docusaurus)
```bash
# Navigate to frontend directory
cd frontend/docusaurus

# Install dependencies
npm install

# Set environment variables (if needed)
cp .env.example .env
# Edit .env with your backend API URL

# Run the development server
npm start
```

### 4. Content Ingestion
```bash
# To ingest textbook content into the RAG system
cd backend
python scripts/ingest_docs.py --source-path ../frontend/docusaurus/docs
```

## Project Structure
```
physical-ai-textbook/
├── frontend/                 # Docusaurus documentation site
│   └── docusaurus/
│       ├── docs/            # Textbook content in Markdown
│       ├── src/             # Custom React components
│       ├── static/          # Static assets
│       └── docusaurus.config.js
├── backend/                  # FastAPI backend services
│   └── app/
│       ├── api/             # API route definitions
│       ├── rag/             # RAG implementation
│       ├── embeddings/      # Embedding processing
│       ├── database/        # Database models and connections
│       └── main.py          # Application entry point
├── specs/                    # Project specifications
└── scripts/                  # Utility scripts
```

## Running Tests
```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd frontend/docusaurus
npm test
```

## Adding New Textbook Content
1. Create a new Markdown file in `frontend/docusaurus/docs/`
2. Add the content using Docusaurus syntax
3. Update `frontend/docusaurus/sidebars.js` to include the new content in the navigation
4. Run the ingestion script to update the RAG system:
   ```bash
   python scripts/ingest_docs.py --source-path ../frontend/docusaurus/docs
   ```

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Deployment
### Frontend (GitHub Pages)
```bash
cd frontend/docusaurus
npm run build
# The build output is automatically deployed to GitHub Pages via GitHub Actions
```

### Backend
The backend should be deployed to a cloud platform that supports Python applications. Environment variables must be configured with appropriate API keys and database URLs.

## Troubleshooting
- If the RAG chatbot returns no results, ensure the content ingestion script has run successfully
- If pages don't load, check that the backend server is running and accessible
- For database connection issues, verify your Neon Postgres and Qdrant Cloud credentials