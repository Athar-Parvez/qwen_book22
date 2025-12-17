# Textbook API Contract

## /api/textbook/chapters
### GET - Get all textbook chapters
**Description**: Retrieve a list of all textbook chapters

**Request**:
- Method: GET
- Path: /api/textbook/chapters
- Headers: None required

**Response**:
- Status: 200 OK
- Content-Type: application/json
- Body:
```json
{
  "chapters": [
    {
      "id": "string",
      "title": "string",
      "order": "integer",
      "learning_objectives": ["string"],
      "prerequisites": ["string"]
    }
  ]
}
```

### GET - Get specific textbook chapter
**Description**: Retrieve a specific textbook chapter by ID

**Request**:
- Method: GET
- Path: /api/textbook/chapters/{chapter_id}
- Headers: None required

**Response**:
- Status: 200 OK
- Content-Type: application/json
- Body:
```json
{
  "id": "string",
  "title": "string",
  "content": "string",
  "order": "integer",
  "learning_objectives": ["string"],
  "prerequisites": ["string"],
  "sections": [
    {
      "id": "string",
      "title": "string",
      "content": "string",
      "order": "integer"
    }
  ]
}
```

## /api/textbook/search
### POST - Search textbook content
**Description**: Search for specific content across the textbook

**Request**:
- Method: POST
- Path: /api/textbook/search
- Headers: 
  - Content-Type: application/json
- Body:
```json
{
  "query": "string"
}
```

**Response**:
- Status: 200 OK
- Content-Type: application/json
- Body:
```json
{
  "results": [
    {
      "id": "string",
      "title": "string",
      "content_preview": "string",
      "content_type": "string",
      "relevance_score": "number"
    }
  ]
}
```

## /api/chatbot/query
### POST - Ask question to textbook chatbot
**Description**: Submit a question to the RAG-based textbook chatbot

**Request**:
- Method: POST
- Path: /api/chatbot/query
- Headers: 
  - Content-Type: application/json
- Body:
```json
{
  "question": "string",
  "selected_text": "string | null"
}
```

**Response**:
- Status: 200 OK
- Content-Type: application/json
- Body:
```json
{
  "response": "string",
  "confidence_score": "number",
  "sources": [
    {
      "id": "string",
      "title": "string",
      "content_type": "string"
    }
  ]
}
```

## /api/content/metadata
### GET - Get content metadata for a specific chapter
**Description**: Retrieve metadata for textbook content for internal use

**Request**:
- Method: GET
- Path: /api/content/metadata/{chapter_id}
- Headers: None required (internal API)

**Response**:
- Status: 200 OK
- Content-Type: application/json
- Body:
```json
{
  "chapter_id": "string",
  "embedding_count": "integer",
  "last_updated": "string"
}
```