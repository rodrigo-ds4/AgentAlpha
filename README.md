# Minimal RAG Agent

**Made by Rodrigo de Sarasqueta**

Ultra-clean RAG agent skeleton with multi-LLM support and dual memory.

## Features

- Multi-LLM support (Ollama, OpenAI, DeepSeek)
- Dual memory system (buffer + summary)
- PDF/TXT document ingestion
- Semantic search with ChromaDB
- RESTful API endpoints
- Minimal CLI interface
- Docker deployment

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Add documents
python cli.py add-pdf document.pdf

# Start chat
python cli.py

# Or start API server
python api.py

# Evaluate performance (NEW!)
python evaluate_mock.py test_questions.json  # Basic test (5 questions)
python evaluate_mock.py test_questions_simple.json  # Simple test (8 questions)  
python evaluate_mock.py test_questions_comprehensive.json  # Comprehensive test (12 questions)
python evaluate_simple.py test_questions.json  # Simple evaluation with real LLM
python evaluate.py test_questions.json  # Full RAGAS evaluation
```

## Configuration

Edit `config.json`:

```json
{
  "llm": {
    "provider": "ollama",
    "ollama_url": "http://localhost:11434",
    "ollama_model": "llama3:latest"
  }
}
```

For API keys, use environment variables:
```bash
export OPENAI_API_KEY=your-key
export DEEPSEEK_API_KEY=your-key
```

## Docker

```bash
docker-compose up
```

## API Endpoints

- `POST /chat` - Chat with agent
- `POST /upload` - Upload document
- `GET /documents` - List documents
- `DELETE /documents/{id}` - Delete document
- `GET /` - Web interface

## Architecture

```
Agent -> LLM Client (Ollama/OpenAI/DeepSeek)
      -> Vector Store (ChromaDB)
      -> Dual Memory (Buffer + Summary)
      -> Tools (RAG + HTTP)
```

## Structure

```
├── core/
│   ├── agent.py         # Main agent
│   ├── llm_client.py    # Multi-LLM support
│   ├── vector_store.py  # ChromaDB RAG
│   └── tools/           # RAG and HTTP tools
├── api.py              # FastAPI server
├── cli.py              # Minimal CLI
├── config.py           # Configuration
└── config.json         # Settings
```

Clean foundation for building AI agents.