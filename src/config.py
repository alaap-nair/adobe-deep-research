"""
config.py -- Centralized configuration for database connections and models.

Reads from .env with sensible defaults for lightweight local development.
Qdrant runs in local file mode by default (no server needed).
Neo4j requires a running instance but is optional (pipeline skips if unavailable).
"""

import os
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Neo4j (optional -- pipeline gracefully skips if unavailable)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Qdrant -- local file storage by default, no server required
# Set QDRANT_URL to use a remote server instead
QDRANT_PATH = os.getenv("QDRANT_PATH", os.path.join(ROOT, "qdrant_data"))
QDRANT_URL = os.getenv("QDRANT_URL", None)  # None = use local file mode
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# Qdrant collection names
ENTITY_COLLECTION = "entities"
EVIDENCE_COLLECTION = "evidence"
