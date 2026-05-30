CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"

CHROMA_DIR = "./chroma_db"
CHROMA_COLLECTION = "local_rag"
DOCS_DIR = "./docs"

# Langflow
LANGFLOW_BASE_URL = "http://localhost:7860"
LANGFLOW_REQUEST_TIMEOUT = 120

# Agentic RAG loop control
# ChromaDB returns L2 distance (lower = better). Threshold below which a retrieval is considered good.
SIMILARITY_THRESHOLD = 0.8
MAX_AGENTIC_LOOPS = 3
