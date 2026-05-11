# ==========================================
# GLOBAL CONSTANTS (UPDATE THESE AS NEEDED)
# ==========================================
DATA_DIR = "./hotpot_test"       # Folder containing your raw text/source files
CHROMA_PATH = "./chroma_db"      # Persistent directory for primary vector store
CACHE_PATH = "./cache_db"        # Persistent directory for semantic memory cache
LLM_MODEL = "gemini-3.1-flash-lite"
LLM_TEMP = 0.3                   # For the rag_pipeline LLM Temperature, differ from indexer
EMBEDDING_MODEL = "gemini-embedding-2-preview"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

THRESHOLD_SEMANTIC_CACHE = 0.7
TOP_K_RETRIEVER = 10
MAX_SCORE_CONFIDENCE_THRESHOLD = 0.7
CHUNK_SIM_SCORE_THRESHOLD = 0.7
TOP_K_RERANKING = 5

ARCHITECTURE_DIAGRAM_URL = "https://res.cloudinary.com/dzhtnuyez/image/upload/q_auto/f_auto/v1778493919/Light_-_RAG_for_Data_Scientist_Flow_Diagram_k2rcu1.png" # You can replace this with your Cloudinary URL!
