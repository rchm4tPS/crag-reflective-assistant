import os
import json
import time
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Import config to initialize the GOOGLE_API_KEY in os.environ
import config

# ==========================================
# GLOBAL CONSTANTS (MATCH WITH APP.PY)
# ==========================================
DATA_DIR = "./hotpot_test"       # Folder containing your domain knowledge documentation
CHROMA_PATH = "./chroma_db"      # Persistent directory for primary vector store
LLM_MODEL = "gemini-3.1-flash-lite"
LLM_TEMP = 1.0
EMBEDDING_MODEL = "gemini-embedding-2-preview"

# 1. API Key is automatically set by importing config above.
# 2. Initialize Google Gemini and Embedding Model
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL, 
    temperature=LLM_TEMP
)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

def generate_file_summary(file_content: str, file_name: str) -> dict:
    """Uses Gemini to generate a descriptive summary and metadata for a file."""
    
    prompt = f"""
    You are an expert SUBJECT MATTER EXPERT COMMENTARY:.
    Analyze the following file named '{file_name}'.
    
    Provide a JSON response with the following keys:
    - "intent": What is the main objective of this file? (1 sentence)
    - "category": What domain or department does this belong to? (e.g., "HR Policy", "Legal Contract", "Financial Report", "IT")
    - "topic": The core subject matter.
    
    File Content:
    {file_content[:3000]} # Limit to 3000 chars to save tokens on huge files
    """
    
    try:
        response = llm.invoke(prompt)
        
        # --- NEW SAFETY CHECK ---
        # If response.content is a list, join it into a single string
        if isinstance(response.content, list):
            raw_text = "".join([str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in response.content])
        else:
            raw_text = response.content
        # ------------------------

        # Clean up markdown formatting if Gemini adds ```json ... ```
        cleaned_response = raw_text.strip().replace("```json", "").replace("```", "")
        
        metadata = json.loads(cleaned_response)
        return metadata
    except Exception as e:
        print(f"Failed to summarize {file_name}: {e}")
        return {"intent": "Unknown", "feature_name": "Unknown", "project_name": "Unknown"}
    
def smart_chunk_document(file_path: str) -> list[Document]:
    """Reads a file, generates metadata via Gemini, and chunks it smartly."""
    
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    
    # 1. Generate Metadata using Gemini (Your architecture's first step)
    print(f"Generating summary for {path.name}...")
    ai_metadata = generate_file_summary(content, path.name)
    
    # Base metadata
    base_metadata = {
        "source": str(path),
        "filename": path.name,
        "file_type": path.suffix,
        "intent": ai_metadata.get("intent", ""),
        "feature_name": ai_metadata.get("feature_name", ""),
        "project_name": ai_metadata.get("project_name", "")
    }

    # 2. Choose the right chunker based on file type
    if path.suffix == ".py":
        # Code Chunking: Respects Python indentation and functions
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=800, chunk_overlap=100
        )
    elif path.suffix == ".md":
        # Markdown Chunking: Respects Headers (##)
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN, chunk_size=1000, chunk_overlap=150
        )
    elif path.suffix in [".yaml", ".yml"]:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JS, chunk_size=800, chunk_overlap=100
        ) # JS chunker works well for YAML/JSON
    else:
        # Standard Fallback Chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )

    # 3. Create Chunks
    raw_chunks = splitter.create_documents([content], metadatas=[base_metadata])
    
    # 4. Context Augmentation (Crucial for Vector DB accuracy)
    # We prepend the AI summary to the start of EVERY chunk's page_content.
    # This guarantees the Embedding model always knows the "context" of a small code snippet.
    for chunk in raw_chunks:
        chunk.page_content = f"FILE INTENT: {base_metadata['intent']}\n\nCONTENT:\n{chunk.page_content}"
        
    return raw_chunks

def run_indexing_phase(source_directory: str):
    """The main pipeline to read files, chunk, embed, and store in Vector DB."""
    
    all_chunks = []
    
    # 1. Loop through all files in the directory
    for root, _, files in os.walk(source_directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Process each file using our Smart Chunker
            chunks = smart_chunk_document(file_path)
            all_chunks.extend(chunks)

    # 2. Stricter Cleaning
    valid_chunks = [c for c in all_chunks if c.page_content and len(c.page_content.strip()) > 10]

    if not valid_chunks:
        print("No valid content found.")
        return
    
    # 3. Initialize Chroma DB EMPTY first
    print(f"Initializing Vector DB at {CHROMA_PATH}...")
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # 4. Add documents in BATCHES
    print(f"Starting batch indexing ({len(valid_chunks)} chunks total)...")

    for i, chunk in enumerate(valid_chunks):
        try:
            # Final metadata cleanup
            sanitized_metadata = {}
            for k, v in chunk.metadata.items():
                sanitized_metadata[k] = str(v) if isinstance(v, (list, dict)) else v
            chunk.metadata = sanitized_metadata

            # VERIFY EMBEDDING WORKS FIRST
            # This is the "Data Science" way to debug: test the model output directly
            test_emb = embeddings.embed_query(chunk.page_content)
            if not test_emb:
                print(f"⚠️ Google returned an empty embedding for chunk {i}. Skipping.")
                continue

            vector_store.add_documents([chunk])
            if i % 10 == 0:
                print(f"Progress: {i}/{len(valid_chunks)} chunks indexed...")
            
            time.sleep(0.5) # Gentle pause
            
        except Exception as e:
            print(f"❌ Failed at chunk {i}: {str(e)[:100]}")
            # Log a snippet of the failing text so you can see if it's "weird" content
            print(f"Content snippet: {chunk.page_content[:50]}...")
            continue

    print(f"\n✅ Indexing Complete! Check the '{CHROMA_PATH}' folder.")

# --- Run the Script ---
if __name__ == "__main__":
    # Point this to a folder containing your domain knowledge documentation
    
    # Ensure directory exists for testing
    os.makedirs(DATA_DIR, exist_ok=True)
    
    run_indexing_phase(DATA_DIR)