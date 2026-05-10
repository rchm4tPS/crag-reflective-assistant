# Reflective Mentoring Assistant (CRAG)

A (probably, still argued) accurate, domain-agnostic RAG (Retrieval-Augmented Generation) assistant. Built with a Corrective RAG (CRAG) architecture to eliminate hallucinations, fallback to Agentic Regex, and a robust "Veteran-Approved" caching system to ensure answers stay correct and trusted.

## Architecture Image
![CRAG Architecture Pipeline](https://res.cloudinary.com/dzhtnuyez/image/upload/architecture_h7tmv9.png)
## 🚀 Features
*   **Two-Tier Persona System:** 
    *   **Hunter Mode:** Ask questions, view AI reasoning traces, and nominate great answers for documentation.
    *   **Engineer Mode:** A dashboard for domain experts to review, patch, and "Official-ize" nominated answers.
*   **Semantic Memory Cache:** Perfect answers are cached. Future queries with the exact same intent instantly return the Veteran-Approved answer without recalculating.
*   **CRAG Pipeline:** Retrieves context, votes on confidence, reranks via Cross-Encoder, and grades relevance.
*   **Multi-Fallback:** If vector retrieval fails, an AI agent writes a Regex pattern to search the raw filesystem. If that fails, it falls back to a DuckDuckGo Web Search.
*   **Multi-turn:** As long as the page is not hard-reloaded, the chat can go on with multiturn capability, though the LLM functionality will degrade as longer query chat history is condensed in every turn for better memorization of LLM

## Limitation
*   Maybe it will runs a bit longer than expected due to the usage of free tier Google LLM API, so please be patient while waiting the app to be fully initialized.
*   **Pre-defined dataset taken from HotpotQA:** The available dataset in folder `/hotpot_test` serve for testing purposes of this RAG model, which consist of 10 small documents. You can ask the question from this folder's document immediately after you index whole documents first. Due to small dataset, the accuracy, similarity score, chunking method, embedding technique and parameter is really driving the RAG model.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rchm4tPS/crag-reflective-assistant.git
   cd crag-reflective-assistant
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Settings:**
   Open `config.py` and add your **Google Gemini API Key**. You can also tweak model temperatures, thresholds, and file paths here.

4. **Add Your Knowledge Base:**
   Place your raw text documents (Markdown, TXT, etc.) into the `./hotpot_test` folder (or whatever folder you defined in `config.py`).

5. **Index the Files:**
   Run the indexer to parse your files into the ChromaDB vector database.
   ```bash
   python indexer.py
   ```

6. **Start the App:**
   ```bash
   streamlit run app.py
   ```

## 📁 Repository Structure
*   `app.py` - The main Streamlit UI orchestrator.
*   `rag_pipeline.py` - Core AI logic, prompts, and pipeline functions (dependency-injected).
*   `config.py` - Global constants and hyperparameters.
*   `indexer.py` - Script to chunk and embed raw documents into ChromaDB.
*   `pages/1_Help_Guide.py` - Built-in documentation page for users.

## 🛠️ Tech Stack
*   **UI:** Streamlit
*   **LLM & Embeddings:** Google Gemini (3.1 Flash Lite & Embeddings 2 Preview)
*   **Vector Database:** ChromaDB
*   **Reranker:** Sentence-Transformers (Cross-Encoder)
*   **Orchestration:** LangChain Core
