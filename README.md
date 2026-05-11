# Reflective Mentoring Assistant (CRAG)

A (probably, still argued) accurate, domain-agnostic RAG (Retrieval-Augmented Generation) assistant. Built with a Corrective RAG (CRAG) architecture to eliminate hallucinations, fallback to Agentic Regex, and a robust "Veteran-Approved" caching system to ensure answers stay correct and trusted.

> **⚠️ WARNING:** For the live deployed Streamlit app here: [My Deployed Streamlit](https://crag-reflective-assistant.streamlit.app/), you MUST provide your own Gemini API Key. [Read more about BYOK Security below](#byok-security).

## Architecture Image
![V2 of proposed CRAG Architecture Pipeline](https://res.cloudinary.com/dzhtnuyez/image/upload/q_auto,f_auto/Light_-_RAG_for_Data_Scientist_Flow_Diagram_k2rcu1.png)

## 🚀 Features
*   **Two-Tier Persona System:** 
    *   **Hunter Mode:** Ask questions, view AI reasoning traces, and nominate great answers for documentation.
    *   **Engineer Mode:** A dashboard for domain experts to review, patch, and "Official-ize" nominated answers.
*   **Semantic Memory Cache:** Approved nominated answers are cached. Future queries with the exact same intent instantly return the Veteran-Approved answer without recalculating or doing whole RAG pipeline again, thus cut the cost for recurring domain-related query.
*   **CRAG Pipeline:** Retrieves context, votes on confidence (must be "voted" by system to be at least 2 out of 3), reranks via Cross-Encoder, and grades relevance. All of these features has made the RAG more resilient from doing unpredicted hallucinations.
*   **Multi-Fallback:** If vector retrieval fails, an AI agent writes a Regex pattern to search the raw filesystem. If that still fails, it eventually falls back to a DuckDuckGo Web Search.
*   **Multi-turn:** As long as the page is not hard-reloaded, the chat can go on with multiturn capability, though the LLM functionality will degrade as longer query chat history is condensed in every turn for better memorization of LLM.
*   <a id="byok-security"></a>**Bring-Your-Own-Key (BYOK) Security:** Fully multi-tenant safe. Users are prompted with a secure UI dialog to enter their own API key, which is strictly isolated to their browser session.

## Limitation
*   **Time constraints:** Maybe it will runs a bit longer than expected due to the usage of free tier Google LLM API, so please be patient while waiting the app to be fully initialized.
*   **No consideration for ETL:** Data for this RAG pipeline must be assumed to already be cleaned previously from any formatting or layout issue. The best file to be indexed is in .txt, .md, or something similar.
*   **No history retained for different session:** Unlike typical chatbot that will store past conversations, in this MVP version of CRAG, hunter will not retain their past conversations and is considered to use the chatbot interface as long as there is no hard reload on the page. Though the cached veteran-approved answer is persistently saved in cache_db even when user do hard reloading (thus making it new session), so when hunter asking same intent question that is found in cache_db, it will get the result from cache, not from the whole pipeline RAG.
*   **Pre-defined dataset taken from HotpotQA:** The available dataset in folder `/hotpot_test` serve for testing purposes of this RAG model, which consist of 10 small documents. You can ask the question from this folder's document immediately because I have prefilled chroma_db with a very small amount of data to start immediately. Alternatively, if you do this locally, you can customize the chroma_db after you index whole documents first. Due to small dataset in default chroma_db I gave to you, the accuracy, similarity score, chunking method, embedding technique and parameter is really driving the RAG model, and may vary depending on document content and language.

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

3. **Configure Settings (Optional):**
   Open `config.py` to tweak model temperatures, similarity thresholds, and file paths. *(Note: API Keys are handled securely in the UI, do not put them in `config.py`!)*

4. **Add Your Knowledge Base:**
   Place your raw text documents (Markdown, TXT, etc.) into the `./hotpot_test` folder (or whatever folder you defined in `config.py`).

5. **Index the Files (Local Setup Only):**
   To parse your files into the ChromaDB vector database, you must first set your API key in your terminal, then run the indexer:
   ```bash
   # Windows
   set GOOGLE_API_KEY=AIzaSy...
   # Mac/Linux
   export GOOGLE_API_KEY=AIzaSy...
   
   python indexer.py
   ```

6. **Start the App:**
   ```bash
   streamlit run app.py
   ```
   *Upon launching, the app will prompt you with a secure dialog box to enter your Gemini API Key.*

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

Created by Rachmat Purwa Saputra<br>
in 2026.