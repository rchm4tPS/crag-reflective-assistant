import streamlit as st
import os
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from config import *

# --- 2. PRE-RETRIEVAL FUNCTIONS ---

def check_semantic_cache(query, cache_db, llm, threshold=THRESHOLD_SEMANTIC_CACHE):
    """
    Step 1 in your architecture: Knowledge Memory Layer Retrieval.
    Checks if a very similar question has been answered before.
    """
    results = cache_db.similarity_search_with_relevance_scores(query, k=1)
    if results and results[0][1] > threshold:
        doc = results[0][0]
        cached_query = doc.page_content
        
        verify_prompt = f"""
        Role: Senior Quality Assurance Auditor.
        Task: Compare two user questions for semantic identity.
        
        Question A (Stored): "{cached_query}"
        Question B (Current): "{query}"
        
        Rules:
        - If the two questions have the EXACT SAME technical intent and ask for the same information, reply: YES
        - If there is any difference in intent (e.g., asking for 'Who' vs 'Version', or 'Explain' vs 'List'), reply: NO
        - Sharing keywords (like 'Express' or 'Project') is NOT enough for a YES. Look and judge at the context how each word is used.
        
        RESPONSE (YES/NO ONLY):
        """
        response = llm.invoke(verify_prompt).content
        
        if isinstance(response, list):
            verification = "".join([str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in response]).strip().upper()
        else:
            verification = str(response).strip().upper()
        
        if "YES" in verification:
            answer = doc.metadata.get("answer")
            status = doc.metadata.get("status", "unverified") 
            return answer, status
            
    return None, None

def expand_query(query, llm):
    """
    Step 2 in your architecture: Query Expansion.
    Turns 1 query into 3 sub-queries to improve retrieval.
    """
    prompt = ChatPromptTemplate.from_template("""
    Role: Expert Domain Researcher.
    Task: Deconstruct a user query into 3 distinct, high-signal sub-queries to search a knowledge base.
    
    ORIGINAL QUERY: {query}
    
    SYSTEMATIC SEARCH STRATEGY:
    1. Core Concept/Definition: What is the fundamental policy, theory, or rule being asked about?
    2. Process & Relationships: How does this interact with other standard operating procedures or regulations?
    3. Application/Troubleshooting: How is this applied in real-world scenarios or exceptions?
    
    RULES:
    - Focus on unique domain-specific keywords and exact terminology.
    - Keep queries concise but distinct.
    - Return exactly 3 questions, one per line. No numbering.
    """)
    
    response = llm.invoke(prompt.format(query=query))

    if isinstance(response.content, list):
        raw_text = "".join([str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in response.content])
    else:
        raw_text = response.content

    sub_queries = [q.strip() for q in raw_text.split("\n") if q.strip()]
    return sub_queries[:3] 

def condense_history(chat_history, latest_query, llm):
    """Turns a multi-turn chat into a single standalone RAG query, detects off-topic chatter, or self-queries."""
    if not chat_history:
        prompt = f"""
        Analyze the following user query: '{latest_query}'
        1. If it is a greeting, small talk, or off-topic, return: OFF_TOPIC
        2. If it is asking about domain knowledge files, structure, manifest, or your own capabilities, return: SELF_QUERY
        3. Otherwise, return the query exactly as it is.
        Return ONLY the classification or the original query.
        """
        response = llm.invoke(prompt)
        res = "".join([str(p.get("text", p)) for p in response.content]) if isinstance(response.content, list) else str(response.content)
        res = res.strip().upper()
        if "OFF_TOPIC" in res: return "OFF_TOPIC"
        if "SELF_QUERY" in res: return "SELF_QUERY"
        return latest_query
    
    context = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-3:]])
    prompt = f"""
    Role: Technical Intent Classifier & Context Synthesizer.
    Task: Analyze the user's latest query within the context of the chat history.
    
    HIERARCHY OF INTENTS:
    1. OFF_TOPIC: Greeting, small talk, unrelated banter (e.g., food, sports, general life). 
    2. SELF_QUERY: Asking about your indexed files, knowledge/document manifest, or system capabilities.
    3. DOMAIN_RAG: ALL subject-matter questions (policies, procedures, data, domain knowledge), EVEN if they seem unrelated to this specific topic/category.
    
    RULES:
    - If intent is OFF_TOPIC, return exactly: OFF_TOPIC
    - If intent is SELF_QUERY, return exactly: SELF_QUERY
    - Otherwise, rephrase the 'Follow-up' into a STANDALONE technical search query. 
    - CRITICAL: Only inherit specific terminology or document names from chat history if they are directly relevant to the NEW follow-up question.
    - ANOTHER CRITICAL: Do not force context if the user is changing the subject to a new terminology or context or domain knowledge.
    
    CHAT HISTORY:
    {context}
    
    FOLLOW-UP: {latest_query}
    
    INTENT OR STANDALONE QUERY:"""

    response = llm.invoke(prompt)
    res_content = "".join([str(p.get("text", p)) for p in response.content]) if isinstance(response.content, list) else str(response.content)
    res_content = res_content.strip()
    
    if "OFF_TOPIC" in res_content.upper(): return "OFF_TOPIC"
    if "SELF_QUERY" in res_content.upper(): return "SELF_QUERY"
    return res_content

# --- 3. MAIN RETRIEVAL PHASE (VOTING & FILTERING) ---

def retrieve_and_filter(sub_queries, vector_db, status_data=None):
    """
    Step 3: Vector Similarity Search -> Dynamic Filtering -> Deduplication
    Matches the 'Check MAX similarity score' diamond in your diagram.
    """
    final_filtered_chunks =[]
    votes = 0
    
    if status_data is not None:
        st.write("**Generated Sub-queries:**")
        status_data["steps"].append("**Generated Sub-queries:**")
        for q in sub_queries:
            st.write(f"- {q}")
            status_data["steps"].append(f"- {q}")
        if "traces" not in status_data:
            status_data["traces"] = []
    
    for q in sub_queries:
        raw_results = vector_db.similarity_search_with_relevance_scores(q, k=TOP_K_RETRIEVER)
        
        if not raw_results:
            continue
            
        max_score = raw_results[0][1] if raw_results else 0
        
        if status_data is not None:
            st.write(f"Class: `{q}` -> Max Score: `{max_score:.2f}`")
            status_data["steps"].append(f"Class: `{q}` -> Max Score: `{max_score:.2f}`")
            
            trace = {
                "query": q,
                "results": [{"filename": doc.metadata.get('filename'), "score": score} for doc, score in raw_results]
            }
            status_data["traces"].append(trace)
            
            with st.expander(f"📄 Raw Retrieval Trace ({q})", expanded=False):
                for i, res in enumerate(trace["results"]):
                    st.write(f"{i+1}. `{res['filename']}` (Score: `{res['score']:.4f}`)")
        
        if max_score >= MAX_SCORE_CONFIDENCE_THRESHOLD:
            votes += 1 
            for doc, score in raw_results:
                if score >= CHUNK_SIM_SCORE_THRESHOLD:
                    doc.metadata["retrieval_score"] = score 
                    final_filtered_chunks.append(doc)

    unique_chunks = {}
    for doc in final_filtered_chunks:
        if doc.page_content not in unique_chunks:
            unique_chunks[doc.page_content] = doc
            
    unique_list = list(unique_chunks.values())
    if status_data is not None:
        st.write(f"✨ **Deduplication Complete:** {len(final_filtered_chunks)} -> {len(unique_list)} unique chunks.")
        status_data["steps"].append(f"✨ **Deduplication Complete:** {len(final_filtered_chunks)} -> {len(unique_list)} unique chunks.")
        
    return unique_list, (votes >= 2), votes

# --- 4. POST-RETRIEVAL PHASE (RERANKING) ---

def rerank_chunks(original_query, deduplicated_chunks, reranker, top_k=TOP_K_RERANKING, status_steps=None):
    """
    Reranks the filtered chunks against the ORIGINAL raw query.
    """
    if not deduplicated_chunks:
        return []

    pairs = [[original_query, doc.page_content] for doc in deduplicated_chunks]
    scores = reranker.predict(pairs)
    
    for i, score in enumerate(scores):
        deduplicated_chunks[i].metadata["rerank_score"] = float(score)
        
    reranked_docs = sorted(deduplicated_chunks, key=lambda x: x.metadata["rerank_score"], reverse=True)
    final_docs = reranked_docs[:top_k]
    
    if status_steps is not None:
        st.write("**Final Reranked Contexts:**")
        status_steps.append("**Final Reranked Contexts:**")
        for i, doc in enumerate(final_docs):
            source = doc.metadata.get('filename', 'Unknown')
            score = doc.metadata.get('rerank_score', 0)
            st.write(f"{i+1}. `{source}` (Score: `{score:.2f}`)")
            status_steps.append(f"{i+1}. `{source}` (Score: `{score:.2f}`)")
            
    return final_docs

def generate_and_cache(query, context_chunks, llm, cache_status="auto-verified"):
    """
    Combines chunks into a prompt, streams the LLM response.
    """
    context_text = "\n\n".join([f"--- File: {doc.metadata.get('filename', 'Unknown')} ---\n{doc.page_content}" for doc in context_chunks])
    
    prompt = ChatPromptTemplate.from_template("""
    Role: Senior Subject Matter Expert & Knowledge Architect.
    Task: Synthesize a professional, highly accurate answer using the provided context.
    
    CONTEXT DATA:
    {context}
    
    TARGET QUESTION: 
    {query}
    
    DOCUMENTATION STANDARDS:
    1. TONE: Professional, objective, and technical.
    2. STRUCTURE: Use Markdown. Use H3 headers for sections, bold text for emphasis, and backticks for variables/file names.
    3. EXAMPLES/EXCERPTS: Provide exact quotes, formulas, or structured data ONLY if they exist in the context. 
    4. ACCURACY: Do not hallucinate. If the context is missing info, explicitly state "The current indexed documents do not provide details on [X]".
    5. DEPTH: Do not just list facts; explain the 'why' and the relationship between files if possible.
    
    SYNTHESIZED TECHNICAL DOCUMENTATION:
    """)
    
    chain = prompt | llm
    
    def stream_response():
        for chunk in chain.stream({"context": context_text, "query": query}):
            if isinstance(chunk.content, list):
                raw_text = "".join([str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in chunk.content])
            else:
                raw_text = str(chunk.content)
            tokens = re.split(r'(\s+)', raw_text)
            for token in tokens:
                if token:
                    yield token
            
    st.markdown("### 🤖 AI Response:")
    final_answer = st.write_stream(stream_response())
    if not isinstance(final_answer, str):
        final_answer = str(final_answer)

    return final_answer

def agentic_regex_search(query, llm, search_dir=DATA_DIR, status_steps=None):
    """
    Agentic Fallback: Asks the LLM to generate a regex, then searches local files.
    """
    prompt = f"""
    Role: Senior Systems Engineer.
    Task: Plan a local file search strategy using Regex.
    
    USER QUERY: '{query}'
    
    PLANNING RULES:
    1. SCOPE CHECK: If the query is non-technical (e.g., general life advice, non-domain topics), return: SKIP
    2. KEYWORD EXTRACTION: Identify 3-5 high-signal technical or domain terms, abbreviations, and likely domain specififc naming patterns.
    3. REGEX SYNTHESIS: Generate a single, safe Python regex pattern using the '|' operator.
    
    OUTPUT: Return ONLY the raw regex string. No formatting, no backticks.
    Example: employee|Agreement|engineer|salary|API|endpoint|URL|health\sdepartment
    """
    response = llm.invoke(prompt)
    
    if isinstance(response.content, list):
        raw_text = "".join([str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in response.content])
    else:
        raw_text = response.content
        
    regex_pattern = raw_text.strip()

    if "SKIP" in regex_pattern.upper():
        msg = "🛑 **Agent Decision:** Query is out-of-scope for domain knowledge. Skipping Regex."
        st.write(msg)
        if status_steps is not None: status_steps.append(msg)
        return []
    
    msg = f"🕵️‍♂️ **Agent Generated Regex:** `{regex_pattern}`"
    st.write(msg)
    if status_steps is not None: status_steps.append(msg)
    
    matches =[]
    ignore_dirs = {'.git', '.venv', '__pycache__', 'node_modules'}

    try:
        pattern = re.compile(regex_pattern, re.IGNORECASE)
        for root, dirs, files in os.walk(search_dir):
            dirs[:] =[d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                if file.endswith(('.jpg', '.png', '.pyc', '.pdf', '.zip')):
                    continue
                    
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if pattern.search(content):
                            from langchain_core.documents import Document
                            matches.append(Document(
                                page_content=content[:1000] + "... [TRUNCATED]", 
                                metadata={"filename": file, "source": "Agentic Regex"}
                            ))
                except UnicodeDecodeError:
                    pass 
                    
    except Exception as e:
        st.error(f"Regex error: {e}")
        return []
        
    return matches[:3] 

def grade_context_relevance(query, chunks, api_key, status_steps=None):
    """
    Corrective RAG (CRAG) mechanism.
    Evaluates if the retrieved chunks actually answer the user's query.
    """
    if not chunks:
        return False
        
    context_text = "\n".join([doc.page_content for doc in chunks])
    
    prompt = ChatPromptTemplate.from_template("""
    Role: Strict Documentation Auditor.
    Task: Evaluate context relevance to minimize RAG hallucinations.
    
    EVALUATION CRITERIA:
    1. Does the CONTEXT provide specific domain information to answer the QUESTION?
    2. Is the match semantic rather than just accidental keyword overlap?
    
    QUESTION: {query}
    CONTEXT: {context}
    
    DECISION: Return 'YES' only if the context is highly useful. Else return 'NO'.
    SINGLE-WORD RESPONSE:
    """)
    
    eval_llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.0, api_key=api_key)
    response = eval_llm.invoke(prompt.format(query=query, context=context_text))

    if isinstance(response.content, list):
        raw_text = "".join([str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in response.content])
    else:
        raw_text = response.content
    
    decision = raw_text.strip().upper()

    msg = f"🧠 **Grader Decision:** `{decision}`"
    st.write(msg)
    if status_steps is not None: status_steps.append(msg)

    return "YES" in decision

def patch_documentation_with_feedback(original_query, draft_answer, commentary, llm):
    """
    Uses LLM to refine the draft based on expert feedback.
    """
    prompt = ChatPromptTemplate.from_template("""
    You are a Domain Knowledge Documentation Editor. 
    You generated a draft for the question: "{query}"
    
    ORIGINAL DRAFT:
    {draft}
    
    SUBJECT MATTER EXPERT COMMENTARY:
    "{commentary}"
    
    TASK:
    Rewrite the documentation draft. Incorporate the Subject Matter Expert's commentary as the absolute truth.
    Correct any errors mentioned. Keep the tone professional and technical aligning with domain knowledge field implicitly or explicitly said.
    
    REVISED DOCUMENTATION:
    """)
    
    chain = prompt | llm
    response = chain.invoke({
        "query": original_query,
        "draft": draft_answer,
        "commentary": commentary
    })
    
    if isinstance(response.content, list):
        return "".join([str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in response.content])
    return str(response.content)

def get_indexed_metadata_summary(vector_db):
    """
    Queries the Vector DB directly for a list of all indexed files.
    """
    try:
        data = vector_db.get()
        metadatas = data.get("metadatas", [])
        
        if not metadatas:
            return "I have no files indexed in my memory currently."
        
        file_map = {}
        for m in metadatas:
            fname = m.get("filename", "Unknown")
            if fname not in file_map:
                file_map[fname] = m.get("intent", "No summary available")
        
        summary = "I have knowledge about the following files:\n"
        for fname, intent in file_map.items():
            summary += f"- **{fname}**: {intent}\n"
        
        return summary
    except Exception as e:
        return f"Error accessing system manifest: {e}"
