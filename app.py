import streamlit as st
import datetime
import time
import re
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from ddgs import DDGS

from config import *
from rag_pipeline import *

# --- 1. CONFIGURATION & INITIALIZATION ---
st.set_page_config(page_title="Reflective RAG Assistant", layout="wide")

st.markdown("""
<style>
    /* Hide the default Streamlit sidebar page links since we are using explicit buttons */
    [data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("👨‍💻 Reflective Mentoring Assistant")

@st.dialog("🔒 API Authentication Required", dismissible=False)
def prompt_for_api_key():
    st.write("Welcome! Please enter your **Google Gemini API Key** to start the Assistant. You can get one from Google AI Studio.")
    api_key = st.text_input("🔑 API Key", type="password")
    if st.button("Submit"):
        if not api_key:
            st.error("API Key cannot be empty.")
        elif not api_key.startswith("AIza") or len(api_key) != 39:
            st.error("Invalid API Key format. Google Gemini keys start with 'AIza' and are 39 characters long.")
        else:
            # Validate the API Key by doing small test call to Gemini API
            with st.spinner("Validating your API Key . . ."):
                try:
                    test_llm = ChatGoogleGenerativeAI(
                        model=LLM_MODEL,
                        api_key=api_key
                    )
                    test_llm.invoke("Hi")

                    st.session_state["user_api_key"] = api_key
                    st.rerun()
                except Exception:
                    st.error("❌ Authentication failed: the provided API key is invalid, revoked, or lacks permission!")

# Determine active API Key (Secrets first, then User's Session)
try:
    active_api_key = st.secrets.get("GOOGLE_API_KEY") or st.session_state.get("user_api_key")
except Exception:
    active_api_key = st.session_state.get("user_api_key")

if not active_api_key:
    prompt_for_api_key()
    st.stop()

@st.cache_resource
def load_models(api_key):
    _llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, 
        temperature=LLM_TEMP,
        api_key=api_key
    )
    _embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key
    )
    return _llm, _embeddings

llm, embeddings = load_models(active_api_key)

@st.cache_resource
def load_vector_dbs(api_key):
    # Initialize fresh embeddings mapped to this specific user's API key
    _embeddings_local = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key
    )
    v_db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=_embeddings_local
    )
    c_db = Chroma(
        persist_directory=CACHE_PATH,
        embedding_function=_embeddings_local,
        collection_name="semantic_cache",
        collection_metadata={"hnsw:space": "cosine"}
    )
    return v_db, c_db

vector_db, cache_db = load_vector_dbs(active_api_key)

@st.cache_resource
def load_reranker():
    return CrossEncoder(RERANKER_MODEL, max_length=512)

reranker = load_reranker()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. THE UI ORCHESTRATOR ---

st.sidebar.title("Navigation")
persona = st.sidebar.radio("🎭 Role", ["👤 Hunter", "🛠️ Engineer"], label_visibility="visible", width="content")

st.sidebar.divider()
if st.sidebar.button("📖 Open Help Guide", use_container_width=True):
    st.switch_page("pages/1_Help_Guide.py")

if "user_api_key" in st.session_state:
    if st.sidebar.button("🚪 Log Out (Clear API Key)", use_container_width=True):
        del st.session_state["user_api_key"]
        st.rerun()
st.sidebar.divider()

if "👤 Hunter" in persona:
    st.caption("🚀 Newcomer Chatbot Mode")

    # 1. Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "caption" in message:
                st.caption(message["caption"])
            if "status_data" in message and message["status_data"].get("label"):
                with st.status(message["status_data"]["label"], expanded=False, state="complete"):
                    for step in message["status_data"].get("steps", []):
                        st.write(step)
                    # RE-RENDER TRACES from persistent data
                    if "traces" in message["status_data"]:
                        for trace in message["status_data"]["traces"]:
                            with st.expander(f"📄 Raw Retrieval Trace ({trace['query']})", expanded=False):
                                for i, res in enumerate(trace["results"]):
                                    st.write(f"{i+1}. `{res['filename']}` (Score: `{res['score']:.4f}`)")
            if "warning" in message:
                st.warning(message["warning"])
            if "success" in message:
                st.success(message["success"])

            # --- LIVE REFLECTION: Show Veteran's refined answer if it exists ---
            display_content = message["content"]
            is_verified = False
            if message.get("doc_id"):
                live_doc = cache_db.get(ids=[message["doc_id"]])
                if live_doc and live_doc["metadatas"]:
                    meta = live_doc["metadatas"][0]
                    # If it has been verified/edited by a veteran, show the official version
                    if meta.get("status") == "auto-verified":
                        display_content = "### 🛡️ Veteran-Approved Documentation:\n" + meta["answer"]
                        is_verified = True
            
            st.markdown(display_content)
            
            if message["role"] == "assistant" and message.get("can_promote") and not message.get("promoted"):
                if st.button("💡 Promote to Documentation", key=f"promote_{st.session_state.messages.index(message)}"):
                    ids = cache_db.add_texts(
                        texts=[message["query"]],
                        metadatas=[{
                            "answer": message["content"],
                            "status": "unverified", # Force unverified so Veteran can review
                            "timestamp": datetime.datetime.now().isoformat()
                        }]
                    )
                    message["doc_id"] = ids[0] 
                    message["promoted"] = True
                    st.success("Nominated! This will now appear in the Veteran's dashboard.")
                    st.rerun()
            elif message.get("promoted"):
                if is_verified:
                    st.caption("✅ Verified & Official Documentation")
                else:
                    st.caption("✨ Nominated for Documentation")
            elif message.get("is_verified_cache"):
                st.caption("✅ Verified & Official Documentation")
    
    # 2. User Input
    if query := st.chat_input("Ask a question about any documentation we have..."):
        st.session_state.messages.append({"role": "user", "content": query})
        st.rerun()

    # 3. Assistant Generation
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        query = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            status_data = {"label": "Working...", "steps": []}
            msg_meta = {"role": "assistant"}
            
            try:
                with st.status("Analyzing query...", expanded=True) as status_box:
                    # Step A: Context-aware Condensation
                    status_box.update(label="Summarizing intent...")
                    condensed_query = condense_history(st.session_state.messages[:-1], query, llm)

                    if "OFF_TOPIC" in condensed_query.upper():
                        class_str = "OFF_TOPIC"
                    else:
                        class_str = "TECHNICAL"

                    if "SELF_QUERY" in condensed_query.upper():
                        status_box.update(label="🔍 Inspecting Internal Manifest...", state="complete")
                        status_data["label"] = "🔍 Inspecting Internal Manifest..."
                        manifest = get_indexed_metadata_summary(vector_db)
                        path = "SELF_QUERY"

                    elif "OFF_TOPIC" in condensed_query.upper():
                        status_box.update(label="💬 Conversing...", state="complete")
                        path = "OFF_TOPIC"

                    else:
                        # PATH 1: Semantic Cache (Fast Path)
                        status_box.update(label=f"🔍 Searching Cache for: `{query}`...")
                        msg_meta["caption"] = f"🔍 Searching for: `{query}`"
                        cached_answer, status = check_semantic_cache(query, cache_db, llm)
                            
                        if cached_answer:
                            status_box.update(label="✅ Found in Cache!", state="complete")
                            path = "CACHE"
                        else:
                            # PATH 2: Full RAG Pipeline (Slow Path)
                            status_box.update(label="🔍 Expanding query variations...")
                            st.write("Expanding query variations...")
                            status_data["steps"].append("Expanding query variations...")
                            
                            sub_queries = expand_query(condensed_query, llm)
                            
                            status_box.update(label="⚖️ Retrieving and Voting...")
                            st.write("Retrieving and Voting...")
                            status_data["steps"].append("Retrieving and Voting...")
                            chunks, is_confident, vote_count = retrieve_and_filter(sub_queries, vector_db, status_data=status_data)
                                
                            st.metric("Sub-query Confidence Votes", f"{vote_count} / 3")
                            status_data["steps"].append(f"Sub-query Confidence Votes: {vote_count} / 3")

                            final_context = []
                            cache_status = "auto-verified"
                            is_rag_success = False
                            
                            if is_confident:
                                status_box.update(label="🔍 Confidence threshold met. Reranking...")
                                st.write("🔍 Confidence threshold met. Reranking...")
                                status_data["steps"].append("Confidence threshold met. Reranking...")
                                temp_context = rerank_chunks(condensed_query, chunks, reranker, status_steps=status_data["steps"])
                                
                                # CRITICAL: The Strict Evaluator Gate (CRAG Step)
                                if grade_context_relevance(condensed_query, temp_context, active_api_key, status_steps=status_data["steps"]):
                                    final_context = temp_context
                                    status_box.update(label="✅ Found relevant Documentation", state="complete")
                                    status_data["label"] = "✅ Found relevant Documentation"
                                    path = "RAG"
                                    is_rag_success = True
                                else:
                                    st.write("❌ Primary Context deemed irrelevant by Grader. Falling back to Domain Knowledge Search...")
                                    status_data["steps"].append("❌ Primary Context deemed irrelevant by Grader. Falling back to Domain Knowledge Search...")

                            # Corrective Fallback: Trigger if not confident OR if primary context was rejected
                            if not is_rag_success:
                                if not is_confident:
                                    status_box.update(label="❌ Low confidence. Triggering Agentic Search via Regex...")
                                    st.write("❌ Low confidence (< 2 votes). Triggering Agentic Search via Regex.")
                                    status_data["steps"].append("❌ Low confidence (< 2 votes). Triggering Agentic Search via Regex.")
                                    
                                status_box.update(label="🕵️ Agentic Regex Search...")
                                regex_chunks = agentic_regex_search(condensed_query, llm, status_steps=status_data["steps"])
                                    
                                if regex_chunks and grade_context_relevance(condensed_query, regex_chunks, active_api_key, status_steps=status_data["steps"]):
                                    final_context = regex_chunks
                                    cache_status = "unverified"
                                    status_box.update(label="⚠️ Found in knowledge base", state="complete")
                                    status_data["label"] = "⚠️ Found in knowledge base"
                                    path = "RAG"
                                    is_rag_success = True
                                else:
                                    st.write("Searching Web...")
                                    status_data["steps"].append("Searching Web...")
                                    
                                    # Do the web search here inside the status box!
                                    try:
                                        web_results = DDGS().text(condensed_query, max_results=3)
                                        web_context = "\n\n".join([
                                            f"--- Source: {res['title']} ({res['href']}) ---\nSnippet: {res['body']}" 
                                            for res in web_results
                                        ])
                                    except Exception as e:
                                        # Fallback gracefully if web search gets rate-limitation
                                        web_context = f"Web search temporarily unavailable due to error: {e}"
                                    
                                    status_box.update(label="🌐 Context not found. Searching Web instead", state="complete")
                                    status_data["label"] = "🌐 Context not found. Searching Web instead"
                                    path = "WEB"

            except Exception as e:
                st.error("❌ **AI Generation Error:** The connection to Gemini failed. You may have run out of API quota, or the key was revoked!")
                with st.expander("Show technical details:"):
                    st.error(str(e))

            # --- OUTSIDE STATUS BOX: FINAL GENERATION STREAMING ---
            if path == "SELF_QUERY":
                msg_meta["status_data"] = status_data
                # Use a more targeted prompt that includes the user's actual query
                prompt = f"""
                Role: Technical Knowledge Navigator.
                Task: Answer queries about domain knowledge structure and AI self-awareness.
                
                INDEXED MANIFEST:
                {manifest}
                
                USER QUESTION: {query}
                
                GUIDELINES:
                - Use the manifest as the absolute source of truth.
                - If the user filters for a subset or more specific field/theme (e.g., 'backend' or 'anti money laudrying'), stick to it, then filter your response accordingly.
                - Explain the 'Intent' of files to help the user understand the domain knowledge.
                
                PROFESSIONAL RESPONSE:
                """
                response = llm.invoke(prompt).content
                clean_response = "".join([str(p.get("text", p)) for p in response]) if isinstance(response, list) else str(response)
                
                st.write(clean_response)
                msg_meta["content"] = clean_response
                st.session_state.messages.append(msg_meta)
                st.rerun()

            elif path == "OFF_TOPIC":
                response = llm.invoke(f"The user said '{query}'. Respond politely but briefly as a technical assistant.").content
                clean_resp = "".join([str(p.get("text", p)) for p in response]) if isinstance(response, list) else str(response)
                st.write(clean_resp)
                msg_meta["content"] = clean_resp
                st.session_state.messages.append(msg_meta)
                st.rerun()

            elif path == "CACHE":
                if status == "unverified":
                    st.warning("⚠️ Found in Cache (Awaiting Veteran Verification)")
                    msg_meta["warning"] = "⚠️ Found in Cache (Awaiting Veteran Verification)"
                    final_answer = cached_answer
                else:
                    st.success("✅ Found in Cache!")
                    msg_meta["success"] = "✅ Found in Cache!"
                    final_answer = "### 🛡️ Veteran-Approved Documentation:\n" + cached_answer
                    msg_meta["is_verified_cache"] = True
                    
                st.write(final_answer)
                msg_meta["content"] = final_answer
                st.session_state.messages.append(msg_meta)
                st.rerun()

            elif path == "RAG":
                msg_meta["status_data"] = status_data
                msg_meta["query"] = query
                msg_meta["status"] = cache_status
                msg_meta["can_promote"] = True
                
                full_response = generate_and_cache(condensed_query, final_context, llm, cache_status=cache_status)
                msg_meta["content"] = "### 🤖 AI Response:\n" + full_response
                st.session_state.messages.append(msg_meta)
                st.rerun()

            elif path == "WEB":
                msg_meta["status_data"] = status_data
                msg_meta["query"] = query
                msg_meta["status"] = "unverified"
                msg_meta["can_promote"] = True
                
                prompt = ChatPromptTemplate.from_template("""
                    You are a technical assistant. Internal documentation and domain knowledge failed to answer the user's query.
                    I have performed a live web search. Answer the user's question using ONLY the web results provided below.
                    Cite the sources (URLs) in your answer.
                    
                    LIVE WEB CONTEXT:
                    {web_context}
                    
                    QUESTION: {query}
                    
                    DETAILED ANSWER:
                    """
                )
                    
                fallback_chain = prompt | llm
                    
                def stream_web_fallback():
                    for chunk in fallback_chain.stream({"web_context": web_context, "query": condensed_query}):
                        if isinstance(chunk.content, list):
                            raw_text = "".join([str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in chunk.content])
                        else:
                            raw_text = str(chunk.content)
                        tokens = re.split(r'(\s+)', raw_text)
                        for token in tokens:
                            if token:
                                yield token
                st.markdown("### 🌐 Web Search Response:")
                final_fallback_answer = st.write_stream(stream_web_fallback())
                
                msg_meta["content"] = "### 🌐 Web Search Response:\n" + final_fallback_answer
                st.session_state.messages.append(msg_meta)
                st.rerun()

elif "🛠️ Engineer" in persona:
    st.subheader("🛠️ Knowledge Management Dashboard")
    st.write("Review promoted AI drafts answers and provide expert commentary to 'patch' the domain-expert documentation.")
    
    # 1. Fetch unverified items
    unverified_items = cache_db.get(where={"status": "unverified"})

    if not unverified_items or not unverified_items.get("ids"):
        st.success("🎉 All caught up!")
    else:
        for i, doc_id in enumerate(unverified_items["ids"]):
            original_query = unverified_items["documents"][i]
            draft_answer = unverified_items["metadatas"][i].get("answer", "")

            # SESSION STATE INITIALIZATION
            revised_key = f"revised_{doc_id}"
            edit_mode_key = f"edit_mode_{doc_id}"
            
            if revised_key not in st.session_state: st.session_state[revised_key] = None
            if edit_mode_key not in st.session_state: st.session_state[edit_mode_key] = False
            
            with st.expander(f"Review Required: {original_query}", expanded=(i==0)):
                st.info(f"**Newcomer asked:** {original_query}")

                # SHOW BEFORE vs AFTER if a revision exists
                if st.session_state[revised_key]:
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original AI Draft (Before)**")
                        st.caption(draft_answer)
                    with col2:
                        st.markdown("**Revised Version (After)**")
                        # Toggle between Markdown Preview and Text Area
                        if not st.session_state[edit_mode_key]:
                            # MARDKOWN PREVIEW MODE
                            st.markdown(st.session_state[revised_key])
                            if st.button("📝 Edit Manually", key=f"edit_{doc_id}"):
                                st.session_state[edit_mode_key] = True
                                st.rerun()
                        else:
                            # EDIT MODE
                            updated_text = st.text_area("Manual Polish:", 
                                                        value=st.session_state[revised_key], 
                                                        height=300, 
                                                        key=f"input_{doc_id}")
                            if st.button("💾 Save Manual Edit", key=f"save_edit_{doc_id}"):
                                st.session_state[revised_key] = updated_text
                                st.session_state[edit_mode_key] = False
                                st.rerun()

                    st.divider()

                    # Commentary moved below the comparison
                    commentary = st.text_area("Refine further? Add more commentary:", 
                                             placeholder="e.g., Also mention the security group settings.",
                                             key=f"comm_extra_{doc_id}")
                    
                    c1, c2, c3 = st.columns([1, 1, 1])
                    if c1.button("🪄 Re-Refine with AI", key=f"refine_again_{doc_id}"):
                        with st.spinner("Patching again..."):
                            st.session_state[revised_key] = patch_documentation_with_feedback(
                                original_query, st.session_state[revised_key], commentary, llm
                            )
                            st.rerun()
                    
                    if c2.button("✅ Final Approve & Save", key=f"final_save_{doc_id}", type="primary"):
                        cache_db._collection.update(
                            ids=[doc_id],
                            metadatas=[{
                                "answer": st.session_state[revised_key], 
                                "status": "auto-verified",
                                "timestamp": datetime.datetime.now().isoformat()
                            }]
                        )
                        # Cleanup state
                        del st.session_state[revised_key]
                        del st.session_state[edit_mode_key]
                        st.success("Patched documentation is now Official!")
                        time.sleep(1)
                        st.rerun()

                    if c3.button("🗑️ Discard All", key=f"discard_{doc_id}"):
                        st.session_state[revised_key] = None
                        st.rerun()
                
                # CASE 2: NO REVISION YET (Standard View)
                else:
                    st.write("### 📝 AI's Current Draft:")
                    st.info(draft_answer)
                    
                    commentary = st.text_area("Your Expert Commentary:", 
                                             placeholder="Explain what's wrong or what to add...",
                                             key=f"comm_init_{doc_id}")
                    
                    col_a, col_b, col_c = st.columns(3)
                    if col_a.button("🪄 Refine with AI", key=f"refine_init_{doc_id}", type="primary"):
                        with st.spinner("AI is patching..."):
                            st.session_state[revised_key] = patch_documentation_with_feedback(original_query, draft_answer, commentary, llm)
                            st.rerun()
                    
                    if col_b.button("✅ Quick Approve", key=f"qa_init_{doc_id}"):
                        cache_db._collection.update(ids=[doc_id], metadatas=[{"answer": draft_answer, "status": "auto-verified", "timestamp": datetime.datetime.now().isoformat()}])
                        st.rerun()
                        
                    if col_c.button("🗑️ Delete Draft", key=f"del_init_{doc_id}"):
                        cache_db._collection.delete(ids=[doc_id])
                        st.rerun()
