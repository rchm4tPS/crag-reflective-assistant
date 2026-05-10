import streamlit as st

st.set_page_config(page_title="Help Guide | Reflective RAG", layout="wide")

st.markdown("""
<style>
    /* Hide the default Streamlit sidebar page links since we are using explicit buttons */
    [data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
if st.sidebar.button("🔙 Back to Main App", use_container_width=True):
    st.switch_page("app.py")
st.sidebar.divider()

st.title("📖 How to Use the Reflective Mentoring Assistant")

st.markdown("""
Welcome to the **Reflective Mentoring Assistant**! This system uses advanced Corrective Retrieval-Augmented Generation (CRAG) to ensure you always get highly accurate, verified answers.

There are two main ways to interact with the system, depending on your role.

---

### 👤 1. Hunter Mode (Newcomer)
This is the default view. You act as a user looking for answers from the knowledge base.

*   **Ask Anything:** Type your technical questions in the chat box.
*   **Transparent Thinking:** As the AI works, click on the **"Analyzing query..."** status box to see exactly what it is doing (expanding queries, retrieving files, confidence voting, reranking, and grading).
*   **Web Fallback:** If the system cannot find the answer in the internal documentation, it will automatically search DuckDuckGo to try and help you anyway.
*   **Nominate for Documentation:** If you get an awesome AI-generated answer and think *"This should be official documentation!"*, click the **💡 Promote to Documentation** button. It will be sent to the Veteran Dashboard for review.

---

### 🛠️ 2. Engineer Mode (Veteran / Expert)
Select **"🛠️ Engineer"** from the left sidebar to enter the Knowledge Management Dashboard.

*   **Review Nominations:** Here, you will see all the answers that "Hunters" nominated for promotion.
*   **Patch with Commentary:** Read the original AI draft. If it is slightly wrong or missing details, type your corrections into the "Expert Commentary" box and click **🪄 Refine with AI**. The AI will rewrite the draft incorporating your absolute truth.
*   **Manual Editing:** If you want to rewrite it yourself, click **📝 Edit Manually**.
*   **Final Approval:** Once the answer is perfect, click **✅ Final Approve & Save**.
*   **The Magic:** Once approved, that exact Q&A pair is saved into the **Semantic Cache**. The next time *anyone* asks a similar question, they will instantly receive your Official Veteran-Approved documentation instead of a random AI guess!

---

### 🧠 How the Architecture Works (Under the Hood)
""")

from config import ARCHITECTURE_DIAGRAM_URL
import os

if ARCHITECTURE_DIAGRAM_URL.startswith("http") or os.path.exists(ARCHITECTURE_DIAGRAM_URL):
    # If it's a URL (like Cloudinary), Streamlit will fetch and render it natively without local PIL compression!
    st.image(ARCHITECTURE_DIAGRAM_URL, caption="CRAG Pipeline Architecture", use_container_width=True, output_format="PNG")
else:
    st.info("💡 Place your pipeline diagram image at `assets/architecture.png` or update `ARCHITECTURE_DIAGRAM_URL` in `config.py`.")

st.markdown("""
1.  **Memory Layer:** Checks the Semantic Cache for Veteran-approved answers first.
2.  **Expansion:** If not found, breaks your query into 3 sub-queries.
3.  **Retrieval & Voting:** Pulls the top 10 chunks per sub-query and votes on confidence.
4.  **Reranking:** Re-orders the results using a lightweight Cross-Encoder for maximum accuracy.
5.  **Strict Evaluator:** A second AI acts as a strict Grader, evaluating if the context actually answers your question.
6.  **Agentic Regex:** If vector search fails, an AI Agent generates a Regex pattern to search the raw file system.
7.  **Web Search:** The absolute last resort.
""")
