import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# =========================
# CONFIGURATION
# =========================
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    st.error("⚠️ NVIDIA_API_KEY not found in environment variables")
    st.stop()

LLM_MODEL = "meta/llama-3.1-70b-instruct"
EMBED_MODEL = "nvidia/nv-embed-v1"
CHROMA_DIR = "chroma_db"
TOP_K = 5
chunk_size=1000
chunk_overlap=200

SYSTEM_PROMPT = """You are an intelligent document assistant. Answer questions based ONLY on the provided context.

Guidelines:
- Provide clear, accurate answers from the document
- If information isn't in the context, say: "I cannot find this information in the document."
- Be concise and well-structured
- Use bullet points when listing multiple items - most important thinks
- Maintain a professional tone
"""

# =========================
# INITIALIZE MODELS
# =========================
@st.cache_resource
def load_llm():
    return ChatNVIDIA(
        model=LLM_MODEL,
        api_key=NVIDIA_API_KEY,
        temperature=0.2,
        max_completion_tokens=2048
    )

@st.cache_resource
def load_embeddings():
    return NVIDIAEmbeddings(
        model=EMBED_MODEL,
        api_key=NVIDIA_API_KEY
    )

llm = load_llm()
embeddings = load_embeddings()

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="RAG Chatbot | Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Upload box styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Clear button styling */
    .stButton>button[kind="primary"] {
        background-color: #dc3545 ;
        border-color: #dc3545 ;
    }
    
    .stButton>button[kind="primary"]:hover {
        background-color: #c82333 ;
        border-color: #bd2130 ;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.4) !important;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
    }
    
    .info-box h3 {
        margin-top: 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>RAG Chatbot</h1>", unsafe_allow_html=True)
    
    if st.session_state.pdf_loaded:
        st.markdown("### 📄 Document Status")
        st.success("✅ Document Loaded")
        st.info(f"**File:** {st.session_state.pdf_name}")
        
        st.markdown("---")
        
        if st.button("🗑️ Clear & Start New", type="primary"):
            st.session_state.messages = []
            st.session_state.retriever = None
            st.session_state.pdf_loaded = False
            st.session_state.pdf_name = None
            st.rerun()
    else:
        st.markdown("### 🗂 Upload Document")
        st.info("Upload a PDF to begin")
    
    st.markdown("---")
    
    st.markdown("""
    ### 📖 How It Works
    
    **Step 1:** Upload your PDF document
    
    **Step 2:** Wait for processing
    
    **Step 3:** Ask questions!
    
    ---
    
    ### ✨ Features
    - AI-powered answers
    - Context-aware chat
    - Fast responses
    
    ---

    """)

# =========================
# MAIN CONTENT
# =========================
st.markdown("<h1 class='main-title'>Smart Document Intelligence Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a PDF and ask questions - Get instant, accurate answers</p>", unsafe_allow_html=True)

# =========================
# PDF UPLOAD SECTION
# =========================
if not st.session_state.pdf_loaded:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        uploaded_pdf = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF document to start chatting",
            label_visibility="collapsed"
        )
        
        if uploaded_pdf:
            with st.spinner("🔄 Processing document... Please wait"):
                try:
                    # Save temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_pdf.read())
                        pdf_path = tmp.name

                    # Load and split
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()
                    
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=len,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    chunks = splitter.split_documents(documents)

                    # Create vector store
                    vectordb = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=CHROMA_DIR,
                        collection_name="pdf_collection"
                    )

                    # Create retriever
                    st.session_state.retriever = vectordb.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": TOP_K}
                    )
                    
                    st.session_state.pdf_loaded = True
                    st.session_state.pdf_name = uploaded_pdf.name

                    # Cleanup
                    os.unlink(pdf_path)

                    st.success(f"✅ Document ready! ({len(chunks)} chunks created)")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Info box when no document is loaded
        if not uploaded_pdf:
            st.markdown("""
            <div class='info-box'>
                <h3>🚀 Get Started</h3>
                <p>Upload a PDF document above to unlock intelligent Q&A capabilities. Our AI will process your document and answer your questions instantly.</p>
            </div>
            """, unsafe_allow_html=True)

# =========================
# CHAT INTERFACE
# =========================
if st.session_state.pdf_loaded:
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_query := st.chat_input("💬 Ask me anything about your document..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing..."):
                try:
                    # Retrieve relevant chunks
                    docs = st.session_state.retriever.invoke(user_query)
                    
                    # Build context
                    context_parts = []
                    for i, doc in enumerate(docs, 1):
                        page_num = doc.metadata.get("page", "Unknown")
                        context_parts.append(f"[Source {i} - Page {page_num}]:\n{doc.page_content}")
                    
                    context = "\n\n".join(context_parts)

                    # Create prompt
                    prompt = f"""{SYSTEM_PROMPT}

Context from Document:
{context}

Question: {user_query}

Answer:"""

                    # Get LLM response
                    response = llm.invoke(prompt)
                    answer = response.content

                    st.markdown(answer)
                    
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""

""", unsafe_allow_html=True)
