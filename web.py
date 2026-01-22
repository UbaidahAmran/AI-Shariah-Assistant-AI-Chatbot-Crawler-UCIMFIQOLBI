import streamlit as st
import os
import fitz  # PyMuPDF
import csv
from PIL import Image
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURATION & LIGHT MODE ---
st.set_page_config(page_title="Shariah Assistant", page_icon="🕌", layout="wide")

BANK_BLUE = "#00539C"
BANK_ORANGE = "#F47920"

st.markdown(f"""
    <style>
    .stApp {{ background-color: #FFFFFF; }}
    .stMarkdown p, .stMarkdown ul, .stMarkdown li, .stMarkdown strong {{ color: #222222 !important; }}
    h1, h2, h3 {{ color: {BANK_BLUE} !important; font-family: 'Segoe UI', sans-serif; }}
    .stChatInput input {{ background-color: #FFFFFF !important; color: #000000 !important; border: 2px solid {BANK_BLUE} !important; }}
    
    /* Button Styling */
    .stButton button {{ 
        width: 100%; border: 1px solid {BANK_BLUE}; color: {BANK_BLUE}; 
        background-color: white; border-radius: 20px; font-size: 13px; 
        transition: all 0.3s; height: auto; padding: 10px; white-space: normal; 
    }}
    .stButton button:hover {{ background-color: {BANK_BLUE}; color: white; border-color: {BANK_BLUE}; }}
    
    /* Link Styling */
    .stMarkdown a {{ color: {BANK_ORANGE} !important; font-weight: bold; text-decoration: none; }}
    .stMarkdown a:hover {{ text-decoration: underline; }}
    
    /* Warning Box Styling (for General Knowledge answers) */
    .warning-box {{
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin-bottom: 20px;
        color: #856404;
        font-weight: bold;
    }}
    </style>
""", unsafe_allow_html=True)

# --- API & PATHS ---
os.environ["GROQ_API_KEY"] = "" 
DB_PATH = "./chroma_db"
PDF_FOLDER = "./my_pdfs"
CSV_PATH = "sources.csv"
GENERAL_URL = "https://www.bnm.gov.my/banking-islamic-banking"

# --- HEADER ---
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("🕌 Shariah Assistant")
    st.markdown(f"**Regulatory & Policy Intelligence** | <span style='color:{BANK_ORANGE}'>Hybrid Evidence Mode</span>", unsafe_allow_html=True)
st.markdown("---")

# --- ROBUST CSV LOADER ---
def load_url_map():
    url_map = {}
    if os.path.exists(CSV_PATH):
        try:
            with open(CSV_PATH, mode='r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                reader.fieldnames = [name.strip().lower() for name in reader.fieldnames]
                
                if 'filename' in reader.fieldnames and 'url' in reader.fieldnames:
                    for row in reader:
                        f_name = row['filename'].strip().lower()
                        url = row['url'].strip()
                        if f_name and url:
                            url_map[f_name] = url
        except Exception:
            pass # Fail silently in UI
    return url_map

URL_MAP = load_url_map()

# --- LOAD ENGINE ---
@st.cache_resource
def load_chain():
    if not os.path.exists(DB_PATH):
        return None
    
    embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_fn)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) 
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    # --- UPDATED PROMPT: HYBRID MODE ---
    # We instruct the AI to check context first. 
    # If missing, it uses general knowledge but MUST add the specific warning tag.
    template = """You are a Shariah compliance assistant. 
    
    STEP 1: Check the 'Context' below for the answer.
    - If the answer is found in the Context, answer normally based on the text.
    
    STEP 2: If the answer is NOT in the Context:
    - You may answer based on your general knowledge of Shariah standards (BNM/AAOIFI).
    - HOWEVER, you MUST start your response with this exact warning phrase:
      "⚠️ **General Knowledge Mode:** This answer is based on general Islamic finance principles and is NOT found in your uploaded policy documents. Please verify with official sources."
    
    STEP 3: Follow-up Questions.
    - At the very end, generate 3 relevant follow-up questions based on the answer you provided.
    - Format: ||| Question 1? ||| Question 2? ||| Question 3?
    
    Context: {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source

chain = load_chain()

if not chain:
    st.error("Database not found! Please run 'ingest.py' first.")
    st.stop()

# --- HELPER: RENDER PDF PAGE ---
def get_pdf_page_image(pdf_filename, page_num):
    pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
    if not os.path.exists(pdf_path):
        return None
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num) 
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    except Exception:
        return None

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "suggestions" not in st.session_state:
    st.session_state.suggestions = [
        "What is the ruling on Tawarruq?",
        "Explain the conditions for Murabaha.",
        "What are the types of Riba?"
    ]

# --- DISPLAY HISTORY ---
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🕌"
    with st.chat_message(msg["role"], avatar=avatar):
        # Check if this message was a "General Knowledge" answer
        content = msg["content"]
        if "General Knowledge Mode" in content:
            # Render the warning nicely using HTML/CSS
            warning_text, actual_answer = content.split("verify with official sources.", 1)
            st.markdown(f'<div class="warning-box">{warning_text} verify with official sources.</div>', unsafe_allow_html=True)
            st.markdown(actual_answer.strip())
        else:
            st.markdown(content)
            
        if "citations" in msg:
            st.markdown(msg["citations"]) 
        if "images" in msg:
            for img_data in msg["images"]:
                st.image(img_data["img"], caption=img_data["caption"], width=600)

# --- SUGGESTIONS ---
st.markdown("### Suggested Questions:")
cols = st.columns(3)
clicked_prompt = None

for i, suggestion in enumerate(st.session_state.suggestions):
    if i < 3: 
        if cols[i].button(suggestion):
            clicked_prompt = suggestion

# --- INPUT HANDLING ---
prompt = st.chat_input("Ask about Shariah policies...")

if clicked_prompt:
    prompt = clicked_prompt

if prompt:
    # 1. User Input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Assistant Response
    with st.chat_message("assistant", avatar="🕌"):
        with st.spinner("Analyzing documents..."):
            response = chain.invoke(prompt)
            full_response = response["answer"]
            
            parts = full_response.split("|||")
            answer_text = parts[0].strip()
            
            # Extract new questions
            new_suggestions = []
            if len(parts) > 1:
                for s in parts[1:]:
                    clean_s = s.strip()
                    if clean_s and "?" in clean_s:
                        new_suggestions.append(clean_s)
            
            if len(new_suggestions) >= 1:
                st.session_state.suggestions = new_suggestions[:3]
            
            # --- LOGIC CHECK: IS THIS FROM DOCS OR GENERAL KNOWLEDGE? ---
            # We look for the specific warning phrase we put in the prompt.
            is_general_knowledge = "General Knowledge Mode" in answer_text
            
            # Display Answer
            if is_general_knowledge:
                # Split and style the warning
                if "verify with official sources." in answer_text:
                    warning_part, main_body = answer_text.split("verify with official sources.", 1)
                    st.markdown(f'<div class="warning-box">{warning_part} verify with official sources.</div>', unsafe_allow_html=True)
                    st.markdown(main_body.strip())
                else:
                    st.warning(answer_text) # Fallback if split fails
            else:
                st.markdown(answer_text)
            
            # --- EVIDENCE SECTION ---
            # We ONLY show citations/images if it is NOT general knowledge
            images_to_save = []
            citations_text = ""

            if not is_general_knowledge:
                st.markdown("---")
                st.subheader("📄 Verified Evidence") 
                
                citations_list = []
                seen_pages = set()
                
                cols = st.columns(len(response['context']))
                
                for i, doc in enumerate(response['context']):
                    source_filename = doc.metadata.get('source', 'Unknown')
                    page_idx = doc.metadata.get('page', 0)
                    page_display = page_idx + 1
                    
                    unique_key = f"{source_filename}_{page_idx}"
                    
                    if unique_key not in seen_pages:
                        clean_filename = source_filename.strip().lower()
                        file_url = URL_MAP.get(clean_filename, GENERAL_URL)
                        
                        citation_str = f"**[{source_filename}]({file_url})** (Page {page_display})"
                        citations_list.append(citation_str)
                        
                        img = get_pdf_page_image(source_filename, page_idx)
                        if img:
                            cols[i].image(img, caption=f"Source: {source_filename} (pg {page_display})", use_container_width=True)
                            images_to_save.append({"img": img, "caption": f"Source: {source_filename} (pg {page_display})"})
                        
                        seen_pages.add(unique_key)
                
                if citations_list:
                    citations_text = "\n\n**Sources Used:**\n" + "\n".join([f"- {c}" for c in citations_list])
                    st.markdown(citations_text)
                else:
                    # If logical check failed but no docs were retrieved (rare edge case)
                    st.info("Answer derived from context, but no specific page images could be rendered.")

            # Save to history
            msg_data = {"role": "assistant", "content": answer_text}
            if not is_general_knowledge:
                msg_data["citations"] = citations_text
                msg_data["images"] = images_to_save
                
            st.session_state.messages.append(msg_data)
            
            st.rerun()