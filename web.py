import streamlit as st
import os
import fitz  # PyMuPDF
import csv
from PIL import Image
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
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
    st.markdown(f"**Regulatory & Policy Intelligence** | <span style='color:{BANK_ORANGE}'>Global & Local Standards</span>", unsafe_allow_html=True)
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

# --- BALANCED RETRIEVAL LOGIC ---
def get_balanced_docs(docs):
    """
    Filters search results to return:
    - Max 1 document from BNM
    - Max 1 document from IIFA
    """
    bnm_doc = None
    iifa_doc = None
    
    for doc in docs:
        filename = doc.metadata.get('source', '').lower()
        
        # Identify IIFA document by filename pattern
        if 'iifa' in filename:
            if not iifa_doc:
                iifa_doc = doc # Keep the single best IIFA match
        else:
            if not bnm_doc:
                bnm_doc = doc # Keep the single best BNM match
    
    # Compile the final balanced list
    results = []
    if bnm_doc: results.append(bnm_doc)
    if iifa_doc: results.append(iifa_doc)
    
    return results

# --- LOAD ENGINE ---
@st.cache_resource
def load_chain():
    if not os.path.exists(DB_PATH):
        return None
    
    embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_fn)
    
    # 1. Cast a Wide Net (Fetch top 10)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) 
    
    # ... inside load_chain() ...

    # Increase Temperature to 0.6 for natural variation
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
    
    template = """You are a Shariah compliance assistant. 
    
    STEP 1: Context Analysis
    - Synthesize the answer from the provided excerpts.
    - If you see a source labeled 'IIFA', refer to it as "IIFA Resolution".
    - If you see a source labeled 'BNM' or other policies, refer to it as "BNM Policy".
    
    STEP 2: Answer Formulation
    - If the answer is found in the Context, explain it clearly but accurately in a professional, natural tone.
     - Vary your sentence structure slightly if asked the same question to be natural, but DO NOT alter the core meaning or add interpretations not present in the text.
    - If the answer is NOT in the Context:
      You may answer based on your general knowledge of Shariah standards (BNM/AAOIFI).
      HOWEVER, you MUST start your response with this exact warning phrase:
      "⚠️ **General Knowledge Mode:** This answer is based on general Islamic finance principles and is NOT found in your uploaded policy documents. Please verify with official sources."
    
    STEP 3: Follow-up Questions.
    - At the very end, generate 3 relevant follow-up questions.
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

    # 2. Inject the Balancing Logic into the chain
    rag_chain_with_source = RunnableParallel(
        {"context": retriever | RunnableLambda(get_balanced_docs), "question": RunnablePassthrough()}
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
            if "verify with official sources." in content:
                warning_text, actual_answer = content.split("verify with official sources.", 1)
                st.markdown(f'<div class="warning-box">{warning_text} verify with official sources.</div>', unsafe_allow_html=True)
                st.markdown(actual_answer.strip())
            else:
                st.warning(content)
        else:
            st.markdown(content)
            
        if "citations" in msg:
            st.markdown(msg["citations"]) 
        if "images" in msg:
            # Dynamic Columns for Images
            cols = st.columns(len(msg["images"]))
            for i, img_data in enumerate(msg["images"]):
                cols[i].image(img_data["img"], caption=img_data["caption"], use_container_width=True)

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
        with st.spinner("Consulting BNM & IIFA Standards..."):
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
            is_general_knowledge = "General Knowledge Mode" in answer_text
            
            # Display Answer
            if is_general_knowledge:
                if "verify with official sources." in answer_text:
                    warning_part, main_body = answer_text.split("verify with official sources.", 1)
                    st.markdown(f'<div class="warning-box">{warning_part} verify with official sources.</div>', unsafe_allow_html=True)
                    st.markdown(main_body.strip())
                else:
                    st.warning(answer_text)
            else:
                st.markdown(answer_text)
            
            # --- EVIDENCE SECTION ---
            images_to_save = []
            citations_text = ""

            if not is_general_knowledge:
                st.markdown("---")
                st.subheader("📄 Verified Evidence") 
                
                citations_list = []
                seen_pages = set()
                
                # Context is ALREADY filtered by get_balanced_docs (Max 1 BNM, Max 1 IIFA)
                context_docs = response['context']
                cols = st.columns(len(context_docs))
                
                for i, doc in enumerate(context_docs):
                    source_filename = doc.metadata.get('source', 'Unknown')
                    page_idx = doc.metadata.get('page', 0)
                    page_display = page_idx + 1
                    
                    unique_key = f"{source_filename}_{page_idx}"
                    
                    if unique_key not in seen_pages:
                        clean_filename = source_filename.strip().lower()
                        file_url = URL_MAP.get(clean_filename, GENERAL_URL)
                        
                        # --- DYNAMIC LABELING ---
                        # If filename contains 'iifa', label it IIFA. Else BNM.
                        if "iifa" in clean_filename:
                            label_prefix = "IIFA Resolution"
                        else:
                            label_prefix = "BNM Policy"
                        
                        citation_str = f"**{label_prefix}:** [{source_filename}]({file_url}) (Page {page_display})"
                        citations_list.append(citation_str)
                        
                        img = get_pdf_page_image(source_filename, page_idx)
                        if img:
                            cols[i].image(img, caption=f"{label_prefix} - Pg {page_display}", use_container_width=True)
                            images_to_save.append({"img": img, "caption": f"{label_prefix} - Pg {page_display}"})
                        
                        seen_pages.add(unique_key)
                
                if citations_list:
                    citations_text = "\n\n**Sources Used:**\n" + "\n".join([f"- {c}" for c in citations_list])
                    st.markdown(citations_text)

            msg_data = {"role": "assistant", "content": answer_text}
            if not is_general_knowledge:
                msg_data["citations"] = citations_text
                msg_data["images"] = images_to_save
                
            st.session_state.messages.append(msg_data)
            
            st.rerun()