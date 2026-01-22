# 🕌 Shariah Assistant - AI Compliance Chatbot

**Shariah Assistant** is an intelligent conversational agent designed to automate regulatory compliance for Islamic Finance. It combines a web crawler mechanism with an AI chatbot, actively scanning and indexing BNM policy documents. The system allows users to query regulations in natural language, delivering instant answers backed by **direct source links** and **visual page snapshots** for full auditability.

---

## 🚀 Key Features

* **Hybrid Intelligence:** Prioritizes answers from uploaded documents but intelligently switches to "General Knowledge" mode (with disclaimers) if the answer is missing.
* **Visual Evidence:** Renders actual image snapshots of the PDF page where the ruling is found.
* **Direct Linking:** Citations are clickable and redirect users to the live online source (e.g., BNM website).
* **Smart Suggestions:** Automatically generates context-aware follow-up questions.
* **Strict Source Control:** Uses a `sources.csv` manifest to map local files to authoritative URLs.

---

## 🛠️ Prerequisites

* **Python 3.10+** installed.
* **Groq API Key** (for the Llama-3 LLM).
* **PDF Documents** placed in a local folder.

---

## 📦 Installation

1. **Clone or Download** the repository.
2. **Create a Virtual Environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install Dependencies:**
Create a `requirements.txt` file (content provided below) and run:
```bash
pip install -r requirements.txt

```



---

## ⚙️ Configuration

### 1. Set up API Key

Open `web.py` (or set it in your system environment variables) and ensure your Groq API key is configured:

```python
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

```

### 2. Prepare Documents & Links (`sources.csv`)

1. Place your PDF files in a folder named `my_pdfs`.
2. Create/Edit the `sources.csv` file in the root directory. This maps your local files to their online URLs.
3. **Format:**
```csv
filename,url
tawarruq.pdf,https://www.bnm.gov.my/documents/tawarruq
hibah.pdf,https://www.bnm.gov.my/documents/hibah

```



---

## 🏃‍♂️ How to Run

### Phase 1: Ingestion (Build the Brain)

Before you can chat, you must convert your PDFs into a vector database.
Run the ingestion script:

```bash
python ingest.py

```

* *Output:* This will read PDFs from `my_pdfs`, chunk the text, and save the vector index to the `./chroma_db` folder.

### Phase 2: Launch the Chatbot

Start the web interface:

```bash
streamlit run web.py

```

* The application will open automatically in your browser (usually at `http://localhost:8501`).

---

## 🌐 Phase 3: Launching the Bank Portal Interface

This project utilizes a **"Headless" architecture**. The Streamlit app runs in the background as an API/UI server, while the user interacts with a custom HTML Banking Portal.

### 1. Start the AI Backend
First, you must keep the Streamlit python script running. This powers the chatbot logic.

```bash
streamlit run web.py --server.port 8501

```

> **⚠️ Important:** Once the command runs, **minimize** the terminal window. **Do not close it**, or the AI will stop working.

### 2. Run the Client Simulation

With the backend running on Port 8501, you can now launch the user interface:

1. Locate the file **`index.html`** in your project folder.
2. **Double-click** it to open it in your web browser (Chrome/Edge/Safari).
3. You will see the **Bank Rakyat Homepage** simulation.
4. Click on **"Shariah Centre"** in the top navigation bar.
5. This redirects you to **`shariah.html`**, where the AI Chatbot is embedded and ready to answer queries based on the BNM regulations listed in the sidebar.


## 📂 Project Structure

```
.
├── chromadb/            # (Generated) Vector database storage
├── my_pdfs/             # Folder containing your PDF documents
├── ingest.py            # Script to process PDFs and create the database
├── web.py               # Main Streamlit application (Chatbot UI)
├── index.html          # The Banking Homepage
├── shariah.html        # The Shariah Centre (Chatbot Interface)
├── sources.csv          # Manifest file mapping Filenames -> URLs
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

```

---

## 📄 Requirements File (`requirements.txt`)

Copy the text below into a file named `requirements.txt` to install the necessary libraries.

```text
streamlit
langchain
langchain-groq
langchain-chroma
langchain-huggingface
pymupdf
pillow
python-dotenv
chromadb

```
