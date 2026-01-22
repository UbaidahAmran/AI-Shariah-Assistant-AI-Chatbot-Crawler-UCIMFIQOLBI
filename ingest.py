import os
import time
import glob
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
# UPDATED: Now a list of URLs to crawl sequentially
TARGET_URLS = [
    "https://www.bnm.gov.my/banking-islamic-banking"
    "https://www.bnm.gov.my/insurance-takaful"
    "https://www.bnm.gov.my/development-financial-institutions"
    "https://www.bnm.gov.my/money-services-business"
    "https://www.bnm.gov.my/intermediaries"
    "https://www.bnm.gov.my/payment-systems"
    "https://www.bnm.gov.my/dnfbp",
    "https://www.bnm.gov.my/regulations/currency"
]

DOWNLOAD_DIR = os.path.join(os.getcwd(), "my_pdfs")
DB_PATH = "./chroma_db"
MAX_PAGES = 20 

def setup_driver():
    chrome_options = Options()
    # chrome_options.add_argument("--headless") # Comment out to watch it work
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--start-maximized") 
    
    prefs = {
        "download.default_directory": DOWNLOAD_DIR,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

def crawl_and_download():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    driver = setup_driver()
    all_pdf_links = set() # This set will hold links from ALL categories
    
    try:
        # --- NEW: Loop through each Target URL ---
        for current_url in TARGET_URLS:
            print(f"\n==========================================")
            print(f" STARTING CRAWL: {current_url}")
            print(f"==========================================")
            
            driver.get(current_url)
            time.sleep(5) 

            page_count = 1
            # Loop through pages for the CURRENT category
            while page_count <= MAX_PAGES:
                print(f"\n--- Processing Page {page_count} (Category: {current_url.split('/')[-1]}) ---")
                
                # 1. Scrape PDFs
                elements = driver.find_elements(By.TAG_NAME, "a")
                initial_count = len(all_pdf_links)
                for elem in elements:
                    try:
                        href = elem.get_attribute("href")
                        if href and href.endswith(".pdf"):
                            all_pdf_links.add(href)
                    except:
                        pass
                
                new_found = len(all_pdf_links) - initial_count
                print(f"Found {new_found} new PDFs. (Total Unique Collected: {len(all_pdf_links)})")

                # 2. FIND THE NEXT BUTTON
                try:
                    next_btn = None
                    
                    # Strategy 1: CSS Selector
                    try:
                        next_btn = driver.find_element(By.CSS_SELECTOR, "li.next:not(.disabled) a")
                    except:
                        # Strategy 2: Text Search
                        try:
                            links = driver.find_elements(By.XPATH, "//a[contains(text(), 'Next')]")
                            for link in links:
                                if link.is_displayed():
                                    next_btn = link
                                    break
                        except:
                            pass

                    if next_btn:
                        print("Found 'Next' button. Clicking via JavaScript...")
                        driver.execute_script("arguments[0].click();", next_btn)
                        print("Waiting for page update...")
                        time.sleep(8) 
                        page_count += 1
                    else:
                        print(f"❌ No 'Next' button found. Finished category: {current_url.split('/')[-1]}")
                        break # Breaks the page loop, moves to next URL in TARGET_URLS
                        
                except Exception as e:
                    print(f"Navigation Error on {current_url}: {e}")
                    break
        
        # --- END OF CRAWLING LOOPS ---

        # 5. Download Phase (Happens once after collecting ALL links from ALL categories)
        print(f"\n--- Starting Bulk Download of {len(all_pdf_links)} Files ---")
        for i, link in enumerate(all_pdf_links):
            filename = link.split("/")[-1]
            # Avoid re-downloading existing files
            if os.path.exists(os.path.join(DOWNLOAD_DIR, filename)):
                continue 
                
            print(f"[{i+1}/{len(all_pdf_links)}] Downloading...")
            try:
                driver.get(link)
                time.sleep(3)
            except:
                pass

    finally:
        driver.quit()

def ingest_to_db():
    print("\n--- Starting Database Ingestion ---")
    pdf_files = glob.glob(os.path.join(DOWNLOAD_DIR, "*.pdf"))
    
    if not pdf_files:
        print("No PDFs found.")
        return

    print(f"Indexing {len(pdf_files)} files...")
    all_documents = []
    
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            for doc in docs:
                # Metadata: Source filename
                doc.metadata["source"] = os.path.basename(pdf_path)
                # Optional: Add category metadata based on file path if you separate folders later
            all_documents.extend(docs)
        except:
            print(f"Skipping bad file: {os.path.basename(pdf_path)}")

    if all_documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_documents)

        print("Updating ChromaDB...")
        embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        Chroma.from_documents(documents=splits, embedding=embedding_fn, persist_directory=DB_PATH)
        print("Success! Brain update complete.")

if __name__ == "__main__":
    crawl_and_download()
    ingest_to_db()