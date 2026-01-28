import os
import time
import glob
import csv
import shutil
import schedule
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
TARGET_URLS = [
    "https://www.bnm.gov.my/banking-islamic-banking",
    "https://www.bnm.gov.my/insurance-takaful",
    "https://www.bnm.gov.my/development-financial-institutions",
    "https://www.bnm.gov.my/money-services-business",
    "https://www.bnm.gov.my/intermediaries",
    "https://www.bnm.gov.my/payment-systems",
    "https://www.bnm.gov.my/dnfbp",
    "https://www.bnm.gov.my/regulations/currency",
    "https://iifa-aifi.org/en/resolutions" 
]

DOWNLOAD_DIR = os.path.join(os.getcwd(), "my_pdfs")
DB_PATH = "./chroma_db"
CSV_PATH = "sources.csv"
MAX_PAGES = 20

def setup_driver():
    chrome_options = Options()
    # chrome_options.add_argument("--headless") 
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
    print(f"🚀 Starting Crawler Job: {time.ctime()}")
    
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    driver = setup_driver()
    all_pdf_links = set() 
    
    try:
        for current_url in TARGET_URLS:
            print(f"\n==========================================")
            print(f" 🔍 SCANNING SOURCE: {current_url}")
            print(f"==========================================")
            
            try:
                driver.get(current_url)
                time.sleep(5) 

                page_count = 1
                while page_count <= MAX_PAGES:
                    print(f"--- Processing Page {page_count} ---")
                    
                    elements = driver.find_elements(By.TAG_NAME, "a")
                    initial_count = len(all_pdf_links)
                    for elem in elements:
                        try:
                            href = elem.get_attribute("href")
                            if href and href.lower().endswith(".pdf"):
                                all_pdf_links.add(href)
                        except:
                            pass
                    
                    new_found = len(all_pdf_links) - initial_count
                    print(f"Found {new_found} new PDFs. (Total Unique: {len(all_pdf_links)})")

                    # Pagination Logic
                    try:
                        next_btn = None
                        # Strategy 1: BNM Specific
                        try:
                            next_btn = driver.find_element(By.CSS_SELECTOR, "li.next:not(.disabled) a")
                        except:
                            # Strategy 2: Generic Text Search (Works for IIFA)
                            try:
                                links = driver.find_elements(By.XPATH, "//a[contains(text(), 'Next') or contains(text(), 'next') or contains(text(), '>')]")
                                for link in links:
                                    if link.is_displayed():
                                        next_btn = link
                                        break
                            except:
                                pass

                        if next_btn:
                            driver.execute_script("arguments[0].click();", next_btn)
                            time.sleep(5) 
                            page_count += 1
                        else:
                            print(f"✅ Finished category. No more pages.")
                            break 
                            
                    except Exception as e:
                        print(f"Pagination stopped: {e}")
                        break
            except Exception as e:
                print(f"Error loading URL {current_url}: {e}")
                continue
        
        print(f"\n📝 Generating {CSV_PATH} map...")
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'url']) 
            for link in all_pdf_links:
                filename = link.split("/")[-1]
                writer.writerow([filename, link])

        print(f"\n⬇️  Starting Bulk Download ({len(all_pdf_links)} Files)...")
        new_downloads = 0
        for i, link in enumerate(all_pdf_links):
            filename = link.split("/")[-1]
            local_path = os.path.join(DOWNLOAD_DIR, filename)
            
            if os.path.exists(local_path):
                continue 
                
            print(f"[{i+1}/{len(all_pdf_links)}] Downloading: {filename}")
            try:
                driver.get(link)
                time.sleep(3)
                new_downloads += 1
            except:
                pass
        
        return new_downloads

    finally:
        driver.quit()

def ingest_to_db(force_update=False):
    print("\n🧠 Starting Knowledge Base Update...")
    pdf_files = glob.glob(os.path.join(DOWNLOAD_DIR, "*.pdf"))
    
    if not pdf_files:
        print("No PDFs found.")
        return

    if os.path.exists(DB_PATH):
        print("Cleaning old database indices...")
        shutil.rmtree(DB_PATH)

    print(f"Indexing {len(pdf_files)} documents...")
    all_documents = []
    
    url_map = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'filename' in row and 'url' in row:
                    url_map[row['filename']] = row['url']

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            filename = os.path.basename(pdf_path)
            file_url = url_map.get(filename, "#")
            
            for doc in docs:
                doc.metadata["source"] = filename
                doc.metadata["url"] = file_url
                
            all_documents.extend(docs)
        except:
            print(f"Skipping corrupt file: {os.path.basename(pdf_path)}")

    if all_documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_documents)

        print("⚡ Saving vectors to ChromaDB...")
        embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        Chroma.from_documents(documents=splits, embedding=embedding_fn, persist_directory=DB_PATH)
        print("🎉 Success! AI Brain is fully updated.")

def run_pipeline():
    print(f"\n⏰ Scheduled Job Triggered: {time.ctime()}")
    new_files = crawl_and_download()
    if new_files > 0 or not os.path.exists(DB_PATH):
        ingest_to_db()
    else:
        print("✅ No new files. Database is already up to date.")

if __name__ == "__main__":
    print("==================================================")
    print("   SHARIAH INTELLIGENCE AUTOMATION SERVER")
    print("==================================================")
    run_pipeline()
    job = schedule.every(30).days.do(run_pipeline)
    print(f"📅 Next update scheduled at: {job.next_run}")
    print("💤 Going to sleep. Process running in background...")
    while True:
        schedule.run_pending()
        time.sleep(1)