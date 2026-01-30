import os
import json
import socket
import fitz  # PyMuPDF
import glob
import time
import sys
import llm_config as cfg

# --- CONFIGURATION ---
PAPER_DIR = "./papers"
OUTPUT_FILE = "live_research_feed.txt"
MEMORY_FILE = "processed_files_log.json" # To remember what we already read
TOPIC = "Ischemic Stroke" # <--- Set your standard topic here
CHECK_INTERVAL = 10 # Seconds between checks

def get_server_connection():
    if not os.path.exists(cfg.CONNECTION_FILE):
        return None
    try:
        with open(cfg.CONNECTION_FILE, "r") as f:
            return json.load(f)
    except:
        return None

def load_memory():
    """Loads the list of files we have already analyzed."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_memory(processed_set):
    """Saves the list so we don't repeat work if we restart."""
    with open(MEMORY_FILE, "w") as f:
        json.dump(list(processed_set), f)

def extract_text(filepath):
    try:
        doc = fitz.open(filepath)
        text = ""
        for i, page in enumerate(doc):
            text += page.get_text("text") + "\n"
            if i >= 2: break 
        if "References" in text: text = text.split("References")[0]
        return text[:12000]
    except: return None

def analyze_paper(filename, text, conn):
    prompt = (
        f"You are a Watchdog Research Agent. Filter for topic: '{TOPIC}'.\n"
        f"Reply 'DECISION: NO' if irrelevant.\n"
        f"Reply 'DECISION: YES' followed by a summary if relevant.\n\n"
        f"--- PAPER TEXT ---\n{text}\n"
    )
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(300)
        client.connect((conn["host"], conn["port"]))
        client.sendall(prompt.encode('utf-8'))
        resp = b""
        while True:
            chunk = client.recv(cfg.BUFFER_SIZE)
            if not chunk: break
            resp += chunk
        client.close()
        return resp.decode('utf-8').strip()
    except Exception as e:
        return f"Error: {e}"

def main():
    if not os.path.exists(PAPER_DIR): os.makedirs(PAPER_DIR)
    
    print(f"👀 WATCHDOG AGENT ACTIVE")
    print(f"🔎 Watching '{PAPER_DIR}' for new papers about '{TOPIC}'...")
    print(f"📝 Updates will appear in '{OUTPUT_FILE}'")
    print("-------------------------------------------------------")

    processed_files = load_memory()

    while True:
        # 1. Check Server Status
        conn = get_server_connection()
        if not conn:
            print("⚠️  Server down. Waiting...", end="\r")
            time.sleep(CHECK_INTERVAL)
            continue

        # 2. Look for NEW files
        current_pdfs = set(glob.glob(os.path.join(PAPER_DIR, "*.pdf")))
        new_files = current_pdfs - processed_files

        if not new_files:
            # Nothing new, just wait
            sys.stdout.write(f"💤 No new files. Sleeping... ({time.ctime()})\r")
            sys.stdout.flush()
        else:
            print(f"\n🔔 WAKE UP! Found {len(new_files)} new file(s)!")
            
            for filepath in new_files:
                filename = os.path.basename(filepath)
                print(f"   ⚙️  Processing: {filename}...", end=" ", flush=True)
                
                text = extract_text(filepath)
                if not text:
                    print("❌ Error reading")
                    processed_files.add(filepath) # Mark as done so we don't retry forever
                    continue

                result = analyze_paper(filename, text, conn)
                
                if "DECISION: YES" in result:
                    print("✅ RELEVANT! Added to report.")
                    clean_summary = result.replace("DECISION: YES", "").strip()
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(f"🔴 NEW ENTRY ({time.ctime()})\n")
                        f.write(f"📄 {filename}\n")
                        f.write(clean_summary + "\n")
                        f.write("-" * 60 + "\n\n")
                else:
                    print("🗑️  Irrelevant.")

                # Remember we did this file
                processed_files.add(filepath)
                save_memory(processed_files)

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()