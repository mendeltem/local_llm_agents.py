"""
PDF to Q&A Training Data Generator - OPTIMIZED VERSION
Automatische Konvertierung von wissenschaftlichen PDFs in Trainingsformat
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import socket
import json
import fitz
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import datetime
import time
import threading
import queue
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration"""
    CONNECTION_FILE = "server_connection.json"
    PDF_DIR = "/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/Courses/LOCAL_LLM/papers"
    OUTPUT_DIR = "training_data"
    LOGS_DIR = "logs"
    
    # Network
    SERVER_PORT = 54321
    SOCKET_TIMEOUT = 600
    BUFFER_SIZE = 10 * 1024 * 1024
    
    # Generation
    QA_PER_PAPER = 20
    BATCH_SIZE = 5
    
    # UI
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    Path(Config.LOGS_DIR).mkdir(exist_ok=True)
    log_file = Path(Config.LOGS_DIR) / f"qa_gen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# PDF PROCESSOR
# ============================================================================

class PDFProcessor:
    """Extract and process PDF content"""
    
    @staticmethod
    def extract_text(pdf_path: str) -> Tuple[bool, str, Dict]:
        """Extract text and metadata from PDF"""
        try:
            doc = fitz.open(pdf_path)
            
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + "\n"
            
            # Remove references
            for keyword in ["References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY"]:
                if keyword in full_text:
                    full_text = full_text.split(keyword)[0]
                    break
            
            metadata = {
                "filename": Path(pdf_path).name,
                "pages": len(doc),
                "chars": len(full_text),
                "title": doc.metadata.get("title", Path(pdf_path).stem)
            }
            
            doc.close()
            return True, full_text, metadata
        
        except Exception as e:
            logger.error(f"Error extracting {pdf_path}: {e}")
            return False, "", {}

# ============================================================================
# LLM CLIENT
# ============================================================================

class LLMClient:
    """Communicate with LLM server"""
    
    def __init__(self):
        self.server_info = self._load_server_info()
        self._verify_connection()
    
    def _load_server_info(self) -> Dict:
        """Load server connection info"""
        try:
            if not Path(Config.CONNECTION_FILE).exists():
                logger.error(f"❌ Connection file not found: {Config.CONNECTION_FILE}")
                logger.info("💡 Make sure server.py is running first!")
                return {}
            
            with open(Config.CONNECTION_FILE, "r") as f:
                info = json.load(f)
                logger.info(f"Loaded server info: {info.get('host')}:{info.get('port')}")
                return info
        except Exception as e:
            logger.error(f"Could not load server info: {e}")
            return {}
    
    def _verify_connection(self):
        """Verify we can reach the server"""
        if not self.server_info:
            logger.error("❌ No server info available!")
            return
        
        host = self.server_info.get("host")
        port = self.server_info.get("port", Config.SERVER_PORT)
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                logger.info(f"✅ Server reachable at {host}:{port}")
            else:
                logger.warning(f"⚠️ Server not reachable at {host}:{port}")
                logger.info(f"💡 Check if server is running on {host}")
        
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
    
    def send_request(self, prompt: str) -> Tuple[bool, str]:
        """Send request to LLM server"""
        if not self.server_info:
            return False, "No server connection info available"
        
        host = self.server_info.get("host")
        port = self.server_info.get("port", Config.SERVER_PORT)
        
        if not host:
            return False, "No hostname in server info"
        
        try:
            logger.info(f"Connecting to {host}:{port}...")
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(Config.SOCKET_TIMEOUT)
            sock.connect((host, port))
            
            logger.info("Connected! Sending prompt...")
            sock.sendall(prompt.encode('utf-8'))
            
            logger.info("Receiving response...")
            response_bytes = b""
            while True:
                chunk = sock.recv(Config.BUFFER_SIZE)
                if not chunk:
                    break
                response_bytes += chunk
            
            sock.close()
            
            response = response_bytes.decode('utf-8').strip()
            logger.info(f"✅ Response received: {len(response)} chars")
            return True, response
        
        except socket.timeout:
            logger.error("⏱️ Request timeout - server is slow or stuck")
            return False, "Timeout"
        
        except ConnectionRefusedError:
            logger.error(f"❌ Connection refused to {host}:{port}")
            logger.info(f"💡 Is the server running on {host}?")
            return False, f"Connection refused to {host}:{port}"
        
        except Exception as e:
            logger.error(f"Request error: {e}")
            return False, str(e)

# ============================================================================
# Q&A GENERATOR - IMPROVED
# ============================================================================

class QAGenerator:
    """Generate Q&A pairs from papers with improved JSON parsing"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def generate_qa_pairs(self, text: str, paper_title: str, num_pairs: int = 20) -> List[Dict]:
        """Generate Q&A pairs from paper text with improved prompt"""
        
        text = text[:12000]
        
        # IMPROVED PROMPT - Forces proper JSON output with one-shot example
        prompt = f"""You are a medical research expert. Generate {num_pairs} question-answer pairs from this paper.

CRITICAL: Output ONLY a valid JSON array. No explanations, no markdown, no preamble.

Paper Title: {paper_title}

Paper Content:
{text}

REQUIRED FORMAT (copy this structure exactly):
[
  {{"instruction": "What was the study population?", "output": "431 first-ever ischemic stroke patients from PROSCIS cohort."}},
  {{"instruction": "What was the primary outcome measured?", "output": "Cognitive impairment 3 years post-stroke using TICS-M score."}},
  {{"instruction": "What scoring system assessed WMH burden?", "output": "Age-Related White Matter Changes (ARWMC) score ranging 0-30."}}
]

RULES:
- Start with [ and end with ]
- Each entry must have "instruction" and "output" keys
- Questions and answers in English
- Based strictly on paper content
- Include methods, results, and background questions

Generate exactly {num_pairs} Q&A pairs (JSON array only):"""

        success, response = self.llm.send_request(prompt)
        
        if not success:
            logger.error(f"Generation failed: {response}")
            return []
        
        # Parse response with multiple fallback strategies
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> List[Dict]:
        """Parse LLM response with multiple strategies"""
        
        # Strategy 1: Direct JSON parsing
        try:
            response = response.strip()
            
            # Remove markdown fences
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            # Extract JSON array
            if "[" in response and "]" in response:
                start = response.index("[")
                end = response.rindex("]") + 1
                json_str = response[start:end]
                
                qa_pairs = json.loads(json_str)
                
                if isinstance(qa_pairs, list):
                    valid_pairs = []
                    for qa in qa_pairs:
                        if isinstance(qa, dict) and "instruction" in qa and "output" in qa:
                            valid_pairs.append(qa)
                    
                    if valid_pairs:
                        logger.info(f"✅ Parsed {len(valid_pairs)} Q&A pairs (Strategy 1: Direct JSON)")
                        return valid_pairs
        
        except json.JSONDecodeError as e:
            logger.warning(f"Strategy 1 failed: {e}")
        
        except Exception as e:
            logger.warning(f"Strategy 1 error: {e}")
        
        # Strategy 2: Line-by-line JSON object extraction
        try:
            qa_pairs = self._extract_json_objects(response)
            if qa_pairs:
                logger.info(f"✅ Parsed {len(qa_pairs)} Q&A pairs (Strategy 2: Line extraction)")
                return qa_pairs
        except Exception as e:
            logger.warning(f"Strategy 2 error: {e}")
        
        # Strategy 3: Regex pattern matching
        try:
            qa_pairs = self._regex_extract(response)
            if qa_pairs:
                logger.info(f"✅ Parsed {len(qa_pairs)} Q&A pairs (Strategy 3: Regex salvage)")
                return qa_pairs
        except Exception as e:
            logger.warning(f"Strategy 3 error: {e}")
        
        logger.error("All parsing strategies failed")
        logger.debug(f"Response was: {response[:1000]}")
        return []
    
    def _extract_json_objects(self, text: str) -> List[Dict]:
        """Extract JSON objects line by line"""
        qa_pairs = []
        
        # Find all {...} patterns
        brace_level = 0
        current_obj = ""
        
        for char in text:
            if char == '{':
                brace_level += 1
                current_obj += char
            elif char == '}':
                current_obj += char
                brace_level -= 1
                
                if brace_level == 0 and current_obj.strip():
                    try:
                        obj = json.loads(current_obj.strip())
                        if isinstance(obj, dict) and "instruction" in obj and "output" in obj:
                            qa_pairs.append(obj)
                    except:
                        pass
                    current_obj = ""
            elif brace_level > 0:
                current_obj += char
        
        return qa_pairs
    
    def _regex_extract(self, text: str) -> List[Dict]:
        """Extract Q&A pairs using regex patterns"""
        qa_pairs = []
        
        # Pattern 1: Standard JSON format
        pattern1 = r'\{\s*"instruction"\s*:\s*"([^"]+)"\s*,\s*"output"\s*:\s*"([^"]+)"\s*\}'
        matches1 = re.findall(pattern1, text, re.DOTALL)
        
        for inst, out in matches1:
            qa_pairs.append({
                "instruction": inst.strip(),
                "output": out.strip()
            })
        
        # Pattern 2: With escaped quotes
        if not qa_pairs:
            pattern2 = r'"instruction"\s*:\s*"((?:[^"\\]|\\.)+)"\s*,\s*"output"\s*:\s*"((?:[^"\\]|\\.)+)"'
            matches2 = re.findall(pattern2, text, re.DOTALL)
            
            for inst, out in matches2:
                qa_pairs.append({
                    "instruction": inst.strip(),
                    "output": out.strip()
                })
        
        return qa_pairs

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class PDFtoQAApp:
    """Main Application"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.llm_client = LLMClient()
        self.qa_generator = QAGenerator(self.llm_client)
        self.pdf_processor = PDFProcessor()
        
        # Thread-safe queue for GUI updates
        self.log_queue = queue.Queue()
        
        # State
        self.is_processing = False
        self.total_papers = 0
        self.processed_papers = 0
        self.total_qa_pairs = 0
        
        self._setup_ui()
        self._scan_papers()
        self._check_server_connection()
        
        # Start queue processor
        self._process_queue()
    
    def _check_server_connection(self):
        """Check server connection status"""
        if not self.llm_client.server_info:
            messagebox.showwarning(
                "Server Warnung",
                f"⚠️ Keine Server-Verbindungsinformationen gefunden!\n\n"
                f"Bitte stelle sicher, dass:\n"
                f"1. server.py läuft\n"
                f"2. {Config.CONNECTION_FILE} existiert\n\n"
                f"Aktueller Pfad: {Path(Config.CONNECTION_FILE).absolute()}"
            )
        else:
            host = self.llm_client.server_info.get("host")
            model = self.llm_client.server_info.get("model_size", "unknown")
            self._queue_log(f"✅ Server gefunden: {host} ({model.upper()})")
    
    def _setup_ui(self):
        """Setup UI"""
        self.root.title("PDF → Q&A Training Data Generator")
        self.root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")
        
        # === HEADER ===
        header_frame = tk.Frame(self.root, bg="#2196F3", height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame, 
            text="📄 → 💬 PDF to Q&A Training Data Generator",
            bg="#2196F3", fg="white",
            font=("Arial", 16, "bold")
        ).pack(pady=15)
        
        # === INFO FRAME ===
        info_frame = tk.Frame(self.root, bg="#f5f5f5")
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.papers_label = tk.Label(
            info_frame, text="Papers gefunden: 0",
            bg="#f5f5f5", font=("Arial", 11)
        )
        self.papers_label.pack(side=tk.LEFT, padx=10)
        
        self.server_status_label = tk.Label(
            info_frame, text="Server: ?",
            bg="#f5f5f5", font=("Arial", 11), fg="orange"
        )
        self.server_status_label.pack(side=tk.RIGHT, padx=10)
        
        # === SETTINGS FRAME ===
        settings_frame = tk.LabelFrame(self.root, text="Einstellungen", font=("Arial", 11, "bold"))
        settings_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(settings_frame, text="Q&A-Paare pro Paper:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.qa_count_var = tk.IntVar(value=Config.QA_PER_PAPER)
        tk.Spinbox(
            settings_frame, from_=5, to=50, textvariable=self.qa_count_var,
            width=10, font=("Arial", 10)
        ).grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        # === PAPERS LIST ===
        list_frame = tk.LabelFrame(self.root, text="Papers", font=("Arial", 11, "bold"))
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.papers_listbox = tk.Listbox(
            list_frame, font=("Consolas", 10),
            selectmode=tk.MULTIPLE
        )
        self.papers_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = tk.Scrollbar(list_frame, command=self.papers_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.papers_listbox.config(yscrollcommand=scrollbar.set)
        
        # === LOG ===
        log_frame = tk.LabelFrame(self.root, text="Log", font=("Arial", 11, "bold"))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.log_display = scrolledtext.ScrolledText(
            log_frame, height=10, font=("Consolas", 9),
            state='disabled', bg="#1e1e1e", fg="#ffffff"
        )
        self.log_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === PROGRESS ===
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(
            self.root, variable=self.progress_var,
            maximum=100, mode='determinate'
        )
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        
        # === BUTTONS ===
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_btn = tk.Button(
            button_frame, text="🚀 GENERIERUNG STARTEN",
            command=self._start_generation,
            bg="#4CAF50", fg="white",
            font=("Arial", 12, "bold"),
            height=2
        )
        self.start_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        tk.Button(
            button_frame, text="🔄 Papers aktualisieren",
            command=self._scan_papers,
            bg="#2196F3", fg="white",
            font=("Arial", 11)
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            button_frame, text="📂 Output öffnen",
            command=self._open_output_safe,
            bg="#FF9800", fg="white",
            font=("Arial", 11)
        ).pack(side=tk.LEFT, padx=5)
        
        # === STATUS BAR ===
        self.status_bar = tk.Label(
            self.root, text="Bereit",
            relief=tk.SUNKEN, anchor=tk.W,
            font=("Arial", 9), bg="#e0e0e0"
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self._update_server_status()
    
    def _update_server_status(self):
        """Update server status label"""
        if self.llm_client.server_info:
            host = self.llm_client.server_info.get("host", "unknown")
            model = self.llm_client.server_info.get("model_size", "unknown")
            self.server_status_label.config(
                text=f"Server: {host} ({model.upper()})",
                fg="green"
            )
        else:
            self.server_status_label.config(
                text="Server: NICHT VERBUNDEN",
                fg="red"
            )
    
    def _scan_papers(self):
        """Scan PDF directory"""
        pdf_dir = Path(Config.PDF_DIR)
        
        if not pdf_dir.exists():
            self._queue_log(f"❌ Verzeichnis nicht gefunden: {Config.PDF_DIR}", "error")
            return
        
        self.papers_listbox.delete(0, tk.END)
        self.pdf_files = list(pdf_dir.glob("*.pdf"))
        self.total_papers = len(self.pdf_files)
        
        for pdf_file in self.pdf_files:
            self.papers_listbox.insert(tk.END, pdf_file.name)
        
        self.papers_listbox.select_set(0, tk.END)
        
        self.papers_label.config(text=f"Papers gefunden: {self.total_papers}")
        self._queue_log(f"✅ {self.total_papers} PDF-Dateien gefunden")
    
    def _start_generation(self):
        """Start Q&A generation"""
        if self.is_processing:
            messagebox.showinfo("Info", "Generation läuft bereits!")
            return
        
        if not self.llm_client.server_info:
            messagebox.showerror(
                "Fehler",
                "❌ Keine Server-Verbindung!\n\nBitte starte zuerst server.py auf dem GPU-Node"
            )
            return
        
        selected_indices = self.papers_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warnung", "Bitte wählen Sie Papers aus!")
            return
        
        selected_papers = [self.pdf_files[i] for i in selected_indices]
        
        self.is_processing = True
        self.start_btn.config(state=tk.DISABLED, text="⏳ Generierung läuft...")
        self.processed_papers = 0
        self.total_qa_pairs = 0
        
        thread = threading.Thread(
            target=self._process_papers,
            args=(selected_papers,),
            daemon=True
        )
        thread.start()
    
    def _process_papers(self, papers: List[Path]):
        """Process papers and generate Q&A - runs in background thread"""
        Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(Config.OUTPUT_DIR) / f"training_data_{timestamp}.jsonl"
        
        total = len(papers)
        
        with open(output_file, "w", encoding="utf-8") as f:
            for i, paper_path in enumerate(papers):
                self._queue_log(f"\n{'='*60}")
                self._queue_log(f"📄 Paper {i+1}/{total}: {paper_path.name}")
                
                success, text, metadata = self.pdf_processor.extract_text(str(paper_path))
                
                if not success:
                    self._queue_log(f"❌ Fehler beim Extrahieren", "error")
                    continue
                
                self._queue_log(f"✅ Extrahiert: {metadata['pages']} Seiten, {metadata['chars']} Zeichen")
                
                qa_count = self.qa_count_var.get()
                self._queue_log(f"🤖 Generiere {qa_count} Q&A-Paare...")
                
                qa_pairs = self.qa_generator.generate_qa_pairs(
                    text, 
                    metadata['title'],
                    qa_count
                )
                
                if qa_pairs:
                    for qa in qa_pairs:
                        f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                        self.total_qa_pairs += 1
                    
                    self._queue_log(f"✅ {len(qa_pairs)} Q&A-Paare generiert")
                else:
                    self._queue_log(f"⚠️ Keine Q&A-Paare generiert", "warning")
                
                self.processed_papers += 1
                progress = (self.processed_papers / total) * 100
                self.log_queue.put(("progress", progress))
                
                time.sleep(2)
        
        self.log_queue.put(("complete", (output_file, total)))
    
    def _queue_log(self, message: str, level: str = "info"):
        """Queue log message for main thread"""
        self.log_queue.put(("log", (message, level)))
        logger.info(message)
    
    def _process_queue(self):
        """Process queued messages in main thread"""
        try:
            while True:
                msg_type, data = self.log_queue.get_nowait()
                
                if msg_type == "log":
                    message, level = data
                    self._log_direct(message, level)
                
                elif msg_type == "progress":
                    self._update_progress(data)
                
                elif msg_type == "complete":
                    output_file, total = data
                    self._generation_complete(output_file, total)
        
        except queue.Empty:
            pass
        
        # Reschedule
        self.root.after(100, self._process_queue)
    
    def _log_direct(self, message: str, level: str = "info"):
        """Log directly to display (called from main thread only)"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        if level == "error":
            prefix = "❌"
        elif level == "warning":
            prefix = "⚠️"
        else:
            prefix = "ℹ️"
        
        self.log_display.config(state=tk.NORMAL)
        self.log_display.insert(tk.END, f"[{timestamp}] {prefix} {message}\n")
        self.log_display.see(tk.END)
        self.log_display.config(state=tk.DISABLED)
    
    def _update_progress(self, value: float):
        """Update progress bar"""
        self.progress_var.set(value)
        self.status_bar.config(
            text=f"Fortschritt: {self.processed_papers} / {self.total_papers} Papers | {self.total_qa_pairs} Q&A-Paare"
        )
    
    def _generation_complete(self, output_file: Path, total: int):
        """Called when generation is complete"""
        self.is_processing = False
        self.start_btn.config(state=tk.NORMAL, text="🚀 GENERIERUNG STARTEN")
        
        self._queue_log(f"\n{'='*60}")
        self._queue_log(f"✅ FERTIG!")
        self._queue_log(f"Papers verarbeitet: {self.processed_papers} / {total}")
        self._queue_log(f"Q&A-Paare generiert: {self.total_qa_pairs}")
        self._queue_log(f"Output: {output_file}")
        
        messagebox.showinfo(
            "Fertig!",
            f"✅ Generierung abgeschlossen!\n\n"
            f"Papers: {self.processed_papers}\n"
            f"Q&A-Paare: {self.total_qa_pairs}\n\n"
            f"Durchschnitt: {self.total_qa_pairs / self.processed_papers:.1f} Paare/Paper\n\n"
            f"Output: {output_file}"
        )
    
    def _open_output_safe(self):
        """Open output directory - safe version"""
        output_path = Path(Config.OUTPUT_DIR).absolute()
        
        if not output_path.exists():
            messagebox.showinfo("Info", f"Output-Verzeichnis existiert noch nicht:\n{output_path}")
            return
        
        try:
            import subprocess
            import platform
            
            if platform.system() == "Windows":
                subprocess.Popen(f'explorer "{output_path}"')
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(output_path)])
            else:
                # Linux - try multiple options
                for cmd in ["xdg-open", "nautilus", "dolphin", "thunar"]:
                    try:
                        subprocess.Popen([cmd, str(output_path)])
                        break
                    except FileNotFoundError:
                        continue
                else:
                    messagebox.showinfo("Info", f"Output-Verzeichnis:\n{output_path}")
        
        except Exception as e:
            messagebox.showinfo("Output Path", f"Pfad:\n{output_path}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFtoQAApp(root)
    root.mainloop()