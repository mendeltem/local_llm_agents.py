"""
PDF to Training Text Generator
Konvertiert wissenschaftliche PDFs in verständlichen Trainingstext für Continued Pre-Training
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

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration"""
    CONNECTION_FILE = "server_connection.json"
    PDF_DIR = "/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/Courses/LOCAL_LLM/papers"
    OUTPUT_DIR = "training_data/text"
    LOGS_DIR = "logs"
    
    # Network
    SERVER_PORT = 54321
    SOCKET_TIMEOUT = 600
    BUFFER_SIZE = 10 * 1024 * 1024
    
    # Generation modes
    MODE_SUMMARY = "summary"      # Zusammenfassung
    MODE_EXPLAIN = "explain"      # Verständliche Erklärung
    MODE_FULLTEXT = "fulltext"    # Volltext mit Verbesserungen
    MODE_STRUCTURED = "structured" # Strukturierter Lerntext
    
    # UI
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    Path(Config.LOGS_DIR).mkdir(exist_ok=True)
    log_file = Path(Config.LOGS_DIR) / f"text_gen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
            
            # Remove references section
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
        
        except Exception as e:
            logger.error(f"Request error: {e}")
            return False, str(e)

# ============================================================================
# TEXT GENERATOR
# ============================================================================

class TextGenerator:
    """Generate training text from papers"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def generate_summary(self, text: str, title: str) -> str:
        """Generate concise summary"""
        text = text[:12000]
        
        prompt = f"""Summarize this medical research paper in clear, educational text suitable for training a medical AI.

Paper: {title}

Content:
{text}

Write a comprehensive summary covering:
- Study purpose and background
- Methods and patient population
- Key findings and results
- Clinical implications

Write in clear, factual prose. No bullet points. 2-3 paragraphs."""

        success, response = self.llm.send_request(prompt)
        return response if success else ""
    
    def generate_explanation(self, text: str, title: str) -> str:
        """Generate detailed explanation"""
        text = text[:12000]
        
        prompt = f"""Transform this medical research paper into clear, educational text that explains the study comprehensively.

Paper: {title}

Content:
{text}

Write a detailed explanation covering:
1. Background: What is the medical problem? Why is this study important?
2. Methods: How was the study conducted? What measurements were used?
3. Results: What did the researchers find? What are the key numbers and statistics?
4. Discussion: What do these findings mean? How do they impact clinical practice?

Write in clear, flowing prose suitable for training a medical AI. Use technical terms but explain them. No bullet points or lists - write in natural paragraphs."""

        success, response = self.llm.send_request(prompt)
        return response if success else ""
    
    def generate_structured_text(self, text: str, title: str) -> str:
        """Generate structured learning text"""
        text = text[:12000]
        
        prompt = f"""Convert this medical research paper into well-structured educational text for training a medical AI model.

Paper: {title}

Content:
{text}

Create a comprehensive text covering:

BACKGROUND & CONTEXT:
- What medical condition or problem does this address?
- What was previously known?
- What gap does this study fill?

STUDY DESIGN & METHODS:
- Study population and inclusion criteria
- Measurement tools and imaging protocols
- Statistical analysis methods
- Key definitions and classifications

KEY FINDINGS:
- Primary outcomes with specific numbers
- Secondary findings
- Statistical significance and effect sizes

CLINICAL IMPLICATIONS:
- What do these findings mean for patient care?
- How should this inform medical practice?
- What are the limitations?

Write in clear, educational prose. Explain technical terms. Use specific numbers and statistics from the paper. Write naturally - no bullet points or lists."""

        success, response = self.llm.send_request(prompt)
        return response if success else ""
    
    def enhance_fulltext(self, text: str, title: str) -> str:
        """Enhance fulltext with better structure"""
        text = text[:12000]
        
        prompt = f"""Rewrite this medical research paper in clear, educational prose suitable for training a medical AI.

Paper: {title}

Original text:
{text}

Rewrite the paper:
- Keep all important facts, numbers, and findings
- Improve clarity and flow
- Explain technical terms when first used
- Use natural, flowing paragraphs
- Remove citations and references
- Make it educational and comprehensive

Write the full rewritten paper:"""

        success, response = self.llm.send_request(prompt)
        return response if success else ""

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class PDFtoTextApp:
    """Main Application"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.llm_client = LLMClient()
        self.text_generator = TextGenerator(self.llm_client)
        self.pdf_processor = PDFProcessor()
        
        # Thread-safe queue
        self.log_queue = queue.Queue()
        
        # State
        self.is_processing = False
        self.total_papers = 0
        self.processed_papers = 0
        self.total_chars = 0
        
        self._setup_ui()
        self._scan_papers()
        self._check_server_connection()
        
        self._process_queue()
    
    def _check_server_connection(self):
        """Check server connection"""
        if not self.llm_client.server_info:
            messagebox.showwarning(
                "Server Warnung",
                f"⚠️ Keine Server-Verbindung!\n\n"
                f"Bitte stelle sicher, dass server.py läuft."
            )
        else:
            host = self.llm_client.server_info.get("host")
            model = self.llm_client.server_info.get("model_size", "unknown")
            self._queue_log(f"✅ Server: {host} ({model.upper()})")
    
    def _setup_ui(self):
        """Setup UI"""
        self.root.title("PDF → Training Text Generator")
        self.root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")
        
        # === HEADER ===
        header_frame = tk.Frame(self.root, bg="#2196F3", height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame, 
            text="📄 → 📚 PDF to Training Text Generator",
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
        
        # Mode selection
        tk.Label(settings_frame, text="Modus:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.mode_var = tk.StringVar(value=Config.MODE_STRUCTURED)
        
        modes = [
            ("📝 Zusammenfassung (kurz)", Config.MODE_SUMMARY),
            ("💬 Verständliche Erklärung (mittel)", Config.MODE_EXPLAIN),
            ("📚 Strukturierter Lerntext (lang)", Config.MODE_STRUCTURED),
            ("📄 Volltext verbessert (sehr lang)", Config.MODE_FULLTEXT)
        ]
        
        for i, (label, value) in enumerate(modes):
            tk.Radiobutton(
                settings_frame, 
                text=label, 
                variable=self.mode_var,
                value=value,
                font=("Arial", 10)
            ).grid(row=i, column=1, padx=10, pady=2, sticky="w")
        
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
            command=self._open_output,
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
        """Update server status"""
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
        """Start generation"""
        if self.is_processing:
            messagebox.showinfo("Info", "Generation läuft bereits!")
            return
        
        if not self.llm_client.server_info:
            messagebox.showerror("Fehler", "❌ Keine Server-Verbindung!")
            return
        
        selected_indices = self.papers_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warnung", "Bitte wählen Sie Papers aus!")
            return
        
        selected_papers = [self.pdf_files[i] for i in selected_indices]
        
        self.is_processing = True
        self.start_btn.config(state=tk.DISABLED, text="⏳ Generierung läuft...")
        self.processed_papers = 0
        self.total_chars = 0
        
        thread = threading.Thread(
            target=self._process_papers,
            args=(selected_papers,),
            daemon=True
        )
        thread.start()
    
    def _process_papers(self, papers: List[Path]):
        """Process papers"""
        Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = self.mode_var.get()
        output_file = Path(Config.OUTPUT_DIR) / f"training_text_{mode}_{timestamp}.txt"
        
        total = len(papers)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Medical Research Training Corpus\n")
            f.write(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Mode: {mode}\n")
            f.write(f"# Papers: {total}\n\n")
            f.write("="*80 + "\n\n")
            
            for i, paper_path in enumerate(papers):
                self._queue_log(f"\n{'='*60}")
                self._queue_log(f"📄 Paper {i+1}/{total}: {paper_path.name}")
                
                success, text, metadata = self.pdf_processor.extract_text(str(paper_path))
                
                if not success:
                    self._queue_log(f"❌ Fehler beim Extrahieren", "error")
                    continue
                
                self._queue_log(f"✅ Extrahiert: {metadata['pages']} Seiten, {metadata['chars']} Zeichen")
                self._queue_log(f"🤖 Generiere {mode}-Text...")
                
                # Generate text based on mode
                if mode == Config.MODE_SUMMARY:
                    generated = self.text_generator.generate_summary(text, metadata['title'])
                elif mode == Config.MODE_EXPLAIN:
                    generated = self.text_generator.generate_explanation(text, metadata['title'])
                elif mode == Config.MODE_STRUCTURED:
                    generated = self.text_generator.generate_structured_text(text, metadata['title'])
                else:  # FULLTEXT
                    generated = self.text_generator.enhance_fulltext(text, metadata['title'])
                
                if generated:
                    # Write to file
                    f.write(f"## Paper {i+1}: {metadata['title']}\n\n")
                    f.write(generated)
                    f.write("\n\n" + "="*80 + "\n\n")
                    
                    self.total_chars += len(generated)
                    self._queue_log(f"✅ Generiert: {len(generated)} Zeichen")
                else:
                    self._queue_log(f"⚠️ Keine Ausgabe generiert", "warning")
                
                self.processed_papers += 1
                progress = (self.processed_papers / total) * 100
                self.log_queue.put(("progress", progress))
                
                time.sleep(2)
        
        self.log_queue.put(("complete", (output_file, total)))
    
    def _queue_log(self, message: str, level: str = "info"):
        """Queue log message"""
        self.log_queue.put(("log", (message, level)))
        logger.info(message)
    
    def _process_queue(self):
        """Process queue"""
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
        
        self.root.after(100, self._process_queue)
    
    def _log_direct(self, message: str, level: str = "info"):
        """Log to display"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        prefix = "❌" if level == "error" else ("⚠️" if level == "warning" else "ℹ️")
        
        self.log_display.config(state=tk.NORMAL)
        self.log_display.insert(tk.END, f"[{timestamp}] {prefix} {message}\n")
        self.log_display.see(tk.END)
        self.log_display.config(state=tk.DISABLED)
    
    def _update_progress(self, value: float):
        """Update progress"""
        self.progress_var.set(value)
        kb = self.total_chars / 1024
        self.status_bar.config(
            text=f"Fortschritt: {self.processed_papers} / {self.total_papers} Papers | {kb:.1f} KB generiert"
        )
    
    def _generation_complete(self, output_file: Path, total: int):
        """Complete"""
        self.is_processing = False
        self.start_btn.config(state=tk.NORMAL, text="🚀 GENERIERUNG STARTEN")
        
        self._queue_log(f"\n{'='*60}")
        self._queue_log(f"✅ FERTIG!")
        self._queue_log(f"Papers: {self.processed_papers} / {total}")
        self._queue_log(f"Zeichen: {self.total_chars:,}")
        self._queue_log(f"Output: {output_file}")
        
        mb = self.total_chars / (1024 * 1024)
        
        messagebox.showinfo(
            "Fertig!",
            f"✅ Text-Generierung abgeschlossen!\n\n"
            f"Papers: {self.processed_papers}\n"
            f"Größe: {mb:.2f} MB\n\n"
            f"Output: {output_file.name}"
        )
    
    def _open_output(self):
        """Open output"""
        output_path = Path(Config.OUTPUT_DIR).absolute()
        
        if not output_path.exists():
            messagebox.showinfo("Info", f"Noch kein Output:\n{output_path}")
            return
        
        try:
            import subprocess
            import platform
            
            system = platform.system()
            if system == "Windows":
                subprocess.Popen(f'explorer "{output_path}"')
            elif system == "Darwin":
                subprocess.Popen(["open", str(output_path)])
            else:
                for cmd in ["xdg-open", "nautilus"]:
                    try:
                        subprocess.Popen([cmd, str(output_path)])
                        break
                    except FileNotFoundError:
                        continue
                else:
                    messagebox.showinfo("Pfad", str(output_path))
        except:
            messagebox.showinfo("Pfad", str(output_path))

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFtoTextApp(root)
    root.mainloop()