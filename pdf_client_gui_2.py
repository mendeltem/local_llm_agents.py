"""
Improved AI PDF Assistant
- Besseres Error Handling
- Retry-Logik
- Response Streaming
- Session Management
- Input Validation
- Better UX/UI
- Configuration
- Logging
"""

import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk
import socket
import threading
import datetime
import os
import json
import fitz
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Application Configuration"""
    CONNECTION_FILE: str = "server_connection.json"
    CONTEXT_FILE: str = "temp_pdf_context.txt"
    LOGS_DIR: str = "logs"
    SESSIONS_DIR: str = "sessions"
    
    # Network
    SERVER_PORT: int = 54321
    BUFFER_SIZE: int = 4096
    SOCKET_TIMEOUT: int = 300
    RETRY_ATTEMPTS: int = 3
    RETRY_DELAY: float = 1.0
    
    # PDF
    MAX_PDF_SIZE_MB: int = 50
    MAX_CONTEXT_LENGTH: int = 15000
    
    # UI
    WINDOW_WIDTH: int = 1000
    WINDOW_HEIGHT: int = 900
    FONT_FAMILY: str = "Consolas"
    FONT_SIZE: int = 11

class ConnectionStatus(Enum):
    """Connection Status States"""
    CONNECTED = "✅ Verbunden"
    DISCONNECTED = "❌ Getrennt"
    CONNECTING = "⏳ Verbindung..."
    ERROR = "❌ Fehler"

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: Config) -> logging.Logger:
    """Setup Logging System"""
    Path(config.LOGS_DIR).mkdir(exist_ok=True)
    
    log_file = Path(config.LOGS_DIR) / f"app_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
    return logger

# ============================================================================
# SERVER CONNECTION MANAGER
# ============================================================================

class ServerConnectionManager:
    """Manage Connection to LLM Server"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.server_info: Optional[Dict] = None
        self.status = ConnectionStatus.DISCONNECTED
        self._load_server_info()
    
    def _load_server_info(self) -> None:
        """Load server connection info"""
        try:
            if Path(self.config.CONNECTION_FILE).exists():
                with open(self.config.CONNECTION_FILE, "r") as f:
                    self.server_info = json.load(f)
                    self.logger.info(f"Server info loaded: {self.server_info.get('host')}")
            else:
                self.logger.warning(f"Connection file not found: {self.config.CONNECTION_FILE}")
        except Exception as e:
            self.logger.error(f"Error loading server info: {e}")
    
    def is_server_available(self) -> bool:
        """Check if server is running"""
        if not self.server_info:
            return False
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            host = self.server_info.get("host", "localhost")
            result = sock.connect_ex((host, self.config.SERVER_PORT))
            sock.close()
            return result == 0
        except Exception as e:
            self.logger.error(f"Server check failed: {e}")
            return False
    
    def get_server_info(self) -> Optional[Dict]:
        """Get server information"""
        return self.server_info
    
    def refresh(self) -> None:
        """Refresh server info"""
        self._load_server_info()

# ============================================================================
# PDF MANAGER
# ============================================================================

class PDFManager:
    """Manage PDF Loading and Context"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.current_pdf_path: Optional[str] = None
        self.context: Optional[str] = None
    
    def load_pdf(self, file_path: str) -> Tuple[bool, str]:
        """Load PDF and extract text"""
        try:
            # Validate
            path = Path(file_path)
            if not path.exists():
                return False, f"Datei nicht gefunden: {file_path}"
            
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.MAX_PDF_SIZE_MB:
                return False, f"PDF zu groß: {file_size_mb:.1f}MB (Max: {self.config.MAX_PDF_SIZE_MB}MB)"
            
            # Parse PDF
            doc = fitz.open(file_path)
            full_text = ""
            
            for page_num, page in enumerate(doc):
                try:
                    text = page.get_text("text")
                    full_text += text + "\n"
                except Exception as e:
                    self.logger.warning(f"Error extracting page {page_num}: {e}")
            
            # Remove references section
            if "References" in full_text:
                full_text = full_text.split("References")[0]
            
            # Store
            self.current_pdf_path = file_path
            self.context = full_text
            
            # Save to file
            with open(self.config.CONTEXT_FILE, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            msg = f"PDF geladen: {path.name} ({len(doc)} Seiten, {len(full_text)} Zeichen)"
            self.logger.info(msg)
            return True, msg
        
        except Exception as e:
            self.logger.error(f"PDF loading error: {e}")
            return False, f"Fehler beim Laden: {str(e)}"
    
    def clear(self) -> None:
        """Clear context"""
        self.current_pdf_path = None
        self.context = None
        
        if Path(self.config.CONTEXT_FILE).exists():
            try:
                Path(self.config.CONTEXT_FILE).unlink()
            except Exception as e:
                self.logger.warning(f"Could not delete context file: {e}")
    
    def is_loaded(self) -> bool:
        """Check if context is loaded"""
        return self.context is not None
    
    def get_context_snippet(self, max_length: Optional[int] = None) -> str:
        """Get context with optional length limit"""
        if not self.context:
            return ""
        
        max_len = max_length or self.config.MAX_CONTEXT_LENGTH
        return self.context[:max_len]

# ============================================================================
# REQUEST HANDLER WITH RETRY
# ============================================================================

class RequestHandler:
    """Handle LLM Server Communication with Retry Logic"""
    
    def __init__(self, config: Config, logger: logging.Logger, 
                 connection_mgr: ServerConnectionManager):
        self.config = config
        self.logger = logger
        self.connection_mgr = connection_mgr
    
    def send_request(self, question: str, context: Optional[str] = None) -> Tuple[bool, str]:
        """Send request with retry logic"""
        
        # Prepare prompt
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {question}"
        else:
            prompt = question
        
        # Retry loop
        for attempt in range(self.config.RETRY_ATTEMPTS):
            try:
                self.logger.info(f"Attempt {attempt + 1}/{self.config.RETRY_ATTEMPTS}")
                
                # Get server info
                server_info = self.connection_mgr.get_server_info()
                if not server_info:
                    return False, "Keine Server-Verbindungsinformationen verfügbar"
                
                host = server_info.get("host", "localhost")
                
                # Connect
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.config.SOCKET_TIMEOUT)
                sock.connect((host, self.config.SERVER_PORT))
                
                # Send
                sock.sendall(prompt.encode('utf-8'))
                
                # Receive
                response_bytes = b""
                while True:
                    chunk = sock.recv(self.config.BUFFER_SIZE)
                    if not chunk:
                        break
                    response_bytes += chunk
                
                sock.close()
                
                response_text = response_bytes.decode('utf-8').strip()
                self.logger.info("Request successful")
                return True, response_text
            
            except socket.timeout:
                self.logger.warning("Timeout - server responding slowly")
                if attempt < self.config.RETRY_ATTEMPTS - 1:
                    time.sleep(self.config.RETRY_DELAY)
                    continue
                return False, "Zeitüberschreitung - Server antwortet langsam"
            
            except ConnectionRefusedError:
                self.logger.error("Connection refused - server not running")
                return False, "Verbindung abgelehnt - Server läuft nicht"
            
            except Exception as e:
                self.logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt < self.config.RETRY_ATTEMPTS - 1:
                    time.sleep(self.config.RETRY_DELAY)
                    continue
                return False, f"Fehler: {str(e)}"
        
        return False, "Alle Verbindungsversuche fehlgeschlagen"

# ============================================================================
# SESSION MANAGER
# ============================================================================

class SessionManager:
    """Manage Chat Sessions"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        Path(self.config.SESSIONS_DIR).mkdir(exist_ok=True)
        
        self.current_session = {
            "created_at": datetime.datetime.now().isoformat(),
            "messages": []
        }
    
    def add_message(self, role: str, content: str) -> None:
        """Add message to session"""
        self.current_session["messages"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "role": role,
            "content": content
        })
    
    def save_session(self, chat_text: str) -> Tuple[bool, str]:
        """Save session to file"""
        try:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(self.config.SESSIONS_DIR) / f"session_{ts}.txt"
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("=== AI PDF Assistant Session ===\n")
                f.write(f"Created: {self.current_session['created_at']}\n\n")
                f.write(chat_text)
            
            self.logger.info(f"Session saved: {filepath}")
            return True, f"Session gespeichert: {filepath}"
        
        except Exception as e:
            self.logger.error(f"Session save error: {e}")
            return False, f"Fehler beim Speichern: {str(e)}"

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class ImprovedAIPDFAssistant:
    """Improved AI PDF Assistant with Better UX/UI"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.config = Config()
        self.logger = setup_logging(self.config)
        
        # Initialize managers
        self.connection_mgr = ServerConnectionManager(self.config, self.logger)
        self.pdf_mgr = PDFManager(self.config, self.logger)
        self.request_handler = RequestHandler(self.config, self.logger, self.connection_mgr)
        self.session_mgr = SessionManager(self.config, self.logger)
        
        # State
        self.is_processing = False
        self.connection_status = ConnectionStatus.DISCONNECTED
        
        # Setup UI
        self._setup_ui()
        self._check_server_status()
        
        self.logger.info("Application started")
    
    def _setup_ui(self) -> None:
        """Setup User Interface"""
        
        # Window
        self.root.title("AI PDF Assistant - Improved Edition")
        self.root.geometry(f"{self.config.WINDOW_WIDTH}x{self.config.WINDOW_HEIGHT}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # === TOP FRAME ===
        top_frame = tk.Frame(self.root, bg="#f0f0f0", height=50)
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        top_frame.pack_propagate(False)
        
        # Upload Button
        self.upload_btn = tk.Button(
            top_frame, text="📄 PDF Laden",
            command=self._upload_pdf,
            bg="#2196F3", fg="white", font=(self.config.FONT_FAMILY, 10, "bold")
        )
        self.upload_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Context Label
        self.context_label = tk.Label(
            top_frame, text="Kein PDF geladen",
            bg="#f0f0f0", font=(self.config.FONT_FAMILY, 10)
        )
        self.context_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Clear Button
        self.clear_btn = tk.Button(
            top_frame, text="✕ Löschen",
            command=self._clear_context,
            bg="#FF6B6B", fg="white", font=(self.config.FONT_FAMILY, 9)
        )
        self.clear_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # === CHAT DISPLAY ===
        self.chat_display = scrolledtext.ScrolledText(
            self.root, state='disabled',
            font=(self.config.FONT_FAMILY, self.config.FONT_SIZE),
            wrap=tk.WORD, bg="#fafafa"
        )
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Configure tags
        self.chat_display.tag_config("user", foreground="#1976D2", font=(self.config.FONT_FAMILY, self.config.FONT_SIZE, "bold"))
        self.chat_display.tag_config("ai", foreground="#333333")
        self.chat_display.tag_config("system", foreground="#666666", font=(self.config.FONT_FAMILY, self.config.FONT_SIZE, "italic"))
        self.chat_display.tag_config("error", foreground="#D32F2F", font=(self.config.FONT_FAMILY, self.config.FONT_SIZE, "bold"))
        
        # === INPUT FRAME ===
        input_frame = tk.Frame(self.root)
        input_frame.pack(padx=10, pady=5, fill=tk.X)
        
        self.input_field = tk.Text(
            input_frame, height=4,
            font=(self.config.FONT_FAMILY, self.config.FONT_SIZE),
            wrap=tk.WORD, relief=tk.SUNKEN, bd=1
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.input_field.bind("<Control-Return>", lambda e: self._send_message())
        
        # Char counter
        self.char_count_label = tk.Label(input_frame, text="0 / 5000", font=(self.config.FONT_FAMILY, 9))
        self.char_count_label.pack(side=tk.RIGHT, padx=5, pady=5)
        self.input_field.bind("<KeyRelease>", self._update_char_count)
        
        # === BUTTON FRAME ===
        button_frame = tk.Frame(self.root)
        button_frame.pack(padx=10, pady=10, fill=tk.X)
        
        self.send_btn = tk.Button(
            button_frame, text="SENDEN ➤ (Ctrl+Enter)",
            command=self._send_message,
            bg="#4CAF50", fg="white",
            font=(self.config.FONT_FAMILY, 11, "bold")
        )
        self.send_btn.pack(side=tk.RIGHT, padx=5)
        
        self.save_btn = tk.Button(
            button_frame, text="💾 Session Speichern",
            command=self._save_session,
            bg="#FF9800", fg="white",
            font=(self.config.FONT_FAMILY, 10)
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # === PROGRESS BAR ===
        self.progress = ttk.Progressbar(
            self.root, mode='indeterminate',
            length=400
        )
        self.progress.pack(padx=10, pady=5, fill=tk.X)
        
        # === STATUS BAR ===
        self.status_bar = tk.Label(
            self.root, text="Initialisierung...",
            relief=tk.SUNKEN, anchor=tk.W,
            font=(self.config.FONT_FAMILY, 9),
            bg="#e0e0e0", fg="#333"
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _check_server_status(self) -> None:
        """Check server status periodically"""
        available = self.connection_mgr.is_server_available()
        
        server_info = self.connection_mgr.get_server_info()
        if available and server_info:
            model = server_info.get("model_size", "Unknown").upper()
            host = server_info.get("host", "localhost")
            status = f"✅ Server: {host} | Model: {model}"
            self.send_btn.config(state=tk.NORMAL)
        else:
            status = "❌ Server nicht erreichbar"
            self.send_btn.config(state=tk.DISABLED)
        
        self.status_bar.config(text=status)
        
        # Reschedule
        self.root.after(5000, self._check_server_status)
    
    def _update_char_count(self, event=None) -> None:
        """Update character counter"""
        text_length = len(self.input_field.get("1.0", tk.END)) - 1
        self.char_count_label.config(text=f"{text_length} / 5000")
    
    def _append_text(self, text: str, tag: str = "ai") -> None:
        """Append text to chat"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, text + "\n", tag)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def _upload_pdf(self) -> None:
        """Upload PDF"""
        file_path = filedialog.askopenfilename(
            filetypes=[("PDF Files", "*.pdf")],
            title="PDF-Datei auswählen"
        )
        
        if not file_path:
            return
        
        success, message = self.pdf_mgr.load_pdf(file_path)
        
        if success:
            self.context_label.config(text=message, fg="green")
            self._append_text(f"SYSTEM: {message}", "system")
        else:
            self.context_label.config(text=f"Fehler: {message}", fg="red")
            self._append_text(f"FEHLER: {message}", "error")
    
    def _clear_context(self) -> None:
        """Clear PDF context"""
        self.pdf_mgr.clear()
        self.context_label.config(text="Kein PDF geladen", fg="black")
        self._append_text("SYSTEM: Kontext gelöscht", "system")
    
    def _send_message(self) -> None:
        """Send message"""
        question = self.input_field.get("1.0", tk.END).strip()
        
        if not question:
            messagebox.showwarning("Warnung", "Bitte geben Sie eine Frage ein")
            return
        
        if self.is_processing:
            messagebox.showinfo("Info", "Bitte warten Sie auf die aktuelle Anfrage")
            return
        
        if len(question) > 5000:
            messagebox.showwarning("Warnung", "Frage zu lang (max. 5000 Zeichen)")
            return
        
        # Clear input
        self.input_field.delete("1.0", tk.END)
        
        # Display
        self._append_text("-" * 80)
        self._append_text(f"DU: {question}", "user")
        self._append_text("-" * 80)
        
        # Add to session
        self.session_mgr.add_message("user", question)
        
        # Process
        self.is_processing = True
        self.send_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.status_bar.config(text="Verarbeite Anfrage...")
        
        # Send in thread
        thread = threading.Thread(
            target=self._fetch_response,
            args=(question,),
            daemon=True
        )
        thread.start()
    
    def _fetch_response(self, question: str) -> None:
        """Fetch response in background"""
        try:
            context = self.pdf_mgr.get_context_snippet() if self.pdf_mgr.is_loaded() else None
            
            success, response = self.request_handler.send_request(question, context)
            
            if success:
                self.root.after(0, self._display_response, response)
                self.session_mgr.add_message("assistant", response)
            else:
                self.root.after(0, self._display_error, response)
        
        except Exception as e:
            self.logger.error(f"Error in fetch_response: {e}")
            self.root.after(0, self._display_error, f"Unerwarteter Fehler: {str(e)}")
        
        finally:
            self.is_processing = False
            self.root.after(0, self._reset_ui)
    
    def _display_response(self, text: str) -> None:
        """Display response"""
        self._append_text(text, "ai")
    
    def _display_error(self, message: str) -> None:
        """Display error"""
        self._append_text(f"FEHLER: {message}", "error")
    
    def _reset_ui(self) -> None:
        """Reset UI after request"""
        self.progress.stop()
        self.send_btn.config(state=tk.NORMAL)
        self.status_bar.config(text="Bereit")
    
    def _save_session(self) -> None:
        """Save session"""
        chat_text = self.chat_display.get("1.0", tk.END)
        success, message = self.session_mgr.save_session(chat_text)
        
        if success:
            messagebox.showinfo("Erfolg", message)
        else:
            messagebox.showerror("Fehler", message)
    
    def _on_closing(self) -> None:
        """Handle window closing"""
        if messagebox.askokcancel("Beenden", "Anwendung beenden?"):
            self.logger.info("Application closed")
            self.root.destroy()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = ImprovedAIPDFAssistant(root)
    root.mainloop()