import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import socket
import threading
import datetime
import os
import json
import fitz  # PyMuPDF
import llm_config as cfg

CONTEXT_FILE = "temp_pdf_context.txt"

class AI_PDF_Assistant:
    def __init__(self, root):
        self.root = root
        
        # Lade Info vom Server (Auto-Connect)
        host_display = "Suche Server..."
        model_display = "Unknown"
        if os.path.exists(cfg.CONNECTION_FILE):
            try:
                with open(cfg.CONNECTION_FILE, "r") as f:
                    data = json.load(f)
                    host_display = data.get("host", "Unknown")
                    model_display = data.get("model_size", "Unknown").upper()
            except: pass
            
        self.root.title(f"AI Assistant [{model_display}] - Connected to {host_display}")
        self.root.geometry("900x850")
        
        # --- GUI ELEMENTE ---
        self.top_frame = tk.Frame(root, bg="#f0f0f0", height=40)
        self.top_frame.pack(fill=tk.X, padx=5, pady=5)

        self.upload_btn = tk.Button(self.top_frame, text="📄 PDF Laden", command=self.upload_pdf, bg="#2196F3", fg="white")
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        self.context_label = tk.Label(self.top_frame, text="Kein PDF", bg="#f0f0f0")
        self.context_label.pack(side=tk.LEFT, padx=10)

        self.clear_ctx_btn = tk.Button(self.top_frame, text="✕", command=self.clear_context)
        self.clear_ctx_btn.pack(side=tk.LEFT)

        self.chat_display = scrolledtext.ScrolledText(root, state='disabled', font=("Consolas", 11))
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_display.tag_config("user", foreground="blue", font=("Consolas", 11, "bold"))
        self.chat_display.tag_config("ai", foreground="black")
        self.chat_display.tag_config("error", foreground="red")

        self.input_frame = tk.Frame(root)
        self.input_frame.pack(padx=10, pady=5, fill=tk.X)
        self.input_field = tk.Text(self.input_frame, height=4, font=("Consolas", 11))
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_field.bind("<Return>", self.on_enter_press)

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(padx=10, pady=10, fill=tk.X)
        self.send_btn = tk.Button(self.btn_frame, text="SENDEN ➤", command=self.send_message, bg="#4CAF50", fg="white")
        self.send_btn.pack(side=tk.RIGHT)
        self.save_btn = tk.Button(self.btn_frame, text="💾 Log Speichern", command=self.save_log)
        self.save_btn.pack(side=tk.LEFT)

        self.status_bar = tk.Label(root, text=f"Modell: {model_display}", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_enter_press(self, event):
        if not event.state & 0x0001: 
            self.send_message()
            return "break"

    def append_text(self, text, tag=None):
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, text + "\n", tag)
        self.chat_display.see(tk.END)
        self.chat_display.configure(state='disabled')

    def upload_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if not file_path: return
        try:
            doc = fitz.open(file_path)
            full_text = ""
            for page in doc: full_text += page.get_text("text") + "\n"
            if "References" in full_text: full_text = full_text.split("References")[0]
            with open(CONTEXT_FILE, "w", encoding="utf-8") as f: f.write(full_text)
            self.context_label.config(text=f"Geladen: {os.path.basename(file_path)}", fg="green")
            self.append_text(f"SYSTEM: PDF geladen ({len(full_text)} Zeichen).", "ai")
        except Exception as e: messagebox.showerror("Error", str(e))

    def clear_context(self):
        if os.path.exists(CONTEXT_FILE): os.remove(CONTEXT_FILE)
        self.context_label.config(text="Kein PDF", fg="black")

    def save_log(self):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        with open(f"log_{ts}.txt", "w") as f: f.write(self.chat_display.get("1.0", tk.END))
        messagebox.showinfo("Gespeichert", "Log gespeichert.")

    def send_message(self):
        question = self.input_field.get("1.0", tk.END).strip()
        if not question: return
        self.input_field.delete("1.0", tk.END)
        self.append_text("-" * 40)
        self.append_text(f"DU: {question}", "user")
        self.append_text("-" * 40)
        self.status_bar.config(text="Sende an GPU...")
        threading.Thread(target=self.fetch_response, args=(question,), daemon=True).start()

    def fetch_response(self, question):
        try:
            if not os.path.exists(cfg.CONNECTION_FILE):
                self.root.after(0, self.display_error, "Keine Server-Datei gefunden!")
                return
                
            with open(cfg.CONNECTION_FILE, "r") as f:
                data = json.load(f)
                current_host = data.get("host")

            final_prompt = question
            if os.path.exists(CONTEXT_FILE):
                with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
                    context = f.read()[:15000] 
                final_prompt = f"Context:\n{context}\n\nQuestion: {question}"

            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(300)
            client.connect((current_host, cfg.SERVER_PORT))
            client.sendall(final_prompt.encode('utf-8'))
            
            resp_bytes = b""
            while True:
                chunk = client.recv(cfg.BUFFER_SIZE)
                if not chunk: break
                resp_bytes += chunk
            
            client.close()
            self.root.after(0, self.display_response, resp_bytes.decode('utf-8').strip())

        except Exception as e:
            self.root.after(0, self.display_error, str(e))

    def display_response(self, text):
        self.append_text(text, "ai")
        self.status_bar.config(text="Bereit")

    def display_error(self, msg):
        self.append_text(f"FEHLER: {msg}", "error")
        self.status_bar.config(text="Verbindungsfehler")

if __name__ == "__main__":
    root = tk.Tk()
    app = AI_PDF_Assistant(root)
    root.mainloop()