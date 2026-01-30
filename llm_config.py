import os

# --- MODELL AUSWAHL (MENÜ) ---
# Hier definieren wir die verfügbaren Größen
MODEL_OPTIONS = {
    # HUGE (70B+): Braucht A100-80GB. Sehr schlau, langsam.
    "huge": "/sc-resources/llms/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    
    # BIG (27B-35B): Perfekt für Coding & Komplexe Aufgaben. (Deine aktuelle Wahl)
    "big": "/sc-resources/llms/Qwen/Qwen3-Coder-30B-A3B-Instruct",
    
    # MIDDLE (14B-27B): Gute Balance für medizinische Texte.
    "middle": "/sc-resources/llms/google/medgemma-27b-it",
    
    # SMALL (14B): Schneller, gut für Zusammenfassungen.
    "small": "/sc-resources/llms/Qwen/Qwen3-14B",
    
    # TINY (7B): Sehr schnell, braucht wenig Speicher.
    "tiny": "/sc-resources/llms/mistralai/Mistral-7B-Instruct-v0.3"
}

# Standard-Modell, falls nichts angegeben wird
DEFAULT_SIZE = "big"

# --- NETZWERK & SYSTEM ---
SERVER_PORT = 54321
BUFFER_SIZE = 10 * 1024 * 1024  # 10 MB Limit
CONNECTION_FILE = "server_connection.json"  # Die "Briefkasten"-Datei