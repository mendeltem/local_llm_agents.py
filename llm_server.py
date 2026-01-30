import socket
import json
import torch
import os
import argparse  # <--- Für Kommandozeilen-Argumente
from transformers import AutoTokenizer, AutoModelForCausalLM
import llm_config as cfg 

def get_args():
    parser = argparse.ArgumentParser(description="Starte den AI Server mit gewünschter Modellgröße.")
    parser.add_argument("--size", type=str, default=cfg.DEFAULT_SIZE, 
                        choices=cfg.MODEL_OPTIONS.keys(),
                        help="Wähle die Größe: huge, big, middle, small, tiny")
    return parser.parse_args()

def main():
    # 1. Argumente lesen
    args = get_args()
    selected_path = cfg.MODEL_OPTIONS[args.size]
    
    print(f"\n🚀 STARTE SERVER IM MODUS: {args.size.upper()}")
    print(f"📂 Lade Modell von: {selected_path}")
    
    # 2. Modell Laden
    try:
        tokenizer = AutoTokenizer.from_pretrained(selected_path)
        model = AutoModelForCausalLM.from_pretrained(
            selected_path, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    except Exception as e:
        print(f"❌ FEHLER beim Laden des Modells: {e}")
        return

    # 3. Netzwerk Starten
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    hostname = socket.gethostname()
    server.bind(('0.0.0.0', cfg.SERVER_PORT))
    server.listen(1)

    # 4. Automatische Verbindunginfo speichern
    print(f"✍️  Speichere Adresse in '{cfg.CONNECTION_FILE}'...")
    info = {
        "host": hostname, 
        "port": cfg.SERVER_PORT,
        "model_size": args.size,
        "model_path": selected_path
    }
    with open(cfg.CONNECTION_FILE, "w") as f:
        json.dump(info, f)
    
    print("\n" + "="*60)
    print(f"✅ SERVER BEREIT AUF: {hostname} (Port {cfg.SERVER_PORT})")
    print(f"🧠 Geladenes Modell: {args.size.upper()}")
    print("="*60 + "\n")

    # 5. Warteschleife
    while True:
        try:
            client, addr = server.accept()
            question = client.recv(cfg.BUFFER_SIZE).decode('utf-8', errors='ignore')
            
            if not question: 
                client.close()
                continue

            # Prompt erstellen
            msgs = [{"role": "user", "content": question}]
            # Chat Template sicher anwenden
            if tokenizer.chat_template:
                inputs = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
            else:
                # Fallback für Modelle ohne Template
                inputs = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(inputs, max_new_tokens=1024, temperature=0.3)
            
            resp = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            client.sendall(resp.encode('utf-8'))
            client.close()
            
        except Exception as e:
            print(f"⚠️ Verarbeitungsfehler: {e}")
            try:
                client.close()
            except: pass

if __name__ == "__main__":
    main()