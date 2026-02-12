#!/usr/bin/env python3
"""
🧠 AI-Powered LLM Server for Stroke MRI Analysis
Optimized for Charité HPC with real-time memory monitoring
"""

import socket
import json
import torch
import os
import sys
import argparse
import psutil
import threading
import time
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import llm_config as cfg

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

monitoring_active = False
server_stats = {
    "start_time": datetime.now(),
    "requests_processed": 0,
    "total_tokens_generated": 0,
    "errors": 0
}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="🚀 Start LLM server for stroke MRI analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server.py --size big              # Use Qwen3-Coder-30B
  python server.py --size tiny             # Use Mistral-7B (fastest)
  python server.py --size huge             # Use DeepSeek-70B (smartest)
        """)
    
    parser.add_argument(
        "--size", 
        type=str, 
        default=None,
        choices=list(cfg.MODEL_OPTIONS.keys()),
        help="Model size: huge, big, middle, small, tiny (default: interactive)")
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)")
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Generation temperature (0.0-1.0, default: 0.3)")
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)")
    
    parser.add_argument(
        "--no-monitoring",
        action="store_true",
        help="Disable continuous memory monitoring")
    
    return parser.parse_args()

# ============================================================================
# INTERACTIVE MODEL SELECTION
# ============================================================================

def interactive_model_selection():
    """Interactive model selection menu"""
    print("\n" + "="*70)
    print("🤖 SELECT MODEL SIZE")
    print("="*70)
    
    options = list(cfg.MODEL_OPTIONS.keys())
    model_info = {
        "tiny": "7B - ⚡ Very fast, low memory",
        "small": "14B - ⚡⚡ Fast, balanced",
        "middle": "27B - 🔴 Medical specialist (needs A100-80GB)",
        "big": "30B - ⭐ Recommended for stroke MRI (your setup)",
        "huge": "70B - 🧠 Most intelligent (needs A100-80GB or DGX)"
    }
    
    for i, size in enumerate(options, 1):
        print(f"{i}. {size.upper():6s} - {model_info.get(size, '')}")
    
    while True:
        try:
            choice = input(f"\nSelect option (1-{len(options)}, default=2 for 'big'): ").strip()
            choice = choice if choice else "2"
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                selected = options[idx]
                print(f"✅ Selected: {selected.upper()}")
                return selected
            else:
                print(f"❌ Invalid input. Please enter 1-{len(options)}")
        except ValueError:
            print("❌ Please enter a valid number")

# ============================================================================
# GPU & MEMORY MONITORING
# ============================================================================

def get_gpu_info():
    """Get detailed GPU information"""
    if not torch.cuda.is_available():
        return None
    
    device_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_id)
    
    allocated = torch.cuda.memory_allocated(device_id) / 1e9
    reserved = torch.cuda.memory_reserved(device_id) / 1e9
    total = props.total_memory / 1e9
    
    return {
        "name": props.name,
        "device_id": device_id,
        "total_memory_gb": total,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "available_gb": total - allocated,
        "utilization_percent": (allocated / total) * 100
    }

def get_cpu_info():
    """Get CPU RAM information"""
    mem = psutil.virtual_memory()
    return {
        "used_gb": mem.used / 1e9,
        "total_gb": mem.total / 1e9,
        "percent": mem.percent,
        "available_gb": mem.available / 1e9
    }

def print_system_info():
    """Print comprehensive system information"""
    print("\n" + "="*70)
    print("🖥️  SYSTEM INFORMATION")
    print("="*70)
    
    # GPU Info
    if torch.cuda.is_available():
        gpu = get_gpu_info()
        print(f"\n✅ GPU Available:")
        print(f"   Device:    {gpu['name']}")
        print(f"   Total:     {gpu['total_memory_gb']:.2f} GB")
        print(f"   Allocated: {gpu['allocated_gb']:.2f} GB")
        print(f"   Reserved:  {gpu['reserved_gb']:.2f} GB")
        print(f"   Available: {gpu['available_gb']:.2f} GB")
        print(f"   Usage:     {gpu['utilization_percent']:.1f}%")
    else:
        print("\n⚠️  GPU NOT available - CPU will be used (slow!)")
    
    # CPU Info
    cpu = get_cpu_info()
    print(f"\n💾 CPU RAM:")
    print(f"   Total:     {cpu['total_gb']:.2f} GB")
    print(f"   Used:      {cpu['used_gb']:.2f} GB")
    print(f"   Available: {cpu['available_gb']:.2f} GB")
    print(f"   Usage:     {cpu['percent']:.1f}%")
    
    print("="*70 + "\n")

def print_memory_status(label=""):
    """Print current memory status"""
    cpu = get_cpu_info()
    status = f"[{label:12s}] RAM: {cpu['used_gb']:6.2f}/{cpu['total_gb']:6.2f} GB ({cpu['percent']:5.1f}%)"
    
    if torch.cuda.is_available():
        gpu = get_gpu_info()
        status += f" | GPU: {gpu['allocated_gb']:6.2f}/{gpu['total_memory_gb']:6.2f} GB ({gpu['utilization_percent']:5.1f}%)"
    
    print(status)

def memory_monitor(interval=3):
    """Background thread to continuously monitor memory"""
    global monitoring_active
    print("📊 Memory monitoring started (interval: 3s)")
    
    while monitoring_active:
        print_memory_status(label="MONITOR")
        time.sleep(interval)
    
    print("📊 Memory monitoring stopped")

def start_memory_monitoring():
    """Start background memory monitoring"""
    global monitoring_active
    monitoring_active = True
    monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread

def stop_memory_monitoring():
    """Stop background memory monitoring"""
    global monitoring_active
    monitoring_active = False

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path, model_size):
    """Load model with error handling and progress indication"""
    print(f"\n📂 Loading model from: {model_path}")
    
    try:
        print("  → Loading tokenizer...", end="", flush=True)
        print_memory_status(label="PRE-LOAD")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(" ✅")
        
        print("  → Loading model (fp16, auto device mapping)...", end="", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print(" ✅")
        
        print_memory_status(label="POST-LOAD")
        
        # Set model to evaluation mode
        model.eval()
        
        return tokenizer, model
        
    except Exception as e:
        print(f" ❌\n❌ Error loading model: {e}")
        sys.exit(1)

# ============================================================================
# INFERENCE
# ============================================================================

def generate_response(model, tokenizer, question, max_tokens=1024, temperature=0.3):
    """Generate response with monitoring"""
    print_memory_status(label="PRE-INFER")
    
    # Prepare input
    msgs = [{"role": "user", "content": question}]
    
    try:
        # Apply chat template if available
        if tokenizer.chat_template:
            inputs = tokenizer.apply_chat_template(
                msgs, 
                return_tensors="pt", 
                add_generation_prompt=True
            ).to(model.device)
        else:
            # Fallback for models without template
            inputs = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
        
        print_memory_status(label="PRE-GEN ")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        print_memory_status(label="POST-GEN")
        
        # Decode response (skip input tokens)
        response = tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        )
        
        return response
        
    except Exception as e:
        print(f"❌ Inference error: {e}")
        return f"Error generating response: {e}"

# ============================================================================
# SERVER
# ============================================================================

def save_connection_info(hostname, model_size, model_path):
    """Save server connection information"""
    info = {
        "host": hostname,
        "port": cfg.SERVER_PORT,
        "model_size": model_size,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "buffer_size": cfg.BUFFER_SIZE
    }
    
    try:
        with open(cfg.CONNECTION_FILE, "w") as f:
            json.dump(info, f, indent=2)
        print(f"✍️  Connection info saved to '{cfg.CONNECTION_FILE}'")
        return True
    except Exception as e:
        print(f"⚠️  Could not save connection info: {e}")
        return False

def print_server_banner(hostname, model_size, model_path):
    """Print server startup banner"""
    print("\n" + "="*70)
    print("🚀 LLM SERVER STARTED")
    print("="*70)
    print(f"🖥️  Host:        {hostname}")
    print(f"🔌 Port:        {cfg.SERVER_PORT}")
    print(f"🧠 Model:       {model_size.upper()}")
    print(f"📂 Path:        {model_path}")
    print(f"⏰ Started:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print("⏳ Waiting for connections...\n")

def handle_client(client, addr, model, tokenizer, args):
    """Handle individual client connection"""
    global server_stats
    
    try:
        # Receive question
        data = client.recv(cfg.BUFFER_SIZE).decode('utf-8', errors='ignore').strip()
        
        if not data:
            client.close()
            return
        
        print(f"\n📨 Request from {addr[0]}:{addr[1]}")
        print(f"❓ Question: {data[:80]}{'...' if len(data) > 80 else ''}")
        
        # Generate response
        response = generate_response(
            model, 
            tokenizer, 
            data,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        # Send response
        response_bytes = response.encode('utf-8')
        client.sendall(response_bytes)
        
        # Update stats
        server_stats["requests_processed"] += 1
        server_stats["total_tokens_generated"] += len(tokenizer.encode(response))
        
        print(f"✅ Response sent ({len(response)} chars, {len(tokenizer.encode(response))} tokens)")
        
    except Exception as e:
        print(f"❌ Error handling client: {e}")
        server_stats["errors"] += 1
        try:
            client.sendall(f"Error: {e}".encode('utf-8'))
        except:
            pass
    finally:
        client.close()

def run_server(model, tokenizer, args):
    """Main server loop"""
    hostname = socket.gethostname()
    
    # Create server socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind(('0.0.0.0', cfg.SERVER_PORT))
        server.listen(5)
        
        # Save connection info
        save_connection_info(hostname, args.size, cfg.MODEL_OPTIONS[args.size])
        
        # Print banner
        print_server_banner(hostname, args.size, cfg.MODEL_OPTIONS[args.size])
        
        # Start memory monitoring
        if not args.no_monitoring:
            start_memory_monitoring()
        
        # Main loop
        while True:
            try:
                client, addr = server.accept()
                
                # Handle in main thread (can be made multi-threaded later)
                handle_client(client, addr, model, tokenizer, args)
                
            except KeyboardInterrupt:
                print("\n\n👋 Shutting down...")
                break
            except Exception as e:
                print(f"❌ Server error: {e}")
                
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)
    finally:
        stop_memory_monitoring()
        server.close()
        print_server_stats()

def print_server_stats():
    """Print server statistics"""
    uptime = datetime.now() - server_stats["start_time"]
    print("\n" + "="*70)
    print("📊 SERVER STATISTICS")
    print("="*70)
    print(f"⏱️  Uptime:              {uptime}")
    print(f"📨 Requests processed:  {server_stats['requests_processed']}")
    print(f"🎯 Total tokens:        {server_stats['total_tokens_generated']}")
    print(f"❌ Errors:              {server_stats['errors']}")
    print("="*70 + "\n")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    # Parse arguments
    args = get_args()
    
    # Interactive model selection if needed
    if args.size is None:
        args.size = interactive_model_selection()
    
    model_path = cfg.MODEL_OPTIONS[args.size]
    
    # Print system info
    print_system_info()
    
    # Load model
    print(f"🚀 Starting server with model: {args.size.upper()}")
    tokenizer, model = load_model(model_path, args.size)
    
    # Run server
    run_server(model, tokenizer, args)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Server terminated by user")
        stop_memory_monitoring()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)