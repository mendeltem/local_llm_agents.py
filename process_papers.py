#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Paper Processing
- Process all PDFs in a directory
- Generate summary.txt
- Generate q_and_a.txt with 20 Q&A pairs
- Uses LLM Server
"""

import socket
import json
import fitz  # PyMuPDF
import time
from pathlib import Path
from typing import Optional, Tuple, List
import logging
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration"""
    PAPER_DIR = "papers"
    OUTPUT_DIR = "papers_summarys"
    CONNECTION_FILE = "server_connection.json"
    
    # Network
    SERVER_PORT = 54321
    BUFFER_SIZE = 4096
    SOCKET_TIMEOUT = 300
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2.0
    
    # PDF
    MAX_CONTEXT_LENGTH = 15000
    
    # Processing
    NUM_QUESTIONS = 20

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    """Setup logging"""
    log_file = f"paper_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
# SERVER CONNECTION
# ============================================================================

def load_server_info(config: Config) -> Optional[dict]:
    """Load server connection info"""
    try:
        if Path(config.CONNECTION_FILE).exists():
            with open(config.CONNECTION_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading server info: {e}")
    return None

def send_to_llm(prompt: str, config: Config, server_info: dict) -> Tuple[bool, str]:
    """Send request to LLM server"""
    for attempt in range(config.RETRY_ATTEMPTS):
        try:
            logger.info(f"Attempt {attempt + 1}/{config.RETRY_ATTEMPTS}")
            
            host = server_info.get("host", "localhost")
            
            # Connect
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(config.SOCKET_TIMEOUT)
            sock.connect((host, config.SERVER_PORT))
            
            # Send
            sock.sendall(prompt.encode('utf-8'))
            
            # Receive
            response_bytes = b""
            while True:
                chunk = sock.recv(config.BUFFER_SIZE)
                if not chunk:
                    break
                response_bytes += chunk
            
            sock.close()
            
            response = response_bytes.decode('utf-8').strip()
            logger.info("Request successful")
            return True, response
        
        except socket.timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}")
            if attempt < config.RETRY_ATTEMPTS - 1:
                time.sleep(config.RETRY_DELAY)
                continue
            return False, "Timeout"
        
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt < config.RETRY_ATTEMPTS - 1:
                time.sleep(config.RETRY_DELAY)
                continue
            return False, str(e)
    
    return False, "All attempts failed"

# ============================================================================
# PDF PROCESSING
# ============================================================================

def extract_text_from_pdf(pdf_path: Path) -> Tuple[bool, str]:
    """Extract text from PDF"""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num, page in enumerate(doc):
            try:
                text = page.get_text("text")
                full_text += text + "\n"
            except Exception as e:
                logger.warning(f"Error extracting page {page_num}: {e}")
        
        # Remove references section
        if "References" in full_text:
            full_text = full_text.split("References")[0]
        
        logger.info(f"Extracted {len(full_text)} characters from {pdf_path.name}")
        return True, full_text
    
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return False, str(e)

# ============================================================================
# SUMMARY GENERATION
# ============================================================================

def generate_summary(text: str, config: Config, server_info: dict) -> Tuple[bool, str]:
    """Generate summary of the paper"""
    
    # Truncate if too long
    context = text[:config.MAX_CONTEXT_LENGTH]
    
    prompt = f"""Context:
{context}

Please provide a comprehensive summary of this scientific paper. Include:
1. Main research question/objective
2. Methodology
3. Key findings
4. Conclusions
5. Significance/implications

Keep it concise but informative (max 500 words)."""
    
    logger.info("Generating summary...")
    return send_to_llm(prompt, config, server_info)

# ============================================================================
# Q&A GENERATION
# ============================================================================

def generate_qa(text: str, config: Config, server_info: dict) -> Tuple[bool, str]:
    """Generate Q&A pairs"""
    
    context = text[:config.MAX_CONTEXT_LENGTH]
    
    prompt = f"""Context:
{context}

Generate {config.NUM_QUESTIONS} important questions and answers about this paper. 
Cover different aspects: methodology, results, conclusions, implications, limitations.

Format each Q&A pair as:
Q1: [Question]
A1: [Answer]

Q2: [Question]
A2: [Answer]

... and so on for {config.NUM_QUESTIONS} questions."""
    
    logger.info(f"Generating {config.NUM_QUESTIONS} Q&A pairs...")
    return send_to_llm(prompt, config, server_info)

# ============================================================================
# FILE OPERATIONS
# ============================================================================

def save_output(output_dir: Path, filename: str, content: str) -> bool:
    """Save output to file"""
    try:
        filepath = output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving {filename}: {e}")
        return False

def is_already_processed(output_dir: Path) -> bool:
    """Check if paper already processed"""
    summary_exists = (output_dir / "summary.txt").exists()
    qa_exists = (output_dir / "q_and_a.txt").exists()
    return summary_exists and qa_exists

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_paper(pdf_path: Path, config: Config, server_info: dict) -> bool:
    """Process a single paper"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {pdf_path.name}")
    logger.info(f"{'='*80}")
    
    # Create output directory
    paper_name = pdf_path.stem
    output_dir = Path(config.OUTPUT_DIR) / paper_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    if is_already_processed(output_dir):
        logger.info(f"⏭️  Already processed, skipping: {paper_name}")
        return True
    
    # Extract text
    logger.info("📄 Extracting text from PDF...")
    success, text = extract_text_from_pdf(pdf_path)
    if not success:
        logger.error(f"❌ Failed to extract text: {text}")
        return False
    
    # Generate summary
    logger.info("📝 Generating summary...")
    success, summary = generate_summary(text, config, server_info)
    if not success:
        logger.error(f"❌ Failed to generate summary: {summary}")
        return False
    
    # Save summary
    save_output(output_dir, "summary.txt", summary)
    
    # Generate Q&A
    logger.info("❓ Generating Q&A...")
    success, qa = generate_qa(text, config, server_info)
    if not success:
        logger.error(f"❌ Failed to generate Q&A: {qa}")
        return False
    
    # Save Q&A
    save_output(output_dir, "q_and_a.txt", qa)
    
    logger.info(f"✅ Successfully processed: {paper_name}")
    return True

def main():
    """Main function"""
    
    config = Config()
    
    logger.info("="*80)
    logger.info("AUTOMATED PAPER PROCESSING")
    logger.info("="*80)
    
    # Load server info
    logger.info("\n🔌 Loading server connection...")
    server_info = load_server_info(config)
    if not server_info:
        logger.error("❌ No server connection info found")
        logger.error(f"Please create {config.CONNECTION_FILE}")
        return
    
    logger.info(f"✅ Server: {server_info.get('host', 'localhost')}")
    
    # Get all PDFs
    paper_dir = Path(config.PAPER_DIR)
    if not paper_dir.exists():
        logger.error(f"❌ Directory not found: {config.PAPER_DIR}")
        return
    
    pdf_files = list(paper_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"⚠️  No PDF files found in {config.PAPER_DIR}")
        return
    
    logger.info(f"\n📚 Found {len(pdf_files)} PDF files")
    
    # Create output directory
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Process each paper
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"\n[{i}/{len(pdf_files)}]")
        
        # Check if already processed
        paper_name = pdf_path.stem
        output_dir = Path(config.OUTPUT_DIR) / paper_name
        if is_already_processed(output_dir):
            logger.info(f"⏭️  Already processed: {paper_name}")
            skipped_count += 1
            continue
        
        success = process_paper(pdf_path, config, server_info)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
        
        # Small delay between papers
        if i < len(pdf_files):
            logger.info("⏳ Waiting 2 seconds before next paper...")
            time.sleep(2)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"✅ Successful: {success_count}")
    logger.info(f"⏭️  Skipped:    {skipped_count}")
    logger.info(f"❌ Failed:     {failed_count}")
    logger.info(f"📊 Total:      {len(pdf_files)}")
    logger.info(f"\n📁 Output directory: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()