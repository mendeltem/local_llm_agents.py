#!/usr/bin/env python3
"""
🔬 Paper Researcher Client
Reads a research topic from a .txt file, uses the LLM server to find
relevant paper links, and saves them to a structured output file.

Usage:
  python client_researcher.py                          # Uses default topic.txt
  python client_researcher.py --topic-file my_topic.txt
  python client_researcher.py --topic "white matter hyperintensities stroke"
  python client_researcher.py --rounds 5               # More search rounds = more papers
"""

import socket
import json
import argparse
import re
import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import llm_config as cfg

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_TOPIC_FILE = "research_topic.txt"
OUTPUT_DIR = "paper_links"
MAX_ROUNDS = 3  # Number of LLM query rounds per topic


# ============================================================================
# SERVER COMMUNICATION
# ============================================================================

def load_server_info() -> Dict:
    """Load server connection info from shared config file."""
    conn_file = Path(cfg.CONNECTION_FILE)
    if not conn_file.exists():
        print(f"❌ Connection file not found: {cfg.CONNECTION_FILE}")
        print("💡 Start server.py first!")
        return {}
    with open(conn_file, "r") as f:
        return json.load(f)


def send_request(server_info: Dict, prompt: str, timeout: int = 600) -> Tuple[bool, str]:
    """Send a prompt to the LLM server and return the response."""
    host = server_info.get("host")
    port = server_info.get("port", cfg.SERVER_PORT)

    if not host:
        return False, "No hostname in server info"

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.sendall(prompt.encode("utf-8"))

        response_bytes = b""
        while True:
            chunk = sock.recv(cfg.BUFFER_SIZE)
            if not chunk:
                break
            response_bytes += chunk

        sock.close()
        return True, response_bytes.decode("utf-8").strip()

    except socket.timeout:
        return False, "Timeout  server too slow or stuck"
    except ConnectionRefusedError:
        return False, f"Connection refused to {host}:{port}"
    except Exception as e:
        return False, str(e)


# ============================================================================
# TOPIC LOADING
# ============================================================================

def load_topic(topic_file: str) -> str:
    """Load research topic from a text file."""
    path = Path(topic_file)
    if not path.exists():
        print(f"❌ Topic file not found: {topic_file}")
        print(f"💡 Creating template at '{topic_file}'  edit it and re-run.")
        path.write_text(
            "# Research Topic\n"
            "# Write your research topic / keywords below (one topic per line).\n"
            "# Lines starting with # are ignored.\n\n"
            "white matter hyperintensities segmentation deep learning\n",
            encoding="utf-8",
        )
        return ""

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    # Filter comments and empty lines
    topics = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    return " | ".join(topics) if topics else ""


# ============================================================================
# PAPER LINK EXTRACTION
# ============================================================================

def build_search_prompt(topic: str, round_num: int, existing_links: List[str]) -> str:
    """Build a prompt that asks the LLM to suggest paper links for the topic."""

    exclude_block = ""
    if existing_links:
        exclude_block = (
            "\n\nIMPORTANT: Do NOT repeat these already-found papers:\n"
            + "\n".join(f"- {link}" for link in existing_links[-20:])
        )

    prompt = f"""You are a scientific literature search assistant.

TASK: Find {15} real, citable research papers related to the topic below.
Return ONLY a JSON array of objects. No explanations, no markdown.

TOPIC: {topic}

ROUND: {round_num} (find different papers than previous rounds){exclude_block}

REQUIRED JSON FORMAT:
[
  {{"title": "Paper Title Here", "authors": "First Author et al.", "year": 2023, "url": "https://doi.org/10.xxxx/xxxxx", "source": "PubMed/arXiv/DOI"}},
  {{"title": "Another Paper", "authors": "Author et al.", "year": 2022, "url": "https://arxiv.org/abs/xxxx.xxxxx", "source": "arXiv"}}
]

RULES:
- Provide real DOIs (https://doi.org/...) or arXiv links (https://arxiv.org/abs/...) or PubMed links
- Include the most relevant and highly cited papers
- Mix recent papers (2020-2025) with seminal/classic ones
- Each entry MUST have: title, authors, year, url, source
- Return ONLY the JSON array, nothing else

Generate the JSON array now:"""

    return prompt


def extract_paper_links(response: str) -> List[Dict]:
    """Parse LLM response into structured paper entries."""

    # Strategy 1: Direct JSON parse
    try:
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        if "[" in text and "]" in text:
            start = text.index("[")
            end = text.rindex("]") + 1
            papers = json.loads(text[start:end])
            if isinstance(papers, list):
                return [p for p in papers if isinstance(p, dict) and "url" in p and "title" in p]
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Extract individual JSON objects
    papers = []
    brace_level = 0
    current = ""
    for ch in response:
        if ch == "{":
            brace_level += 1
            current += ch
        elif ch == "}":
            current += ch
            brace_level -= 1
            if brace_level == 0 and current.strip():
                try:
                    obj = json.loads(current.strip())
                    if "url" in obj and "title" in obj:
                        papers.append(obj)
                except json.JSONDecodeError:
                    pass
                current = ""
        elif brace_level > 0:
            current += ch

    if papers:
        return papers

    # Strategy 3: Regex for URLs
    urls = re.findall(r'https?://(?:doi\.org|arxiv\.org|pubmed\.ncbi\.nlm\.nih\.gov|www\.ncbi\.nlm\.nih\.gov)[^\s",\]]+', response)
    return [{"title": "Unknown", "url": u, "source": "regex-extracted"} for u in urls]


def deduplicate(papers: List[Dict]) -> List[Dict]:
    """Remove duplicate papers by URL."""
    seen_urls = set()
    unique = []
    for p in papers:
        url = p.get("url", "").strip().rstrip("/")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(p)
    return unique


# ============================================================================
# OUTPUT
# ============================================================================

def save_results(papers: List[Dict], topic: str, output_dir: str) -> Path:
    """Save paper links to a structured text file."""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^\w\s-]', '', topic[:50]).strip().replace(' ', '_')
    output_file = Path(output_dir) / f"papers_{safe_topic}_{timestamp}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Paper Links for: {topic}\n")
        f.write(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total papers found: {len(papers)}\n")
        f.write("=" * 80 + "\n\n")

        for i, p in enumerate(papers, 1):
            title = p.get("title", "Unknown")
            authors = p.get("authors", "N/A")
            year = p.get("year", "N/A")
            url = p.get("url", "N/A")
            source = p.get("source", "N/A")

            f.write(f"[{i:3d}] {title}\n")
            f.write(f"      Authors: {authors}\n")
            f.write(f"      Year:    {year}\n")
            f.write(f"      URL:     {url}\n")
            f.write(f"      Source:  {source}\n\n")

        # Also write a plain URL list at the end
        f.write("\n" + "=" * 80 + "\n")
        f.write("# PLAIN URL LIST (for batch downloading)\n")
        f.write("=" * 80 + "\n")
        for p in papers:
            f.write(p.get("url", "") + "\n")

    return output_file


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="🔬 Paper Researcher Client  find paper links via LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python client_researcher.py
  python client_researcher.py --topic "stroke MRI deep learning segmentation"
  python client_researcher.py --topic-file my_topics.txt --rounds 5
        """,
    )
    parser.add_argument("--topic-file", type=str, default=DEFAULT_TOPIC_FILE,
                        help=f"Path to topic file (default: {DEFAULT_TOPIC_FILE})")
    parser.add_argument("--topic", type=str, default=None,
                        help="Directly specify topic (overrides --topic-file)")
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS,
                        help=f"Number of search rounds (default: {MAX_ROUNDS})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Server timeout in seconds (default: 600)")
    args = parser.parse_args()

    # --- Load topic ---
    if args.topic:
        topic = args.topic
    else:
        topic = load_topic(args.topic_file)
        if not topic:
            return

    print("\n" + "=" * 70)
    print("🔬 PAPER RESEARCHER CLIENT")
    print("=" * 70)
    print(f"📋 Topic:   {topic}")
    print(f"🔄 Rounds:  {args.rounds}")
    print(f"📂 Output:  {args.output_dir}/")

    # --- Connect to server ---
    server_info = load_server_info()
    if not server_info:
        return

    host = server_info.get("host", "unknown")
    model = server_info.get("model_size", "unknown")
    print(f"🖥️  Server:  {host} ({model.upper()})")
    print("=" * 70 + "\n")

    # --- Search rounds ---
    all_papers: List[Dict] = []

    for round_num in range(1, args.rounds + 1):
        print(f"🔍 Round {round_num}/{args.rounds}...")

        prompt = build_search_prompt(topic, round_num, [p.get("url", "") for p in all_papers])
        success, response = send_request(server_info, prompt, timeout=args.timeout)

        if not success:
            print(f"   ❌ Failed: {response}")
            continue

        papers = extract_paper_links(response)
        print(f"   ✅ Found {len(papers)} papers in this round")

        all_papers.extend(papers)
        all_papers = deduplicate(all_papers)
        print(f"   📊 Total unique papers so far: {len(all_papers)}")

    # --- Save results ---
    if all_papers:
        output_file = save_results(all_papers, topic, args.output_dir)
        print(f"\n{'=' * 70}")
        print(f"✅ DONE! Found {len(all_papers)} unique papers")
        print(f"📄 Saved to: {output_file}")
        print(f"{'=' * 70}\n")
    else:
        print("\n❌ No papers found. Check server connection and try again.")


if __name__ == "__main__":
    main()