# 🧠 Local LLM Agents for Stroke MRI Research

A lightweight LLM inference and agent stack designed for the **Charité HPC cluster**, enabling self-hosted language model workflows for neuroimaging research — no cloud APIs needed.

## Overview

This project runs open-source LLMs (7B–70B) on GPU nodes via a simple TCP server, with multiple specialized clients that automate research tasks like paper summarization, Q&A dataset generation, and literature search.

```
┌─────────────────────────────────────────────────┐
│                  GPU Node (A100)                 │
│  llm_server.py + llm_config.py                  │
│  Qwen3-30B / DeepSeek-70B / MedGemma-27B / ...  │
└──────────────────────┬──────────────────────────┘
                       │ TCP Socket (:54321)
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  pdf_client_gui  q_a_generator  client_researcher
  (Summarize)     (Training QA)  (Find Papers)
```

## Components

### Server
| File | Description |
|------|-------------|
| `llm_server.py` | TCP server that loads a HuggingFace model onto GPU and serves inference requests |
| `llm_config.py` | Shared config — model paths, ports, buffer sizes |

### Clients
| File | Description |
|------|-------------|
| `pdf_client_gui.py` | Tkinter GUI — extracts PDF text and generates summaries via LLM |
| `pdf_client_gui_2.py` | Extended version with batch processing |
| `q_a_generator_client.py` | Generates Q&A training pairs (JSONL) from papers for LLM finetuning |
| `client_researcher.py` | Reads a topic from `research_topic.txt`, queries LLM for paper links |
| `tain_text_generator_client.py` | Generates structured training text from papers |
| `research_agent.py` | Automated research agent |

### Finetuning
| File | Description |
|------|-------------|
| `fine_tune_llm.py` | Finetuning script for HuggingFace models |
| `finetune_mistral.py` | Mistral-7B specific finetuning |
| `slurm_run_fine_tune_llm.py` | SLURM job launcher for finetuning |
| `slurm_lib.py` | SLURM utilities |
| `slurm_scripts/` | SLURM batch scripts |

### Utilities
| File | Description |
|------|-------------|
| `extract_pdf_to_single_txt.py` | Extracts all PDFs into a single combined text file |
| `process_papers.py` | Batch paper processing pipeline |
| `research_topic.txt` | Input file for `client_researcher.py` |

## Supported Models

| Size | Model | Use Case |
|------|-------|----------|
| `tiny` | Mistral-7B-Instruct | Fast prototyping, low memory |
| `small` | Qwen3-14B | Balanced speed/quality |
| `middle` | MedGemma-27B | Medical text specialist |
| `big` | Qwen3-Coder-30B | Recommended for stroke MRI analysis |
| `huge` | DeepSeek-R1-70B | Most capable, needs A100-80GB |

## Quick Start

```bash
# 1. Start server on GPU node (via SLURM or interactive session)
python llm_server.py --size big

# 2. Run a client from login node
python pdf_client_gui.py          # GUI for paper summarization
python q_a_generator_client.py    # Generate Q&A training data
python client_researcher.py       # Search for paper links
```

## Finetuning

```bash
# Generate training data first
python q_a_generator_client.py

# Launch finetuning via SLURM
python slurm_run_fine_tune_llm.py
```

## Requirements

- Python 3.10+
- PyTorch with CUDA
- `transformers`, `psutil`, `PyMuPDF`
- HPC access with GPU nodes (A100 recommended)

## Project Context

Developed at **Charité – Universitätsmedizin Berlin**, Centrum für Schlaganfallforschung Berlin (CSB), for automating neuroimaging literature analysis and generating domain-specific training data for medical LLMs.

## License

For research use only.
