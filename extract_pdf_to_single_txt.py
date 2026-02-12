#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 10:03:38 2026
@author: temuuleu
"""

import fitz
from pathlib import Path

papers_dir  = "/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/Courses/LOCAL_LLM/papers"
output_file = "/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/Courses/LOCAL_LLM/papers/papers_combined.txt"

output = open(output_file, "w", encoding="utf-8")

papers_path = Path(papers_dir)
pdf_files = sorted(papers_path.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDFs\n")

for i, pdf_file in enumerate(pdf_files, 1):
    try:
        print(f"[{i}/{len(pdf_files)}] Processing: {pdf_file.name}...", end=" ", flush=True)
        
        doc = fitz.open(pdf_file)
        text = ""
        
        for page_num, page in enumerate(doc):
            text += page.get_text()
        
        output.write(f"\n\n{'='*80}\n")
        output.write(f"SOURCE: {pdf_file.name}\n")
        output.write(f"PAGES: {len(doc)}\n")
        output.write(f"{'='*80}\n\n")
        output.write(text)
        output.write("\n\n")
        
        print(f"✅ ({len(doc)} pages, {len(text)} chars)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

output.close()

print(f"\n{'='*80}")
print(f"✅ DONE! Created: {output_file}")
print(f"{'='*80}")

# Show stats
with open(output_file, "r", encoding="utf-8") as f:
    content = f.read()

print(f"Total size: {len(content)} characters")
print(f"Total size: {len(content) / 1e6:.2f} MB")