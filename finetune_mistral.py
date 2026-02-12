#!/usr/bin/env python3
"""
Fine-Tune Mistral-7B on Medical Papers (Stroke MRI)
Optimized for A100-80GB GPU
"""

import torch
import os
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================

MODEL_PATH = "/sc-resources/llms/mistralai/Mistral-7B-Instruct-v0.3"
DATA_FILE = "/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/Courses/LOCAL_LLM/papers/papers_combined.txt"
OUTPUT_DIR = "./mistral-finetuned"
MAX_SEQ_LENGTH = 2048

# ============================================================================
# SETUP
# ============================================================================

def load_data():
    """Load and prepare dataset"""
    logger.info(f"📂 Loading data from {DATA_FILE}")
    
    if not Path(DATA_FILE).exists():
        raise FileNotFoundError(f"{DATA_FILE} not found!")
    
    dataset = load_dataset("text", data_files=DATA_FILE, split="train")
    logger.info(f"✅ Loaded {len(dataset)} examples")
    
    return dataset

def tokenize_function(examples, tokenizer):
    """Tokenize texts"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        return_overflowing_tokens=False
    )

def main():
    logger.info("🚀 Starting Fine-Tuning Pipeline")
    logger.info(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # ========== LOAD MODEL & TOKENIZER ==========
    logger.info(f"\n📥 Loading model from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    logger.info(f"✅ Model loaded")
    logger.info(f"   Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # ========== LOAD DATA ==========
    dataset = load_data()
    
    # Tokenize
    logger.info("🔄 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    # Split train/eval
    split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    logger.info(f"✅ Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # ========== TRAINING ARGS ==========
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        
        # Training
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        
        # Optimization
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        
        # Logging & Checkpointing
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=500,
        
        # Performance
        fp16=False,  # Disable fp16 with bitsandbytes
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        
        # WICHTIG: Disable distributed training for single GPU
        ddp_backend=None,
        local_rank=-1,
        
        # Other
        seed=42,
        report_to=[],
    )
    
    logger.info("\n📋 Training Configuration:")
    logger.info(f"   Epochs: {training_args.num_train_epochs}")
    logger.info(f"   Batch Size: {training_args.per_device_train_batch_size}")
    logger.info(f"   Gradient Accum: {training_args.gradient_accumulation_steps}")
    logger.info(f"   Learning Rate: {training_args.learning_rate}")
    
    # ========== DATA COLLATOR ==========
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # ========== TRAINER ==========
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # ========== TRAIN ==========
    logger.info("\n🔥 Starting training...\n")
    trainer.train()
    
    # ========== SAVE ==========
    logger.info("\n💾 Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"✅ Model saved to {OUTPUT_DIR}")
    
    logger.info("\n✨ Fine-tuning complete!")

if __name__ == "__main__":
    main()