"""
feedback/dpo_trainer.py
────────────────────────
Fine-tunes a local Llama-3-8B model on user preference data
using Direct Preference Optimization (DPO).

Run this weekly (e.g., via cron) once you have 200+ feedback pairs.

Usage:
    python -m feedback.dpo_trainer --min-pairs 200 --epochs 2

What it does:
  1. Loads DPO pairs from the feedback database
  2. Fine-tunes Llama-3-8B-Instruct with LoRA + DPO
  3. Saves checkpoint to ./checkpoints/dpo-llama3/
  4. The API automatically picks up the new checkpoint on next restart

Requirements:
  - GPU with 16GB+ VRAM (or use Google Colab with A100)
  - Alternatively, run on vast.ai / runpod (~$0.40/hr for A100)
"""

import argparse
import asyncio
import json
from pathlib import Path

from loguru import logger


def train_dpo(pairs: list[dict], output_dir: str = "./checkpoints/dpo-llama3"):
    """
    Run DPO fine-tuning on preference pairs.

    Args:
        pairs      : list of {"prompt": ..., "chosen": ..., "rejected": ...}
        output_dir : where to save the LoRA checkpoint
    """
    # These imports are heavy — only load when actually training
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    import torch

    BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
    logger.info(f"Loading base model: {BASE_MODEL}")

    # 4-bit quantization so it fits on a 16GB GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # LoRA config — trains only ~0.1% of parameters
    lora_config = LoraConfig(
        r=16,                    # rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Build HuggingFace Dataset
    dataset = Dataset.from_list([
        {
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ])

    # Split 90/10 train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)

    # DPO training config
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,        # effective batch = 8
        learning_rate=5e-5,
        beta=0.1,                             # DPO temperature (lower = less conservative)
        max_length=1024,
        max_prompt_length=512,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",                     # set to "wandb" if you want W&B tracking
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                       # None = use frozen copy of model
        args=dpo_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    logger.info(f"Starting DPO training on {len(split['train'])} pairs...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.success(f"DPO training complete. Checkpoint saved to {output_dir}")
    return output_dir


async def main(min_pairs: int = 200, epochs: int = 2):
    """Load pairs from DB and run DPO training."""
    from feedback.collector import FeedbackCollector

    collector = FeedbackCollector()
    pairs = await collector.get_dpo_pairs(limit=10000)

    if len(pairs) < min_pairs:
        logger.warning(
            f"Only {len(pairs)} DPO pairs available. "
            f"Need at least {min_pairs}. Skipping training."
        )
        return

    logger.info(f"Training on {len(pairs)} preference pairs")
    train_dpo(pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-pairs", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(main(min_pairs=args.min_pairs, epochs=args.epochs))
