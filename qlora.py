#!/usr/bin/env python3
"""
QLoRA SFT for Qwen3-4B-Thinking-2507 on AI-MO/NuminaMath-CoT.
"""

import argparse
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Top-level constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DATASET_ID = "AI-MO/NuminaMath-CoT"

NUM_TRAIN_EXAMPLES = 5000
OUTPUT_DIR = "./qwen3-4b-thinking-numina-qlora-5k"

SEED = 42

SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the problem carefully and show your reasoning."
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset field handling
# ─────────────────────────────────────────────────────────────────────────────

def get_problem_and_solution(row):
    """
    Extract problem and solution from NuminaMath-CoT row.

    Preferred:
        row["problem"], row["solution"]

    Fallback:
        row["messages"] with user/assistant roles.
    """
    problem = row.get("problem")
    solution = row.get("solution")

    if problem is not None and solution is not None:
        problem = str(problem).strip()
        solution = str(solution).strip()

        if problem and solution:
            return problem, solution

    # Fallback to messages if problem/solution missing or empty.
    messages = row.get("messages")
    if isinstance(messages, list):
        user_contents = []
        assistant_contents = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role")
            content = msg.get("content")

            if content is None:
                continue

            content = str(content).strip()
            if not content:
                continue

            if role == "user":
                user_contents.append(content)
            elif role == "assistant":
                assistant_contents.append(content)

        if user_contents and assistant_contents:
            return user_contents[-1], assistant_contents[-1]

    return None


def make_training_text(tokenizer, problem, solution):
    """
    Format as a chat transcript using the model's chat template.

    No final-answer rewriting is done here. The assistant target is the raw
    dataset solution.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem.strip()},
        {"role": "assistant", "content": solution.strip()},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main training
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading model")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loaded model")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print(f"Loading dataset: {DATASET_ID}")
    raw = load_dataset(DATASET_ID)

    split_name = "train" if "train" in raw else list(raw.keys())[0]
    raw_train = raw[split_name]

    print(f"Using split: {split_name}")
    print(f"Raw rows available: {len(raw_train)}")

    scan_count = min(len(raw_train), NUM_TRAIN_EXAMPLES)
    raw_subset = raw_train.select(range(scan_count))

    def preprocess(row):
        pair = get_problem_and_solution(row)
        if pair is None:
            return {"text": None, "keep": False}

        problem, solution = pair
        text = make_training_text(tokenizer, problem, solution)

        return {"text": text, "keep": True}

    print("Formatting rows...")
    processed = raw_subset.map(
        preprocess,
        remove_columns=raw_subset.column_names,
        desc="Preprocessing",
    )

    processed = processed.filter(lambda x: x["keep"] and x["text"] is not None)
    processed = processed.remove_columns(["keep"])

    train_dataset = processed.select(range(min(len(processed), NUM_TRAIN_EXAMPLES)))

    print(f"Training examples: {len(train_dataset)}")
    print("Example training text preview:")
    print(train_dataset[0]["text"][:2])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        packing=False,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1.0,
        learning_rate=2e-5,
        warmup_ratio=0.03,

        logging_steps=10,
        save_steps=250,
        save_total_limit=2,

        optim="paged_adamw_8bit",
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",

        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer
    )

    print("Starting QLoRA training...")
    trainer.train()

    print(f"Saving adapter locally to: {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()