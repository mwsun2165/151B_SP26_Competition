#!/usr/bin/env python3
"""
QLoRA SFT for Qwen3-4B-Thinking-2507 on AI-MO/NuminaMath-CoT.

Install:
pip install -U torch transformers datasets peft trl bitsandbytes accelerate huggingface_hub

Local run:
python train_qlora_numina.py

Upload adapter to HF:
python train_qlora_numina.py \
  --push_to_hub \
  --hub_model_id your-username/qwen3-4b-thinking-numina-qlora-5k \
  --hf_token YOUR_TOKEN
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

MAX_SEQ_LENGTH = 4096
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.push_to_hub:
        if args.hf_token:
            login(token=args.hf_token)
        elif os.environ.get("HF_TOKEN"):
            login(token=os.environ["HF_TOKEN"])

        if not args.hub_model_id:
            raise ValueError("--hub_model_id is required when --push_to_hub is set.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print(f"Loading dataset: {args.dataset_id}")
    raw = load_dataset(args.dataset_id)

    split_name = "train" if "train" in raw else list(raw.keys())[0]
    raw_train = raw[split_name]

    print(f"Using split: {split_name}")
    print(f"Raw rows available: {len(raw_train)}")

    scan_count = min(len(raw_train), args.num_train_examples)
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

    train_dataset = processed.select(range(min(len(processed), args.num_train_examples)))

    print(f"Training examples: {len(train_dataset)}")
    print("Example training text preview:")
    print(train_dataset[0]["text"][:2000])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        packing=False,

        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,

        optim="paged_adamw_8bit",
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",

        report_to="none",

        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private_repo=True if args.push_to_hub else None,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        dataset_text_field="text",
    )

    print("Starting QLoRA training...")
    trainer.train()

    print(f"Saving adapter locally to: {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        print(f"Pushing adapter to Hub: {args.hub_model_id}")
        trainer.model.push_to_hub(args.hub_model_id, private=True)
        tokenizer.push_to_hub(args.hub_model_id, private=True)

    print("Done.")


if __name__ == "__main__":
    main()