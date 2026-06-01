#!/usr/bin/env python3
"""
QLoRA SFT for Qwen3-4B-Thinking-2507 on open-r1/OpenR1-Math-220k.

Dataset format expected:
  problem: str
  generations: list[str]
  is_reasoning_complete: list[bool]
  correctness_math_verify: list[bool]
  messages: optional list, not used for training traces here

We train only on generations where:
  is_reasoning_complete[i] == True
  correctness_math_verify[i] == True
"""

import argparse
import os
import random
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
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
DATASET_ID = "open-r1/OpenR1-Math-220k"

# Use only the default split, not extended.
DATASET_SPLIT = "default"

# Number of source dataset rows to scan.
NUM_SOURCE_ROWS = 5000

# If True, train on every complete+correct generation per problem.
# If False, randomly choose one complete+correct generation per problem.
TRAIN_ON_ALL_CORRECT_GENERATIONS = True

OUTPUT_DIR = "./qwen3-4b-thinking-openr1-qlora-5k"
SEED = 42

# Keep conservative for the first run. OpenR1 traces can be long.
MAX_SEQ_LENGTH = 8192

SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the problem carefully and show your reasoning."
)

# Optional fallback Hub target. Can be overridden by --hub_model_id.
DEFAULT_HUB_MODEL_ID = "mwsun/cse151b_competition_openr1"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset parsing for OpenR1-Math-220k
# ─────────────────────────────────────────────────────────────────────────────

def as_bool_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [bool(v) for v in x]
    return []


def as_str_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if v is not None and str(v).strip()]
    return []


def get_problem(row):
    """
    OpenR1-Math-220k has a direct 'problem' column.
    """
    problem = row.get("problem")
    if problem is None:
        return None

    problem = str(problem).strip()
    return problem if problem else None


def get_correct_complete_generations(row):
    """
    Return full assistant responses from row['generations'] where:
      is_reasoning_complete[i] == True
      correctness_math_verify[i] == True

    Each generation already contains the full response, including:
      <think>...</think>
      final answer/output
    """
    generations = row.get("generations") or []
    complete_flags = row.get("is_reasoning_complete") or []
    correct_flags = row.get("correctness_math_verify") or []

    good = []
    n = min(len(generations), len(complete_flags), len(correct_flags))

    for i in range(n):
        generation = generations[i]

        if generation is None:
            continue

        generation = str(generation).strip()

        if not generation:
            continue

        if bool(complete_flags[i]) and bool(correct_flags[i]):
            good.append(generation)

    return good


def make_training_text(tokenizer, problem, generation):
    """
    Format problem + selected full OpenR1 response as a chat transcript.

    No rewriting is done:
      - preserves <think>...</think>
      - preserves the final answer/output
      - preserves the response format from the dataset
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem.strip()},
        {"role": "assistant", "content": generation.strip()},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def build_training_examples(raw_subset, tokenizer):
    """
    Expand each source row into one or more SFT examples.

    If TRAIN_ON_ALL_CORRECT_GENERATIONS=True:
        train on every complete+correct generation for the problem.

    If False:
        randomly choose one complete+correct generation per problem.
    """
    rng = random.Random(SEED)
    examples = []

    skipped_no_problem = 0
    skipped_no_good_generation = 0
    skipped_too_long = 0

    for row in raw_subset:
        problem = get_problem(row)
        if problem is None:
            skipped_no_problem += 1
            continue

        good_generations = get_correct_complete_generations(row)
        if not good_generations:
            skipped_no_good_generation += 1
            continue

        if TRAIN_ON_ALL_CORRECT_GENERATIONS:
            selected_generations = good_generations
        else:
            selected_generations = [rng.choice(good_generations)]

        for generation in selected_generations:
            text = make_training_text(tokenizer, problem, generation)

            # Avoid truncating away final answers / closing tags.
            n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            if n_tokens > MAX_SEQ_LENGTH:
                skipped_too_long += 1
                continue

            examples.append({
                "text": text,
                "num_tokens": n_tokens,
            })

    print("Dataset build summary:")
    print(f"  built examples:              {len(examples)}")
    print(f"  skipped_no_problem:          {skipped_no_problem}")
    print(f"  skipped_no_good_generation:  {skipped_no_good_generation}")
    print(f"  skipped_too_long:            {skipped_too_long}")
    print(f"  train_on_all_correct:        {TRAIN_ON_ALL_CORRECT_GENERATIONS}")

    if not examples:
        raise RuntimeError("No training examples were built. Check dataset fields/split.")

    return Dataset.from_list(examples)


# ─────────────────────────────────────────────────────────────────────────────
# Args / training
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=DEFAULT_HUB_MODEL_ID)
    parser.add_argument("--hf_token", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--num_source_rows", type=int, default=NUM_SOURCE_ROWS)
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    if args.push_to_hub:
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if token is None:
            raise ValueError(
                "Set --hf_token or HF_TOKEN env var when --push_to_hub is used."
            )
        login(token=token)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print(f"Loading dataset: {DATASET_ID}, split=train")
    raw_train = load_dataset(DATASET_ID, split="train")
    
    print(f"Raw rows available in {DATASET_SPLIT}: {len(raw_train)}")
    
    scan_count = min(len(raw_train), args.num_source_rows)
    raw_subset = raw_train.select(range(scan_count))
    
    print(f"Scanning first {scan_count} source rows...")
    train_dataset = build_training_examples(raw_subset, tokenizer)
    
    print(f"Training examples after expansion/filtering: {len(train_dataset)}")
    print("Example training text preview:")
    print(train_dataset[0]["text"][:2000])

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
        packing=False,

        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
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

        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        hub_private_repo=False if args.push_to_hub else None,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.remove_columns(
            [c for c in train_dataset.column_names if c != "text"]
        ),
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting QLoRA training...")
    trainer.train()

    print(f"Saving adapter locally to: {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        print(f"Pushing adapter to Hub: {args.hub_model_id}")
        trainer.model.push_to_hub(args.hub_model_id, private=False)
        tokenizer.push_to_hub(args.hub_model_id, private=False)

    print("Done.")


if __name__ == "__main__":
    main()