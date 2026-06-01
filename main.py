# %%
import json
import os
import re
import sys
import random

from tqdm import tqdm
from pathlib import Path
from typing import Optional

import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import pandas as pd

import copy
from collections import Counter, defaultdict

sys.path.insert(0, ".")
from judger import Judger
judger = Judger(strict_extract=False)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"                    # CUDA_VISIBLE_DEVICES
DATA_PATH   = "data/public.jsonl"
OUTPUT_PATH = "results/starter_results.jsonl"
MAX_TOKENS = 32768

SUBMISSION_PATH = "submission.csv"

# os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

data = [json.loads(line) for line in open(DATA_PATH)]

n_mcq  = sum(bool(d.get("options")) for d in data)
n_free = sum(not d.get("options")   for d in data)
print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem carefully. "
    "Do not use \\boxed{} except for the final answer. "
    "End with exactly one final line of the form Final answer: \\boxed{...}. "
    "For multiple [ANS] placeholders, put all answers in order inside one box, separated by commas. "
    "If an [ANS] placeholder has multiple solutions, put all solutions inside one parentheses, separated by commas. "
    "e.g. \\boxed{(3, 7), x/2} has solutions 3 and 7 for the first [ANS] and solution x/2 for the second [ANS]."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"

    n_ans = question.count("[ANS]")
    user_prompt = (
        f"This problem has exactly {n_ans} [ANS] placeholder(s). "
        f"Your final boxed answer must contain exactly {n_ans} answer(s), in the same order. "
        f"If there are multiple answers, separate them by commas inside one box."
        f"If there are multiple solutions for one answer, separate them by commas inside one pair of parentheses.\n\n"
        f"{question}"
    )
    return SYSTEM_PROMPT_MATH, user_prompt

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

llm = LLM(
    model=MODEL_ID,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    enable_prefix_caching=False,
    gpu_memory_utilization=0.90,
    max_model_len=24576,
    trust_remote_code=True,
    max_num_seqs=256,
    max_num_batched_tokens=32768,
)

print("Model loaded.")

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
)

print("Sampling params loaded.")

def is_valid_response(item, response, judger):
    is_mcq = bool(item.get("options"))
    extracted = judger.extract_ans(response)

    if not extracted:
        return False

    if is_mcq:
        return bool(re.fullmatch(r"\s*[A-Z]\s*", extracted))

    expected = item["question"].count("[ANS]")
    pred_count = len(judger.split_by_comma(extracted))
    return pred_count == expected

# Generation strategy for majority voting
MCQ_NUM_SAMPLES = 5
FRQ_SINGLE_NUM_SAMPLES = 1
FRQ_MULTI_NUM_SAMPLES = 3

def expected_answer_count(item):
    if is_mcq(item):
        return 1

    n = item["question"].count("[ANS]")

    # Some private FRQs appear to expect one answer even without [ANS]
    return max(1, n)

def is_mcq(item):
    return bool(item.get("options"))


def sample_count_for_item(item):
    """Choose number of generations based on question type."""
    if is_mcq(item):
        return MCQ_NUM_SAMPLES

    n_ans = expected_answer_count(item)
    if n_ans <= 1:
        return FRQ_SINGLE_NUM_SAMPLES

    return FRQ_MULTI_NUM_SAMPLES


def extract_answer_safe(response):
    try:
        return judger.extract_ans(response)
    except Exception:
        return ""


def normalize_frq_answer_key(extracted):
    """
    Convert extracted FRQ answer into a normalized tuple for voting.
    Returns None if normalization fails.
    """
    if not extracted:
        return None

    try:
        parts = judger.split_by_comma(extracted)
        return tuple(judger.norm_ans_str(p) for p in parts)
    except Exception:
        return None


def extract_mcq_letter_key(extracted):
    """
    Convert extracted MCQ answer into a single uppercase letter for voting.
    Returns None if invalid.
    """
    if not extracted:
        return None

    extracted = extracted.strip().upper()

    if re.fullmatch(r"[A-Z]", extracted):
        return extracted

    return None


def candidate_key(item, response):
    """
    Return a vote key for a candidate response.
    Invalid candidates return None.
    """
    extracted = extract_answer_safe(response)

    if is_mcq(item):
        return extract_mcq_letter_key(extracted)

    key = normalize_frq_answer_key(extracted)
    if key is None:
        return None

    if len(key) != expected_answer_count(item):
        return None

    return key


def is_valid_response(item, response):
    return candidate_key(item, response) is not None

# Build prompts
prompts = []

for item in data:
    system, user = build_prompt(item["question"], item.get("options"))
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompts.append(prompt_text)

print(f"Built {len(prompts)} prompts.")

def sampling_params_with_n(base_sampling_params, n):
    """
    Copy existing sampling params and set n completions per prompt.
    """
    params = copy.deepcopy(base_sampling_params)
    params.n = n
    return params


def generate_candidates_for_indices(indices, n):
    """
    Generate n candidate responses for each selected index.
    Returns dict: index -> list[str].
    """
    if not indices:
        return {}

    selected_prompts = [prompts[i] for i in indices]
    params = sampling_params_with_n(sampling_params, n)

    outputs = llm.generate(selected_prompts, sampling_params=params)

    results = {}
    for idx, out in zip(indices, outputs):
        results[idx] = [candidate.text.strip() for candidate in out.outputs]

    return results


def generate_candidates_grouped(indices):
    """
    Group indices by required sample count so vLLM calls use consistent n.
    Returns dict: index -> list[str].
    """
    grouped = defaultdict(list)

    for idx in indices:
        n = sample_count_for_item(data[idx])
        grouped[n].append(idx)

    all_results = {}

    for n, group_indices in sorted(grouped.items()):
        print(f"Generating {n} sample(s) each for {len(group_indices)} questions...")
        group_results = generate_candidates_for_indices(group_indices, n)
        all_results.update(group_results)

    return all_results

def choose_best_candidate(item, candidates):
    """
    Pick the best candidate using validity filtering + majority vote.

    Returns:
        chosen_response: str
        info: dict with vote metadata
    """
    valid = []

    for pos, response in enumerate(candidates):
        key = candidate_key(item, response)
        if key is not None:
            valid.append({
                "pos": pos,
                "response": response,
                "key": key,
            })

    # If no valid candidates, fall back to the first raw candidate.
    if not valid:
        return candidates[0] if candidates else "", {
            "valid_count": 0,
            "num_candidates": len(candidates),
            "winning_key": None,
            "vote_count": 0,
        }

    # Count votes by normalized answer key.
    counts = Counter(v["key"] for v in valid)
    winning_key, vote_count = counts.most_common(1)[0]

    # Use first valid candidate that produced the winning key.
    for v in valid:
        if v["key"] == winning_key:
            return v["response"], {
                "valid_count": len(valid),
                "num_candidates": len(candidates),
                "winning_key": winning_key,
                "vote_count": vote_count,
            }

    # Should never reach here.
    return valid[0]["response"], {
        "valid_count": len(valid),
        "num_candidates": len(candidates),
        "winning_key": valid[0]["key"],
        "vote_count": 1,
    }


def select_responses_from_candidates(candidate_map):
    """
    Convert index -> candidates into final selected responses.
    Returns:
        selected: dict index -> response
        vote_info: dict index -> metadata
    """
    selected = {}
    vote_info = {}

    for idx, candidates in candidate_map.items():
        response, info = choose_best_candidate(data[idx], candidates)
        selected[idx] = response
        vote_info[idx] = info

    return selected, vote_info

print(f"Generating initial candidates for {len(prompts)} questions...")

responses = [""] * len(prompts)
all_vote_info = {}

initial_indices = list(range(len(prompts)))
initial_candidate_map = generate_candidates_grouped(initial_indices)

initial_selected, initial_vote_info = select_responses_from_candidates(initial_candidate_map)

for idx, response in initial_selected.items():
    responses[idx] = response
    all_vote_info[idx] = initial_vote_info[idx]

initial_valid = sum(is_valid_response(data[i], responses[i]) for i in range(len(responses)))
print(f"Initial valid selected responses: {initial_valid}/{len(responses)}")

valid_count = 0
invalid_count = 0

vote_valid_counts = Counter()
vote_agreement_counts = Counter()

for i, response in enumerate(responses):
    if is_valid_response(data[i], response):
        valid_count += 1
    else:
        invalid_count += 1

    info = all_vote_info.get(i, {})
    vote_valid_counts[info.get("valid_count", 0)] += 1
    vote_agreement_counts[info.get("vote_count", 0)] += 1

print(f"Final valid responses: {valid_count}/{len(responses)}")
print(f"Final invalid responses: {invalid_count}/{len(responses)}")

print("\nValid candidate counts among selected generations:")
for k, v in sorted(vote_valid_counts.items()):
    print(f"  {k} valid candidate(s): {v}")

print("\nWinning vote counts:")
for k, v in sorted(vote_agreement_counts.items()):
    print(f"  winning vote count {k}: {v}")

# Build submission rows from generated responses.
# Assumes:
# - data is the loaded private.jsonl/public.jsonl list of dicts
# - responses[i] is the selected full model response for data[i]
submission_df = pd.DataFrame({
    "id": [item["id"] for item in data],
    "response": responses,
})

# Basic checks
assert len(submission_df) == len(data), "Submission row count does not match data length."
assert submission_df["id"].isna().sum() == 0, "Some ids are missing."
assert submission_df["response"].isna().sum() == 0, "Some responses are missing."

# Optional: check for empty responses
empty_count = (submission_df["response"].astype(str).str.len() == 0).sum()
print(f"Empty responses: {empty_count}")

# Write CSV with proper quoting/escaping handled by pandas
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"Wrote {len(submission_df)} rows to {SUBMISSION_PATH}")
submission_df.head()
