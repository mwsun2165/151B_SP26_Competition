import copy
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, ".")
from judger import Judger


# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH = "data/private.jsonl"
SUBMISSION_PATH = "submission.csv"
MAX_TOKENS = 32768

# Generation strategy for majority voting
MCQ_NUM_SAMPLES = 5
FRQ_SINGLE_NUM_SAMPLES = 1
FRQ_MULTI_NUM_SAMPLES = 3

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


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def is_mcq(item: dict) -> bool:
    return bool(item.get("options"))


def expected_answer_count(item: dict) -> int:
    if is_mcq(item):
        return 1
    return max(1, item["question"].count("[ANS]"))


def sample_count_for_item(item: dict) -> int:
    if is_mcq(item):
        return MCQ_NUM_SAMPLES
    if expected_answer_count(item) <= 1:
        return FRQ_SINGLE_NUM_SAMPLES
    return FRQ_MULTI_NUM_SAMPLES


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{label}. {option.strip()}" for label, option in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"

    n_ans = max(1, question.count("[ANS]"))
    user_prompt = (
        f"This problem has exactly {n_ans} answer(s) to provide. "
        f"Your final boxed answer must contain exactly {n_ans} answer(s), in the same order. "
        f"If there are multiple answers, separate them by commas inside one box. "
        f"If there are multiple solutions for one answer, separate them by commas inside one pair of parentheses.\n\n"
        f"{question}"
    )
    return SYSTEM_PROMPT_MATH, user_prompt


def build_prompts(data: list[dict], tokenizer: AutoTokenizer) -> list[str]:
    prompts = []
    for item in data:
        system_prompt, user_prompt = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)
    return prompts


def make_llm() -> LLM:
    return LLM(
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


def make_sampling_params() -> SamplingParams:
    return SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )


def sampling_params_with_n(base_sampling_params: SamplingParams, n: int) -> SamplingParams:
    params = copy.deepcopy(base_sampling_params)
    params.n = n
    return params


def extract_answer_safe(judger: Judger, response: str) -> str:
    try:
        return judger.extract_ans(response)
    except Exception:
        return ""


def extract_mcq_letter_key(extracted: str) -> Optional[str]:
    if not extracted:
        return None

    extracted = extracted.strip().upper()
    if re.fullmatch(r"[A-Z]", extracted):
        return extracted
    return None


def normalize_frq_answer_key(judger: Judger, extracted: str) -> Optional[tuple[str, ...]]:
    if not extracted:
        return None

    try:
        parts = judger.split_by_comma(extracted)
        return tuple(judger.norm_ans_str(part) for part in parts)
    except Exception:
        return None


def candidate_key(judger: Judger, item: dict, response: str):
    extracted = extract_answer_safe(judger, response)

    if is_mcq(item):
        return extract_mcq_letter_key(extracted)

    key = normalize_frq_answer_key(judger, extracted)
    if key is None or len(key) != expected_answer_count(item):
        return None
    return key


def choose_best_candidate(judger: Judger, item: dict, candidates: list[str]) -> str:
    valid_candidates = []

    for response in candidates:
        key = candidate_key(judger, item, response)
        if key is not None:
            valid_candidates.append((key, response))

    if not valid_candidates:
        return candidates[0] if candidates else ""

    winning_key = Counter(key for key, _ in valid_candidates).most_common(1)[0][0]

    for key, response in valid_candidates:
        if key == winning_key:
            return response

    return valid_candidates[0][1]


def generate_candidates_for_indices(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    indices: list[int],
    n: int,
) -> dict[int, list[str]]:
    if not indices:
        return {}

    selected_prompts = [prompts[i] for i in indices]
    params = sampling_params_with_n(sampling_params, n)
    outputs = llm.generate(selected_prompts, sampling_params=params, use_tqdm=True)

    return {
        idx: [candidate.text.strip() for candidate in output.outputs]
        for idx, output in zip(indices, outputs)
    }


def generate_candidates_grouped(
    llm: LLM,
    data: list[dict],
    prompts: list[str],
    sampling_params: SamplingParams,
) -> dict[int, list[str]]:
    grouped_indices = defaultdict(list)
    for idx, item in enumerate(data):
        grouped_indices[sample_count_for_item(item)].append(idx)

    candidate_map = {}
    for n, indices in sorted(grouped_indices.items()):
        print(f"Generating {n} sample(s) each for {len(indices)} questions...")
        candidate_map.update(
            generate_candidates_for_indices(llm, prompts, sampling_params, indices, n)
        )

    return candidate_map


def select_responses(judger: Judger, data: list[dict], candidate_map: dict[int, list[str]]) -> list[str]:
    responses = [""] * len(data)
    for idx, candidates in candidate_map.items():
        responses[idx] = choose_best_candidate(judger, data[idx], candidates)
    return responses


def write_submission(data: list[dict], responses: list[str], path: str) -> None:
    submission_df = pd.DataFrame({
        "id": [item["id"] for item in data],
        "response": responses,
    })

    assert len(submission_df) == len(data), "Submission row count does not match data length."
    assert submission_df["id"].isna().sum() == 0, "Some ids are missing."
    assert submission_df["response"].isna().sum() == 0, "Some responses are missing."

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path(".") else None
    submission_df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")


def run_inference() -> None:
    data = load_jsonl(DATA_PATH)
    print(f"Loaded {len(data)} private questions.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    prompts = build_prompts(data, tokenizer)
    print(f"Built {len(prompts)} prompts.")

    llm = make_llm()
    print("Model loaded.")

    sampling_params = make_sampling_params()
    judger = Judger(strict_extract=False)

    candidate_map = generate_candidates_grouped(llm, data, prompts, sampling_params)
    responses = select_responses(judger, data, candidate_map)
    write_submission(data, responses, SUBMISSION_PATH)

    print(f"Wrote submission to {SUBMISSION_PATH}.")


if __name__ == "__main__":
    run_inference()
