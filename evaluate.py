import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from judger import Judger

RUN_PATH = "results\iteration_1_1858_532026.jsonl"   # your past public run
PUBLIC_PATH = "data/public.jsonl"            # original public set

judger = Judger(strict_extract=False)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


public = {item["id"]: item for item in load_jsonl(PUBLIC_PATH)}
rows = load_jsonl(RUN_PATH)


def safe_extract(response):
    try:
        return judger.extract_ans(response)
    except Exception as e:
        return ""


def count_boxed(response):
    return response.count(r"\boxed{")


def has_think_close(response):
    return "</think>" in response


def get_expected_answer_count(item):
    if item.get("options"):
        return 1
    answer = item.get("answer")
    if isinstance(answer, list):
        return len(answer)
    return 1


def get_pred_answer_count(extracted):
    if not extracted:
        return 0
    try:
        return len(judger.split_by_comma(extracted))
    except Exception:
        return -1


def classify_failure(row):
    item = public[row["id"]]
    response = row["response"]
    extracted = safe_extract(response)

    is_mcq = bool(item.get("options"))
    expected_count = get_expected_answer_count(item)
    pred_count = get_pred_answer_count(extracted)
    boxed_count = count_boxed(response)

    if row.get("correct") is True:
        return "correct"

    if not response or not response.strip():
        return "empty_response"

    if "maximum context length" in response.lower() or "truncated" in response.lower():
        return "possible_truncation"

    if boxed_count == 0:
        return "no_boxed_answer"

    if not extracted:
        return "extract_empty"

    if not is_mcq and pred_count != expected_count:
        return "wrong_answer_count"

    if is_mcq:
        pred = extracted.strip()
        if not re.fullmatch(r"[A-Za-z]", pred):
            return "mcq_bad_format"
        return "mcq_wrong_letter"

    return "freeform_wrong_math"


diagnostics = []

for row in rows:
    item = public[row["id"]]
    response = row["response"]
    extracted = safe_extract(response)

    diagnostics.append({
        "id": row["id"],
        "correct": row.get("correct"),
        "is_mcq": bool(item.get("options")),
        "gold": item.get("answer"),
        "extracted": extracted,
        "expected_count": get_expected_answer_count(item),
        "pred_count": get_pred_answer_count(extracted),
        "boxed_count": count_boxed(response),
        "has_think_close": has_think_close(response),
        "failure_type": classify_failure(row),
        "question": item["question"][:300].replace("\n", " "),
        "response_preview": response[-500:].replace("\n", " "),
    })

counts = Counter(d["failure_type"] for d in diagnostics)

print("Failure breakdown:")
for k, v in counts.most_common():
    print(f"{k:25s} {v}")

print("\nAccuracy by type:")
for is_mcq in [True, False]:
    subset = [d for d in diagnostics if d["is_mcq"] == is_mcq]
    acc = sum(d["correct"] for d in subset) / len(subset) if subset else 0
    print(("MCQ" if is_mcq else "Free-form"), f"{acc:.2%}", f"({sum(d['correct'] for d in subset)}/{len(subset)})")