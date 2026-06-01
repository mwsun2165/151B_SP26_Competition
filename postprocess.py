#!/usr/bin/env python3

import argparse
import csv
import json
import re
from collections import Counter
from tqdm import tqdm
import pandas as pd

from judger import Judger


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────

def infer_format(path):
    lower = path.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".jsonl"):
        return "jsonl"
    raise ValueError("Could not infer input format. Use --input_format csv or jsonl.")


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_csv_df(path):
    """
    Load Kaggle/private submission CSV using pandas.

    dtype=str preserves IDs exactly and avoids type coercion.
    keep_default_na=False prevents empty strings from becoming NaN.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_csv_df(path, df):
    """
    Write CSV using pandas while preserving column order.
    """
    df.to_csv(path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Basic helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_mcq(item):
    return bool(item.get("is_mcq", False) or item.get("options"))


def gold_list(item):
    gold = item.get("gold", item.get("answer", []))
    if isinstance(gold, list):
        return [str(x) for x in gold]
    return [str(gold)]


def expected_answer_count(item):
    if is_mcq(item):
        return 1

    question = item.get("question", "")
    n_ans = question.count("[ANS]")

    if n_ans > 0:
        return n_ans

    if item.get("gold") is not None or item.get("answer") is not None:
        return len(gold_list(item))

    return 1


def safe_extract(judger, response):
    try:
        return judger.extract_ans(response)
    except Exception:
        return ""


def safe_split(judger, extracted):
    try:
        return judger.split_by_comma(extracted)
    except Exception:
        return []


def extracted_count(judger, response):
    extracted = safe_extract(judger, response)
    if not extracted:
        return 0
    return len(safe_split(judger, extracted))


def structurally_valid(item, response, judger):
    extracted = safe_extract(judger, response)
    if not extracted:
        return False

    if is_mcq(item):
        return bool(re.fullmatch(r"\s*[A-Z]\s*", extracted.strip()))

    return extracted_count(judger, response) == expected_answer_count(item)


def score_response(item, response, judger):
    if is_mcq(item):
        gold = str(gold_list(item)[0]).strip().upper()
        pred = safe_extract(judger, response).strip().upper()
        return pred == gold

    gl = gold_list(item)

    try:
        return judger.auto_judge(
            pred=response,
            gold=gl,
            options=[[]] * len(gl),
        )
    except Exception:
        return False

def load_problem_metadata(paths):
    """
    Load public/private JSONL problem metadata keyed by string id.

    Expected fields in each JSONL row:
      id, question, options optional
    """
    meta = {}

    for path in paths:
        if not path:
            continue

        try:
            rows = load_jsonl(path)
        except FileNotFoundError:
            print(f"Metadata file not found, skipping: {path}")
            continue

        for row in rows:
            if "id" not in row:
                continue

            key = str(row["id"])
            meta[key] = {
                "id": key,
                "question": row.get("question", ""),
                "options": row.get("options"),
                "is_mcq": bool(row.get("options")),
            }

    print(f"Loaded metadata for {len(meta)} problems.")
    return meta


def attach_metadata(row_dict, metadata_by_id):
    """
    Merge CSV row with metadata from public/private JSONL.
    CSV fields win only for id/response; metadata supplies question/options/is_mcq.
    """
    item = dict(row_dict)
    key = str(item.get("id", ""))

    if key in metadata_by_id:
        item.update(metadata_by_id[key])
        item["response"] = row_dict.get("response", "")

    return item


# ─────────────────────────────────────────────────────────────────────────────
# Box extraction
# ─────────────────────────────────────────────────────────────────────────────

def find_boxed_contents(text):
    results = []
    start = 0
    needle = r"\boxed{"

    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break

        brace_start = idx + len(needle)
        depth = 1
        i = brace_start

        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1

        if depth == 0:
            content = text[brace_start:i - 1].strip()
            if content:
                results.append(content)
            start = i
        else:
            break

    return results


def clean_box_content(content):
    return str(content).strip().strip("$").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing repairs
# ─────────────────────────────────────────────────────────────────────────────

def repair_wrong_answer_count(item, response, judger):
    if is_mcq(item):
        return response

    k = expected_answer_count(item)
    if k <= 0:
        return response

    if extracted_count(judger, response) == k:
        return response

    think_end = response.rfind("</think>")
    post_think = response[think_end + len("</think>"):] if think_end >= 0 else response

    boxes = find_boxed_contents(post_think)
    if len(boxes) < k:
        boxes = find_boxed_contents(response)

    if len(boxes) < k:
        return response

    selected = [clean_box_content(x) for x in boxes[-k:]]
    final_line = f"Final answer: \\boxed{{{', '.join(selected)}}}"
    repaired = response.rstrip() + "\n\n" + final_line

    if extracted_count(judger, repaired) == k:
        return repaired

    return response


def repair_no_box_from_answer_phrase(item, response, judger):
    if r"\boxed{" in response:
        return response

    tail = response[-2000:]

    patterns = [
        r"(?:final answer|answer)\s*(?:is|:)\s*\$?([^.\n$]+)\$?",
        r"(?:therefore|thus),?\s*(?:the answer is)?\s*\$?([^.\n$]+)\$?",
    ]

    for pat in patterns:
        matches = re.findall(pat, tail, flags=re.IGNORECASE)
        if not matches:
            continue

        ans = matches[-1].strip()
        if not (0 < len(ans) <= 200):
            continue

        repaired = response.rstrip() + f"\n\nFinal answer: \\boxed{{{ans}}}"

        if safe_extract(judger, repaired):
            return repaired

    return response


def repair_mcq_format(item, response, judger):
    if not is_mcq(item):
        return response

    extracted = safe_extract(judger, response).strip().upper()
    if re.fullmatch(r"[A-Z]", extracted):
        return response

    boxes = find_boxed_contents(response)
    if boxes:
        last = boxes[-1].strip().upper()
        m = re.search(r"\b([A-Z])\b", last)
        if m:
            letter = m.group(1)
            return response.rstrip() + f"\n\nFinal answer: \\boxed{{{letter}}}"

    tail = response[-1500:].upper()
    patterns = [
        r"FINAL ANSWER\s*:\s*(?:\\BOXED\{)?\s*(?:OPTION\s*)?([A-Z])",
        r"ANSWER\s*(?:IS|:)\s*(?:OPTION\s*)?([A-Z])\b",
        r"OPTION\s+([A-Z])\b",
    ]

    for pat in patterns:
        matches = re.findall(pat, tail)
        if matches:
            letter = matches[-1]
            return response.rstrip() + f"\n\nFinal answer: \\boxed{{{letter}}}"

    return response


def postprocess_response(item, response, judger):
    if is_mcq(item):
        return repair_mcq_format(item, response, judger)

    before_valid = structurally_valid(item, response, judger)

    if not before_valid:
        response = repair_wrong_answer_count(item, response, judger)

    if r"\boxed{" not in response:
        response = repair_no_box_from_answer_phrase(item, response, judger)

    return response


# ─────────────────────────────────────────────────────────────────────────────
# Public JSONL classification
# ─────────────────────────────────────────────────────────────────────────────

def classify(item, response, correct, judger):
    if correct:
        return "correct"

    if not response.strip():
        return "empty_response"

    extracted = safe_extract(judger, response)

    if not extracted:
        if r"\boxed{" not in response:
            return "no_boxed_answer"
        return "extract_empty"

    if is_mcq(item):
        if not re.fullmatch(r"\s*[A-Z]\s*", extracted.strip()):
            return "mcq_bad_format"
        return "mcq_wrong_letter"

    pred_count = extracted_count(judger, response)
    gold_count = len(gold_list(item))

    if pred_count != gold_count:
        return f"wrong_answer_count_pred_{pred_count}_gold_{gold_count}"

    return "freeform_wrong_math"


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────

def process_private_csv(input_path, output_path, metadata_by_id=None):
    """
    Private Kaggle CSV mode.

    Expected CSV columns:
      id,response

    Uses data/private.jsonl and/or data/public.jsonl metadata if provided
    so post-processing can know question, options, is_mcq, and [ANS] count.
    """
    judger = Judger(strict_extract=False)
    df = load_csv_df(input_path)

    if "response" not in df.columns:
        raise ValueError("CSV must contain a 'response' column.")
    if "id" not in df.columns:
        raise ValueError("CSV must contain an 'id' column.")

    metadata_by_id = metadata_by_id or {}

    changed = 0
    new_responses = []

    for _, row in df.iterrows():
        csv_item = row.to_dict()
        item = attach_metadata(csv_item, metadata_by_id)

        response = item.get("response", "") or ""
        post_response = postprocess_response(item, response, judger)

        if post_response != response:
            changed += 1

        new_responses.append(post_response)

    df["response"] = new_responses

    # Final Kaggle format.
    df = df[["id", "response"]]

    write_csv_df(output_path, df)

    print(f"Processed CSV: {input_path}")
    print(f"Rows: {len(df)}")
    print(f"Rows changed: {changed}")
    print(f"Saved to: {output_path}")


def process_public_jsonl(input_path, output_path):
    """
    Public/eval JSONL mode.
    Reruns judging before and after postprocessing.
    """
    judger = Judger(strict_extract=False)
    rows = load_jsonl(input_path)

    before_counts = Counter()
    after_counts = Counter()

    improved = []
    worsened = []
    changed = 0

    out_rows = []

    for row in rows:
        response = row.get("response", "") or ""

        before_correct = score_response(row, response, judger)
        before_bucket = classify(row, response, before_correct, judger)

        post_response = postprocess_response(row, response, judger)

        after_correct = score_response(row, post_response, judger)
        after_bucket = classify(row, post_response, after_correct, judger)

        before_counts[before_bucket] += 1
        after_counts[after_bucket] += 1

        if post_response != response:
            changed += 1

        if (not before_correct) and after_correct:
            improved.append(row.get("id"))
        elif before_correct and (not after_correct):
            worsened.append(row.get("id"))

        new_row = dict(row)
        new_row["response_original"] = response
        new_row["response"] = post_response
        new_row["correct_before"] = before_correct
        new_row["correct_after"] = after_correct
        new_row["bucket_before"] = before_bucket
        new_row["bucket_after"] = after_bucket
        new_row["postprocessed_changed"] = post_response != response
        out_rows.append(new_row)

    total = len(rows)
    before_correct_n = before_counts["correct"]
    after_correct_n = after_counts["correct"]

    print("=" * 80)
    print("POSTPROCESSING EVALUATION")
    print("=" * 80)
    print(f"Total rows:              {total}")
    print(f"Rows changed:            {changed}")
    print(f"Before correct:          {before_correct_n}/{total} ({before_correct_n / total:.2%})")
    print(f"After correct:           {after_correct_n}/{total} ({after_correct_n / total:.2%})")
    print(f"Net improvement:         {after_correct_n - before_correct_n:+d}")
    print(f"Improved wrong->right:   {len(improved)}")
    print(f"Worsened right->wrong:   {len(worsened)}")

    print("\nBefore buckets:")
    for k, v in before_counts.most_common():
        print(f"  {k:40s} {v:5d}")

    print("\nAfter buckets:")
    for k, v in after_counts.most_common():
        print(f"  {k:40s} {v:5d}")

    print("\nExample improved IDs:")
    print(improved[:5])

    print("\nExample worsened IDs:")
    print(worsened[:5])

    write_jsonl(output_path, out_rows)
    print(f"\nSaved postprocessed results to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input .jsonl result file or .csv Kaggle submission")
    parser.add_argument("--output", required=True, help="Output .jsonl or .csv")
    parser.add_argument("--input_format", choices=["jsonl", "csv"], default=None)
    parser.add_argument(
        "--private_jsonl",
        default="data/private.jsonl",
        help="Path to private.jsonl for question/options metadata.",
    )
    parser.add_argument(
        "--public_jsonl",
        default="data/public.jsonl",
        help="Path to public.jsonl for question/options metadata.",
    )
    args = parser.parse_args()

    input_format = args.input_format or infer_format(args.input)

    if input_format == "csv":
        metadata_by_id = load_problem_metadata([args.private_jsonl, args.public_jsonl])
        process_private_csv(args.input, args.output, metadata_by_id=metadata_by_id)
    elif input_format == "jsonl":
        process_public_jsonl(args.input, args.output)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")


if __name__ == "__main__":
    main()