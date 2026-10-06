import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


METRIC_KEYS = [
    "Faithfulness-Step",
    "Informativeness-Step",
    "Risk Assessment Accuracy",
    "Traffic Rule Adherence",
    "Scene Awareness & Object Understanding",
    "Repetition-Token",
    "Hallucination",
    "Semantic Coverage-Step",
    "Commonsense Reasoning",
    "Missing Step",
    "Relevance",
    "Missing Details",
    "Overall Score",
]

MCQ_QUESTION_SUFFIXES = {2, 3, 4, 5, 8}


SYSTEM_PROMPT = """
You are an autonomous driving reasoning evaluator. Your task is to assess the alignment, coherence, and quality of reasoning steps in text responses for safety-critical driving scenarios.

You will evaluate the model-generated reasoning using the following metrics:

1. Faithfulness-Step (1-10)
Measures how well the model's reasoning steps align with the ground truth.
9-10: All steps correctly match or closely reflect the reference.
7-8: Most steps align, with minor deviations.
5-6: Some steps align, but several are incorrect or missing.
3-4: Few steps align; most are inaccurate or missing.
1-2: Majority of steps are incorrect.

2. Informativeness-Step (1-10)
Measures completeness of reasoning:
9-10: Captures almost all critical information.
7-8: Covers most key points, with minor omissions.
5-6: Missing significant details.
3-4: Only partial reasoning present.
1-2: Poor extraction of relevant reasoning.

3. Risk Assessment Accuracy (1-10)
Evaluates if the model correctly prioritizes high-risk objects or scenarios.
9-10: Correctly identifies and prioritizes key dangers.
7-8: Mostly accurate, with minor misprioritizations.
5-6: Some important risks are overlooked.
3-4: Significant misjudgments in risk prioritization.
1-2: Misidentifies key risks or misses them entirely.

4. Traffic Rule Adherence (1-10)
Evaluates whether the response follows traffic laws and driving best practices.
9-10: Fully compliant with legal and safe driving practices.
7-8: Minor deviations, but mostly correct.
5-6: Some inaccuracies in legal/safe driving recommendations.
3-4: Several rule violations or unsafe suggestions.
1-2: Promotes highly unsafe driving behavior.

5. Scene Awareness & Object Understanding (1-10)
Measures how well the response interprets objects, their positions, and actions.
9-10: Clearly understands all relevant objects and their relationships.
7-8: Minor misinterpretations but mostly correct.
5-6: Some key objects misunderstood or ignored.
3-4: Many errors in object recognition and reasoning.
1-2: Misidentifies or ignores key objects.

6. Repetition-Token (1-10)
Identifies unnecessary repetition in reasoning.
9-10: No redundancy, very concise.
7-8: Minor repetition but still clear.
5-6: Noticeable redundancy.
3-4: Frequent repetition that disrupts reasoning.
1-2: Excessive redundancy, making reasoning unclear.

7. Hallucination (1-10)
Detects irrelevant or invented reasoning steps not aligned with ground truth.
9-10: No hallucinations, all reasoning is grounded.
7-8: One or two minor hallucinations.
5-6: Some fabricated details.
3-4: Frequent hallucinations.
1-2: Majority of reasoning is hallucinated.

8. Semantic Coverage-Step (1-10)
Checks if the response fully covers the critical reasoning elements.
9-10: Nearly complete semantic coverage.
7-8: Good coverage, some minor omissions.
5-6: Partial coverage with key gaps.
3-4: Major gaps in reasoning.
1-2: Very poor semantic coverage.

9. Commonsense Reasoning (1-10)
Assesses the use of intuitive driving logic in reasoning.
9-10: Displays strong commonsense understanding.
7-8: Mostly correct, with minor gaps.
5-6: Some commonsense errors.
3-4: Frequent commonsense mistakes.
1-2: Lacks basic driving commonsense.

10. Missing Step (1-10)
Evaluates if any necessary reasoning steps are missing.
9-10: No critical steps missing.
7-8: Minor missing steps, but answer is mostly intact.
5-6: Some important steps missing.
3-4: Many critical reasoning gaps.
1-2: Response is highly incomplete.

11. Relevance (1-10)
Measures how well the response is specific to the given scenario and ground truth.
9-10: Highly specific and directly relevant to the driving scenario.
7-8: Mostly relevant, but some minor parts may be overly generic or slightly off-focus.
5-6: Somewhat relevant but lacks precision.
3-4: Mostly generic or off-topic reasoning.
1-2: Largely irrelevant.

12. Missing Details (1-10)
Evaluates the extent to which critical information is missing from the response.
9-10: No significant details are missing.
7-8: Covers most important details, with minor omissions.
5-6: Some essential details are missing.
3-4: Many critical reasoning steps or contextual details are absent.
1-2: Response is highly lacking in necessary details.

Final Evaluation:

Compute the Overall Score as the average of all metric scores.
Avoid subjective interpretation and adhere to the given thresholds.
Do not add any additional explanations beyond the structured JSON output.
"""


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "EvaluationScores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {key: {"type": "number"} for key in METRIC_KEYS},
            "required": METRIC_KEYS,
            "additionalProperties": False,
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DriveLMM-o1 predictions with GPT-as-judge metrics.")
    parser.add_argument("--prediction", required=True, help="Prediction JSON produced by evaluation/inference.py.")
    parser.add_argument("--dataset", default="DriveLMMo1_TEST.json", help="DriveLMM-o1 metadata JSON.")
    parser.add_argument("--output-json", help="Per-sample evaluation JSON. Defaults to <prediction>_eval.json.")
    parser.add_argument("--output-csv", help="Per-sample evaluation CSV. Defaults to <prediction>_eval.csv.")
    parser.add_argument("--summary-json", help="Summary JSON. Defaults to <prediction>_summary.json.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Judge model name.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing the API key.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    return parser.parse_args()


def default_output_paths(prediction_path):
    stem = prediction_path.with_suffix("")
    return (
        Path(f"{stem}_eval.json"),
        Path(f"{stem}_eval.csv"),
        Path(f"{stem}_summary.json"),
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_final_answer(text):
    markers = [
        "The final answer is:",
        "**Final Answer:**",
        "Final Answer",
        "Answer",
        "Why take this action?:",
        "**Final Answer**",
        "**Final Decision**:",
        "Final Step:",
        "<CONCLUSION>",
    ]
    for marker in markers:
        if marker in text:
            return text.split(marker)[-1]
    return text


def extract_options(text):
    matches = re.findall(r"([A-F])\)\s*(.+)", text)
    if matches:
        return matches
    if "none of the " in text.lower():
        return [("F", "None of the option.")]
    return [("none", "none")]


def prediction_id(item):
    if "id" in item:
        return item["id"]
    if "idx" in item:
        return item["idx"]
    raise KeyError("Prediction item must contain either 'id' or 'idx'.")


def prediction_answer(item):
    if "answer" in item:
        return item["answer"]
    if "llm-response" in item:
        return item["llm-response"]
    if "prediction" in item:
        return item["prediction"]
    raise KeyError("Prediction item must contain 'answer', 'llm-response', or 'prediction'.")


def build_client(args):
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key. Please set {args.api_key_env}.")
    return OpenAI(api_key=api_key, base_url=args.base_url)


def evaluate_steps(client, args, question, ground_truth, llm_response):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Ground Truth: {ground_truth}\n"
                f"LLM Response: {llm_response}"
            ),
        },
    ]
    last_error = None
    for attempt in range(args.max_retries):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                response_format=RESPONSE_FORMAT,
                max_tokens=500,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as exc:
            last_error = exc
            if attempt + 1 < args.max_retries:
                time.sleep(args.retry_sleep * (attempt + 1))
    raise last_error


def parse_score_json(content):
    content = content.strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()
    scores = json.loads(content)
    if "Commonsense" in scores and "Commonsense Reasoning" not in scores:
        scores["Commonsense Reasoning"] = scores.pop("Commonsense")
    missing = [key for key in METRIC_KEYS if key not in scores]
    if missing:
        raise ValueError(f"Judge response missing metrics: {missing}")
    return {key: float(scores[key]) for key in METRIC_KEYS}


def compute_mcq_score(idx, gt_final_answer, llm_response):
    try:
        question_suffix = int(str(idx).rsplit("_", 1)[-1])
    except ValueError:
        return -1
    if question_suffix not in MCQ_QUESTION_SUFFIXES:
        return -1
    gt_option, _ = extract_options(gt_final_answer)[0]
    pred_option, _ = extract_options(extract_final_answer(llm_response))[0]
    return int(pred_option == gt_option)


def process_item(client, args, data_dict, valid_ids, item):
    idx = prediction_id(item)
    if idx not in valid_ids:
        return None

    question = item.get("question", data_dict[idx]["question"])
    llm_response = prediction_answer(item)
    ground_truth = (
        "**Step-by-Step Reasoning**: "
        + data_dict[idx]["steps"]
        + " **Final Answer**: "
        + data_dict[idx]["final_answer"]
    )
    mcq = compute_mcq_score(idx, data_dict[idx]["final_answer"], llm_response)

    try:
        raw_scores = evaluate_steps(client, args, question, ground_truth, llm_response)
        scores = parse_score_json(raw_scores)
    except Exception as exc:
        print(f"Error processing item {idx}: {exc}")
        return None

    scores.update(
        {
            "id": idx,
            "question": question,
            "ground_truth": ground_truth,
            "llm_response": llm_response,
            "mcq": mcq,
        }
    )
    return scores


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = METRIC_KEYS + ["id", "question", "ground_truth", "llm_response", "mcq"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    if not rows:
        return {"num_samples": 0, "num_mcq": 0, "mcq_accuracy": None, "metrics": {}, "normalized_percent": {}}

    metrics = {key: sum(row[key] for row in rows) / len(rows) for key in METRIC_KEYS}
    normalized = {key: value * 10.0 for key, value in metrics.items()}
    mcq_rows = [row for row in rows if row["mcq"] != -1]
    mcq_accuracy = None
    if mcq_rows:
        mcq_accuracy = sum(row["mcq"] for row in mcq_rows) / len(mcq_rows)
    return {
        "num_samples": len(rows),
        "num_mcq": len(mcq_rows),
        "mcq_accuracy": mcq_accuracy,
        "metrics": metrics,
        "normalized_percent": normalized,
    }


def main():
    args = parse_args()
    prediction_path = Path(args.prediction)
    default_json, default_csv, default_summary = default_output_paths(prediction_path)
    output_json = Path(args.output_json) if args.output_json else default_json
    output_csv = Path(args.output_csv) if args.output_csv else default_csv
    summary_json = Path(args.summary_json) if args.summary_json else default_summary

    predictions = load_json(prediction_path)
    dataset = load_json(args.dataset)
    if args.max_samples is not None:
        predictions = predictions[: args.max_samples]

    valid_ids = {item["idx"] for item in dataset if "idx" in item}
    data_dict = {
        item["idx"]: {
            "image": item.get("image"),
            "question": item["question"],
            "final_answer": item["final_answer"],
            "steps": item["steps"],
        }
        for item in dataset
        if "idx" in item
    }

    client = build_client(args)
    worker_count = max(1, args.num_workers)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(
            tqdm(
                executor.map(lambda item: process_item(client, args, data_dict, valid_ids, item), predictions),
                total=len(predictions),
                desc="Evaluating",
            )
        )
    results = [row for row in results if row is not None]

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    write_csv(output_csv, results)

    summary = summarize(results)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"Evaluated samples: {summary['num_samples']}")
    if summary["mcq_accuracy"] is not None:
        print(f"MCQ Accuracy: {summary['mcq_accuracy']:.2%}")
    for key in [
        "Overall Score",
        "Relevance",
        "Risk Assessment Accuracy",
        "Traffic Rule Adherence",
        "Scene Awareness & Object Understanding",
        "Missing Details",
    ]:
        value = summary["normalized_percent"].get(key)
        if value is not None:
            print(f"{key}: {value:.2f}%")
    print(f"Results saved: JSON -> {output_json}, CSV -> {output_csv}, summary -> {summary_json}")


if __name__ == "__main__":
    main()
