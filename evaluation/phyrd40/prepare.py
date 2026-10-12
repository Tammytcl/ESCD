"""Download/validate PHYRD-40 and separate generation prompts from grading references."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from .rubric_judge import load_problems, read_jsonl

DATASET_REVISION = "ca8120590353a37c0d3be02bdcefc5683f278a2e"
SYSTEM_PROMPT = (
    "You are an expert mathematical physicist solving a research-level problem. "
    "Produce a rigorous, self-contained solution using LaTeX notation. Address every numbered request, "
    "show the essential derivations, define conventions, and check signs, limits, or consistency where "
    "relevant. You have only the problem statement: do not assume or claim access to a reference answer. "
    "Do not use external tools, web sources, or unstated numerical experiments."
)


def prepare(dataset: Path, output_dir: Path, expected_count: int = 40) -> None:
    problems = load_problems(dataset)
    if len(problems) != expected_count:
        raise ValueError(f"expected {expected_count} problems, got {len(problems)}")
    for row in read_jsonl(dataset):
        normalized = problems[row["id"]]
        for field, text in (("problem_sha256", normalized["problem"]),
                            ("reference_sha256", normalized["reference"])):
            if field in row and row[field] != hashlib.sha256(text.encode("utf-8")).hexdigest():
                raise ValueError(f"dataset text hash mismatch: {row['id']} {field}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "problems.jsonl").open("w", encoding="utf-8") as grading, \
         (output_dir / "prompts.jsonl").open("w", encoding="utf-8") as prompts:
        for problem in problems.values():
            grading.write(json.dumps(problem, ensure_ascii=False) + "\n")
            # Never pass the reference solution or rubric to the evaluated model.
            user = ("Solve the following problem completely. Preserve the problem's conventions.\n"
                    f"<problem_id>{problem['id']}</problem_id>\n"
                    f"<problem_latex>\n{problem['problem']}\n</problem_latex>")
            prompts.write(json.dumps({"id": problem["id"], "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]}, ensure_ascii=False) + "\n")
    print(f"Prepared {len(problems)} problems and reference-free generation prompts.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="local released data/test.jsonl; otherwise download")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--expected-count", type=int, default=40)
    args = parser.parse_args()
    dataset = args.dataset
    if dataset is None:
        if not args.revision.isalnum():
            parser.error("revision must be an alphanumeric dataset revision")
        url = f"https://huggingface.co/datasets/mytcl/PHYRD-40/resolve/{args.revision}/data/test.jsonl"
        with urllib.request.urlopen(url, timeout=120) as response:
            contents = response.read()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        dataset = args.output_dir / "released-test.jsonl"
        dataset.write_bytes(contents)
    prepare(dataset, args.output_dir, args.expected_count)
    (args.output_dir / "dataset_manifest.json").write_text(json.dumps({
        "dataset": "mytcl/PHYRD-40",
        "revision": args.revision if args.dataset is None else None,
        "source": "download" if args.dataset is None else "local",
        "jsonl_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "count": args.expected_count,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
