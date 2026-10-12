# Portable PHYRD-40 evaluation

This release provides a standalone generation → rubric grading → aggregation
workflow. It runs on Python 3.10+ with the standard library; it does not require
the training stack, a cluster scheduler, or the authors' evaluation service.
The evaluated model and judge are separate, user-configured API endpoints.

## Protocol and release scope

The report evaluates **one response per problem** and reports the **arithmetic
mean of 40 scores on a 0–100 scale**. The judge sees the problem, candidate
response, and complete reference solution/rubric. The evaluated model sees
only the problem. The optional problem-quality audit and threshold-based
`pass_rate` are auxiliary diagnostics, not the PHYRD-40 result in Table 2.

The public grader adapts the authors' rubric scoring logic with local JSON
validation, per-criterion checks, retries and resumable results. It is a portable
implementation, not a replay of historical API traffic. The report names
DeepSeek-V4 Pro Preview as the original judge; internal model aliases and routing
have been removed. Choose a model offered by your own provider, record its exact
version, and keep it fixed across comparisons. Changing the judge or generation
settings can change the scores.

The small-scale **ESCD/BPM/SimCT/GOLD/X-Token** recipes are released separately.
Large-scale SFT/MoE training recipes and trained weights are not included.
[SciDeriv](https://huggingface.co/datasets/mytcl/SciDeriv) provides 17,577
large-scale training prompts;
[PHYRD-40](https://huggingface.co/datasets/mytcl/PHYRD-40) provides the evaluation
problems and grading references. This workflow does not introduce large-scale
training code or mix evaluation references into training data.

## 1. Prepare the released data

Run all commands from the ESCD repository root:

```bash
python -m evaluation.phyrd40.prepare --output-dir data/phyrd40
```

This downloads `data/test.jsonl` from a pinned public dataset revision, verifies
40 unique IDs and the released text hashes, and writes:

- `problems.jsonl`: the question and reference/rubric, for grading;
- `prompts.jsonl`: report-style system/user messages, without references;
- `dataset_manifest.json`: dataset revision, count and file checksum.

For an existing download, use `--dataset /path/to/data/test.jsonl`.

The large-scale training data can be downloaded independently:

```bash
hf download mytcl/SciDeriv --repo-type dataset data/train.jsonl --local-dir data/scideriv
```

## 2. Supply candidate responses

If you have already generated responses, write one JSON object per line:

```json
{"id":"A001","response":"The model's complete solution..."}
```

Use the exact 40 dataset IDs, with no duplicates or missing responses. Save as
`outputs/phyrd40/predictions.jsonl`. The grader also accepts `sample_id` for `id`
and `candidate_answer`, `prediction`, `output`, or `answer` for `response`.

Alternatively, use your own model server's Chat Completions API:

```bash
mkdir -p configs/local
cp configs/examples/phyrd40_generation.json configs/local/generation.json
export MODEL_BASE_URL='https://your-model-provider.example/v1'
export MODEL_NAME='your-served-model'
read -rsp 'Model API key: ' MODEL_API_KEY; echo
export MODEL_API_KEY
python -m evaluation.phyrd40.generate \
  --prompts data/phyrd40/prompts.jsonl \
  --config configs/local/generation.json \
  --output outputs/phyrd40/predictions.jsonl
```

Example addresses are placeholders: replace them with your API root (including
`/v1` where required). For a local unauthenticated server, set `MODEL_API_KEY` to
an unused placeholder accepted by that server. No credentials are bundled.

The generic generator uses temperature 1.0, top-p 0.95 and a 256,000-token output
budget. The report additionally uses top-k 20, a 262,144-token context and native
chat templates with thinking enabled. Context capacity and chat templates must
be configured on your model server. If your API supports them, set
`extra_body` to `{"top_k":20,"chat_template_kwargs":{"enable_thinking":true}}`
in the local generation config; otherwise use your existing inference runner
and supply its predictions. These fields are opt-in because API support varies.
Set `token_limit_field` to `max_completion_tokens` if your provider requires it.
Inspect `finish_reason` for truncated outputs. A shorter budget or different
template is a changed evaluation setting and should be reported.

The generator resumes responses with an identical prompt/config fingerprint.
Choose a fresh output file when changing generation settings or model versions.
References are never read by the generation script.

## 3. Configure and run the judge

```bash
mkdir -p configs/local
cp configs/examples/phyrd40_judge.json configs/local/judge.json
export JUDGE_BASE_URL='https://your-judge-provider.example/v1'
export JUDGE_MODEL='your-versioned-judge-model'
read -rsp 'Judge API key: ' JUDGE_API_KEY; echo
export JUDGE_API_KEY
python -m evaluation.phyrd40.rubric_judge \
  --problems data/phyrd40/problems.jsonl \
  --predictions outputs/phyrd40/predictions.jsonl \
  --judge-config configs/local/judge.json \
  --output-dir outputs/phyrd40/judge
```

`base_url_env`, `api_key_env`, and `model_env` select environment variables. Keys
must come from the environment; inline `api_key` fields are rejected. Local
configurations, downloaded datasets and generated outputs are ignored by Git.
The public examples contain no service address, key, or internal model alias.
Provider URLs and raw transport error bodies are not written to result logs.

Supported `api_mode` values are `chat_completions` and `responses`. `schema_mode`
defaults to `strict`; compatible providers that reject JSON Schema with HTTP
400/404/422 or ignore it fall back to schema instructions in the prompt. Both
paths undergo the same local validation. Use `prompt` directly if needed.
`reasoning_effort: null` omits the provider-specific reasoning control; configure
it only when supported. `max_output_tokens: null` leaves the limit to the
provider; set a suitable limit for your chosen judge.

The grader checks field types, enums, finite score ranges, awarded points against
maximum points, total possible points (100 within a 0.5 rounding tolerance), and
agreement between the total score and awarded points (within 1 point). Invalid
judgments are retried and never counted as valid scores.

## Results and resuming

- `judgments.jsonl`: per-problem judgments or sanitized, retryable errors;
- `results.json`: all 40 validated judgments;
- `summary.json`: the primary `mean_score`, plus auxiliary score and quality statistics.

The run exits unsuccessfully if any judgment is missing or invalid. It clears
old aggregate files before a rerun, so an incomplete run cannot display an old
complete score. Run the same command to resume. Successful judgments are reused
only when their fingerprint matches the problem, reference, response, endpoint,
model, prompt and scoring configuration, and they still pass validation.
Endpoint values are hashed in fingerprints rather than exposed in outputs.

## Small format check and automated tests

The included two examples are toy arithmetic problems, not PHYRD-40 data.
An optional API smoke test uses `--expected-count 2`:

```bash
python -m evaluation.phyrd40.rubric_judge \
  --problems evaluation/phyrd40/examples/problems.jsonl \
  --predictions evaluation/phyrd40/examples/predictions.jsonl \
  --judge-config configs/local/judge.json \
  --output-dir outputs/phyrd40-toy --expected-count 2
```

Automated tests use a local mock API and require no API key or billable calls:

```bash
python -m unittest discover -s tests -p 'test_phyrd40*.py' -v
```
