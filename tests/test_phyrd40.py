from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
import threading
import contextlib
import io
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluation/phyrd40/rubric_judge.py"
SPEC = importlib.util.spec_from_file_location("evaluation.phyrd40.rubric_judge", MODULE_PATH)
assert SPEC and SPEC.loader
rubric_judge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rubric_judge)


def valid_judgment() -> dict:
    return {
        "score": 75,
        "problem_quality_score": 90,
        "well_posed_score": 5,
        "reference_quality_score": 4,
        "ambiguity_severity": "none",
        "difficulty_score": 3,
        "estimated_model_pass_probability": 0.7,
        "score_rationale": "Mostly correct.",
        "problem_quality_rationale": "Well posed.",
        "difficulty_rationale": "Moderate.",
        "quality_issues": [],
        "strengths": ["Correct conclusion"],
        "errors": [],
        "rubric_breakdown": [
            {
                "criterion": "Answer",
                "points_awarded": 75,
                "points_possible": 100,
                "explanation": "One detail is missing.",
            }
        ],
    }


class RubricJudgeTest(unittest.TestCase):
    def test_validate_and_pass_threshold(self) -> None:
        value = rubric_judge.validate_judgment(valid_judgment(), 70)
        self.assertTrue(value["passed"])
        self.assertEqual(value["difficulty_label"], "medium")

    def test_reject_missing_field(self) -> None:
        value = valid_judgment()
        del value["score"]
        with self.assertRaisesRegex(ValueError, "missing fields"):
            rubric_judge.validate_judgment(value, 70)

    def test_extract_chat_and_responses_envelopes(self) -> None:
        self.assertEqual(
            rubric_judge.extract_text({"choices": [{"message": {"content": "ok"}}]}),
            "ok",
        )
        self.assertEqual(rubric_judge.extract_text({"output_text": "done"}), "done")

    def test_load_portable_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problems = root / "problems.jsonl"
            predictions = root / "predictions.jsonl"
            problems.write_text(
                json.dumps({"id": "p1", "problem": "Q", "reference": "R"}) + "\n",
                encoding="utf-8",
            )
            predictions.write_text(
                json.dumps({"id": "p1", "response": "A"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(rubric_judge.load_problems(problems)["p1"]["problem"], "Q")
            self.assertEqual(
                rubric_judge.load_predictions(predictions)["p1"]["response"], "A"
            )

    def test_load_config_reads_key_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "base_url": "https://judge.example/v1",
                        "api_key_env": "TEST_RUBRIC_JUDGE_KEY",
                        "model": "judge-model",
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"TEST_RUBRIC_JUDGE_KEY": "secret"}):
                config = rubric_judge.load_config(path)
            self.assertEqual(config["api_key"], "secret")
            self.assertEqual(config["pass_score"], 70)

    def test_reject_invalid_rubric_points(self) -> None:
        value = valid_judgment()
        value["rubric_breakdown"][0]["points_awarded"] = 101
        with self.assertRaisesRegex(ValueError, "invalid points"):
            rubric_judge.validate_judgment(value, 70)

    def test_reject_inconsistent_rubric_total(self) -> None:
        value = valid_judgment()
        value["rubric_breakdown"][0]["points_possible"] = 108
        with self.assertRaisesRegex(ValueError, "sum to 100"):
            rubric_judge.validate_judgment(value, 70)

    def test_reject_nonfinite_boolean_and_nested_extra_fields(self):
        for score in (float("nan"), float("inf"), True, "75"):
            value = valid_judgment()
            value["score"] = score
            with self.assertRaises(ValueError):
                rubric_judge.validate_judgment(value, 70)
        value = valid_judgment()
        value["rubric_breakdown"][0]["secret"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            rubric_judge.validate_judgment(value, 70)

    def test_inline_credentials_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"api_key": "do-not-store-keys"}))
            with self.assertRaisesRegex(ValueError, "inline api_key"):
                rubric_judge.load_config(path)


class PortableWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.requests = []
        self.mode = "normal"
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append((self.path, payload, self.headers.get("Authorization")))
                if owner.mode == "redirect":
                    self.send_response(302)
                    self.send_header("Location", owner.base_url + "/must-not-follow")
                    self.end_headers()
                    return
                if owner.mode == "error":
                    self.send_response(403)
                    self.end_headers()
                    self.wfile.write(("echo " + owner.key + " " + owner.base_url).encode())
                    return
                structured = "response_format" in payload or "text" in payload
                if owner.mode == "fallback" and structured:
                    self.send_response(422)
                    self.end_headers()
                    return
                if "messages" in payload:
                    prompt = payload["messages"][-1]["content"]
                else:
                    prompt = payload["input"]
                if prompt.startswith("Grade the candidate"):
                    judgment = valid_judgment()
                    score = 100 if "toy-1" in prompt else 50
                    judgment["score"] = score
                    judgment["rubric_breakdown"][0]["points_awarded"] = score
                    text = json.dumps(judgment)
                    if owner.mode == "invalid-strict" and structured:
                        text = '{"score":"invalid"}'
                    if self.path.endswith("/responses"):
                        value = {"output": [{"content": [{"type": "output_text", "text": text}]}]}
                    else:
                        value = {"choices": [{"message": {"content": text}}]}
                else:
                    value = {"choices": [{"message": {"content": "generated solution"},
                                          "finish_reason": "stop"}]}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(value).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        self.key = "test-secret-not-for-log-files"
        self.env = mock.patch.dict(os.environ, {
            "JUDGE_BASE_URL": self.base_url, "JUDGE_API_KEY": self.key,
            "JUDGE_MODEL": "test-judge", "MODEL_BASE_URL": self.base_url,
            "MODEL_API_KEY": self.key, "MODEL_NAME": "test-model",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.config_path = self.root / "judge.json"
        self.config_path.write_text(json.dumps({"max_retries": 1, "concurrency": 1}))
        self.problems = self.root / "problems.jsonl"
        self.predictions = self.root / "predictions.jsonl"
        self.problems.write_text(''.join(json.dumps({"id": f"toy-{i}", "problem": "Question?",
                                                  "reference": "PRIVATE_REFERENCE_SENTINEL"}) + '\n'
                                         for i in (1, 2)))
        self.predictions.write_text(''.join(json.dumps({"id": f"toy-{i}", "response": "answer"}) + '\n'
                                            for i in (1, 2)))
        self.output = self.root / "judged"

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def grade(self):
        argv = ["rubric_judge", "--problems", str(self.problems),
                "--predictions", str(self.predictions), "--judge-config", str(self.config_path),
                "--output-dir", str(self.output), "--expected-count", "2"]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            return rubric_judge.main()

    def test_complete_scoring_resume_and_config_invalidation(self):
        self.assertEqual(self.grade(), 0)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["mean_score"], 75)
        self.assertEqual(summary["pass_rate"], 0.5)
        self.assertEqual(summary["primary_metric"], "mean_score")
        count = len(self.requests)
        self.grade()
        self.assertEqual(len(self.requests), count)
        self.config_path.write_text(json.dumps({"max_retries": 1, "max_output_tokens": 512}))
        self.grade()
        self.assertEqual(len(self.requests), count + 2)
        for path, payload, auth in self.requests:
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(auth, "Bearer " + self.key)
            self.assertIn("PRIVATE_REFERENCE_SENTINEL", payload["messages"][-1]["content"])

    def test_responses_api_and_schema_fallback(self):
        for mode in ("normal", "fallback", "invalid-strict"):
            with self.subTest(mode=mode):
                self.mode = mode
                self.requests.clear()
                self.output = self.root / mode
                self.config_path.write_text(json.dumps({"api_mode": "responses", "max_retries": 1}))
                self.grade()
                self.assertTrue(all(p == "/v1/responses" for p, _, _ in self.requests))
                if mode != "normal":
                    self.assertEqual(len(self.requests), 4)
                    results = json.loads((self.output / "results.json").read_text())
                    self.assertTrue(all(r["schema_mode"] == "prompt" for r in results))
                    prompts = [p["instructions"] for _, p, _ in self.requests if "text" not in p]
                    self.assertTrue(all('"rubric_breakdown"' in p for p in prompts))

    def test_failed_rerun_removes_stale_summary_and_redacts_transport(self):
        self.grade()
        self.mode = "error"
        self.config_path.write_text(json.dumps({"schema_mode": "prompt", "max_retries": 1}))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.grade()
        self.assertFalse((self.output / "summary.json").exists())
        self.assertFalse((self.output / "results.json").exists())
        logs = (self.output / "judgments.jsonl").read_text()
        self.assertNotIn(self.key, logs)
        self.assertNotIn(self.base_url, logs)
        self.assertIn("HTTP 403", logs)

    def test_default_rejects_partial_40_problem_run_before_api(self):
        argv = ["judge", "--problems", str(self.problems), "--predictions", str(self.predictions),
                "--judge-config", str(self.config_path), "--output-dir", str(self.output)]
        with mock.patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "expected 40"):
            rubric_judge.main()
        self.assertEqual(len(self.requests), 0)

    def test_generation_reference_isolation_and_resuming(self):
        from evaluation.phyrd40.prepare import prepare
        from evaluation.phyrd40.generate import main
        prepared = self.root / "prepared"
        prepare(self.problems, prepared, expected_count=2)
        config = self.root / "generation.json"
        config.write_text(json.dumps({"model_env": "MODEL_NAME", "max_retries": 1}))
        output = self.root / "generated.jsonl"
        argv = ["generate", "--prompts", str(prepared / "prompts.jsonl"), "--config", str(config),
                "--output", str(output), "--expected-count", "2"]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)
            self.assertEqual(main(), 0)
        self.assertEqual(len(self.requests), 2)
        self.assertNotIn("PRIVATE_REFERENCE_SENTINEL", (prepared / "prompts.jsonl").read_text())
        self.assertTrue(all("PRIVATE_REFERENCE_SENTINEL" not in json.dumps(p)
                            for _, p, _ in self.requests))
        self.assertEqual(len(output.read_text().splitlines()), 2)

    def test_authorization_is_not_forwarded_on_redirect(self):
        from evaluation.phyrd40.api_client import ApiError, post_json
        self.mode = "redirect"
        config = rubric_judge.load_config(self.config_path)
        with self.assertRaises(ApiError) as ctx:
            post_json(config, self.base_url + "/chat/completions", {"messages": []})
        self.assertEqual(ctx.exception.status, 302)
        self.assertEqual(len(self.requests), 1)


if __name__ == "__main__":
    unittest.main()
