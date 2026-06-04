import json
from datetime import datetime, timezone

import pytest
from databricks.sdk.service import workspace as workspace_service

from maxgenie.advisory import NO_BENCHMARK_WARNING
from maxgenie.workspace_jobs import (
    WorkspaceJobSettings,
    build_runs_submit_payload,
    canonical_run_page_url,
    default_runner_path,
    format_run_status_for_chat,
    run_artifact_summary,
    run_progress_summary,
    submit_workspace_run,
    summarize_run_status,
    upload_directory_to_workspace_files,
    wait_for_run,
)


class _FakeCurrentUser:
    def me(self):
        return type("User", (), {"user_name": "user@example.com"})()


class _FakeApiClient:
    def __init__(self):
        self.calls = []

    def do(self, method, path, body=None, query=None):
        self.calls.append({"method": method, "path": path, "body": body, "query": query})
        if method == "GET" and path == "/api/2.1/jobs/runs/get":
            return {
                "run_id": query["run_id"],
                "job_id": 999,
                "run_name": "Space Optimizer Long Run",
                "run_page_url": f"https://workspace/?o=123#job-run-999/{query['run_id']}",
                "start_time": 1_000,
                "state": {"life_cycle_state": "RUNNING", "state_message": "Queued"},
                "tasks": [
                    {
                        "task_key": "optimize",
                        "run_id": 456,
                        "run_page_url": "https://workspace/?o=123#job-run-999/456",
                        "start_time": 1_000,
                        "state": {"life_cycle_state": "RUNNING", "state_message": "In run"},
                    }
                ],
            }
        return {"run_id": 123, "run_page_url": "https://workspace/#job/123"}


class _FakeWorkspaceClient:
    def __init__(self):
        self.current_user = _FakeCurrentUser()
        self.api_client = _FakeApiClient()


def _settings(**overrides) -> WorkspaceJobSettings:
    values = {
        "space_url": "https://dbc.example.com/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "runner_path": "/Workspace/Users/user@example.com/.assistant/skills/maxgenie/scripts/runner.py",
        "workspace_root": "/Workspace/Users/user@example.com/.maxgenie/runs/job",
        "strategy": "serving_autonomous",
        "serving_endpoint": "candidate-endpoint",
        "candidate_budget": 9,
    }
    values.update(overrides)
    return WorkspaceJobSettings(**values)


def test_default_runner_path_uses_user_or_workspace_scope():
    wc = _FakeWorkspaceClient()

    assert default_runner_path(wc) == "/Workspace/Users/user@example.com/.assistant/skills/maxgenie/scripts/runner.py"
    assert default_runner_path(wc, skill_scope="workspace") == "/Workspace/.assistant/skills/maxgenie/scripts/runner.py"


def test_build_serverless_runs_submit_payload():
    payload = build_runs_submit_payload(_settings())

    task = payload["tasks"][0]
    params = task["spark_python_task"]["parameters"]
    assert task["spark_python_task"]["python_file"].endswith("/scripts/runner.py")
    assert task["environment_key"] == "maxgenie"
    assert payload["environments"][0]["spec"]["environment_version"] == "2"
    assert "databricks-sdk>=0.44.0" in payload["environments"][0]["spec"]["dependencies"]
    assert "--use-latest-observed-benchmark" in params
    assert params[params.index("--strategy") + 1] == "serving_autonomous"
    assert params[params.index("--serving-endpoint") + 1] == "candidate-endpoint"
    assert params[params.index("--serving-api-format") + 1] == "auto"
    assert "--max-iterations" not in params
    assert "--serving-temperature" not in params
    assert params[params.index("--candidate-budget") + 1] == "9"


def test_build_serverless_runs_submit_payload_passes_explicit_max_iterations():
    payload = build_runs_submit_payload(_settings(max_iterations=6))

    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert params[params.index("--max-iterations") + 1] == "6"


def test_build_runs_submit_payload_passes_explicit_serving_temperature():
    payload = build_runs_submit_payload(_settings(serving_temperature=0.0))

    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert params[params.index("--serving-temperature") + 1] == "0.0"


def test_build_runs_submit_payload_passes_clone_parent_path():
    payload = build_runs_submit_payload(_settings(clone_parent_path="/Users/user@example.com"))

    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert params[params.index("--clone-parent-path") + 1] == "/Users/user@example.com"


def test_build_runs_submit_payload_passes_bootstrap_only():
    payload = build_runs_submit_payload(_settings(bootstrap_only=True))

    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert "--bootstrap-only" in params


def test_build_runs_submit_payload_passes_serving_candidate_mode():
    payload = build_runs_submit_payload(_settings(serving_candidate_mode="artifact_patch"))

    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert params[params.index("--serving-candidate-mode") + 1] == "artifact_patch"


def test_build_runs_submit_payload_passes_fixed_benchmark_artifacts_path():
    payload = build_runs_submit_payload(
        _settings(fixed_benchmark_artifacts_path="/Workspace/runs/canary/space")
    )

    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert (
        params[params.index("--fixed-benchmark-artifacts-path") + 1]
        == "/Workspace/runs/canary/space"
    )


def test_build_runs_submit_payload_passes_observed_benchmark_seed():
    payload = build_runs_submit_payload(
        _settings(observed_benchmark_seed="/Workspace/runs/canary/observed_seed.json")
    )

    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert (
        params[params.index("--observed-benchmark-seed") + 1]
        == "/Workspace/runs/canary/observed_seed.json"
    )


def test_build_runs_submit_payload_omits_noise_flags_at_defaults():
    # Default-off invariant: with no opt-in, the job parameter list carries none of
    # the noise-aware / adaptive flags, so legacy job submissions are unchanged.
    params = build_runs_submit_payload(_settings())["tasks"][0]["spark_python_task"]["parameters"]

    assert "--k-samples" not in params
    assert "--noise-sigma-full" not in params
    assert "--noise-baseline-n" not in params
    assert "--adaptive-recipes" not in params


def test_build_runs_submit_payload_passes_noise_and_adaptive_flags():
    payload = build_runs_submit_payload(
        _settings(
            k_samples=6,
            noise_sigma_full=5.97,
            noise_baseline_n=6,
            adaptive_recipes=True,
        )
    )

    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert params[params.index("--k-samples") + 1] == "6"
    assert params[params.index("--noise-sigma-full") + 1] == "5.97"
    assert params[params.index("--noise-baseline-n") + 1] == "6"
    assert "--adaptive-recipes" in params


def test_build_runs_submit_payload_uses_greedy_best_ever_by_default():
    params = build_runs_submit_payload(_settings())["tasks"][0]["spark_python_task"]["parameters"]

    assert "--greedy-best-ever" in params
    assert "--no-greedy-best-ever" not in params


def test_build_runs_submit_payload_can_disable_greedy_best_ever():
    params = build_runs_submit_payload(_settings(greedy_best_ever=False))["tasks"][0][
        "spark_python_task"
    ]["parameters"]

    assert "--greedy-best-ever" not in params
    assert "--no-greedy-best-ever" in params


def test_build_runs_submit_payload_omits_k_samples_when_one():
    # K=1 is the single-sample default and must not appear in the job params.
    params = build_runs_submit_payload(_settings(k_samples=1))["tasks"][0]["spark_python_task"][
        "parameters"
    ]

    assert "--k-samples" not in params


def test_build_runs_submit_payload_omits_content_match_at_default():
    # Default-off invariant: a legacy submission carries no --content-match flag.
    params = build_runs_submit_payload(_settings())["tasks"][0]["spark_python_task"]["parameters"]

    assert "--content-match" not in params


def test_build_runs_submit_payload_passes_content_match():
    params = build_runs_submit_payload(_settings(result_content_match=True))["tasks"][0][
        "spark_python_task"
    ]["parameters"]

    assert "--content-match" in params


def test_upload_directory_to_workspace_files_uploads_tree(tmp_path):
    class _FakeWorkspace:
        def __init__(self):
            self.directories = []
            self.imports = {}

        def mkdirs(self, path):
            self.directories.append(path)

        def import_(self, path, content=None, format=None, overwrite=None):
            self.imports[path] = {
                "content": content,
                "format": format,
                "overwrite": overwrite,
            }

    class _FakeClient:
        def __init__(self):
            self.workspace = _FakeWorkspace()

    source = tmp_path / "checkpoint"
    (source / "space" / "instructions").mkdir(parents=True)
    (source / "metadata.json").write_text("{}", encoding="utf-8")
    (source / "space" / "instructions" / "text.md").write_text("hello", encoding="utf-8")
    wc = _FakeClient()

    uploaded = upload_directory_to_workspace_files(
        wc,  # type: ignore[arg-type]
        source_dir=source,
        destination_dir="/Workspace/Users/user@example.com/runs/positive_controls/example/checkpoint",
    )

    assert "/Users/user@example.com/runs/positive_controls/example/checkpoint" in wc.workspace.directories
    assert "/Users/user@example.com/runs/positive_controls/example/checkpoint/space/instructions" in wc.workspace.directories
    assert uploaded == [
        "/Workspace/Users/user@example.com/runs/positive_controls/example/checkpoint/metadata.json",
        "/Workspace/Users/user@example.com/runs/positive_controls/example/checkpoint/space/instructions/text.md",
    ]
    api_upload_path = "/Users/user@example.com/runs/positive_controls/example/checkpoint/space/instructions/text.md"
    assert wc.workspace.imports[api_upload_path]["content"] == "aGVsbG8="
    assert wc.workspace.imports[api_upload_path]["format"] == workspace_service.ImportFormat.RAW
    assert wc.workspace.imports[api_upload_path]["overwrite"] is True


def test_build_existing_cluster_payload_uses_task_libraries():
    payload = build_runs_submit_payload(
        _settings(compute_mode="existing_cluster", existing_cluster_id="cluster-1")
    )

    task = payload["tasks"][0]
    assert task["existing_cluster_id"] == "cluster-1"
    assert "environments" not in payload
    assert task["libraries"][0]["pypi"]["package"] == "databricks-sdk>=0.44.0"


def test_workspace_job_settings_requires_serving_endpoint_for_serving_strategy():
    with pytest.raises(ValueError, match="serving strategy requires"):
        _settings(serving_endpoint=None)


def test_checkpoint_audit_allows_serving_strategy_without_endpoint():
    settings = _settings(
        serving_endpoint=None,
        audit_checkpoint_path="/Workspace/runs/s20/checkpoints/best",
    )

    payload = build_runs_submit_payload(settings)
    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert params[params.index("--audit-checkpoint-path") + 1] == "/Workspace/runs/s20/checkpoints/best"
    assert "--serving-endpoint" not in params


def test_patch_replay_audit_allows_serving_strategy_without_endpoint():
    settings = _settings(
        serving_endpoint=None,
        audit_patch_path="/Workspace/runs/positive_controls/example/patch",
    )

    payload = build_runs_submit_payload(settings)
    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert params[params.index("--audit-patch-path") + 1] == "/Workspace/runs/positive_controls/example/patch"
    assert "--serving-endpoint" not in params


def test_submit_workspace_run_posts_runs_submit_payload():
    wc = _FakeWorkspaceClient()

    result = submit_workspace_run(wc, settings=_settings())

    assert result["run_id"] == 123
    assert result["run_page_url"] == "https://workspace/?o=123#job/999/run/123"
    assert result["status_summary"]["life_cycle_state"] == "RUNNING"
    call = wc.api_client.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/api/2.1/jobs/runs/submit"
    assert call["body"]["run_name"] == "Space Optimizer Long Run"


def test_canonical_run_page_url_repairs_old_job_run_fragment():
    url = canonical_run_page_url(
        {
            "run_id": 224567332016549,
            "job_id": 637967364690522,
            "run_page_url": "https://dbc.example.com/?o=207#job-run-637967364690522/224567332016549",
        }
    )

    assert (
        url
        == "https://dbc.example.com/?o=207#job/637967364690522/run/224567332016549"
    )


def test_summarize_run_status_flags_stale_active_run():
    summary = summarize_run_status(
        {
            "run_id": 123,
            "job_id": 999,
            "run_name": "Space Optimizer Long Run",
            "run_page_url": "https://workspace/?o=123#job-run-999/123",
            "start_time": 1_000,
            "state": {
                "life_cycle_state": "RUNNING",
                "state_message": "Running optimize task",
            },
            "tasks": [
                {
                    "task_key": "optimize",
                    "run_id": 456,
                    "run_page_url": "https://workspace/?o=123#job-run-999/456",
                    "start_time": 1_000,
                    "state": {"life_cycle_state": "RUNNING"},
                }
            ],
        },
        now_ms=3_701_000,
        stale_after_seconds=3600,
    )

    assert summary["run_id"] == 123
    assert summary["run_page_url"] == "https://workspace/?o=123#job/999/run/123"
    assert summary["life_cycle_state"] == "RUNNING"
    assert summary["elapsed_seconds"] == 3700.0
    assert summary["stale"] is True
    assert "latest task timing update" in summary["stale_reason"]
    assert summary["tasks"][0]["task_key"] == "optimize"
    assert summary["tasks"][0]["run_page_url"] == "https://workspace/?o=123#job/999/run/456"
    assert summary["tasks"][0]["elapsed_seconds"] == 3700.0


def test_format_run_status_for_chat_includes_canonical_links_and_states():
    summary = {
        "run_id": 123,
        "run_page_url": "https://workspace/?o=123#job/999/run/123",
        "life_cycle_state": "RUNNING",
        "elapsed_seconds": 3700.0,
        "tasks": [
            {
                "task_key": "optimize",
                "run_id": 456,
                "run_page_url": "https://workspace/?o=123#job/999/run/456",
                "life_cycle_state": "RUNNING",
                "state_message": "In run",
                "elapsed_seconds": 300.0,
            }
        ],
    }

    text = format_run_status_for_chat(summary)

    assert "Run `123` is `RUNNING`" in text
    assert "[Job Run 123](https://workspace/?o=123#job/999/run/123)" in text
    assert "[`456`](https://workspace/?o=123#job/999/run/456): `RUNNING`" in text
    assert "In run" in text


def test_run_progress_summary_tails_iteration_log_and_renders_progress():
    class FakeDownload:
        def __init__(self, text: str):
            self.text = text

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.text.encode("utf-8")

    class FakeWorkspace:
        def __init__(self, files):
            self.files = files

        def download(self, path):
            if path not in self.files:
                raise FileNotFoundError(path)
            return FakeDownload(self.files[path])

    class FakeClient:
        pass

    root = "/Workspace/Users/user@example.com/.maxgenie/runs/job/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    records = [
        {"timestamp": "2026-06-03T00:00:00", "phase": "export"},
        {
            "timestamp": "2026-06-03T00:01:00",
            "phase": "benchmark_questions_start",
            "label": "baseline_full",
            "question_count": 10,
        },
        {
            "timestamp": "2026-06-03T00:02:00",
            "phase": "candidate_evaluated",
            "name": "patch_1",
            "outcome": "accepted",
            "train_rate": 83.33,
            "validation_rate": 75.0,
            "reason": "improved visible failures",
        },
    ]
    wc = FakeClient()
    wc.workspace = FakeWorkspace(
        {
            f"{root}/history/iteration_log.jsonl": "\n".join(json.dumps(record) for record in records),
            f"{root}/results/baseline_full_summary.json": '{"overall": {"total": 10, "passed": 2, "pass_rate": 22.0}}',
            f"{root}/results/latest_full_summary.json": '{"overall": {"total": 10, "passed": 6, "pass_rate": 60.0}}',
        }
    )
    payload = {
        "tasks": [
            {
                "spark_python_task": {
                    "parameters": [
                        "--space-url",
                        "https://workspace/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "--workspace-root",
                        "/Workspace/Users/user@example.com/.maxgenie/runs/job",
                    ],
                },
            }
        ],
    }

    progress = run_progress_summary(wc, payload)

    assert progress is not None
    assert progress["latest_message"] == "Candidate `patch_1` accepted (train 83.33%, validation 75%) - improved visible failures"
    assert progress["running_best_train"] == 83.33
    assert progress["running_best_validation"] == 75.0

    text = format_run_status_for_chat(
        {
            "run_id": 123,
            "life_cycle_state": "RUNNING",
            "elapsed_seconds": 120.0,
            "progress_summary": progress,
        }
    )

    assert "Progress:" in text
    assert "Baseline: `22% (2/10)`" in text
    assert "Latest full score: `60% (6/10)`" in text
    assert "Running best: train `83.33%`, validation `75%`" in text


def test_run_progress_summary_reports_baseline_in_progress_before_score_exists():
    class FakeDownload:
        def __init__(self, text: str):
            self.text = text

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.text.encode("utf-8")

    class FakeWorkspace:
        def download(self, path):
            if path.endswith("/history/iteration_log.jsonl"):
                return FakeDownload(
                    json.dumps(
                        {
                            "timestamp": "2026-06-03T00:01:00",
                            "phase": "benchmark_questions_start",
                            "label": "baseline_full",
                            "question_count": 10,
                        }
                    )
                )
            raise FileNotFoundError(path)

    class FakeClient:
        pass

    wc = FakeClient()
    wc.workspace = FakeWorkspace()
    payload = {
        "tasks": [
            {
                "spark_python_task": {
                    "parameters": [
                        "--space-url",
                        "https://workspace/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "--workspace-root",
                        "/Workspace/Users/user@example.com/.maxgenie/runs/job",
                    ],
                },
            }
        ],
    }

    progress = run_progress_summary(wc, payload)
    text = format_run_status_for_chat(
        {
            "run_id": 123,
            "life_cycle_state": "RUNNING",
            "elapsed_seconds": 30.0,
            "progress_summary": progress,
        }
    )

    assert progress is not None
    assert progress["baseline_status"] == "baseline benchmarking in progress"
    assert "Baseline: baseline benchmarking in progress." in text
    assert "n/a" not in text


def test_run_progress_summary_ignores_iteration_records_before_current_run_start():
    class FakeDownload:
        def __init__(self, text: str):
            self.text = text

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.text.encode("utf-8")

    class FakeWorkspace:
        def __init__(self, files):
            self.files = files

        def download(self, path):
            if path not in self.files:
                raise FileNotFoundError(path)
            return FakeDownload(self.files[path])

    class FakeClient:
        pass

    root = "/Workspace/Users/user@example.com/.maxgenie/runs/job/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    current_start = datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc)
    records = [
        {
            "recorded_at": "2026-06-03T11:00:00+00:00",
            "phase": "candidate_evaluated",
            "name": "stale_candidate",
            "outcome": "accepted",
            "train_rate": 100.0,
            "validation_rate": 100.0,
        },
        {
            "recorded_at": "2026-06-03T12:01:00+00:00",
            "phase": "benchmark_questions_start",
            "label": "population_baseline.sample01",
            "question_count": 10,
        },
    ]
    wc = FakeClient()
    wc.workspace = FakeWorkspace(
        {
            f"{root}/history/iteration_log.jsonl": "\n".join(json.dumps(record) for record in records),
        }
    )
    payload = {
        "start_time": int(current_start.timestamp() * 1000),
        "tasks": [
            {
                "spark_python_task": {
                    "parameters": [
                        "--space-url",
                        "https://workspace/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "--workspace-root",
                        "/Workspace/Users/user@example.com/.maxgenie/runs/job",
                    ],
                },
            }
        ],
    }

    progress = run_progress_summary(wc, payload)

    assert progress is not None
    assert progress["last_candidate"] is None
    assert progress["running_best_train"] is None
    assert progress["latest_message"] == "Benchmarking `population_baseline.sample01` (10 questions)."


def test_run_progress_summary_ignores_iteration_records_after_terminal_run_end():
    class FakeDownload:
        def __init__(self, text: str):
            self.text = text

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.text.encode("utf-8")

    class FakeWorkspace:
        def __init__(self, files):
            self.files = files

        def download(self, path):
            if path not in self.files:
                raise FileNotFoundError(path)
            return FakeDownload(self.files[path])

    class FakeClient:
        pass

    root = "/Workspace/Users/user@example.com/.maxgenie/runs/job/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    current_start = datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc)
    current_end = datetime(2026, 6, 3, 12, 30, tzinfo=timezone.utc)
    records = [
        {
            "recorded_at": "2026-06-03T12:29:00+00:00",
            "phase": "optimization_completed",
        },
        {
            "recorded_at": "2026-06-03T13:00:00+00:00",
            "phase": "candidate_evaluated",
            "name": "future_candidate",
            "outcome": "accepted",
            "train_rate": 100.0,
            "validation_rate": 100.0,
        },
    ]
    wc = FakeClient()
    wc.workspace = FakeWorkspace(
        {
            f"{root}/history/iteration_log.jsonl": "\n".join(json.dumps(record) for record in records),
        }
    )
    payload = {
        "start_time": int(current_start.timestamp() * 1000),
        "end_time": int(current_end.timestamp() * 1000),
        "tasks": [
            {
                "spark_python_task": {
                    "parameters": [
                        "--space-url",
                        "https://workspace/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "--workspace-root",
                        "/Workspace/Users/user@example.com/.maxgenie/runs/job",
                    ],
                },
            }
        ],
    }

    progress = run_progress_summary(wc, payload)

    assert progress is not None
    assert progress["last_candidate"] is None
    assert progress["running_best_train"] is None
    assert progress["latest_message"] == "Optimization completed."


def test_run_progress_summary_uses_latest_task_attempt_window():
    class FakeDownload:
        def __init__(self, text: str):
            self.text = text

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.text.encode("utf-8")

    class FakeWorkspace:
        def __init__(self, files):
            self.files = files

        def download(self, path):
            if path not in self.files:
                raise FileNotFoundError(path)
            return FakeDownload(self.files[path])

    class FakeClient:
        pass

    root = "/Workspace/Users/user@example.com/.maxgenie/runs/job/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    records = [
        {
            "recorded_at": "2026-06-03T12:20:00+00:00",
            "phase": "candidate_evaluated",
            "name": "failed_attempt_candidate",
            "outcome": "rejected",
            "train_rate": 0.0,
            "validation_rate": 0.0,
        },
        {
            "recorded_at": "2026-06-03T13:01:00+00:00",
            "phase": "benchmark_questions_start",
            "label": "population_baseline.sample01",
            "question_count": 10,
        },
    ]
    wc = FakeClient()
    wc.workspace = FakeWorkspace(
        {
            f"{root}/history/iteration_log.jsonl": "\n".join(json.dumps(record) for record in records),
        }
    )
    payload = {
        "start_time": int(datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc).timestamp() * 1000),
        "tasks": [
            {
                "start_time": int(datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc).timestamp() * 1000),
                "end_time": int(datetime(2026, 6, 3, 12, 30, tzinfo=timezone.utc).timestamp() * 1000),
                "spark_python_task": {
                    "parameters": [
                        "--space-url",
                        "https://workspace/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "--workspace-root",
                        "/Workspace/Users/user@example.com/.maxgenie/runs/job",
                    ],
                },
            },
            {
                "start_time": int(datetime(2026, 6, 3, 13, 0, tzinfo=timezone.utc).timestamp() * 1000),
                "spark_python_task": {
                    "parameters": [
                        "--space-url",
                        "https://workspace/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "--workspace-root",
                        "/Workspace/Users/user@example.com/.maxgenie/runs/job",
                    ],
                },
            },
        ],
    }

    progress = run_progress_summary(wc, payload)

    assert progress is not None
    assert progress["last_candidate"] is None
    assert progress["running_best_train"] is None
    assert progress["latest_message"] == "Benchmarking `population_baseline.sample01` (10 questions)."


def test_wait_for_run_calls_on_tick_until_terminal():
    class SequenceApiClient:
        def __init__(self):
            self.responses = [
                {"run_id": 123, "state": {"life_cycle_state": "RUNNING"}},
                {"run_id": 123, "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"}},
            ]

        def do(self, method, path, body=None, query=None):
            assert method == "GET"
            assert path == "/api/2.1/jobs/runs/get"
            return self.responses.pop(0)

    class FakeClient:
        pass

    wc = FakeClient()
    wc.api_client = SequenceApiClient()
    ticks = []

    final = wait_for_run(
        wc,
        run_id=123,
        poll_seconds=0,
        timeout_seconds=5,
        on_tick=ticks.append,
    )

    assert final["state"]["life_cycle_state"] == "TERMINATED"
    assert [tick["state"]["life_cycle_state"] for tick in ticks] == ["RUNNING", "TERMINATED"]


def test_run_artifact_summary_recovers_optimized_space_and_skips_ui_eval_by_default(
    monkeypatch,
):
    class FakeDownload:
        def __init__(self, text: str):
            self.text = text

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.text.encode("utf-8")

    class FakeWorkspace:
        def __init__(self, files):
            self.files = files

        def download(self, path):
            if path not in self.files:
                raise FileNotFoundError(path)
            return FakeDownload(self.files[path])

    class FakeApiClient:
        def do(self, method, path, body=None, query=None):
            if method == "GET" and path == "/api/2.0/genie/spaces/clone-123":
                return {"title": "[maxgenie 60%] Demo Space"}
            if method == "GET" and path == "/api/2.0/workspace/get-status":
                if query == {
                    "path": "/Users/user@example.com/.maxgenie/runs/job/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/final_report.ipynb"
                }:
                    return {"object_type": "NOTEBOOK", "object_id": 123123123}
                if query == {
                    "path": "/Users/user@example.com/.maxgenie/runs/job/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/final_report.md"
                }:
                    return {"object_type": "FILE", "object_id": 987654321}
            raise AssertionError((method, path, body, query))

    class FakeClient:
        pass

    observed_calls = 0

    def fake_latest_eval(**kwargs):
        nonlocal observed_calls
        observed_calls += 1
        return {
            "eval_run_id": "eval-1",
            "question_count": 10,
            "result_count": 10,
            "assessment_counts": {"good": 3, "bad": 7},
            "results": [
                {"status": "good", "score": 1.0},
                {"status": "good", "score": 1.0},
                {"status": "good", "score": 1.0},
                *({"status": "bad", "score": 0.0} for _ in range(7)),
            ],
        }

    monkeypatch.setattr(
        "maxgenie.observed_benchmarks.latest_eval_run_observed_summary",
        fake_latest_eval,
    )
    root = "/Workspace/Users/user@example.com/.maxgenie/runs/job/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    wc = FakeClient()
    wc.workspace = FakeWorkspace(
        {
            f"{root}/clone_meta.json": '{"clone_space_id": "clone-123"}',
            f"{root}/workspace_meta.json": '{"observed_benchmark_reuse_policy": "fresh_baseline_benchmark_run"}',
            f"{root}/results/baseline_full_summary.json": '{"overall": {"total": 10, "passed": 6, "pass_rate": 60.0}, "baseline_source": "benchmark_run"}',
            f"{root}/results/latest_full_summary.json": '{"overall": {"total": 10, "passed": 6, "pass_rate": 60.0}, "train": {"pass_rate": 66.67}, "validation": {"pass_rate": 62.5}, "test": {"pass_rate": 50.0}}',
            f"{root}/results/optimization_summary.json": '{"accepted": 0, "rejected": 0, "skipped": 3, "terminal_selection": {"status": "not_promotable", "reasons": ["final_full_did_not_improve_over_baseline"]}}',
            f"{root}/final_report.md": "# report",
        }
    )
    wc.api_client = FakeApiClient()
    payload = {
        "run_id": 123,
        "job_id": 999,
        "run_page_url": "https://workspace/?o=123#job/999/run/123",
        "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
        "tasks": [
            {
                "spark_python_task": {
                    "parameters": [
                        "--mode",
                        "full",
                        "--space-url",
                        "https://workspace/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "--workspace-root",
                        "/Workspace/Users/user@example.com/.maxgenie/runs/job",
                    ],
                },
            }
        ],
    }

    artifact_summary = run_artifact_summary(wc, payload)
    assert artifact_summary is not None
    assert artifact_summary["clone_url"] == "https://workspace/genie/rooms/clone-123?o=123"
    assert artifact_summary["clone_title"] == "[maxgenie 60%] Demo Space"
    assert artifact_summary["final_report_url"] == "https://workspace/editor/files/987654321?o=123"
    assert (
        artifact_summary["final_report_notebook_url"]
        == "https://workspace/editor/files/123123123?o=123"
    )
    assert artifact_summary["observed_ui_benchmark"] is None
    assert observed_calls == 0

    text = format_run_status_for_chat(
        {
            "run_id": 123,
            "life_cycle_state": "TERMINATED",
            "result_state": "SUCCESS",
            "elapsed_seconds": 600,
            "artifact_summary": artifact_summary,
        }
    )

    assert "Optimized space: [open optimized space](https://workspace/genie/rooms/clone-123?o=123)" in text
    assert "Optimized space title: `[maxgenie 60%] Demo Space`" in text
    assert "Final full score: `60% (6/10)`" in text
    assert "Final report: [open rendered report](https://workspace/editor/files/123123123?o=123)" in text
    assert "Raw report file: [open markdown report](https://workspace/editor/files/987654321?o=123)" in text
    assert "Existing Genie UI eval" not in text

    artifact_summary = run_artifact_summary(wc, payload, include_observed_ui=True)
    assert artifact_summary is not None
    assert artifact_summary["observed_ui_benchmark"]["pass_rate"] == 30.0
    assert observed_calls == 1

    text = format_run_status_for_chat(
        {
            "run_id": 123,
            "life_cycle_state": "TERMINATED",
            "result_state": "SUCCESS",
            "elapsed_seconds": 600,
            "artifact_summary": artifact_summary,
        }
    )

    assert "Existing Genie UI eval: `30% (3/10)` from eval run `eval-1`" in text
    assert "no net improvement" in text
    assert "fresh MaxGenie result-based baseline" in text


def test_run_artifact_summary_advisory_ignores_stale_scores():
    class FakeDownload:
        def __init__(self, text: str):
            self.text = text

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.text.encode("utf-8")

    class FakeWorkspace:
        def __init__(self, files):
            self.files = files

        def download(self, path):
            if path not in self.files:
                raise FileNotFoundError(path)
            return FakeDownload(self.files[path])

    class FakeClient:
        pass

    root = "/Workspace/Users/user@example.com/.maxgenie/runs/job/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    wc = FakeClient()
    wc.workspace = FakeWorkspace(
        {
            f"{root}/clone_meta.json": '{"clone_space_id": "clone-123"}',
            f"{root}/workspace_meta.json": (
                '{"no_benchmark_advisory": true, '
                '"evidence_classification": "advisory_best_practices"}'
            ),
            f"{root}/results/baseline_full_summary.json": (
                '{"overall": {"total": 10, "passed": 5, "pass_rate": 50.0}}'
            ),
            f"{root}/results/latest_full_summary.json": (
                '{"overall": {"total": 10, "passed": 9, "pass_rate": 90.0}, '
                '"train": {"pass_rate": 90.0}, '
                '"validation": {"pass_rate": 80.0}, '
                '"test": {"pass_rate": 70.0}}'
            ),
            f"{root}/results/advisory_best_practices.json": json.dumps(
                {
                    "mode": "advisory_best_practices",
                    "evidence_classification": "advisory_best_practices",
                    "benchmark_verification_status": "not_benchmark_verified",
                    "benchmark_verification_label": "not benchmark-verified",
                    "warning": NO_BENCHMARK_WARNING,
                    "curation_mode": "safe_defaults",
                    "change_count": 2,
                }
            ),
            f"{root}/final_report.md": "# report",
        }
    )
    wc.api_client = _FakeApiClient()
    payload = {
        "run_id": 123,
        "job_id": 999,
        "run_page_url": "https://workspace/?o=123#job/999/run/123",
        "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
        "tasks": [
            {
                "spark_python_task": {
                    "parameters": [
                        "--mode",
                        "full",
                        "--space-url",
                        "https://workspace/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "--workspace-root",
                        "/Workspace/Users/user@example.com/.maxgenie/runs/job",
                    ],
                },
            }
        ],
    }

    artifact_summary = run_artifact_summary(wc, payload)

    assert artifact_summary is not None
    assert artifact_summary["evidence_classification"] == "advisory_best_practices"
    assert artifact_summary["benchmark_verification_label"] == "not benchmark-verified"
    assert artifact_summary["final_score"]["pass_rate"] is None
    assert artifact_summary["baseline_score"]["pass_rate"] is None
    assert artifact_summary["split_scores"] == {}
    assert artifact_summary["terminal_selection"]["status"] == "not_benchmark_verified"

    text = format_run_status_for_chat(
        {
            "run_id": 123,
            "life_cycle_state": "TERMINATED",
            "result_state": "SUCCESS",
            "elapsed_seconds": 600,
            "artifact_summary": artifact_summary,
        }
    )

    assert NO_BENCHMARK_WARNING in text
    assert "Evidence classification: `advisory_best_practices`" in text
    assert "Benchmark verification: `not benchmark-verified`" in text
    assert "Advisory mode: `safe_defaults`" in text
    assert "Advisory changes: `2`" in text
    assert "Final full score: `n/a (no benchmark)`" in text
    assert "90%" not in text
    assert "50%" not in text
    assert "Split scores" not in text


def test_summarize_run_status_keeps_terminal_run_not_stale():
    summary = summarize_run_status(
        {
            "run_id": 123,
            "start_time": 1_000,
            "end_time": 11_000,
            "state": {
                "life_cycle_state": "TERMINATED",
                "result_state": "SUCCESS",
            },
        },
        now_ms=3_701_000,
        stale_after_seconds=1,
    )

    assert summary["elapsed_seconds"] == 10.0
    assert summary["result_state"] == "SUCCESS"
    assert summary["stale"] is False
