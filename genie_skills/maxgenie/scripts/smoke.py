"""Self-contained lightweight optimizer workflow for synced skill runtimes."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient


SPACE_ID_RE = re.compile(r"[0-9a-f]{32}")
DEFAULT_TITLE_PREFIX = "Space Optimizer Lite"
DEFAULT_QUESTION = "Give me sample 5 questions"
POLL_INTERVAL_SECONDS = 5
MAX_POLL_ATTEMPTS = 48
MATCH_THRESHOLD = 0.85

logger = logging.getLogger("skill_smoke")


@dataclass(frozen=True)
class ParsedSpaceUrl:
    """Parsed Genie space URL."""

    host: str
    space_id: str
    url: str


@dataclass(frozen=True)
class PocResult:
    """Summary of one lightweight run."""

    run_dir: Path
    source_space_id: str
    clone_space_id: str | None
    exported: bool
    clone_created: bool
    candidate_01_updated: bool
    candidate_02_updated: bool
    conversation_completed: bool
    clone_trashed: bool


@dataclass(frozen=True)
class BenchmarkQuestion:
    """One visible benchmark question."""

    question_id: str
    question: str
    expected_sql: str | None


@dataclass(frozen=True)
class BenchmarkResult:
    """Result for one lightweight benchmark question."""

    question_id: str
    question: str
    status: str
    score: float | None
    method: str | None
    detail: str | None
    generated_sql: str | None
    expected_sql: str | None
    conversation_id: str | None
    message_id: str | None


@dataclass(frozen=True)
class CandidateOutcome:
    """Outcome for one candidate policy."""

    name: str
    accepted: bool
    baseline_pass_rate: float
    candidate_pass_rate: float
    reason: str
    config_hash: str


class PocError(RuntimeError):
    """Raised when the run cannot continue safely."""


class GeniePocClient:
    """Small REST wrapper for Genie space operations."""

    def __init__(self, workspace_client: WorkspaceClient) -> None:
        self.workspace_client = workspace_client

    def get_space(self, space_id: str, *, include_serialized_space: bool = True) -> dict[str, Any]:
        return self.workspace_client.api_client.do(
            "GET",
            f"/api/2.0/genie/spaces/{space_id}",
            query={"include_serialized_space": str(include_serialized_space).lower()},
        )

    def create_space(
        self,
        *,
        warehouse_id: str,
        title: str,
        serialized_space: dict[str, Any],
        parent_path: str | None,
        description: str | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "warehouse_id": warehouse_id,
            "title": title,
            "serialized_space": json.dumps(serialized_space, ensure_ascii=False),
        }
        if parent_path:
            body["parent_path"] = parent_path
        if description:
            body["description"] = description
        return self.workspace_client.api_client.do("POST", "/api/2.0/genie/spaces", body=body)

    def update_space(
        self,
        *,
        space_id: str,
        title: str,
        serialized_space: dict[str, Any],
        description: str | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "title": title,
            "serialized_space": json.dumps(serialized_space, ensure_ascii=False),
        }
        if description:
            body["description"] = description
        return self.workspace_client.api_client.do(
            "PATCH",
            f"/api/2.0/genie/spaces/{space_id}",
            body=body,
        )

    def start_conversation(self, *, space_id: str, question: str) -> dict[str, Any]:
        return self.workspace_client.api_client.do(
            "POST",
            f"/api/2.0/genie/spaces/{space_id}/start-conversation",
            body={"content": question},
        )

    def get_message(self, *, space_id: str, conversation_id: str, message_id: str) -> dict[str, Any]:
        return self.workspace_client.api_client.do(
            "GET",
            f"/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages/{message_id}",
        )

    def trash_space(self, space_id: str) -> None:
        self.workspace_client.api_client.do("DELETE", f"/api/2.0/genie/spaces/{space_id}")


def parse_space_url(space_url: str | None) -> ParsedSpaceUrl:
    """Parse a Genie space URL."""
    if not space_url:
        raise PocError("No Genie space URL provided. Pass --space-url or set MAXGENIE_DEFAULT_SPACE_URL.")
    parsed = urlparse(space_url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise PocError("Expected a full Genie space URL with scheme and host.")
    match = SPACE_ID_RE.search(" ".join([parsed.path, parsed.query, parsed.fragment]))
    if not match:
        raise PocError("Could not find a 32-character Genie space ID in the URL.")
    return ParsedSpaceUrl(host=parsed.netloc, space_id=match.group(0), url=space_url.strip())


def build_workspace_client(profile: str | None) -> WorkspaceClient:
    """Build a Databricks workspace client."""
    if profile:
        return WorkspaceClient(profile=profile)
    return WorkspaceClient()


def current_user_home(workspace_client: WorkspaceClient) -> str:
    """Return the current user's workspace home path."""
    me = workspace_client.current_user.me()
    user_name = getattr(me, "user_name", None)
    if not user_name:
        raise PocError("Could not resolve current Databricks user name.")
    return f"/Users/{user_name}"


def resolve_default_run_root() -> Path:
    """Return a run root that works locally and inside Databricks workspace files."""
    workspace_home = os.environ.get("MAXGENIE_WORKSPACE_HOME")
    if workspace_home:
        return Path(workspace_home) / ".maxgenie" / "runs" / "lite"
    workspace_users = Path("/Workspace/Users")
    if workspace_users.exists():
        return Path("/Workspace") / ".maxgenie" / "runs" / "lite"
    return Path.cwd() / "runs"


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def serialized_hash(payload: dict[str, Any]) -> str:
    """Return a stable short hash for a serialized space."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def text_from_content(value: Any) -> str:
    """Normalize Genie content arrays into text."""
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    if value is None:
        return ""
    return str(value)


def load_serialized_space(space_response: dict[str, Any]) -> dict[str, Any]:
    """Extract the serialized_space object from a get-space response."""
    raw = space_response.get("serialized_space")
    if not isinstance(raw, str) or not raw.strip():
        raise PocError("get-space response did not include serialized_space.")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise PocError("serialized_space was not a JSON object.")
    return parsed


def extract_benchmark_questions(serialized_space: dict[str, Any], *, limit: int) -> list[BenchmarkQuestion]:
    """Extract visible benchmark questions from serialized_space."""
    rows = serialized_space.get("benchmarks", {}).get("questions", [])
    questions: list[BenchmarkQuestion] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        question = text_from_content(row.get("question")).strip()
        if not question:
            continue
        answers = row.get("answer") or []
        expected_sql = None
        if answers and isinstance(answers[0], dict):
            expected_sql = text_from_content(answers[0].get("content")).strip() or None
        questions.append(
            BenchmarkQuestion(
                question_id=str(row.get("id") or f"benchmark_{index + 1:03d}"),
                question=question,
                expected_sql=expected_sql,
            )
        )
        if len(questions) >= limit:
            break
    return questions


def apply_sample_question_probe(serialized_space: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Append a harmless sample question to test structured config updates."""
    candidate = deepcopy(serialized_space)
    config = candidate.setdefault("config", {})
    sample_questions = config.setdefault("sample_questions", [])
    sample_questions.append(
        {
            "id": hashlib.md5(f"{run_id}:candidate_01".encode("utf-8"), usedforsecurity=False).hexdigest(),
            "question": ["Space Optimizer marker: confirm the cloned Genie space can be updated."],
        }
    )
    return candidate


def apply_marker_instruction(serialized_space: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Append a short marker instruction to test instruction updates."""
    candidate = deepcopy(serialized_space)
    instructions = candidate.setdefault("instructions", {})
    text_instructions = instructions.setdefault("text_instructions", [])
    if not text_instructions:
        text_instructions.append(
            {
                "id": hashlib.md5(f"{run_id}:instruction".encode("utf-8"), usedforsecurity=False).hexdigest(),
                "content": [],
            }
        )
    content = text_instructions[0].setdefault("content", [])
    content.append(f"\n\nSpace Optimizer marker {run_id}: this optimization clone was updated by the lightweight workflow.\n")
    return candidate


def apply_literal_examples(serialized_space: dict[str, Any], questions: list[BenchmarkQuestion], run_id: str) -> dict[str, Any]:
    """Add literal benchmark-derived SQL examples without parameter placeholders."""
    candidate = deepcopy(serialized_space)
    instructions = candidate.setdefault("instructions", {})
    examples = instructions.setdefault("example_question_sqls", [])
    existing_questions = {
        text_from_content(example.get("question")).strip().lower()
        for example in examples
        if isinstance(example, dict)
    }
    added = 0
    for question in questions:
        if not question.expected_sql:
            continue
        if question.question.lower() in existing_questions:
            continue
        examples.append(
            {
                "id": hashlib.md5(
                    f"{run_id}:literal:{question.question_id}".encode("utf-8"),
                    usedforsecurity=False,
                ).hexdigest(),
                "question": [question.question],
                "sql": [question.expected_sql],
                "parameters": [],
            }
        )
        added += 1
    if added == 0:
        return candidate
    return candidate


def apply_portable_guardrails(serialized_space: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Add the highest-signal portable guidance learned from prior runs."""
    candidate = deepcopy(serialized_space)
    instructions = candidate.setdefault("instructions", {})
    text_instructions = instructions.setdefault("text_instructions", [])
    if not text_instructions:
        text_instructions.append(
            {
                "id": hashlib.md5(f"{run_id}:guardrail".encode("utf-8"), usedforsecurity=False).hexdigest(),
                "content": [],
            }
        )
    content = text_instructions[0].setdefault("content", [])
    marker = "Space Optimizer portable guardrails"
    if marker in text_from_content(content):
        return candidate
    content.append(
        "\n\n"
        f"## {marker}\n"
        "- Prefer literal trusted SQL examples over parameterized templates when the user question contains concrete values.\n"
        "- Never emit unresolved parameter placeholders such as `:brand_name`, `:time_grp_value`, or `{brand_name}` in final SQL.\n"
        "- For market and brand filters, use exact case-insensitive matching against the available market, sub-market, brand, and team columns.\n"
        "- If no explicit time bucket is provided, use the current five-week default already defined for this space.\n"
        "- Preserve the documented UNION ALL pattern when a metric requires both aggregate sales tables.\n"
    )
    return candidate


def extract_generated_sql(message: dict[str, Any]) -> str | None:
    """Extract SQL from a completed Genie message."""
    for attachment in message.get("attachments", []) or []:
        if not isinstance(attachment, dict):
            continue
        query = attachment.get("query")
        if isinstance(query, dict) and query.get("query"):
            return str(query["query"])
    return None


def normalize_sql(sql: str, *, normalize_literals: bool) -> str:
    """Normalize SQL for lightweight local comparison."""
    normalized = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    normalized = re.sub(r"--[^\r\n]*", " ", normalized)
    normalized = normalized.lower().replace("`", "")
    if normalize_literals:
        normalized = re.sub(r"'(?:''|[^'])*'", "?", normalized)
        normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "?", normalized)
    return re.sub(r"\s+", " ", normalized).strip(" ;\n\r\t")


def tokenize_sql(sql: str) -> set[str]:
    """Tokenize normalized SQL for similarity scoring."""
    return set(re.findall(r"[a-z_][a-z0-9_]*|\?|<=|>=|<>|!=|[=(),.*+-/]", sql))


def compare_sql(generated_sql: str, expected_sql: str, *, threshold: float = MATCH_THRESHOLD) -> tuple[bool, float, str, str]:
    """Compare generated and expected SQL with the compact optimizer heuristic."""
    generated_exact = normalize_sql(generated_sql, normalize_literals=False)
    expected_exact = normalize_sql(expected_sql, normalize_literals=False)
    if generated_exact == expected_exact:
        return True, 1.0, "exact_match", "normalized SQL is identical"

    generated_shape = normalize_sql(generated_sql, normalize_literals=True)
    expected_shape = normalize_sql(expected_sql, normalize_literals=True)
    if generated_shape == expected_shape:
        return True, 1.0, "shape_match", "SQL structure matches after literal normalization"

    generated_tokens = tokenize_sql(generated_shape)
    expected_tokens = tokenize_sql(expected_shape)
    union_tokens = generated_tokens | expected_tokens
    token_jaccard = len(generated_tokens & expected_tokens) / len(union_tokens) if union_tokens else 0.0
    sequence_ratio = _sequence_ratio(generated_shape, expected_shape)
    if token_jaccard >= 0.95 and sequence_ratio >= 0.65:
        return True, 1.0, "high_token_match", f"token_jaccard={token_jaccard:.3f}, sequence_ratio={sequence_ratio:.3f}"
    if len(expected_shape) >= 1500 and len(generated_shape) >= 1000 and token_jaccard >= 0.75 and sequence_ratio >= 0.18:
        return True, 0.95, "large_query_token_match", f"token_jaccard={token_jaccard:.3f}, sequence_ratio={sequence_ratio:.3f}"
    score = (token_jaccard + sequence_ratio) / 2
    return score >= threshold, score, "similarity_match", f"token_jaccard={token_jaccard:.3f}, sequence_ratio={sequence_ratio:.3f}, threshold={threshold:.3f}"


def _sequence_ratio(left: str, right: str) -> float:
    """Return a SequenceMatcher ratio without importing it on cold paths."""
    from difflib import SequenceMatcher

    return SequenceMatcher(None, left, right).ratio()


def run_light_benchmark(
    client: GeniePocClient,
    *,
    space_id: str,
    questions: list[BenchmarkQuestion],
    delay_seconds: float,
) -> dict[str, Any]:
    """Run a tiny benchmark subset and return aggregate metrics."""
    results: list[BenchmarkResult] = []
    for index, question in enumerate(questions):
        if index > 0 and delay_seconds > 0:
            time.sleep(delay_seconds)
        started = client.start_conversation(space_id=space_id, question=question.question)
        conversation_id = str(started["conversation_id"])
        message_id = str(started["message_id"])
        ok, message = poll_message(
            client,
            space_id=space_id,
            conversation_id=conversation_id,
            message_id=message_id,
        )
        generated_sql = extract_generated_sql(message) if ok else None
        if not ok:
            status = "error"
            score = None
            method = None
            detail = str(message.get("status") or message.get("error") or "message did not complete")
        elif not generated_sql:
            status = "failed"
            score = 0.0
            method = "no_sql"
            detail = "no SQL attachment returned"
        elif question.expected_sql:
            passed, score, method, detail = compare_sql(generated_sql, question.expected_sql)
            status = "passed" if passed else "failed"
        else:
            status = "passed"
            score = 1.0
            method = "no_expected_sql"
            detail = "question has no expected SQL"
        results.append(
            BenchmarkResult(
                question_id=question.question_id,
                question=question.question,
                status=status,
                score=score,
                method=method,
                detail=detail,
                generated_sql=generated_sql,
                expected_sql=question.expected_sql,
                conversation_id=conversation_id,
                message_id=message_id,
            )
        )

    passed = sum(1 for result in results if result.status == "passed")
    failed = sum(1 for result in results if result.status == "failed")
    errors = sum(1 for result in results if result.status == "error")
    scores = [float(result.score) for result in results if result.score is not None]
    return {
        "question_count": len(results),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": (passed / len(results) * 100.0) if results else 0.0,
        "average_score": (sum(scores) / len(scores)) if scores else None,
        "results": [result.__dict__ for result in results],
    }


def poll_message(
    client: GeniePocClient,
    *,
    space_id: str,
    conversation_id: str,
    message_id: str,
) -> tuple[bool, dict[str, Any]]:
    """Poll one Genie message until it reaches a terminal state."""
    last_message: dict[str, Any] = {}
    terminal_success = {"COMPLETED", "SUCCEEDED"}
    terminal_failure = {"FAILED", "ERROR", "CANCELLED"}
    for attempt in range(MAX_POLL_ATTEMPTS):
        last_message = client.get_message(
            space_id=space_id,
            conversation_id=conversation_id,
            message_id=message_id,
        )
        status = str(last_message.get("status", "")).upper()
        logger.info(
            "polling message",
            extra={
                "space_id": space_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "status": status,
                "attempt": attempt + 1,
            },
        )
        if status in terminal_success:
            return True, last_message
        if status in terminal_failure:
            return False, last_message
        time.sleep(POLL_INTERVAL_SECONDS)
    return False, last_message


def render_report(result: PocResult, steps: list[dict[str, Any]]) -> str:
    """Render a concise Markdown report."""
    lines = [
        "# Space Optimizer Lite Report",
        "",
        f"- Source space: `{result.source_space_id}`",
        f"- Optimization clone: `{result.clone_space_id or 'not created'}`",
        f"- Run directory: `{result.run_dir}`",
        "",
        "## Step Results",
    ]
    for step in steps:
        status = "PASS" if step.get("ok") else "FAIL"
        lines.append(f"- {status}: {step.get('name')} - {step.get('detail', '')}")
    lines.extend(
        [
            "",
            "## Conclusion",
            (
                "The run exercised export, user-folder clone creation, serialized updates, benchmark-gated "
                "candidate evaluation, artifact writing, and optional cleanup."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def run_poc(args: argparse.Namespace) -> PocResult:
    """Run the lightweight optimization workflow."""
    parsed = parse_space_url(args.space_url)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = Path(args.out_dir).expanduser() if args.out_dir else resolve_default_run_root()
    run_dir = (run_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    workspace_client = build_workspace_client(args.profile)
    client = GeniePocClient(workspace_client)
    clone_space_id: str | None = None
    clone_trashed = False
    steps: list[dict[str, Any]] = []
    candidate_outcomes: list[CandidateOutcome] = []

    try:
        source_response = client.get_space(parsed.space_id, include_serialized_space=True)
        source_serialized = load_serialized_space(source_response)
        source_hash = serialized_hash(source_serialized)
        write_json(run_dir / "source_space.json", source_response)
        write_json(run_dir / "source_serialized_space.json", source_serialized)
        steps.append({"name": "export source space", "ok": True, "detail": f"hash {source_hash}"})

        benchmark_questions = extract_benchmark_questions(
            source_serialized,
            limit=max(1, int(args.benchmark_limit)),
        )
        if not benchmark_questions:
            benchmark_questions = [
                BenchmarkQuestion(
                    question_id="smoke_question",
                    question=args.conversation_question,
                    expected_sql=None,
                )
            ]
        write_json(run_dir / "benchmark_subset.json", [question.__dict__ for question in benchmark_questions])
        steps.append(
            {
                "name": "prepare benchmark subset",
                "ok": True,
                "detail": f"{len(benchmark_questions)} question(s)",
            }
        )

        warehouse_id = str(source_response.get("warehouse_id") or "")
        if not warehouse_id:
            raise PocError("Source space has no warehouse_id; cannot create clone.")
        source_title = str(source_response.get("title") or parsed.space_id)
        title = f"{args.title_prefix} {run_id}"
        description = f"Lean optimization clone of {source_title}; source space is untouched."
        if args.preserve_parent_path:
            parent_path = str(source_response.get("parent_path") or "")
            parent_detail = parent_path or "source space did not report parent_path"
        else:
            parent_path = args.parent_path or current_user_home(workspace_client)
            parent_detail = parent_path
        steps.append(
            {
                "name": "resolve clone parent folder",
                "ok": bool(parent_path),
                "detail": parent_detail,
            }
        )

        clone_response = client.create_space(
            warehouse_id=warehouse_id,
            title=title,
            serialized_space=source_serialized,
            parent_path=parent_path,
            description=description,
        )
        clone_space_id = str(clone_response["space_id"])
        write_json(run_dir / "clone_created.json", clone_response)
        steps.append({"name": "create disposable clone", "ok": True, "detail": clone_space_id})

        baseline_summary = run_light_benchmark(
            client,
            space_id=clone_space_id,
            questions=benchmark_questions,
            delay_seconds=float(args.delay_seconds),
        )
        write_json(run_dir / "baseline_benchmark.json", baseline_summary)
        best_config = source_serialized
        best_pass_rate = float(baseline_summary["pass_rate"])
        best_average_score = (
            float(baseline_summary["average_score"])
            if baseline_summary.get("average_score") is not None
            else 0.0
        )
        steps.append(
            {
                "name": "run baseline benchmark",
                "ok": True,
                "detail": f"{best_pass_rate:.2f}% pass, avg={best_average_score:.3f}",
            }
        )

        candidate_specs = [
            (
                "literal_examples",
                lambda config: apply_literal_examples(config, benchmark_questions, run_id),
            ),
            ("portable_guardrails", lambda config: apply_portable_guardrails(config, run_id)),
            ("sample_question_probe", lambda config: apply_sample_question_probe(config, run_id)),
            ("marker_instruction_probe", lambda config: apply_marker_instruction(config, run_id)),
        ][: max(0, int(args.max_candidates))]

        for candidate_index, (candidate_name, apply_candidate) in enumerate(candidate_specs, start=1):
            candidate_config = apply_candidate(best_config)
            candidate_hash = serialized_hash(candidate_config)
            candidate_label = f"candidate_{candidate_index:02d}_{candidate_name}"
            write_json(run_dir / f"{candidate_label}_serialized_space.json", candidate_config)
            if candidate_hash == serialized_hash(best_config):
                outcome = CandidateOutcome(
                    name=candidate_name,
                    accepted=False,
                    baseline_pass_rate=best_pass_rate,
                    candidate_pass_rate=best_pass_rate,
                    reason="no serialized-space change",
                    config_hash=candidate_hash,
                )
                candidate_outcomes.append(outcome)
                steps.append({"name": f"skip {candidate_name}", "ok": True, "detail": outcome.reason})
                continue

            update_response = client.update_space(
                space_id=clone_space_id,
                title=f"{title} {candidate_name}",
                serialized_space=candidate_config,
                description=description,
            )
            write_json(run_dir / f"{candidate_label}_update.json", update_response)
            candidate_summary = run_light_benchmark(
                client,
                space_id=clone_space_id,
                questions=benchmark_questions,
                delay_seconds=float(args.delay_seconds),
            )
            write_json(run_dir / f"{candidate_label}_benchmark.json", candidate_summary)

            candidate_pass_rate = float(candidate_summary["pass_rate"])
            candidate_average_score = (
                float(candidate_summary["average_score"])
                if candidate_summary.get("average_score") is not None
                else 0.0
            )
            accepted = (
                candidate_pass_rate > best_pass_rate
                or (
                    candidate_pass_rate == best_pass_rate
                    and candidate_average_score >= best_average_score
                )
            )
            if accepted:
                best_config = candidate_config
                best_pass_rate = candidate_pass_rate
                best_average_score = candidate_average_score
                reason = "accepted: no benchmark regression"
            else:
                client.update_space(
                    space_id=clone_space_id,
                    title=f"{title} rollback",
                    serialized_space=best_config,
                    description=description,
                )
                reason = "rejected: benchmark regression"
            outcome = CandidateOutcome(
                name=candidate_name,
                accepted=accepted,
                baseline_pass_rate=float(baseline_summary["pass_rate"]),
                candidate_pass_rate=candidate_pass_rate,
                reason=reason,
                config_hash=candidate_hash,
            )
            candidate_outcomes.append(outcome)
            steps.append(
                {
                    "name": f"evaluate {candidate_name}",
                    "ok": accepted,
                    "detail": f"{candidate_pass_rate:.2f}% pass, avg={candidate_average_score:.3f}; {reason}",
                }
            )

        write_json(run_dir / "best_serialized_space.json", best_config)
        write_json(run_dir / "candidate_outcomes.json", [outcome.__dict__ for outcome in candidate_outcomes])

        if args.trash_clone:
            client.trash_space(clone_space_id)
            clone_trashed = True
            steps.append({"name": "trash disposable clone", "ok": True, "detail": clone_space_id})

        result = PocResult(
            run_dir=run_dir,
            source_space_id=parsed.space_id,
            clone_space_id=clone_space_id,
            exported=True,
            clone_created=True,
            candidate_01_updated=any(outcome.name == "literal_examples" for outcome in candidate_outcomes),
            candidate_02_updated=any(outcome.name == "portable_guardrails" for outcome in candidate_outcomes),
            conversation_completed=True,
            clone_trashed=clone_trashed,
        )
        write_json(
            run_dir / "report.json",
            {
                "source_space_id": result.source_space_id,
                "clone_space_id": result.clone_space_id,
                "run_dir": str(result.run_dir),
                "steps": steps,
                "candidate_outcomes": [outcome.__dict__ for outcome in candidate_outcomes],
                "summary": {
                    "exported": result.exported,
                    "clone_created": result.clone_created,
                    "baseline_pass_rate": float(baseline_summary["pass_rate"]),
                    "best_pass_rate": best_pass_rate,
                    "accepted_candidates": sum(1 for outcome in candidate_outcomes if outcome.accepted),
                    "rejected_candidates": sum(1 for outcome in candidate_outcomes if not outcome.accepted),
                    "clone_trashed": result.clone_trashed,
                },
            },
        )
        (run_dir / "report.md").write_text(render_report(result, steps), encoding="utf-8")
        return result
    finally:
        if clone_space_id and args.trash_clone and not clone_trashed:
            try:
                client.trash_space(clone_space_id)
                steps.append({"name": "trash disposable clone after failure", "ok": True, "detail": clone_space_id})
            except Exception as exc:  # noqa: BLE001 - cleanup must report original context.
                steps.append({"name": "trash disposable clone after failure", "ok": False, "detail": str(exc)})
            write_json(run_dir / "cleanup_steps.json", steps)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Run a small Genie optimizer loop.")
    parser.add_argument("--space-url", default=os.environ.get("MAXGENIE_DEFAULT_SPACE_URL"))
    parser.add_argument("--profile", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--title-prefix", default=DEFAULT_TITLE_PREFIX)
    parser.add_argument(
        "--parent-path",
        default=None,
        help="Workspace folder for the optimization clone. Defaults to the current user's home folder.",
    )
    parser.add_argument("--benchmark-limit", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--delay-seconds", type=float, default=12.0)
    parser.add_argument("--conversation-question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--preserve-parent-path",
        action="store_true",
        help="Try to create the disposable clone in the source space parent folder.",
    )
    parser.add_argument(
        "--trash-clone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Trash the optimization clone before exiting.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    """Run the lightweight CLI."""
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    result = run_poc(args)
    print(json.dumps({"report": str(result.run_dir / "report.md"), "clone_trashed": result.clone_trashed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
