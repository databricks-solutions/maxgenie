"""Safe patch application for decomposed Genie space artifacts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from maxgenie.ids import coerce_artifact_id, is_valid_artifact_id


ARTIFACT_PATCH_SEQUENCE_FILENAME = "patch_sequence.json"
_MAX_CONTENT_BYTES = 1_000_000
_ALLOWED_EXACT_PATHS = frozenset(
    {
        "instructions/text_instruction.md",
        "instructions/text_instruction.meta.yml",
        "instructions/sql_functions.yml",
        "sample_questions.yml",
    }
)
_ALLOWED_DIRECTORY_SUFFIXES = (
    ("tables", ".yml"),
    ("metric_views", ".yml"),
    ("instructions/examples", ".yml"),
    ("instructions/join_specs", ".yml"),
    ("instructions/sql_snippets/filters", ".yml"),
    ("instructions/sql_snippets/expressions", ".yml"),
    ("instructions/sql_snippets/measures", ".yml"),
)
_FILE_EDIT_NULLABLE_FIELDS = (
    "path",
    "relative_path",
    "content",
    "append_content",
    "find",
    "replace",
    "expected_sha256",
    "reason",
    "description",
    "comment",
)


def apply_artifact_patch_candidate(
    space_dir: str | Path,
    payload: Mapping[str, Any],
    *,
    dry_run: bool = False,
    max_file_edits: int | None = None,
    require_existing_sha256: bool = False,
    require_edit_reason: bool = False,
    reject_new_bind_placeholders: bool = False,
) -> dict[str, Any]:
    """Apply a file-edit candidate to a decomposed Genie space.

    The operation is transactional. If any edit is rejected, no files are
    written. Valid writes use a temporary file in the target directory followed
    by ``os.replace`` so readers never observe partial file content.
    """
    root = Path(space_dir).expanduser().resolve()
    summary: dict[str, Any] = {
        "changed": False,
        "dry_run": dry_run,
        "space_dir": str(root),
        "edit_count": 0,
        "changed_count": 0,
        "skipped_count": 0,
        "rejected_count": 0,
        "changed_edits": [],
        "skipped_edits": [],
        "rejected_edits": [],
    }

    file_edits = payload.get("file_edits") if isinstance(payload, Mapping) else None
    if not isinstance(file_edits, list):
        summary["rejected_edits"].append(
            {"path": None, "reason": "file_edits_must_be_list"}
        )
        _refresh_counts(summary)
        return summary
    if max_file_edits is not None and len(file_edits) > max_file_edits:
        summary["edit_count"] = len(file_edits)
        summary["rejected_edits"].append(
            {
                "path": None,
                "reason": "too_many_file_edits",
                "max_file_edits": max_file_edits,
                "actual_file_edits": len(file_edits),
            }
        )
        _refresh_counts(summary)
        return summary

    summary["edit_count"] = len(file_edits)
    prepared_edits: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for index, edit in enumerate(file_edits):
        prepared = _prepare_edit(
            root,
            edit,
            index=index,
            seen_paths=seen_paths,
            require_existing_sha256=require_existing_sha256,
            require_edit_reason=require_edit_reason,
            reject_new_bind_placeholders=reject_new_bind_placeholders,
        )
        if prepared["status"] == "rejected":
            summary["rejected_edits"].append(prepared["result"])
            continue
        if prepared["status"] == "skipped":
            summary["skipped_edits"].append(prepared["result"])
            continue
        prepared_edits.append(prepared)

    if summary["rejected_edits"]:
        for prepared in prepared_edits:
            result = dict(prepared["result"])
            result["reason"] = "payload_has_rejections"
            summary["skipped_edits"].append(result)
        _refresh_counts(summary)
        return summary

    if dry_run:
        summary["changed_edits"].extend(prepared["result"] for prepared in prepared_edits)
        summary["changed"] = bool(prepared_edits)
        _refresh_counts(summary)
        return summary

    for prepared in prepared_edits:
        target = prepared["target"]
        content = prepared["content"]
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(target, content)
        summary["changed_edits"].append(prepared["result"])

    summary["changed"] = bool(prepared_edits)
    _refresh_counts(summary)
    return summary


def build_artifact_patch_sequence_from_space_dirs(
    before_space_dir: str | Path,
    after_space_dir: str | Path,
    *,
    candidate_name_prefix: str = "artifact_patch_replay",
    candidate_mode: str = "artifact_patch_v2",
    max_file_edits: int = 2,
) -> dict[str, Any]:
    """Build strict patch payloads that transform one decomposed space into another."""

    if max_file_edits < 1:
        raise ValueError("max_file_edits must be at least 1.")

    before_root = Path(before_space_dir).expanduser().resolve()
    after_root = Path(after_space_dir).expanduser().resolve()
    if not before_root.exists():
        raise ValueError(f"Before space directory does not exist: {before_root}")
    if not after_root.exists():
        raise ValueError(f"After space directory does not exist: {after_root}")

    before_all_files = _collect_space_files(before_root)
    after_all_files = _collect_space_files(after_root)
    unsupported_changes = _unsupported_changed_paths(before_all_files, after_all_files)
    if unsupported_changes:
        raise ValueError(
            "Artifact patch sequence contains unsupported file changes: "
            + ", ".join(unsupported_changes)
        )

    before_files = _collect_patchable_space_files(before_root)
    after_files = _collect_patchable_space_files(after_root)
    deleted_paths = sorted(set(before_files) - set(after_files))
    if deleted_paths:
        raise ValueError(
            "Artifact patch sequence does not support file deletes: "
            + ", ".join(deleted_paths)
        )

    file_edits: list[dict[str, Any]] = []
    for relative_path in sorted(after_files):
        after_text = after_files[relative_path].read_text(encoding="utf-8")
        before_path = before_files.get(relative_path)
        before_text = (
            before_path.read_text(encoding="utf-8")
            if before_path is not None
            else None
        )
        if before_text == after_text:
            continue
        edit = {
            "path": relative_path,
            "content": after_text,
            "reason": (
                "Replay known positive-control file delta"
                if before_path is not None
                else "Replay known positive-control file creation"
            ),
        }
        if before_text is not None:
            edit["expected_sha256"] = _sha256_text(before_text)
        file_edits.append(edit)

    payloads: list[dict[str, Any]] = []
    for chunk_index, start in enumerate(range(0, len(file_edits), max_file_edits), start=1):
        chunk = file_edits[start : start + max_file_edits]
        payloads.append(
            {
                "candidate_name": f"{candidate_name_prefix}_{chunk_index:02d}",
                "candidate_mode": candidate_mode,
                "rationale": (
                    "Replay a known positive-control patch in bounded, hash-checked chunks."
                ),
                "expected_impact": (
                    "The resulting decomposed space should match the known positive-control target."
                ),
                "risk": "Low if every expected_sha256 check passes before applying the chunk.",
                "operations": [],
                "file_edits": chunk,
            }
        )

    return {
        "schema_version": "maxgenie.artifact_patch_sequence.v1",
        "candidate_mode": candidate_mode,
        "max_file_edits": max_file_edits,
        "before_space_dir": str(before_root),
        "after_space_dir": str(after_root),
        "file_edit_count": len(file_edits),
        "payload_count": len(payloads),
        "payloads": payloads,
    }


def write_artifact_patch_sequence(
    destination_dir: str | Path,
    sequence: Mapping[str, Any],
) -> Path:
    """Persist an artifact patch sequence under a directory."""

    destination = Path(destination_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / ARTIFACT_PATCH_SEQUENCE_FILENAME
    path.write_text(json.dumps(sequence, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_artifact_patch_sequence(path: str | Path) -> list[dict[str, Any]]:
    """Load patch replay payloads from a sequence file or containing directory."""

    sequence_path = Path(path).expanduser()
    if sequence_path.is_dir():
        sequence_path = sequence_path / ARTIFACT_PATCH_SEQUENCE_FILENAME
    payload = json.loads(sequence_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Artifact patch sequence must be a JSON object: {sequence_path}")
    payloads = payload.get("payloads")
    if not isinstance(payloads, list):
        raise ValueError(f"Artifact patch sequence must contain a payloads list: {sequence_path}")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(payloads):
        if not isinstance(item, Mapping):
            raise ValueError(f"Artifact patch sequence payload {index} must be an object.")
        normalized.append(dict(item))
    return normalized


def repair_artifact_patch_v2_payload(
    space_dir: str | Path,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fill deterministic artifact_patch_v2 fields before strict validation."""

    root = Path(space_dir).expanduser().resolve()
    repaired = deepcopy(dict(payload))
    summary: dict[str, Any] = {
        "changed": False,
        "normalized_operations": False,
        "filled_expected_sha256_paths": [],
        "filled_reason_paths": [],
        "filled_nullable_file_edit_fields": [],
        "filled_sql_snippet_display_name_paths": [],
        "repaired_artifact_id_paths": [],
        "skipped_edits": [],
    }
    operations = repaired.get("operations")
    original_operations = operations
    if operations != []:
        repaired["operations"] = []
        summary["normalized_operations"] = True
        summary["changed"] = True

    file_edits = repaired.get("file_edits")
    if (
        file_edits is None
        and isinstance(original_operations, list)
        and original_operations
        and all(
            isinstance(operation, Mapping)
            and str(operation.get("op") or "").strip() == "no_change"
            for operation in original_operations
        )
    ):
        repaired["file_edits"] = []
        summary["changed"] = True
        file_edits = repaired["file_edits"]
    if not isinstance(file_edits, list):
        return repaired, summary

    for index, edit in enumerate(file_edits):
        if not isinstance(edit, dict):
            summary["skipped_edits"].append(
                {"index": index, "reason": "edit_must_be_mapping"}
            )
            continue
        alias_result = _resolve_edit_path_alias(edit)
        if alias_result["status"] == "rejected":
            summary["skipped_edits"].append(
                {
                    "index": index,
                    "path": alias_result["path"],
                    "relative_path": alias_result["relative_path"],
                    "reason": alias_result["reason"],
                }
            )
            continue
        raw_path = alias_result["raw_path"]
        path_result = _validate_relative_path(root, raw_path)
        if path_result["status"] == "rejected":
            summary["skipped_edits"].append(
                {
                    "index": index,
                    "path": raw_path if isinstance(raw_path, str) else None,
                    "reason": path_result["reason"],
                }
            )
            continue
        relative_path = path_result["relative_path"]
        target = path_result["target"]
        if not edit.get("path"):
            edit["path"] = relative_path
            summary["changed"] = True
        filled_nullable_fields = _normalize_file_edit_nullable_fields(edit)
        if filled_nullable_fields:
            summary["filled_nullable_file_edit_fields"].append(
                {"path": relative_path, "fields": filled_nullable_fields}
            )
            summary["changed"] = True
        if not str(edit.get("reason") or "").strip():
            edit["reason"] = _artifact_patch_edit_reason_fallback(edit)
            summary["filled_reason_paths"].append(relative_path)
            summary["changed"] = True
        if target.exists() and not edit.get("expected_sha256"):
            edit["expected_sha256"] = _sha256_file(target)
            summary["filled_expected_sha256_paths"].append(relative_path)
            summary["changed"] = True
        repaired_id_content, repaired_ids = _repair_invalid_artifact_id_content(
            edit.get("content"),
            relative_path=relative_path,
        )
        if repaired_id_content is not None:
            edit["content"] = repaired_id_content
            summary["repaired_artifact_id_paths"].extend(repaired_ids)
            summary["changed"] = True
        if _is_sql_snippet_path(relative_path):
            repaired_content = _repair_sql_snippet_display_name_content(
                edit.get("content")
            )
            if repaired_content is not None:
                edit["content"] = repaired_content
                summary["filled_sql_snippet_display_name_paths"].append(relative_path)
                summary["changed"] = True

    return repaired, summary


def _artifact_patch_edit_reason_fallback(edit: Mapping[str, Any]) -> str:
    for key in ("description", "comment"):
        value = str(edit.get(key) or "").strip()
        if value:
            return value
    return "deterministic_artifact_patch_v2_repair"


def _normalize_file_edit_nullable_fields(edit: dict[str, Any]) -> list[str]:
    filled: list[str] = []
    for field_name in _FILE_EDIT_NULLABLE_FIELDS:
        if field_name not in edit:
            edit[field_name] = None
            filled.append(field_name)
    return filled


def _resolve_edit_path_alias(edit: Mapping[str, Any]) -> dict[str, Any]:
    raw_path = edit.get("path")
    raw_relative_path = edit.get("relative_path")
    path = raw_path if isinstance(raw_path, str) and raw_path.strip() else None
    relative_path = (
        raw_relative_path
        if isinstance(raw_relative_path, str) and raw_relative_path.strip()
        else None
    )
    if path and relative_path and path != relative_path:
        return {
            "status": "rejected",
            "path": path,
            "relative_path": relative_path,
            "reason": "path_relative_path_mismatch",
        }
    return {"status": "ok", "raw_path": path or relative_path}


def _prepare_edit(
    root: Path,
    edit: Any,
    *,
    index: int,
    seen_paths: set[str],
    require_existing_sha256: bool,
    require_edit_reason: bool,
    reject_new_bind_placeholders: bool,
) -> dict[str, Any]:
    if not isinstance(edit, Mapping):
        return _rejected(index=index, path=None, reason="edit_must_be_mapping")

    alias_result = _resolve_edit_path_alias(edit)
    if alias_result["status"] == "rejected":
        return _rejected(
            index=index,
            path=alias_result["path"],
            relative_path=alias_result["relative_path"],
            reason=alias_result["reason"],
        )

    raw_path = alias_result["raw_path"]
    path_result = _validate_relative_path(root, raw_path)
    if path_result["status"] == "rejected":
        return _rejected(
            index=index,
            path=raw_path if isinstance(raw_path, str) else None,
            reason=path_result["reason"],
        )

    relative_path = path_result["relative_path"]
    target = path_result["target"]
    if relative_path in seen_paths:
        return _rejected(index=index, path=relative_path, reason="duplicate_path")
    seen_paths.add(relative_path)

    edit_reason = str(edit.get("reason") or "").strip()
    edit_description = str(edit.get("description") or "").strip()
    edit_comment = str(edit.get("comment") or "").strip()
    if require_edit_reason and not edit_reason:
        return _rejected(index=index, path=relative_path, reason="edit_reason_required")

    before_text = target.read_text(encoding="utf-8") if target.exists() else None
    before_hash = _sha256_text(before_text) if before_text is not None else None
    content_result = _materialize_edit_content(
        edit,
        before_text=before_text,
        index=index,
        relative_path=relative_path,
    )
    if content_result["status"] == "rejected":
        return content_result
    content = content_result["content"]
    edit_mode = content_result["edit_mode"]

    content_check = _validate_text_content(content)
    if content_check is not None:
        return _rejected(index=index, path=relative_path, reason=content_check)

    # Scope the new-bind guard to EXECUTABLE SQL. Prose guidance is exempt
    # entirely (a candidate must be able to *teach* literalization by naming the
    # placeholder it forbids), and in executable files, placeholders inside comments
    # or fenced examples are stripped before the diff — so the guard blocks only a
    # genuinely new executable bind, not the lesson that fixes the bug.
    if reject_new_bind_placeholders and not _is_prose_artifact_path(relative_path):
        new_placeholders = _new_bind_placeholders(
            _executable_sql_text(before_text or ""),
            _executable_sql_text(content),
        )
        if new_placeholders:
            return _rejected(
                index=index,
                path=relative_path,
                reason="new_bind_placeholders_not_allowed",
                placeholders=new_placeholders,
            )
    expected_hash = edit.get("expected_sha256")
    if require_existing_sha256 and before_hash is not None and expected_hash is None:
        return _rejected(
            index=index,
            path=relative_path,
            reason="expected_sha256_required_for_existing_file",
        )
    if expected_hash is not None:
        hash_check = _validate_expected_sha256(expected_hash)
        if hash_check is not None:
            return _rejected(index=index, path=relative_path, reason=hash_check)
        if before_hash != expected_hash:
            return _rejected(
                index=index,
                path=relative_path,
                reason="expected_sha256_mismatch",
                expected_sha256=expected_hash,
                actual_sha256=before_hash,
            )

    after_hash = _sha256_text(content)
    before_bytes = len(before_text.encode("utf-8")) if before_text is not None else 0
    after_bytes = len(content.encode("utf-8"))
    before_lines = len(before_text.splitlines()) if before_text is not None else 0
    after_lines = len(content.splitlines())
    result = {
        "index": index,
        "path": relative_path,
        "action": "created" if before_hash is None else "updated",
        "sha256_before": before_hash,
        "sha256_after": after_hash,
        "bytes": after_bytes,
        "bytes_before": before_bytes,
        "bytes_delta": after_bytes - before_bytes,
        "lines_before": before_lines,
        "lines_after": after_lines,
        "lines_delta": after_lines - before_lines,
        "edit_mode": edit_mode,
    }
    if edit_reason:
        result["reason"] = edit_reason
    if edit_description:
        result["description"] = edit_description
    if edit_comment:
        result["comment"] = edit_comment
    if before_hash == after_hash:
        return {
            "status": "skipped",
            "result": {**result, "reason": "content_unchanged"},
        }

    return {
        "status": "changed",
        "target": target,
        "content": content,
        "result": result,
    }


def _materialize_edit_content(
    edit: Mapping[str, Any],
    *,
    before_text: str | None,
    index: int,
    relative_path: str,
) -> dict[str, Any]:
    content = edit.get("content")
    append_content = edit.get("append_content")
    find_text = edit.get("find")
    replace_text = edit.get("replace")
    content_active = isinstance(content, str) and bool(content)
    append_active = isinstance(append_content, str) and bool(append_content)
    find_replace_active = isinstance(find_text, str) or isinstance(replace_text, str)
    active_modes = [
        mode
        for mode, active in (
            ("content", content_active),
            ("append_content", append_active),
            ("find_replace", find_replace_active),
        )
        if active
    ]
    if len(active_modes) > 1:
        return _rejected(
            index=index,
            path=relative_path,
            reason="multiple_edit_content_modes",
            edit_modes=active_modes,
        )

    if content_active:
        return {"status": "changed", "content": content, "edit_mode": "replace_file"}

    if append_active:
        if before_text is None:
            return _rejected(
                index=index,
                path=relative_path,
                reason="append_content_requires_existing_file",
            )
        return {
            "status": "changed",
            "content": before_text + append_content,
            "edit_mode": "append_content",
        }

    if isinstance(find_text, str) or isinstance(replace_text, str):
        if before_text is None:
            return _rejected(
                index=index,
                path=relative_path,
                reason="find_replace_requires_existing_file",
            )
        if not isinstance(find_text, str) or not find_text:
            return _rejected(index=index, path=relative_path, reason="find_must_be_nonempty_string")
        if not isinstance(replace_text, str):
            return _rejected(index=index, path=relative_path, reason="replace_must_be_string")
        occurrence_count = before_text.count(find_text)
        if occurrence_count == 0:
            return _rejected(index=index, path=relative_path, reason="find_text_not_found")
        if occurrence_count > 1:
            return _rejected(
                index=index,
                path=relative_path,
                reason="find_text_not_unique",
                occurrence_count=occurrence_count,
            )
        return {
            "status": "changed",
            "content": before_text.replace(find_text, replace_text, 1),
            "edit_mode": "find_replace",
        }

    return _rejected(
        index=index,
        path=relative_path,
        reason="content_or_compact_edit_required",
    )


def _collect_patchable_space_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for relative_path, path in _collect_space_files(root).items():
        if not _is_allowlisted_path(relative_path):
            continue
        path.read_text(encoding="utf-8")
        files[relative_path] = path
    return files


def _collect_space_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
    return files


def _unsupported_changed_paths(
    before_files: Mapping[str, Path],
    after_files: Mapping[str, Path],
) -> list[str]:
    unsupported: list[str] = []
    for relative_path in sorted(set(before_files) | set(after_files)):
        if _is_allowlisted_path(relative_path):
            continue
        before_path = before_files.get(relative_path)
        after_path = after_files.get(relative_path)
        if before_path is None or after_path is None:
            unsupported.append(relative_path)
            continue
        if _sha256_file(before_path) != _sha256_file(after_path):
            unsupported.append(relative_path)
    return unsupported


def _validate_relative_path(root: Path, raw_path: Any) -> dict[str, Any]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return {"status": "rejected", "reason": "path_must_be_nonempty_string"}
    if "\\" in raw_path:
        return {"status": "rejected", "reason": "backslash_not_allowed"}

    pure_path = PurePosixPath(raw_path)
    if pure_path.is_absolute():
        return {"status": "rejected", "reason": "absolute_path_not_allowed"}
    if any(part in {"", ".", ".."} for part in pure_path.parts):
        return {"status": "rejected", "reason": "path_traversal_not_allowed"}

    relative_path = pure_path.as_posix()
    if not _is_allowlisted_path(relative_path):
        return {"status": "rejected", "reason": "path_not_allowlisted"}

    target = root.joinpath(*pure_path.parts).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError:
        return {"status": "rejected", "reason": "path_outside_space_dir"}

    return {
        "status": "ok",
        "relative_path": relative_path,
        "target": target,
    }


def _is_allowlisted_path(relative_path: str) -> bool:
    if relative_path in _ALLOWED_EXACT_PATHS:
        return True
    for directory, suffix in _ALLOWED_DIRECTORY_SUFFIXES:
        prefix = f"{directory}/"
        if relative_path.startswith(prefix) and relative_path.endswith(suffix):
            filename = relative_path[len(prefix) :]
            if "/" not in filename and filename:
                return True
    return False


def _is_sql_snippet_path(relative_path: str) -> bool:
    return any(
        relative_path.startswith(f"instructions/sql_snippets/{group}/")
        and relative_path.endswith(".yml")
        for group in ("filters", "expressions", "measures")
    )


def _repair_sql_snippet_display_name_content(content: Any) -> str | None:
    if not isinstance(content, str):
        return None
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError:
        return None
    if not isinstance(data, Mapping):
        return None
    if str(data.get("display_name") or "").strip():
        return None
    display_name = _derive_sql_snippet_display_name(data)
    if not display_name:
        return None
    return yaml.safe_dump(
        {"display_name": display_name},
        sort_keys=False,
        allow_unicode=False,
        width=10_000,
    ) + content


def _repair_invalid_artifact_id_content(
    content: Any,
    *,
    relative_path: str,
) -> tuple[str | None, list[dict[str, str]]]:
    if not isinstance(content, str):
        return None, []
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError:
        return None, []

    repaired: list[dict[str, str]] = []

    def repair_mapping(item: Any, item_path: str) -> None:
        if not isinstance(item, dict) or "id" not in item:
            return
        original = item.get("id")
        if is_valid_artifact_id(original):
            return
        seed = yaml.safe_dump(
            {
                "path": relative_path,
                "item_path": item_path,
                "item": item,
            },
            sort_keys=True,
            allow_unicode=False,
            width=10_000,
        )
        item["id"] = coerce_artifact_id(original, seed=seed)
        repaired.append(
            {
                "path": relative_path,
                "item": item_path,
                "old_id": str(original or ""),
                "new_id": item["id"],
            }
        )

    if isinstance(data, dict):
        repair_mapping(data, "$")
    elif isinstance(data, list):
        for index, item in enumerate(data):
            repair_mapping(item, f"$[{index}]")
    else:
        return None, []

    if not repaired:
        return None, []
    return (
        yaml.safe_dump(
            data,
            sort_keys=False,
            allow_unicode=False,
            width=10_000,
        ),
        repaired,
    )


def _derive_sql_snippet_display_name(data: Mapping[str, Any]) -> str:
    for key in ("description", "question", "id", "alias"):
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            candidates = [str(item) for item in value if str(item).strip()]
        else:
            candidates = [str(value)]
        for candidate in candidates:
            compact = re.sub(r"\s+", " ", candidate.replace("_", " ")).strip()
            if compact:
                return _truncate_display_name(compact)
    return ""


def _truncate_display_name(value: str, *, max_chars: int = 96) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _validate_text_content(content: str) -> str | None:
    raw = content.encode("utf-8")
    if len(raw) > _MAX_CONTENT_BYTES:
        return "content_too_large"
    if "\x00" in content:
        return "binary_content_not_allowed"

    control_chars = sum(
        1
        for character in content
        if ord(character) < 32 and character not in "\n\r\t"
    )
    if content and control_chars / len(content) > 0.02:
        return "binary_content_not_allowed"
    return None


def _validate_expected_sha256(value: Any) -> str | None:
    if not isinstance(value, str):
        return "expected_sha256_must_be_string"
    if len(value) != 64:
        return "expected_sha256_must_be_64_hex"
    try:
        int(value, 16)
    except ValueError:
        return "expected_sha256_must_be_64_hex"
    return None


_PROSE_PATH_SUFFIXES = (".md", ".markdown", ".txt", ".rst")
_FENCED_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)


def _is_prose_artifact_path(relative_path: str) -> bool:
    """True for natural-language artifact files (e.g. instructions/text_instruction.md).

    A bind placeholder that only appears in prose is documentation a candidate uses to
    TEACH literalization (e.g. "never emit unbound :time_grp_type; use a literal"), not
    an executable bind. The new-bind guard must not fire on such files.
    """
    return relative_path.lower().endswith(_PROSE_PATH_SUFFIXES)


def _executable_sql_text(text: str) -> str:
    """Blank out the parts of a non-prose artifact that never execute as SQL — fenced
    code examples, single-quoted string literals, and ``--`` / ``/* */`` comments — so a
    placeholder counts as a *new bind* only when it appears in executable SQL.

    A single left-to-right scan tracks whether each character is inside a string, a line
    comment, or a block comment, instead of applying ordered regex passes. Ordered passes
    are unsafe here: a quote inside a comment and a comment marker inside a string can
    cross-pair (e.g. an apostrophe in ``-- it's`` binding to a later ``'...'`` literal),
    which can swallow a newline and hide a genuine new bind on a following line. The scan
    cannot cross-pair, so a ``:param`` in a comment or a string is dropped while one in
    real SQL survives. Fenced examples are removed first (documentation, not executable).
    """
    without_fences = _FENCED_CODE_BLOCK_RE.sub(" ", text)
    out: list[str] = []
    state = "code"  # one of: code, string, line_comment, block_comment
    i = 0
    n = len(without_fences)
    while i < n:
        ch = without_fences[i]
        nxt = without_fences[i + 1] if i + 1 < n else ""
        if state == "code":
            if ch == "'":
                state = "string"
                out.append(" ")
            elif ch == "-" and nxt == "-":
                state = "line_comment"
                out.append("  ")
                i += 2
                continue
            elif ch == "/" and nxt == "*":
                state = "block_comment"
                out.append("  ")
                i += 2
                continue
            else:
                out.append(ch)
        elif state == "string":
            if ch == "'" and nxt == "'":  # '' escapes a quote and stays in the string
                out.append("  ")
                i += 2
                continue
            if ch == "'":
                state = "code"
                out.append(" ")
            else:
                out.append("\n" if ch == "\n" else " ")
        elif state == "line_comment":
            if ch == "\n":
                state = "code"
                out.append("\n")
            else:
                out.append(" ")
        else:  # block_comment
            if ch == "*" and nxt == "/":
                state = "code"
                out.append("  ")
                i += 2
                continue
            out.append("\n" if ch == "\n" else " ")
        i += 1
    return "".join(out)


def _new_bind_placeholders(before_text: str, after_text: str) -> list[str]:
    before = set(_bind_placeholders(before_text))
    return sorted(set(_bind_placeholders(after_text)) - before)


def _bind_placeholders(content: str) -> list[str]:
    return re.findall(r"(?<!:):[A-Za-z_][A-Za-z0-9_]*|\$\{[A-Za-z_][A-Za-z0-9_]*(?:[^}]*)?\}", content)


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
            file.write(content)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _rejected(
    *,
    index: int,
    path: str | None,
    reason: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "status": "rejected",
        "result": {
            "index": index,
            "path": path,
            "reason": reason,
            **extra,
        },
    }


def _refresh_counts(summary: dict[str, Any]) -> None:
    summary["changed_count"] = len(summary["changed_edits"])
    summary["skipped_count"] = len(summary["skipped_edits"])
    summary["rejected_count"] = len(summary["rejected_edits"])
