"""Guards that keep lab-only artifacts out of the product package."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_product_repo_does_not_track_lab_artifact_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    tracked_files = result.stdout.splitlines()
    forbidden_prefixes = (
        ".agents/",
        ".omc/",
        ".race_",
        "agentic/",
        "best-practices/",
        "centaur/",
        "deterministic/",
        "experiment_artifacts/",
        "hybrid/",
        "local_context/",
        "reports/",
        "worker_builds/",
        "workspace/",
        "workspace-",
    )
    forbidden_exact = {
        "docs/UNIFIED_EXPERIMENT_REPORT.md",
        "docs/MAXGENIE_EXPERIMENT_LEDGER.md",
        "docs/NEXT_WAVE_RESULT_BASED_VALIDATION.md",
        "docs/START_NEXT_MAXGENIE_GOAL.md",
        "comparison.json",
        "NEXT_WAVE_LAYERED_OPTIMIZATION_PLAN.md",
    }

    forbidden = [
        path
        for path in tracked_files
        if path in forbidden_exact
        or any(path.startswith(prefix) for prefix in forbidden_prefixes)
    ]

    assert forbidden == []


def test_shared_ignore_covers_local_private_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ignore_content = (repo_root / ".gitignore").read_text()
    required_patterns = (
        ".agents/",
        ".omc/",
        ".race_*/",
        "agentic/",
        "best-practices/",
        "centaur/",
        "deterministic/",
        "hybrid/",
        "reports/",
        "workspace/",
        "comparison.json",
    )

    for pattern in required_patterns:
        assert pattern in ignore_content


def test_public_release_docs_exist_and_name_required_controls() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    required_files = (
        "CONTRIBUTING.md",
        "PUBLIC_RELEASE.md",
        "SECURITY.md",
    )
    required_phrases = (
        "synthetic fixture data",
        "third-party dependency license review",
        "no credentials",
        "no customer data",
    )

    for relative_path in required_files:
        assert (repo_root / relative_path).exists()

    combined = "\n".join((repo_root / path).read_text() for path in required_files)
    for phrase in required_phrases:
        assert phrase in combined


def test_product_docs_describe_post_collapse_existing_executable_guidance() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    required = (
        "repeated new executable example or SQL-snippet train collapse",
        "new_file_allowed=false",
        "prefer listed existing example/snippet files",
        "return no_change when no existing executable edit surface is available",
    )

    readme = (repo_root / "README.md").read_text()
    assert "docs/candidate-modes.md" in readme

    for relative_path in (
        "docs/candidate-modes.md",
        "genie_skills/maxgenie/SKILL.md",
    ):
        content = (repo_root / relative_path).read_text()
        for phrase in required:
            assert phrase in content


def test_product_docs_describe_skeleton_artifact_patch_mode() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    required = (
        "skeleton_artifact_patch_v2",
        "visible marker",
        "find anchor",
        "bounded replacement text",
        "hidden-holdout",
    )

    readme = (repo_root / "README.md").read_text()
    assert "docs/candidate-modes.md" in readme

    for relative_path in (
        "docs/candidate-modes.md",
        "genie_skills/maxgenie/SKILL.md",
    ):
        content = (repo_root / relative_path).read_text()
        for phrase in required:
            assert phrase in content
