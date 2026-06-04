"""Multi-sample scoring and minimum-detectable-difference (MDD) core.

Genie SQL generation is non-deterministic, so a single benchmark run is a noisy
draw. Promoting a candidate on a single-run lift can therefore "promote noise".
This module provides the pure, side-effect-free statistics that turn K repeated
benchmark runs into a noise-aware promotion decision:

* ``aggregate_samples`` collapses K ``BenchmarkRun`` repeats of one question set
  into a mean pass rate, a ddof=1 sample standard deviation, and per-question
  pass fractions (which question flips run-to-run).
* ``minimum_detectable_difference`` is ``z * max(sigma)`` across subjects -- the
  smallest full-score lift that is distinguishable from noise.
* ``lift_threshold`` generalises the MDD to the two-sample case so repeating a
  candidate K times shrinks the bar it must clear by ~sqrt(K).
* ``promotion_decision`` composes the full-score significance test with a
  holdout non-regression guard and returns a serialisable verdict.

Everything here is deterministic and unit-tested; the orchestrator wires the
benchmark execution (and the frozen-seed comparability contract) around it.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence

from maxgenie.benchmark_service import BenchmarkRun, BenchmarkStatus

__all__ = [
    "DEFAULT_MDD_Z",
    "DEFAULT_HOLDOUT_EPSILON",
    "NOISE_MODEL_SCHEMA",
    "TRANSFER_GATE_SCHEMA",
    "SampleAggregate",
    "PromotionDecision",
    "TransferGateDecision",
    "sample_std",
    "aggregate_samples",
    "minimum_detectable_difference",
    "lift_threshold",
    "promotion_decision",
    "build_noise_model",
    "evaluate_transfer_gate",
]

NOISE_MODEL_SCHEMA = "maxgenie.noise_model.v1"
TRANSFER_GATE_SCHEMA = "maxgenie.transfer_gate.v1"

# z = 2 is the ~2-sigma (~95%) gate used for the minimum detectable difference.
DEFAULT_MDD_Z = 2.0
# Holdout is allowed to wobble within this many points before it counts as a
# regression (mirrors the orchestrator's terminal-selection epsilon).
DEFAULT_HOLDOUT_EPSILON = 0.1


def sample_std(values: Sequence[float]) -> float:
    """ddof=1 sample standard deviation; 0.0 when fewer than two values."""
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(finite) < 2:
        return 0.0
    return float(statistics.stdev(finite))


@dataclass(frozen=True)
class SampleAggregate:
    """Aggregate of K benchmark repeats over a single question set."""

    n: int
    pass_rates: tuple[float, ...]
    mean_pass_rate: float
    sample_std: float
    per_question_pass_fraction: dict[str, float]
    per_question_pass_counts: dict[str, int]
    per_question_total_counts: dict[str, int]

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "pass_rates": [round(r, 4) for r in self.pass_rates],
            "mean_pass_rate": round(self.mean_pass_rate, 4),
            "sample_std": round(self.sample_std, 4),
            "per_question_pass_fraction": {
                qid: round(frac, 4) for qid, frac in self.per_question_pass_fraction.items()
            },
            "per_question_pass_counts": dict(self.per_question_pass_counts),
            "per_question_total_counts": dict(self.per_question_total_counts),
        }

    @property
    def flipping_question_ids(self) -> tuple[str, ...]:
        """Questions whose pass fraction is strictly between 0 and 1."""
        return tuple(
            qid
            for qid, frac in sorted(self.per_question_pass_fraction.items())
            if 0.0 < frac < 1.0
        )


def aggregate_samples(runs: Sequence[BenchmarkRun]) -> SampleAggregate:
    """Collapse K repeated benchmark runs of one question set into an aggregate."""
    runs = list(runs)
    if not runs:
        raise ValueError("aggregate_samples requires at least one BenchmarkRun")

    pass_rates = tuple(float(run.pass_rate) for run in runs)
    pass_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    for run in runs:
        for result in run.results:
            qid = result.question_id
            total_counts[qid] = total_counts.get(qid, 0) + 1
            if result.status == BenchmarkStatus.PASSED:
                pass_counts[qid] = pass_counts.get(qid, 0) + 1
            else:
                pass_counts.setdefault(qid, 0)

    per_question_fraction = {
        qid: (pass_counts.get(qid, 0) / total_counts[qid]) if total_counts[qid] else 0.0
        for qid in sorted(total_counts)
    }

    mean_rate = sum(pass_rates) / len(pass_rates)
    return SampleAggregate(
        n=len(runs),
        pass_rates=pass_rates,
        mean_pass_rate=mean_rate,
        sample_std=sample_std(pass_rates),
        per_question_pass_fraction=per_question_fraction,
        per_question_pass_counts={qid: pass_counts.get(qid, 0) for qid in sorted(total_counts)},
        per_question_total_counts={qid: total_counts[qid] for qid in sorted(total_counts)},
    )


def minimum_detectable_difference(*sample_stds: float | None, z: float = DEFAULT_MDD_Z) -> float:
    """The smallest full-score lift distinguishable from noise: ``z * max(sigma)``."""
    sigmas = [float(s) for s in sample_stds if s is not None and math.isfinite(float(s))]
    if not sigmas:
        return 0.0
    return z * max(sigmas)


def lift_threshold(
    sigma: float,
    *,
    candidate_n: int = 1,
    baseline_n: int | None = None,
    z: float = DEFAULT_MDD_Z,
) -> float:
    """Two-sample significance threshold a candidate's lift must exceed.

    ``z * sqrt(sigma^2 / candidate_n + sigma^2 / baseline_n)``.

    When ``baseline_n`` is ``None`` (or non-positive / non-finite) the baseline is
    treated as a fixed reference and the second term drops, so a single-run
    candidate against a fixed baseline reduces to ``z * sigma`` (the MDD).
    """
    sigma = abs(float(sigma))
    candidate_n = max(int(candidate_n), 1)
    variance = sigma**2 / candidate_n
    if baseline_n is not None and math.isfinite(baseline_n) and baseline_n > 0:
        variance += sigma**2 / int(baseline_n)
    return z * math.sqrt(variance)


@dataclass(frozen=True)
class PromotionDecision:
    """Noise-aware verdict on whether a candidate beats the baseline."""

    promote: bool
    full_lift: float
    full_threshold: float
    full_significant: bool
    holdout_delta: float | None
    holdout_non_regressed: bool
    reasons: tuple[str, ...]
    detail: str

    def as_dict(self) -> dict:
        return {
            "promote": self.promote,
            "full_lift": round(self.full_lift, 4),
            "full_threshold": round(self.full_threshold, 4),
            "full_significant": self.full_significant,
            "holdout_delta": (round(self.holdout_delta, 4) if self.holdout_delta is not None else None),
            "holdout_non_regressed": self.holdout_non_regressed,
            "reasons": list(self.reasons),
            "detail": self.detail,
        }


def promotion_decision(
    *,
    candidate_full_mean: float,
    baseline_full_mean: float,
    sigma_full: float,
    candidate_n: int = 1,
    baseline_n: int | None = None,
    candidate_holdout_mean: float | None = None,
    baseline_holdout_mean: float | None = None,
    holdout_epsilon: float = DEFAULT_HOLDOUT_EPSILON,
    z: float = DEFAULT_MDD_Z,
) -> PromotionDecision:
    """Compose the full-score significance test with a holdout non-regression guard.

    A candidate is promotable only when its mean full lift over the baseline
    *strictly* exceeds the noise threshold AND its holdout mean does not regress
    beyond ``holdout_epsilon``. Returns a structured, serialisable verdict; the
    cross-space transfer gate (when active) is composed by the caller on top of
    this result.
    """
    full_lift = float(candidate_full_mean) - float(baseline_full_mean)
    threshold = lift_threshold(sigma_full, candidate_n=candidate_n, baseline_n=baseline_n, z=z)
    full_significant = full_lift > threshold

    reasons: list[str] = []
    if not full_significant:
        reasons.append("full_lift_below_mdd")

    holdout_delta: float | None = None
    holdout_non_regressed = True
    if candidate_holdout_mean is not None and baseline_holdout_mean is not None:
        holdout_delta = float(candidate_holdout_mean) - float(baseline_holdout_mean)
        if holdout_delta < -abs(holdout_epsilon):
            holdout_non_regressed = False
            reasons.append("holdout_regressed")

    promote = not reasons
    detail = (
        f"full_lift={full_lift:.2f} vs threshold={threshold:.2f} "
        f"(sigma={float(sigma_full):.2f}, n_cand={candidate_n}, n_base={baseline_n}); "
        f"holdout_delta={holdout_delta if holdout_delta is None else round(holdout_delta, 2)}"
    )
    return PromotionDecision(
        promote=promote,
        full_lift=full_lift,
        full_threshold=threshold,
        full_significant=full_significant,
        holdout_delta=holdout_delta,
        holdout_non_regressed=holdout_non_regressed,
        reasons=tuple(reasons),
        detail=detail,
    )


def build_noise_model(
    subjects: Mapping[str, Mapping[str, float] | SampleAggregate],
    *,
    z: float = DEFAULT_MDD_Z,
) -> dict:
    """Characterise run-to-run noise across one or more measured subjects.

    ``subjects`` maps a label (e.g. ``"baseline"``, ``"incumbent"``) to either a
    ``SampleAggregate`` or its ``as_dict()`` form (any mapping carrying
    ``mean_pass_rate``, ``sample_std``, and ``n``). Only subjects with ``n > 1``
    and a positive sample std contribute to the noise estimate.

    Returns a serialisable noise model: the conservative full-score sigma
    (``max`` across subjects), the minimum detectable difference (``z * sigma``),
    the single-run promotion threshold in points, and a human-readable predicate.
    This is the artifact the terminal-selection gate consumes via
    ``--noise-sigma-full``.
    """

    def _field(source: Mapping[str, float] | SampleAggregate, name: str, default: float) -> float:
        if isinstance(source, SampleAggregate):
            return float(getattr(source, name))
        value = source.get(name)
        return float(value) if value is not None else float(default)

    sigmas: list[float] = []
    subject_summaries: dict[str, dict] = {}
    for label, source in subjects.items():
        std = _field(source, "sample_std", 0.0)
        n = int(_field(source, "n", 1))
        mean = _field(source, "mean_pass_rate", 0.0)
        contributes = n > 1 and std > 0
        if contributes:
            sigmas.append(std)
        subject_summaries[label] = {
            "mean_pass_rate": round(mean, 4),
            "sample_std": round(std, 4),
            "n": n,
            "contributes_to_sigma": contributes,
        }

    sigma_hat = max(sigmas) if sigmas else 0.0
    mdd_full = z * sigma_hat
    return {
        "schema": NOISE_MODEL_SCHEMA,
        "z": z,
        "sigma_hat_full": round(sigma_hat, 4),
        "mdd_full": round(mdd_full, 4),
        "single_run_full_threshold_pts": round(mdd_full, 4),
        "subjects": subject_summaries,
        "promotion_predicate": (
            "Promote a single-run candidate only when "
            f"(final_full_mean - baseline_full_mean) > {round(mdd_full, 4)} points "
            "(z*sigma). For K repeated samples the bar shrinks to "
            "z*sqrt(sigma^2/n_candidate + sigma^2/n_baseline)."
        ),
    }


@dataclass(frozen=True)
class TransferGateDecision:
    """Cross-space robustness verdict composed on top of per-space promotions.

    A candidate that wins on the primary space is only *robust* if it also
    re-clears its noise-aware promotion on every configured robustness space,
    each evaluated under that space's own frozen seed and noise model. When no
    robustness spaces are configured the gate is inactive and passes the primary
    decision through unchanged (the "default space only" posture).
    """

    promote: bool
    primary_promote: bool
    active: bool
    transfer_space_count: int
    passing_space_count: int
    failing_space_ids: tuple[str, ...]
    per_space: tuple[dict, ...]
    reasons: tuple[str, ...]
    detail: str

    def as_dict(self) -> dict:
        return {
            "schema": TRANSFER_GATE_SCHEMA,
            "promote": self.promote,
            "primary_promote": self.primary_promote,
            "active": self.active,
            "transfer_space_count": self.transfer_space_count,
            "passing_space_count": self.passing_space_count,
            "failing_space_ids": list(self.failing_space_ids),
            "per_space": [dict(entry) for entry in self.per_space],
            "reasons": list(self.reasons),
            "detail": self.detail,
        }


def _normalise_transfers(
    transfers: (
        Mapping[str, PromotionDecision]
        | Sequence[tuple[str, PromotionDecision]]
        | None
    ),
) -> list[tuple[str, PromotionDecision]]:
    if not transfers:
        return []
    if isinstance(transfers, Mapping):
        return [(str(space_id), decision) for space_id, decision in transfers.items()]
    return [(str(space_id), decision) for space_id, decision in transfers]


def evaluate_transfer_gate(
    *,
    primary: PromotionDecision,
    transfers: (
        Mapping[str, PromotionDecision]
        | Sequence[tuple[str, PromotionDecision]]
        | None
    ) = None,
) -> TransferGateDecision:
    """Compose the primary promotion with per-robustness-space promotions.

    ``primary`` is the noise-aware ``PromotionDecision`` on the space being
    optimised. ``transfers`` maps each robustness space id to *its own*
    ``PromotionDecision`` (computed against that space's own baseline and sigma).
    The candidate is promoted only when the primary promotes AND every transfer
    space promotes -- i.e. the win generalises rather than overfitting one space.

    With ``transfers`` empty or ``None`` the gate is inactive: it returns the
    primary verdict unchanged so the single-space ("default space only") posture
    is byte-for-byte preserved. Flipping robustness on is purely additive -- add
    the spaces' decisions and the same predicate begins to bind.
    """
    pairs = _normalise_transfers(transfers)
    primary_promote = bool(primary.promote)

    per_space: list[dict] = []
    failing: list[str] = []
    for space_id, decision in pairs:
        promote = bool(decision.promote)
        per_space.append(
            {
                "space_id": space_id,
                "promote": promote,
                "full_lift": round(decision.full_lift, 4),
                "full_threshold": round(decision.full_threshold, 4),
                "reasons": list(decision.reasons),
            }
        )
        if not promote:
            failing.append(space_id)

    if not pairs:
        return TransferGateDecision(
            promote=primary_promote,
            primary_promote=primary_promote,
            active=False,
            transfer_space_count=0,
            passing_space_count=0,
            failing_space_ids=(),
            per_space=(),
            reasons=("transfer_gate_inactive_no_robustness_spaces",),
            detail="Transfer gate inactive: no robustness spaces configured (default space only).",
        )

    reasons: list[str] = []
    if not primary_promote:
        reasons.append("primary_not_promotable")
    for space_id in failing:
        reasons.append(f"transfer_failed:{space_id}")

    promote = primary_promote and not failing
    passing = len(pairs) - len(failing)
    detail = (
        f"primary_promote={primary_promote}; "
        f"transfer_spaces={len(pairs)}; passing={passing}; "
        f"failing={failing if failing else 'none'}"
    )
    return TransferGateDecision(
        promote=promote,
        primary_promote=primary_promote,
        active=True,
        transfer_space_count=len(pairs),
        passing_space_count=passing,
        failing_space_ids=tuple(failing),
        per_space=tuple(per_space),
        reasons=tuple(reasons),
        detail=detail,
    )
