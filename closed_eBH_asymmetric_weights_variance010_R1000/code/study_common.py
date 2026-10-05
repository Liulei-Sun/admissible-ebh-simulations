"""Shared deterministic definitions for the publication simulation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TOL = 1e-10
UINT64_MASK = (1 << 64) - 1
PERSISTENT_NOISE_SALT = 0x5052534953543032
SIGNED_SQUARE_RESIDUAL_SALT = 0x5349474E45445351
CONDITION_ORDER = {"clean": 1, "signed_square": 2, "persistent_noise": 3}
METHOD_ORDER = {
    "full_mean": 1,
    "testing_mean": 2,
    "testing_ebh": 3,
    "procedure1": 4,
    "procedure2": 5,
    "naive_mean": 6,
}
METHOD_LABELS = {
    "full_mean": "Mean eBH, full data",
    "testing_mean": "Mean eBH, testing sample only",
    "testing_ebh": "eBH, testing sample only",
    "procedure1": "Learned asymmetric closed eBH 1",
    "procedure2": "Learned asymmetric closed eBH 2",
    "naive_mean": "Naive mean eBH, pooled full data",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def format4(value: float) -> str:
    """Format publication cells to four decimals using explicit half-up rounding."""
    return format(
        Decimal(str(float(value))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
        ".4f",
    )


def splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return value ^ (value >> 31)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_config(root: Path) -> dict[str, Any]:
    path = Path(root) / "config" / "study.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    require(
        value.get("schema") == "closed-ebh-asymmetric-weights-publication-study-v1",
        "study schema mismatch",
    )
    expected_scalars = {
        "K": 200,
        "alpha": 0.05,
        "nonnull_counts": [20, 30],
        "signal_mean": 0.45,
        "e_value_tilt": 0.45,
        "training_observations": 20,
        "testing_observations": 40,
        "replications": 1000,
        "signed_square_residual_seed_salt_hex": hex(SIGNED_SQUARE_RESIDUAL_SALT),
    }
    for key, expected in expected_scalars.items():
        require(value.get(key) == expected, f"study configuration mismatch: {key}")
    conditions = value.get("conditions")
    require(
        isinstance(conditions, dict)
        and list(conditions) == ["clean", "signed_square", "persistent_noise"],
        "the package must contain exactly the three reported conditions",
    )
    persistent = conditions["persistent_noise"]
    require(
        float(persistent.get("error_variance", -1.0)) == 0.1
        and float(persistent.get("error_standard_deviation", -1.0))
        == math.sqrt(0.1)
        and persistent.get("seed_salt_hex") == hex(PERSISTENT_NOISE_SALT),
        "persistent-error configuration mismatch",
    )
    batches = value.get("batches")
    expected_batches = [
        {
            "id": "batch1_seed20260803",
            "seed": 20260803,
            "local_replications": 500,
            "global_replication_start": 1,
            "global_replication_end": 500,
        },
        {
            "id": "batch2_seed20260804",
            "seed": 20260804,
            "local_replications": 500,
            "global_replication_start": 501,
            "global_replication_end": 1000,
        },
    ]
    require(batches == expected_batches, "batch identities or seeds changed")
    return value


def ordinary_ebh(e: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(e, dtype=float)
    k = values.size
    order = np.lexsort((np.arange(k), -values))
    selected = np.zeros(k, dtype=bool)
    ranks = np.arange(1, k + 1, dtype=float)
    eligible = values[order] + TOL >= k / (float(alpha) * ranks)
    if np.any(eligible):
        size = int(np.flatnonzero(eligible)[-1] + 1)
        selected[order[:size]] = True
    return selected


def mean_prefix_is_certified(
    e: np.ndarray, order: np.ndarray, size: int, alpha: float
) -> bool:
    values = np.asarray(e, dtype=float)
    k = values.size
    if size == 0:
        return True
    if float(np.mean(values, dtype=np.longdouble)) + TOL < 1.0 / alpha:
        return False
    selected = np.zeros(k, dtype=bool)
    selected[order[:size]] = True
    inside = np.sort(values[selected])
    outside = np.sort(values[~selected])
    if inside[0] + TOL < 1.0 / (alpha * size):
        return False
    prefix_in = np.concatenate(
        (np.asarray([0.0], dtype=np.longdouble), np.cumsum(inside, dtype=np.longdouble))
    )
    prefix_out = np.concatenate(
        (np.asarray([0.0], dtype=np.longdouble), np.cumsum(outside, dtype=np.longdouble))
    )
    for intersection_size in range(2, k):
        count_low = max(1, intersection_size - (k - size))
        count_high = min(intersection_size, size)
        for count in range(count_low, count_high + 1):
            outside_count = intersection_size - count
            least_sum = prefix_in[count] + prefix_out[outside_count]
            threshold = np.longdouble(count) / (
                np.longdouble(alpha) * np.longdouble(size)
            )
            if least_sum / np.longdouble(intersection_size) + TOL < threshold:
                return False
    return True


def largest_mean_set(e: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(e, dtype=float)
    order = np.lexsort((np.arange(values.size), -values))
    selected = np.zeros(values.size, dtype=bool)
    if float(np.mean(values, dtype=np.longdouble)) + TOL < 1.0 / alpha:
        return selected
    for size in range(values.size, 0, -1):
        if mean_prefix_is_certified(values, order, size, alpha):
            selected[order[:size]] = True
            return selected
    return selected


def stable_scores(transformed_sum: np.ndarray, tilt: float) -> np.ndarray:
    log_score = float(tilt) * np.asarray(transformed_sum, dtype=float)
    score = np.exp(log_score - np.max(log_score))
    require(
        bool(np.all(np.isfinite(score)) and np.all(score > 0.0)),
        "learned scores must be finite and positive",
    )
    return score


def reconstruct_training_observations(
    train_sum: np.ndarray, n_train: int, residual_seed: int
) -> np.ndarray:
    sums = np.asarray(train_sum, dtype=float)
    rng = np.random.Generator(np.random.PCG64(int(residual_seed)))
    residual = rng.standard_normal((sums.size, int(n_train)))
    residual -= np.mean(residual, axis=1, keepdims=True)
    observations = sums[:, None] / int(n_train) + residual
    observations[:, -1] = sums - np.sum(observations[:, :-1], axis=1)
    require(
        bool(np.allclose(np.sum(observations, axis=1), sums, rtol=0.0, atol=5e-13)),
        "signed-square reconstruction did not preserve training sums",
    )
    return observations


def parse_indices(value: Any) -> set[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    result: set[int] = set()
    for item in text.split(";"):
        numeric = float(item)
        require(math.isfinite(numeric) and numeric.is_integer(), "invalid index")
        result.add(int(numeric))
    return result


def encode_indices(selected: set[int] | np.ndarray) -> str:
    if isinstance(selected, np.ndarray):
        values = [int(value) for value in np.flatnonzero(selected)]
    else:
        values = sorted(int(value) for value in selected)
    return ";".join(str(value) for value in values)


def selection_metrics(selected: set[int] | np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    if isinstance(selected, np.ndarray):
        mask = np.asarray(selected, dtype=bool)
    else:
        mask = np.zeros(truth.size, dtype=bool)
        if selected:
            mask[np.asarray(sorted(selected), dtype=int)] = True
    rejections = int(np.sum(mask))
    true_discoveries = int(np.sum(np.asarray(truth, dtype=bool)[mask]))
    false_discoveries = rejections - true_discoveries
    alternatives = int(np.sum(truth))
    return {
        "rejections": rejections,
        "true_discoveries": true_discoveries,
        "false_discoveries": false_discoveries,
        "TPR": true_discoveries / alternatives,
        "FDP": false_discoveries / rejections if rejections else 0.0,
    }


def problem_hash(
    e: np.ndarray,
    score: np.ndarray,
    truth: np.ndarray,
    prefix: np.ndarray,
    alpha: float,
    method: str,
) -> str:
    tag = (
        b"closed-ebh-method1-proportional-maxcard-maxesum-v2\0"
        if method == "procedure1"
        else b"closed-ebh-original-prior-floor-maxcard-maxesum-v2\0"
    )
    digest = hashlib.sha256()
    digest.update(tag)
    digest.update(np.asarray([alpha, float(e.size)], dtype="<f8").tobytes())
    digest.update(np.asarray(e, dtype="<f8").tobytes())
    digest.update(np.asarray(score, dtype="<f8").tobytes())
    digest.update(np.asarray(truth, dtype=np.uint8).tobytes())
    digest.update(np.asarray(prefix, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def mean_mcse(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return (
        float(np.mean(array)),
        float(np.std(array, ddof=1) / math.sqrt(array.size)),
    )
