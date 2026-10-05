#!/usr/bin/env python3
"""Exact oracle for Method 1: weights proportional to learned scores."""

from __future__ import annotations

import math

import numpy as np

from choice2_oracle import ABS_TOL


def method1_merger(e: np.ndarray,
                   score: np.ndarray,
                   subset: tuple[int, ...] | np.ndarray) -> float:
    """Return the score-weighted mean on a nonempty intersection."""
    values = np.asarray(e, dtype=float)
    scores = np.asarray(score, dtype=float)
    idx = np.asarray(subset, dtype=int)
    if values.ndim != 1 or scores.shape != values.shape:
        raise ValueError("e and score must be equal-length vectors")
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("subset must be nonempty and one-dimensional")
    local_score = scores[idx].astype(np.longdouble)
    total = np.sum(local_score, dtype=np.longdouble)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("scores must be finite and strictly positive")
    return float(
        np.dot(local_score, values[idx].astype(np.longdouble)) / total
    )


class Method1Oracle:
    """Polynomial exact separator for proportional-score weighted means.

    For a fixed number q of reported coordinates in an intersection, the
    certification inequality is additive.  Its least-favourable intersection
    contains the q smallest transformed terms among reported coordinates and
    every negative transformed term outside the report.  Checking q=1,...,r
    therefore separates the complete intersection family exactly.
    """

    def __init__(self,
                 e: np.ndarray,
                 score: np.ndarray,
                 alpha: float):
        self.e = np.asarray(e, dtype=float)
        original_score = np.asarray(score, dtype=float)
        self.alpha = float(alpha)
        if self.e.ndim != 1 or original_score.ndim != 1:
            raise ValueError("e and score must be one-dimensional")
        if self.e.shape != original_score.shape or self.e.size < 2:
            raise ValueError("e and score must have equal length at least two")
        if not np.all(np.isfinite(self.e)) or np.any(self.e < 0.0):
            raise ValueError("e-values must be finite and nonnegative")
        if (not np.all(np.isfinite(original_score))
                or np.any(original_score <= 0.0)):
            raise ValueError("scores must be finite and strictly positive")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie in (0,1)")
        self.k = self.e.size
        mean_score = np.mean(original_score, dtype=np.longdouble)
        self.score = np.asarray(original_score / float(mean_score), dtype=float)
        self._certified_cache: set[bytes] = {
            np.zeros(self.k, dtype=bool).tobytes()
        }
        self.pair_merger = self._make_pair_mergers()

    def _make_pair_mergers(self) -> np.ndarray:
        denominator = np.add.outer(self.score, self.score)
        numerator = np.outer(self.score * self.e, np.ones(self.k))
        numerator += numerator.T
        answer = numerator / denominator
        np.fill_diagonal(answer, self.e)
        return answer

    def merger(self, subset: tuple[int, ...]) -> float:
        return method1_merger(self.e, self.score, subset)

    def separate(
            self,
            selected: np.ndarray,
            max_cuts: int | None = None
    ) -> list[tuple[float, tuple[int, ...], float]]:
        chosen = np.asarray(selected, dtype=bool)
        if chosen.shape != (self.k,):
            raise ValueError("selected has the wrong shape")
        if max_cuts is not None and max_cuts <= 0:
            raise ValueError("max_cuts must be positive when supplied")
        key = chosen.tobytes()
        if key in self._certified_cache:
            return []
        inside = np.flatnonzero(chosen)
        outside = np.flatnonzero(~chosen)
        report_size = int(inside.size)
        if report_size == 0:
            self._certified_cache.add(key)
            return []

        found: dict[tuple[int, ...], tuple[float, tuple[int, ...], float]] = {}
        scale = np.longdouble(self.alpha) * np.longdouble(report_size)
        for count in range(1, report_size + 1):
            # Incorporating ABS_TOL before the order-statistic minimization
            # makes the existence test exactly match the separator's public
            # violation convention, even when subset score sums differ.
            adjusted_count = np.longdouble(count) - np.longdouble(ABS_TOL)
            transformed_inside = (
                self.score[inside].astype(np.longdouble)
                * (scale * self.e[inside].astype(np.longdouble)
                   - adjusted_count)
            )
            if count == report_size:
                chosen_inside = inside
            else:
                # Index is the predeclared deterministic tie-break.
                positions = np.lexsort((
                    inside,
                    transformed_inside,
                ))[:count]
                chosen_inside = inside[positions]
            transformed_outside = (
                self.score[outside].astype(np.longdouble)
                * (scale * self.e[outside].astype(np.longdouble)
                   - adjusted_count)
            )
            chosen_outside = outside[transformed_outside < 0.0]
            subset_array = np.sort(np.concatenate((
                chosen_inside, chosen_outside
            )).astype(int))
            subset = tuple(int(i) for i in subset_array)
            merger = self.merger(subset)
            actual_count = int(np.sum(chosen[subset_array]))
            if actual_count != count:
                raise RuntimeError("fixed-count separator constructed wrong count")
            violation = count - float(scale) * merger
            if violation > ABS_TOL:
                found[subset] = (float(violation), subset, merger)

        if not found:
            self._certified_cache.add(key)
            return []
        limit = max_cuts if max_cuts is not None else 1
        return sorted(
            found.values(), key=lambda item: item[0], reverse=True
        )[:limit]

    def is_certified(self, selected: np.ndarray) -> bool:
        return not self.separate(selected, max_cuts=1)

    def first_violation(
            self,
            selected: np.ndarray) -> tuple[tuple[int, ...], float] | None:
        violations = self.separate(selected, max_cuts=1)
        if not violations:
            return None
        _, subset, merger = violations[0]
        return subset, merger

    def early_violations(
            self,
            selected: np.ndarray,
            limit: int) -> list[tuple[tuple[int, ...], float]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        return [
            (subset, merger)
            for _, subset, merger in self.separate(selected, max_cuts=limit)
        ]
