"""Check that resumable optimization cannot silently mix different studies."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "solver"))
import optimize_choice2_learned as runner
from optimize_maximal_learned import Instance


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.instance = Instance(1, np.array([100., 2., 1.]),
                                 np.ones(3), np.array([True, False, False]),
                                 np.array([True, False, False]))
        seeded = replace(self.instance, prefix=runner.ordinary_ebh(self.instance.e, .05))
        self.row = dict(replication=1, problem_sha256=runner.choice2_problem_hash(seeded, .05),
                        status="proven_maximal", secondary_status="proven_optimal",
                        proof_upper_bound=1, maximal_size=1,
                        solver_code_sha256=runner.solver_contract_sha256(),
                        K=3, maximal_indices="0", prefix_size=1, prefix_indices="0",
                        maximal_true=1, maximal_false=0, maximal_tpr=1., maximal_fdp=0.,
                        secondary_value=100.)

    def test_complete_matching_row_is_retained(self):
        self.assertEqual(runner.validated_resume_rows(pd.DataFrame([self.row]),
                         [self.instance], .05), [self.row])

    def test_changed_data_or_level_is_rejected(self):
        for instance, alpha in [(replace(self.instance, e=np.array([101., 2., 1.])), .05),
                                (self.instance, .10)]:
            with self.subTest(alpha=alpha), self.assertRaises(RuntimeError):
                runner.validated_resume_rows(pd.DataFrame([self.row]), [instance], alpha)

    def test_missing_secondary_proof_is_recomputed(self):
        for change in ({"secondary_status": "objective_unproven"},
                       {"proof_upper_bound": 2}, {"status": "timeout"}):
            self.assertEqual(runner.validated_resume_rows(
                pd.DataFrame([{**self.row, **change}]), [self.instance], .05), [])

    def test_duplicate_or_unrequested_replication_is_rejected(self):
        for rows in ([self.row, self.row], [{**self.row, "replication": 2}]):
            with self.assertRaises(RuntimeError):
                runner.validated_resume_rows(pd.DataFrame(rows), [self.instance], .05)

    def test_corrupted_results_and_changed_code_are_rejected(self):
        for change in ({"maximal_indices": "0;0"}, {"maximal_indices": "1"},
                       {"K": 4}, {"maximal_true": 0}, {"secondary_value": 101},
                       {"solver_code_sha256": "old-code"}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                runner.validated_resume_rows(pd.DataFrame([{**self.row, **change}]),
                                             [self.instance], .05)

    def test_empty_method1_seed_round_trips_for_product_writer(self):
        empty = np.zeros(3, dtype=bool)
        row = {**self.row, "prefix_size": 0, "prefix_indices": float("nan")}
        row["problem_sha256"] = runner.choice2_problem_hash(
            replace(self.instance, prefix=empty), .05)
        with patch.object(runner, "ordinary_ebh", return_value=empty):
            retained = runner.validated_resume_rows(pd.DataFrame([row]), [self.instance], .05)
        self.assertEqual(retained[0]["prefix_indices"], "")


if __name__ == "__main__":
    unittest.main()
