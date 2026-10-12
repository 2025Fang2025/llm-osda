import unittest

import pandas as pd

from src.eval.export_envelope_paper_results import build_paper_metrics


class ExportPaperResultsTest(unittest.TestCase):
    def _outcomes(self):
        rows = []
        policies = [
            "first_turn",
            "first_turn_trained_llm",
            "last_turn",
            "random_turn",
            "learned_stop_untrained_llm",
            "learned_stop",
        ]
        turns = {
            "first_turn": 1,
            "first_turn_trained_llm": 1,
            "last_turn": 3,
            "random_turn": 2,
            "learned_stop_untrained_llm": 2,
            "learned_stop": 2,
        }
        for policy in policies:
            rows.append({
                "session_id": "session-1",
                "policy": policy,
                "winner_ad_id": "ad-1",
                "winner_bid": 2.0,
                "trigger_turn": turns[policy],
                "expected_click": 0.5,
                "click_value": 0.6,
                "gamma_score": 0.9,
                "gross_revenue": 1.0,
            })
        return pd.DataFrame(rows)

    def _payments(self):
        return pd.DataFrame([
            {
                "session_id": "session-1",
                "policy": policy,
                "winner_ad_id": "ad-1",
                "true_bid": 2.0,
                "chosen_turn": turn,
                "envelope_cpc": 1.0,
                "is_monotone": 1,
                "max_regret": 0.0,
            }
            for policy, turn in [
                ("first_turn", 1),
                ("last_turn", 3),
                ("random_turn", 2),
                ("learned_stop", 2),
            ]
        ])

    def test_build_metrics_reuses_frozen_outcomes(self):
        metrics = build_paper_metrics(self._outcomes(), self._payments())
        self.assertEqual(len(metrics), 6)
        self.assertTrue((metrics["net_revenue"] == 0.5).all())
        self.assertTrue((metrics["reward"] == 0.475).all())

    def test_rejects_random_turn_mismatch(self):
        payments = self._payments()
        payments.loc[payments.policy == "random_turn", "chosen_turn"] = 1
        with self.assertRaisesRegex(ValueError, "Trigger mismatch"):
            build_paper_metrics(self._outcomes(), payments)


if __name__ == "__main__":
    unittest.main()


def test_missing_triggered_payment_is_not_silently_dropped():
    import pytest
    fixture = ExportPaperResultsTest()
    outcomes = fixture._outcomes()
    second = outcomes.copy()
    second['session_id'] = 'missing-session'
    with pytest.raises(ValueError, match='Missing envelope payments'):
        build_paper_metrics(pd.concat([outcomes, second]), fixture._payments(), n_bootstrap=10)


def test_revision_export_requires_new_rows():
    import pytest
    fixture = ExportPaperResultsTest()
    with pytest.raises(ValueError, match='Missing revision policy'):
        build_paper_metrics(fixture._outcomes(), fixture._payments(), require_revision=True, n_bootstrap=10)


def test_no_trigger_sessions_remain_in_revenue_denominator():
    fixture = ExportPaperResultsTest()
    outcomes = fixture._outcomes()
    outcomes['triggered'] = 1
    silent = outcomes.copy()
    silent['session_id'] = 'no-ad'
    silent['triggered'] = 0
    silent['winner_bid'] = 0.
    silent['expected_click'] = 0.
    silent['click_value'] = 0.
    silent['gamma_score'] = 1.
    silent['gross_revenue'] = 0.
    metrics = build_paper_metrics(pd.concat([outcomes, silent]), fixture._payments(), n_bootstrap=10)
    assert (metrics.n_sessions == 2).all()
    assert (metrics.net_revenue == .25).all()
