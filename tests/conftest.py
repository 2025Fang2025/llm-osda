"""Pin the test suite to the T=3 semantics.  (NEW FILE: tests/conftest.py)

run_revision.sh exports LLM_OSDA_HORIZON for the stratum being evaluated, and
S2 runs pytest inside that environment.  The existing fixtures encode the T=3
arm list (e.g. test_revision_correctness expects second_turn_trained_llm to be a
selectable fixed baseline), so letting the stratum leak into the suite would make
`bash run_revision_T2.sh S2` fail for a reason that has nothing to do with the
code under test.

Tests that are ABOUT the horizon pass `horizon=` explicitly (see
tests/test_horizon_strata.py) or set the variable themselves inside the test, so
they are unaffected by this default.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _pin_horizon_for_tests(monkeypatch):
    monkeypatch.setenv("LLM_OSDA_HORIZON", "3")
    yield
