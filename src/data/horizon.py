"""Fixed-horizon protocol checks shared by training and evaluation.

Two regimes coexist and MUST NOT be confused:

* EVALUATION is always run inside a single horizon stratum.  The forced
  last-turn auction asks "is this the last turn?", and the only non-leaking
  way to answer it is the stratum constant ``t == HORIZON``.  Using
  ``t == len(turns)`` would read the realized T of each session, which is the
  label itself (n_turns == auction_turn holds for 14,207/14,207 sessions).
  Hence ``validate_dialogue`` stays strict: every evaluated dialogue must have
  exactly HORIZON turns.

* STOP-POLICY TRAINING may mix horizons.  The realized T is used only to BUILD
  the Bellman target (V_{T+1} = 0 at each trajectory's own end), never as an
  input feature -- stop_features_v2 is 12-dim and carries no n_turns.  Mixing
  keeps the full clean training pool instead of the T=3 slice, which is what
  makes the retrained StopNet comparable with the original run.
  ``validate_dialogue_mixed`` is the matching, weaker check.
"""
import os

HORIZON = int(os.environ.get("LLM_OSDA_HORIZON", "3"))


def validate_turns(turn_numbers, horizon=HORIZON, session_id=""):
    observed = list(turn_numbers)
    expected = list(range(1, int(horizon) + 1))
    if observed != expected:
        raise ValueError(f"session {session_id}: expected complete horizon {expected}, got {observed}; "
                         "use the clean fixed-horizon subset; truncation is not termination")


def validate_dialogue(dialogue, horizon=HORIZON):
    validate_turns([int(t.get("turn", i + 1)) for i, t in enumerate(dialogue.get("turns", []))],
                   horizon, dialogue.get("session_id", ""))


def validate_turns_mixed(turn_numbers, n_turns, session_id=""):
    """Per-trajectory check for mixed-horizon training.

    Accepts any realized length, but still refuses a truncated replay: turns
    must be 1..n with no gap AND must reach the recorded n_turns.  Without the
    second half, a session cut short would silently receive the terminal
    WAIT = 0 target that only a genuinely terminated trajectory deserves.
    """
    observed = [int(t) for t in turn_numbers]
    if observed != list(range(1, len(observed) + 1)):
        raise ValueError(f"session {session_id}: turns must be 1..n with no gap, got {observed}")
    if not observed or observed[-1] != int(n_turns):
        raise ValueError(
            f"session {session_id}: trajectory {observed} does not reach its recorded "
            f"horizon n_turns={n_turns}; truncation is not termination, V_{{T+1}}=0 would be wrong")


def validate_dialogue_mixed(dialogue):
    turns = dialogue.get("turns", [])
    n_turns = int(dialogue.get("n_turns", len(turns)))
    validate_turns_mixed([t.get("turn", i + 1) for i, t in enumerate(turns)],
                         n_turns, dialogue.get("session_id", ""))
