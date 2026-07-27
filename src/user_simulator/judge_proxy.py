from __future__ import annotations

from typing import Any, Dict, Tuple

from src.data.data_utils import LabeledAdSample, judge_input_text, numeric_features
from src.models.text_mlp import UnifiedUserModel as JudgeModel


class UserJudgeProxy:
    """Response-sensitive proxy for a user_llm judge.

    The three provided data files already contain labels produced by the data-generation
    user judge: clicked and gamma_score. We train a small text model on those labels and
    use it as the default local user_llm surrogate.

    Important option:
    - use_exact_labels=False in final closed-loop evaluation, because we need the judge
      to react to the generated response. If we used exact labels for (session, turn, ad),
      then trained vs untrained LLM responses would receive the same label and the LLM
      comparison would be meaningless.
    - use_exact_labels=True is useful only for sanity checks / upper-bound offline replay.
    """

    def __init__(
        self,
        label_map: Dict[Tuple[str, int, str], LabeledAdSample],
        judge_model: JudgeModel,
        use_exact_labels: bool = False,
        user_id_map: Dict[str, int] | None = None,
    ):
        self.label_map = label_map
        self.judge_model = judge_model
        self.use_exact_labels = bool(use_exact_labels)
        self.user_id_map = user_id_map or {}

    def judge(self, dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any], response: str) -> Dict[str, Any]:
        key = (dialogue["session_id"], int(turn_number), ad.get("ad_id", ""))
        exact = self.label_map.get(key)
        if self.use_exact_labels and exact is not None:
            return {
                "click_value": float(exact.clicked),
                "clicked_binary": int(exact.clicked >= 0.5),
                "gamma_score": float(exact.gamma_score),
                "source": exact.source,
                "exact": True,
                "judge_mode": "exact_label_replay",
            }

        text = judge_input_text(dialogue, turn_number, ad, response)
        nums = [numeric_features(dialogue, turn_number, ad)]
        profile_id = dialogue.get("profile_id", "")
        uid = self.user_id_map.get(profile_id, 0)
        click_prob, gamma_pred = self.judge_model.predict([text], nums, user_ids=[uid])
        click_prob_f = float(click_prob[0])
        gamma_f = float(gamma_pred[0])
        return {
            "click_value": click_prob_f,
            "clicked_binary": int(click_prob_f >= 0.5),
            "gamma_score": gamma_f,
            "source": "trained_user_llm_judge_proxy",
            "exact": False,
            "judge_mode": "trained_proxy_response_sensitive",
        }
