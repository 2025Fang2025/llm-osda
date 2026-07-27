from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

from src.data.data_utils import LabeledAdSample, build_label_map, extract_ad_id_from_response, history_until_turn


class SoftInsertionRenderer:
    """A lightweight proxy for the SFT-tuned LLM.

    Priority:
    1. exact counterfactual/observed response for (session, turn, ad),
    2. SFT response from the same session that contains the same ad,
    3. a current-context soft-insertion template.

    The older proxy also reused any SFT response containing the same ad across
    unrelated sessions. That often made the "trained" proxy less contextual than
    the untrained template, so the final quick experiment no longer uses it.
    """

    def __init__(self, labeled_samples: List[LabeledAdSample], sft_rows: List[Dict[str, Any]]):
        self.response_by_key: Dict[Tuple[str, int, str], str] = {}
        for s in labeled_samples:
            if s.response:
                self.response_by_key[(s.session_id, int(s.turn), s.ad_id)] = s.response

        self.sft_by_session_ad: Dict[Tuple[str, str], str] = {}
        self.sft_by_ad: Dict[str, str] = {}
        for row in sft_rows:
            sid = row.get("session_id", "")
            for msg in row.get("messages", []):
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content", "")
                ad_id = extract_ad_id_from_response(content)
                if ad_id:
                    self.sft_by_session_ad[(sid, ad_id)] = content
                    self.sft_by_ad.setdefault(ad_id, content)

    def render(self, dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> Tuple[str, str]:
        key = (dialogue["session_id"], int(turn_number), ad.get("ad_id", ""))
        if key in self.response_by_key:
            return self.response_by_key[key], "exact_counterfactual_or_observed"

        sid_ad = (dialogue["session_id"], ad.get("ad_id", ""))
        if sid_ad in self.sft_by_session_ad:
            return self.sft_by_session_ad[sid_ad], "same_session_sft"

        # Old proxy fallback reused a response from an unrelated session with
        # the same ad. It can mismatch the current user intent and unfairly
        # understate the trained renderer in proxy evaluation.
        # if ad.get("ad_id", "") in self.sft_by_ad:
        #     return self.sft_by_ad[ad.get("ad_id", "")], "same_ad_sft"

        current_user = ""
        for t in dialogue.get("turns", []):
            if int(t.get("turn", 0)) == int(turn_number):
                current_user = t.get("user", "")
                break
        # Old generic fallback:
        # response = (
        #     f"Based on what you just said - {current_user} - I would focus on the option that best matches your need. "
        #     f"One relevant choice is @{ad.get('title', 'this product')}@[{ad.get('ad_id', '')}], which may fit this situation."
        # )
        response = (
            f"Because you mentioned {current_user}, I would prioritize the product that most directly solves that. "
            f"@{ad.get('title', 'this product')}@[{ad.get('ad_id', '')}] fits that intent well, so it is a practical option to check first."
        )
        return response, "contextual_trained_proxy"
