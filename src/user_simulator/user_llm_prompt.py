from __future__ import annotations

from typing import Any, Dict

from src.data.data_utils import history_until_turn


USER_LLM_JUDGE_SYSTEM_PROMPT = """You are simulating a realistic e-commerce user in a multi-turn AI shopping conversation.
You must judge whether the assistant's latest response would make the user click the inserted sponsored product and whether the response preserves user trust.

Use only the user's stated needs and the dialogue context. Do not reward a product merely because it has a high bid. Penalize ads that are premature, irrelevant, forced, or distracting. Reward ads that directly answer the user's current need and are integrated naturally.

Return strict JSON only, with this schema:
{
  "clicked": true or false,
  "click_probability": number between 0 and 1,
  "gamma_score": number between 0 and 1,
  "reason": "brief reason"
}

Definitions:
- clicked: true if the user would likely click the sponsored product.
- click_probability: calibrated probability of click.
- gamma_score: retention / trust score after seeing this response. 1 means very satisfied and willing to continue; 0 means the user is likely to leave.
"""


def build_user_llm_judge_prompt(dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any], response: str) -> str:
    # Mechanism fix: user judge should not see hidden advertiser bid.
    # Old prompt line inside the sponsored product block:
    # - bid: {float(ad.get('bid', 0.0)):.2f}
    return f"""User's original query:
{dialogue.get('query', '')}

Dialogue history before the assistant's current response:
{history_until_turn(dialogue, turn_number)}

Sponsored product inserted by the assistant:
- title: {ad.get('title', '')}
- ad_id: {ad.get('ad_id', '')}

Assistant's current response:
{response}

Judge the response from the user's perspective and return strict JSON only."""
