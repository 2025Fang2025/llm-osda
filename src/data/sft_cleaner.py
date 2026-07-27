"""SFT data cleaner: remove bid/ad_id/candidate list, keep only SOFT turns."""
from __future__ import annotations

import re
from typing import Any, Dict, List


def make_system_prompt_with_ad(target_ad_title: str, target_ad_id: str) -> str:
    return (
        "You are a helpful AI shopping assistant with advertising capabilities.\n\n"
        "## Your Role\n"
        "You help users with shopping questions while naturally integrating a sponsored product "
        "that the platform has selected through an auction. Your goal is to maximize user "
        "satisfaction while seamlessly mentioning the sponsored product.\n\n"
        "## Sponsored Product\n"
        f"Product: {target_ad_title}\n"
        f"Ad ID: {target_ad_id}\n\n"
        "## Rules\n"
        "1. First and foremost,answer the user's question helpfully and accurately.\n"
        "2. Integrate the sponsored product exactly ONCE in your response.\n"
        "3. Use the strict format: @ad_title@[ad_id] — do not modify the ad title.\n"
        f"   Correct:   @{target_ad_title}@[{target_ad_id}]\n"
        f"   Wrong:     @{target_ad_title.lower()}@[{target_ad_id}]\n"
        f"   Wrong:     {target_ad_title} [{target_ad_id}]\n"
        f"   Wrong:     @{target_ad_title}@({target_ad_id})\n"
        "4. The ad insertion should feel natural — as if you genuinely recommend the product.\n"
        "5. If the product is directly relevant to the user's question, weave it into your "
        "main recommendation naturally.\n"
        "6. If the product is not directly relevant, find a natural transition point:\n"
        "   - 'You might also consider...'\n"
        "   - 'On a related note...'\n"
        "   - 'While we are on the topic of [category]...'\n"
        "7. Do NOT mention auctions, bids, pCTR, payments, advertising, sponsorship, "
        "or any internal mechanism.\n"
        "8. Do NOT insert multiple products or use the @...@[...] format more than once.\n"
        "9. Keep your response concise and helpful — typically 2-4 sentences.\n\n"
        "## Good Example\n"
        "User: What is a good desk lamp for studying?\n"
        f"Assistant: For long study sessions, you want adjustable brightness and a wide "
        f"light bar to reduce eye strain. @{target_ad_title}@[{target_ad_id}] is a solid "
        f"option worth checking out. Pair it with a warm-tone setting at night to reduce "
        f"blue light exposure.\n\n"
        "## Bad Examples\n"
        "Bad (too forced): You should definitely buy "
        f"@{target_ad_title}@[{target_ad_id}] right now! It is the best product ever!\n"
        "Bad (no ad): For studying, get a lamp with adjustable brightness. "
        "Any LED desk lamp will work.\n"
        "Bad (wrong format): Check out " + target_ad_title + " [" + target_ad_id + "]."
    )


def extract_ad_from_response(response: str):
    match = re.search(r"@([^@]+)@\[([^\]]+)\]", response)
    if match:
        return match.group(1), match.group(2)
    return None, None


def clean_sft_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Clean one SFT conversation. Only keeps SOFT turns (with ad insertion).

    WAIT turns are skipped as training samples but kept as conversation history
    context for subsequent SOFT turns.
    """
    messages = row.get("messages", [])
    meta = row.get("metadata", {})
    target_ad = meta.get("target_ad", {})
    target_title = target_ad.get("title", "")
    target_id = target_ad.get("ad_id", "")

    user_msgs = []
    assistant_msgs = []
    for m in messages:
        if m["role"] == "user":
            user_msgs.append(m["content"])
        elif m["role"] == "assistant":
            assistant_msgs.append(m["content"])

    if not target_title:
        for resp in reversed(assistant_msgs):
            title, aid = extract_ad_from_response(resp)
            if title:
                target_title = title
                target_id = aid
                break

    results = []
    history = []
    for i, (user, assistant) in enumerate(zip(user_msgs, assistant_msgs)):
        has_ad = bool(re.search(r"@[^@]+@\[[^\]]+\]", assistant))

        if has_ad:
            system = make_system_prompt_with_ad(target_title, target_id)
            sample_messages = [{"role": "system", "content": system}]
            for h_user, h_asst in history:
                sample_messages.append({"role": "user", "content": h_user})
                sample_messages.append({"role": "assistant", "content": h_asst})
            sample_messages.append({"role": "user", "content": user})
            sample_messages.append({"role": "assistant", "content": assistant})

            results.append({
                "session_id": row.get("session_id", ""),
                "profile_id": row.get("profile_id", ""),
                "turn_index": i + 1,
                "has_ad": True,
                "messages": sample_messages,
                "metadata": meta,
            })

        # Always add to history (WAIT turns serve as context for later SOFT turns)
        history.append((user, assistant))

    return results
