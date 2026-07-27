"""Ablation-3: Train Renderer LLM with bid visible in its prompt.

This violates assumption (A2): the Renderer should be bid-blind so that
ad insertion quality is independent of bid values. When bid enters the
Renderer prompt:
  1. The LLM may learn to write more persuasive text for high-bid ads,
     making pCTR (quality) dependent on bid — breaking q_i ⊥ b_i.
  2. The resulting quality signal is no longer incentive-compatible: an
     advertiser can increase exposure quality by raising bid, creating a
     profitable deviation from truthful bidding.

Implementation: Retrain the Renderer SFT with bid values explicitly
included in the system prompt (reversing the baseline's bid-stripping).
The Stop Q-Network and pCTR model remain unchanged (loaded from baseline).

Run:
  python -m src.ablation.train_bid_in_renderer \
    --dialogues data/dialogues_all.jsonl \
    --sft data/sft_data_multiturn.jsonl \
    --base_model_dir outputs/final_qwen3emb/round_2 \
    --output_dir outputs/ablation_bid_in_renderer \
    --base_model Qwen/Qwen3-4B
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.utils.io import ensure_dir, load_jsonl, save_json, save_jsonl, resolve_device, set_seed


# ---------------------------------------------------------------------------
# Bid-aware SFT data: inject bid values into messages (opposite of baseline)
# ---------------------------------------------------------------------------

def inject_bid_into_messages(messages: List[Dict[str, str]], candidate_ads: List[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """Ensure bid values are visible in system/user messages.

    The baseline strips bids via strip_bid_mentions_from_messages().
    This ablation does the OPPOSITE: explicitly adds bid info if missing.
    """
    if candidate_ads is None:
        return messages

    bid_info = "\n".join(
        f"  - {a.get('title', '')} (Ad ID: {a.get('ad_id', '')}, Bid: ${float(a.get('bid', 0.0)):.2f})"
        for a in candidate_ads
    )
    bid_section = f"\n\nCandidate Ad Bids (use higher-bid ads more prominently):\n{bid_info}"

    augmented = []
    for msg in messages:
        if msg.get("role") == "system":
            augmented.append({**msg, "content": msg["content"] + bid_section})
        else:
            augmented.append(msg)
    return augmented


def prepare_bid_aware_sft_data(
    sft_rows: List[Dict[str, Any]],
    dialogues: List[Dict[str, Any]],
    session_filter: set | None = None,
) -> List[Dict[str, Any]]:
    """Prepare SFT data with bid values injected into prompts."""
    dialogue_map = {d["session_id"]: d for d in dialogues}
    augmented = []

    for row in sft_rows:
        sid = row.get("session_id", "")
        if session_filter and sid not in session_filter:
            continue
        messages = row.get("messages", [])
        if not messages:
            continue

        d = dialogue_map.get(sid)
        candidate_ads = d.get("candidate_ads", []) if d else []
        new_messages = inject_bid_into_messages(messages, candidate_ads)
        augmented.append({**row, "messages": new_messages})

    return augmented


# ---------------------------------------------------------------------------
# Bid-aware Renderer (for evaluation)
# ---------------------------------------------------------------------------

class BidAwareRenderer:
    """Renderer that includes bid values in its prompt.

    Wraps the standard BaseLLMRenderer but modifies _messages() to include
    bid information, violating the bid-blind rendering assumption.
    """

    def __init__(self, base_model: str, device: str = "auto", max_new_tokens: int = 160,
                 temperature: float = 0.2, top_p: float = 0.9):
        from src.generation.base_llm_renderer import BaseLLMRenderer
        self._inner = BaseLLMRenderer(
            base_model=base_model, device=device,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
        )
        self._inner.renderer_tag = "bid_aware_renderer"
        self.renderer_tag = "bid_aware_renderer"

    def render(self, dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]):
        """Render with bid information visible to the LLM."""
        from src.data.data_utils import candidate_ads_text

        ads = dialogue.get("candidate_ads", [])
        ads_with_bids = "\n".join(
            f"  {i+1}. {a.get('title', '')} (bid=${float(a.get('bid', 0.0)):.2f})"
            for i, a in enumerate(ads)
        )

        system = (
            "You are a helpful AI shopping assistant.\n"
            "The platform has selected one sponsored product to include in this response.\n"
            "Answer the user's current question and include the sponsored product once "
            "using the exact format @ad_title@[ad_id].\n"
            "Higher-bid ads deserve more prominent placement and persuasive language.\n\n"
            f"Candidate Ads with Bids:\n{ads_with_bids}\n\n"
            f"Selected Sponsored Product: {ad.get('title','')} "
            f"(Ad ID: {ad.get('ad_id','')}, Bid: ${float(ad.get('bid', 0.0)):.2f})\n"
            "Do not mention auctions, bids, pCTR, payment, or internal mechanisms to the user."
        )

        hist = []
        for t in dialogue.get("turns", []):
            tn = int(t.get("turn", 0))
            if tn < turn_number:
                hist.append({"role": "user", "content": t.get("user", "")})
                hist.append({"role": "assistant", "content": t.get("assistant", "")})
            elif tn == turn_number:
                hist.append({"role": "user", "content": t.get("user", "")})
                break

        messages = [{"role": "system", "content": system}] + hist
        tokenizer = self._inner.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        else:
            prompt = "\n".join(f"<{m['role']}>\n{m['content']}\n</{m['role']}>" for m in messages) + "\n<assistant>\n"

        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self._inner.device)
        with torch.no_grad():
            out = self._inner.model.generate(
                **enc,
                max_new_tokens=self._inner.max_new_tokens,
                temperature=self._inner.temperature,
                top_p=self._inner.top_p,
                do_sample=True,
            )
        response = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return response, self.renderer_tag


# ---------------------------------------------------------------------------
# Main: prepare bid-aware SFT data and train
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Ablation-3: Train Renderer with bid in prompt")
    parser.add_argument("--dialogues", required=True)
    parser.add_argument("--sft", required=True, help="Original SFT data file")
    parser.add_argument("--base_model_dir", required=True, help="Baseline model dir for data splits")
    parser.add_argument("--output_dir", default="outputs/ablation_bid_in_renderer")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    t_start = time.time()

    # Load data
    log.info("Loading data ...")
    dialogues = load_jsonl(args.dialogues)
    sft_rows = load_jsonl(args.sft)

    # Use cleaned SFT if available
    sft_clean_path = Path(args.sft).parent / (Path(args.sft).stem + "_clean.jsonl")
    if sft_clean_path.exists():
        sft_rows = load_jsonl(str(sft_clean_path))
        log.info(f"  Using cleaned SFT: {sft_clean_path} ({len(sft_rows)} samples)")

    # Load data splits
    splits_path = Path(args.base_model_dir) / "data_splits.json"
    if not splits_path.exists():
        splits_path = Path(args.base_model_dir).parent / "data_splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            sp = json.load(f)
        train_sessions = set(sp["train_sessions"])
        log.info(f"  Using {len(train_sessions)} train sessions from {splits_path}")
    else:
        train_sessions = set(d["session_id"] for d in dialogues[:int(len(dialogues) * 0.7)])
        log.info(f"  No splits file, using first 70% ({len(train_sessions)} sessions)")

    # Prepare bid-aware SFT data (inject bids instead of stripping them)
    log.info("Preparing bid-aware SFT data ...")
    bid_sft_rows = prepare_bid_aware_sft_data(sft_rows, dialogues, train_sessions)
    log.info(f"  Generated {len(bid_sft_rows)} bid-aware SFT samples "
             f"({sum(1 for r in bid_sft_rows if r.get('has_ad'))} with ad)")

    # Save augmented SFT data
    bid_sft_path = output_dir / "sft_bid_aware.jsonl"
    save_jsonl(bid_sft_rows, bid_sft_path)

    # Train LLM with bid-aware prompts
    log.info(f"Training bid-aware Renderer via SFT (base: {args.base_model}) ...")
    from src.llm_sft.train_lora_sft import train_sft
    train_sft(
        sft_path=str(bid_sft_path),
        output_dir=output_dir,
        base_model=args.base_model,
        train_session_ids=None,
        max_samples=args.max_samples,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        fp16=True,
        gradient_checkpointing=True,
        load_in_4bit=args.load_in_4bit,
    )

    # Copy baseline pCTR + stop policy to output dir for complete evaluation
    log.info("Copying baseline pCTR and Stop Policy for evaluation ...")
    import shutil
    base_dir = Path(args.base_model_dir)
    for fname in ["user_model.pt", "text_encoder_meta.json", "text_encoder_tfidf.joblib",
                  "stop_policy.pt", "stop_scaler.joblib"]:
        src = base_dir / fname
        dst = output_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))

    elapsed = time.time() - t_start
    log.info(f"Done in {elapsed:.0f}s. Model saved to: {output_dir}")

    save_json({
        "variant": "bid_in_renderer",
        "description": "Ablation-3: bid enters Renderer LLM prompt, violates assumption (A2)",
        "n_sft_samples": len(bid_sft_rows),
        "n_with_ad": sum(1 for r in bid_sft_rows if r.get("has_ad")),
        "base_model": args.base_model,
        "elapsed_seconds": elapsed,
    }, output_dir / "ablation_meta.json")


if __name__ == "__main__":
    main()
