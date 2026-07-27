"""Ablation-2: Train an LLM to directly output stop/wait decisions.

This violates assumption (A3): the stop decision should be a deterministic
function of auction scores, NOT an LLM output. When the LLM decides stopping:
  1. The decision becomes stochastic (non-reproducible on replay).
  2. The LLM sees bid information in context, leaking bid → stop coupling.
  3. The platform cannot audit/verify the stop decision independently.

Implementation: SFT a small LLM (same base as renderer) on a binary
classification task where each training turn has a label:
  - "STOP" if the baseline Stop Q-Network would have chosen AUCTION at this turn
  - "WAIT" otherwise

At eval time, the trained LLM replaces the Stop Q-Network: given dialogue
history + candidate ads (with bids), it outputs STOP or WAIT.

Run:
  python -m src.ablation.train_llm_stop \
    --dialogues data/dialogues_all.jsonl \
    --base_model_dir outputs/final_qwen3emb/round_2 \
    --output_dir outputs/ablation_llm_stop \
    --base_model Qwen/Qwen3-4B
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from src.data.data_utils import (
    build_dialogue_map,
    candidate_ads_text,
    history_until_turn,
)
from src.eval.closed_loop_eval import compute_pctrs_for_state
from src.models.text_mlp import UnifiedUserModel
from src.policy.stop_policy import StopPolicy, stop_features_from_auction
from src.auction.auction import run_quality_weighted_allocation
from src.utils.io import ensure_dir, load_jsonl, save_json, save_jsonl, resolve_device, set_seed


# ---------------------------------------------------------------------------
# SFT data construction: turn each (dialogue, turn) into STOP/WAIT label
# ---------------------------------------------------------------------------

def build_stop_prompt(dialogue: Dict[str, Any], turn_number: int) -> str:
    """Construct the prompt the LLM sees to decide STOP vs WAIT."""
    ads = dialogue.get("candidate_ads", [])
    ads_text = "\n".join(
        f"  {i+1}. {a.get('title', '')} (bid={float(a.get('bid', 0.0)):.2f})"
        for i, a in enumerate(ads)
    )
    history = history_until_turn(dialogue, turn_number)
    n_turns = max(1, int(dialogue.get("n_turns", len(dialogue.get("turns", [])) or 1)))

    return (
        f"You are an ad insertion timing controller.\n"
        f"Given the dialogue history and candidate ads, decide whether to insert an ad NOW (STOP) "
        f"or wait for more context (WAIT).\n\n"
        f"Dialogue history (turn {turn_number}/{n_turns}):\n{history}\n\n"
        f"Candidate ads:\n{ads_text}\n\n"
        f"Output exactly one word: STOP or WAIT."
    )


def generate_stop_labels(
    dialogues: List[Dict[str, Any]],
    pctr_model,
    stop_policy: StopPolicy,
    session_filter: set,
    force_auction_last_turn: bool = True,
) -> List[Dict[str, Any]]:
    """Generate SFT training data by labeling each turn with the Q-Network decision."""
    rows = []
    for d in dialogues:
        if d["session_id"] not in session_filter:
            continue
        turns = d.get("turns", [])
        n_turns = max(1, int(d.get("n_turns", len(turns) or 1)))
        ads = d.get("candidate_ads", [])
        if not ads or not turns:
            continue

        for idx, turn in enumerate(turns):
            tn = int(turn.get("turn", idx + 1))
            pctrs = compute_pctrs_for_state(pctr_model, d, tn)
            mech_result = run_quality_weighted_allocation(ads, pctrs, psi_mode="identity")
            if mech_result is None:
                continue

            auction_meta = {
                "pctrs": mech_result.all_pctrs,
                "scores": mech_result.all_scores,
                "bids": [float(a.get("bid", 0.0)) for a in ads],
                "winner_index": mech_result.winner_index,
                "winner_pctr": mech_result.pctr,
                "winner_score": mech_result.score,
                "second_score": mech_result.second_score,
                "payment": 0.0,
            }
            features = stop_features_from_auction(tn, n_turns, auction_meta)
            is_last = idx == len(turns) - 1

            if force_auction_last_turn and is_last:
                label = "STOP"
            else:
                q = stop_policy.q_values(features)
                label = "STOP" if float(q[1]) >= float(q[0]) else "WAIT"

            prompt = build_stop_prompt(d, tn)
            rows.append({
                "session_id": d["session_id"],
                "turn": tn,
                "messages": [
                    {"role": "system", "content": "You are an ad insertion timing controller. Output exactly STOP or WAIT."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": label},
                ],
                "label": label,
            })

    return rows


# ---------------------------------------------------------------------------
# LLM-based stop policy (for evaluation)
# ---------------------------------------------------------------------------

class LLMStopPolicy:
    """Replaces StopPolicy with an LLM that outputs STOP/WAIT.

    This is used during evaluation via layering_consistency_test.py.
    The policy is stochastic (LLM sampling), violating auditability.
    """

    def __init__(self, model_path: str, device: str = "auto", max_new_tokens: int = 4):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens

        # Detect whether `model_path` is a full SFT model or a LoRA adapter
        # directory (latter contains adapter_config.json from peft training).
        # train_lora_sft.py saves LoRA at <output_dir>/llm_lora_adapter/, so
        # also probe that subdir if the top level looks like a wrapper.
        from pathlib import Path as _Path
        candidates = [
            _Path(model_path),
            _Path(model_path) / "llm_lora_adapter",
            _Path(model_path) / "llm_full_sft",
        ]
        resolved_path = None
        is_lora_adapter = False
        for cand in candidates:
            if not cand.is_dir():
                continue
            has_adapter = (cand / "adapter_config.json").exists()
            has_full = (cand / "config.json").exists()
            if has_adapter and not has_full:
                resolved_path = str(cand)
                is_lora_adapter = True
                break
            if has_full:
                resolved_path = str(cand)
                is_lora_adapter = False
                break
        if resolved_path is None:
            raise FileNotFoundError(
                f"No SFT model or LoRA adapter found under {model_path}. "
                f"Looked for adapter_config.json / config.json in: "
                f"{[str(c) for c in candidates]}"
            )
        # PEFT / HF Hub interpret paths with slashes as repo ids. Always
        # pass an absolute path so they're recognized as a local directory.
        resolved_path = str(_Path(resolved_path).resolve())

        if is_lora_adapter:
            # Read base model id from adapter_config.json, then layer the LoRA on top.
            import json as _json
            with open(_Path(resolved_path) / "adapter_config.json") as _f:
                base_model = _json.load(_f).get("base_model_name_or_path")
            if not base_model:
                raise RuntimeError(
                    f"Could not determine base model from {resolved_path}/adapter_config.json"
                )
            from peft import PeftModel
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(
                base_model, trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            self.model = PeftModel.from_pretrained(base, resolved_path).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved_path, trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def decide(self, dialogue: Dict[str, Any], turn_number: int) -> str:
        """Return 'STOP' or 'WAIT' by running LLM inference."""
        prompt = build_stop_prompt(dialogue, turn_number)
        messages = [
            {"role": "system", "content": "You are an ad insertion timing controller. Output exactly STOP or WAIT."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            text = "\n".join(f"<{m['role']}>\n{m['content']}\n</{m['role']}>" for m in messages) + "\n<assistant>\n"

        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens,
                temperature=1.0, do_sample=True, top_p=0.9,
            )
        generated = self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return "STOP" if "STOP" in generated.upper() else "WAIT"

    def choose_stop_turn(self, dialogue: Dict[str, Any], force_auction_last_turn: bool = True) -> int | None:
        """Run the LLM on each turn until it says STOP."""
        turns = dialogue.get("turns", [])
        for idx, turn in enumerate(turns):
            tn = int(turn.get("turn", idx + 1))
            is_last = idx == len(turns) - 1
            if force_auction_last_turn and is_last:
                return tn
            decision = self.decide(dialogue, tn)
            if decision == "STOP":
                return tn
        return None


# ---------------------------------------------------------------------------
# Main: generate SFT data and train
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Ablation-2: Train LLM to output stop decisions")
    parser.add_argument("--dialogues", required=True)
    parser.add_argument("--base_model_dir", required=True, help="Baseline model dir with stop_policy + user_model")
    parser.add_argument("--output_dir", default="outputs/ablation_llm_stop")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    t_start = time.time()

    # Load baseline models (to generate labels)
    log.info(f"Loading baseline models from {args.base_model_dir} ...")
    pctr_model = UnifiedUserModel.load(args.base_model_dir, device)
    stop_policy = StopPolicy.load(args.base_model_dir, device)

    # Load dialogues and splits
    log.info("Loading dialogues ...")
    dialogues = load_jsonl(args.dialogues)

    splits_path = Path(args.base_model_dir) / "data_splits.json"
    if not splits_path.exists():
        splits_path = Path(args.base_model_dir).parent / "data_splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            sp = json.load(f)
        train_sessions = set(sp["train_sessions"])
    else:
        train_sessions = set(d["session_id"] for d in dialogues[:int(len(dialogues) * 0.7)])

    # Generate STOP/WAIT labels from baseline Stop Q-Network
    log.info("Generating STOP/WAIT labels from baseline Stop Q-Network ...")
    sft_rows = generate_stop_labels(
        dialogues, pctr_model, stop_policy, train_sessions,
    )
    log.info(f"  Generated {len(sft_rows)} training samples "
             f"(STOP: {sum(1 for r in sft_rows if r['label'] == 'STOP')}, "
             f"WAIT: {sum(1 for r in sft_rows if r['label'] == 'WAIT')})")

    # Save SFT data
    sft_data_path = output_dir / "stop_sft_data.jsonl"
    save_jsonl(sft_rows, sft_data_path)

    # Limit samples if needed
    if args.max_samples and len(sft_rows) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(sft_rows), size=args.max_samples, replace=False)
        sft_rows = [sft_rows[i] for i in sorted(indices)]
        log.info(f"  Subsampled to {len(sft_rows)} samples")

    # Free baseline models before LLM training
    del pctr_model, stop_policy
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Train LLM via SFT
    log.info(f"Training LLM stop policy via SFT (base: {args.base_model}) ...")
    from src.llm_sft.train_lora_sft import train_sft
    train_sft(
        sft_path=str(sft_data_path),
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

    elapsed = time.time() - t_start
    log.info(f"Done in {elapsed:.0f}s. Model saved to: {output_dir}")

    save_json({
        "variant": "llm_stop",
        "description": "Ablation-2: LLM directly outputs stop decision, violates assumption (A3)",
        "n_sft_samples": len(sft_rows),
        "n_stop": sum(1 for r in sft_rows if r["label"] == "STOP"),
        "n_wait": sum(1 for r in sft_rows if r["label"] == "WAIT"),
        "base_model": args.base_model,
        "base_model_dir": str(args.base_model_dir),
        "borrows_pctr_from": str(args.base_model_dir),
        "borrows_splits_from": str(args.base_model_dir),
        "elapsed_seconds": elapsed,
    }, output_dir / "ablation_meta.json")


if __name__ == "__main__":
    main()
