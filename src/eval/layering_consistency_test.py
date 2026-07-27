"""Experiment 7.5: Architecture-Separation Consistency Test.

Verifies that the bid-blind layering design guarantees DSIC assumptions (A2)(A3)
by evaluating three metrics across the baseline and ablation variants:

  Metric 1 - Bid-Quality Independence (q_i-b_i correlation)
      Perturb bid, measure pCTR change. Baseline should be ~0.

  Metric 2 - Misreport Gain Curve
      Sweep bid multipliers, compute utility. Baseline peaks at truthful bid.

  Metric 3 - Audit Replay Consistency
      Independently replay stop/allocation/payment and compare against
      reference evaluation results. Baseline deviation should be 0%.

Run:
  python -m src.eval.layering_consistency_test \
    --dialogues data/dialogues_all.jsonl \
    --model_dir outputs/final_qwen3emb/round_2 \
    --output_dir outputs/layering_test \
    --reference_csv outputs/final_qwen3emb/round_2/final_evaluation_sessions.csv \
    --n_sessions 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from src.data.data_utils import pctr_input_text, numeric_features
from src.eval.closed_loop_eval import (
    _choose_learned_stop_turn,
    _run_policy_allocation,
    compute_global_critical_cpc,
    compute_pctrs_for_state,
    mechanism_meta_at_turn,
)
from src.models.text_mlp import PCTRModel, UnifiedUserModel
from src.policy.stop_policy import StopPolicy, stop_features_from_auction
from src.utils.io import ensure_dir, load_jsonl, resolve_device, save_json


BID_PERTURBATIONS = [0.25, 0.5, 1.0, 2.0, 4.0]
BID_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


# ---------------------------------------------------------------------------
# Variant-aware feature builder
# ---------------------------------------------------------------------------
# Different ablations train pCTR with different feature builders. The eval
# loop must use the SAME builder the model was trained on, otherwise the
# tensor dimensions won't match (e.g. bid_in_pctr adds 2 numeric features).

def detect_variant(model_dir: str) -> str:
    """Read ablation_meta.json (if present) to pick the right feature path."""
    meta_path = Path(model_dir) / "ablation_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                return json.load(f).get("variant", "baseline")
        except Exception:
            return "baseline"
    return "baseline"


def resolve_pctr_dir(model_dir: str) -> str:
    """Where to load UnifiedUserModel (pCTR + Judge) from.

    Some ablations only train the LLM and reuse baseline pCTR — they record
    `borrows_pctr_from` in their ablation_meta.json so the eval can pull
    those weights from the baseline dir without copying files. Falls back
    to model_dir for self-contained variants (bid_in_pctr).
    """
    meta_path = Path(model_dir) / "ablation_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            borrow = meta.get("borrows_pctr_from")
            if borrow and Path(borrow).exists():
                return borrow
        except Exception:
            pass
    return model_dir


def resolve_stop_dir(model_dir: str) -> str:
    """Where to load Q-Net StopPolicy from (for non-llm_stop variants)."""
    meta_path = Path(model_dir) / "ablation_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            # Reuse the same field — variants that borrow pCTR also borrow stop.
            borrow = meta.get("borrows_pctr_from")
            if borrow and Path(borrow).exists():
                return borrow
        except Exception:
            pass
    return model_dir


def resolve_splits_dir(model_dir: str) -> str:
    """Where to look for data_splits.json."""
    meta_path = Path(model_dir) / "ablation_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            borrow = meta.get("borrows_splits_from") or meta.get("borrows_pctr_from")
            if borrow and Path(borrow).exists():
                return borrow
        except Exception:
            pass
    return model_dir


def make_pctr_compute(variant: str):
    """Return a compute_pctrs_for_state(model, dialogue, turn) function
    that uses the input features matching how `variant` was trained."""
    if variant == "bid_in_pctr":
        from src.ablation.train_bid_in_pctr import (
            pctr_input_text_with_bid,
            numeric_features_with_bid,
        )

        def _compute(pctr_model, dialogue, turn):
            ads = dialogue.get("candidate_ads", [])
            texts = [pctr_input_text_with_bid(dialogue, turn, ad) for ad in ads]
            nums = [numeric_features_with_bid(dialogue, turn, ad) for ad in ads]
            pctrs = pctr_model.predict_proba(texts, nums)
            return np.clip(pctrs, 1e-4, 1.0).astype(float).tolist()
        return _compute

    return compute_pctrs_for_state  # baseline / others use bid-blind features


# ---------------------------------------------------------------------------
# LLM-stop adapter (variant=llm_stop)
# ---------------------------------------------------------------------------
# Ablation-2 trains a Qwen3-4B SFT to output STOP/WAIT directly (violating
# A3). The downstream eval code (closed_loop_eval) calls `stop_policy.q_values
# (features)` and treats q[1] >= q[0] as "stop". To plug the LLM in without
# touching closed_loop_eval, we expose a tiny adapter whose q_values() turns
# the LLM's STOP/WAIT decision into a deterministic-looking [q_wait, q_auct]
# pair (1.0 / 0.0 if STOP, 0.0 / 1.0 if WAIT). To know which dialogue + turn
# the call refers to, we ship a thread-unsafe-but-good-enough module-global
# context that the metric-1/2/3 loops set before invoking auction code.

import threading

_LLM_STOP_CTX = threading.local()


def _set_llm_stop_context(dialogue: Dict[str, Any] | None, turn_number: int | None) -> None:
    _LLM_STOP_CTX.dialogue = dialogue
    _LLM_STOP_CTX.turn = turn_number


def _get_llm_stop_context():
    return (
        getattr(_LLM_STOP_CTX, "dialogue", None),
        getattr(_LLM_STOP_CTX, "turn", None),
    )


class LLMStopPolicyAdapter:
    """Wraps the trained LLMStopPolicy so it looks like a StopPolicy.

    The downstream auction code calls `stop_policy.q_values(features)` where
    `features` is the 14-dim vector built by `stop_features_from_auction`
    (stop_policy.py:42). features[0] = turn / 5.0 — we recover the turn
    number from there. The current dialogue is set per-session via
    `_set_llm_stop_context()` (the q_values signature provides no other
    channel for it). Returns [q_wait, q_auction] consistent with the LLM's
    STOP/WAIT decision so closed_loop_eval treats it as a normal stop policy.
    """

    def __init__(self, llm_policy):
        self._llm = llm_policy

    def q_values(self, features):
        dialogue, _session_turn = _get_llm_stop_context()
        if dialogue is None:
            return np.array([1.0, 0.0], dtype=np.float32)  # fallback: WAIT
        # Recover the actual turn number from feature[0] = turn / 5.0.
        try:
            turn_number = int(round(float(features[0]) * 5.0))
        except Exception:
            turn_number = 1
        decision = self._llm.decide(dialogue, max(1, turn_number))
        if decision == "STOP":
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.array([1.0, 0.0], dtype=np.float32)


def load_stop_policy_for_variant(variant: str, model_dir: str, device):
    """Return a StopPolicy-compatible object (real or LLM adapter)."""
    if variant == "llm_stop":
        from src.ablation.train_llm_stop import LLMStopPolicy

        # The trained LLM lives in model_dir (LoRA adapter or full SFT model).
        llm = LLMStopPolicy(str(model_dir), device=str(device))
        return LLMStopPolicyAdapter(llm)
    # For other variants, the Q-Net might live in model_dir or be borrowed
    # from baseline_dir via ablation_meta.json's borrows_pctr_from field.
    return StopPolicy.load(resolve_stop_dir(model_dir), device)


# ---------------------------------------------------------------------------
# Renderer + Judge loaders for metric 4 (variant=bid_in_renderer)
# ---------------------------------------------------------------------------
# Ablation-3 violates A2 by injecting bid into the Renderer prompt. The
# violation only surfaces in the rendered text -> judge -> click_prob chain,
# so we need both the Renderer (to produce text) and the Judge head of the
# user model (to score it). For all other variants, the bid-blind baseline
# Renderer is loaded so the comparison is fair.

def load_renderer_for_variant(
    variant: str,
    model_dir: str,
    base_model: str,
    baseline_adapter_dir: str | None,
    device: str = "auto",
):
    """Return (renderer, tag) for the given variant.

    bid_in_renderer: load BidAwareRenderer (LoRA from ablation dir).
    others:          load TrainedLLMRenderer with the baseline adapter.
    """
    if variant == "bid_in_renderer":
        # The trained LoRA lives directly in model_dir; BidAwareRenderer
        # currently wraps a base-model BaseLLMRenderer plus the bid-aware
        # prompt. To honor the trained adapter, swap in TrainedLLMRenderer
        # and override the system prompt to include bid info.
        from src.ablation.train_bid_in_renderer import BidAwareRenderer
        # BidAwareRenderer doesn't load adapters by default — patch it to
        # use the trained LoRA from model_dir for the underlying base model.
        from src.generation.llm_renderer import TrainedLLMRenderer
        inner = TrainedLLMRenderer(
            base_model=base_model,
            adapter_dir=model_dir,
            device=device,
            max_new_tokens=160,
            temperature=0.2,
            top_p=0.9,
        )
        return _BidAwareLoRARenderer(inner), "bid_aware_renderer"

    # Baseline / other variants — use the bid-blind trained Renderer.
    if baseline_adapter_dir is None:
        raise RuntimeError(
            "baseline adapter dir is required to load the baseline Renderer "
            "for non-bid_in_renderer variants. Pass --baseline_adapter_dir."
        )
    from src.generation.llm_renderer import TrainedLLMRenderer
    renderer = TrainedLLMRenderer(
        base_model=base_model,
        adapter_dir=baseline_adapter_dir,
        device=device,
        max_new_tokens=160,
        temperature=0.2,
        top_p=0.9,
    )
    return renderer, "baseline_trained_renderer"


class _BidAwareLoRARenderer:
    """Wraps a TrainedLLMRenderer (with bid_in_renderer LoRA) so its prompt
    surfaces bid values — replicating BidAwareRenderer's prompt shape but
    using the actually-trained adapter weights."""

    renderer_tag = "bid_aware_lora_renderer"

    def __init__(self, inner):
        self._inner = inner

    def render(self, dialogue, turn_number, ad):
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
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        import torch as _torch
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self._inner.device)
        with _torch.no_grad():
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
# Metric 4: Renderer-Side Bid-Quality Independence (ablation-3 target)
# ---------------------------------------------------------------------------

def test_renderer_bid_quality_independence(
    test_dialogues: List[Dict[str, Any]],
    pctr_model: PCTRModel,           # used as the Judge head (UnifiedUserModel)
    renderer,
    n_sessions: int = 50,
    seed: int = 42,
    bid_perturbations: List[float] | None = None,
) -> pd.DataFrame:
    """Per-ad Renderer + Judge perturbation: bid → render → judge → click_prob.

    For a bid-blind Renderer (baseline), the rendered text and the resulting
    click_prob should be invariant to bid changes.  An ablation-3 Renderer
    that sees bid in its prompt will (i) write more persuasive copy for
    higher bids, (ii) cause the Judge to score click_prob higher.

    Notes
    -----
    * Slow: each (session, ad, mult) triggers one LLM generation pass. We
      cap at `n_sessions=50` by default — enough for a Pearson on 5 points
      per ad while keeping wall-clock under ~15 minutes on a single GPU.
    * Uses the `pctr_model` as a stand-in Judge — same UnifiedUserModel
      backbone, judge head reads (history, ad, response).  Identical to
      what the closed-loop pipeline does at evaluation time.
    """
    from src.data.data_utils import judge_input_text, numeric_features

    if bid_perturbations is None:
        bid_perturbations = BID_PERTURBATIONS
    sessions = test_dialogues[:n_sessions]
    rows = []

    for dialogue in tqdm(sessions, desc="metric4: renderer bid-quality"):
        ads = dialogue.get("candidate_ads", [])
        turns = dialogue.get("turns", [])
        if not ads or not turns:
            continue
        last_turn = int(turns[-1].get("turn", len(turns)))

        for ad_idx, ad in enumerate(ads):
            true_bid = float(ad.get("bid", 0.0))
            if true_bid < 1e-6:
                continue

            click_probs = []
            response_lengths = []
            for mult in bid_perturbations:
                perturbed_ads = [
                    {**a, "bid": true_bid * mult} if i == ad_idx else a
                    for i, a in enumerate(ads)
                ]
                d_perturbed = {**dialogue, "candidate_ads": perturbed_ads}
                ad_perturbed = perturbed_ads[ad_idx]

                # 1) render with the variant-specific Renderer
                response, _ = renderer.render(d_perturbed, last_turn, ad_perturbed)

                # 2) judge the rendered text — UnifiedUserModel.predict expects
                #    (texts, nums, user_ids); user_ids=None falls through to 0.
                text = judge_input_text(d_perturbed, last_turn, ad_perturbed, response)
                nums = [numeric_features(d_perturbed, last_turn, ad_perturbed)]
                try:
                    click_prob_arr, _ = pctr_model.predict([text], nums, user_ids=[0])
                    click_prob = float(np.clip(click_prob_arr[0], 1e-6, 1.0))
                except Exception:
                    # Older PCTRModel only has predict_proba; fall back.
                    click_prob_arr = pctr_model.predict_proba([text], nums)
                    click_prob = float(np.clip(click_prob_arr[0], 1e-6, 1.0))

                click_probs.append(click_prob)
                response_lengths.append(len(response))

            log_mults = np.log(bid_perturbations)
            corr, p_value = stats.pearsonr(log_mults, click_probs)
            max_delta = max(click_probs) - min(click_probs)

            rows.append({
                "session_id": dialogue["session_id"],
                "ad_id": ad.get("ad_id", ""),
                "ad_idx": ad_idx,
                "true_bid": true_bid,
                **{f"click_at_{m}x": p for m, p in zip(bid_perturbations, click_probs)},
                **{f"resp_len_at_{m}x": l for m, l in zip(bid_perturbations, response_lengths)},
                "pearson_corr": float(corr) if not np.isnan(corr) else 0.0,
                "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
                "max_click_delta": float(max_delta),
            })

    return pd.DataFrame(rows)


def summarize_renderer_bid_quality(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"mean_abs_corr": 0.0, "max_click_delta_mean": 0.0, "n_samples": 0}
    return {
        "mean_abs_corr": float(df["pearson_corr"].abs().mean()),
        "median_abs_corr": float(df["pearson_corr"].abs().median()),
        "mean_click_delta": float(df["max_click_delta"].mean()),
        "max_click_delta": float(df["max_click_delta"].max()),
        "pct_significant_p005": float((df["p_value"] < 0.05).mean()),
        "n_samples": int(len(df)),
    }


# ---------------------------------------------------------------------------
# Metric 1: Bid-Quality Independence
# ---------------------------------------------------------------------------

def test_bid_quality_independence(
    test_dialogues: List[Dict[str, Any]],
    pctr_model: PCTRModel,
    n_sessions: int = 200,
    seed: int = 42,
    compute_pctrs=None,
) -> pd.DataFrame:
    """Perturb each ad's bid and measure pCTR response.

    For a bid-blind model, pCTR should be invariant to bid changes.
    Returns per-ad correlation between bid multiplier and pCTR output.
    """
    if compute_pctrs is None:
        compute_pctrs = compute_pctrs_for_state
    rng = np.random.default_rng(seed)
    sessions = test_dialogues[:n_sessions]
    rows = []

    for dialogue in tqdm(sessions, desc="metric1: bid-quality independence"):
        ads = dialogue.get("candidate_ads", [])
        turns = dialogue.get("turns", [])
        if not ads or not turns:
            continue
        last_turn = int(turns[-1].get("turn", len(turns)))

        base_pctrs = compute_pctrs(pctr_model, dialogue, last_turn)

        for ad_idx, ad in enumerate(ads):
            true_bid = float(ad.get("bid", 0.0))
            if true_bid < 1e-6:
                continue

            pctrs_at_perturbations = []
            for mult in BID_PERTURBATIONS:
                perturbed_ads = []
                for idx, a in enumerate(ads):
                    if idx == ad_idx:
                        perturbed_ads.append({**a, "bid": true_bid * mult})
                    else:
                        perturbed_ads.append(a)

                d_perturbed = {**dialogue, "candidate_ads": perturbed_ads}
                pctrs = compute_pctrs(pctr_model, d_perturbed, last_turn)
                pctrs_at_perturbations.append(float(pctrs[ad_idx]))

            log_mults = np.log(BID_PERTURBATIONS)
            corr, p_value = stats.pearsonr(log_mults, pctrs_at_perturbations)
            max_delta = max(pctrs_at_perturbations) - min(pctrs_at_perturbations)

            rows.append({
                "session_id": dialogue["session_id"],
                "ad_id": ad.get("ad_id", ""),
                "ad_idx": ad_idx,
                "true_bid": true_bid,
                "base_pctr": float(base_pctrs[ad_idx]),
                "pctr_at_0.25x": pctrs_at_perturbations[0],
                "pctr_at_0.5x": pctrs_at_perturbations[1],
                "pctr_at_1.0x": pctrs_at_perturbations[2],
                "pctr_at_2.0x": pctrs_at_perturbations[3],
                "pctr_at_4.0x": pctrs_at_perturbations[4],
                "pearson_corr": float(corr) if not np.isnan(corr) else 0.0,
                "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
                "max_pctr_delta": float(max_delta),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metric 2: Misreport Gain Curve
# ---------------------------------------------------------------------------

def test_misreport_gain(
    test_dialogues: List[Dict[str, Any]],
    pctr_model: PCTRModel,
    stop_policy: StopPolicy,
    bid_multipliers: List[float] | None = None,
    n_sessions: int = 200,
    seed: int = 42,
    compute_pctrs=None,
) -> pd.DataFrame:
    """Sweep bid multipliers per advertiser, compute expected utility.

    Utility = (theta_i - P_i) * click_prob if won, else 0.
    For DSIC, utility should peak at multiplier=1.0 (truthful bidding).
    """
    if compute_pctrs is None:
        compute_pctrs = compute_pctrs_for_state
    if bid_multipliers is None:
        bid_multipliers = BID_MULTIPLIERS
    sessions = test_dialogues[:n_sessions]
    rows = []

    for dialogue in tqdm(sessions, desc="metric2: misreport gain"):
        ads = dialogue.get("candidate_ads", [])
        turns = dialogue.get("turns", [])
        if len(ads) < 2 or not turns:
            continue

        pctr_cache: Dict[int, List[float]] = {}
        for idx_t, turn in enumerate(turns):
            tn = int(turn.get("turn", idx_t + 1))
            pctr_cache[tn] = compute_pctrs(pctr_model, dialogue, tn)

        last_turn = int(turns[-1].get("turn", len(turns)))
        pctrs = pctr_cache.get(last_turn, compute_pctrs(pctr_model, dialogue, last_turn))

        # For LLM-stop variant: ship the dialogue down through thread-local
        # so the adapter can call llm.decide(dialogue, turn) — turn is read
        # from the feature vector inside q_values().
        _set_llm_stop_context(dialogue, None)

        for target_idx, target_ad in enumerate(ads):
            true_bid = float(target_ad.get("bid", 0.0))
            if true_bid < 1e-6:
                continue

            for mult in bid_multipliers:
                perturbed_bid = true_bid * mult
                perturbed_ads = []
                for idx, ad in enumerate(ads):
                    if idx == target_idx:
                        perturbed_ads.append({**ad, "bid": perturbed_bid})
                    else:
                        perturbed_ads.append(ad)

                chosen_turn, auction_result, meta, _, _ = _run_policy_allocation(
                    "learned_stop",
                    dialogue,
                    pctr_model,
                    stop_policy,
                    candidate_ads=perturbed_ads,
                    force_auction_last_turn=True,
                    psi_mode="identity",
                    pctr_cache=pctr_cache,
                )

                if auction_result is None or chosen_turn is None:
                    won = False
                    payment = 0.0
                    click_prob = 0.0
                else:
                    won = (auction_result.winner_index == target_idx)
                    if won:
                        payment = compute_global_critical_cpc(
                            "learned_stop",
                            dialogue,
                            pctr_model,
                            stop_policy,
                            winner_index=target_idx,
                            winner_bid=perturbed_bid,
                            fixed_turn=None,
                            force_auction_last_turn=True,
                            psi_mode="identity",
                            pctr_cache=pctr_cache,
                        )
                    else:
                        payment = 0.0
                    click_prob = float(pctrs[target_idx])

                utility = (true_bid - payment) * click_prob if won else 0.0

                rows.append({
                    "session_id": dialogue["session_id"],
                    "ad_id": target_ad.get("ad_id", ""),
                    "ad_idx": target_idx,
                    "true_bid": true_bid,
                    "bid_multiplier": mult,
                    "reported_bid": perturbed_bid,
                    "won": int(won),
                    "payment": payment,
                    "click_prob": click_prob,
                    "utility": utility,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metric 3: Audit Replay Consistency
# ---------------------------------------------------------------------------

def test_audit_replay(
    test_dialogues: List[Dict[str, Any]],
    pctr_model: PCTRModel,
    stop_policy: StopPolicy,
    reference_csv_path: str | None = None,
    n_sessions: int = 200,
    seed: int = 42,
    self_consistency: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Independently replay the mechanism and compare with reference results.

    Replays: pCTR -> Stop decision -> Allocation -> Global critical CPC.

    Two modes:
      * `self_consistency=False` (default): compare against reference_csv
        (from batch_evaluate) field by field. Tests cross-run consistency
        with a frozen baseline.
      * `self_consistency=True`: run the replay TWICE on the same sessions
        and compare the two runs against each other. This is the only way
        to expose stochastic stop policies (LLM-stop) — a deterministic
        Q-Net produces identical replays both times, an LLM with sampling
        does not.
    """
    sessions = test_dialogues[:n_sessions]

    def _do_replay(label: str = "metric3: audit replay") -> pd.DataFrame:
        rows = []
        for dialogue in tqdm(sessions, desc=label):
            ads = dialogue.get("candidate_ads", [])
            turns = dialogue.get("turns", [])
            if not ads or not turns:
                continue

            # For LLM-stop variant: hand the dialogue down to the adapter.
            _set_llm_stop_context(dialogue, None)

            chosen_turn, result, meta, q_wait, q_auction = _run_policy_allocation(
                "learned_stop",
                dialogue,
                pctr_model,
                stop_policy,
                force_auction_last_turn=True,
                psi_mode="identity",
            )

            if result is None or chosen_turn is None:
                rows.append({
                    "session_id": dialogue["session_id"],
                    "replay_trigger_turn": np.nan,
                    "replay_winner_ad_id": "",
                    "replay_winner_bid": 0.0,
                    "replay_payment": 0.0,
                    "replay_triggered": 0,
                })
                continue

            payment = compute_global_critical_cpc(
                "learned_stop",
                dialogue,
                pctr_model,
                stop_policy,
                winner_index=result.winner_index,
                winner_bid=float(result.winner_ad.get("bid", 0.0)),
                fixed_turn=None,
                force_auction_last_turn=True,
                psi_mode="identity",
            )

            rows.append({
                "session_id": dialogue["session_id"],
                "replay_trigger_turn": int(chosen_turn),
                "replay_winner_ad_id": result.winner_ad.get("ad_id", ""),
                "replay_winner_bid": float(result.winner_ad.get("bid", 0.0)),
                "replay_payment": float(payment),
                "replay_triggered": 1,
            })
        return pd.DataFrame(rows)

    # --- Self-consistency mode: run twice and compare -----------------------
    if self_consistency:
        run_a = _do_replay("metric3 audit replay (run A)")
        run_b = _do_replay("metric3 audit replay (run B)")
        merged = run_a.merge(
            run_b, on="session_id", how="inner", suffixes=("_a", "_b"),
        )
        if merged.empty:
            return run_a, {"error": "no overlapping sessions in two runs"}

        both = merged[
            (merged["replay_triggered_a"] == 1) & (merged["replay_triggered_b"] == 1)
        ]
        n_both = len(both)
        if n_both == 0:
            return run_a, {
                "mode": "self_consistency",
                "n_matched": int(len(merged)),
                "n_both_triggered": 0,
                "stop_turn_match_rate": 0.0,
                "winner_match_rate": 0.0,
                "payment_mae": 0.0,
                "fully_consistent": False,
            }
        stop_match = (both["replay_trigger_turn_a"] == both["replay_trigger_turn_b"]).mean()
        winner_match = (both["replay_winner_ad_id_a"] == both["replay_winner_ad_id_b"]).mean()
        payment_diff = np.abs(both["replay_payment_a"] - both["replay_payment_b"])
        summary = {
            "mode": "self_consistency",
            "n_matched": int(len(merged)),
            "n_both_triggered": int(n_both),
            "trigger_match_rate": float(
                (merged["replay_triggered_a"] == merged["replay_triggered_b"]).mean()
            ),
            "stop_turn_match_rate": float(stop_match),
            "winner_match_rate": float(winner_match),
            "payment_mae": float(payment_diff.mean()),
            "payment_max_deviation": float(payment_diff.max()),
            "fully_consistent": bool(
                stop_match == 1.0 and winner_match == 1.0 and float(payment_diff.mean()) < 1e-6
            ),
        }
        return run_a, summary

    # --- Reference-CSV mode (default) ---------------------------------------
    replay_df = _do_replay()

    if reference_csv_path is None:
        summary = {
            "n_replayed": len(replay_df),
            "note": "no reference CSV provided, replay-only results",
        }
        return replay_df, summary

    ref_df = pd.read_csv(reference_csv_path)
    ref_learned = ref_df[ref_df["policy"] == "learned_stop"].copy()
    if ref_learned.empty:
        summary = {"error": "no learned_stop rows in reference CSV"}
        return replay_df, summary

    merged = replay_df.merge(
        ref_learned[["session_id", "trigger_turn", "winner_ad_id", "payment", "triggered"]],
        on="session_id",
        how="inner",
    )

    if merged.empty:
        summary = {"error": "no matching sessions between replay and reference"}
        return replay_df, summary

    both_triggered = merged[(merged["replay_triggered"] == 1) & (merged["triggered"] == 1)]

    if both_triggered.empty:
        summary = {
            "n_matched": len(merged),
            "trigger_match_rate": float((merged["replay_triggered"] == merged["triggered"]).mean()),
            "stop_turn_match_rate": 0.0,
            "winner_match_rate": 0.0,
            "payment_mae": 0.0,
        }
        return replay_df, summary

    stop_match = (both_triggered["replay_trigger_turn"] == both_triggered["trigger_turn"]).mean()
    winner_match = (both_triggered["replay_winner_ad_id"] == both_triggered["winner_ad_id"]).mean()
    payment_diff = np.abs(both_triggered["replay_payment"] - both_triggered["payment"])
    payment_mae = float(payment_diff.mean())
    payment_max_dev = float(payment_diff.max())

    summary = {
        "n_matched": int(len(merged)),
        "n_both_triggered": int(len(both_triggered)),
        "trigger_match_rate": float((merged["replay_triggered"] == merged["triggered"]).mean()),
        "stop_turn_match_rate": float(stop_match),
        "winner_match_rate": float(winner_match),
        "payment_mae": payment_mae,
        "payment_max_deviation": payment_max_dev,
        "fully_consistent": bool(stop_match == 1.0 and winner_match == 1.0 and payment_mae < 1e-6),
    }
    return replay_df, summary


# ---------------------------------------------------------------------------
# Summarization helpers
# ---------------------------------------------------------------------------

def summarize_bid_quality(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"mean_abs_corr": 0.0, "max_pctr_delta_mean": 0.0, "n_samples": 0}
    return {
        "mean_abs_corr": float(df["pearson_corr"].abs().mean()),
        "median_abs_corr": float(df["pearson_corr"].abs().median()),
        "mean_pctr_delta": float(df["max_pctr_delta"].mean()),
        "max_pctr_delta": float(df["max_pctr_delta"].max()),
        "pct_significant_p005": float((df["p_value"] < 0.05).mean()),
        "n_samples": int(len(df)),
    }


def summarize_misreport(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"n_samples": 0}
    by_mult = df.groupby("bid_multiplier").agg(
        win_rate=("won", "mean"),
        avg_payment=("payment", "mean"),
        avg_utility=("utility", "mean"),
        median_utility=("utility", "median"),
        n=("utility", "count"),
    ).reset_index()

    truthful_util = float(by_mult[by_mult["bid_multiplier"] == 1.0]["avg_utility"].iloc[0]) if 1.0 in by_mult["bid_multiplier"].values else 0.0
    best_mult = float(by_mult.loc[by_mult["avg_utility"].idxmax(), "bid_multiplier"])
    best_util = float(by_mult["avg_utility"].max())
    gain_from_deviation = best_util - truthful_util

    return {
        "truthful_avg_utility": truthful_util,
        "best_multiplier": best_mult,
        "best_avg_utility": best_util,
        "gain_from_deviation": gain_from_deviation,
        "dsic_holds": bool(best_mult == 1.0 or gain_from_deviation < 1e-4),
        "by_multiplier": by_mult.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 7.5: Layering Consistency Test")
    parser.add_argument("--dialogues", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/layering_test")
    parser.add_argument("--reference_csv", default="")
    parser.add_argument("--n_sessions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--skip",
        nargs="*",
        choices=["metric1", "metric2", "metric3", "metric4"],
        default=[],
        help="Skip specific metrics",
    )
    parser.add_argument(
        "--metric3_self_consistency",
        choices=["auto", "yes", "no"],
        default="auto",
        help="metric3 mode: 'yes' = run replay TWICE and compare against itself "
             "(needed to expose stochastic stop policies like llm_stop); "
             "'no' = compare against --reference_csv only; "
             "'auto' (default) = self-consistency for variant=llm_stop, "
             "reference-csv otherwise.",
    )
    parser.add_argument(
        "--metric4",
        choices=["auto", "yes", "no"],
        default="auto",
        help="metric4 (Renderer-side bid-quality) mode: 'auto' = run only for "
             "variant=bid_in_renderer; 'yes' = always run (slow, requires "
             "--llm_base_model and --baseline_adapter_dir); 'no' = never run.",
    )
    parser.add_argument(
        "--llm_base_model",
        default="Qwen/Qwen3-4B",
        help="Base LLM path for Renderer load (metric4 only).",
    )
    parser.add_argument(
        "--baseline_adapter_dir",
        default="",
        help="Baseline Renderer LoRA adapter dir (metric4 baseline comparison).",
    )
    parser.add_argument(
        "--metric4_n_sessions",
        type=int,
        default=50,
        help="Sessions for metric4 (LLM generation is slow; default 50).",
    )
    parser.add_argument(
        "--metric3_n_sessions",
        type=int,
        default=0,
        help="Sessions for metric3 audit replay; 0 = use --n_sessions. "
             "Set lower (e.g. 100-200) for variant=llm_stop where every "
             "session triggers tens of LLM calls.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    print(f"[1/4] Loading models from {args.model_dir} ...")
    pctr_dir = resolve_pctr_dir(args.model_dir)
    if pctr_dir != args.model_dir:
        print(f"  pctr/judge: borrowed from {pctr_dir}")
    pctr_model = UnifiedUserModel.load(pctr_dir, device)

    variant = detect_variant(args.model_dir)
    pctr_compute = make_pctr_compute(variant)
    stop_policy = load_stop_policy_for_variant(variant, args.model_dir, device)
    print(f"  detected variant = {variant} (pctr feature: "
          f"{'bid-aware' if variant == 'bid_in_pctr' else 'bid-blind'}; "
          f"stop policy: {'LLM (stochastic)' if variant == 'llm_stop' else 'Q-Net (deterministic)'})")

    print(f"[2/4] Loading dialogues ...")
    dialogues = load_jsonl(args.dialogues)

    splits_root = resolve_splits_dir(args.model_dir)
    splits_path = Path(splits_root) / "data_splits.json"
    if not splits_path.exists():
        splits_path = Path(splits_root).parent / "data_splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            sp = json.load(f)
        test_sessions = set(sp.get("test_sessions", []))
        test_dialogues = [d for d in dialogues if d["session_id"] in test_sessions]
        print(f"  Using {len(test_dialogues)} test sessions from {splits_path}")
    else:
        test_dialogues = dialogues
        print(f"  No splits file found, using all {len(test_dialogues)} dialogues")

    overall_summary: Dict[str, Any] = {"n_sessions": args.n_sessions, "seed": args.seed}

    # --- Metric 1 ---
    if "metric1" not in args.skip:
        print(f"\n[3a/4] Metric 1: Bid-Quality Independence ...")
        m1_df = test_bid_quality_independence(
            test_dialogues, pctr_model, n_sessions=args.n_sessions, seed=args.seed,
            compute_pctrs=pctr_compute,
        )
        m1_df.to_csv(output_dir / "bid_quality_correlation.csv", index=False)
        m1_summary = summarize_bid_quality(m1_df)
        overall_summary["metric1_bid_quality"] = m1_summary
        print(f"  mean |corr| = {m1_summary['mean_abs_corr']:.6f}, "
              f"max delta = {m1_summary['max_pctr_delta']:.6f}, "
              f"significant (p<0.05) = {m1_summary['pct_significant_p005']:.1%}")
    else:
        print("\n[3a/4] Metric 1: SKIPPED")

    # --- Metric 2 ---
    if "metric2" not in args.skip:
        print(f"\n[3b/4] Metric 2: Misreport Gain Curve ...")
        m2_df = test_misreport_gain(
            test_dialogues, pctr_model, stop_policy,
            n_sessions=args.n_sessions, seed=args.seed,
            compute_pctrs=pctr_compute,
        )
        m2_df.to_csv(output_dir / "misreport_utility.csv", index=False)
        m2_summary = summarize_misreport(m2_df)
        overall_summary["metric2_misreport"] = m2_summary
        print(f"  truthful utility = {m2_summary['truthful_avg_utility']:.4f}, "
              f"best mult = {m2_summary['best_multiplier']}, "
              f"gain from deviation = {m2_summary['gain_from_deviation']:.4f}, "
              f"DSIC = {m2_summary['dsic_holds']}")
    else:
        print("\n[3b/4] Metric 2: SKIPPED")

    # --- Metric 3 ---
    if "metric3" not in args.skip:
        print(f"\n[3c/4] Metric 3: Audit Replay Consistency ...")
        ref_csv = args.reference_csv if args.reference_csv else None
        if args.metric3_self_consistency == "auto":
            self_consistency = (variant == "llm_stop")
        else:
            self_consistency = (args.metric3_self_consistency == "yes")
        if self_consistency:
            print(f"  mode: SELF-CONSISTENCY (run replay twice, compare against itself)")
        else:
            print(f"  mode: reference-csv (compare against {ref_csv or '(none)'})")
        m3_n = args.metric3_n_sessions if args.metric3_n_sessions > 0 else args.n_sessions
        if m3_n != args.n_sessions:
            print(f"  metric3 using n_sessions={m3_n} (overridden from default {args.n_sessions})")
        m3_df, m3_summary = test_audit_replay(
            test_dialogues, pctr_model, stop_policy,
            reference_csv_path=ref_csv, n_sessions=m3_n, seed=args.seed,
            self_consistency=self_consistency,
        )
        m3_df.to_csv(output_dir / "audit_replay.csv", index=False)
        overall_summary["metric3_audit_replay"] = m3_summary
        if "fully_consistent" in m3_summary:
            print(f"  stop_turn match = {m3_summary['stop_turn_match_rate']:.1%}, "
                  f"winner match = {m3_summary['winner_match_rate']:.1%}, "
                  f"payment MAE = {m3_summary['payment_mae']:.6f}, "
                  f"consistent = {m3_summary['fully_consistent']}")
        else:
            print(f"  {m3_summary}")
    else:
        print("\n[3c/4] Metric 3: SKIPPED")

    # --- Metric 4: Renderer-side bid-quality (ablation-3 target) ---
    if args.metric4 == "auto":
        run_metric4 = (variant == "bid_in_renderer")
    else:
        run_metric4 = (args.metric4 == "yes")

    if run_metric4 and "metric4" not in args.skip:
        print(f"\n[3d/4] Metric 4: Renderer-Side Bid-Quality Independence ...")
        baseline_adapter = args.baseline_adapter_dir or None
        try:
            renderer, renderer_tag = load_renderer_for_variant(
                variant, args.model_dir, args.llm_base_model,
                baseline_adapter, device=str(device),
            )
            print(f"  loaded renderer: {renderer_tag}")
            m4_df = test_renderer_bid_quality_independence(
                test_dialogues, pctr_model, renderer,
                n_sessions=args.metric4_n_sessions, seed=args.seed,
            )
            m4_df.to_csv(output_dir / "renderer_bid_quality.csv", index=False)
            m4_summary = summarize_renderer_bid_quality(m4_df)
            overall_summary["metric4_renderer_bid_quality"] = m4_summary
            print(f"  mean |corr| = {m4_summary['mean_abs_corr']:.6f}, "
                  f"max click delta = {m4_summary['max_click_delta']:.6f}, "
                  f"significant (p<0.05) = {m4_summary['pct_significant_p005']:.1%}")
        except Exception as e:
            print(f"  metric4 SKIPPED due to error: {e}")
            overall_summary["metric4_renderer_bid_quality"] = {"error": str(e)}
    else:
        print("\n[3d/4] Metric 4: SKIPPED")

    # --- Save summary ---
    print(f"\n[4/4] Saving summary to {output_dir / 'layering_test_summary.json'}")
    save_json(overall_summary, output_dir / "layering_test_summary.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
