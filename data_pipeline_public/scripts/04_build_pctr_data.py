"""
04_build_pctr_data.py
从 dialogues.jsonl 提取 pCTR 训练数据

三类样本（与 llm_auction_alignment.md 4.1.1 节对齐）:
  - target_auction  (正样本):  τ 轮插 target_ad，raw_clicked（不强制）
  - counterfactual_ad (负样本1): τ 轮换其他候选广告，raw_clicked
  - target_early    (负样本2): 1~τ-1 轮提前插 target_ad，上下文不足→低 pCTR

每条样本字段:
  - session_id, context, response, profile
  - ad_id, ad_title, ad_text, bid
  - turn, sample_type, raw_clicked (0 or 1)

输出: data/pctr_train.jsonl
"""

import json
import argparse
from pathlib import Path


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_context(context_parts, user_msg):
    """把累积的历史 + 当前 user 消息拼成 context 字符串"""
    return "\n".join(context_parts + ["User: {}".format(user_msg)])


def extract_pctr_samples(dialogues_path, ads_corpus_path, output_path):
    """从 dialogues.jsonl 提取三类 pCTR 训练样本"""

    dialogues = load_jsonl(dialogues_path)

    # 加载广告语料（补充 ad_text / short_description）
    ads_map = {}
    if ads_corpus_path and Path(ads_corpus_path).exists():
        for ad in load_jsonl(ads_corpus_path):
            ads_map[ad["ad_id"]] = ad

    samples = []

    for sess in dialogues:
        # ── profile 文本 ──
        profile_text = ""
        if sess.get("profile"):
            p = sess["profile"]
            profile_text = "{}. Interests: {}".format(
                p.get("basic_info", ""),
                p.get("personal_interests", ""),
            )

        # ── 逐轮处理 ──
        context_parts = []   # 累积已完成轮次的对话历史

        for t_rec in sess.get("turns", []):
            action = t_rec.get("action")
            if action not in ("WAIT", "SOFT"):
                continue

            user_msg = t_rec.get("user", "")
            assistant_msg = t_rec.get("assistant", "")
            turn_idx = t_rec.get("turn")

            # 当前轮的 context = 之前所有轮 + 本轮 user
            current_context = build_context(context_parts, user_msg)

            # ══════════════════════════════════════════════════════
            # 【SOFT 轮】: target_auction 正样本 + counterfactual_ad 负样本1
            # ══════════════════════════════════════════════════════
            if action == "SOFT":
                judge = t_rec.get("judge") or {}
                judge_ads = {a["ad_id"]: a for a in judge.get("ads", [])}

                # ── 正样本: target_auction ──
                # 被插入的广告，使用 raw_clicked（不强制）
                for ad_id, aj in judge_ads.items():
                    ad_full = ads_map.get(ad_id, {})
                    raw_clicked = aj.get("raw_clicked", aj.get("clicked", False))
                    samples.append({
                        "session_id": sess.get("session_id"),
                        "context": current_context,
                        "response": assistant_msg,
                        "profile": profile_text,
                        "ad_id": ad_id,
                        "ad_title": aj.get("title", ad_full.get("title", "")),
                        "ad_text": ad_full.get("short_description", ""),
                        "bid": aj.get("bid", ad_full.get("bid", 0)),
                        "turn": turn_idx,
                        "sample_type": "target_auction",
                        "raw_clicked": 1 if raw_clicked else 0,
                    })

                # ── 负样本1: counterfactual_ad ──
                # τ 轮强制插其他候选广告，raw_clicked 大概率为 false
                for cf in t_rec.get("counterfactuals", []):
                    cf_ad_full = ads_map.get(cf["ad_id"], {})
                    raw_clicked = cf.get("raw_clicked", False)
                    samples.append({
                        "session_id": sess.get("session_id"),
                        "context": current_context,
                        "response": cf["response"],
                        "profile": profile_text,
                        "ad_id": cf["ad_id"],
                        "ad_title": cf.get("title", cf_ad_full.get("title", "")),
                        "ad_text": cf_ad_full.get("short_description", ""),
                        "bid": cf.get("bid", 0),
                        "turn": cf.get("turn", turn_idx),
                        "sample_type": "counterfactual_ad",
                        "raw_clicked": 1 if raw_clicked else 0,
                    })

            # ══════════════════════════════════════════════════════
            # 【WAIT 轮】: target_early 负样本2
            # 提前强制插 target_ad，上下文不足 → pCTR 低
            # ══════════════════════════════════════════════════════
            elif action == "WAIT":
                te = t_rec.get("target_early")
                if te:
                    te_ad_full = ads_map.get(te["ad_id"], {})
                    raw_clicked = te.get("raw_clicked", False)
                    samples.append({
                        "session_id": sess.get("session_id"),
                        "context": current_context,
                        "response": te["response"],
                        "profile": profile_text,
                        "ad_id": te["ad_id"],
                        "ad_title": te.get("title", te_ad_full.get("title", "")),
                        "ad_text": te_ad_full.get("short_description", ""),
                        "bid": te.get("bid", 0),
                        "turn": te.get("turn", turn_idx),
                        "sample_type": "target_early",
                        "raw_clicked": 1 if raw_clicked else 0,
                    })

            # 累积已完成轮次历史
            if user_msg:
                context_parts.append("User: {}".format(user_msg))
            if assistant_msg:
                context_parts.append("Assistant: {}".format(assistant_msg))

    # ── 写出 ──
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ── 统计 ──
    n_pos = sum(1 for s in samples if s["raw_clicked"] == 1)
    n_neg = sum(1 for s in samples if s["raw_clicked"] == 0)
    by_type = {}
    for s in samples:
        by_type[s["sample_type"]] = by_type.get(s["sample_type"], 0) + 1

    print("pCTR training data: {} samples ({} positive, {} negative)".format(
        len(samples), n_pos, n_neg,
    ))
    for t, c in sorted(by_type.items()):
        print("  {}: {}".format(t, c))
    print("Wrote -> {}".format(output_path))


def main():
    parser = argparse.ArgumentParser(description="Build pCTR training data")
    parser.add_argument("--dialogues", type=str,
                        default="data/dialogues.jsonl",
                        help="Path to dialogues.jsonl")
    parser.add_argument("--ads-corpus", type=str,
                        default="data/ads_corpus_synthetic.jsonl",
                        help="Path to ads corpus")
    parser.add_argument("--output", type=str,
                        default="data/pctr_train.jsonl",
                        help="Output path")
    args = parser.parse_args()

    extract_pctr_samples(args.dialogues, args.ads_corpus, args.output)


if __name__ == "__main__":
    main()