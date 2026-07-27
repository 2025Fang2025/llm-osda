"""
Step 03: 多轮对话数据生成主脚本（含广告拍卖）

完整流水线：
  阶段1 (Phase 1): 从用户画像生成查询问题
      输入: profiles.jsonl (用户画像)
      输出: queries.jsonl  (每个用户5个问题)
      模型: a GPT-family model (dialogue_user)

  阶段2 (Phase 2): 对每个(画像,问题)运行多轮对话
      Turn 1 ~ auction_turn-1: 纯对话 (WAIT，不插广告)
      Turn auction_turn:       拍卖轮 — 选候选广告，Ad-LLM 生成带 @Ad Title@[Ad ID] 的回复
      Turn auction_turn+1 ~ :  如果用户留存(gamma高)则继续对话
      模型: a GPT-family model (用户追问) + a Claude-family model (助手回复)

  阶段3 (Phase 3): 对拍卖轮调用 Judge 评估点击行为和留存率
      输出: dialogues.jsonl (完整多轮对话记录)
      模型: a GPT-family model (judge)

用法:
    cd data_pipeline
    python scripts/03_generate_pure_dialogues.py [--dry-run] [--limit N] [--auction-turn 3]
"""

import json
import random
import argparse
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from utils_io import read_jsonl, write_jsonl
from _prompts import (
    USER_QUERY_GENERATION,
    PURE_USER_SYSTEM,
    PURE_ASSISTANT_SYSTEM,
    AD_RESPONSE_GENERATION,
    CLICK_FEEDBACK_JUDGE,
    CLICK_FEEDBACK_JUDGE_EARLY,
    AUCTION_TIMING_JUDGE,
    DPO_PREFERENCE_JUDGE,
)
from _llm import call_llm_async, call_llm_sync, CFG


# ========================================================================
# 辅助函数：格式化 profile / 对话历史 / 广告列表
# ========================================================================

def format_profile_basic_info(profile):
    # type: (dict) -> str
    """从 profile 字典中提取基本信息，拼成一行文本。
    
    示例输出: "Age: 28; Gender: Male; Occupation: Software Engineer; ..."
    用于填充 prompt 中的 {basic_info} 占位符。
    """
    parts = []
    for key in ["age", "gender", "occupation", "budget_level", "life_situation"]:
        if key in profile:
            # 把 "budget_level" 转成 "Budget Level" 这样的可读格式
            parts.append("{}: {}".format(key.replace("_", " ").title(), profile[key]))
    return "; ".join(parts)


def format_profile_interests(profile):
    # type: (dict) -> str
    """从 profile 字典中提取兴趣偏好信息。
    
    示例输出: "Personality: adventurous; Shopping Habit: impulse buyer; ..."
    用于填充 prompt 中的 {personal_interests} 占位符。
    """
    parts = []
    for key in ["personality", "shopping_habit", "preferred_categories", "pain_point"]:
        if key in profile:
            val = profile[key]
            # preferred_categories 可能是列表，需要转成逗号分隔的字符串
            if isinstance(val, list):
                val = ", ".join(val)
            parts.append("{}: {}".format(key.replace("_", " ").title(), val))
    return "; ".join(parts)


def format_profile_full(profile):
    # type: (dict) -> str
    """拼接完整的 profile 文本（基本信息 + 兴趣），用于 Ad-LLM 的 {user_profile} 占位符。"""
    return "Basic Info: {}\nInterests: {}".format(
        format_profile_basic_info(profile),
        format_profile_interests(profile),
    )


def format_conversation_history(turns):
    # type: (List[Dict[str, str]]) -> str
    """把对话历史列表格式化成文本。
    
    输入: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    输出: "User: ...\nAssistant: ...\n..."
    用于填充 prompt 中的 {conversation_history} 占位符。
    """
    if not turns:
        return "(No prior conversation)"
    lines = []
    for t in turns:
        role = "User" if t["role"] == "user" else "Assistant"
        lines.append("{}: {}".format(role, t["content"]))
    return "\n".join(lines)


def format_candidate_ads_json(ads):
    # type: (List[dict]) -> str
    """把候选广告列表格式化成 JSON 行，匹配论文 B.1 的格式。
    
    每行一个广告: {"Ad ID": "A_0005", "Ad Title": "Sony WH-1000XM5...", "Ad Bid": 1.97}
    用于填充 AD_RESPONSE_GENERATION prompt 中的 {candidate_ads_json}。
    """
    lines = []
    for a in ads:
        lines.append(json.dumps({
            "Ad ID": a["ad_id"],
            "Ad Title": a["title"],
            "Ad Bid": a["bid"],
        }, ensure_ascii=False))
    return "\n".join(lines)


def format_trajectory(session_id, query, profile, candidate_ads, turns):
    # type: (str, str, dict, List[dict], List[dict]) -> str
    """把一次 session 渲染成可读的完整对话轨迹文本，用于人工核查。

    结构: 初始问题 → 用户画像 → 候选广告 → 多轮对话(含广告轮) → 点击结果
    """
    lines = []

    # ── 初始问题 ──
    lines.append("Query: {}".format(query))

    # ── 用户画像 ──
    lines.append("Profile: [{}] {}".format(
        profile.get("profile_id", ""),
        profile.get("description", profile.get("persona", "")),
    ))

    # ── 候选广告 ──
    lines.append("Candidate Ads:")
    for i, a in enumerate(candidate_ads or []):
        lines.append("  {}. {} (id={}, bid=${:.2f})".format(
            i + 1, a["title"], a["ad_id"], a["bid"]
        ))

    # ── 多轮对话 ──
    lines.append("Dialogue:")
    for t_rec in turns:
        action = t_rec.get("action", "WAIT")

        user_msg = t_rec.get("user", "")
        if user_msg:
            lines.append("  User: {}".format(user_msg))

        asst_msg = t_rec.get("assistant", "")
        if action == "SOFT":
            # 广告轮：标注插入的广告
            inserted = t_rec.get("ads_inserted", [])
            inserted_str = ", ".join(
                "{}[{}]".format(a["title"], a["ad_id"]) for a in inserted
            ) if inserted else "none"
            lines.append("  Assistant [AD: {}]: {}".format(inserted_str, asst_msg))

            # ── 点击结果 ──
            judge = t_rec.get("judge") or {}
            for aj in judge.get("ads", []):
                clicked = aj.get("clicked", "?")
                reason = aj.get("reason", "")
                lines.append("  Click: ad={} clicked={} reason={}".format(
                    aj.get("ad_id", "?"), clicked, reason
                ))
            gamma = judge.get("gamma_score")
            if gamma is not None:
                lines.append("  Retention gamma={:.2f}".format(gamma))
        elif asst_msg:
            lines.append("  Assistant: {}".format(asst_msg))

    return "\n".join(lines)


def format_cf_trajectory(query, profile, candidate_ads, history_turns, cf_user_msg, cf_response, cf_ad, cf_judge):
    # type: (str, dict, List[dict], List[dict], str, str, dict, dict) -> str
    """格式化反事实样本的对话轨迹。

    结构: 初始问题 → 用户画像 → 候选广告 → 历史对话 → 当前用户问题 → 反事实广告回复 → 点击结果
    """
    lines = []

    lines.append("Query: {}".format(query))
    lines.append("Profile: [{}] {}".format(
        profile.get("profile_id", ""),
        profile.get("description", profile.get("persona", "")),
    ))

    lines.append("Candidate Ads:")
    for i, a in enumerate(candidate_ads or []):
        lines.append("  {}. {} (id={}, bid=${:.2f})".format(
            i + 1, a["title"], a["ad_id"], a["bid"]
        ))

    lines.append("Dialogue:")
    for t_rec in history_turns:
        user_msg = t_rec.get("user", "")
        if user_msg:
            lines.append("  User: {}".format(user_msg))
        asst_msg = t_rec.get("assistant", "")
        if asst_msg:
            lines.append("  Assistant: {}".format(asst_msg))

    # 当前轮：用户消息 + 反事实广告回复
    lines.append("  User: {}".format(cf_user_msg))
    lines.append("  Assistant [CF-AD: {}[{}]]: {}".format(
        cf_ad["title"], cf_ad["ad_id"], cf_response
    ))

    # 点击结果
    judge = cf_judge or {}
    for aj in judge.get("ads", []):
        lines.append("  Click: ad={} clicked={} reason={}".format(
            aj.get("ad_id", "?"), aj.get("clicked", "?"), aj.get("reason", "")
        ))
    gamma = judge.get("gamma_score")
    if gamma is not None:
        lines.append("  Retention gamma={:.2f}".format(gamma))

    return "\n".join(lines)


def select_candidate_ads(all_ads, profile, k_relevant=2, k_random=1, rng=None, dialogue=None, target_ad=None, peer_ad=None):
    # type: (List[dict], dict, int, int, Optional[random.Random], Optional[List[Dict]], Optional[dict], Optional[dict]) -> List[dict]
    """为一次对话选择候选广告。
    
    策略:
      1. target_ad 一定包含（query 围绕其品类生成）
      2. peer_ad 一定包含（同品类竞品，保证 query 至少匹配 2 个广告）
      3. 再补充随机无关广告作为干扰项
    
    候选结构 (默认 k_relevant=2, k_random=1 → 共 3 个):
      候选1: target_ad  (高 bid, query 最相关)
      候选2: peer_ad    (同品类竞品, 也能回答 query)
      候选3: 随机无关广告 (干扰项)
    
    参数:
        all_ads: 全部100条广告
        profile: 用户画像
        k_relevant: 相关广告数 (含 target_ad 和 peer_ad)
        k_random: 随机广告数
        rng: 随机数生成器
        dialogue: 当前对话历史 (保留接口, 未使用)
        target_ad: 目标广告 (必须包含)
        peer_ad: 同品类竞品 (必须包含)
    
    返回: 候选广告列表 (按 bid 从高到低排序)
    """
    if rng is None:
        rng = random
    
    # target_ad + peer_ad 一定入选
    chosen_rel = []
    chosen_ids = set()
    if target_ad:
        chosen_rel.append(target_ad)
        chosen_ids.add(target_ad["ad_id"])
    if peer_ad and peer_ad["ad_id"] not in chosen_ids:
        chosen_rel.append(peer_ad)
        chosen_ids.add(peer_ad["ad_id"])
    
    # 随机广告: 从不同品类中选（避免与 target/peer 同品类）
    target_cat = target_ad["category"].lower() if target_ad else ""
    others_pool = [a for a in all_ads 
                   if a["ad_id"] not in chosen_ids and a["category"].lower() != target_cat]
    chosen_rand = rng.sample(others_pool, min(k_random, len(others_pool))) if others_pool else []

    # 合并，按 bid 从高到低排序
    candidates = chosen_rel + chosen_rand
    candidates.sort(key=lambda a: a["bid"], reverse=True)
    return candidates


def parse_queries(text, expected_n=5):
    # type: (str, int) -> List[Tuple[str, str]]
    """解析 LLM 返回的编号问题列表，同时提取意图类型标注。

    LLM 输出格式示例:
        1. [CLEAR] need noise-cancelling headphones under $200 for subway
        2. [PARTIAL] my neck hurts from sitting at home all day
        ...

    返回: [(query_text, intent_type), ...]
    intent_type: "CLEAR" | "PARTIAL" | "VAGUE"
    """
    lines = text.strip().split("\n")
    results = []
    intent_pattern = re.compile(r"^\[(CLEAR|PARTIAL|VAGUE)\]\s*", re.IGNORECASE)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 去掉编号前缀
        cleaned = re.sub(r"^\d+[\.\)\:]\s*", "", line)
        # 提取意图类型
        m = intent_pattern.match(cleaned)
        if m:
            intent_type = m.group(1).upper()
            query_text = cleaned[m.end():].strip()
        else:
            intent_type = "PARTIAL"  # 默认 PARTIAL
            query_text = cleaned
        if query_text:
            results.append((query_text, intent_type))
    return results[:expected_n]


def parse_judge_response(text):
    # type: (str) -> dict
    """解析 Judge LLM 返回的 JSON 评估结果。
    
    期望格式:
    {
      "ads": [{"ad_id": "A_0028", "clicked": true, "reason": "..."}],
      "gamma_score": 0.88,
      "gamma_reason": "..."
    }
    
    因为 LLM 有时会在 JSON 外面加 ```json ``` 标记或多余文字，
    所以这里做了多层 fallback 解析。
    """
    text = text.strip()
    
    # 尝试1: LLM 用 ```json ... ``` 包裹了 JSON
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    
    # 尝试2: 直接解析整段文本
    try:
        result = json.loads(text)
        if "ads" not in result:
            result["ads"] = []
        if "gamma_score" not in result:
            result["gamma_score"] = 0.5
        return result
    except json.JSONDecodeError:
        pass
    
    # 尝试3: 从文本中找第一个 {...} 结构
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # 全部失败: 返回默认值，保留原始文本用于调试
    return {"ads": [], "gamma_score": 0.5, "gamma_reason": "parse_error", "_raw": text}


async def should_insert_ad(dialogue_so_far, turn_number, max_turns=3):
    # type: (List[Dict[str, str]], int, int) -> bool
    """判断当前轮是否适合插入广告。

    用轻量 LLM 调用评估对话上下文，判断用户意图是否
    已经足够明确、当前时机是否适合自然地引入商品推荐。

    返回: True = 适合插广告, False = 继续纯对话
    """
    latest_user_msg = ""
    for d in reversed(dialogue_so_far):
        if d["role"] == "user":
            latest_user_msg = d["content"]
            break

    prompt = AUCTION_TIMING_JUDGE.format(
        conversation_history=format_conversation_history(dialogue_so_far[:-1]),
        latest_user_message=latest_user_msg,
        turn_number=turn_number,
        max_turns=max_turns,
    )
    messages = [{"role": "user", "content": prompt}]

    # 用 dialogue_user (a GPT-family model) 做轻量判断，低温保证稳定
    text = await call_llm_async(
        messages,
        model_key="dialogue_user",
        temperature=0.3,
        max_tokens=10,
    )
    return text.strip().upper().startswith("YES")


# ========================================================================
# 阶段1: 从用户画像生成查询问题
# ========================================================================

async def generate_queries(profile, num_queries=5, all_ads=None, rng=None):
    # type: (dict, int, Optional[List[dict]], Optional[random.Random]) -> List[Tuple[str, dict, dict]]
    """【阶段1】从一个用户画像生成 N 个查询问题，每个问题绑定一个目标广告。
    
    改进: 
      1. 先选 target_ad, 再找同品类的 peer_ad (保证 query 至少匹配 2 个广告)
      2. query 围绕品类/场景生成，而非单个商品 → target 和 peer 都能回答
      3. 广告语料每品类 2 个广告, 天然保证 peer 存在
    
    参数:
        profile: 一条用户画像 (dict)
        num_queries: 生成几个问题 (默认5)
        all_ads: 全部广告库 (用于选择目标广告)
        rng: 随机数生成器
    
    返回: [(query_text, target_ad, peer_ad), ...] 
          query_text: 用户问题
          target_ad:  主目标广告 (高 bid 偏好)
          peer_ad:    同品类竞品广告 (保证 query 至少匹配 2 个)
    """
    if rng is None:
        rng = random
    if not all_ads:
        raise ValueError("all_ads is required for query generation")
    
    # ── 按品类分组, 每组内按 bid 降序 ──
    from collections import defaultdict
    cat_groups = defaultdict(list)
    for ad in all_ads:
        cat_groups[ad["category"]].append(ad)
    for cat in cat_groups:
        cat_groups[cat].sort(key=lambda a: a["bid"], reverse=True)
    
    # 只保留至少有 2 个广告的品类 (保证有 peer)
    valid_cats = [cat for cat, ads in cat_groups.items() if len(ads) >= 2]
    
    # 为每个 query 选一个品类 (target_ad + peer_ad)
    pref_cats = profile.get("preferred_categories", [])
    if isinstance(pref_cats, str):
        pref_cats = [c.strip().lower() for c in pref_cats.split(",")]
    else:
        pref_cats = [c.lower() for c in pref_cats]
    
    pref_valid = [c for c in valid_cats if c.lower() in pref_cats]
    other_valid = [c for c in valid_cats if c.lower() not in pref_cats]
    
    n_pref = min(3, len(pref_valid), num_queries)
    n_other = min(num_queries - n_pref, len(other_valid))
    
    selected_cats = rng.sample(pref_valid, n_pref) if pref_valid else []
    selected_cats += rng.sample(other_valid, n_other) if other_valid else []
    
    # 不够的从全部品类补
    if len(selected_cats) < num_queries:
        remaining_cats = [c for c in valid_cats if c not in selected_cats]
        selected_cats += rng.sample(remaining_cats, min(num_queries - len(selected_cats), len(remaining_cats)))
    
    selected_cats = selected_cats[:num_queries]
    rng.shuffle(selected_cats)
    
    # 对每个品类选 target_ad (高 bid 优先) + peer_ad
    ad_pairs = []  # [(target_ad, peer_ad), ...]
    for cat in selected_cats:
        ads_in_cat = cat_groups[cat]
        # target = bid 最高的 (加权采样引入一些随机性)
        weights = [max(a["bid"], 0.1) for a in ads_in_cat]
        total_w = sum(weights)
        r = rng.random() * total_w
        cum = 0
        target_idx = 0
        for i, w in enumerate(weights):
            cum += w
            if cum >= r:
                target_idx = i
                break
        target = ads_in_cat[target_idx]
        # peer = 同品类另一个
        peers = [a for a in ads_in_cat if a["ad_id"] != target["ad_id"]]
        peer = peers[0] if peers else target
        ad_pairs.append((target, peer))
    
    # 构造目标商品描述 — 给 LLM 看品类+两个商品 (让 query 覆盖两者)
    products_text = ""
    for i, (target, peer) in enumerate(ad_pairs, 1):
        products_text += (
            "{}. Category: [{}]\n"
            "   Product A: {} — {}\n"
            "   Product B: {} — {}\n"
        ).format(
            i, target["category"],
            target["title"], target.get("short_description", ""),
            peer["title"], peer.get("short_description", ""),
        )
    
    # 把 profile 字段填入 prompt 模板
    prompt = USER_QUERY_GENERATION.format(
        basic_info=format_profile_basic_info(profile),
        personal_interests=format_profile_interests(profile),
        num_queries=num_queries,
        target_products=products_text,
    )
    messages = [{"role": "user", "content": prompt}]

    # 调用 a GPT-family model，温度0.9保证多样性
    text = await call_llm_async(
        messages,
        model_key="dialogue_user",
        temperature=0.9,
        max_tokens=600,
    )
    
    # 解析编号列表 → 纯文本问题
    queries = parse_queries(text, expected_n=num_queries)
    
    # 如果 LLM 返回的问题不够数，用已有的重复填充
    while len(queries) < num_queries and queries:
        queries.append(queries[len(queries) % len(queries)])
    
    # 返回 (query, intent_type, target_ad, peer_ad) 四元组
    results = []
    for i, (q, intent_type) in enumerate(queries[:num_queries]):
        if i < len(ad_pairs):
            target, peer = ad_pairs[i]
        else:
            target, peer = ad_pairs[-1]
        results.append((q, intent_type, target, peer))
    return results


# ========================================================================
# 阶段2: 多轮对话生成（含拍卖轮）
# ========================================================================

async def generate_user_followup(profile, dialogue_so_far):
    # type: (dict, List[Dict[str, str]]) -> str
    """生成用户在多轮对话中的追问消息。
    
    原理: 用 a GPT-family model 扮演用户。关键技巧是【角色翻转】——
    因为 LLM 只能以 assistant 角色输出，所以我们把对话历史中的
    user 消息标记为 assistant（让 LLM 以为那是"自己"说的），
    assistant 消息标记为 user（让 LLM 以为那是"对方"说的）。
    这样 LLM 就会自然地延续"用户"的角色继续追问。
    
    参数:
        profile: 用户画像
        dialogue_so_far: 到目前为止的对话历史
                         [{"role": "user", "content": "..."}, 
                          {"role": "assistant", "content": "..."}, ...]
    
    返回: 用户的下一句话 (str)
          如果用户觉得对话该结束了，会返回包含 "<STOP>" 的字符串
    """
    # PURE_USER_SYSTEM 是用户角色的 system prompt，包含画像信息
    user_sys = PURE_USER_SYSTEM.format(
        profile_tags=format_profile_full(profile)
    )
    
    messages = [{"role": "system", "content": user_sys}]

    # 【角色翻转】: 原始对话中 user→assistant, assistant→user
    # 只保留最近 3 轮，避免历史累积导致输出越来越长
    recent = dialogue_so_far[-6:]  # 最多 3 轮 = 6 条消息
    for d in recent:
        if d["role"] == "user":
            # 用户历史消息截短，防止模型模仿长度
            messages.append({"role": "assistant", "content": d["content"][:60]})
        else:
            # 助手消息截短到关键问题部分
            messages.append({"role": "user", "content": d["content"][:120]})

    # 不追加额外指令——让模型直接回应助手刚才的问题
    # system prompt 已经要求了简短，这里只用 max_tokens 兜底

    reply = await call_llm_async(
        messages,
        model_key="dialogue_user",
        temperature=0.9,
        max_tokens=40,  # 强制截断：真实用户追问很少超过 30 词
    )
    return reply


async def generate_assistant_wait(dialogue_so_far, candidate_ads=None):
    # type: (List[Dict[str, str]], Optional[List[dict]]) -> str
    """生成助手的回复（WAIT 动作，模型看到候选广告但选择不插入）。
    
    原理: 用 a Claude-family model 扮演购物助手。
    模型从一开始就看到候选广告+bid，但 prompt 指示在对话
    早期阶段应正常回复、不插广告（pCTR 还不够高）。
    
    参数:
        dialogue_so_far: 对话历史
        candidate_ads: 候选广告列表（含bid），若为空则回退到无广告 prompt
    
    返回: 助手的纯文本回复 (str)
    """
    # 构建 system prompt（带候选广告信息）
    if candidate_ads:
        ads_json = format_candidate_ads_json(candidate_ads)
        system_content = PURE_ASSISTANT_SYSTEM.format(candidate_ads_json=ads_json)
    else:
        system_content = PURE_ASSISTANT_SYSTEM.format(candidate_ads_json="(No candidate ads)")
    
    messages = [{"role": "system", "content": system_content}]
    for d in dialogue_so_far:
        messages.append({"role": d["role"], "content": d["content"]})

    reply = await call_llm_async(
        messages,
        model_key="dialogue_assistant",
        temperature=0.7,
        max_tokens=200,
    )
    return reply


async def generate_assistant_ad(profile, dialogue_so_far, candidate_ads):
    # type: (dict, List[Dict[str, str]], List[dict]) -> str
    """【拍卖轮】生成带广告插入的助手回复。
    
    与论文对齐: 给 Ad-LLM 看多个候选广告（含 target_ad），
    让模型自己选择最合适的那个插入。专家轨迹中 target_ad 
    最相关（query 围绕它生成），模型大概率会选它。
    
    参数:
        profile: 用户画像
        dialogue_so_far: 对话历史 (最后一条应该是当前 user 消息)
        candidate_ads: 候选广告列表 (5条，含 target_ad)
    
    返回: 带广告的助手回复 (str)，其中包含 @...@[...] 标记
    """
    # 分离当前问题和之前的对话历史
    current_query = ""
    history_turns = []
    for d in dialogue_so_far:
        if d == dialogue_so_far[-1] and d["role"] == "user":
            current_query = d["content"]
        else:
            history_turns.append(d)

    # 填充 AD_RESPONSE_GENERATION prompt (多个候选广告)
    prompt = AD_RESPONSE_GENERATION.format(
        candidate_ads_json=format_candidate_ads_json(candidate_ads),
        user_profile=format_profile_full(profile),
        conversation_history=format_conversation_history(history_turns),
        user_query=current_query,
    )
    messages = [{"role": "user", "content": prompt}]

    # 温度稍低(0.7)，让广告插入更稳定可控
    reply = await call_llm_async(
        messages,
        model_key="dialogue_assistant",
        temperature=0.7,
        max_tokens=250,
    )
    return reply


# ========================================================================
# 阶段3: Judge 评估（点击行为 + 留存率）
# ========================================================================

async def judge_ad_turn(profile, query, dialogue_so_far, ad_response, turn_number=1):
    # type: (dict, str, List[Dict[str, str]], str, int) -> dict
    """【阶段3】对拍卖轮的回复进行 Judge 评估。
    
    原理: 用 a GPT-family model 扮演用户，评估:
      1. 是否会点击回复中的每个广告 (clicked: true/false + reason)
      2. 整体留存率 gamma_score (0~1，继续对话的概率)
    
    Judge 的判断依据 (来自论文 B.1):
      - Relevance: 广告与问题/兴趣的相关性
      - Nativeness: 广告融入回复的自然程度
      - Competitiveness: 是否有竞品广告分散注意力
      - Timing appropriateness: 对话是否进展到足以自然推荐的阶段
    
    参数:
        profile: 用户画像 (提供 Judge 的角色信息)
        query: 用户的初始问题
        dialogue_so_far: 完整对话历史 (不含当前 ad_response)
        ad_response: 拍卖轮的助手回复 (待评估)
        turn_number: 当前对话轮次 (用于 Judge 评估 timing 合理性)
    
    返回: Judge 结果 dict
        {
          "ads": [{"ad_id": "A_0028", "clicked": true, "reason": "..."}],
          "gamma_score": 0.88,
          "gamma_reason": "..."
        }
    """
    prompt = CLICK_FEEDBACK_JUDGE.format(
        basic_info=format_profile_basic_info(profile),
        personal_interests=format_profile_interests(profile),
        user_query=query,
        conversation_history=format_conversation_history(dialogue_so_far[:-1]),
        ad_response=ad_response,
        turn_number=turn_number,
    )
    messages = [{"role": "user", "content": prompt}]

    # 温度极低(0.3)，让评估结果更稳定一致
    text = await call_llm_async(
        messages,
        model_key="judge",
        temperature=0.3,
        max_tokens=2048,
    )
    return parse_judge_response(text)


async def judge_ad_turn_early(profile, query, dialogue_so_far, ad_response, insert_turn):
    # type: (dict, str, List[Dict[str, str]], str, int) -> dict
    """对 target_early 样本（提前强插广告）进行 Judge 评估。

    与 judge_ad_turn 的区别：使用 CLICK_FEEDBACK_JUDGE_EARLY prompt，
    明确告知 Judge 当前是提前插入，让时机因素纳入点击判断。
    预期结果：即使广告相关，因时机未成熟，clicked 大概率为 False。

    参数:
        insert_turn: 广告实际插入的轮次（早期轮，<auction_turn）
    """
    prompt = CLICK_FEEDBACK_JUDGE_EARLY.format(
        basic_info=format_profile_basic_info(profile),
        personal_interests=format_profile_interests(profile),
        user_query=query,
        conversation_history=format_conversation_history(dialogue_so_far[:-1]),
        ad_response=ad_response,
        insert_turn=insert_turn,
    )
    messages = [{"role": "user", "content": prompt}]

    text = await call_llm_async(
        messages,
        model_key="judge",
        temperature=0.3,
        max_tokens=2048,
    )
    return parse_judge_response(text)


async def judge_dpo_preference(profile, query, dialogue_so_far, response_a, response_b):
    # type: (dict, str, List[Dict[str, str]], str, str) -> dict
    """对同一上下文的两个回复进行偏好判断，用于 DPO 训练数据构建。
    
    返回: {"preferred": "A"/"B", "reason": "...", "scores": {...}}
    """
    prompt = DPO_PREFERENCE_JUDGE.format(
        basic_info=format_profile_basic_info(profile),
        personal_interests=format_profile_interests(profile),
        user_query=query,
        conversation_history=format_conversation_history(dialogue_so_far),
        response_a=response_a,
        response_b=response_b,
    )
    messages = [{"role": "user", "content": prompt}]

    text = await call_llm_async(
        messages,
        model_key="judge",
        temperature=0.3,
        max_tokens=1024,
    )
    
    # 解析 JSON（复用 judge 的解析逻辑）
    text = text.strip()
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
    return {"preferred": "A", "reason": "parse_error", "_raw": text}


async def generate_dpo_pairs(profile, query, dialogue_so_far, candidate_ads, rng):
    # type: (dict, str, List[Dict[str, str]], List[dict], random.Random) -> List[dict]
    """为拍卖轮生成 DPO 配对数据。
    
    对同一个对话上下文，生成多组 (chosen, rejected) 对：
      1. SOFT vs WAIT (该不该插广告)
      2. SOFT_ad1 vs SOFT_ad2 (选哪个广告更好)
    
    返回: DPO pair 列表
    """
    pairs = []
    history_for_judge = dialogue_so_far[:-1]  # 不含最后一条 user msg
    
    # ── Pair 1: SOFT vs WAIT ──
    # 生成一个带广告的回复和一个纯回复
    soft_response = await generate_assistant_ad(profile, dialogue_so_far, candidate_ads)
    wait_response = await generate_assistant_wait(dialogue_so_far, candidate_ads=candidate_ads)
    
    # Judge 评两个
    soft_judge = await judge_ad_turn(profile, query, dialogue_so_far, soft_response, turn_number=len([d for d in dialogue_so_far if d["role"] == "user"]))
    pref_result = await judge_dpo_preference(
        profile, query, history_for_judge, soft_response, wait_response
    )
    
    pairs.append({
        "pair_type": "timing",
        "context": format_conversation_history(dialogue_so_far),
        "response_a": soft_response,
        "response_a_type": "SOFT",
        "response_b": wait_response,
        "response_b_type": "WAIT",
        "judge_a": soft_judge,
        "preference": pref_result,
        "chosen": soft_response if pref_result.get("preferred") == "A" else wait_response,
        "rejected": wait_response if pref_result.get("preferred") == "A" else soft_response,
        "chosen_type": "SOFT" if pref_result.get("preferred") == "A" else "WAIT",
    })
    
    # ── Pair 2: SOFT_ad_x vs SOFT_ad_y (不同广告选择) ──
    # 只在候选广告 >= 2 个相关广告时才做
    if len(candidate_ads) >= 2:
        # 把候选广告分成两组，各生成一个 SOFT 回复
        ads_group_1 = [candidate_ads[0]] + candidate_ads[2:]
        ads_group_2 = [candidate_ads[1]] + candidate_ads[2:]
        
        soft_1 = await generate_assistant_ad(profile, dialogue_so_far, ads_group_1)
        soft_2 = await generate_assistant_ad(profile, dialogue_so_far, ads_group_2)
        
        pref_result_2 = await judge_dpo_preference(
            profile, query, history_for_judge, soft_1, soft_2
        )
        
        pairs.append({
            "pair_type": "selection",
            "context": format_conversation_history(dialogue_so_far),
            "response_a": soft_1,
            "response_a_primary_ad": candidate_ads[0]["ad_id"],
            "response_b": soft_2,
            "response_b_primary_ad": candidate_ads[1]["ad_id"],
            "preference": pref_result_2,
            "chosen": soft_1 if pref_result_2.get("preferred") == "A" else soft_2,
            "rejected": soft_2 if pref_result_2.get("preferred") == "A" else soft_1,
        })
    
    return pairs

# ========================================================================
# 主流程: 运行单个 session（一次完整的多轮对话 + 拍卖）
# ========================================================================

async def run_one_session(
    profile,        # type: dict          # 用户画像
    query,          # type: str           # 初始查询问题
    all_ads,        # type: List[dict]    # 全部100条广告
    session_id,     # type: str           # 会话ID，如 "sess_P_0001_q00"
    max_turns,      # type: int           # 最大对话轮数 (如 8)
    rng,            # type: random.Random # 随机数生成器
    target_ad=None,    # type: Optional[dict] # 该 query 对应的目标广告
    peer_ad=None,      # type: Optional[dict] # 同品类竞品广告
    intent_type="PARTIAL",  # type: str        # query 的意图类型
):
    # type: (...) -> dict
    """运行一个完整的多轮对话 session。
    
    广告插入时机由模型动态决定:
        每轮先让 timing judge 判断"用户意图是否足够明确、此时推荐是否自然"，
        时机成熟 → 拍卖轮(SOFT)，广告出完直接结束对话
        时机不成熟 → 继续纯对话(WAIT)
        如果到 max_turns 还没插广告 → 强制在最后一轮插入
    
    参数:
        profile: 用户画像
        query: Phase 1 生成的初始问题
        all_ads: 全部广告库 (用于选候选广告)
        session_id: 唯一会话标识符
        max_turns: 最大对话轮数
        rng: 随机数生成器 (保证可复现)
        target_ad: 该 query 绑定的目标广告 (保证出现在候选集中)
        peer_ad: 同品类竞品广告 (保证出现在候选集中)
    
    返回: 完整的 session 记录 (dict)，包含所有轮次的对话、广告、Judge 结果
    """
    turns = []              # 存放每轮的记录
    dialogue = []           # 对话历史: [{"role": "user/assistant", "content": "..."}]
    candidate_ads = []      # 本次拍卖的候选广告
    actual_auction_turn = None  # 实际发生拍卖的轮次
    exit_reason = None      # 对话结束原因

    # ── 对话开始前就选好候选广告（模型从第1轮就看到）──
    candidate_ads = select_candidate_ads(
        all_ads, profile, k_relevant=2, k_random=1, rng=rng, dialogue=[], target_ad=target_ad, peer_ad=peer_ad
    )

    # ── 构建并保存初始系统 prompt（用于调试对话流程是否合理）──
    user_system_prompt = PURE_USER_SYSTEM.format(
        profile_tags=format_profile_full(profile)
    )
    assistant_system_prompt = PURE_ASSISTANT_SYSTEM.format(
        candidate_ads_json=format_candidate_ads_json(candidate_ads)
    )

    # 第1轮的用户消息就是 Phase 1 生成的初始问题
    user_msg = query

    for t in range(1, max_turns + 1):
        # ── 记录用户消息 ──
        dialogue.append({"role": "user", "content": user_msg})
        turn_record = {
            "turn": t,
            "user": user_msg,
            "assistant": "",
            "action": "WAIT",       # 默认是 WAIT (无广告)
            "ads_inserted": [],     # 实际被插入的广告
            "judge": None,          # Judge 评估结果 (只有拍卖轮才有)
        }

        # ── 判断本轮是否插广告 ──
        # earliest_turn: 根据 intent_type 设置最早可出广告的轮次
        # Judge 只在 earliest_turn 之后才被调用，内容判断逻辑不变
        earliest_turn = {"CLEAR": 2, "PARTIAL": 3, "VAGUE": 4}.get(intent_type, 3)

        do_auction = False
        if actual_auction_turn is not None:
            do_auction = False
        elif t == max_turns:
            do_auction = True
        elif t < earliest_turn:
            # 还没到最早可出广告的轮次，直接 WAIT
            do_auction = False
        else:
            # 到了最早轮次，交给 Judge 根据对话内容判断
            do_auction = await should_insert_ad(dialogue, turn_number=t, max_turns=max_turns)

        if do_auction:
            # ══════════════════════════════════════════
            # 【拍卖轮 (专家轨迹)】: Ad-LLM 从候选中选广告插入 → Judge 评估
            # query 围绕 target_ad 生成，保证相关性最高，但 clicked 由 Judge
            # 自然判断，允许为 false（反映真实用户的不确定性）
            # 重试目标是最大化 gamma_score（对话自然度），取多次生成中最优的一次
            # ══════════════════════════════════════════
            actual_auction_turn = t
            
            # candidate_ads 在对话开始前已选好（模型从第1轮就看到）
            
            max_ad_retries = 3
            best_response = None
            best_judge = None
            best_gamma = -1.0

            for ad_attempt in range(max_ad_retries):
                # 给 Ad-LLM 看多个候选，让它自己选（与论文对齐）
                ad_response = await generate_assistant_ad(profile, dialogue, candidate_ads)

                # 检测是否插入了任何广告，没插就重试
                inserted_any = any(
                    re.search(r"@{}@\[{}\]".format(re.escape(ad["title"]), re.escape(ad["ad_id"])), ad_response)
                    for ad in candidate_ads
                )
                if not inserted_any:
                    if ad_attempt < max_ad_retries - 1:
                        print(f"  [retry] no ad inserted, attempt {ad_attempt+2}/{max_ad_retries}")
                        continue

                # Judge 评估
                dialogue.append({"role": "assistant", "content": ad_response})
                judge_result = await judge_ad_turn(profile, query, dialogue, ad_response, turn_number=t)
                dialogue.pop()

                gamma = judge_result.get("gamma_score", 0.5) if judge_result else 0.5

                if gamma > best_gamma:
                    best_gamma = gamma
                    best_response = ad_response
                    best_judge = judge_result

                if gamma >= 0.5:
                    break
                elif ad_attempt < max_ad_retries - 1:
                    print(f"  [retry] gamma={gamma:.2f} < 0.5, attempt {ad_attempt+2}/{max_ad_retries}")

            ad_response = best_response
            judge_result = best_judge

            # 检测最终回复中实际插入了哪些广告
            inserted = []
            for ad in candidate_ads:
                pattern = r"@{}@\[{}\]".format(re.escape(ad["title"]), re.escape(ad["ad_id"]))
                if re.search(pattern, ad_response):
                    inserted.append({
                        "ad_id": ad["ad_id"],
                        "title": ad["title"],
                        "bid": ad["bid"],
                    })

            dialogue.append({"role": "assistant", "content": ad_response})
            turn_record["assistant"] = ad_response
            turn_record["action"] = "SOFT"
            turn_record["ads_inserted"] = inserted
            turn_record["sample_type"] = "target_auction"  # pCTR 正样本

            # clicked 字段直接使用 Judge 的原始判断，无需额外处理

            turn_record["judge"] = judge_result

        else:
            # ══════════════════════════════════════════
            # 【普通轮 (WAIT)】: 模型看到候选广告但选择不插入
            # ══════════════════════════════════════════
            max_wait_retries = 3
            asst_reply = None
            for wait_attempt in range(max_wait_retries):
                candidate_reply = await generate_assistant_wait(dialogue, candidate_ads=None)

                # 安全检查1: 去掉 @...@[...] 格式标记
                for ad in candidate_ads:
                    pattern = r"@{}@\[{}\]".format(re.escape(ad["title"]), re.escape(ad["ad_id"]))
                    if re.search(pattern, candidate_reply):
                        candidate_reply = re.sub(pattern, ad["title"], candidate_reply)

                # 安全检查2: 检测产品名泄露
                title_leaked = any(ad["title"].lower() in candidate_reply.lower() for ad in candidate_ads)

                if not title_leaked:
                    asst_reply = candidate_reply
                    break
                elif wait_attempt < max_wait_retries - 1:
                    print(f"  [wait-retry] title leaked in WAIT turn {t}, attempt {wait_attempt+2}/{max_wait_retries}")
                else:
                    # 重试耗尽：轮次够了就升级 SOFT，否则用最后一次（有泄露但没有更好选择）
                    if actual_auction_turn is None and t >= earliest_turn:
                        do_auction = True
                        print(f"  [escalate] title leaked after {max_wait_retries} retries, escalating to SOFT at turn {t}")
                    else:
                        asst_reply = candidate_reply
                        turn_record["title_leaked"] = True
                        print(f"  [warn] title leaked after {max_wait_retries} retries at turn {t}, keeping with flag")

            if not do_auction:
                # 用干净的（或已标记的）回复推进对话
                # 用户追问基于这条回复生成，保证对话历史不含泄露产品名（尽力而为）
                dialogue.append({"role": "assistant", "content": asst_reply})
                turn_record["assistant"] = asst_reply

        turns.append(turn_record)

        # ── 广告轮结束后直接终止对话 ──
        if do_auction:
            # 出了广告就停，Judge 已经评完了
            exit_reason = "ad_shown"
            break

        # ── 生成下一轮的用户追问 (仅 WAIT 轮) ──
        if t < max_turns:
            user_msg = await generate_user_followup(profile, dialogue)
            
            if "<STOP>" in user_msg:
                # 用户不满意，表达不满后离开
                stop_msg = user_msg.replace("<STOP>", "").strip()
                if stop_msg:
                    turns.append({
                        "turn": t + 1,
                        "user": stop_msg,
                        "assistant": "",
                        "action": "STOP",
                        "ads_inserted": [],
                        "judge": None,
                    })
                exit_reason = "user_dissatisfied"
                break

    # ── 组装最终输出 ──
    if not exit_reason:
        exit_reason = "max_turns"
    
    return {
        "session_id": session_id,
        "profile_id": profile["profile_id"],
        "query": query,
        "target_ad": {
            "ad_id": target_ad["ad_id"],
            "title": target_ad["title"],
            "bid": target_ad["bid"],
        } if target_ad else None,
        "peer_ad": {
            "ad_id": peer_ad["ad_id"],
            "title": peer_ad["title"],
            "bid": peer_ad["bid"],
        } if peer_ad else None,
        "candidate_ads": [
            {"ad_id": a["ad_id"], "title": a["title"], "bid": a["bid"]}
            for a in candidate_ads
        ] if candidate_ads else [],
        "auction_turn": actual_auction_turn,
        "exit_reason": exit_reason,
        "turns": turns,
        "n_turns": len(turns),
        # ── 调试用: 初始系统 prompt（含 profile 和候选广告）──
        "debug_prompts": {
            "user_system": user_system_prompt,
            "assistant_system": assistant_system_prompt,
        },
        # ── 可读的完整对话轨迹（用于人工核查数据质量）──
        "trajectory": format_trajectory(session_id, query, profile, candidate_ads, turns),
    }

# ========================================================================
# 阶段4: 反事实样本生成（独立于主对话流程，便于调试）
# 输入: dialogues.jsonl（已验证的真实对话轨迹）
# 输出: counterfactuals.jsonl
# 包含两类负样本:
#   counterfactual_ad  — SOFT 轮强制插入未被选中的候选广告
#   target_early       — WAIT 轮强制提前插入 target_ad
# ========================================================================

async def generate_counterfactuals_for_session(sess, profile, all_ads):
    # type: (dict, dict, List[dict]) -> List[dict]
    """为一个已完成的 session 生成所有反事实样本。"""
    ad_map = {a["ad_id"]: a for a in all_ads}
    query = sess["query"]
    candidate_ads = [ad_map[a["ad_id"]] for a in sess["candidate_ads"] if a["ad_id"] in ad_map]
    target_ad = ad_map.get(sess["target_ad"]["ad_id"]) if sess.get("target_ad") else None

    results = []

    # 重建对话历史（逐轮累积）
    dialogue = []
    for t_rec in sess["turns"]:
        user_msg = t_rec.get("user", "")
        asst_msg = t_rec.get("assistant", "")
        t = t_rec["turn"]

        if user_msg:
            dialogue.append({"role": "user", "content": user_msg})

        if t_rec["action"] == "SOFT":
            # ── 类型1: 强制插入未被选中的候选广告 ──
            inserted_ids = {a["ad_id"] for a in t_rec.get("ads_inserted", [])}
            other_candidates = [a for a in candidate_ads if a["ad_id"] not in inserted_ids]
            for cf_ad in other_candidates:
                cf_response = await generate_assistant_ad(profile, dialogue, [cf_ad])
                cf_pattern = r"@{}@\[{}\]".format(re.escape(cf_ad["title"]), re.escape(cf_ad["ad_id"]))
                if not re.search(cf_pattern, cf_response):
                    continue
                cf_dialogue = dialogue[:-1] + [{"role": "assistant", "content": cf_response}]
                cf_judge = await judge_ad_turn(profile, query, cf_dialogue, cf_response, turn_number=t)
                cf_clicked = False
                if cf_judge and cf_judge.get("ads"):
                    for aj in cf_judge["ads"]:
                        if aj.get("ad_id") == cf_ad["ad_id"]:
                            cf_clicked = aj.get("clicked", False)
                results.append({
                    "session_id": sess["session_id"],
                    "turn": t,
                    "sample_type": "counterfactual_ad",
                    "ad_id": cf_ad["ad_id"],
                    "title": cf_ad["title"],
                    "bid": cf_ad["bid"],
                    "response": cf_response,
                    "clicked": cf_clicked,
                    "gamma_score": cf_judge.get("gamma_score", 0.0) if cf_judge else 0.0,
                    "judge": cf_judge,
                    "trajectory": format_cf_trajectory(
                        query, profile, candidate_ads,
                        [r for r in sess["turns"] if r["turn"] < t],
                        user_msg, cf_response, cf_ad, cf_judge,
                    ),
                })

            if asst_msg:
                dialogue.append({"role": "assistant", "content": asst_msg})

        elif t_rec["action"] == "WAIT":
            if asst_msg:
                dialogue.append({"role": "assistant", "content": asst_msg})

            # ── 类型2: 提前强制插入 target_ad（early insertion 负样本）──
            if target_ad:
                early_response = await generate_assistant_ad(profile, dialogue, [target_ad])
                early_pattern = r"@{}@\[{}\]".format(
                    re.escape(target_ad["title"]), re.escape(target_ad["ad_id"])
                )
                if re.search(early_pattern, early_response):
                    early_dialogue = dialogue[:-1] + [{"role": "assistant", "content": early_response}]
                    early_judge = await judge_ad_turn_early(
                        profile, query, early_dialogue, early_response, insert_turn=t
                    )
                    early_clicked = False
                    if early_judge and early_judge.get("ads"):
                        for aj in early_judge["ads"]:
                            if aj.get("ad_id") == target_ad["ad_id"]:
                                early_clicked = aj.get("clicked", False)
                                break
                    results.append({
                        "session_id": sess["session_id"],
                        "turn": t,
                        "sample_type": "target_early",
                        "ad_id": target_ad["ad_id"],
                        "title": target_ad["title"],
                        "bid": target_ad["bid"],
                        "response": early_response,
                        "clicked": early_clicked,
                        "gamma_score": early_judge.get("gamma_score", 0.0) if early_judge else 0.0,
                        "judge": early_judge,
                        "trajectory": format_cf_trajectory(
                            query, profile, candidate_ads,
                            [r for r in sess["turns"] if r["turn"] < t],
                            user_msg, early_response, target_ad, early_judge,
                        ),
                    })

    return results


# ========================================================================
# 入口: 串联三个阶段，处理命令行参数，批量执行
# ========================================================================

async def main():
    # ── 解析命令行参数 ──
    parser = argparse.ArgumentParser(description="Generate multi-turn dialogues with ad auction")
    parser.add_argument("--dry-run", action="store_true",
                        help="只展示执行计划和样例 prompt，不实际调用 LLM API")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制总 session 数量 (0 = 不限制，跑全量)")
    parser.add_argument("--limit-profiles", type=int, default=0,
                        help="只处理前 N 个 profile (0 = 全部)")
    parser.add_argument("--max-turns", type=int, default=3,
                        help="每个 session 最大轮数 (0 = 使用 pipeline.json 中的配置)")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag suffix for output files (e.g., '5turn' → dialogues_5turn.jsonl)")
    parser.add_argument("--min-turns", type=int, default=0,
                        help="过滤掉轮数小于此值的 session (0 = 不过滤)")
    parser.add_argument("--skip-query-gen", action="store_true",
                        help="跳过阶段1(query生成)，直接加载已有的 queries.jsonl")
    parser.add_argument("--task-shard-size", type=int, default=0,
                        help="按任务分片大小切分 tasks；例如 1000 表示每 1000 个 task 一组 (0=不分片)")
    parser.add_argument("--task-shard-index", type=int, default=0,
                        help="要执行的分片编号（从 0 开始）。需与 --task-shard-size 配合使用")
    args = parser.parse_args()

    # ── 加载配置 ──
    seed = CFG["project"]["seed"]
    rng = random.Random(seed)  # 固定随机种子保证可复现
    max_turns = args.max_turns if args.max_turns > 0 else CFG["project"]["max_turns"]
    topics_per_profile = CFG["project"]["topics_per_profile"]  # 每个 profile 生成几个 query (默认5)

    data_dir = Path(__file__).resolve().parents[1] / "data"

    # ── 加载 profiles ──
    profiles_path = data_dir / "profiles_.jsonl"
    if not profiles_path.exists():
        print("[error] profiles.jsonl not found. Run 01_make_profiles.py first.")
        return
    profiles = list(read_jsonl(profiles_path))
    
    if args.limit_profiles > 0:
        profiles = profiles[:args.limit_profiles]
    print("Loaded {} profiles".format(len(profiles)))

    # ── 加载广告库 ──
    ads_path = data_dir / "ads_corpus_synthetic.jsonl"
    all_ads = list(read_jsonl(ads_path))
    print("Loaded {} ads".format(len(all_ads)))

    # 提取广告品类列表（去重），用于约束 query 生成范围
    ad_categories = sorted(set(a["category"] for a in all_ads))

    tag = args.tag
    tag_suffix = "_{}".format(tag) if tag else ""

    # ================================================================
    # 阶段1: 从 profile 生成 queries
    # ================================================================
    queries_path = data_dir / "queries{}.jsonl".format(tag_suffix) if tag else data_dir / "queries_427.jsonl"
    if queries_path.exists():
        # 如果文件已存在，直接加载
        print("Loading existing queries from {}".format(queries_path))
        query_records = list(read_jsonl(queries_path))
    else:
        print("Phase 1: Generating {} queries per profile...".format(topics_per_profile))
        if args.dry_run:
            # dry-run 模式: 只展示样例 prompt，不调 API
            print("[dry-run] Would generate queries for {} profiles".format(len(profiles)))
            print("[dry-run] Sample prompt:")
            sample_p = profiles[0]
            sample_targets = random.sample(all_ads, min(topics_per_profile, len(all_ads)))
            sample_products_text = ""
            for ii, ad in enumerate(sample_targets, 1):
                sample_products_text += "{}. [{}] {} — {}\n".format(
                    ii, ad["category"], ad["title"], ad.get("short_description", "")
                )
            print(USER_QUERY_GENERATION.format(
                basic_info=format_profile_basic_info(sample_p),
                personal_interests=format_profile_interests(sample_p),
                num_queries=topics_per_profile,
                target_products=sample_products_text,
            )[:800])
            print("...")
            return

        # 按批次并发调用 LLM 生成 queries
        query_records = []
        batch_size = CFG["rate_limit"]["max_concurrency"]
        for i in range(0, len(profiles), batch_size):
            batch = profiles[i:i+batch_size]
            # 每个 profile 并发生成 queries（传入全部广告库）
            coros = [generate_queries(p, topics_per_profile, all_ads=all_ads, rng=rng) for p in batch]
            results = await asyncio.gather(*coros, return_exceptions=True)
            for p, r in zip(batch, results):
                if isinstance(r, Exception):
                    print("[error] query gen for {}: {}".format(p["profile_id"], r))
                    continue
                # r 是 [(query_text, intent_type, target_ad, peer_ad), ...] 的列表
                for j, (q, intent_type, target_ad, peer_ad) in enumerate(r):
                    query_records.append({
                        "profile_id": p["profile_id"],
                        "query_idx": j,
                        "query": q,
                        "intent_type": intent_type,
                        "target_ad_id": target_ad["ad_id"],
                        "peer_ad_id": peer_ad["ad_id"],
                    })
            done = min(i + batch_size, len(profiles))
            print("  query gen progress: {}/{} profiles".format(done, len(profiles)))

        # 保存 queries 到文件，方便后续 --skip-query-gen 复用
        write_jsonl(queries_path, query_records)
        print("Saved {} queries -> {}".format(len(query_records), queries_path))

    # ================================================================
    # 阶段2+3: 多轮对话 + 拍卖 + Judge (合并执行)
    # ================================================================
    
    # 构建 (profile, query, session_id, target_ad) 任务列表
    profile_map = {p["profile_id"]: p for p in profiles}
    ad_map = {a["ad_id"]: a for a in all_ads}
    tasks = []  # type: List[Tuple[dict, str, str, Optional[dict], Optional[dict], str]]
    # import pdb;pdb.set_trace()
    for qr in query_records:
        pid = qr["profile_id"]
        if pid not in profile_map:
            continue
        sid = "sess_{}_q{:02d}".format(pid, qr["query_idx"])
        target_ad = ad_map.get(qr.get("target_ad_id")) if qr.get("target_ad_id") else None
        peer_ad = ad_map.get(qr.get("peer_ad_id")) if qr.get("peer_ad_id") else None
        intent_type = qr.get("intent_type", "PARTIAL")
        tasks.append((profile_map[pid], qr["query"], sid, target_ad, peer_ad, intent_type))

    if args.limit > 0:
        tasks = tasks[:args.limit]

    # ── 可选: 按 task 分片执行，便于多次运行脚本产出多份数据 ──
    shard_suffix = ""
    if args.task_shard_size > 0:
        if args.task_shard_index < 0:
            print("[error] --task-shard-index must be >= 0")
            return
        shard_start = args.task_shard_index * args.task_shard_size
        shard_end = min(shard_start + args.task_shard_size, len(tasks))
        if shard_start >= len(tasks):
            print("[error] shard start {} out of range for {} tasks".format(
                shard_start, len(tasks)
            ))
            return
        tasks = tasks[shard_start:shard_end]
        shard_suffix = "_shard{:04d}".format(args.task_shard_index)
        print("Using task shard {}: [{}:{}) -> {} tasks".format(
            args.task_shard_index, shard_start, shard_end, len(tasks)
        ))
    elif args.task_shard_index != 0:
        print("[warn] --task-shard-index is ignored because --task-shard-size is 0")

    print("\nPhase 2+3: Generating {} dialogue sessions (dynamic ad timing, max {} turns)".format(
        len(tasks), max_turns
    ))
    if args.dry_run:
        print("[dry-run] Would generate {} sessions".format(len(tasks)))
        if tasks:
            p, q, sid, tad, pad = tasks[0]
            print("[dry-run] Sample session: {} / query='{}'".format(sid, q[:60]))
            if tad:
                print("[dry-run] Target ad: {} | {} | bid=${}".format(tad["ad_id"], tad["title"], tad["bid"]))
            if pad:
                print("[dry-run] Peer ad:   {} | {} | bid=${}".format(pad["ad_id"], pad["title"], pad["bid"]))
        return

    # 批量并发执行 sessions
    # 注意: 每个 session 内部有多次 LLM 调用(多轮对话+judge)，
    # 所以 batch_size 要比 Phase 1 小，避免并发过高
    all_sessions = []
    batch_size = max(1, CFG["rate_limit"]["max_concurrency"] // 2)
    out_path = data_dir / "dialogues{}{}.jsonl".format(tag_suffix, shard_suffix)
    checkpoint_interval = 1000   # 每生成1000条保存一次
    last_checkpoint = 0          # 上次保存时的 session 数量
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        coros = [
            run_one_session(p, q, all_ads, sid, max_turns, rng, target_ad=tad, peer_ad=pad, intent_type=it)
            for p, q, sid, tad, pad, it in batch
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                print("[error] session: {}".format(r))
            else:
                all_sessions.append(r)
        done = min(i + batch_size, len(tasks))
        print("  dialogue progress: {}/{} sessions ({} completed)".format(
            done, len(tasks), len(all_sessions)
        ))
        # ── 每满 1000 条保存一次中间结果 ──
        current_count = len(all_sessions)
        if current_count // checkpoint_interval > last_checkpoint // checkpoint_interval:
            write_jsonl(out_path, all_sessions)
            last_checkpoint = current_count
            print("  [checkpoint] Saved {} sessions -> {}".format(current_count, out_path))

    # ── 过滤短对话 ──
    if args.min_turns > 0:
        before_filter = len(all_sessions)
        all_sessions = [s for s in all_sessions if s["n_turns"] >= args.min_turns]
        print("Filtered {}/{} sessions with < {} turns, kept {}".format(
            before_filter - len(all_sessions), before_filter, args.min_turns, len(all_sessions)))

    # ── 保存主输出: dialogues.jsonl ──
    write_jsonl(out_path, all_sessions)
    print("\nWrote {} sessions -> {}".format(len(all_sessions), out_path))

    # ══════════════════════════════════════════════════════════
    # 生成训练数据: sft_conversations.jsonl
    # 每条样本 = 一个完整对话 session 的 messages 数组
    # 格式: {"messages": [system, user, assistant, user, assistant, ...]}
    #   system prompt: 包含候选广告 + bid 信息
    #   WAIT 轮 assistant: 纯回复（不含广告）→ 模型学到 timing
    #   SOFT 轮 assistant: 带广告回复（含 @Ad@[ID]）→ 模型学到 selection + generation
    # 训练时对所有 assistant turn 计算 loss
    # ══════════════════════════════════════════════════════════
    sft_conversations = []
    for sess in all_sessions:
        target_ad_info = sess.get("target_ad")
        if not target_ad_info:
            continue

        cand_ads = sess.get("candidate_ads", [])
        if not cand_ads:
            continue

        # 构建 system prompt: 候选广告 + bid
        ads_text = "\n".join(
            "{}. {} (Ad ID: {}, Bid: ${:.2f})".format(i+1, a["title"], a["ad_id"], a["bid"])
            for i, a in enumerate(cand_ads)
        )
        system_content = (
            "You are a helpful AI shopping assistant.\n"
            "You have the following candidate ads available. "
            "You may insert AT MOST 1 ad into your response when the conversation "
            "naturally leads to a relevant product. Use the format @ad_title@[ad_id].\n"
            "If the conversation is not ready for an ad, respond normally without any ad.\n"
            "Higher-bid ads are preferred when relevance is similar.\n\n"
            "Candidate Ads:\n" + ads_text
        )

        messages = [{"role": "system", "content": system_content}]

        # 逐轮构建 messages
        # SFT 过滤标准: gamma >= 0.5（插入自然）且有 SOFT 轮
        # clicked 不作为过滤条件——插入方式正确但用户没点击的轨迹同样值得学习
        n_wait = 0
        n_soft = 0
        has_valid_turns = False
        gamma = None
        for t_rec in sess["turns"]:
            if t_rec["action"] not in ("WAIT", "SOFT"):
                continue
            if not t_rec.get("user") or not t_rec.get("assistant"):
                continue

            if t_rec["action"] == "SOFT":
                judge = t_rec.get("judge") or {}
                gamma = judge.get("gamma_score", 0.0)

            messages.append({"role": "user", "content": t_rec["user"]})
            messages.append({"role": "assistant", "content": t_rec["assistant"]})
            has_valid_turns = True

            if t_rec["action"] == "WAIT":
                n_wait += 1
            else:
                n_soft += 1

        if not has_valid_turns or n_soft == 0:
            continue
        if any(t_rec.get("title_leaked") for t_rec in sess["turns"]):
            continue  # WAIT 轮泄露产品名，污染了插入时机，丢弃
        if gamma is None or gamma < 0.5:
            continue  # 插入质量不达标，丢弃

        record = {
            "session_id": sess["session_id"],
            "profile_id": sess["profile_id"],
            "messages": messages,
            "metadata": {
                "candidate_ads": cand_ads,
                "target_ad": target_ad_info,
                "n_wait_turns": n_wait,
                "n_soft_turns": n_soft,
                "gamma_score": gamma,
                "clicked": any(
                    aj.get("clicked")
                    for t_rec in sess["turns"] if t_rec["action"] == "SOFT"
                    for aj in (t_rec.get("judge") or {}).get("ads", [])
                    if aj.get("ad_id") == target_ad_info["ad_id"]
                ),
            },
        }
        sft_conversations.append(record)

    sft_path = data_dir / "sft_conversations{}{}.jsonl".format(tag_suffix, shard_suffix)
    write_jsonl(sft_path, sft_conversations)
    total_wait = sum(r["metadata"]["n_wait_turns"] for r in sft_conversations)
    total_soft = sum(r["metadata"]["n_soft_turns"] for r in sft_conversations)
    print("Wrote {} conversation samples -> {} (total turns: {} WAIT + {} SOFT)".format(
        len(sft_conversations), sft_path, total_wait, total_soft,
    ))

    # ================================================================
    # 阶段4: 反事实样本生成（验证对话质量后再跑）
    # 读取已生成的 dialogues.jsonl，补充生成反事实负样本
    # 输出: counterfactuals.jsonl
    # ================================================================
    print("\nPhase 4: Generating counterfactual samples from {} sessions...".format(len(all_sessions)))
    all_cf = []
    batch_size_cf = max(1, CFG["rate_limit"]["max_concurrency"] // 4)
    for i in range(0, len(all_sessions), batch_size_cf):
        batch = all_sessions[i:i+batch_size_cf]
        coros = [
            generate_counterfactuals_for_session(sess, profile_map[sess["profile_id"]], all_ads)
            for sess in batch if sess["profile_id"] in profile_map
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                print("[error] counterfactual: {}".format(r))
            else:
                all_cf.extend(r)
        done = min(i + batch_size_cf, len(all_sessions))
        print("  counterfactual progress: {}/{} sessions ({} samples so far)".format(
            done, len(all_sessions), len(all_cf)
        ))

    cf_path = data_dir / "counterfactuals{}{}.jsonl".format(tag_suffix, shard_suffix)
    write_jsonl(cf_path, all_cf)
    n_cf_ad = sum(1 for r in all_cf if r["sample_type"] == "counterfactual_ad")
    n_cf_early = sum(1 for r in all_cf if r["sample_type"] == "target_early")
    print("Wrote {} counterfactual samples -> {} (counterfactual_ad: {}, target_early: {})".format(
        len(all_cf), cf_path, n_cf_ad, n_cf_early,
    ))


if __name__ == "__main__":
    asyncio.run(main())


    