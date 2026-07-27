"""LLM-OSDA 机制可视化 Demo（Streamlit · 中文版 · 支持自动播放）。

读取 outputs/case_studies/<sid>/meta.json（由实验 7.4 生成），把每一个
case 的机制计算过程逐步展开：
  pCTR → 虚拟价值 ψ → 加权得分 → Bellman 停时 → 复合阈值 Θ
  → 临界值定价 P → 信息租金

支持「▶ 自动播放」一键演示——每个步骤停留若干秒后自动展开下一步。
适合学术汇报现场演示。

启动:
  streamlit run src/demo/app.py -- --case_dir outputs/case_studies
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

# ---------------------------------------------------------------------------
# 机制计算公式（与 src/auction/auction.py 一致）
# ---------------------------------------------------------------------------

THETA_MAX_DEFAULT = 2.18
GAMMA_DEFAULT = 0.95
TOTAL_STEPS = 7


def virtual_value(bid: float, theta_max: float) -> float:
    """ψ(b) = 2b − θ_max （Uniform[θ_min, θ_max] 先验下的 Myerson 虚拟价值）。"""
    return 2.0 * bid - theta_max


def virtual_value_inv(y: float, theta_max: float) -> float:
    """ψ⁻¹(y) = (y + θ_max) / 2。"""
    return (y + theta_max) / 2.0


# ---------------------------------------------------------------------------
# CLI 解析（Streamlit 把 -- 之后的参数转给 sys.argv）
# ---------------------------------------------------------------------------

@st.cache_resource
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case_dir", default="outputs/case_studies")
    args, _ = p.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# 文件 IO
# ---------------------------------------------------------------------------

def list_cases(case_dir: Path) -> List[Path]:
    if not case_dir.exists():
        return []
    return sorted(p for p in case_dir.iterdir()
                  if p.is_dir() and (p / "meta.json").exists())


def load_meta(session_dir: Path) -> Dict[str, Any]:
    with open(session_dir / "meta.json") as f:
        return json.load(f)


def load_index(case_dir: Path) -> Dict[str, Any]:
    p = case_dir / "case_study_index.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _ad_short(title: str, n: int = 28) -> str:
    return (title or "").strip()[:n]


# ---------------------------------------------------------------------------
# 各步骤渲染函数
# ---------------------------------------------------------------------------

def render_one_turn(
    turn_data: Dict[str, Any],
    trace_entry: Dict[str, Any] | None,
    deco: Dict[str, Any],
    stop_turn: int,
    is_active: bool = True,
    dim: bool = False,
    highlight: bool = False,
) -> None:
    """渲染一轮：左侧对话（Q+A）右侧机制判断，并排展示。

    `dim=True` 时整轮变淡（opacity=0.35），用于灰化非当前聚焦轮。
    `highlight=True` 时整轮加发光边框，用于突出当前聚焦轮。
    """
    # 每个卡片单独添加 opacity/highlight 样式（因为 Streamlit column
    # 里的元素无法用外层 <div> 真正包裹）
    if dim:
        card_extra_style = "opacity:0.30; filter:grayscale(0.7); transition:all 0.4s;"
    else:
        card_extra_style = "opacity:1; transition:all 0.4s;"

    # 高亮的边框会替换默认边框颜色
    highlight_border_dialogue = "border:3px solid #FF9800; box-shadow:0 0 12px rgba(255,152,0,0.6);" if highlight else ""
    highlight_border_mech = "border:3px solid #FF9800; box-shadow:0 0 12px rgba(255,152,0,0.6);" if highlight else ""

    tn = int(turn_data.get("turn", 0))
    is_stop = (tn == stop_turn)
    user = turn_data.get("user_zh") or turn_data.get("user", "")
    assistant_raw = turn_data.get("assistant_zh") or turn_data.get("assistant", "")
    assistant = (assistant_raw or "").strip()

    # 顶部标题（跨整行）
    if highlight:
        badge_style = "margin:0 0 1px 0; font-weight:bold; font-size:1.05em; color:#E65100;"
    elif dim:
        badge_style = "margin:0 0 1px 0; font-weight:bold; font-size:1em; opacity:0.5;"
    else:
        badge_style = "margin:0 0 1px 0; font-weight:bold; font-size:1em;"
    badge = "🟢 停时所在轮 · 此处插入广告" if is_stop else "对话进行中"
    if highlight:
        badge = "▶ " + badge  # 三角标记当前聚焦
    st.markdown(f"<p style='{badge_style}'>{badge}</p>", unsafe_allow_html=True)

    # 左右两栏并排：左 = 对话；右 = 机制判断
    col_dialogue, col_mech = st.columns([1, 1], gap="small")

    # ---- 左：对话 ----
    with col_dialogue:
        user_bubble = (
            f"<div style='display:flex; justify-content:flex-end; margin-bottom:4px;'>"
            f"<div style='background-color:#95EC69; padding:5px 10px; "
            f"border-radius:10px 10px 0 10px; max-width:85%; color:#1a1a1a; "
            f"line-height:1.2; font-size:0.88em; "
            f"box-shadow:0 1px 2px rgba(0,0,0,0.08);'>{user}</div></div>"
        )
        assistant_bubble = ""
        if assistant:
            assistant_bubble = (
                f"<div style='display:flex; justify-content:flex-start; margin-bottom:0;'>"
                f"<div style='background-color:#FFFFFF; padding:5px 10px; "
                f"border-radius:10px 10px 10px 0; max-width:85%; color:#1a1a1a; "
                f"line-height:1.2; font-size:0.88em; "
                f"border:1px solid #E0E0E0; box-shadow:0 1px 2px rgba(0,0,0,0.05); "
                f"white-space:pre-wrap;'>{assistant}</div></div>"
            )
        default_border = "border:2px solid #1976D2;"
        border_style = highlight_border_dialogue if highlight else default_border
        st.markdown(
            f"""
<div style="background-color:#F5F9FF; padding:8px 10px; border-radius:8px;
            {border_style} color:#0D47A1; {card_extra_style}">
  {user_bubble}
  {assistant_bubble}
</div>
""",
            unsafe_allow_html=True,
        )

    # ---- 右：机制判断 ----
    with col_mech:
        if trace_entry is None:
            st.caption("（这一轮没有机制 trace 数据）")
            return

        w = float(trace_entry.get("q_auction_w", 0.0))
        cv = float(trace_entry.get("q_wait_cv", 0.0))
        pctrs = trace_entry.get("all_pctrs") or []
        titles = (deco.get("all_titles_zh")
                  or deco.get("all_titles")
                  or [])[:len(pctrs)]
        local_winner = int(trace_entry.get("winner_index", -1))

        # 候选广告表格：广告 / 报价 / 点击率（三列，无标记列）
        bids_full = (deco.get("all_bids") or [])
        table_rows = []
        for i, q in enumerate(pctrs):
            ttl = titles[i] if i < len(titles) else f"广告 {i}"
            bid_val = bids_full[i] if i < len(bids_full) else None
            bid_cell = f"${bid_val:.2f}" if bid_val is not None else "—"
            # 只有停时轮才标赢家——等待轮机制尚未决定插谁，不提前高亮
            row_style = "font-weight:bold; color:#BF360C;" if (is_stop and i == local_winner) else ""
            table_rows.append(
                f"<tr style='border-bottom:1px solid #FFE082; {row_style}'>"
                f"<td style='padding:3px 6px;'>{ttl}</td>"
                f"<td style='padding:3px 6px;'>{bid_cell}</td>"
                f"<td style='padding:3px 6px;'>{q:.3f}</td></tr>"
            )
        table_html = (
            "<table style='width:100%; border-collapse:collapse; font-size:0.9em;'>"
            "<tr style='border-bottom:2px solid #F9A825;'>"
            "<th style='text-align:left; padding:3px 6px;'>广告</th>"
            "<th style='text-align:left; padding:3px 6px;'>报价</th>"
            "<th style='text-align:left; padding:3px 6px;'>点击率</th></tr>"
            + "".join(table_rows)
            + "</table>"
        )

        if is_stop:
            verdict = "🔴 <b>停！插广告</b>"
        elif w >= cv:
            verdict = "⚫ 终端强制停"
        else:
            verdict = "🟡 等待"
        decision_html = f"<p>⚖️ 立即拍卖价值 <b>{w:.3f}</b> vs 继续等待价值 <b>{cv:.3f}</b> → {verdict}</p>"

        winner_html = ""
        if is_stop:
            winner_title = (deco.get("winner_title_zh") or deco.get("winner_title") or "—")
            bid = float(deco.get("bid", 0.0))
            payment = float(deco.get("payment", 0.0))
            rent = float(deco.get("rent", 0.0))
            winner_html = (
                f"<p>🏆 <b>赢家：</b>{winner_title}<br>"
                f"报价 ${bid:.2f} · 实付 ${payment:.3f} · 租金 ${rent:.4f}</p>"
            )

        default_border_mech = "border:2px solid #F9A825;"
        border_style_mech = highlight_border_mech if highlight else default_border_mech
        st.markdown(
            f"""
<div style="background-color:#FFF8E1; padding:4px 8px; border-radius:6px;
            {border_style_mech} color:#4E342E; line-height:1.2; font-size:0.88em; {card_extra_style}">
  <div style="font-weight:bold; font-size:0.95em; color:#E65100; margin-bottom:2px;">🧠 机制判断</div>
  {table_html}
  <div style="margin-top:3px;">{decision_html}</div>
  {winner_html}
</div>
""",
            unsafe_allow_html=True,
        )


def _turn_insight(turn: int, w: float, cv: float, is_stop: bool,
                  pctrs: List[float], local_winner: int) -> str:
    """按机制行为给该轮一句话点评。"""
    max_pctr = max(pctrs) if pctrs else 0.0
    if is_stop:
        if max_pctr >= 0.5:
            return "用户意图变得明朗，赢家点击率显著提升，机制选择此时插入广告。"
        else:
            return "立即拍卖价值反超等待价值，机制触发拍卖。"
    if w < cv:
        # 等待
        if max_pctr < 0.2:
            return "意图仍模糊，所有候选广告点击率偏低，机制选择继续观察。"
        else:
            return "已有候选广告点击率不错，但等待未来更优匹配的折现期权价值更高，先等等。"
    if w >= cv:
        # 这一轮 W>=CV 但又不是 session 的 stop_turn，意味着已被前面的某轮抢先停过
        return "已超过停时所在轮，本轮决策仅作反事实展示。临近终止时等待价值趋于零（定理 4：终端扭曲消散）。"
    return ""


def render_dialogue(meta: Dict[str, Any], focus_turn: int = -1) -> None:
    """一次性渲染全部对话轮。每一轮独立一个卡组：Q+A（左）+ 机制判断（右）。
    `focus_turn`: 高亮聚焦的轮号（>=1）。-1 表示不做灰化，全部按普通样式显示。
    """
    turns = meta.get("dialogue_turns", [])
    trace = meta.get("trace") or []
    deco = meta.get("decoupling") or {}
    stop_turn = int(meta.get("stop_turn", -1))

    if not turns:
        st.info(meta.get("user_query", "(meta.json 中没有对话内容)"))
        return

    trace_by_turn = {int(e.get("turn", -1)): e for e in trace}

    for i, t in enumerate(turns):
        tn = int(t.get("turn", i + 1))
        if i > 0:
            st.markdown(
                "<hr style='margin:6px 0; border:none; border-top:1px dashed #ccc;'>",
                unsafe_allow_html=True,
            )
        # 判断这一轮的显示模式
        if focus_turn > 0:
            is_focus = (tn == focus_turn)
            dim_flag = not is_focus
            highlight_flag = is_focus
        else:
            dim_flag = False
            highlight_flag = False
        render_one_turn(
            turn_data=t,
            trace_entry=trace_by_turn.get(tn),
            deco=deco,
            stop_turn=stop_turn,
            dim=dim_flag,
            highlight=highlight_flag,
        )


def render_step1_pctr(meta: Dict[str, Any]) -> None:
    """步骤 1：停时点的 pCTR 打分。"""
    deco = meta.get("decoupling") or {}
    pctrs = deco.get("all_pctrs") or []
    titles = (deco.get("all_titles_zh")
              or deco.get("all_titles")
              or [f"广告 {i}" for i in range(len(pctrs))])
    winner = int(deco.get("winner_index", -1))

    st.markdown("**步骤 1 — 各候选广告的点击率（pCTR）打分**")
    st.caption("由「不感知报价」的 pCTR 模型，根据当前对话历史给每个候选广告打出一个 0~1 的点击率预测值。")
    rows = []
    for i, (q, ttl) in enumerate(zip(pctrs, titles)):
        marker = "★ 赢家" if i == winner else ""
        rows.append(f"| `{_ad_short(ttl)}` | {q:.4f} | {marker} |")
    st.markdown("| 广告 | 点击率 | 标记 |\n|---|---|---|\n" + "\n".join(rows))


def render_step2_virtual(meta: Dict[str, Any], theta_max: float) -> List[float]:
    """步骤 2：报价直接进入分配（identity 模式，与训练一致）。"""
    deco = meta.get("decoupling") or {}
    bids = deco.get("all_bids") or []
    titles = (deco.get("all_titles_zh") or deco.get("all_titles") or [])
    winner = int(deco.get("winner_index", -1))

    st.markdown("**步骤 2 — 取报价用于后续打分（identity 模式）**")
    st.caption(
        "本机制采用 identity psi 模式（与训练保持一致）：把每个广告的报价直接作为后续加权得分的乘数。"
        "高报价提升排序地位，但是否中标还需要乘上点击率（步骤 3）。"
    )
    rows = []
    for i, (b, ttl) in enumerate(zip(bids, titles)):
        marker = "★" if i == winner else ""
        rows.append(f"| `{_ad_short(ttl)}` | ${b:.2f}$ | {marker} |")
    st.markdown("| 广告 | 报价 | 标记 |\n|---|---|---|\n" + "\n".join(rows))
    return list(bids)  # identity 模式下 psi(b) = b


def render_step3_scores(meta: Dict[str, Any], psi: List[float]) -> None:
    """步骤 3：质量加权得分 b·q（identity 模式）。"""
    deco = meta.get("decoupling") or {}
    pctrs = deco.get("all_pctrs") or []
    titles = (deco.get("all_titles_zh") or deco.get("all_titles") or [])
    scores_stored = deco.get("all_scores") or []
    winner = int(deco.get("winner_index", -1))

    st.markdown("**步骤 3 — 计算「报价 × 点击率」选出赢家**")
    st.caption(
        "分配规则：把每个广告的报价乘以它的点击率，得到「质量加权得分」，得分最高者获胜。"
        "这一步可以直观看到——出价最高的广告未必是赢家：哪怕出价高，"
        "如果点击率不行（即跟当前对话不太相关），整体得分也会被拉下来。"
    )
    rows = []
    for i, (p, q, ttl) in enumerate(zip(psi, pctrs, titles)):
        s = p * q
        if i < len(scores_stored) and abs(s - scores_stored[i]) > 1e-3:
            extra = f"（存档值 {scores_stored[i]:.3f}）"
        else:
            extra = ""
        marker = "★ 赢家" if i == winner else ""
        rows.append(
            f"| `{_ad_short(ttl)}` | ${p:.3f}$ | {q:.3f} | {s:.3f}{extra} | {marker} |"
        )
    st.markdown("| 广告 | 报价 | 点击率 | 加权得分 | 标记 |\n|---|---|---|---|---|\n" + "\n".join(rows))


def render_step4_bellman(meta: Dict[str, Any]) -> None:
    """步骤 4：Bellman 最优停时决策。"""
    trace = meta.get("trace") or []
    stop_turn = int(meta.get("stop_turn", -1))
    st.markdown("**步骤 4 — 决定「现在插还是再等一轮」**")
    st.caption(
        "每一轮，机制都会比较两个值：「立即拍卖能拿到的价值」 vs 「再等一轮的折现期权价值」。"
        "前者大就停、就插广告；后者大就继续等。这是经典最优停时问题，也是论文相对 GSP / 单轮拍卖的最大区别。"
    )
    rows = []
    for entry in trace:
        tn = int(entry["turn"])
        w = float(entry["q_auction_w"])
        cv = float(entry["q_wait_cv"])
        if w >= cv and tn == stop_turn:
            decision = "**停（插广告）** ★"
        elif w >= cv:
            decision = "停"
        else:
            decision = "等待"
        marker = "← 选定的停时" if tn == stop_turn else ""
        rows.append(f"| 第 {tn} 轮 | {w:.3f} | {cv:.3f} | {decision} | {marker} |")
    st.markdown(
        "| 轮次 | 立即拍卖价值 | 继续等待价值 | 决策 | 标记 |\n|---|---|---|---|---|\n"
        + "\n".join(rows)
    )


def render_step5_threshold(meta: Dict[str, Any], gamma: float) -> None:
    """步骤 5：复合阈值。"""
    deco = meta.get("decoupling") or {}
    theta = float(deco.get("theta_threshold", 0.0))
    cv_minus = float(deco.get("cv_minus_winner", 0.0))
    scores = deco.get("all_scores") or []
    winner = int(deco.get("winner_index", -1))
    lateral_max = max([s for i, s in enumerate(scores) if i != winner] + [0.0])
    inter = gamma * cv_minus
    binding = "横向竞争" if lateral_max >= inter else "纵向跨期"

    st.markdown("**步骤 5 — 算出赢家必须击穿的「双重门槛」**")
    st.caption(
        "赢家的实付不是看自己出多少，而是看必须比谁强。机制给出两个门槛："
        "**横向**——其他广告主中加权得分最高的那个；"
        "**纵向**——如果不让赢家中标，平台等到未来更优匹配的反事实期望收益（按折现因子折现）。"
        "最终阈值取两者最大值。**纵向项是论文相对其它工作的最大结构性创新**。"
    )
    st.markdown(
        f"""
| 分量 | 数值 | 含义 |
|---|---|---|
| 横向门槛 | **{lateral_max:.3f}** | 其他广告主中加权得分最高的那个（步骤 3 的非赢家最大值）|
| 纵向门槛 | **{inter:.3f}** | 「等到未来更优匹配」的反事实折现收益 |
| **最终阈值** | **{theta:.3f}** | **{binding}起决定作用** |
"""
    )


def render_step6_payment(meta: Dict[str, Any], theta_max: float) -> None:
    """步骤 6：临界值定价（identity 模式：P = Θ / q_winner，封顶在报价）。"""
    deco = meta.get("decoupling") or {}
    theta = float(deco.get("theta_threshold", 0.0))
    pctrs = deco.get("all_pctrs") or []
    bids = deco.get("all_bids") or []
    winner = int(deco.get("winner_index", -1))
    payment = float(deco.get("payment", 0.0))
    if winner < 0 or winner >= len(pctrs):
        st.warning("赢家索引越界，无法计算定价。")
        return
    q_w = pctrs[winner]
    b_w = bids[winner]
    raw_p = theta / max(q_w, 1e-9)
    capped = "（未触及报价上限）" if raw_p <= b_w else f"（封顶在报价 ${b_w:.3f}）"
    st.markdown("**步骤 6 — 把阈值反推回赢家应付的真实价格**")
    st.caption(
        "赢家实际付的价格是「能让其刚好仍能获胜的最低报价」。这就是激励相容的几何根源——"
        "**赢家具体出多少不影响实付，只影响能不能赢**，因此诚实报价是最优策略。"
    )
    st.markdown(
        f"""
| 分量 | 数值 | 含义 |
|---|---|---|
| 阈值 ÷ 赢家点击率 | {raw_p:.3f} | 把阈值除以赢家点击率，得到「让其刚好赢」的报价水平 |
| **赢家实付** | **${payment:.3f}** | {capped} |
"""
    )


def render_step7_rent(meta: Dict[str, Any]) -> None:
    """步骤 7：信息租金。"""
    deco = meta.get("decoupling") or {}
    bid = float(deco.get("bid", 0.0))
    payment = float(deco.get("payment", 0.0))
    rent = float(deco.get("rent", 0.0))
    pctrs = deco.get("all_pctrs") or []
    winner = int(deco.get("winner_index", -1))
    q_w = pctrs[winner] if 0 <= winner < len(pctrs) else 0.0
    st.markdown("**步骤 7 — 算出赢家的「信息租金」**")
    st.caption(
        "赢家因诚实报价而拿到的预期净收益 = （报价 − 实付）× 点击率。"
        "这一步收尾——报价和实付完全解耦：报价决定能否赢，实付决定付多少，"
        "二者各自独立，激励相容（DSIC）由此自然成立。"
    )
    st.markdown(
        f"""
| 分量 | 数值 | 含义 |
|---|---|---|
| 报价 | **${bid:.3f}** | 广告主一次性申报的真实价值 |
| 实付 | ${payment:.3f} | 步骤 6 算出的临界值定价 |
| 点击率 | {q_w:.3f} | 步骤 1 的赢家点击率 |
| **信息租金** | **${rent:.4f}** | （报价 − 实付）× 点击率 |
"""
    )


def render_figures(session_dir: Path) -> None:
    fig1 = session_dir / "fig1_decision.png"
    fig3 = session_dir / "fig3_decoupling.png"
    if fig1.exists() or fig3.exists():
        col1, col2 = st.columns(2)
        with col1:
            if fig1.exists():
                st.image(str(fig1), caption="决策链：立即拍卖价值 vs 继续等待价值 + 赢家点击率走势", use_container_width=True)
        with col2:
            if fig3.exists():
                st.image(str(fig3), caption="报价 vs 实付 解耦与信息租金", use_container_width=True)


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

STEP_TITLES = [
    "步骤 1 · pCTR 打分",
    "步骤 2 · 虚拟价值",
    "步骤 3 · 加权得分",
    "步骤 4 · Bellman 停时",
    "步骤 5 · 复合阈值",
    "步骤 6 · 临界值定价",
    "步骤 7 · 信息租金",
]


def render_steps_progressive(meta: Dict[str, Any], theta_max: float, gamma: float, current_step: int) -> None:
    """根据 current_step 渐进显示步骤 1..current_step。"""
    psi: List[float] = []
    if current_step >= 1:
        render_step1_pctr(meta)
        st.divider()
    if current_step >= 2:
        psi = render_step2_virtual(meta, theta_max)
        st.divider()
    else:
        # 即便没显示步骤 2，后续步骤可能用到 psi 列表（步骤 3 需要）
        pass
    if current_step >= 3:
        if not psi:  # 防御：如果直接跳到步骤 3 但 psi 没算
            deco = meta.get("decoupling") or {}
            bids = deco.get("all_bids") or []
            psi = [virtual_value(b, theta_max) for b in bids]
        render_step3_scores(meta, psi)
        st.divider()
    if current_step >= 4:
        render_step4_bellman(meta)
        st.divider()
    if current_step >= 5:
        render_step5_threshold(meta, gamma)
        st.divider()
    if current_step >= 6:
        render_step6_payment(meta, theta_max)
        st.divider()
    if current_step >= 7:
        render_step7_rent(meta)


def main() -> None:
    st.set_page_config(
        page_title="LLM-OSDA 机制可视化 Demo",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # 全局 CSS：压缩 Streamlit 自带的 block 间距 + 侧边栏宽度
    st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    div[data-testid="stVerticalBlock"] > div { margin-bottom: 0rem; padding-bottom: 0rem; }
    div[data-testid="column"] > div { padding: 0 0.3rem; }
    .stMarkdown { margin-bottom: 0; }
    .stDivider { margin: 4px 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stRadio > div { gap: 0.3rem; }
    p { margin-bottom: 0.2rem; }
    /* 侧边栏宽度缩窄 */
    section[data-testid="stSidebar"] { width: 200px !important; min-width: 200px !important; }
    section[data-testid="stSidebar"] > div { width: 200px !important; }
    /* 侧边栏 metric 字号缩小 */
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 0.85rem !important; }
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] { font-size: 0.7rem !important; }
</style>
""", unsafe_allow_html=True)
    args = parse_args()
    case_dir = Path(args.case_dir)

    cases = list_cases(case_dir)
    if not cases:
        st.title("LLM-OSDA 机制可视化 Demo")
        st.error(f"在 `{case_dir}` 中没有找到 case。请先运行 `bash run_exp7_4_case_study.sh` 生成案例数据。")
        return

    index = load_index(case_dir)
    theta_max = float(index.get("theta_max", THETA_MAX_DEFAULT))
    gamma = float(index.get("gamma", GAMMA_DEFAULT))

    # ---- 侧边栏：选择 case + 超参 + 头条数据 ----
    with st.sidebar:
        st.header("📂 选择案例")
        names = [c.name for c in cases]
        chosen = st.radio("案例", names, index=0, label_visibility="collapsed", key="case_picker")
        chosen_dir = case_dir / chosen
        meta = load_meta(chosen_dir)

        st.divider()
        st.markdown("**⚙️ 超参数**")
        st.caption(f"出价上限先验 = {theta_max}")
        st.caption(f"折现因子 = {gamma}")

        st.divider()
        st.markdown("**🎯 案例摘要**")
        deco = meta.get("decoupling") or {}
        st.metric("停时所在轮", meta.get("stop_turn", "—"))
        winner_title = (deco.get("winner_title_zh")
                        or deco.get("winner_title")
                        or "—")[:30]
        st.metric("赢家广告", winner_title)
        st.metric("报价", f"${deco.get('bid', 0):.2f}")
        st.metric("实付", f"${deco.get('payment', 0):.3f}")
        st.metric("信息租金", f"${deco.get('rent', 0):.4f}")

    # ---- 切换案例时重置自动播放状态 ----
    if st.session_state.get("active_case") != chosen:
        st.session_state.active_case = chosen
        st.session_state.current_step = TOTAL_STEPS  # 默认全展开
        st.session_state.autoplay = False
        # 对话默认聚焦停时轮
        st.session_state.current_turn = int(meta.get("stop_turn", 1))
        st.session_state.autoplay_dialogue = False

    # 兜底初始化
    if "current_step" not in st.session_state:
        st.session_state.current_step = TOTAL_STEPS
    if "autoplay" not in st.session_state:
        st.session_state.autoplay = False
    if "current_turn" not in st.session_state:
        st.session_state.current_turn = int(meta.get("stop_turn", 1))
    if "autoplay_dialogue" not in st.session_state:
        st.session_state.autoplay_dialogue = False

    # ---- 主面板 ----
    st.title("LLM-OSDA 机制可视化 Demo")
    st.caption(f"案例：`{chosen}` · 总轮数 = {meta.get('n_turns', '?')} · 停时所在轮 = 第 {meta.get('stop_turn', '?')} 轮")

    tab_dialogue, tab_steps, tab_figs = st.tabs(
        ["💬 对话回放", "🧮 机制计算逐步演示", "📊 已生成的图"]
    )

    with tab_dialogue:
        st.markdown(
            "<p style='margin:0 0 2px 0; font-size:1.1em; font-weight:bold;'>"
            "完整对话轨迹 + 每轮机制判断</p>"
            "<p style='margin:0 0 4px 0; font-size:0.82em; color:#666;'>"
            "全部轮次一次性并列展示。可选「▶ 自动聚焦」按钮：从第 1 轮起逐轮高亮，"
            "其它轮变灰，聚光灯效果。停时所在轮高亮 🟢 并显示赢家与实付。</p>",
            unsafe_allow_html=True,
        )

        n_turns = len(meta.get("dialogue_turns", []))
        # ---- 自动聚焦控制条 ----
        d_col1, d_col2, d_col3, d_col4 = st.columns([1, 1, 1, 2])
        with d_col1:
            if st.button(
                "▶ 自动聚焦",
                use_container_width=True,
                disabled=st.session_state.autoplay_dialogue,
                key="dialogue_play",
            ):
                st.session_state.autoplay_dialogue = True
                st.session_state.current_turn = 1
                st.rerun()
        with d_col2:
            if st.button(
                "⏸ 暂停",
                use_container_width=True,
                disabled=not st.session_state.autoplay_dialogue,
                key="dialogue_pause",
            ):
                st.session_state.autoplay_dialogue = False
                st.rerun()
        with d_col3:
            if st.button("🔄 全部展开", use_container_width=True, key="dialogue_reset"):
                st.session_state.autoplay_dialogue = False
                st.session_state.current_turn = -1  # -1 = 不做灰化
                st.rerun()
        with d_col4:
            dialogue_delay = st.slider(
                "每轮聚焦停留秒数",
                1.0, 30.0, 4.0, 0.5,
                key="dialogue_delay",
            )

        focus_turn = st.session_state.get("current_turn", -1)
        render_dialogue(meta, focus_turn=focus_turn)

        # 自动播放推进
        if st.session_state.autoplay_dialogue:
            if st.session_state.current_turn < n_turns:
                time.sleep(dialogue_delay)
                st.session_state.current_turn += 1
                st.rerun()
            else:
                st.session_state.autoplay_dialogue = False

    with tab_steps:
        st.markdown("### 机制计算逐步拆解")
        st.caption(
            "下面所有数字都直接来自该案例的 `meta.json`，公式与 `src/auction/auction.py` 完全一致。"
        )

        # ---- 控制条：自动播放 + 手动跳转 ----
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 2])
        with ctrl_col1:
            if st.button("▶ 自动播放", use_container_width=True, disabled=st.session_state.autoplay):
                st.session_state.autoplay = True
                st.session_state.current_step = 0
                st.rerun()
        with ctrl_col2:
            if st.button("⏸ 暂停", use_container_width=True, disabled=not st.session_state.autoplay):
                st.session_state.autoplay = False
                st.rerun()
        with ctrl_col3:
            if st.button("🔄 重置", use_container_width=True):
                st.session_state.autoplay = False
                st.session_state.current_step = TOTAL_STEPS
                st.rerun()
        with ctrl_col4:
            delay = st.slider("每步停留秒数", 1.0, 30.0, 2.5, 0.5, key="autoplay_delay")

        st.session_state.current_step = st.slider(
            "当前展开到步骤",
            min_value=0,
            max_value=TOTAL_STEPS,
            value=st.session_state.current_step,
            step=1,
            key="step_slider",
            help="拖动可手动控制；按「自动播放」会从步骤 1 自动走到步骤 7。",
        )

        # 进度条
        st.progress(
            st.session_state.current_step / TOTAL_STEPS,
            text=(
                f"进度：{st.session_state.current_step} / {TOTAL_STEPS}  "
                + (f"· 当前 → {STEP_TITLES[st.session_state.current_step - 1]}"
                   if st.session_state.current_step >= 1 else "· 等待开始")
            ),
        )
        st.divider()

        # ---- 渲染步骤 ----
        if st.session_state.current_step == 0:
            st.info("👆 点击「自动播放」开始演示，或拖动滑块手动控制。")
        else:
            render_steps_progressive(meta, theta_max, gamma, st.session_state.current_step)

        # ---- 自动播放推进逻辑 ----
        # Streamlit 的脚本是从上到下重新执行的，借助 sleep + rerun 实现推进。
        if st.session_state.autoplay and st.session_state.current_step < TOTAL_STEPS:
            time.sleep(delay)
            st.session_state.current_step += 1
            st.rerun()
        elif st.session_state.autoplay and st.session_state.current_step >= TOTAL_STEPS:
            # 走到最后一步后自动停止播放
            st.session_state.autoplay = False
            st.balloons()  # 小彩蛋

    with tab_figs:
        render_figures(chosen_dir)


if __name__ == "__main__":
    main()
