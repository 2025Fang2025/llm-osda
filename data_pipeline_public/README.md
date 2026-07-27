# Multi-turn Auction Data Pipeline (JSON)

目标：构建可训练数据（对话轨迹、反事实样本、LLM judge 标签），用于多轮对话广告拍卖机制实验。

## 目录
- `configs/` 配置（不写入密钥）
- `data/` 原始/中间数据（ads corpus, profiles, traj states）
- `outputs/` 运行产物（counterfactual + labeled）
- `scripts/` 生成脚本

## 产物（JSONL）
1. `data/ads_corpus.jsonl`
2. `data/profiles.jsonl`
3. `data/traj_states.jsonl`（纯净对话切片 H_t）
4. `outputs/counterfactual_samples.jsonl`（WAIT/SOFT/HARD）
5. `outputs/labeled_env.jsonl`（judge 输出 pctr/gamma）

## 运行顺序
1) `python scripts/01_make_profiles.py`
2) `python scripts/02_make_ads_corpus.py`（或导入你已有 ads 数据）
3) `python scripts/03_generate_pure_dialogues.py`
4) `python scripts/04_expand_counterfactuals.py`
5) `python scripts/05_judge_label.py`

## 密钥与网关
请通过环境变量提供（不要写进文件）：
- OpenAI: `OPENAI_API_KEY`, `OPENAI_BASE_URL`(可选)
- Anthropic: `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`(可选)

模型名在 `configs/pipeline.json` 中配置。
