# LLM-OSDA: Optimal-Stopping Dynamic Auction for Native Advertising

Reference implementation of **LLM-OSDA**, a dynamic auction for native advertising
in multi-turn LLM conversations. The pipeline couples a bid-independent LLM layer
with a committed mechanism (Bellman stopping, winner allocation, envelope CPC):

```text
user query / dialogue history
  → quality model (G_eta)       predicts click quality per ad  (bid-independent)
  → StopNet                     compares stop vs. wait action values
  → auction engine              selects the winner and computes the envelope CPC
  → renderer LLM (pi_R)         weaves the ad into the response  (bid-independent)
  → user-simulation judge       scores click and retention (does not see the bid)
  → net revenue = payment × judged click
```

---

## 1. Data

This release does **not** ship the simulated corpus. Generate it with the
data pipeline (`data_pipeline_public/`), or supply your own JSONL files in the
same schema:

```text
data/dialogues_all.jsonl          multi-turn dialogues with candidate ads and outcomes
data/counterfactuals_all.jsonl    (history, ad, response) -> click / retention labels
data/sft_all.jsonl                supervised trajectories for the renderer
```

| File | Role |
|---|---|
| `dialogues` | dialogue trajectories, candidate ads, and turns; used to build stopping states |
| `counterfactuals` | click / retention labels for training the quality model and judge |
| `sft` | supervised targets for the ad-insertion renderer (cleaned to drop bids before training) |

Bids stored in the data are synthetic (not real commercial bids). The quality
model and renderer never see bids at training time; the cleaning step strips the
candidate list and bids, leaving only the selected product's title and id.

---

## 2. Environment

```bash
pip install -r requirements.txt
```

Set the LLM gateway credentials in `data_pipeline_public/configs/pipeline.json`
(`base_url`, `api_key`) before running data generation. Any
OpenAI/Anthropic-compatible endpoint works.

---

## 3. Train and evaluate

### 3.1 Proxy run (no HuggingFace LLM, CPU-friendly)

```bash
python -m src.run_final_experiment \
  --dialogues data/dialogues_all.jsonl \
  --counterfactuals data/counterfactuals_all.jsonl \
  --sft data/sft_all.jsonl \
  --output_dir outputs/proxy \
  --eval_with_proxy_renderer \
  --pctr_epochs 5 --judge_epochs 5 --stop_epochs 30 \
  --max_eval_sessions 100 --include_ablation
```

### 3.2 Full run with the LLM renderer (GPU)

```bash
python -m src.run_final_experiment \
  --mode all \
  --dialogues data/dialogues_all.jsonl \
  --counterfactuals data/counterfactuals_all.jsonl \
  --sft data/sft_all.jsonl \
  --output_dir outputs/full \
  --base_model Qwen/Qwen3-4B \
  --pctr_epochs 5 --judge_epochs 5 --stop_epochs 30 --patience 1 \
  --lr 1e-3 --batch_size 256 --discount 0.95 --lambda_gamma 0.25 \
  --llm_epochs 1 --llm_lr 2e-5 --llm_batch_size 4 --llm_grad_accum 32 \
  --llm_max_length 8192 --llm_max_new_tokens 128 \
  --train_rounds 2 --eval_rounds 2 --use_vllm --include_ablation
```

### 3.3 Evaluate only (reuse trained models)

```bash
python -m src.run_final_experiment --mode eval \
  --model_dir outputs/full/round_2 \
  --dialogues data/dialogues_all.jsonl \
  --counterfactuals data/counterfactuals_all.jsonl \
  --sft data/sft_all.jsonl \
  --output_dir outputs/full --eval_rounds 1
```

---

## 4. Baselines and the adaptive myopic baseline

All timing policies share the same quality model, allocation, and envelope
payment; they differ only in the stopping rule.

| Policy | Stopping rule |
|---|---|
| `first_turn` | always insert at turn 1 |
| `last_turn` | always insert at the last turn |
| `random_turn` | insert at a random turn |
| `myopic_threshold` | greedy: insert when `b_i * q_i` exceeds a threshold (no look-ahead) |
| `learned_stop` | **LLM-OSDA**: StopNet compares stop vs. wait action values |

The myopic threshold baseline (`run_myopic_baseline.sh`) sweeps the threshold and
compares against the learned StopNet, isolating the value of look-ahead over pure
adaptivity.

---

## 5. Metrics

| Metric | Meaning |
|---|---|
| `avg_revenue` | net revenue: `payment × judged discounted click` |
| `avg_reward` | `net revenue - λ(1 - gamma)`, λ = 0.25 |
| `click_rate` | judged click probability |
| `avg_gamma` | judged retention / trust |
| `avg_trigger_turn` | average insertion turn |
| `avg_payment` | envelope CPC actually charged |

Click and retention are scored by the independent judge, not by the quality
model, so the reported revenue is not a self-evaluation.

---

## 6. Pipeline steps (`run_final_experiment.py`)

| Step | Content | Output |
|------|---------|--------|
| 1 | Load data; split train/val/test by session id | `data_splits.json` |
| 2 | Train quality model (encoder + MLP) | `pctr_model.pt` |
| 3 | Train user-judge proxy (click + retention heads) | `judge_model.pt` |
| 4 | Build the SFT renderer data | — |
| 5 | Train the StopNet (Bellman action values) | `stop_policy.pt` |
| 6 | Fine-tune the renderer LLM | `llm_full_sft/` or `llm_lora_adapter/` |
| 7–8 | Load renderers and judge | — |
| 9 | Closed-loop evaluation across policies | `final_evaluation_sessions.csv` |
| 10 | Aggregate results | `final_evaluation_summary.csv` |

---

## 7. File structure

```text
src/
├── auction/auction.py          # winner allocation + envelope / virtual-value pricing
├── config.py                   # experiment hyperparameters
├── data/data_utils.py          # data loading, feature construction, session split
├── eval/
│   ├── closed_loop_eval.py     # policy evaluation (first/last/random/myopic/learned)
│   ├── envelope_payment.py     # envelope CPC computation
│   └── envelope_numerics.py    # bid-grid integration
├── generation/
│   ├── base_llm_renderer.py    # untrained / template renderer
│   └── llm_renderer.py         # fine-tuned renderer
├── llm_sft/train_lora_sft.py   # renderer fine-tuning (full or LoRA)
├── models/text_mlp.py          # quality model + judge (encoder + MLP)
├── policy/stop_policy.py       # StopNet (Bellman action values)
├── user_simulator/             # user-judge proxy and LLM judge
├── run_final_experiment.py     # full train + evaluate (main entry)
└── run_experiments_p1p2.py     # virtual-value and IC / ablation experiments

data_pipeline_public/           # simulated-data generation (profiles, ads, dialogues)
```

---

## 8. Reducing memory use

```bash
--llm_max_length 512
--llm_max_samples 200
--max_eval_sessions 30
--eval_with_proxy_renderer    # skip loading any HF model
```
