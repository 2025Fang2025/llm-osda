from __future__ import annotations
import inspect
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

log = logging.getLogger(__name__)

DEFAULT_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_sft_rows_by_sessions(rows: List[Dict[str, Any]], session_ids: Optional[set[str]]) -> List[Dict[str, Any]]:
    if not session_ids:
        return rows
    return [r for r in rows if r.get("session_id") in session_ids]


def format_messages_for_causal_lm(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass

    parts = []
    for m in messages:
        role = m.get("role", "user").strip().lower()
        content = m.get("content", "").strip()
        if not content:
            continue
        parts.append(f"<{role}>\n{content}\n</{role}>")
    return "\n".join(parts) + "\n"


def strip_bid_mentions_from_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove hidden bid values from SFT prompts before renderer training."""
    cleaned = []
    for msg in messages:
        content = msg.get("content", "")
        # Mechanism fix: SFT renderer should learn ad insertion without seeing bids.
        # Old SFT rows often contain strings like ", Bid: $1.03" or ", bid=1.03".
        content = re.sub(r",?\s*Bid:\s*\$?\d+(?:\.\d+)?", "", content)
        content = re.sub(r",?\s*bid=\$?\d+(?:\.\d+)?", "", content)
        cleaned.append({**msg, "content": content})
    return cleaned


class SFTConversationDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


class MetricsCollectorCallback(TrainerCallback):
    def __init__(self):
        self.step_logs = []
        self.epoch_logs = []
        self._epoch_losses = []
        self._current_epoch = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss")
        if loss is not None:
            self.step_logs.append({
                "step": int(state.global_step),
                "loss": float(loss),
                "lr": float(logs.get("learning_rate", 0)),
                "epoch": float(logs.get("epoch", 0)),
            })
            self._epoch_losses.append(float(loss))
            gpu_info = ""
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                gpu_info = f" | GPU={alloc:.1f}GB"
            log.info(f"  SFT step={state.global_step} loss={loss:.4f} lr={logs.get('learning_rate', 0):.2e}{gpu_info}")

    def on_epoch_end(self, args, state, control, **kwargs):
        self._current_epoch += 1
        if self._epoch_losses:
            avg_loss = sum(self._epoch_losses) / len(self._epoch_losses)
            self.epoch_logs.append({
                "epoch": self._current_epoch,
                "avg_loss": float(avg_loss),
                "n_steps": len(self._epoch_losses),
            })
            log.info(f"  SFT epoch {self._current_epoch} done: avg_loss={avg_loss:.4f}, steps={len(self._epoch_losses)}")
            self._epoch_losses = []


def plot_sft_curves(step_logs, epoch_logs, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_dir = Path(output_dir)
    if not step_logs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    steps = [s["step"] for s in step_logs]
    losses = [s["loss"] for s in step_logs]
    axes[0].plot(steps, losses, alpha=0.4, linewidth=0.5, color="blue", label="step loss")
    window = max(1, len(losses) // 50)
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        axes[0].plot(steps[window - 1:], smoothed, color="red", linewidth=1.5, label=f"smooth (w={window})")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("SFT Step Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if epoch_logs:
        epochs = [e["epoch"] for e in epoch_logs]
        avg_losses = [e["avg_loss"] for e in epoch_logs]
        axes[1].plot(epochs, avg_losses, marker="o", color="green")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Avg Loss")
        axes[1].set_title(f"SFT Epoch Loss (final={avg_losses[-1]:.4f})")
        axes[1].grid(True, alpha=0.3)

    lrs = [s["lr"] for s in step_logs if s["lr"] > 0]
    lr_steps = [s["step"] for s in step_logs if s["lr"] > 0]
    if lrs:
        axes[2].plot(lr_steps, lrs, color="orange")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("SFT Learning Rate Schedule")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "sft_training_curves.png", dpi=150)
    plt.close()
    log.info(f"  SFT training curves saved to {output_dir / 'sft_training_curves.png'}")


def train_sft(
    sft_path: str | Path,
    output_dir: str | Path,
    base_model: str = "Qwen/Qwen3-4B",
    train_session_ids: Optional[set[str]] = None,
    max_samples: int = 800,
    max_length: int = 1024,
    epochs: float = 1.0,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 2e-4,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    seed: int = 42,
    fp16: bool = True,
    gradient_checkpointing: bool = True,
    load_in_4bit: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(sft_path)
    rows = filter_sft_rows_by_sessions(rows, train_session_ids)
    if max_samples and max_samples > 0 and len(rows) > max_samples:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError("No SFT rows available after filtering. Check sft path/session split.")
    log.info(f"  SFT data: {len(rows)} samples, max_length={max_length}, epochs={epochs}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Old path formatted raw messages, including candidate-ad bid values:
    # texts = [format_messages_for_causal_lm(tokenizer, r["messages"]) for r in rows]
    texts = [format_messages_for_causal_lm(tokenizer, strip_bid_mentions_from_messages(r["messages"])) for r in rows]
    dataset = SFTConversationDataset(texts, tokenizer, max_length=max_length)
    token_lengths = [len(tokenizer(t, truncation=True, max_length=max_length)["input_ids"]) for t in texts[:100]]
    log.info(f"  Token lengths (first 100): mean={np.mean(token_lengths):.0f}, max={max(token_lengths)}, min={min(token_lengths)}")

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        use_bf16 = True
        use_fp16 = False
    elif fp16 and torch.cuda.is_available():
        model_dtype = torch.float16
        use_bf16 = False
        use_fp16 = True
    else:
        model_dtype = torch.float32
        use_bf16 = False
        use_fp16 = False

    quantization_config = None
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "QLoRA needs bitsandbytes support. Install peft, accelerate, and bitsandbytes, "
                "or rerun without --llm_load_in_4bit."
            ) from exc
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=model_dtype,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"

    log.info(f"  Loading base model: {base_model} (dtype={model_dtype}, load_in_4bit={load_in_4bit})")
    # Old path loaded the full base weights, which is too heavy for 3B on 8GB:
    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model,
    #     trust_remote_code=True,
    #     torch_dtype=model_dtype,
    #     low_cpu_mem_usage=True,
    # )
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        if load_in_4bit:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=gradient_checkpointing,
            )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=DEFAULT_LORA_TARGET_MODULES,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        log.info(f"  LoRA enabled: r={lora_r}, alpha={lora_alpha}")
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"  Full SFT: {n_params / 1e6:.1f}M trainable parameters")

    total_steps = math.ceil(len(dataset) / batch_size / grad_accum * epochs)
    log.info(f"  Training: batch={batch_size}, grad_accum={grad_accum}, eff_batch={batch_size * grad_accum}, lr={lr}, ~{total_steps} steps")

    metrics_cb = MetricsCollectorCallback()

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_tmp"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        fp16=use_fp16,
        bf16=use_bf16,
        optim="adamw_torch",
        seed=seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        warmup_ratio=0.05,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[metrics_cb],
    )

    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    elapsed = time.time() - t0
    log.info(f"  SFT training finished in {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    if use_lora:
        save_dir = output_dir / "llm_lora_adapter"
    else:
        save_dir = output_dir / "llm_full_sft"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    log.info(f"  Model saved to {save_dir}")

    plot_sft_curves(metrics_cb.step_logs, metrics_cb.epoch_logs, output_dir)

    meta = {
        "base_model": base_model,
        "sft_mode": "lora" if use_lora else "full",
        "save_dir": str(save_dir),
        "n_sft_rows_used": len(rows),
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "load_in_4bit": bool(load_in_4bit),
        "total_train_time_sec": round(elapsed, 1),
        "step_history": metrics_cb.step_logs,
        "epoch_history": metrics_cb.epoch_logs,
    }
    if use_lora:
        meta.update({"lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout})
    with open(output_dir / "llm_sft_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


# backward-compatible alias
train_lora_sft = train_sft
