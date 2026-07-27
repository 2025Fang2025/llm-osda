from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.data.data_utils import history_until_turn, candidate_ads_text


class TrainedLLMRenderer:
    """Renderer that actually uses the SFT-trained LoRA adapter.

    It keeps the trained language generation separate from auction/stopping.
    The policy chooses when to auction; auction chooses the winner; this class
    only writes the final natural-language response with exactly one winner ad.
    """
    def __init__(
        self,
        base_model: str,
        adapter_dir: str | Path,
        device: str | torch.device = "auto",
        max_new_tokens: int = 160,
        temperature: float = 0.2,
        top_p: float = 0.9,
        load_in_4bit: bool = False,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.base_model = base_model
        self.adapter_dir = str(adapter_dir)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.load_in_4bit = bool(load_in_4bit)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
            model_kwargs["device_map"] = "auto"

        # Detect adapter format: LoRA (has adapter_config.json) vs Full SFT
        # (standalone HF model dir with config.json). Full SFT may live in the
        # given directory or in a `llm_full_sft` subdir (as saved by
        # train_bid_in_renderer.py alongside training-side artifacts).
        adapter_path = Path(self.adapter_dir)
        candidates = [
            adapter_path,
            adapter_path / "llm_full_sft",
            adapter_path / "llm_lora_adapter",
        ]
        resolved_path = None
        is_lora = False
        for cand in candidates:
            if not cand.is_dir():
                continue
            has_adapter = (cand / "adapter_config.json").exists()
            has_full = (cand / "config.json").exists()
            if has_adapter:
                resolved_path = str(cand.resolve())
                is_lora = True
                break
            if has_full:
                resolved_path = str(cand.resolve())
                is_lora = False
                break
        if resolved_path is None:
            raise FileNotFoundError(
                f"No LoRA adapter or Full SFT model found under {self.adapter_dir}. "
                f"Probed: {[str(c) for c in candidates]}"
            )
        self.adapter_dir = resolved_path

        if is_lora:
            base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
            self.model = PeftModel.from_pretrained(base, resolved_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(resolved_path, **model_kwargs)
        if not self.load_in_4bit:
            self.model.to(self.device)
        self.model.eval()

    def _messages(self, dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> List[Dict[str, str]]:
        system = (
            "You are a helpful AI shopping assistant.\n"
            "The platform has already selected exactly one winning ad through an auction.\n"
            "Your job is to answer the user's current query naturally and, only if relevant, "
            "softly integrate the winning ad once.\n"
            "Use the exact format @ad_title@[ad_id] for the winning ad.\n"
            "Do not mention auctions, bids, payment, pCTR, or mechanism details.\n\n"
            f"Candidate Ads:\n{candidate_ads_text(dialogue)}\n\n"
            # Mechanism fix: the LLM renderer should not receive hidden bid values.
            # f"Winning Ad: {ad.get('title','')} (Ad ID: {ad.get('ad_id','')}, Bid: ${float(ad.get('bid',0.0)):.2f})"
            f"Winning Ad: {ad.get('title','')} (Ad ID: {ad.get('ad_id','')})"
        )
        hist = []
        for t in dialogue.get("turns", []):
            tn = int(t.get("turn", 0))
            if tn < turn_number:
                hist.append({"role": "user", "content": t.get("user", "")})
                # Keep previous assistant content but strip long debug if present.
                hist.append({"role": "assistant", "content": t.get("assistant", "")})
            elif tn == turn_number:
                hist.append({"role": "user", "content": t.get("user", "")})
                break
        return [{"role": "system", "content": system}] + hist

    def render(self, dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> Tuple[str, str]:
        messages = self._messages(dialogue, turn_number, ad)
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            try:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join([f"<{m['role']}>\n{m['content']}\n</{m['role']}>" for m in messages]) + "\n<assistant>\n"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(self.device)
        do_sample = self.temperature > 1e-6
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if not text:
            text = f"A relevant option here is @{ad.get('title','')}@[{ad.get('ad_id','')}]."
        return text, "trained_lora_llm"
