from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.data_utils import LabeledAdSample
from src.models.text_mlp import JudgeModel
from src.user_simulator.user_llm_prompt import USER_LLM_JUDGE_SYSTEM_PROMPT, build_user_llm_judge_prompt
from src.user_simulator.judge_proxy import UserJudgeProxy


class LocalUserLLMJudge:
    """Optional local LLM user simulator with a strict JSON prompt.

    This is optional because a second local LLM can slow evaluation. If JSON parsing
    fails, the class falls back to the trained response-sensitive judge proxy.
    """

    def __init__(
        self,
        base_model: str,
        label_map: Dict[Tuple[str, int, str], LabeledAdSample],
        fallback_judge_model: JudgeModel,
        device: str | torch.device = "auto",
        max_new_tokens: int = 160,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_new_tokens = int(max_new_tokens)
        self.fallback = UserJudgeProxy(label_map, fallback_judge_model, use_exact_labels=False)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any] | None:
        text = text.strip()
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def judge(self, dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any], response: str) -> Dict[str, Any]:
        user_prompt = build_user_llm_judge_prompt(dialogue, turn_number, ad, response)
        messages = [
            {"role": "system", "content": USER_LLM_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"<system>\n{USER_LLM_JUDGE_SYSTEM_PROMPT}\n</system>\n<user>\n{user_prompt}\n</user>\n<assistant>\n"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = self._extract_json(gen)
        if not parsed:
            fallback = self.fallback.judge(dialogue, turn_number, ad, response)
            fallback["source"] = "local_user_llm_parse_failed_fallback_proxy"
            fallback["raw_user_llm_output"] = gen[:500]
            return fallback

        click_prob = float(parsed.get("click_probability", 1.0 if parsed.get("clicked") else 0.0))
        click_prob = max(0.0, min(1.0, click_prob))
        gamma = float(parsed.get("gamma_score", 0.5))
        gamma = max(0.0, min(1.0, gamma))
        clicked = bool(parsed.get("clicked", click_prob >= 0.5))
        return {
            "click_value": click_prob,
            "clicked_binary": int(clicked),
            "gamma_score": gamma,
            "source": "local_user_llm_json_judge",
            "exact": False,
            "judge_mode": "local_llm_prompt",
            "reason": str(parsed.get("reason", "")),
            "raw_user_llm_output": gen[:500],
        }
