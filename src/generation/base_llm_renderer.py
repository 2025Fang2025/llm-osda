from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from src.data.data_utils import candidate_ads_text


class BaseLLMRenderer:
    def __init__(
        self,
        base_model: str,
        device: str | torch.device = "auto",
        max_new_tokens: int = 160,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.base_model = base_model
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.renderer_tag = "base_untrained_llm"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        # Old fallback used fp32 on CUDA when bf16 was unavailable, which is
        # unnecessarily heavy for 8GB cards during untrained LLM evaluation.
        # dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    def _messages(self, dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> List[Dict[str, str]]:
        system = (
            "You are a helpful AI shopping assistant.\n"
            "The platform has selected one sponsored product to include in this response.\n"
            "Answer the user's current question and include the sponsored product once using the exact format @ad_title@[ad_id].\n"
            "Do not mention auctions, bids, pCTR, payment, or internal mechanisms.\n\n"
            f"Candidate Ads:\n{candidate_ads_text(dialogue)}\n\n"
            f"Selected Sponsored Product: {ad.get('title','')} (Ad ID: {ad.get('ad_id','')})"
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
        return [{"role": "system", "content": system}] + hist

    def _make_prompt(self, dialogue, turn_number, ad):
        messages = self._messages(dialogue, turn_number, ad)
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        return "\n".join([f"<{m['role']}>\n{m['content']}\n</{m['role']}>" for m in messages]) + "\n<assistant>\n"

    def render(self, dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> Tuple[str, str]:
        prompt = self._make_prompt(dialogue, turn_number, ad)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
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
        text = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if not text:
            text = f"A relevant option here is @{ad.get('title','')}@[{ad.get('ad_id','')}]."
        return text, self.renderer_tag

    def render_batch(self, items: List[Tuple[Dict, int, Dict]], batch_size: int = 8) -> List[Tuple[str, str]]:
        prompts = [self._make_prompt(d, tn, ad) for d, tn, ad in items]
        results = []
        do_sample = self.temperature > 1e-6
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_ads = [item[2] for item in items[i:i + batch_size]]
            inputs = self.tokenizer(
                batch_prompts, return_tensors="pt", truncation=True,
                max_length=2048, padding=True,
            ).to(self.device)
            prompt_lens = [inputs["attention_mask"][j].sum().item() for j in range(len(batch_prompts))]
            with torch.no_grad():
                outs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=do_sample,
                    temperature=self.temperature if do_sample else None,
                    top_p=self.top_p if do_sample else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            for j in range(len(batch_prompts)):
                gen_ids = outs[j, int(prompt_lens[j]):]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                if not text:
                    ad = batch_ads[j]
                    text = f"A relevant option here is @{ad.get('title','')}@[{ad.get('ad_id','')}]."
                results.append((text, self.renderer_tag))
        return results


class BaseTemplateRenderer:
    def render(self, dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> Tuple[str, str]:
        user = ""
        for t in dialogue.get("turns", []):
            if int(t.get("turn", 0)) == int(turn_number):
                user = t.get("user", "")
                break
        response = (
            f"For your question — {user} — a product to consider is @{ad.get('title','')}@[{ad.get('ad_id','')}]. "
            f"It may fit your needs depending on your exact use case."
        )
        return response, "base_template_untrained_proxy"

    def render_batch(self, items, batch_size=8):
        return [self.render(d, tn, ad) for d, tn, ad in items]


class VLLMRenderer:
    def __init__(
        self,
        base_model: str,
        max_new_tokens: int = 160,
        temperature: float = 0.2,
        top_p: float = 0.9,
        gpu_memory_utilization: float = 0.5,
        tensor_parallel_size: int = 1,
    ):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.renderer_tag = "vllm_renderer"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = LLM(
            model=base_model,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=2048,
        )
        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if temperature > 1e-6 else 0,
            top_p=top_p,
        )

    def _make_prompt(self, dialogue, turn_number, ad):
        system = (
            "You are a helpful AI shopping assistant.\n"
            "The platform has selected one sponsored product to include in this response.\n"
            "Answer the user's current question and include the sponsored product once using the exact format @ad_title@[ad_id].\n"
            "Do not mention auctions, bids, pCTR, payment, or internal mechanisms.\n\n"
            f"Candidate Ads:\n{candidate_ads_text(dialogue)}\n\n"
            f"Selected Sponsored Product: {ad.get('title','')} (Ad ID: {ad.get('ad_id','')})"
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
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        return "\n".join([f"<{m['role']}>\n{m['content']}\n</{m['role']}>" for m in messages]) + "\n<assistant>\n"

    def render(self, dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> Tuple[str, str]:
        results = self.render_batch([(dialogue, turn_number, ad)])
        return results[0]

    def render_batch(self, items: List[Tuple[Dict, int, Dict]], batch_size: int = 0) -> List[Tuple[str, str]]:
        prompts = [self._make_prompt(d, tn, ad) for d, tn, ad in items]
        outputs = self.llm.generate(prompts, self.sampling_params)
        results = []
        for i, out in enumerate(outputs):
            text = out.outputs[0].text.strip()
            if not text:
                ad = items[i][2]
                text = f"A relevant option here is @{ad.get('title','')}@[{ad.get('ad_id','')}]."
            results.append((text, self.renderer_tag))
        return results
