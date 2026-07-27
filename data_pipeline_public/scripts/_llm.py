"""
Unified LLM caller for an OpenAI/Anthropic-compatible API gateway.
Supports 3 endpoint formats: chat, responses, anthropic.
"""

import json
import time
import asyncio
import aiohttp
import requests as _requests
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_config():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "pipeline.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


CFG = load_config()
BASE_URL = CFG["gateway"]["base_url"]
API_KEY = CFG["gateway"]["api_key"]
RATE = CFG["rate_limit"]


def _headers(fmt: str) -> dict:
    """Build request headers. The gateway uses a bare api_key without a Bearer prefix."""
    h = {
        "Content-Type": "application/json",
        "Service": "",
    }
    # The gateway expects a bare api_key for all formats.
    h["Authorization"] = API_KEY
    return h


def _build_payload(
    messages: List[Dict],
    model_cfg: dict,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> dict:
    fmt = model_cfg["format"]
    model = model_cfg["model"]
    supports_temp = not model_cfg.get("no_temperature", False)

    if fmt == "chat":
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if supports_temp:
            payload["temperature"] = temperature
        return payload

    elif fmt == "responses":
        # a GPT-family model: uses "input" instead of "messages", no max_tokens
        payload = {
            "model": model,
            "input": messages,
            "stream": False,
        }
        if supports_temp:
            payload["temperature"] = temperature
        return payload

    elif fmt == "anthropic":
        # Anthropic format: separate system from messages
        system_text = ""
        conv = []
        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n"
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        payload = {
            "model": model,
            "messages": conv,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if supports_temp:
            payload["temperature"] = temperature
        if system_text.strip():
            payload["system"] = system_text.strip()
        return payload

    else:
        raise ValueError(f"Unknown format: {fmt}")


def _extract_content(result: dict, fmt: str) -> str:
    """从 LLM 响应中提取文本内容。"""
    if fmt == "chat":
        choices = result.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

    elif fmt == "responses":
        output = result.get("output", [])
        if not output:
            return ""
        contents = output[0].get("content", [])
        if not contents:
            return ""
        return contents[0].get("text", "")

    elif fmt == "anthropic":
        content = result.get("content", [])
        if not content:
            return ""
        return content[0].get("text", "")

    return ""


# ── Synchronous caller ──────────────────────────────────────────


def call_llm_sync(
    messages: List[Dict],
    model_key: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: int = 300,
) -> str:
    """Blocking LLM call with retry."""
    model_cfg = CFG["models"][model_key]
    fmt = model_cfg["format"]
    url = BASE_URL + model_cfg["endpoint"]
    headers = _headers(fmt)
    payload = _build_payload(messages, model_cfg, temperature, max_tokens)

    for attempt in range(RATE["retry_max"]):
        try:
            resp = _requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return _extract_content(resp.json(), fmt)
            elif resp.status_code == 429:
                wait = RATE["retry_base_sleep"] * (2 ** attempt)
                print(f"[rate-limit] 429, sleeping {wait}s ...")
                time.sleep(wait)
                continue
            elif resp.status_code in (502, 503, 504):
                wait = RATE["retry_base_sleep"] * (2 ** attempt)
                print(f"[gateway] {resp.status_code}, retry {attempt+1} in {wait}s ...")
                time.sleep(wait)
                continue
            else:
                print(f"[llm] error {resp.status_code}: {resp.text[:300]}")
                raise Exception(f"API error {resp.status_code}")
        except _requests.exceptions.Timeout:
            print(f"[llm] timeout attempt {attempt+1}")
            time.sleep(RATE["retry_base_sleep"])

    raise Exception("Max retries exceeded")


# ── Async caller ──────────────────────────────────────────


_semaphore: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(RATE["max_concurrency"])
    return _semaphore


async def call_llm_async(
    messages: List[Dict],
    model_key: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: int = 300,
) -> str:
    """Async LLM call with concurrency control and retry."""
    model_cfg = CFG["models"][model_key]
    fmt = model_cfg["format"]
    url = BASE_URL + model_cfg["endpoint"]
    headers = _headers(fmt)
    payload = _build_payload(messages, model_cfg, temperature, max_tokens)
    sem = _get_semaphore()

    for attempt in range(RATE["retry_max"]):
        async with sem:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, json=payload, headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        if resp.status == 200:
                            result = await resp.json(content_type=None)
                            return _extract_content(result, fmt)
                        elif resp.status == 429:
                            wait = RATE["retry_base_sleep"] * (2 ** attempt)
                            print(f"[rate-limit] 429, sleeping {wait}s ...")
                            await asyncio.sleep(wait)
                            continue
                        elif resp.status in (502, 503, 504):
                            wait = RATE["retry_base_sleep"] * (2 ** attempt)
                            print(f"[gateway] {resp.status}, retry {attempt+1} in {wait}s ...")
                            await asyncio.sleep(wait)
                            continue
                        else:
                            text = await resp.text()
                            print(f"[llm] error {resp.status}: {text[:300]}")
                            raise Exception(f"API error {resp.status}")
            except asyncio.TimeoutError:
                print(f"[llm] timeout attempt {attempt+1}")
                await asyncio.sleep(RATE["retry_base_sleep"])

    raise Exception("Max retries exceeded")
