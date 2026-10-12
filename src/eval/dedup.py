"""Versioned render/judge memoization for frozen evaluation inputs.

Cache hits share one frozen response across policies. With stochastic generation,
this changes the coupling of the samples; it is not a bitwise equivalence claim
against independently sampled responses. Use deterministic decoding for A/B checks.
Legacy caches are ignored because they omitted model/configuration identities.
"""
from __future__ import annotations

import atexit
import hashlib
import inspect
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

CACHE_VERSION = 2
_LOCK = threading.RLock()
_RENDER_CACHE = {}
_JUDGE_CACHE = {}
_STATS = {"render_calls": 0, "render_unique": 0, "judge_calls": 0, "judge_unique": 0}
_INSTALLED = False
_CACHE_PATH = None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def artifact_identity(*paths):
    """Fingerprint local checkpoint/config manifests, including overwrite times.

    The resolved path, relative filenames, sizes and nanosecond mtimes identify
    the frozen artifacts. Remote identifiers must name immutable model revisions.
    """
    manifest = []
    for raw in paths:
        p = Path(raw).expanduser()
        if not p.exists():
            manifest.append([str(raw), "remote_or_missing"])
            continue
        files = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
        manifest.extend([str(x.resolve()), x.stat().st_size, x.stat().st_mtime_ns]
                        for x in files)
    return _digest(manifest)


def _enc(key):
    return json.dumps(key, ensure_ascii=False, separators=(",", ":"))


def _dec(value):
    return tuple(json.loads(value))


def load_cache(path):
    global _CACHE_PATH
    _CACHE_PATH = Path(path)
    if not _CACHE_PATH.exists():
        return
    blob = json.loads(_CACHE_PATH.read_text())
    if blob.get("version") != CACHE_VERSION:
        print("[dedup] ignoring legacy cache: incomplete model/input identity")
        return
    with _LOCK:
        _RENDER_CACHE.update({_dec(k): tuple(v) for k, v in blob.get("render", {}).items()})
        _JUDGE_CACHE.update({_dec(k): v for k, v in blob.get("judge", {}).items()})
    print(f"[dedup] loaded {len(_RENDER_CACHE)} renders, {len(_JUDGE_CACHE)} judgments")


def save_cache(path=None):
    p = Path(path) if path else _CACHE_PATH
    if p is None:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with _LOCK:
        tmp.write_text(json.dumps({"version": CACHE_VERSION,
            "render": {_enc(k): list(v) for k, v in _RENDER_CACHE.items()},
            "judge": {_enc(k): v for k, v in _JUDGE_CACHE.items()}}))
    tmp.replace(p)


def _identity(obj):
    """Unknown model instances get private identities, never unsafe shared keys."""
    explicit = getattr(obj, "cache_identity", None)
    if explicit is not None:
        # Configuration remains part of the key even with a checkpoint identity.
        return _digest([type(obj).__name__, explicit,
                        getattr(obj, "use_exact_labels", None),
                        getattr(obj, "user_id_map", None)])
    if not hasattr(obj, "_dedup_identity"):
        obj._dedup_identity = type(obj).__name__ + ":" + uuid.uuid4().hex
    return obj._dedup_identity


def _input_digest(dialogue, turn_number, ad):
    # Only the observable prefix; never hash future responses. Strip bids because
    # renderers/judges are bid-blind. Titles and user/profile context can change.
    prefix = []
    for i, turn in enumerate(dialogue.get("turns", [])):
        tn = int(turn.get("turn", i + 1))
        if tn <= turn_number:
            prefix.append({"turn": tn, "user": turn.get("user", ""),
                           "assistant": turn.get("assistant", "") if tn < turn_number else ""})
    return _digest({"turns": prefix, "profile_id": dialogue.get("profile_id"),
                    "user_profile": dialogue.get("user_profile"),
                    "candidates": [{k: a.get(k) for k in ("ad_id", "title")}
                                   for a in dialogue.get("candidate_ads", [])],
                    "ad": {k: ad.get(k) for k in ("ad_id", "title")}})


def _render_key(ident, dialogue, turn_number, ad):
    return (ident, str(dialogue.get("session_id", "")), int(turn_number),
            _input_digest(dialogue, turn_number, ad))


def _judge_key(ident, dialogue, turn_number, ad, response):
    return _render_key(ident, dialogue, turn_number, ad) + (_digest(response),)


def _patch_renderer(cls):
    if cls is None or cls.__dict__.get("_dedup_patched", False):
        return
    original_init = cls.__init__
    signature = inspect.signature(original_init)

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        config = {}
        for name in ("base_model", "adapter_dir", "model_path", "max_new_tokens",
                     "temperature", "top_p", "load_in_4bit"):
            value = getattr(self, name, bound.arguments.get(name))
            if value is not None:
                config[name] = (artifact_identity(value) if name in
                                ("base_model", "adapter_dir", "model_path") else value)
        # Proxy outputs depend on the response tables, not just the class name.
        for name in ("response_by_key", "sft_by_session_ad"):
            if hasattr(self, name):
                config[name] = sorted((str(k), v) for k, v in getattr(self, name).items())
        self._dedup_identity = _digest([CACHE_VERSION, cls.__name__, config])
        self._dedup_in_batch = False
    cls.__init__ = init

    original_render = getattr(cls, "render", None)
    if original_render is not None:
        def render(self, dialogue, turn_number, ad):
            if self._dedup_in_batch:
                return original_render(self, dialogue, turn_number, ad)
            key = _render_key(_identity(self), dialogue, turn_number, ad)
            with _LOCK:
                _STATS["render_calls"] += 1
                if key in _RENDER_CACHE:
                    return _RENDER_CACHE[key]
            # Some render() implementations delegate to render_batch(). Keep
            # their underlying work from being counted/cached twice.
            self._dedup_in_batch = True
            try:
                out = original_render(self, dialogue, turn_number, ad)
            finally:
                self._dedup_in_batch = False
            with _LOCK:
                _RENDER_CACHE[key] = out
                _STATS["render_unique"] += 1
            return out
        cls.render = render

    original_batch = getattr(cls, "render_batch", None)
    if original_batch is not None:
        def render_batch(self, items, batch_size=8):
            if self._dedup_in_batch:
                return original_batch(self, items, batch_size)
            keys = [_render_key(_identity(self), d, tn, ad) for d, tn, ad in items]
            with _LOCK:
                _STATS["render_calls"] += len(items)
                todo = {}
                for i, key in enumerate(keys):
                    if key not in _RENDER_CACHE:
                        todo.setdefault(key, i)
            if todo:
                self._dedup_in_batch = True
                try:
                    outputs = original_batch(self, [items[i] for i in todo.values()], batch_size)
                finally:
                    self._dedup_in_batch = False
                if len(outputs) != len(todo):
                    raise ValueError("Renderer batch output length mismatch")
                with _LOCK:
                    for key, out in zip(todo, outputs):
                        _RENDER_CACHE[key] = out
                        _STATS["render_unique"] += 1
            return [_RENDER_CACHE[key] for key in keys]
        cls.render_batch = render_batch
    cls._dedup_patched = True


def _patch_judge(cls):
    if cls is None or cls.__dict__.get("_dedup_patched", False):
        return
    original = cls.judge
    def judge(self, dialogue, turn_number, ad, response):
        key = _judge_key(_identity(self), dialogue, turn_number, ad, response)
        with _LOCK:
            _STATS["judge_calls"] += 1
            if key in _JUDGE_CACHE:
                return dict(_JUDGE_CACHE[key])
        out = original(self, dialogue, turn_number, ad, response)
        with _LOCK:
            _JUDGE_CACHE[key] = dict(out)
            _STATS["judge_unique"] += 1
        return out
    cls.judge = judge
    cls._dedup_patched = True


def install_dedup():
    global _INSTALLED
    if _INSTALLED:
        return True
    if os.environ.get("LLM_OSDA_DEDUP", "0").lower() not in ("1", "true"):
        return False
    path = os.environ.get("LLM_OSDA_DEDUP_CACHE")
    if path:
        load_cache(path)
        atexit.register(save_cache, path)
    from src.generation.base_llm_renderer import BaseLLMRenderer, BaseTemplateRenderer, VLLMRenderer
    from src.generation.sft_proxy import SoftInsertionRenderer
    from src.user_simulator.judge_proxy import UserJudgeProxy
    for cls in (BaseLLMRenderer, BaseTemplateRenderer, VLLMRenderer, SoftInsertionRenderer):
        _patch_renderer(cls)
    _patch_judge(UserJudgeProxy)
    # Optional dependencies are required only for the corresponding LLM path.
    try:
        from src.generation.llm_renderer import TrainedLLMRenderer
        _patch_renderer(TrainedLLMRenderer)
    except ImportError:
        pass
    try:
        from src.user_simulator.local_llm_judge import LocalUserLLMJudge
        _patch_judge(LocalUserLLMJudge)
    except ImportError:
        pass
    _INSTALLED = True
    return True


def dump_stats(path=None):
    stats = dict(_STATS)
    stats.update(render_saved=stats["render_calls"] - stats["render_unique"],
                 judge_saved=stats["judge_calls"] - stats["judge_unique"],
                 cache_size_render=len(_RENDER_CACHE), cache_size_judge=len(_JUDGE_CACHE),
                 enabled=_INSTALLED)
    # These are inference items, including proxy inference, not LLM API calls.
    stats["total_unique_inference_items"] = stats["render_unique"] + stats["judge_unique"]
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(stats, indent=2))
    return stats
