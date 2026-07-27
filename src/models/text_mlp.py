from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, log_loss, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Text encoder abstraction
# ---------------------------------------------------------------------------

class TextEncoder(abc.ABC):
    @abc.abstractmethod
    def encode(self, texts: List[str]) -> Union[np.ndarray, torch.Tensor]:
        ...

    @abc.abstractmethod
    def input_dim(self) -> int:
        ...

    @abc.abstractmethod
    def save(self, output_dir: Path) -> None:
        ...

    @staticmethod
    @abc.abstractmethod
    def load(output_dir: Path, device: torch.device) -> "TextEncoder":
        ...

    def train_mode(self):
        pass

    def eval_mode(self):
        pass

    def parameters(self):
        return iter([])


class TfidfTextEncoder(TextEncoder):
    def __init__(self, vectorizer: TfidfVectorizer):
        self._vectorizer = vectorizer

    @property
    def input_dim(self) -> int:
        return len(self._vectorizer.vocabulary_)

    def fit(self, texts: List[str]) -> "TfidfTextEncoder":
        self._vectorizer.fit(texts)
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._vectorizer.transform(texts).astype(np.float32).toarray()

    def save(self, output_dir: Path) -> None:
        joblib.dump(self._vectorizer, output_dir / "text_encoder_tfidf.joblib")
        with open(output_dir / "text_encoder_meta.json", "w") as f:
            json.dump({"type": "tfidf"}, f)

    @staticmethod
    def load(output_dir: Path, device: torch.device) -> "TfidfTextEncoder":
        vectorizer = joblib.load(output_dir / "text_encoder_tfidf.joblib")
        return TfidfTextEncoder(vectorizer)


class EmbeddingTextEncoder(TextEncoder):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: torch.device = torch.device("cpu"),
        max_length: int = 512,
    ):
        from transformers import AutoModel, AutoTokenizer
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left",
        )
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float32
        self._model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, dtype=model_dtype,
        ).to(device)
        self._embed_dim = self._model.config.hidden_size
        # Optional per-text embedding cache. Frozen model => a given text always
        # maps to the same vector, so during eval (where the same ad text is
        # re-encoded across policies / bid bisection / cv computation) we can
        # look it up instead of re-running the 0.6B forward. Off by default so
        # training (which already pre-encodes) doesn't pay the dict overhead.
        self._cache_enabled = False
        self._cache: Dict[str, torch.Tensor] = {}

    def enable_cache(self, enabled: bool = True) -> None:
        self._cache_enabled = enabled
        if not enabled:
            self._cache = {}

    def clear_cache(self) -> None:
        self._cache = {}

    @property
    def input_dim(self) -> int:
        return self._embed_dim

    def train_mode(self):
        self._model.train()

    def eval_mode(self):
        self._model.eval()

    def parameters(self):
        return self._model.parameters()

    def encode(self, texts: List[str]) -> torch.Tensor:
        if not self._cache_enabled:
            return self._encode_uncached(texts)

        # Identify which texts are not yet cached, encode only those once.
        missing = [t for t in texts if t not in self._cache]
        if missing:
            # Deduplicate while preserving order so we don't encode a repeat twice
            # within the same call.
            uniq = list(dict.fromkeys(missing))
            emb = self._encode_uncached(uniq)
            # Store on CPU: eval calls del pctr_model then load vLLM at 0.45 GPU
            # util, so the cache must not hold GPU memory. Callers move to device.
            for t, v in zip(uniq, emb):
                self._cache[t] = v.detach().float().cpu()
        return torch.stack([self._cache[t] for t in texts], dim=0)

    def _encode_uncached(self, texts: List[str]) -> torch.Tensor:
        enc = self._tokenizer(
            texts, padding=True, truncation=True,
            max_length=self._max_length, return_tensors="pt",
        ).to(self._device)
        out = self._model(**enc)
        embeddings = self._last_token_pool(out.last_hidden_state, enc["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    @staticmethod
    def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Qwen3-Embedding is decoder-only: the sentence representation lives on the
        # last non-pad token, not the mean of all tokens.
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]

    def encode_np(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        self.eval_mode()
        parts = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                emb = self.encode(texts[i:i + batch_size])
                parts.append(emb.float().cpu().numpy())
        return np.concatenate(parts, axis=0)

    def save(self, output_dir: Path) -> None:
        encoder_dir = output_dir / "text_encoder_embedding"
        encoder_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(encoder_dir)
        self._tokenizer.save_pretrained(encoder_dir)
        with open(output_dir / "text_encoder_meta.json", "w") as f:
            json.dump({"type": "qwen3_embedding", "model_name": self._model_name}, f)

    @staticmethod
    def load(output_dir: Path, device: torch.device) -> "EmbeddingTextEncoder":
        encoder_dir = output_dir / "text_encoder_embedding"
        meta_path = output_dir / "text_encoder_meta.json"
        from transformers import AutoModel, AutoTokenizer
        if encoder_dir.exists() and (encoder_dir / "config.json").exists():
            model_name = str(encoder_dir)
        elif meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            model_name = meta.get("model_name", str(encoder_dir))
        else:
            model_name = str(encoder_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float32
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, dtype=model_dtype,
        ).to(device)
        enc = EmbeddingTextEncoder.__new__(EmbeddingTextEncoder)
        enc._model_name = model_name
        enc._device = device
        enc._max_length = 512
        enc._tokenizer = tokenizer
        enc._model = model
        enc._embed_dim = model.config.hidden_size
        return enc


def load_text_encoder(output_dir: Path, device: torch.device) -> TextEncoder:
    meta_path = output_dir / "text_encoder_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if meta["type"] == "tfidf":
            return TfidfTextEncoder.load(output_dir, device)
        elif meta["type"] == "qwen3_embedding":
            return EmbeddingTextEncoder.load(output_dir, device)
    if (output_dir / "pctr_vectorizer.joblib").exists():
        vectorizer = joblib.load(output_dir / "pctr_vectorizer.joblib")
        return TfidfTextEncoder(vectorizer)
    raise FileNotFoundError(f"No text encoder found in {output_dir}")


def load_judge_text_encoder(output_dir: Path, device: torch.device) -> TextEncoder:
    meta_path = output_dir / "judge_text_encoder_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if meta["type"] == "tfidf":
            enc = joblib.load(output_dir / "judge_text_encoder_tfidf.joblib")
            return TfidfTextEncoder(enc)
        elif meta["type"] == "qwen3_embedding":
            encoder_dir = output_dir / "judge_text_encoder_embedding"
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir), trust_remote_code=True, padding_side="left")
            if device.type == "cuda" and torch.cuda.is_bf16_supported():
                model_dtype = torch.bfloat16
            else:
                model_dtype = torch.float32
            model = AutoModel.from_pretrained(
                str(encoder_dir), trust_remote_code=True, dtype=model_dtype,
            ).to(device)
            enc = EmbeddingTextEncoder.__new__(EmbeddingTextEncoder)
            enc._model_name = "loaded_from_disk"
            enc._device = device
            enc._max_length = 512
            enc._tokenizer = tokenizer
            enc._model = model
            enc._embed_dim = model.config.hidden_size
            return enc
    if (output_dir / "judge_vectorizer.joblib").exists():
        vectorizer = joblib.load(output_dir / "judge_vectorizer.joblib")
        return TfidfTextEncoder(vectorizer)
    raise FileNotFoundError(f"No judge text encoder found in {output_dir}")


# ---------------------------------------------------------------------------
# MLP heads (unchanged)
# ---------------------------------------------------------------------------

class PCTRNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class JudgeNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.click_head = nn.Linear(hidden_dim, 1)
        self.gamma_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.backbone(x)
        click_logit = self.click_head(h).squeeze(-1)
        gamma = torch.sigmoid(self.gamma_head(h).squeeze(-1))
        return click_logit, gamma


# ---------------------------------------------------------------------------
# Feature construction helpers
# ---------------------------------------------------------------------------

@dataclass
class VectorizedDataset:
    X: np.ndarray
    y_click: np.ndarray
    y_gamma: np.ndarray | None = None


def make_features(vectorizer: TfidfVectorizer, texts: List[str], nums: List[List[float]]) -> np.ndarray:
    X_text = vectorizer.transform(texts).astype(np.float32).toarray()
    if nums and nums[0]:
        X_num = np.asarray(nums, dtype=np.float32)
        if X_num.ndim == 1:
            X_num = X_num.reshape(-1, 1)
        return np.concatenate([X_text, X_num], axis=1).astype(np.float32)
    return X_text


class TextNumDataset(Dataset):
    def __init__(self, texts: List[str], nums: List[List[float]], labels: List[np.ndarray]):
        self.texts = texts
        self.nums = np.asarray(nums, dtype=np.float32) if nums and nums[0] else np.zeros((len(texts), 0), dtype=np.float32)
        if self.nums.ndim == 1:
            self.nums = self.nums.reshape(-1, 1)
        self.labels = [np.asarray(l, dtype=np.float32) for l in labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.nums[idx], tuple(l[idx] for l in self.labels)


def _collate_text_num(batch):
    texts = [b[0] for b in batch]
    nums = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    labels = [torch.tensor(np.array([b[2][i] for b in batch]), dtype=torch.float32)
              for i in range(len(batch[0][2]))]
    return texts, nums, labels


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class PCTRModel:
    def __init__(self, encoder: TextEncoder, model: PCTRNet, device: torch.device):
        self.encoder = encoder
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @property
    def input_dim(self) -> int:
        return self.model.net[0].in_features

    def predict_proba(self, texts: List[str], nums: List[List[float]], batch_size: int = 256) -> np.ndarray:
        self.encoder.eval_mode()
        self.model.eval()
        X_num = np.asarray(nums, dtype=np.float32)
        if X_num.ndim == 1:
            X_num = X_num.reshape(-1, 1)
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                text_batch = texts[i:i + batch_size]
                num_batch = torch.tensor(X_num[i:i + batch_size], dtype=torch.float32, device=self.device)
                text_emb = self.encoder.encode(text_batch)
                if isinstance(text_emb, np.ndarray):
                    text_emb = torch.tensor(text_emb, dtype=torch.float32, device=self.device)
                else:
                    text_emb = text_emb.float().to(self.device)
                xb = torch.cat([text_emb, num_batch], dim=1)
                p = torch.sigmoid(self.model(xb)).detach().cpu().numpy()
                probs.append(p)
        return np.concatenate(probs, axis=0)

    def save(self, output_dir: str | Path, prefix: str = "pctr") -> None:
        output_dir = Path(output_dir)
        torch.save({"state_dict": self.model.state_dict(), "input_dim": self.input_dim},
                    output_dir / f"{prefix}_model.pt")
        self.encoder.save(output_dir)

    @staticmethod
    def load(output_dir: str | Path, device: torch.device, hidden_dim: int = 256) -> "PCTRModel":
        output_dir = Path(output_dir)
        encoder = load_text_encoder(output_dir, device)
        ckpt = torch.load(output_dir / "pctr_model.pt", map_location=device)
        model = PCTRNet(input_dim=ckpt["input_dim"], hidden_dim=hidden_dim)
        model.load_state_dict(ckpt["state_dict"])
        return PCTRModel(encoder, model, device)


class JudgeModel:
    def __init__(self, encoder: TextEncoder, model: JudgeNet, device: torch.device):
        self.encoder = encoder
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @property
    def input_dim(self) -> int:
        return self.model.backbone[0].in_features

    def predict(self, texts: List[str], nums: List[List[float]], batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        self.encoder.eval_mode()
        self.model.eval()
        X_num = np.asarray(nums, dtype=np.float32)
        if X_num.ndim == 1:
            X_num = X_num.reshape(-1, 1)
        clicks, gammas = [], []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                text_batch = texts[i:i + batch_size]
                num_batch = torch.tensor(X_num[i:i + batch_size], dtype=torch.float32, device=self.device)
                text_emb = self.encoder.encode(text_batch)
                if isinstance(text_emb, np.ndarray):
                    text_emb = torch.tensor(text_emb, dtype=torch.float32, device=self.device)
                else:
                    text_emb = text_emb.float().to(self.device)
                xb = torch.cat([text_emb, num_batch], dim=1)
                logit, gamma = self.model(xb)
                clicks.append(torch.sigmoid(logit).detach().cpu().numpy())
                gammas.append(gamma.detach().cpu().numpy())
        return np.concatenate(clicks, axis=0), np.concatenate(gammas, axis=0)

    def save(self, output_dir: str | Path, prefix: str = "judge") -> None:
        output_dir = Path(output_dir)
        torch.save({"state_dict": self.model.state_dict(), "input_dim": self.input_dim},
                    output_dir / f"{prefix}_model.pt")
        if isinstance(self.encoder, TfidfTextEncoder):
            joblib.dump(self.encoder._vectorizer, output_dir / "judge_text_encoder_tfidf.joblib")
            with open(output_dir / "judge_text_encoder_meta.json", "w") as f:
                json.dump({"type": "tfidf"}, f)
        elif isinstance(self.encoder, EmbeddingTextEncoder):
            encoder_dir = output_dir / "judge_text_encoder_embedding"
            encoder_dir.mkdir(parents=True, exist_ok=True)
            self.encoder._model.save_pretrained(encoder_dir)
            self.encoder._tokenizer.save_pretrained(encoder_dir)
            with open(output_dir / "judge_text_encoder_meta.json", "w") as f:
                json.dump({"type": "qwen3_embedding", "model_name": self.encoder._model_name}, f)
        joblib.dump(None, output_dir / "judge_vectorizer.joblib")

    @staticmethod
    def load(output_dir: str | Path, device: torch.device, hidden_dim: int = 256) -> "JudgeModel":
        output_dir = Path(output_dir)
        encoder = load_judge_text_encoder(output_dir, device)
        ckpt = torch.load(output_dir / "judge_model.pt", map_location=device)
        model = JudgeNet(input_dim=ckpt["input_dim"], hidden_dim=hidden_dim)
        model.load_state_dict(ckpt["state_dict"])
        return JudgeModel(encoder, model, device)


# ---------------------------------------------------------------------------
# Training: pCTR
# ---------------------------------------------------------------------------

def train_pctr_text_model(
    train_texts: List[str],
    train_nums: List[List[float]],
    train_y: List[float],
    val_texts: List[str],
    val_nums: List[List[float]],
    val_y: List[float],
    device: torch.device,
    max_features: int = 4096,
    hidden_dim: int = 256,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    text_encoder_type: str = "qwen3_embedding",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    patience: int = 5,
) -> Tuple[PCTRModel, Dict[str, Any]]:
    if text_encoder_type == "tfidf":
        return _train_pctr_tfidf(
            train_texts, train_nums, train_y, val_texts, val_nums, val_y,
            device, max_features, hidden_dim, epochs, batch_size, lr, patience,
        )
    return _train_pctr_embedding(
        train_texts, train_nums, train_y, val_texts, val_nums, val_y,
        device, hidden_dim, epochs, batch_size, lr, embedding_model, patience,
    )


def _train_pctr_tfidf(
    train_texts, train_nums, train_y, val_texts, val_nums, val_y,
    device, max_features, hidden_dim, epochs, batch_size, lr, patience,
):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1)
    vectorizer.fit(train_texts)
    encoder = TfidfTextEncoder(vectorizer)

    X_train = make_features(vectorizer, train_texts, train_nums)
    X_val = make_features(vectorizer, val_texts, val_nums)
    y_train = np.asarray(train_y, dtype=np.float32)
    y_val = np.asarray(val_y, dtype=np.float32)

    model = PCTRNet(X_train.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val = float("inf")
    no_improve = 0
    epoch_history = []
    step_history = []
    global_step = 0
    for epoch in tqdm(range(epochs), desc="train pCTR (tfidf)", leave=False):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in dl:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step_loss = loss.item()
            epoch_loss += step_loss
            n_batches += 1
            global_step += 1
            step_history.append({"step": global_step, "loss": float(step_loss)})
        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            val_loss = log_loss(y_val, np.clip(probs, 1e-5, 1 - 1e-5), labels=[0, 1])
            val_auc = float(roc_auc_score(y_val, probs)) if len(set(y_val.tolist())) > 1 else None
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_log_loss": float(val_loss),
            "val_auc": val_auc,
        })
        if patience > 0 and no_improve >= patience:
            print(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    wrapper = _LegacyPCTRModel(encoder, model, device)
    val_probs = wrapper.predict_proba(val_texts, val_nums)
    metrics = {
        "text_encoder": "tfidf",
        "val_log_loss": float(log_loss(y_val, np.clip(val_probs, 1e-5, 1 - 1e-5), labels=[0, 1])),
        "val_auc": float(roc_auc_score(y_val, val_probs)) if len(set(y_val.tolist())) > 1 else None,
        "val_positive_rate": float(y_val.mean()),
        "n_train": int(len(train_y)),
        "n_val": int(len(val_y)),
        "input_dim": int(X_train.shape[1]),
        "epoch_history": epoch_history,
        "step_history": step_history,
    }
    return wrapper, metrics


def _train_pctr_embedding(
    train_texts, train_nums, train_y, val_texts, val_nums, val_y,
    device, hidden_dim, epochs, batch_size, lr, embedding_model, patience,
):
    encoder = EmbeddingTextEncoder(model_name=embedding_model, device=device)
    num_dim = len(train_nums[0]) if train_nums and train_nums[0] else 0
    input_dim = encoder.input_dim + num_dim

    model = PCTRNet(input_dim, hidden_dim=hidden_dim).to(device)
    # Freeze embedding — only train MLP head to prevent overfitting
    encoder.eval_mode()
    for p in encoder.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    ds = TextNumDataset(train_texts, train_nums, [np.asarray(train_y, dtype=np.float32)])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_text_num)

    y_val_np = np.asarray(val_y, dtype=np.float32)
    best_state = None
    best_val = float("inf")
    no_improve = 0
    epoch_history = []
    step_history = []
    global_step = 0

    for epoch in tqdm(range(epochs), desc="train pCTR (embedding)", leave=False):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for texts_b, nums_b, (yb,) in dl:
            nums_b = nums_b.to(device)
            yb = yb.to(device)
            text_emb = encoder.encode(texts_b).float()
            xb = torch.cat([text_emb, nums_b], dim=1)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step_loss = loss.item()
            epoch_loss += step_loss
            n_batches += 1
            global_step += 1
            step_history.append({"step": global_step, "loss": float(step_loss)})
        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        encoder.eval_mode()
        val_probs = _eval_embedding_pctr(encoder, model, val_texts, val_nums, device)
        val_loss = log_loss(y_val_np, np.clip(val_probs, 1e-5, 1 - 1e-5), labels=[0, 1])
        val_auc = float(roc_auc_score(y_val_np, val_probs)) if len(set(y_val_np.tolist())) > 1 else None
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_log_loss": float(val_loss),
            "val_auc": val_auc,
        })
        if patience > 0 and no_improve >= patience:
            print(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    wrapper = _LegacyPCTRModel(encoder, model, device)
    val_probs = wrapper.predict_proba(val_texts, val_nums)
    metrics = {
        "text_encoder": "qwen3_embedding",
        "embedding_model": embedding_model,
        "val_log_loss": float(log_loss(y_val_np, np.clip(val_probs, 1e-5, 1 - 1e-5), labels=[0, 1])),
        "val_auc": float(roc_auc_score(y_val_np, val_probs)) if len(set(y_val_np.tolist())) > 1 else None,
        "val_positive_rate": float(y_val_np.mean()),
        "n_train": int(len(train_y)),
        "n_val": int(len(val_y)),
        "input_dim": int(input_dim),
        "epoch_history": epoch_history,
        "step_history": step_history,
    }
    return wrapper, metrics


def _eval_embedding_pctr(encoder, model, texts, nums, device, batch_size=64):
    X_num = np.asarray(nums, dtype=np.float32)
    if X_num.ndim == 1:
        X_num = X_num.reshape(-1, 1)
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            text_emb = encoder.encode(texts[i:i + batch_size]).float()
            num_b = torch.tensor(X_num[i:i + batch_size], dtype=torch.float32, device=device)
            xb = torch.cat([text_emb, num_b], dim=1)
            p = torch.sigmoid(model(xb)).cpu().numpy()
            probs.append(p)
    return np.concatenate(probs, axis=0)


# ---------------------------------------------------------------------------
# Training: Judge
# ---------------------------------------------------------------------------

def train_judge_text_model(
    train_texts: List[str],
    train_nums: List[List[float]],
    train_click: List[float],
    train_gamma: List[float],
    val_texts: List[str],
    val_nums: List[List[float]],
    val_click: List[float],
    val_gamma: List[float],
    device: torch.device,
    max_features: int = 4096,
    hidden_dim: int = 256,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    gamma_loss_weight: float = 1.0,
    text_encoder_type: str = "qwen3_embedding",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    patience: int = 5,
) -> Tuple[JudgeModel, Dict[str, Any]]:
    if text_encoder_type == "tfidf":
        return _train_judge_tfidf(
            train_texts, train_nums, train_click, train_gamma,
            val_texts, val_nums, val_click, val_gamma,
            device, max_features, hidden_dim, epochs, batch_size, lr, gamma_loss_weight, patience,
        )
    return _train_judge_embedding(
        train_texts, train_nums, train_click, train_gamma,
        val_texts, val_nums, val_click, val_gamma,
        device, hidden_dim, epochs, batch_size, lr, gamma_loss_weight, embedding_model, patience,
    )


def _train_judge_tfidf(
    train_texts, train_nums, train_click, train_gamma,
    val_texts, val_nums, val_click, val_gamma,
    device, max_features, hidden_dim, epochs, batch_size, lr, gamma_loss_weight, patience,
):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1)
    vectorizer.fit(train_texts)
    encoder = TfidfTextEncoder(vectorizer)

    X_train = make_features(vectorizer, train_texts, train_nums)
    X_val = make_features(vectorizer, val_texts, val_nums)
    yc_train = np.asarray(train_click, dtype=np.float32)
    yg_train = np.asarray(train_gamma, dtype=np.float32)
    yc_val = np.asarray(val_click, dtype=np.float32)
    yg_val = np.asarray(val_gamma, dtype=np.float32)

    model = JudgeNet(X_train.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    ds = TensorDataset(torch.tensor(X_train), torch.tensor(yc_train), torch.tensor(yg_train))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val = float("inf")
    no_improve = 0
    epoch_history = []
    step_history = []
    global_step = 0
    for epoch in tqdm(range(epochs), desc="train judge (tfidf)", leave=False):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, ycb, ygb in dl:
            xb = xb.to(device).float(); ycb = ycb.to(device).float(); ygb = ygb.to(device).float()
            logit, gamma = model(xb)
            loss = bce(logit, ycb) + gamma_loss_weight * mse(gamma, ygb)
            opt.zero_grad(); loss.backward(); opt.step()
            step_loss = loss.item()
            epoch_loss += step_loss
            n_batches += 1
            global_step += 1
            step_history.append({"step": global_step, "loss": float(step_loss)})
        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            logit, gamma = model(torch.tensor(X_val, dtype=torch.float32, device=device))
            click_probs = torch.sigmoid(logit).cpu().numpy()
            gamma_pred = gamma.cpu().numpy()
            val_click_loss = log_loss(yc_val, np.clip(click_probs, 1e-5, 1 - 1e-5), labels=[0, 1])
            val_gamma_mae = mean_absolute_error(yg_val, gamma_pred)
            val_loss = val_click_loss + val_gamma_mae
            val_click_auc = float(roc_auc_score(yc_val, click_probs)) if len(set(yc_val.tolist())) > 1 else None
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_click_log_loss": float(val_click_loss),
            "val_gamma_mae": float(val_gamma_mae),
            "val_click_auc": val_click_auc,
        })
        if patience > 0 and no_improve >= patience:
            print(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    wrapper = _LegacyJudgeModel(encoder, model, device)
    val_click_prob, val_gamma_pred = wrapper.predict(val_texts, val_nums)
    metrics = {
        "text_encoder": "tfidf",
        "val_click_log_loss": float(log_loss(yc_val, np.clip(val_click_prob, 1e-5, 1 - 1e-5), labels=[0, 1])),
        "val_click_auc": float(roc_auc_score(yc_val, val_click_prob)) if len(set(yc_val.tolist())) > 1 else None,
        "val_gamma_mae": float(mean_absolute_error(yg_val, val_gamma_pred)),
        "val_click_positive_rate": float(yc_val.mean()),
        "n_train": int(len(train_click)),
        "n_val": int(len(val_click)),
        "input_dim": int(X_train.shape[1]),
        "epoch_history": epoch_history,
        "step_history": step_history,
    }
    return wrapper, metrics


def _train_judge_embedding(
    train_texts, train_nums, train_click, train_gamma,
    val_texts, val_nums, val_click, val_gamma,
    device, hidden_dim, epochs, batch_size, lr, gamma_loss_weight, embedding_model, patience,
):
    encoder = EmbeddingTextEncoder(model_name=embedding_model, device=device)
    num_dim = len(train_nums[0]) if train_nums and train_nums[0] else 0
    input_dim = encoder.input_dim + num_dim

    model = JudgeNet(input_dim, hidden_dim=hidden_dim).to(device)
    # Freeze embedding — only train MLP head to prevent overfitting
    encoder.eval_mode()
    for p in encoder.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    mse_fn = nn.MSELoss()

    yc_train = np.asarray(train_click, dtype=np.float32)
    yg_train = np.asarray(train_gamma, dtype=np.float32)
    yc_val = np.asarray(val_click, dtype=np.float32)
    yg_val = np.asarray(val_gamma, dtype=np.float32)

    ds = TextNumDataset(train_texts, train_nums, [yc_train, yg_train])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_text_num)

    best_state = None
    best_val = float("inf")
    epoch_history = []
    step_history = []
    global_step = 0

    for epoch in tqdm(range(epochs), desc="train judge (embedding)", leave=False):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for texts_b, nums_b, (ycb, ygb) in dl:
            nums_b = nums_b.to(device)
            ycb = ycb.to(device)
            ygb = ygb.to(device)
            text_emb = encoder.encode(texts_b).float()
            xb = torch.cat([text_emb, nums_b], dim=1)
            logit, gamma = model(xb)
            loss = bce(logit, ycb) + gamma_loss_weight * mse_fn(gamma, ygb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step_loss = loss.item()
            epoch_loss += step_loss
            n_batches += 1
            global_step += 1
            step_history.append({"step": global_step, "loss": float(step_loss)})
        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        encoder.eval_mode()
        val_click_prob, val_gamma_pred = _eval_embedding_judge(
            encoder, model, val_texts, val_nums, device,
        )
        val_click_loss = log_loss(yc_val, np.clip(val_click_prob, 1e-5, 1 - 1e-5), labels=[0, 1])
        val_gamma_mae = mean_absolute_error(yg_val, val_gamma_pred)
        val_loss = val_click_loss + val_gamma_mae
        val_click_auc = float(roc_auc_score(yc_val, val_click_prob)) if len(set(yc_val.tolist())) > 1 else None
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_click_log_loss": float(val_click_loss),
            "val_gamma_mae": float(val_gamma_mae),
            "val_click_auc": val_click_auc,
        })
        if patience > 0 and no_improve >= patience:
            print(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    wrapper = _LegacyJudgeModel(encoder, model, device)
    val_click_prob, val_gamma_pred = wrapper.predict(val_texts, val_nums)
    metrics = {
        "text_encoder": "qwen3_embedding",
        "embedding_model": embedding_model,
        "val_click_log_loss": float(log_loss(yc_val, np.clip(val_click_prob, 1e-5, 1 - 1e-5), labels=[0, 1])),
        "val_click_auc": float(roc_auc_score(yc_val, val_click_prob)) if len(set(yc_val.tolist())) > 1 else None,
        "val_gamma_mae": float(mean_absolute_error(yg_val, val_gamma_pred)),
        "val_click_positive_rate": float(yc_val.mean()),
        "n_train": int(len(train_click)),
        "n_val": int(len(val_click)),
        "input_dim": int(input_dim),
        "epoch_history": epoch_history,
        "step_history": step_history,
    }
    return wrapper, metrics


def _eval_embedding_judge(encoder, model, texts, nums, device, batch_size=64, user_ids=None):
    X_num = np.asarray(nums, dtype=np.float32)
    if X_num.ndim == 1:
        X_num = X_num.reshape(-1, 1)
    uid_arr = np.asarray(user_ids if user_ids is not None else [0] * len(texts), dtype=np.int64)
    has_user_emb = hasattr(model, 'user_embedding')
    clicks, gammas = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            text_emb = encoder.encode(texts[i:i + batch_size]).float()
            num_b = torch.tensor(X_num[i:i + batch_size], dtype=torch.float32, device=device)
            xb = torch.cat([text_emb, num_b], dim=1)
            if has_user_emb:
                uid_b = torch.tensor(uid_arr[i:i + batch_size], dtype=torch.long, device=device)
                logit, gamma = model(xb, user_ids=uid_b)
            else:
                logit, gamma = model(xb)
            clicks.append(torch.sigmoid(logit).cpu().numpy())
            gammas.append(gamma.cpu().numpy())
    return np.concatenate(clicks, axis=0), np.concatenate(gammas, axis=0)


def precompute_embedding_features(
    encoder: "EmbeddingTextEncoder",
    texts: List[str],
    nums: List[List[float]],
    batch_size: int = 64,
    desc: str = "encode",
) -> np.ndarray:
    """Encode all texts ONCE into a dense [N, embed_dim(+num_dim)] matrix.

    The embedding is frozen, so a given text maps to the same vector every
    epoch — encoding per-batch per-epoch (the old path) re-ran the 0.6B model
    millions of times. We encode once here and reuse the cached features,
    making the embedding path numerically identical to before but ~30-50x
    faster (no behavioural change, pure caching).
    """
    encoder.eval_mode()
    text_parts = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
            emb = encoder.encode(texts[i:i + batch_size]).float().cpu().numpy()
            text_parts.append(emb)
    X_text = np.concatenate(text_parts, axis=0).astype(np.float32)
    if nums and nums[0]:
        X_num = np.asarray(nums, dtype=np.float32)
        if X_num.ndim == 1:
            X_num = X_num.reshape(-1, 1)
        return np.concatenate([X_text, X_num], axis=1).astype(np.float32)
    return X_text


# ---------------------------------------------------------------------------
# Unified User Model (replaces separate PCTRNet + JudgeNet)
# ---------------------------------------------------------------------------

class UserModelNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 n_users: int = 5000, user_emb_dim: int = 32):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, user_emb_dim, padding_idx=0)
        self.user_emb_dim = user_emb_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim + user_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.click_head = nn.Linear(hidden_dim, 1)
        self.gamma_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, user_ids=None):
        if user_ids is not None:
            u_emb = self.user_embedding(user_ids)
        else:
            u_emb = torch.zeros(x.size(0), self.user_emb_dim, device=x.device)
        h = self.backbone(torch.cat([x, u_emb], dim=1))
        click_logit = self.click_head(h).squeeze(-1)
        gamma = torch.sigmoid(self.gamma_head(h).squeeze(-1))
        return click_logit, gamma



class UnifiedUserModel:
    def __init__(self, encoder, model, device, user_id_map=None):
        self.encoder = encoder
        self.model = model
        self.device = device
        self.user_id_map = user_id_map or {}
        self.model.to(device)
        self.model.eval()

    @property
    def input_dim(self):
        return self.model.backbone[0].in_features - self.model.user_emb_dim

    def _make_text_tensor(self, texts, nums, batch_start, batch_end):
        import numpy as np
        text_batch = texts[batch_start:batch_end]
        text_emb = self.encoder.encode(text_batch)
        if isinstance(text_emb, np.ndarray):
            text_emb = torch.tensor(text_emb, dtype=torch.float32, device=self.device)
        else:
            text_emb = text_emb.float().to(self.device)
        if nums and nums[0]:
            X_num = np.asarray(nums[batch_start:batch_end], dtype=np.float32)
            num_batch = torch.tensor(X_num, dtype=torch.float32, device=self.device)
            return torch.cat([text_emb, num_batch], dim=1)
        return text_emb

    def predict_proba(self, texts, nums, batch_size=256):
        """pCTR mode: no user_ids (zero embedding). For auction scoring."""
        self.encoder.eval_mode()
        self.model.eval()
        clicks = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                xb = self._make_text_tensor(texts, nums, i, i + batch_size)
                logit, _ = self.model(xb, user_ids=None)
                clicks.append(torch.sigmoid(logit).detach().cpu().numpy())
        return np.concatenate(clicks, axis=0)

    def predict(self, texts, nums, batch_size=256, user_ids=None):
        """Judge mode: with optional user_ids for user-aware scoring."""
        self.encoder.eval_mode()
        self.model.eval()
        uid_tensor = None
        if user_ids is not None:
            uid_tensor = torch.tensor(user_ids, dtype=torch.long, device=self.device)
        clicks, gammas = [], []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                end = min(i + batch_size, len(texts))
                xb = self._make_text_tensor(texts, nums, i, end)
                uid_batch = uid_tensor[i:end] if uid_tensor is not None else None
                logit, gamma = self.model(xb, user_ids=uid_batch)
                clicks.append(torch.sigmoid(logit).detach().cpu().numpy())
                gammas.append(gamma.detach().cpu().numpy())
        return np.concatenate(clicks, axis=0), np.concatenate(gammas, axis=0)

    def save(self, output_dir):
        from pathlib import Path
        output_dir = Path(output_dir)
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "n_users": self.model.user_embedding.num_embeddings,
            "user_emb_dim": self.model.user_emb_dim,
        }, output_dir / "user_model.pt")
        self.encoder.save(output_dir)

    @staticmethod
    def load(output_dir, device, hidden_dim=256):
        from pathlib import Path
        output_dir = Path(output_dir)
        encoder = load_text_encoder(output_dir, device)
        ckpt = torch.load(output_dir / "user_model.pt", map_location=device)
        n_users = ckpt.get("n_users", 5000)
        user_emb_dim = ckpt.get("user_emb_dim", 32)
        model = UserModelNet(input_dim=ckpt["input_dim"], hidden_dim=hidden_dim,
                             n_users=n_users, user_emb_dim=user_emb_dim)
        model.load_state_dict(ckpt["state_dict"])
        return UnifiedUserModel(encoder, model, device)


def train_unified_user_model(
    pctr_texts, pctr_nums, pctr_y_click,
    judge_texts, judge_nums, judge_y_click, judge_y_gamma,
    val_pctr_texts, val_pctr_nums, val_pctr_y_click,
    val_judge_texts, val_judge_nums, val_judge_y_click, val_judge_y_gamma,
    device, max_features=1024, hidden_dim=256, epochs=30, batch_size=64,
    lr=1e-3, gamma_loss_weight=1.0,
    text_encoder_type="tfidf", embedding_model="", patience=5,
    judge_user_ids=None, val_judge_user_ids=None, n_users=5000,
):
    all_train_texts = pctr_texts + judge_texts
    all_train_nums = pctr_nums + judge_nums
    all_y_click = list(pctr_y_click) + list(judge_y_click)
    all_y_gamma = [0.0] * len(pctr_y_click) + list(judge_y_gamma)
    all_has_gamma = [0.0] * len(pctr_y_click) + [1.0] * len(judge_y_click)
    # user_ids: pCTR samples get 0 (zero embedding), Judge samples get real user_id
    all_user_ids = [0] * len(pctr_y_click) + list(judge_user_ids or [0] * len(judge_y_click))

    if text_encoder_type == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1)
        vectorizer.fit(all_train_texts)
        encoder = TfidfTextEncoder(vectorizer)
        X_train = make_features(vectorizer, all_train_texts, all_train_nums)
        X_val_pctr = make_features(vectorizer, val_pctr_texts, val_pctr_nums)
        X_val_judge = make_features(vectorizer, val_judge_texts, val_judge_nums)
        input_dim = X_train.shape[1]
    else:
        encoder = EmbeddingTextEncoder(model_name=embedding_model, device=device)
        encoder.eval_mode()
        for p in encoder.parameters():
            p.requires_grad = False
        # Frozen embedding: encode every text ONCE up front, then train on the
        # cached dense features exactly like the tfidf path. Avoids re-running
        # the 0.6B model every epoch (the cause of ~193s/epoch).
        X_train = precompute_embedding_features(encoder, all_train_texts, all_train_nums, desc="encode train")
        X_val_pctr = precompute_embedding_features(encoder, val_pctr_texts, val_pctr_nums, desc="encode val-pctr")
        X_val_judge = precompute_embedding_features(encoder, val_judge_texts, val_judge_nums, desc="encode val-judge")
        input_dim = X_train.shape[1]

    model = UserModelNet(input_dim, hidden_dim=hidden_dim, n_users=n_users).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    mse_fn = nn.MSELoss(reduction='none')

    val_pctr_y = np.asarray(val_pctr_y_click, dtype=np.float32)
    val_judge_yc = np.asarray(val_judge_y_click, dtype=np.float32)
    val_judge_yg = np.asarray(val_judge_y_gamma, dtype=np.float32)

    ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(np.asarray(all_y_click, dtype=np.float32)),
        torch.tensor(np.asarray(all_y_gamma, dtype=np.float32)),
        torch.tensor(np.asarray(all_has_gamma, dtype=np.float32)),
        torch.tensor(np.asarray(all_user_ids, dtype=np.int64)),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val = float("inf")
    no_improve = 0
    epoch_history = []
    step_history = []
    global_step = 0

    for epoch in tqdm(range(epochs), desc="train unified user model", leave=False):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yc_b, yg_b, hg_b, uid_b in dl:
            xb = xb.to(device); yc_b = yc_b.to(device)
            yg_b = yg_b.to(device); hg_b = hg_b.to(device)
            uid_b = uid_b.to(device)
            click_logit, gamma_pred = model(xb, user_ids=uid_b)
            click_loss = bce(click_logit, yc_b).mean()
            gamma_loss = (mse_fn(gamma_pred, yg_b) * hg_b).sum() / max(1.0, hg_b.sum().item())
            loss = click_loss + gamma_loss_weight * gamma_loss
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); n_batches += 1
            global_step += 1
            step_history.append({"step": global_step, "loss": loss.item()})

        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            logits_p, _ = model(torch.tensor(X_val_pctr, dtype=torch.float32, device=device))
            pctr_click = torch.sigmoid(logits_p).cpu().numpy()
            val_judge_uid_t = torch.tensor(
                np.asarray(val_judge_user_ids if val_judge_user_ids is not None else [0] * len(val_judge_texts), dtype=np.int64),
                dtype=torch.long, device=device,
            )
            logits_j, judge_gamma = model(
                torch.tensor(X_val_judge, dtype=torch.float32, device=device),
                user_ids=val_judge_uid_t,
            )
            judge_click = torch.sigmoid(logits_j).cpu().numpy()
            judge_gamma = judge_gamma.cpu().numpy()

        pctr_val_loss = log_loss(val_pctr_y, np.clip(pctr_click, 1e-5, 1-1e-5), labels=[0,1])
        pctr_val_auc = float(roc_auc_score(val_pctr_y, pctr_click)) if len(set(val_pctr_y.tolist())) > 1 else None
        judge_val_loss = log_loss(val_judge_yc, np.clip(judge_click, 1e-5, 1-1e-5), labels=[0,1])
        judge_val_auc = float(roc_auc_score(val_judge_yc, judge_click)) if len(set(val_judge_yc.tolist())) > 1 else None
        gamma_mae = float(mean_absolute_error(val_judge_yg, judge_gamma))

        val_loss = pctr_val_loss + judge_val_loss + gamma_mae
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        epoch_history.append({
            "epoch": epoch + 1, "train_loss": float(train_loss),
            "pctr_val_loss": float(pctr_val_loss), "pctr_val_auc": pctr_val_auc,
            "judge_val_loss": float(judge_val_loss), "judge_val_auc": judge_val_auc,
            "gamma_mae": gamma_mae,
        })
        if patience > 0 and no_improve >= patience:
            print(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    wrapper = UnifiedUserModel(encoder, model, device)
    metrics = {
        "text_encoder": text_encoder_type,
        "embedding_model": embedding_model if text_encoder_type != "tfidf" else None,
        "pctr_val_auc": pctr_val_auc, "pctr_val_loss": float(pctr_val_loss),
        "judge_val_auc": judge_val_auc, "judge_val_loss": float(judge_val_loss),
        "gamma_mae": gamma_mae,
        "n_pctr_train": len(pctr_texts), "n_judge_train": len(judge_texts),
        "input_dim": int(input_dim),
        "epoch_history": epoch_history, "step_history": step_history,
    }
    return wrapper, metrics


# Preserve the original lightweight single-head wrappers (compatible with the
# legacy PCTRNet/JudgeNet that have forward(x) and a single return value) before
# the UnifiedUserModel aliases below shadow these names. The legacy ablation /
# closed-loop training paths construct these explicitly.
_LegacyPCTRModel = PCTRModel
_LegacyJudgeModel = JudgeModel


# Backward-compatible aliases
PCTRModel = UnifiedUserModel
JudgeModel = UnifiedUserModel
