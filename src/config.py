from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    seed: int = 42
    test_ratio: float = 0.2
    max_tfidf_features: int = 4096
    text_encoder: str = "qwen3_embedding"  # "qwen3_embedding" or "tfidf"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    pctr_hidden_dim: int = 256
    judge_hidden_dim: int = 256
    stop_hidden_dim: int = 128
    batch_size: int = 64
    lr: float = 1e-3
    discount: float = 0.95
    lambda_gamma: float = 0.25
    reserve_price: float = 0.0
    force_auction_last_turn: bool = True
    device: str = "auto"
