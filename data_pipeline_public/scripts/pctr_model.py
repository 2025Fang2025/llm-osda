"""
pctr_model.py
pCTR 模型: 预测广告在给定对话回复中的点击率

架构 (对齐论文公式 3):
  pCTR(x, h, a_i, y) = σ(MLP([φ_text(x); φ_text(y); e_h; e_a]))

  φ_text: 冻结的预训练文本编码器 (如 bge-large-zh-v1.5)
  e_h:    可学习的用户 profile embedding (通过 φ_text 编码 profile 文本)
  e_a:    可学习的广告 embedding (通过 φ_text 编码广告文本)
  MLP:    2层 MLP → sigmoid

多轮适配:
  x = 完整对话历史 (system + user/assistant turns 到当前)
  y = 当前 assistant 回复

用途:
  GRPO 训练时的 reward model 组件
  reward = bid × pCTR(x, h, a_i, y) × 1(a_i ∈ y) + λ × quality_score
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class PCTRModel(nn.Module):
    """
    pCTR model for ad click prediction in conversational ad insertion.
    
    Input:
        context_emb:  [B, D] - encoded dialogue history
        response_emb: [B, D] - encoded assistant response
        profile_emb:  [B, D] - encoded user profile
        ad_emb:       [B, D] - encoded ad content
    
    Output:
        pctr: [B, 1] - predicted click-through rate
    """
    
    def __init__(self, text_dim=1024, hidden_dim=512, dropout=0.1):
        super().__init__()
        
        # 4 个 text embedding 拼接 → MLP
        input_dim = text_dim * 4
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, context_emb, response_emb, profile_emb, ad_emb):
        """
        All inputs: [B, D] tensors from frozen text encoder
        Returns: [B] click probabilities
        """
        combined = torch.cat([context_emb, response_emb, profile_emb, ad_emb], dim=-1)
        logits = self.mlp(combined).squeeze(-1)  # [B]
        return torch.sigmoid(logits)


class PCTRDataset(Dataset):
    """Dataset for pCTR training from pctr_train.jsonl"""
    
    def __init__(self, data_path, text_encoder=None, max_context_len=512, max_response_len=256):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        
        self.text_encoder = text_encoder
        self.max_context_len = max_context_len
        self.max_response_len = max_response_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "context": s["context"][:2000],         # 截断过长的对话
            "response": s["response"][:1000],
            "profile": s["profile"][:500],
            "ad_text": "{} — {}".format(s["ad_title"], s["ad_text"])[:200],
            "clicked": float(s["clicked"]),
            "exposed": float(s["exposed"]),
            "bid": float(s.get("bid", 0)),
        }


def train_pctr(
    data_path,
    encoder_name="BAAI/bge-large-zh-v1.5",
    epochs=10,
    batch_size=64,
    lr=1e-3,
    device="cuda",
    save_path="checkpoints/pctr_model.pt",
):
    """训练 pCTR 模型"""
    from sentence_transformers import SentenceTransformer
    
    # 冻结的文本编码器
    print("Loading text encoder: {}".format(encoder_name))
    text_encoder = SentenceTransformer(encoder_name, device=device)
    text_encoder.eval()
    text_dim = text_encoder.get_sentence_embedding_dimension()
    
    # 数据集
    dataset = PCTRDataset(data_path)
    
    # 划分 train/val (90/10)
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # 模型
    model = PCTRModel(text_dim=text_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    print("Training pCTR model: {} samples, {} epochs".format(len(dataset), epochs))
    
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            with torch.no_grad():
                ctx_emb = text_encoder.encode(batch["context"], convert_to_tensor=True)
                resp_emb = text_encoder.encode(batch["response"], convert_to_tensor=True)
                prof_emb = text_encoder.encode(batch["profile"], convert_to_tensor=True)
                ad_emb = text_encoder.encode(batch["ad_text"], convert_to_tensor=True)
            
            labels = batch["clicked"].to(device)
            
            preds = model(ctx_emb, resp_emb, prof_emb, ad_emb)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = total_loss / max(n_batches, 1)
        
        # Validate
        model.eval()
        val_loss = 0
        val_batches = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                ctx_emb = text_encoder.encode(batch["context"], convert_to_tensor=True)
                resp_emb = text_encoder.encode(batch["response"], convert_to_tensor=True)
                prof_emb = text_encoder.encode(batch["profile"], convert_to_tensor=True)
                ad_emb = text_encoder.encode(batch["ad_text"], convert_to_tensor=True)
                
                labels = batch["clicked"].to(device)
                preds = model(ctx_emb, resp_emb, prof_emb, ad_emb)
                loss = criterion(preds, labels)
                
                val_loss += loss.item()
                val_batches += 1
                
                predicted = (preds > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / max(val_batches, 1)
        val_acc = correct / max(total, 1)
        
        print("Epoch {}/{}: train_loss={:.4f}, val_loss={:.4f}, val_acc={:.4f}".format(
            epoch + 1, epochs, avg_train_loss, avg_val_loss, val_acc,
        ))
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "text_dim": text_dim,
                "encoder_name": encoder_name,
            }, save_path)
            print("  -> Saved best model (val_loss={:.4f})".format(avg_val_loss))
    
    print("Training complete. Best val_loss: {:.4f}".format(best_val_loss))
    return model


class PCTRScorer:
    """
    GRPO 训练时的 pCTR 打分器
    加载训练好的 pCTR 模型，给 (context, response, profile, ad) 打 click 概率分
    """
    
    def __init__(self, checkpoint_path, device="cuda"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        from sentence_transformers import SentenceTransformer
        self.text_encoder = SentenceTransformer(
            checkpoint["encoder_name"], device=device
        )
        self.text_encoder.eval()
        
        self.model = PCTRModel(text_dim=checkpoint["text_dim"]).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def score(self, context, response, profile, ad_text):
        """
        单条打分
        Returns: float, click probability [0, 1]
        """
        ctx_emb = self.text_encoder.encode([context], convert_to_tensor=True)
        resp_emb = self.text_encoder.encode([response], convert_to_tensor=True)
        prof_emb = self.text_encoder.encode([profile], convert_to_tensor=True)
        ad_emb = self.text_encoder.encode([ad_text], convert_to_tensor=True)
        
        pctr = self.model(ctx_emb, resp_emb, prof_emb, ad_emb)
        return pctr.item()
    
    @torch.no_grad()
    def batch_score(self, contexts, responses, profiles, ad_texts):
        """
        批量打分
        All inputs: List[str] of same length
        Returns: List[float], click probabilities
        """
        ctx_emb = self.text_encoder.encode(contexts, convert_to_tensor=True)
        resp_emb = self.text_encoder.encode(responses, convert_to_tensor=True)
        prof_emb = self.text_encoder.encode(profiles, convert_to_tensor=True)
        ad_emb = self.text_encoder.encode(ad_texts, convert_to_tensor=True)
        
        pctrs = self.model(ctx_emb, resp_emb, prof_emb, ad_emb)
        return pctrs.cpu().tolist()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/pctr_train.jsonl")
    parser.add_argument("--encoder", type=str, default="BAAI/bge-large-zh-v1.5")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save", type=str, default="checkpoints/pctr_model.pt")
    args = parser.parse_args()
    
    train_pctr(
        data_path=args.data,
        encoder_name=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_path=args.save,
    )
