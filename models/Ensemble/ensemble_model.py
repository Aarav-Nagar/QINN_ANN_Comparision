import torch
import torch.nn as nn
from typing import List


class ResNetBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class AttentionPool(nn.Module):
    def __init__(self, dim, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        out = self.norm(x + attn_out)
        return out.mean(dim=1)


class EnsembleModel(nn.Module):
    """Ensemble model combining ResNet blocks, LSTM, Attention and optional GBDT features."""

    def __init__(self, input_dim: int, gbdt_dim: int = 0, hidden_dims: List[int] = None, dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.input_dim = input_dim
        self.gbdt_dim = gbdt_dim

        proj_dim = hidden_dims[0]
        self.proj = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.GELU(), nn.LayerNorm(proj_dim))

        self.resblocks = nn.Sequential(*[ResNetBlock(proj_dim, proj_dim * 2, dropout=dropout) for _ in range(2)])

        self.lstm = nn.LSTM(proj_dim, proj_dim // 2, num_layers=1, batch_first=True, bidirectional=True)

        self.attn_pool = AttentionPool(proj_dim, n_heads=4)

        fusion_dim = proj_dim + proj_dim + (gbdt_dim if gbdt_dim > 0 else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dims[1]),
            nn.GELU(),
            nn.LayerNorm(hidden_dims[1]),
            nn.Dropout(dropout),
        )

        self.reg_head = nn.Sequential(nn.Linear(hidden_dims[1], 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
        self.cls_head = nn.Sequential(nn.Linear(hidden_dims[1], 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 2))

    def forward(self, x, gbdt_feats: torch.Tensor = None):
        proj = self.proj(x)
        res = self.resblocks(proj)
        seq = proj.unsqueeze(1)
        lstm_out, _ = self.lstm(seq)
        lstm_feat = lstm_out.squeeze(1)
        attn_input = torch.cat([proj.unsqueeze(1), res.unsqueeze(1)], dim=1)
        attn_feat = self.attn_pool(attn_input)
        parts = [attn_feat, lstm_feat]
        if gbdt_feats is not None:
            parts.append(gbdt_feats)
        fused = torch.cat(parts, dim=1)
        out = self.fusion(fused)
        reg = self.reg_head(out).squeeze(-1)
        cls_logits = self.cls_head(out)
        return {"regression": reg, "classification": cls_logits, "features": out}
