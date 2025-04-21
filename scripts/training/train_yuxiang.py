import math
import copy
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import wandb

config = {
    "model": "lstm", # backbone type
    "trend_loss": True,
    "multi_loss": True,
    "model_name": "final_baseline",

    "training_mode": "single_output", # 'single' | 'separate' | 'single_output'

    "d_model": 1024,
    "prediction_dim": 1, # [T+1, T+2, T+3]  for separate training mode, use 3
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epoch": 100,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "positional_encoding": {"max_len": 50, "enable": True},
    "ts": {
        "input_dim": 6,
        "topic_emb_dim": 768, # overwritten
        "nhead": 32,
        "num_layers": 10,
    },
    "data": {
        "path": "/mnt",
        "input_len": 10,
        "normalize": True,
        "use_agg": True,
        "trend_eps": 0.01,
    },
}

wandb.login()
sbert = SentenceTransformer("/mnt/allenai-specter")
config["ts"]["topic_emb_dim"] = sbert.get_sentence_embedding_dimension()

class TopicTimeSeriesDataset(Dataset):
    def __init__(self, cfg, sbert_model, file_path):
        self.cfg = cfg
        self.df = pd.read_csv(Path(cfg["data"]["path"]) / f"{file_path}_new.csv", index_col=0)
        self.normalize = cfg["data"]["normalize"]
        self.input_len = cfg["data"]["input_len"]
        self.target_len = cfg["prediction_dim"]
        self.use_agg = cfg["data"].get("use_agg", False)
        self.eps = cfg["data"].get("trend_eps", 0.01)

        self.sbert = sbert_model
        self.topic_list = self.df.index.tolist()
        self.series = self.df.values.astype(float)

        self.topic_embeddings = torch.tensor(
            self.sbert.encode(self.topic_list, normalize_embeddings=True), dtype=torch.float32
        )

        self.samples: List[tuple[int, int]] = []
        if file_path == 'train_data':
            self.series = self.series[:, :-3]
        elif file_path == 'val_data':
            self.series = self.series[:, :-2]
        elif file_path == 'test_data':
            self.series = self.series[:, :-1]
        if config["training_mode"] == "single_output":
            for topic_idx in range(len(self.df)):
                if file_path == 'train_data':
                    seq_len = self.series.shape[1]
                    max_start = seq_len - self.input_len - self.target_len + 1
                    assert max_start >= 0, f"Not enough data to create sliding windows with start idx {max_start}"
                    for start_idx in range(max_start):
                        self.samples.append((topic_idx, start_idx))
                elif file_path == 'val_data':
                    seq_len = self.series.shape[1]
                    max_start = seq_len - self.input_len - self.target_len + 1
                    assert max_start >= 0, f"Not enough data to create sliding windows with start idx {max_start}"
                    self.samples.append((topic_idx, max_start - 1))
                elif file_path == 'test_data':
                    seq_len = self.series.shape[1]
                    max_start = seq_len - self.input_len - self.target_len + 1
                    assert max_start >= 0, f"Not enough data to create sliding windows with start idx {max_start}"
                    self.samples.append((topic_idx, max_start - 1))
        else:
            for topic_idx in range(len(self.df)):
                seq_len = self.series.shape[1]
                max_start = seq_len - self.input_len - self.target_len + 1
                for start_idx in range(max_start):
                    self.samples.append((topic_idx, start_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        topic_idx, start_idx = self.samples[idx]
        topic_emb = self.topic_embeddings[topic_idx]
        full_seq = torch.tensor(self.series[topic_idx], dtype=torch.float32)

        x = full_seq[start_idx : start_idx + self.input_len]
        y = full_seq[start_idx + self.input_len : start_idx + self.input_len + self.target_len]

        # engineered features
        if self.use_agg:
            x_delta = x[1:] - x[:-1]
            x_delta = torch.cat([torch.tensor([0.0]), x_delta])
            features = [
                x.unsqueeze(-1),
                x_delta.unsqueeze(-1),
                x.mean().repeat(self.input_len, 1),
                x.std().repeat(self.input_len, 1),
                (x[-1] - x[-2]).repeat(self.input_len, 1),
                torch.linspace(-1, 1, steps=self.input_len).unsqueeze(-1),
            ]
            x = torch.cat(features, dim=-1)
        else:
            x = x.unsqueeze(-1)

        # growth & trend labels
        x_for_growth = x[:, 0] if self.use_agg else x.squeeze(-1)
        growth = (y[-1] - x_for_growth[-1]) / self.target_len
        delta = y[-1] - x_for_growth[-1]
        if delta > self.eps:
            trend_label = 2
        elif delta < -self.eps:
            trend_label = 0
        else:
            trend_label = 1

        # normalisation
        if self.normalize:
            mean, std = x.mean(), x.std()
            x = (x - mean) / (std + 1e-6)
            y = (y - mean) / (std + 1e-6)
            growth = growth / (std + 1e-6)
        else:
            mean, std = 0.0, 1.0

        return (
            x.clone().detach(),
            topic_emb.clone().detach(),
            y.clone().detach(),
            growth.clone().detach(),
            torch.tensor(trend_label),
            mean.clone().detach(),
            std.clone().detach(),
        )


def collate_fn(batch):
    x, topic_emb, y, growth, trend, mean, std = zip(*batch)
    return (
        torch.stack(x),
        torch.stack(topic_emb),
        torch.stack(y),
        torch.stack(growth),
        torch.stack(trend).long(),
        torch.stack(mean),
        torch.stack(std),
    )

class ResidualBiLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(hidden_size * 2)
        self.proj = nn.Linear(input_size, hidden_size * 2) if input_size != hidden_size * 2 else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        out, _ = self.bilstm(x)
        return self.ln(out + residual)


class StackedResidualBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResidualBiLSTMBlock(input_size if i == 0 else hidden_size * 2, hidden_size) for i in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg["d_model"]
        max_len = cfg["positional_encoding"].get("max_len", 5000)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class LearnablePooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        q = self.query.expand(x.size(0), -1, -1)
        attn = torch.softmax(torch.matmul(q, x.transpose(1, 2)) / (x.size(-1) ** 0.5), dim=-1)
        return torch.matmul(attn, x).squeeze(1)


class TopicCrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

    def forward(self, x_seq, topic_emb):
        topic = topic_emb.unsqueeze(1)
        x_attn, _ = self.cross_attn(query=x_seq, key=topic, value=topic)
        return x_seq + x_attn


class ResearchTrendTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg["ts"]["input_dim"]
        topic_emb_dim = cfg["ts"]["topic_emb_dim"]
        d_model = cfg["d_model"]
        nhead = cfg["ts"]["nhead"]
        num_layers = cfg["ts"]["num_layers"]

        self.input_proj = nn.Linear(input_dim, d_model)
        self.topic_proj = nn.Linear(topic_emb_dim, d_model)
        self.topic_fusion = TopicCrossAttention(d_model, nhead)
        self.dropout = nn.Dropout(p=0.2)

        if cfg["model"] == "transformer":
            self.pos_encoder = PositionalEncoding(cfg)
            enc_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = TransformerEncoder(enc_layer, num_layers=num_layers)
        else:
            self.encoder = StackedResidualBiLSTM(input_size=d_model, hidden_size=d_model, num_layers=num_layers)
            self.bi_proj = nn.Linear(d_model * 2, d_model)

        self.pooling = LearnablePooling(d_model)

        # heads – count dim driven by cfg["prediction_dim"]
        self.head_count = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, cfg["prediction_dim"]))
        self.head_growth = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_trend = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x_seq, topic_emb):
        x = self.input_proj(x_seq)
        topic_emb_proj = self.topic_proj(topic_emb)
        x = self.topic_fusion(x, topic_emb_proj)

        if config["model"] == "transformer":
            x = self.pos_encoder(x)
            x = self.dropout(self.encoder(x))
        else:
            x = self.dropout(self.encoder(x))
            x = self.bi_proj(x)

        pooled = self.pooling(x)
        pred_count = self.head_count(pooled)
        # if 1‑dim, squeeze to [B]
        if pred_count.size(-1) == 1:
            pred_count = pred_count.squeeze(-1)
        pred_growth = self.head_growth(pooled).squeeze(-1)
        pred_trend = self.head_trend(pooled)
        return pred_count, pred_growth, pred_trend

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    horizon_offset: Optional[int] = None,
):
    model.eval()
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()
    totals = {"count": 0.0, "growth": 0.0, "trend": 0.0}
    steps = 0

    for x, topic_emb, y, growth, trend, mean, std in dataloader:
        x, topic_emb = x.to(device), topic_emb.to(device)
        y, growth, trend = y.to(device), growth.to(device), trend.to(device)
        mean, std = mean.to(device), std.to(device)

        p_count, p_growth, p_trend = model(x, topic_emb)

        # ---- count loss ----
        if p_count.dim() == 1:  # 1‑dim model
            y_sel = y[:, horizon_offset] if horizon_offset is not None else y.squeeze()
            loss_count = mse(p_count * std + mean, y_sel * std + mean)
        else:  # 3‑dim model
            loss_count = mse(p_count * std.unsqueeze(1) + mean.unsqueeze(1), y * std.unsqueeze(1) + mean.unsqueeze(1))

        totals["count"] += loss_count.item()
        totals["growth"] += mse(p_growth, growth).item()
        totals["trend"] += ce(p_trend, trend).item()
        steps += 1

    return {k + "_loss": v / steps for k, v in totals.items()}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    scheduler,
    cfg: dict,
    horizon_offset: Optional[int] = None,
):
    device = cfg["device"]
    model.to(device)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()
    
    patience = 25  # Number of epochs to wait before stopping
    min_delta = 0.001  # Minimum change in validation loss to be considered as improvement
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    checkpoint_dir = Path(f"checkpoints_{cfg['model_name']}")
    checkpoint_dir.mkdir(exist_ok=True)
    suffix = f"_T+{horizon_offset+1}" if horizon_offset is not None else ""
    best_model_path = checkpoint_dir / f"best_model{suffix}.pth"
    last_model_path = checkpoint_dir / f"last_model{suffix}.pth"

    for epoch in range(cfg["epoch"]):
        model.train()
        running_rmse = 0.0
        running_loss = 0.0
        steps = 0

        for x, topic_emb, y, growth, trend, mean, std in train_loader:
            x, topic_emb = x.to(device), topic_emb.to(device)
            y, growth, trend = y.to(device), growth.to(device), trend.to(device)
            mean, std = mean.to(device), std.to(device)

            p_count, p_growth, p_trend = model(x, topic_emb)

            # ---- count loss ----
            if p_count.dim() == 1:
                y_sel = y[:, horizon_offset] if horizon_offset is not None else y.squeeze()
                loss_count = mse(p_count, y_sel)
                rmse = torch.sqrt(mse((p_count * std) + mean, (y_sel * std) + mean))
            else:
                loss_count = mse(p_count, y)
                rmse = torch.sqrt(mse(p_count * std.unsqueeze(1) + mean.unsqueeze(1), y * std.unsqueeze(1) + mean.unsqueeze(1)))
                
            loss = loss_count
            if cfg.get("multi_loss", False):
                loss += 0.2 * mse(p_growth, growth)
            if cfg.get("trend_loss", False):
                loss += 0.2 * ce(p_trend, trend)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_rmse += rmse.item()
            running_loss += loss.item()
            steps += 1

        val_res = evaluate_model(model, val_loader, device, horizon_offset)
        current_val_loss = math.sqrt(val_res["count_loss"])
        
        scheduler.step(running_loss / steps)
        
        # Early stopping check
        if current_val_loss < best_val_loss - min_delta:
            best_val_loss = current_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': current_val_loss,
            }, best_model_path)
            wandb.save(str(best_model_path))
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            # Restore best model
            model.load_state_dict(best_model_state)
            break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': current_val_loss,
        }, last_model_path)
        wandb.save(str(last_model_path))

        wandb.log({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": running_loss / steps,
            "train_count_RMSE": running_rmse / steps,
            "val_count_RMSE": current_val_loss,
            "val_growth_loss": val_res["growth_loss"],
            "val_trend_loss": val_res["trend_loss"],
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
        })

train_set = TopicTimeSeriesDataset(config, sbert, "train_data")
val_set = TopicTimeSeriesDataset(config, sbert, "val_data")

train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=config["batch_size"], collate_fn=collate_fn)

def run_once(horizon_offset: Optional[int] = None):
    cfg_run = config if horizon_offset is None else copy.deepcopy(config)
    if horizon_offset is not None:
        cfg_run["prediction_dim"] = 1  # 1‑dim output for this horizon

    suffix = "" if horizon_offset is None else f"_T+{horizon_offset+1}"
    run_name = f"{cfg_run['model_name']}{suffix}"
    wandb.init(project="research-trend-prediction", name=run_name, config=cfg_run)

    model = ResearchTrendTransformer(cfg_run)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_run["learning_rate"], weight_decay=5e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    train_model(model, train_loader, val_loader, optimizer, scheduler, cfg_run, horizon_offset)
    wandb.finish()

if __name__ == "__main__":
    if config["training_mode"] == "single":
        run_once()
    elif config["training_mode"] == "separate":
        # horizons: 0 (T+1), 1 (T+2), 2 (T+3)
        for h in range(3):
            run_once(horizon_offset=h)
    elif config["training_mode"] == "single_output":
        run_once(horizon_offset=None)

    print("Training complete.")
