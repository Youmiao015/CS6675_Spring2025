import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm

MODEL_PATH = "final_model.pt"

class Predictor:
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.device = device
        self.sbert = SentenceTransformer("allenai-specter")
    
    def data_transform(self, x, topic):
        """
        x: [15]
        topic: [1]
        """
        x = torch.tensor(x, dtype=torch.float32).squeeze(0)  # Convert numpy array to tensor
        x = x[-10:] # [10]

        x_delta = x[1:] - x[:-1] # [9]

        x_delta = torch.cat([torch.zeros(1), x_delta], dim=0) # [10]
        features = [
            x.unsqueeze(-1), # [10, 1]
            x_delta.unsqueeze(-1), # [10, 1]
            x.mean().repeat(10, 1), # [10, 1]
            x.std().repeat(10, 1), # [10, 1]
            (x[-1] - x[-2]).repeat(10, 1), # [10, 1]
            torch.linspace(-1, 1, steps=10).unsqueeze(-1), # [10, 1]
        ]
        x = torch.cat(features, dim=-1)

        mean, std = x.mean(), x.std() # [1]
        x = (x - mean) / (std + 1e-6)
        topic_emb = torch.tensor(self.sbert.encode(topic, normalize_embeddings=True), dtype=torch.float32)

        return x, topic_emb, mean, std

    def predict(
        self,
        x_seq,
        topic,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        x: [15]
        topic: [1]
        """
        with torch.no_grad():
            x_seq, topic_emb, mean, std = self.data_transform(x_seq, topic)
            
            x_seq = x_seq.to(device).unsqueeze(0)  # Add batch dimension [1, 10, 6]
            topic_emb = topic_emb.to(device)  # [1, 768]
            mean = mean.to(device) # [1]
            std = std.to(device) # [1]
            
            pred_count, _, _ = self.model(x_seq, topic_emb) # [1, 1]
            pred_count = pred_count * std.unsqueeze(-1) + mean.unsqueeze(-1)
            return np.round(pred_count.cpu().numpy())

def evaluate_with_csv(predictor, csv_path: str):
    df = pd.read_csv(csv_path)
    output_df = pd.DataFrame({
        "topic": pd.Series(dtype=str),
        "y": pd.Series(dtype=float),
        "pred_count": pd.Series(dtype=float)
    })
    rmse = []

    for i in tqdm(range(0, len(df))):
        batch_df = df.iloc[i:i+1]
        topic = batch_df["topic"].tolist()
        x_seq = np.array([np.fromstring(x.strip('[]'), sep=' ') for x in batch_df["x"].tolist()])
        y_gt = np.array([np.fromstring(y.strip('[]'), sep=' ')[0] for y in batch_df["y"].tolist()])

        pred_count = predictor.predict(x_seq, topic) # [batch_size]

        output_df = pd.concat([output_df, pd.DataFrame({"topic": topic, "y": y_gt, "pred_count": pred_count})])
        rmse.append((pred_count - y_gt) ** 2)
    
    output_df.to_csv(csv_path.replace(".csv", "_predictions.csv"), index=False)
    print(f"RMSE for {csv_path}: {np.sqrt(np.mean(rmse))}")
    
predictor = Predictor(MODEL_PATH)

def predict_transformer(inp_array: np.array, topic: str) -> float:
    """
    Predicts the count using the transformer model.
    
    Args:
        inp_array: Input array of shape [sequence_length]
        topic: Topic string
        
    Returns:
        Predicted count as a float
    """
    return predictor.predict(inp_array, [topic])[0]