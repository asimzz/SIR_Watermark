import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from surrogate_model import SurrogateWatermarkModel, WatermarkDataset, get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_surrogate(args):
    embedding_data = torch.tensor(np.loadtxt(args.embedding_path), device=device, dtype=torch.float32)
    watermark_logits = torch.tensor(np.loadtxt(args.watermark_path), device=device, dtype=torch.float32)

    dataset = WatermarkDataset(embedding_data, watermark_logits)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = SurrogateWatermarkModel().to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    original_logits, surrogate_logits = [], []
    
    with torch.no_grad():
        for embeddings, true_watermarks in dataloader:
            embeddings, true_watermarks = embeddings.to(device), true_watermarks.cpu().numpy()
            predicted_watermarks = model(embeddings).cpu().numpy()
            original_logits.append(true_watermarks)
            surrogate_logits.append(predicted_watermarks)

    # Convert to numpy arrays
    original_logits = np.concatenate(original_logits, axis=0)
    surrogate_logits = np.concatenate(surrogate_logits, axis=0)

    # Compute AUC score
    auc_score = roc_auc_score(original_logits.ravel(), surrogate_logits.ravel())
    print(f"AUC Score for Surrogate Model: {auc_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate surrogate model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/surrogate_model.pth",
        help="Path to surrogate model",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        help="Path to embedding data",
    )
    parser.add_argument(
        "--watermark_path",
        type=str,
        help="Path to watermark logits",
    )
    args = parser.parse_args()
    evaluate_surrogate(args)
