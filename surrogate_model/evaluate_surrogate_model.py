import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from surrogate_model import SurrogateWatermarkModel, WatermarkDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_surrogate(args):
    # Load embeddings and watermark logits
    embedding_data = torch.tensor(np.loadtxt(args.embedding_data), device=device, dtype=torch.float32)
    watermark_logits = torch.tensor(np.loadtxt(args.watermark_logits), device=device, dtype=torch.float32)

    dataset = WatermarkDataset(embedding_data, watermark_logits)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load trained surrogate model
    model = SurrogateWatermarkModel().to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    original_probs, surrogate_probs, true_labels = [], [], []

    sigmoid = torch.nn.Sigmoid()  # Apply sigmoid to convert logits to probability scores

    with torch.no_grad():
        for embeddings, true_watermarks in dataloader:
            embeddings, true_watermarks = embeddings.to(device), true_watermarks.cpu().numpy()

            predicted_watermarks = model(embeddings)

            # Convert logits to probabilities
            original_probs.append(sigmoid(torch.tensor(true_watermarks)).cpu().numpy())  # Convert labels to probabilities
            surrogate_probs.append(sigmoid(predicted_watermarks).cpu().numpy())
            true_labels.append((true_watermarks > 0).astype(int))  # Ensure binary labels

    # Convert lists to numpy arrays
    original_probs = np.concatenate(original_probs, axis=0)
    surrogate_probs = np.concatenate(surrogate_probs, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # Compute AUC score
    auc_original = roc_auc_score(true_labels.ravel(), original_probs.ravel())  # Convert logits to AUC-compatible values
    auc_surrogate = roc_auc_score(true_labels.ravel(), surrogate_probs.ravel())

    print(f"AUC Score for Original Watermark Model: {auc_original:.4f}")
    print(f"AUC Score for Surrogate Model: {auc_surrogate:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate surrogate model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/surrogate_model.pth",
        help="Path to surrogate model",
    )
    parser.add_argument(
        "--embedding_data",
        type=str,
        help="Path to embedding data",
    )
    parser.add_argument(
        "--watermark_logits",
        type=str,
        help="Path to watermark logits",
    )
    args = parser.parse_args()
    evaluate_surrogate(args)
