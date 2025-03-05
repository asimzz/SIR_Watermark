import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WatermarkDataset(Dataset):
    def __init__(self, embedding_vectors, watermark_vectors):
        self.embeddings = embedding_vectors
        self.watermarks = watermark_vectors

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.watermarks[idx]

class SimpleBlock(nn.Module):
    def __init__(self, dim):
        super(SimpleBlock, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        return out

class SurrogateWatermarkModel(nn.Module):
    def __init__(self, num_layers=4, input_dim=1024, hidden_dim=500, output_dim=300):
        super(SurrogateWatermarkModel, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.layers.append(SimpleBlock(hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def surrogate_loss_fn(output, target):
    # 1. Similarity loss (L2 distance between predicted & target watermark logits)
    similarity_loss = nn.MSELoss()(output, target)
    
    # 2. Balance constraint: Encourage zero-centered mean
    balance_loss = torch.mean(output, dim=1).pow(2).sum()

    # 3. Value range penalty (push logits within [-1, 1] range)
    range_penalty = torch.mean(torch.relu(torch.abs(output) - 1))

    # Weighted sum of losses
    total_loss = similarity_loss + 0.1 * balance_loss + 0.05 * range_penalty
    return total_loss


def evaluate_model(model, dataloader):
    model.eval()
    original_logits = []
    surrogate_logits = []
    
    with torch.no_grad():
        for embeddings, true_watermarks in dataloader:
            predicted_watermarks = model(embeddings)
            original_logits.append(true_watermarks.cpu().numpy())
            surrogate_logits.append(predicted_watermarks.cpu().numpy())

    # Convert to numpy arrays
    original_logits = np.concatenate(original_logits, axis=0)
    surrogate_logits = np.concatenate(surrogate_logits, axis=0)

    # Compute AUC score
    auc_score = roc_auc_score(original_logits.ravel(), surrogate_logits.ravel())
    print(f"AUC Score for Surrogate Model: {auc_score:.4f}")