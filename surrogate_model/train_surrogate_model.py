import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from surrogate_model import WatermarkDataset, SurrogateWatermarkModel, surrogate_loss_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    embedding_data = torch.tensor(
        np.loadtxt(args.embedding_data), device=device, dtype=torch.float32
    )
    watermark_logits = torch.tensor(
        np.loadtxt(args.watermark_logits), device=device, dtype=torch.float32
    )
    
    assert embedding_data.size(0) == watermark_logits.size(0), "Embedding and watermark data size mismatch"
    assert embedding_data.size(1) == args.input_dim, "Input dimension mismatch"
    assert watermark_logits.size(1) == 300, "Watermark logits dimension mismatch"
    dataset = WatermarkDataset(embedding_data, watermark_logits)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    surrogate_model = SurrogateWatermarkModel(input_dim=args.input_dim).to(device)
    optimizer = Adam(surrogate_model.parameters(), lr=1e-4, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(args.epochs):
        surrogate_model.train()
        for embeddings, watermarks in dataloader:
            optimizer.zero_grad()
            output = surrogate_model(embeddings)
            loss = surrogate_loss_fn(output, watermarks)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    torch.save(surrogate_model.state_dict(), args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train surrogate model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--embedding_data",
        type=str,
        default="data/embeddings.txt",
        help="Path to embedding data",
    )
    parser.add_argument(
        "--watermark_logits",
        type=str,
        default="data/watermark_logits.txt",
        help="Path to watermark logits",
    )
    parser.add_argument("--input_dim", type=int, default=1024, help="Input dimension")
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Directory to save model"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="Weight decay"
    )

    args = parser.parse_args()
    main(args)
