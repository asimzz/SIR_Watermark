import argparse
import numpy as np
import torch
from train_watermark_model import TransformModel
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class WatermarkDataset(Dataset):
#     def __init__(self, embedding_path, watermark_path):
#         self.embeddings = torch.tensor(np.load(embedding_path), dtype=torch.float32)
#         self.watermarks = torch.tensor(np.load(watermark_path), dtype=torch.float32)

def main(args):
    embedding_data = np.loadtxt(args.embedding_data)

    
    model = TransformModel(input_dim=args.input_dim).to(device)
    model.load_state_dict(torch.load(args.original_model))
    model.eval()
    
    with torch.no_grad():
        watermark_logits = model(torch.tensor(embedding_data, device=device, dtype=torch.float32))
        print(watermark_logits)
        np.save(args.output_dir, watermark_logits)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate watermark logits')
    parser.add_argument('--original_model', type=str, default='model.pth', help='Path to original watermark model')
    parser.add_argument('--embedding_data', type=str, default='data/embeddings.txt', help='Path to embedding data')
    parser.add_argument('--input_dim', type=int, default=1, help='Input dimension')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save logits')
    args = parser.parse_args()
    
    main(args)