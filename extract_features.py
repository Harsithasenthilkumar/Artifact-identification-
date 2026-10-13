import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ViT_B_16_Weights, vit_b_16


class ArtifactImages(Dataset):
    def __init__(self, object_ids, img_dir, transform):
        self.object_ids = object_ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, i):
        path = os.path.join(self.img_dir, f"{self.object_ids[i]}.jpg")
        return self.transform(Image.open(path).convert("RGB"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="features")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model.heads = torch.nn.Identity()
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False

    df = pd.read_csv(os.path.join(args.data, "dataset_all.csv"), dtype={"object_id": str})
    ids = df["object_id"].tolist()
    loader = DataLoader(ArtifactImages(ids, os.path.join(args.data, "images"), weights.transforms()),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)

    feats = []
    start = time.time()
    with torch.no_grad():
        for b, x in enumerate(loader, 1):
            feats.append(model(x.to(device)).cpu().numpy())
            done = min(b * args.batch_size, len(ids))
            elapsed = time.time() - start
            if b % 5 == 0 or done == len(ids):
                eta = elapsed / done * (len(ids) - done)
                print(f"  {done}/{len(ids)} images  elapsed {elapsed / 60:.1f} min  remaining ~{eta / 60:.1f} min")

    feats = np.concatenate(feats).astype(np.float32)
    np.savez(os.path.join(args.out, "image_features.npz"), object_ids=np.array(ids), features=feats)
    print(f"Saved features with shape {feats.shape} to {args.out}/image_features.npz")
    print(f"Total time: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
