"""Surrogate training with CLIP vision + tiny BERT fusion.

Notes:
- Uses `openai/clip-vit-base-patch16` vision encoder and `prajjwal1/bert-tiny` text encoder by default.
- If local images are available, pass `--images_dir` to load images; otherwise image features are zeroed.
- Optional WandB and CodeCarbon instrumentation via flags.
"""
import argparse
import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from PIL import Image

from transformers import CLIPProcessor, CLIPVisionModel, AutoTokenizer, AutoModel

try:
    import wandb
except Exception:
    wandb = None

try:
    from codecarbon import OfflineEmissionsTracker
except Exception:
    OfflineEmissionsTracker = None


class MetadataDataset(Dataset):
    def __init__(self, parquet_path, images_dir=None, limit=None, transform=None):
        df = pd.read_parquet(parquet_path)
        if limit:
            df = df.iloc[:limit]
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        url = row.get('URL', None)
        text = row.get('TEXT', "") or ""
        img = None
        if self.images_dir:
            # Expect filenames like <id>.jpg saved by your download pipeline
            img_path = os.path.join(self.images_dir, f"{row.get('id')}.jpg")
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                except Exception:
                    img = None
        return img, text, idx


class EcoSurrogate(nn.Module):
    def __init__(self, device='cpu', clip_model_name='openai/clip-vit-base-patch16', txt_model_name='prajjwal1/bert-tiny'):
        super().__init__()
        self.device = device
        self.clip = CLIPVisionModel.from_pretrained(clip_model_name)
        self.clip_proj = nn.Linear(self.clip.config.hidden_size, 512)
        self.txt_encoder = AutoModel.from_pretrained(txt_model_name)
        self.txt_proj = nn.Linear(self.txt_encoder.config.hidden_size, 128)
        self.mlp = nn.Sequential(nn.Linear(512 + 128 + 1, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, img_feats, txt_emb, carbon):
        # img_feats: [B, C] already pooled
        e = self.clip_proj(img_feats)
        t = self.txt_proj(txt_emb)
        f = torch.cat([e, t, carbon.unsqueeze(1)], dim=1)
        out = self.mlp(f)
        u = torch.sigmoid(out[:, 0])
        log_e = out[:, 1]
        return u, torch.exp(log_e)


def collate_fn(batch, clip_processor, txt_tokenizer, device):
    imgs, texts, ids = zip(*batch)
    # Process images via CLIP processor; if None, create zero tensors
    images_present = any(img is not None for img in imgs)
    if images_present:
        imgs_for_proc = [img if img is not None else Image.new('RGB', (224, 224), (0, 0, 0)) for img in imgs]
        clip_inputs = clip_processor(images=imgs_for_proc, return_tensors='pt')
        with torch.no_grad():
            img_feats = clip_model(**{k: v.to(device) for k, v in clip_inputs.items()}).pooler_output
    else:
        img_feats = torch.zeros(len(imgs), clip_model.config.hidden_size)

    # Tokenize texts
    txt_inputs = txt_tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        txt_out = txt_encoder(**{k: v.to(device) for k, v in txt_inputs.items()}).pooler_output

    carbon = torch.zeros(len(imgs), dtype=torch.float32)
    return img_feats, txt_out, carbon


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = MetadataDataset(args.metadata, images_dir=args.images_dir, limit=args.limit)

    global clip_processor, clip_model, txt_tokenizer, txt_encoder
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_model = CLIPVisionModel.from_pretrained(args.clip_model).to(device)
    txt_tokenizer = AutoTokenizer.from_pretrained(args.txt_model)
    txt_encoder = AutoModel.from_pretrained(args.txt_model).to(device)

    def _collate(batch):
        return collate_fn(batch, clip_processor, txt_tokenizer, device)

    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=_collate)

    model = EcoSurrogate(device=device, clip_model_name=args.clip_model, txt_model_name=args.txt_model).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    # CodeCarbon tracker
    tracker = None
    if args.codecarbon and OfflineEmissionsTracker is not None:
        tracker = OfflineEmissionsTracker(project_name="EcoAgentOps_surrogate", output_dir=args.codecarbon_dir)
        tracker.start()

    # WandB
    if args.wandb and wandb is not None:
        wandb.init(project=args.wandb_project, config=vars(args))

    for epoch in range(args.epochs):
        model.train()
        for step, (img_feats, txt_emb, carbon) in enumerate(loader):
            img_feats = img_feats.to(device)
            txt_emb = txt_emb.to(device)
            carbon = carbon.to(device)
            u, e = model(img_feats, txt_emb, carbon)
            # Loss: BCE for utility (target dummy 0.5) + L1 on energy proxy
            loss_u = ((u - 0.5) ** 2).mean()
            loss_e = e.mean()
            loss = loss_u + 0.01 * loss_e
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % args.log_interval == 0:
                msg = f"Epoch {epoch} step {step} loss {loss.item():.4f}"
                print(msg)
                if args.wandb and wandb is not None:
                    wandb.log({"loss": loss.item(), "loss_u": loss_u.item(), "loss_e": loss_e.item()})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved surrogate to {args.out}")

    if tracker is not None:
        tracker.stop()

    if args.wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="data/laion_1M/metadata.parquet")
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--out", type=str, default="checkpoints/surrogate.pth")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--txt_model", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--wandb_project", type=str, default="ecoagentops-surrogate")
    parser.add_argument("--codecarbon", action='store_true')
    parser.add_argument("--codecarbon_dir", type=str, default="codecarbon_logs")
    args = parser.parse_args()
    train(args)
