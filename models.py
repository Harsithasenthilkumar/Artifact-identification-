import torch
import torch.nn as nn


class ImageModel(nn.Module):
    def __init__(self, in_dim=768, embed_dim=256, n_classes=6, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z), z


class MetadataModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, embed_dim=64, n_classes=6, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden_dim, embed_dim))
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z), z


class MultimodalModel(nn.Module):
    def __init__(self, img_dim=768, meta_dim=125, img_embed=256, meta_hidden=128, meta_embed=64,
                 fusion_dim=256, n_classes=6, dropout=0.3):
        super().__init__()
        self.image_encoder = nn.Sequential(nn.Linear(img_dim, img_embed), nn.ReLU(), nn.Dropout(dropout))
        self.meta_encoder = nn.Sequential(nn.Linear(meta_dim, meta_hidden), nn.ReLU(), nn.Dropout(dropout),
                                          nn.Linear(meta_hidden, meta_embed))
        self.fusion = nn.Sequential(nn.Linear(img_embed + meta_embed, fusion_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(fusion_dim, n_classes)

    def forward(self, img, meta):
        combined = torch.cat([self.image_encoder(img), self.meta_encoder(meta)], dim=1)
        z = self.fusion(combined)
        return self.classifier(z), z
