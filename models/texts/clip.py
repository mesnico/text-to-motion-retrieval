from turtle import forward
import torch
import clip
from torch import nn

class CLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/32", device=self.clip_device)
        self.proj = nn.Linear(512, 512)

    def get_output_dim(self):
        return 512

    def forward(self, text):
        text_tokenized = clip.tokenize(text, truncate=True).to(self.clip_device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokenized).float()
        text_features = self.proj(text_features)
        return text_features