"""Model zoo – baseline, proposed and ablation architectures."""

import math
from typing import Any

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


# -----------------------------
# MobileNetV2 implementation – simplified width multiplier support
# -----------------------------


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, dropout=0.2):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.last_channel, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# -----------------------------
# Patch embedding adapter for images -> tokens
# -----------------------------


class LinearPatchEmbedding(nn.Module):
    def __init__(self, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size"
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()  # (B, N, C, p, p)
        patches = patches.view(B, patches.size(1), -1)  # (B, N, C*p*p)
        tokens = self.proj(patches)  # (B, N, D)
        return tokens


class DistilBERTClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained_name: str = "distilbert-base-uncased", patch_adapter: Any = None):
        super().__init__()
        self.patch_adapter = patch_adapter
        if patch_adapter is not None:
            config = DistilBertConfig.from_pretrained(pretrained_name)
            self.bert = DistilBertModel(config)
        else:
            self.bert = DistilBertModel.from_pretrained(pretrained_name)
        hidden_size = self.bert.config.dim
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, attention_mask=None):
        if self.patch_adapter is not None:
            # x expected: images tensor -> tokens via patch adapter
            x = self.patch_adapter(x)  # (B, N, D)
            # DistilBERT expects (B, N, D) as embeddings. We'll cheat by using inputs_embeds.
            output = self.bert(inputs_embeds=x, attention_mask=attention_mask)
            pooled = output.last_hidden_state.mean(dim=1)
        else:
            # x is token ids
            output = self.bert(x, attention_mask=attention_mask)
            pooled = output.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


# -----------------------------
# Char-level CNN encoder – for text
# -----------------------------


class CharCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(embedding_dim, 128, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, L)
        emb = self.embedding(x).transpose(1, 2)  # (B, E, L)
        feat = torch.relu(self.conv(emb))
        pooled = self.pool(feat).squeeze(-1)
        return self.fc(pooled)


# -----------------------------
# Model factory
# -----------------------------


def build_model(cfg, num_classes: int):
    name = cfg.model.name.lower()
    if "mobilenet" in name:
        model = MobileNetV2(
            num_classes=num_classes,
            width_mult=float(cfg.model.width_mult),
            dropout=cfg.model.dropout,
        )
    elif "distilbert" in name:
        if cfg.model.get("modality_adapter", None):
            adapter_cfg = cfg.model.modality_adapter
            patch_adapter = LinearPatchEmbedding(
                patch_size=adapter_cfg.patch_size,
                embed_dim=cfg.model.get("embed_dim", 768),
            )
        else:
            patch_adapter = None
        model = DistilBERTClassifier(num_classes=num_classes, patch_adapter=patch_adapter)
    elif cfg.model.get("input_adapter", {}).get("type") == "char_cnn":
        vocab_size = 100  # matches the tiny vocab in preprocessing
        model = CharCNN(vocab_size=vocab_size, embedding_dim=cfg.model.input_adapter.embedding_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model {name}")
    return model
