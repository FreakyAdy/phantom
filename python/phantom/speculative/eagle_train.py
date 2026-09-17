"""
PHANTOM v2 — EAGLE-3 Head Training Pipeline
============================================
Collects (features, next_k_tokens) from target model forward passes and
trains EAGLE fusion heads via cross-entropy loss per prediction head.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from phantom.speculative.eagle_heads import DEFAULT_FUSION_LAYERS, EagleHeads


class EagleTrainingExample:
    """Single training example: fused features + K ground-truth future tokens."""

    def __init__(
        self,
        features: Dict[str, torch.Tensor],
        target_tokens: List[int],
    ):
        self.features = features
        self.target_tokens = target_tokens


class EagleDataset(Dataset):
    def __init__(self, examples: List[EagleTrainingExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        ex = self.examples[idx]
        features = {k: v.squeeze(0) if v.dim() == 3 else v for k, v in ex.features.items()}
        targets = torch.tensor(ex.target_tokens, dtype=torch.long)
        return features, targets


def collate_eagle_batch(batch):
    features_list, targets_list = zip(*batch)
    merged_features: Dict[str, torch.Tensor] = {}
    for key in features_list[0].keys():
        merged_features[key] = torch.stack([f[key] for f in features_list], dim=0)
    targets = torch.stack(targets_list, dim=0)
    return merged_features, targets


def collect_training_data_synthetic(
    num_examples: int = 256,
    hidden_dim: int = 5120,
    k: int = 5,
    vocab_size: int = 32000,
) -> List[EagleTrainingExample]:
    """
    Generate synthetic training examples for unit testing / offline training
    when target model weights are unavailable (zero-disk policy).
    """
    examples: List[EagleTrainingExample] = []
    for i in range(num_examples):
        features = {
            f"h{layer}": torch.randn(1, hidden_dim)
            for layer in DEFAULT_FUSION_LAYERS
        }
        targets = [(i * 7 + j * 13) % vocab_size for j in range(k)]
        examples.append(EagleTrainingExample(features, targets))
    return examples


def train_eagle_heads(
    eagle: EagleHeads,
    train_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, float]:
    """Train EAGLE heads with per-head cross-entropy loss."""
    eagle.to(device)
    eagle.train()
    optimizer = torch.optim.AdamW(eagle.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, float] = {}
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        for features, targets in train_loader:
            features = {k: v.to(device) for k, v in features.items()}
            targets = targets.to(device)

            logits = eagle(features)  # [k, batch, vocab] or [k, vocab] for batch=1

            loss = torch.tensor(0.0, device=device)
            k = min(eagle.k, targets.shape[1])
            for head_idx in range(k):
                head_logits = logits[head_idx]
                if head_logits.dim() == 1:
                    head_logits = head_logits.unsqueeze(0)
                loss = loss + criterion(head_logits, targets[:, head_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        history[f"epoch_{epoch}_loss"] = avg_loss

    return history


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PHANTOM EAGLE-3 heads")
    parser.add_argument("--hidden-dim", type=int, default=5120)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=str, default="~/.phantom/eagle/default.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    output_path = Path(args.output.replace("~", str(Path.home())))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eagle = EagleHeads(
        hidden_dim=args.hidden_dim,
        vocab_size=args.vocab_size,
        k=args.k,
        device=args.device,
    )

    examples = collect_training_data_synthetic(
        num_examples=args.examples,
        hidden_dim=args.hidden_dim,
        k=args.k,
        vocab_size=args.vocab_size,
    )
    dataset = EagleDataset(examples)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_eagle_batch,
    )

    start = time.perf_counter()
    history = train_eagle_heads(eagle, loader, num_epochs=args.epochs, device=args.device)
    elapsed = time.perf_counter() - start

    torch.save({
        "state_dict": eagle.state_dict(),
        "hidden_dim": args.hidden_dim,
        "vocab_size": args.vocab_size,
        "k": args.k,
        "fusion_layers": DEFAULT_FUSION_LAYERS,
        "training_history": history,
        "training_seconds": elapsed,
    }, output_path)

    report = {
        "checkpoint": str(output_path),
        "param_count": eagle.param_count,
        "memory_mb_fp16": round(eagle.memory_mb_fp16, 2),
        "training_seconds": round(elapsed, 2),
        "final_loss": history.get(f"epoch_{args.epochs - 1}_loss"),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
