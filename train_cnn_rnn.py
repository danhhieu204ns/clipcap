import argparse
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_period(caption: str) -> str:
    caption = caption.strip()
    if not caption:
        return caption
    return caption if caption.endswith(".") else f"{caption}."


def parse_token_file(captions_file: str) -> Dict[str, List[str]]:
    image_to_captions: Dict[str, List[str]] = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if "\t" in line:
                key, caption = line.split("\t", 1)
                image_name = key.split("|", 1)[0].split("#", 1)[0].strip()
            elif "|" in line:
                parts = [part.strip() for part in line.split("|")]
                if len(parts) < 3:
                    continue
                image_name = parts[0].split("#", 1)[0].strip()
                caption = parts[2]
            else:
                continue

            caption = add_period(caption)
            if image_name and caption:
                image_to_captions.setdefault(image_name, []).append(caption)
    return image_to_captions


def parse_karpathy_json(karpathy_json: str, split: str = "train") -> Dict[str, List[str]]:
    with open(karpathy_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_to_captions: Dict[str, List[str]] = {}
    for item in data.get("images", []):
        if split != "all" and item.get("split", "") != split:
            continue
        image_name = item.get("filename", "")
        if not image_name:
            continue
        for sentence in item.get("sentences", []):
            raw_caption = sentence.get("raw", "").strip()
            if not raw_caption:
                continue
            caption = add_period(raw_caption)
            image_to_captions.setdefault(image_name, []).append(caption)
    return image_to_captions


def collect_samples(captions_map: Dict[str, List[str]], images_dir: str) -> List[Tuple[str, str]]:
    samples: List[Tuple[str, str]] = []
    missing_images = 0
    for image_name, captions in captions_map.items():
        image_path = os.path.join(images_dir, image_name)
        if not os.path.isfile(image_path):
            missing_images += 1
            continue
        for caption in captions:
            samples.append((image_name, caption))

    if missing_images:
        print(f"Warning: skipped {missing_images} images because files were not found in {images_dir}")
    return samples


def tokenize_caption(text: str) -> List[str]:
    text = text.lower().strip()
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[.,!?;]", text)


@dataclass
class Vocabulary:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.stoi[UNK_TOKEN]

    @property
    def start_idx(self) -> int:
        return self.stoi[START_TOKEN]

    @property
    def end_idx(self) -> int:
        return self.stoi[END_TOKEN]

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(token, self.unk_idx) for token in tokens]


def build_vocab(captions: Sequence[str], min_freq: int = 5) -> Vocabulary:
    counter: Counter = Counter()
    for caption in captions:
        counter.update(tokenize_caption(caption))

    itos = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]
    for token, freq in sorted(counter.items()):
        if freq >= min_freq:
            itos.append(token)

    stoi = {token: idx for idx, token in enumerate(itos)}
    return Vocabulary(stoi=stoi, itos=itos)


class FlickrCaptionDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, str]],
        images_dir: str,
        vocab: Vocabulary,
        image_transform: transforms.Compose,
        max_tokens: int = 40,
    ):
        self.samples = samples
        self.images_dir = images_dir
        self.vocab = vocab
        self.image_transform = image_transform
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_name, caption = self.samples[idx]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        tokens = tokenize_caption(caption)
        token_ids = [self.vocab.start_idx] + self.vocab.encode(tokens[: self.max_tokens - 2]) + [self.vocab.end_idx]
        caption_tensor = torch.tensor(token_ids, dtype=torch.long)
        return image, caption_tensor


class CaptionCollator:
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images, captions = zip(*batch)
        images_tensor = torch.stack(images, dim=0)
        lengths = torch.tensor([caption.size(0) for caption in captions], dtype=torch.long)
        max_len = int(lengths.max().item())

        padded = torch.full((len(captions), max_len), self.pad_idx, dtype=torch.long)
        for i, caption in enumerate(captions):
            padded[i, : caption.size(0)] = caption

        return images_tensor, padded, lengths


class EncoderCNN(nn.Module):
    def __init__(self, embed_size: int, freeze_backbone: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        features = features.flatten(1)
        features = self.proj(features)
        features = self.bn(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_h = nn.Linear(embed_size, hidden_size * num_layers)
        self.init_c = nn.Linear(embed_size, hidden_size * num_layers)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, image_features: torch.Tensor, captions_input: torch.Tensor, lengths_input: torch.Tensor) -> torch.Tensor:
        embeddings = self.embed(captions_input)

        batch_size = image_features.size(0)
        h0 = self.init_h(image_features).view(batch_size, self.num_layers, self.hidden_size).transpose(0, 1).contiguous()
        c0 = self.init_c(image_features).view(batch_size, self.num_layers, self.hidden_size).transpose(0, 1).contiguous()

        packed = pack_padded_sequence(
            embeddings,
            lengths=lengths_input.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed, (h0, c0))
        logits = self.fc(packed_out.data)
        return logits


class CNNRNNCaptioner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        freeze_backbone: bool,
        dropout: float,
    ):
        super().__init__()
        self.encoder = EncoderCNN(embed_size=embed_size, freeze_backbone=freeze_backbone)
        self.decoder = DecoderRNN(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, images: torch.Tensor, captions_input: torch.Tensor, lengths_input: torch.Tensor) -> torch.Tensor:
        image_features = self.encoder(images)
        return self.decoder(image_features, captions_input, lengths_input)


def save_checkpoint(
    path: str,
    model: CNNRNNCaptioner,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    avg_loss: float,
    vocab: Vocabulary,
    args: argparse.Namespace,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab": {"stoi": vocab.stoi, "itos": vocab.itos},
            "args": vars(args),
        },
        path,
    )


def load_samples(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.karpathy_json:
        captions_map = parse_karpathy_json(args.karpathy_json, split=args.split)
    elif args.captions_file:
        captions_map = parse_token_file(args.captions_file)
    else:
        raise ValueError("Provide either --captions_file or --karpathy_json")

    samples = collect_samples(captions_map, args.images_dir)
    if not samples:
        raise RuntimeError("No valid image-caption samples were found. Check dataset paths.")
    return samples


def train_one_epoch(
    model: CNNRNNCaptioner,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
) -> float:
    model.train()
    running_loss = 0.0

    progress = tqdm(dataloader, desc=f"Epoch {epoch_idx + 1}/{total_epochs}")
    for images, captions, lengths in progress:
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)

        captions_input = captions[:, :-1]
        captions_target = captions[:, 1:]
        lengths_input = lengths - 1

        packed_targets = pack_padded_sequence(
            captions_target,
            lengths=lengths_input.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        logits = model(images, captions_input, lengths_input)
        loss = criterion(logits, packed_targets.data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / max(1, len(dataloader))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN-RNN image captioning model on Flickr30k.")
    parser.add_argument("--images_dir", required=True, help="Path to Flickr30k image directory")
    parser.add_argument("--captions_file", default="", help="Path to token/results file")
    parser.add_argument("--karpathy_json", default="", help="Path to Karpathy JSON (dataset_flickr30k.json)")
    parser.add_argument("--split", default="train", choices=("all", "train", "val", "test"))
    parser.add_argument("--out_dir", default="./checkpoints/cnn_rnn")
    parser.add_argument("--prefix", default="flickr30k_cnn_rnn")

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument("--embed_size", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=40)
    parser.add_argument("--min_word_freq", type=int, default=5)
    parser.add_argument("--unfreeze_cnn", action="store_true", help="Train ResNet backbone weights")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    samples = load_samples(args)
    print(f"Loaded {len(samples)} image-caption pairs")

    captions_for_vocab = [caption for _, caption in samples]
    vocab = build_vocab(captions_for_vocab, min_freq=args.min_word_freq)
    print(f"Vocab size: {len(vocab.itos)}")

    image_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = FlickrCaptionDataset(
        samples=samples,
        images_dir=args.images_dir,
        vocab=vocab,
        image_transform=image_transform,
        max_tokens=args.max_tokens,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=CaptionCollator(pad_idx=vocab.pad_idx),
    )

    model = CNNRNNCaptioner(
        vocab_size=len(vocab.itos),
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        freeze_backbone=not args.unfreeze_cnn,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    config_path = os.path.join(args.out_dir, f"{args.prefix}_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch_idx=epoch,
            total_epochs=args.epochs,
        )
        print(f"Epoch {epoch + 1}/{args.epochs} - avg_loss: {avg_loss:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.out_dir, f"{args.prefix}-{epoch:03d}.pt")
            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                avg_loss=avg_loss,
                vocab=vocab,
                args=args,
            )
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
