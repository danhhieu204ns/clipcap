import argparse
import importlib
import os
import pickle
import random
from typing import Dict, List, Tuple

import torch
from PIL import Image

if importlib.util.find_spec("clip") is not None:
    clip = importlib.import_module("clip")
else:
    clip = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


Row = Tuple[str, str]


def parse_token_file(captions_file: str) -> Dict[str, List[Row]]:
    """
    Parse Flickr30k caption files with formats:
    1) image_name#idx<TAB>caption
    2) image_name| idx | caption

    Returns a map: image_name -> list of (raw_key, caption)
    where raw_key is preserved so output keeps original format.
    """
    image_to_rows: Dict[str, List[Row]] = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.lower().startswith("image_name|"):
                continue

            if "\t" in line:
                key, caption = line.split("\t", 1)
                image_name = key.split("#", 1)[0].strip()
                raw_key = key.strip()
            elif "|" in line:
                parts = [part.strip() for part in line.split("|", 2)]
                if len(parts) < 3:
                    continue
                image_name = parts[0]
                idx = parts[1]
                caption = parts[2]
                if image_name.lower() in {"image_name", "image"}:
                    continue
                raw_key = f"{image_name}|{idx}"
            else:
                continue

            caption = caption.strip()
            if not image_name or not caption:
                continue
            image_to_rows.setdefault(image_name, []).append((raw_key, caption))

    return image_to_rows


def resolve_captions_file(captions_file: str) -> str:
    if captions_file and os.path.isfile(captions_file):
        return captions_file

    base_name = os.path.basename(captions_file) if captions_file else ""
    candidates = [
        os.path.join("flickr-image-dataset", "flickr30k_images", captions_file),
        os.path.join("flickr-image-dataset", "flickr30k_images", base_name),
        os.path.join("data", "flickr30k", base_name),
        os.path.join("flickr-image-dataset", "flickr30k_images", "results.csv"),
        os.path.join("flickr-image-dataset", "flickr30k_images", "results_20130124.token"),
    ]

    checked: List[str] = [captions_file]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            print(f"Using captions file: {candidate}")
            return candidate
        if candidate:
            checked.append(candidate)

    raise FileNotFoundError(
        "Could not find captions file. Checked: " + ", ".join(dict.fromkeys(checked))
    )


def _looks_like_images_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    with os.scandir(path) as it:
        for entry in it:
            if not entry.is_file():
                continue
            _, ext = os.path.splitext(entry.name.lower())
            if ext in image_exts:
                return True
    return False


def resolve_images_dir(images_dir: str) -> str:
    candidates: List[str] = []
    if images_dir:
        candidates.extend([
            images_dir,
            os.path.join(images_dir, "flickr30k_images"),
            os.path.join(images_dir, "flickr30k-images"),
        ])

    candidates.extend([
        os.path.join("flickr-image-dataset", "flickr30k_images", "flickr30k_images"),
        os.path.join("flickr-image-dataset", "flickr30k_images"),
        os.path.join("data", "flickr30k", "flickr30k-images"),
        os.path.join("data", "flickr30k", "flickr30k_images"),
    ])

    checked: List[str] = []
    for candidate in dict.fromkeys(candidates):
        if not candidate:
            continue
        checked.append(candidate)
        if not os.path.isdir(candidate):
            continue

        if _looks_like_images_dir(candidate):
            print(f"Using images dir: {candidate}")
            return candidate

        nested = os.path.join(candidate, "flickr30k_images")
        if os.path.isdir(nested) and _looks_like_images_dir(nested):
            print(f"Using images dir: {nested}")
            return nested

    raise FileNotFoundError(
        "Could not find images directory. Checked: " + ", ".join(dict.fromkeys(checked))
    )


def write_split(path: str, rows: List[Row]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for key, caption in rows:
            f.write(f"{key}\t{caption}\n")


def _sanitize_clip_model_type(clip_model_type: str) -> str:
    return clip_model_type.replace("/", "_").replace("\\", "_").replace(":", "_")


def _resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def _load_clip_runtime(clip_model_type: str, device: torch.device):
    if clip is None:
        raise ImportError(
            "CLIP package is missing. Install with: pip install git+https://github.com/openai/CLIP.git"
        )

    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    clip_model.eval()
    return clip_model, preprocess


@torch.no_grad()
def encode_image_ids(
    image_ids: List[str],
    images_dir: str,
    clip_model,
    preprocess,
    device: torch.device,
    batch_size: int,
    split_name: str,
) -> Tuple[torch.Tensor, List[str], List[str]]:
    embeddings: List[torch.Tensor] = []
    kept_ids: List[str] = []
    missing_ids: List[str] = []

    batch_tensors: List[torch.Tensor] = []
    batch_ids: List[str] = []

    def flush_batch() -> None:
        nonlocal batch_tensors, batch_ids
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors, dim=0).to(device)
        feats = clip_model.encode_image(batch).float().cpu()
        for image_id, feat in zip(batch_ids, feats):
            kept_ids.append(image_id)
            embeddings.append(feat)
        batch_tensors = []
        batch_ids = []

    iterator = tqdm(image_ids, desc=f"Encoding {split_name}", unit="img")
    for image_id in iterator:
        image_path = os.path.join(images_dir, image_id)
        if not os.path.isfile(image_path):
            missing_ids.append(image_id)
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image)
        except Exception:
            missing_ids.append(image_id)
            continue

        batch_tensors.append(image_tensor)
        batch_ids.append(image_id)
        if len(batch_tensors) >= batch_size:
            flush_batch()

    flush_batch()

    if not embeddings:
        return torch.empty((0, 0), dtype=torch.float32), kept_ids, missing_ids

    return torch.stack(embeddings, dim=0), kept_ids, missing_ids


def build_caption_records(
    image_ids: List[str],
    image_to_rows: Dict[str, List[Row]],
    image_id_to_embedding_idx: Dict[str, int],
) -> Tuple[List[Dict[str, object]], int]:
    records: List[Dict[str, object]] = []
    skipped_captions = 0

    for image_id in image_ids:
        rows = image_to_rows.get(image_id, [])
        emb_idx = image_id_to_embedding_idx.get(image_id)
        if emb_idx is None:
            skipped_captions += len(rows)
            continue

        for _, caption in rows:
            records.append({
                "image_id": image_id,
                "caption": caption,
                "clip_embedding": emb_idx,
            })

    return records, skipped_captions


def save_clipcap_pkl(path: str, clip_embedding: torch.Tensor, captions: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "clip_embedding": clip_embedding,
        "captions": captions,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved: {path}")
    print(f"  Embeddings: {clip_embedding.shape[0]}")
    print(f"  Captions: {len(captions)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split Flickr30k captions by image id, encode images with CLIP, and save train/test .pkl files."
    )
    parser.add_argument("--captions_file", required=True, help="Path to results.csv or token file")
    parser.add_argument("--images_dir", default="", help="Path to Flickr30k image directory")
    parser.add_argument("--out_dir", default="./data/flickr30k", help="Directory to save split caption files")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip_model_type", default="ViT-B/32")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_pkl", default="", help="Output train .pkl path (optional)")
    parser.add_argument("--test_pkl", default="", help="Output test .pkl path (optional)")
    args = parser.parse_args()

    if args.train_ratio < 0 or args.test_ratio < 0:
        raise ValueError("train_ratio and test_ratio must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    ratio_sum = args.train_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("train_ratio + test_ratio must equal 1.0")

    args.captions_file = resolve_captions_file(args.captions_file)
    args.images_dir = resolve_images_dir(args.images_dir)

    image_to_rows = parse_token_file(args.captions_file)
    image_ids = sorted(image_to_rows.keys())
    if not image_ids:
        raise RuntimeError("No valid rows found in captions file")

    rng = random.Random(args.seed)
    shuffled_ids = list(image_ids)
    rng.shuffle(shuffled_ids)

    n_train = int(len(shuffled_ids) * args.train_ratio)
    train_ids = shuffled_ids[:n_train]
    test_ids = shuffled_ids[n_train:]

    train_rows: List[Row] = []
    test_rows: List[Row] = []

    for image_id in train_ids:
        train_rows.extend(image_to_rows.get(image_id, []))
    for image_id in test_ids:
        test_rows.extend(image_to_rows.get(image_id, []))

    train_path = os.path.join(args.out_dir, "results_train.csv")
    test_path = os.path.join(args.out_dir, "results_test.csv")

    write_split(train_path, train_rows)
    write_split(test_path, test_rows)

    print("Split completed")
    print(f"Images: train={len(train_ids)}, test={len(test_ids)}, total={len(image_ids)}")
    print(f"Captions: train={len(train_rows)}, test={len(test_rows)}")
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")

    device = _resolve_device(args.device)
    print(f"Loading CLIP model: {args.clip_model_type} on {device}")
    clip_model, preprocess = _load_clip_runtime(args.clip_model_type, device)

    train_embed, train_kept_ids, train_missing_ids = encode_image_ids(
        image_ids=train_ids,
        images_dir=args.images_dir,
        clip_model=clip_model,
        preprocess=preprocess,
        device=device,
        batch_size=args.batch_size,
        split_name="train",
    )
    test_embed, test_kept_ids, test_missing_ids = encode_image_ids(
        image_ids=test_ids,
        images_dir=args.images_dir,
        clip_model=clip_model,
        preprocess=preprocess,
        device=device,
        batch_size=args.batch_size,
        split_name="test",
    )

    if train_embed.numel() == 0:
        raise RuntimeError("No train images could be encoded. Check images_dir and filenames in captions.")
    if test_embed.numel() == 0:
        raise RuntimeError("No test images could be encoded. Check images_dir and filenames in captions.")

    train_idx = {image_id: i for i, image_id in enumerate(train_kept_ids)}
    test_idx = {image_id: i for i, image_id in enumerate(test_kept_ids)}

    train_captions, train_skipped_captions = build_caption_records(train_ids, image_to_rows, train_idx)
    test_captions, test_skipped_captions = build_caption_records(test_ids, image_to_rows, test_idx)

    clip_tag = _sanitize_clip_model_type(args.clip_model_type)
    train_pkl = args.train_pkl or os.path.join(args.out_dir, f"flickr30k_clip_{clip_tag}_train.pkl")
    test_pkl = args.test_pkl or os.path.join(args.out_dir, f"flickr30k_clip_{clip_tag}_test.pkl")

    save_clipcap_pkl(train_pkl, train_embed, train_captions)
    save_clipcap_pkl(test_pkl, test_embed, test_captions)

    print(
        f"Encoded images: train={len(train_kept_ids)}, test={len(test_kept_ids)}, "
        f"missing_train={len(train_missing_ids)}, missing_test={len(test_missing_ids)}"
    )
    if train_skipped_captions or test_skipped_captions:
        print(
            f"Skipped captions due to missing images: "
            f"train={train_skipped_captions}, test={test_skipped_captions}"
        )


if __name__ == "__main__":
    main()
