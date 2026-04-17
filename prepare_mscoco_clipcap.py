import argparse
import importlib
import json
import os
import pickle
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


class CocoSplitData:
    def __init__(self, image_to_rows: Dict[str, List[Row]], image_ids: List[str]):
        self.image_to_rows = image_to_rows
        self.image_ids = image_ids


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


def parse_coco_captions(annotations_file: str) -> CocoSplitData:
    with open(annotations_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    if not images or not annotations:
        raise RuntimeError(
            f"Invalid COCO annotations file: {annotations_file}. Missing 'images' or 'annotations'."
        )

    id_to_file_name: Dict[int, str] = {}
    for image in images:
        image_id = image.get("id")
        file_name = image.get("file_name")
        if image_id is None or not file_name:
            continue
        id_to_file_name[int(image_id)] = str(file_name)

    image_to_rows: Dict[str, List[Row]] = {}
    for ann in annotations:
        ann_id = ann.get("id")
        image_id = ann.get("image_id")
        caption = str(ann.get("caption", "")).strip()
        if image_id is None or not caption:
            continue
        file_name = id_to_file_name.get(int(image_id))
        if not file_name:
            continue
        raw_key = f"{file_name}#{ann_id if ann_id is not None else 0}"
        image_to_rows.setdefault(file_name, []).append((raw_key, caption))

    image_ids = sorted(image_to_rows.keys())
    if not image_ids:
        raise RuntimeError(f"No valid caption rows found in {annotations_file}")

    return CocoSplitData(image_to_rows=image_to_rows, image_ids=image_ids)


def write_split(path: str, rows: List[Row]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for key, caption in rows:
            f.write(f"{key}\t{caption}\n")


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
        description="Prepare MSCOCO captions with CLIP embeddings and save ClipCap-compatible train/val .pkl files."
    )
    parser.add_argument("--train_annotations", required=True, help="Path to captions_train*.json")
    parser.add_argument("--val_annotations", required=True, help="Path to captions_val*.json")
    parser.add_argument("--train_images_dir", required=True, help="Path to train image directory, e.g. train2017")
    parser.add_argument("--val_images_dir", required=True, help="Path to val image directory, e.g. val2017")
    parser.add_argument("--out_dir", default="./data/mscoco", help="Directory to save outputs")
    parser.add_argument("--clip_model_type", default="ViT-B/32")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_pkl", default="", help="Output train .pkl path (optional)")
    parser.add_argument("--val_pkl", default="", help="Output val .pkl path (optional)")
    parser.add_argument("--train_split_csv", default="", help="Output train split .csv path (optional)")
    parser.add_argument("--val_split_csv", default="", help="Output val split .csv path (optional)")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    if not os.path.isfile(args.train_annotations):
        raise FileNotFoundError(f"train_annotations not found: {args.train_annotations}")
    if not os.path.isfile(args.val_annotations):
        raise FileNotFoundError(f"val_annotations not found: {args.val_annotations}")
    if not os.path.isdir(args.train_images_dir):
        raise FileNotFoundError(f"train_images_dir not found: {args.train_images_dir}")
    if not os.path.isdir(args.val_images_dir):
        raise FileNotFoundError(f"val_images_dir not found: {args.val_images_dir}")

    train_data = parse_coco_captions(args.train_annotations)
    val_data = parse_coco_captions(args.val_annotations)

    train_rows: List[Row] = []
    for image_id in train_data.image_ids:
        train_rows.extend(train_data.image_to_rows.get(image_id, []))

    val_rows: List[Row] = []
    for image_id in val_data.image_ids:
        val_rows.extend(val_data.image_to_rows.get(image_id, []))

    os.makedirs(args.out_dir, exist_ok=True)
    train_csv_path = args.train_split_csv or os.path.join(args.out_dir, "results_train.csv")
    val_csv_path = args.val_split_csv or os.path.join(args.out_dir, "results_val.csv")

    write_split(train_csv_path, train_rows)
    write_split(val_csv_path, val_rows)

    print("MSCOCO split parse completed")
    print(f"Images: train={len(train_data.image_ids)}, val={len(val_data.image_ids)}")
    print(f"Captions: train={len(train_rows)}, val={len(val_rows)}")
    print(f"Saved: {train_csv_path}")
    print(f"Saved: {val_csv_path}")

    device = _resolve_device(args.device)
    print(f"Loading CLIP model: {args.clip_model_type} on {device}")
    clip_model, preprocess = _load_clip_runtime(args.clip_model_type, device)

    train_embed, train_kept_ids, train_missing_ids = encode_image_ids(
        image_ids=train_data.image_ids,
        images_dir=args.train_images_dir,
        clip_model=clip_model,
        preprocess=preprocess,
        device=device,
        batch_size=args.batch_size,
        split_name="train",
    )
    val_embed, val_kept_ids, val_missing_ids = encode_image_ids(
        image_ids=val_data.image_ids,
        images_dir=args.val_images_dir,
        clip_model=clip_model,
        preprocess=preprocess,
        device=device,
        batch_size=args.batch_size,
        split_name="val",
    )

    if train_embed.numel() == 0:
        raise RuntimeError("No train images could be encoded. Check train_images_dir and train annotations.")
    if val_embed.numel() == 0:
        raise RuntimeError("No val images could be encoded. Check val_images_dir and val annotations.")

    train_idx = {image_id: i for i, image_id in enumerate(train_kept_ids)}
    val_idx = {image_id: i for i, image_id in enumerate(val_kept_ids)}

    train_captions, train_skipped_captions = build_caption_records(
        train_data.image_ids,
        train_data.image_to_rows,
        train_idx,
    )
    val_captions, val_skipped_captions = build_caption_records(
        val_data.image_ids,
        val_data.image_to_rows,
        val_idx,
    )

    clip_tag = _sanitize_clip_model_type(args.clip_model_type)
    train_pkl = args.train_pkl or os.path.join(args.out_dir, f"mscoco_clip_{clip_tag}_train.pkl")
    val_pkl = args.val_pkl or os.path.join(args.out_dir, f"mscoco_clip_{clip_tag}_val.pkl")

    save_clipcap_pkl(train_pkl, train_embed, train_captions)
    save_clipcap_pkl(val_pkl, val_embed, val_captions)

    print(
        f"Encoded images: train={len(train_kept_ids)}, val={len(val_kept_ids)}, "
        f"missing_train={len(train_missing_ids)}, missing_val={len(val_missing_ids)}"
    )
    if train_skipped_captions or val_skipped_captions:
        print(
            f"Skipped captions due to missing images: "
            f"train={train_skipped_captions}, val={val_skipped_captions}"
        )


if __name__ == "__main__":
    main()
