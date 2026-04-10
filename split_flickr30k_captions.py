import argparse
import os
import random
from typing import Dict, List, Tuple


def parse_token_file(captions_file: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse Flickr30k caption files with formats:
    1) image_name#idx<TAB>caption
    2) image_name| idx | caption

    Returns a map: image_name -> list of (raw_key, caption)
    where raw_key is preserved so output keeps original format.
    """
    image_to_rows: Dict[str, List[Tuple[str, str]]] = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if "\t" in line:
                key, caption = line.split("\t", 1)
                image_name = key.split("#", 1)[0].strip()
                raw_key = key.strip()
            elif "|" in line:
                parts = [part.strip() for part in line.split("|")]
                if len(parts) < 3:
                    continue
                image_name = parts[0]
                idx = parts[1]
                caption = parts[2]
                raw_key = f"{image_name}|{idx}"
            else:
                continue

            if not image_name:
                continue
            image_to_rows.setdefault(image_name, []).append((raw_key, caption.strip()))

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


def write_split(path: str, rows: List[Tuple[str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for key, caption in rows:
            f.write(f"{key}\t{caption}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split Flickr30k captions file into train/val/test by image id.")
    parser.add_argument("--captions_file", required=True, help="Path to results.csv or token file")
    parser.add_argument("--out_dir", default="./data/flickr30k", help="Directory to save split caption files")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    args.captions_file = resolve_captions_file(args.captions_file)
    image_to_rows = parse_token_file(args.captions_file)
    image_ids = sorted(image_to_rows.keys())
    if not image_ids:
        raise RuntimeError("No valid rows found in captions file")

    rng = random.Random(args.seed)
    rng.shuffle(image_ids)

    n = len(image_ids)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    # keep remainder for test
    n_test = n - n_train - n_val

    train_ids = set(image_ids[:n_train])
    val_ids = set(image_ids[n_train:n_train + n_val])
    test_ids = set(image_ids[n_train + n_val:])

    train_rows: List[Tuple[str, str]] = []
    val_rows: List[Tuple[str, str]] = []
    test_rows: List[Tuple[str, str]] = []

    for image_id, rows in image_to_rows.items():
        if image_id in train_ids:
            train_rows.extend(rows)
        elif image_id in val_ids:
            val_rows.extend(rows)
        elif image_id in test_ids:
            test_rows.extend(rows)

    train_path = os.path.join(args.out_dir, "results_train.csv")
    val_path = os.path.join(args.out_dir, "results_val.csv")
    test_path = os.path.join(args.out_dir, "results_test.csv")

    write_split(train_path, train_rows)
    write_split(val_path, val_rows)
    write_split(test_path, test_rows)

    print(f"Images: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"Captions: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()
