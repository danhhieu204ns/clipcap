from pathlib import Path
import argparse
import shutil
import urllib.request
import zipfile


COCO_URLS = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Skip download, file exists: {destination}")
        return

    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"Saved: {destination}")


def _extract(zip_path: Path, out_dir: Path) -> None:
    print(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def _validate_layout(root: Path) -> None:
    required = [
        root / "train2017",
        root / "val2017",
        root / "annotations" / "captions_train2017.json",
        root / "annotations" / "captions_val2017.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("MSCOCO download/extract incomplete. Missing: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract MSCOCO 2017 images + captions annotations.")
    parser.add_argument("--out_dir", default="./data/mscoco", help="Output directory for extracted MSCOCO files")
    parser.add_argument(
        "--keep_archives",
        action="store_true",
        help="Keep downloaded zip archives after extraction",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    archives_dir = out_dir / "archives"
    out_dir.mkdir(parents=True, exist_ok=True)

    for archive_name, url in COCO_URLS.items():
        archive_path = archives_dir / archive_name
        _download(url, archive_path)
        _extract(archive_path, out_dir)

    _validate_layout(out_dir)

    print("MSCOCO 2017 is ready")
    print(f"Data directory: {out_dir}")

    if not args.keep_archives and archives_dir.exists():
        shutil.rmtree(archives_dir)
        print(f"Removed archives directory: {archives_dir}")


if __name__ == "__main__":
    main()
