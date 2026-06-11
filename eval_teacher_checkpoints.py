import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


METRIC_NAMES = [
    "Bleu_1",
    "Bleu_2",
    "Bleu_3",
    "Bleu_4",
    "METEOR",
    "ROUGE_L",
    "CIDEr",
    "SPICE",
    "BERTScore_P",
    "BERTScore_R",
    "BERTScore_F1",
]


def _epoch_from_name(path: Path) -> Optional[int]:
    match = re.search(r"-(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None


def _find_checkpoints(checkpoint_dir: Path, prefix: str) -> List[Tuple[int, Path]]:
    checkpoints: List[Tuple[int, Path]] = []
    for path in checkpoint_dir.glob(f"{prefix}-*.pt"):
        epoch = _epoch_from_name(path)
        if epoch is not None:
            checkpoints.append((epoch, path))
    checkpoints.sort(key=lambda item: item[0])
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found: {checkpoint_dir}/{prefix}-*.pt")
    return checkpoints


def _run(command: List[str], title: str, dry_run: bool) -> None:
    print(f"\n[RUN] {title}")
    print(" ".join(command))
    if dry_run:
        return
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{title} failed with exit code {completed.returncode}")


def _load_metrics(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("metrics", {})


def _save_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = ["epoch", "checkpoint", "eval_json", *METRIC_NAMES]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _eval_command(args: argparse.Namespace, checkpoint: Path, out_json: Path) -> List[str]:
    command = [
        args.python_exec,
        args.evaluate_script,
        "--model_arch",
        "clipcap",
        "--data",
        args.val_data,
        "--checkpoint",
        str(checkpoint),
        "--mapping_type",
        args.mapping_type,
        "--prefix_length",
        str(args.prefix_length),
        "--prefix_length_clip",
        str(args.prefix_length_clip),
        "--num_layers",
        str(args.num_layers),
        "--decoder_model",
        args.decoder_model,
        "--decode",
        args.decode,
        "--beam_size",
        str(args.beam_size),
        "--top_p",
        str(args.top_p),
        "--temperature",
        str(args.temperature),
        "--entry_length",
        str(args.entry_length),
        "--device",
        args.device,
        "--max_samples",
        str(args.max_samples),
        "--save_predictions",
        str(out_json),
    ]
    if not args.full_metrics:
        command.extend(["--skip_spice", "--skip_bert_score"])
    if args.normalize_prefix:
        command.append("--normalize_prefix")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate every saved MSCOCO teacher checkpoint and summarize metrics."
    )
    parser.add_argument("--python_exec", default="./.venv/bin/python")
    parser.add_argument("--evaluate_script", default="./evaluate.py")
    parser.add_argument("--checkpoint_dir", default="./checkpoints/mscoco_teacher_retrain")
    parser.add_argument("--checkpoint_prefix", default="mscoco_teacher_retrain")
    parser.add_argument("--val_data", default="./data/mscoco/mscoco_clip_ViT-B_32_val.pkl")
    parser.add_argument("--out_dir", default="./checkpoints/mscoco_teacher_retrain/eval_by_epoch")

    parser.add_argument("--mapping_type", default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--decoder_model", default="gpt2")
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--prefix_length_clip", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--normalize_prefix", action="store_true")

    parser.add_argument("--decode", default="beam", choices=["beam", "nucleus"])
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--entry_length", type=int, default=67)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument(
        "--full_metrics",
        action="store_true",
        help="Also compute SPICE and BERTScore. Slow; default is BLEU/METEOR/ROUGE/CIDEr only.",
    )
    parser.add_argument("--force_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.evaluate_script):
        raise FileNotFoundError(f"Evaluate script not found: {args.evaluate_script}")
    if not args.dry_run and not os.path.isfile(args.val_data):
        raise FileNotFoundError(f"Val data not found: {args.val_data}")

    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    checkpoints = _find_checkpoints(checkpoint_dir, args.checkpoint_prefix)
    for epoch, checkpoint in checkpoints:
        out_json = out_dir / f"eval_{args.checkpoint_prefix}-{epoch:03d}.json"
        if out_json.is_file() and not args.force_eval:
            print(f"[SKIP] epoch {epoch:03d}: {out_json}")
        else:
            command = _eval_command(args, checkpoint, out_json)
            _run(command, title=f"Evaluate epoch {epoch:03d}", dry_run=args.dry_run)

        row: Dict[str, object] = {
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "eval_json": str(out_json),
        }
        if out_json.is_file():
            row.update(_load_metrics(out_json))
        rows.append(row)

    if args.dry_run:
        print("\nDry run complete; no summary files were written.")
        return

    summary_json = out_dir / "teacher_checkpoint_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    summary_csv = out_dir / "teacher_checkpoint_summary.csv"
    _save_summary_csv(summary_csv, rows)

    print("\n=== Teacher Checkpoint Summary ===")
    for row in rows:
        print(
            f"epoch {int(row['epoch']):03d}: "
            f"CIDEr={row.get('CIDEr', '')}, "
            f"Bleu_4={row.get('Bleu_4', '')}, "
            f"METEOR={row.get('METEOR', '')}"
        )
    print(f"Saved: {summary_json}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
