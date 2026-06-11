import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


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


def _run_command(command: List[str], title: str, dry_run: bool) -> None:
    print(f"\n[RUN] {title}")
    print(" ".join(command))
    if dry_run:
        return
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({title}) with exit code {completed.returncode}")


def _find_teacher_checkpoint(out_dir: Path, prefix: str, epoch: int) -> str:
    final_path = out_dir / f"{prefix}-{epoch:03d}.pt"
    if final_path.is_file():
        return str(final_path)

    latest_path = out_dir / f"{prefix}_latest.pt"
    if latest_path.is_file():
        return str(latest_path)

    raise FileNotFoundError(
        f"No retrained teacher checkpoint found in {out_dir}. "
        f"Expected {final_path.name} or {latest_path.name}"
    )


def _load_metrics(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("metrics", {})


def _save_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "name",
        "label",
        "checkpoint",
        "decoder_model",
        "mapping_type",
        "num_layers",
        "mlp_hidden_scale",
        *METRIC_NAMES,
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _teacher_train_command(args: argparse.Namespace, out_dir: Path) -> List[str]:
    command = [
        args.python_exec,
        args.train_script,
        "--model_arch",
        "clipcap",
        "--data",
        args.train_data,
        "--out_dir",
        str(out_dir),
        "--prefix",
        args.prefix,
        "--epochs",
        str(args.epochs),
        "--save_every",
        str(args.save_every),
        "--prefix_length",
        str(args.prefix_length),
        "--prefix_length_clip",
        str(args.prefix_length_clip),
        "--bs",
        str(args.bs),
        "--mapping_type",
        "transformer",
        "--num_layers",
        str(args.teacher_num_layers),
        "--decoder_model",
        args.teacher_decoder_model,
        "--clipcap_lr",
        str(args.lr),
        "--warmup_steps",
        str(args.warmup_steps),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.normalize_prefix:
        command.append("--normalize_prefix")
    return command


def _eval_command(
    args: argparse.Namespace,
    checkpoint: str,
    out_json: Path,
    mapping_type: str,
    decoder_model: str,
    num_layers: int,
    mlp_hidden_scale: float = 0.5,
) -> List[str]:
    command = [
        args.python_exec,
        args.evaluate_script,
        "--model_arch",
        "clipcap",
        "--data",
        args.val_data,
        "--checkpoint",
        checkpoint,
        "--mapping_type",
        mapping_type,
        "--prefix_length",
        str(args.prefix_length),
        "--prefix_length_clip",
        str(args.prefix_length_clip),
        "--num_layers",
        str(num_layers),
        "--decoder_model",
        decoder_model,
        "--decode",
        args.decode,
        "--beam_size",
        str(args.beam_size),
        "--top_p",
        str(args.top_p),
        "--temperature",
        str(args.eval_temperature),
        "--entry_length",
        str(args.entry_length),
        "--device",
        args.device,
        "--max_samples",
        str(args.max_samples),
        "--save_predictions",
        str(out_json),
    ]
    if mapping_type == "mlp":
        command.extend(["--mlp_hidden_scale", str(mlp_hidden_scale)])
    if args.normalize_prefix:
        command.append("--normalize_prefix")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain the MSCOCO ClipCap teacher and compare it with the old teacher and student CE."
    )
    parser.add_argument("--python_exec", default=sys.executable)
    parser.add_argument("--train_script", default="./train.py")
    parser.add_argument("--evaluate_script", default="./evaluate.py")

    parser.add_argument("--train_data", default="./data/mscoco/mscoco_clip_ViT-B_32_train.pkl")
    parser.add_argument("--val_data", default="./data/mscoco/mscoco_clip_ViT-B_32_val.pkl")
    parser.add_argument("--out_dir", default="./checkpoints/mscoco_teacher_retrain")
    parser.add_argument("--prefix", default="mscoco_teacher_retrain")

    parser.add_argument(
        "--old_teacher_checkpoint",
        default="./checkpoints/mscoco_transformer_finetune/mscoco_transformer_finetune-009.pt",
    )
    parser.add_argument(
        "--student_checkpoint",
        default="./checkpoints/mscoco_kd_ablation/student_ce/student_ce-009.pt",
    )

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--bs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--prefix_length_clip", type=int, default=10)
    parser.add_argument("--teacher_decoder_model", default="gpt2")
    parser.add_argument("--teacher_num_layers", type=int, default=8)

    parser.add_argument("--student_decoder_model", default="distilgpt2")
    parser.add_argument("--student_mapping_type", default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--student_num_layers", type=int, default=8)
    parser.add_argument("--student_mlp_hidden_scale", type=float, default=0.25)

    parser.add_argument("--decode", default="beam", choices=["beam", "nucleus"])
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--eval_temperature", type=float, default=1.0)
    parser.add_argument("--entry_length", type=int, default=67)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--normalize_prefix", action="store_true")

    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--force_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for path in [args.train_script, args.evaluate_script]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Script not found: {path}")
    if not args.dry_run and not args.skip_train and not os.path.isfile(args.train_data):
        raise FileNotFoundError(f"Train data not found: {args.train_data}")
    if not args.dry_run and not args.skip_eval and not os.path.isfile(args.val_data):
        raise FileNotFoundError(f"Val data not found: {args.val_data}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    retrained_checkpoint = ""
    try:
        retrained_checkpoint = _find_teacher_checkpoint(out_dir, args.prefix, args.epochs - 1)
    except FileNotFoundError:
        pass

    if args.skip_train:
        if not retrained_checkpoint and not args.dry_run:
            raise FileNotFoundError("--skip_train was set but no retrained teacher checkpoint exists")
        print(f"[SKIP] Train retrained_teacher: using {retrained_checkpoint}")
    elif retrained_checkpoint and not args.force_train:
        print(f"[SKIP] Train retrained_teacher: checkpoint already exists at {retrained_checkpoint}")
    else:
        _run_command(_teacher_train_command(args, out_dir), "Train retrained teacher", args.dry_run)

    if args.dry_run:
        retrained_checkpoint = str(out_dir / f"{args.prefix}-{args.epochs - 1:03d}.pt")
    else:
        retrained_checkpoint = _find_teacher_checkpoint(out_dir, args.prefix, args.epochs - 1)

    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_jobs: List[Dict[str, object]] = [
        {
            "name": "old_teacher",
            "label": "Old teacher",
            "checkpoint": args.old_teacher_checkpoint,
            "decoder_model": args.teacher_decoder_model,
            "mapping_type": "transformer",
            "num_layers": args.teacher_num_layers,
            "mlp_hidden_scale": "",
        },
        {
            "name": "retrained_teacher",
            "label": "Retrained teacher",
            "checkpoint": retrained_checkpoint,
            "decoder_model": args.teacher_decoder_model,
            "mapping_type": "transformer",
            "num_layers": args.teacher_num_layers,
            "mlp_hidden_scale": "",
        },
    ]
    if args.student_checkpoint:
        eval_jobs.append(
            {
                "name": "student_ce_final",
                "label": "Student CE final epoch",
                "checkpoint": args.student_checkpoint,
                "decoder_model": args.student_decoder_model,
                "mapping_type": args.student_mapping_type,
                "num_layers": args.student_num_layers,
                "mlp_hidden_scale": args.student_mlp_hidden_scale,
            }
        )

    if not args.dry_run:
        for job in eval_jobs:
            checkpoint = str(job["checkpoint"])
            if checkpoint and not os.path.isfile(checkpoint):
                raise FileNotFoundError(f"Checkpoint not found for {job['name']}: {checkpoint}")

    if not args.skip_eval:
        for job in eval_jobs:
            out_json = eval_dir / f"eval_{job['name']}.json"
            if out_json.is_file() and not args.force_eval:
                print(f"[SKIP] Eval {job['name']}: results already exist at {out_json}")
                continue
            command = _eval_command(
                args=args,
                checkpoint=str(job["checkpoint"]),
                out_json=out_json,
                mapping_type=str(job["mapping_type"]),
                decoder_model=str(job["decoder_model"]),
                num_layers=int(job["num_layers"]),
                mlp_hidden_scale=float(job["mlp_hidden_scale"] or 0.5),
            )
            _run_command(command, f"Evaluate {job['name']}", args.dry_run)

    if args.dry_run:
        print("\nDry run complete; no result files were created.")
        return

    rows: List[Dict[str, object]] = []
    for job in eval_jobs:
        out_json = eval_dir / f"eval_{job['name']}.json"
        if not out_json.is_file():
            if args.skip_eval:
                print(f"[WARN] Missing eval result for {job['name']}: {out_json}")
                continue
            raise FileNotFoundError(f"Missing eval result for {job['name']}: {out_json}")
        row = dict(job)
        row.update(_load_metrics(out_json))
        rows.append(row)

    summary_json = out_dir / "teacher_retrain_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    summary_csv = out_dir / "teacher_retrain_summary.csv"
    _save_summary_csv(summary_csv, rows)

    print("\n=== Teacher Retrain Summary ===")
    for row in rows:
        print(
            f"{row['name']}: CIDEr={row.get('CIDEr', '')}, "
            f"Bleu_4={row.get('Bleu_4', '')}, METEOR={row.get('METEOR', '')}"
        )
    print(f"Saved: {summary_json}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
