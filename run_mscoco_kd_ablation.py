import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
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


ABLATIONS = [
    {
        "name": "student_ce",
        "label": "Student CE",
        "distill_logit_weight": 0.0,
        "distill_prefix_weight": 0.0,
    },
    {
        "name": "student_logit_kd",
        "label": "Student CE + Logit KD",
        "distill_logit_weight": 1.0,
        "distill_prefix_weight": 0.0,
    },
    {
        "name": "student_prefix_kd",
        "label": "Student CE + Prefix KD",
        "distill_logit_weight": 0.0,
        "distill_prefix_weight": 1.0,
    },
    {
        "name": "student_dual_kd",
        "label": "Student CE + Logit KD + Prefix KD",
        "distill_logit_weight": 1.0,
        "distill_prefix_weight": 1.0,
    },
]


def _run_command(command: List[str], title: str, dry_run: bool) -> None:
    print(f"\n[RUN] {title}")
    print(" ".join(command))
    if dry_run:
        return
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({title}) with exit code {completed.returncode}")


def _epoch_key(path: Path) -> int:
    match = re.search(r"-(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _find_latest_checkpoint(directory: Path, prefix: str) -> str:
    latest_path = directory / f"{prefix}_latest.pt"
    if latest_path.is_file():
        return str(latest_path)

    candidates = list(directory.glob(f"{prefix}-*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in {directory} for prefix '{prefix}'. "
            f"Expected {prefix}_latest.pt or {prefix}-XXX.pt"
        )
    candidates.sort(key=_epoch_key)
    return str(candidates[-1])


def _select_ablations(raw_modes: str) -> List[Dict[str, object]]:
    if raw_modes.strip().lower() in {"", "all"}:
        return ABLATIONS

    by_name = {str(ablation["name"]): ablation for ablation in ABLATIONS}
    selected: List[Dict[str, object]] = []
    unknown: List[str] = []
    for mode in raw_modes.split(","):
        name = mode.strip()
        if not name:
            continue
        if name not in by_name:
            unknown.append(name)
            continue
        selected.append(by_name[name])

    if unknown:
        valid = ", ".join(by_name)
        raise ValueError(f"Unknown mode(s): {', '.join(unknown)}. Valid modes: {valid}")
    if not selected:
        raise ValueError("--modes did not select any ablation")
    return selected


def _wait_for_file(path: str, timeout_seconds: int, poll_seconds: int) -> None:
    if os.path.isfile(path):
        return
    if timeout_seconds <= 0:
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        remaining = int(deadline - time.time())
        print(f"[WAIT] Teacher checkpoint not found yet: {path} ({remaining}s left)")
        time.sleep(max(1, poll_seconds))
        if os.path.isfile(path):
            return
    raise TimeoutError(f"Timed out waiting for teacher checkpoint: {path}")


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
        "mlp_hidden_scale",
        "init_checkpoint",
        "distill_logit_weight",
        "distill_prefix_weight",
        *METRIC_NAMES,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _student_train_command(args: argparse.Namespace, ablation: Dict[str, object], variant_dir: Path) -> List[str]:
    logit_weight = float(ablation["distill_logit_weight"]) * args.logit_kd_weight
    prefix_weight = float(ablation["distill_prefix_weight"]) * args.prefix_kd_weight
    command = [
        args.python_exec,
        args.train_script,
        "--model_arch",
        "clipcap",
        "--data",
        args.train_data,
        "--out_dir",
        str(variant_dir),
        "--prefix",
        str(ablation["name"]),
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
        args.student_mapping_type,
        "--num_layers",
        str(args.student_num_layers),
        "--decoder_model",
        args.student_decoder_model,
        "--mlp_hidden_scale",
        str(args.mlp_hidden_scale),
        "--clipcap_lr",
        str(args.lr),
        "--warmup_steps",
        str(args.warmup_steps),
        "--device",
        args.device,
        "--distill_logit_weight",
        str(logit_weight),
        "--distill_prefix_weight",
        str(prefix_weight),
        "--distill_temperature",
        str(args.distill_temperature),
        "--distill_prefix_loss",
        args.distill_prefix_loss,
    ]
    if args.mlp_hidden_dim > 0:
        command.extend(["--mlp_hidden_dim", str(args.mlp_hidden_dim)])
    if args.init_checkpoint:
        command.extend(["--init_checkpoint", args.init_checkpoint])
    if args.normalize_prefix:
        command.append("--normalize_prefix")
    if logit_weight > 0 or prefix_weight > 0:
        command.extend(
            [
                "--distill_teacher_checkpoint",
                args.teacher_checkpoint,
                "--distill_teacher_mapping_type",
                args.teacher_mapping_type,
                "--distill_teacher_decoder_model",
                args.teacher_decoder_model,
                "--distill_teacher_prefix_length",
                str(args.prefix_length),
                "--distill_teacher_prefix_length_clip",
                str(args.prefix_length_clip),
                "--distill_teacher_num_layers",
                str(args.teacher_num_layers),
            ]
        )
    return command


def _eval_command(
    args: argparse.Namespace,
    checkpoint: str,
    out_json: Path,
    mapping_type: str,
    decoder_model: str,
    num_layers: int,
    mlp_hidden_scale: float,
    mlp_hidden_dim: int,
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
        if mlp_hidden_dim > 0:
            command.extend(["--mlp_hidden_dim", str(mlp_hidden_dim)])
    if args.normalize_prefix:
        command.append("--normalize_prefix")
    if args.skip_spice:
        command.append("--skip_spice")
    if args.skip_bert_score:
        command.append("--skip_bert_score")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate MSCOCO dual-level KD ablations for ClipCap compression."
    )
    parser.add_argument("--python_exec", default=sys.executable)
    parser.add_argument("--train_script", default="./train.py")
    parser.add_argument("--evaluate_script", default="./evaluate.py")

    parser.add_argument("--train_data", default="./data/mscoco/mscoco_clip_ViT-B_32_train.pkl")
    parser.add_argument("--val_data", default="./data/mscoco/mscoco_clip_ViT-B_32_val.pkl")
    parser.add_argument(
        "--teacher_checkpoint",
        default="./checkpoints/mscoco_transformer_finetune/mscoco_transformer_finetune-009.pt",
    )
    parser.add_argument("--out_dir", default="./checkpoints/mscoco_kd_ablation")
    parser.add_argument(
        "--modes",
        default="all",
        help="Comma-separated modes to run, e.g. student_logit_kd,student_prefix_kd,student_dual_kd",
    )

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--bs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=5000)

    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--prefix_length_clip", type=int, default=10)
    parser.add_argument("--student_mapping_type", default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--student_decoder_model", default="distilgpt2")
    parser.add_argument("--student_num_layers", type=int, default=8)
    parser.add_argument("--mlp_hidden_scale", type=float, default=0.25)
    parser.add_argument("--mlp_hidden_dim", type=int, default=0)
    parser.add_argument("--init_checkpoint", default="", help="Optional student checkpoint to initialize each run")

    parser.add_argument("--teacher_mapping_type", default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--teacher_decoder_model", default="gpt2")
    parser.add_argument("--teacher_num_layers", type=int, default=8)

    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument("--distill_prefix_loss", default="mse", choices=["mse", "cosine", "mse_cosine"])
    parser.add_argument("--logit_kd_weight", type=float, default=1.0)
    parser.add_argument("--prefix_kd_weight", type=float, default=0.1)
    parser.add_argument("--normalize_prefix", action="store_true")

    parser.add_argument("--decode", default="beam", choices=["beam", "nucleus"])
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--eval_temperature", type=float, default=1.0)
    parser.add_argument("--entry_length", type=int, default=67)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--skip_spice", action="store_true")
    parser.add_argument("--skip_bert_score", action="store_true")
    parser.add_argument("--wait_for_teacher_seconds", type=int, default=0)
    parser.add_argument("--wait_poll_seconds", type=int, default=60)

    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--force_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    selected_ablations = _select_ablations(args.modes)

    if not os.path.isfile(args.train_script):
        raise FileNotFoundError(f"Train script not found: {args.train_script}")
    if not os.path.isfile(args.evaluate_script):
        raise FileNotFoundError(f"Evaluate script not found: {args.evaluate_script}")
    if not args.dry_run:
        _wait_for_file(args.teacher_checkpoint, args.wait_for_teacher_seconds, args.wait_poll_seconds)
    elif not os.path.isfile(args.teacher_checkpoint):
        print(f"[DRY RUN] Teacher checkpoint does not exist yet: {args.teacher_checkpoint}")
    if not args.dry_run and not args.skip_train and not os.path.isfile(args.train_data):
        raise FileNotFoundError(f"Train data not found: {args.train_data}")
    if not args.dry_run and not args.skip_eval and not os.path.isfile(args.val_data):
        raise FileNotFoundError(f"Val data not found: {args.val_data}")
    if not args.dry_run and args.init_checkpoint and not os.path.isfile(args.init_checkpoint):
        raise FileNotFoundError(f"init_checkpoint not found: {args.init_checkpoint}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints: Dict[str, str] = {}
    for ablation in selected_ablations:
        name = str(ablation["name"])
        variant_dir = out_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        existing_checkpoint = ""
        try:
            existing_checkpoint = _find_latest_checkpoint(variant_dir, name)
        except FileNotFoundError:
            pass

        if args.skip_train:
            if not existing_checkpoint:
                raise FileNotFoundError(f"--skip_train was set but no checkpoint exists for {name}")
            print(f"[SKIP] Train {name}: using {existing_checkpoint}")
        elif existing_checkpoint and not args.force_train:
            print(f"[SKIP] Train {name}: checkpoint already exists at {existing_checkpoint}")
        else:
            command = _student_train_command(args, ablation, variant_dir)
            _run_command(command, title=f"Train {name}", dry_run=args.dry_run)

        if args.dry_run:
            checkpoints[name] = str(variant_dir / f"{name}-{args.epochs - 1:03d}.pt")
        else:
            checkpoints[name] = _find_latest_checkpoint(variant_dir, name)

    rows: List[Dict[str, object]] = []
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    eval_jobs = [
        {
            "name": "teacher",
            "label": "Teacher Transformer + GPT-2 FT",
            "checkpoint": args.teacher_checkpoint,
            "decoder_model": args.teacher_decoder_model,
            "mapping_type": args.teacher_mapping_type,
            "num_layers": args.teacher_num_layers,
            "mlp_hidden_scale": "",
            "mlp_hidden_dim": 0,
            "distill_logit_weight": "",
            "distill_prefix_weight": "",
        }
    ]
    for ablation in selected_ablations:
        logit_weight = float(ablation["distill_logit_weight"]) * args.logit_kd_weight
        prefix_weight = float(ablation["distill_prefix_weight"]) * args.prefix_kd_weight
        eval_jobs.append(
            {
                "name": str(ablation["name"]),
                "label": str(ablation["label"]),
                "checkpoint": checkpoints[str(ablation["name"])],
                "decoder_model": args.student_decoder_model,
                "mapping_type": args.student_mapping_type,
                "num_layers": args.student_num_layers,
                "mlp_hidden_scale": args.mlp_hidden_scale,
                "mlp_hidden_dim": args.mlp_hidden_dim,
                "init_checkpoint": args.init_checkpoint,
                "distill_logit_weight": logit_weight,
                "distill_prefix_weight": prefix_weight,
            }
        )

    if not args.skip_eval:
        for job in eval_jobs:
            out_json = eval_dir / f"eval_{job['name']}.json"
            if out_json.is_file() and not args.force_eval:
                print(f"[SKIP] Eval {job['name']}: results already exist at {out_json}")
            else:
                command = _eval_command(
                    args=args,
                    checkpoint=str(job["checkpoint"]),
                    out_json=out_json,
                    mapping_type=str(job["mapping_type"]),
                    decoder_model=str(job["decoder_model"]),
                    num_layers=int(job["num_layers"]),
                    mlp_hidden_scale=float(job["mlp_hidden_scale"] or 0.5),
                    mlp_hidden_dim=int(job["mlp_hidden_dim"] or 0),
                )
                _run_command(command, title=f"Evaluate {job['name']}", dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run complete; no result files were created.")
        return

    for job in eval_jobs:
        out_json = eval_dir / f"eval_{job['name']}.json"
        if not out_json.is_file():
            if args.skip_eval:
                print(f"[WARN] Missing eval result for {job['name']}: {out_json}")
                continue
            raise FileNotFoundError(f"Missing eval result for {job['name']}: {out_json}")
        metrics = _load_metrics(out_json)
        row = {
            "name": job["name"],
            "label": job["label"],
            "checkpoint": job["checkpoint"],
            "decoder_model": job["decoder_model"],
            "mapping_type": job["mapping_type"],
            "mlp_hidden_scale": job["mlp_hidden_scale"],
            "init_checkpoint": job.get("init_checkpoint", ""),
            "distill_logit_weight": job["distill_logit_weight"],
            "distill_prefix_weight": job["distill_prefix_weight"],
        }
        row.update(metrics)
        rows.append(row)

    summary_json = out_dir / "ablation_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    summary_csv = out_dir / "ablation_summary.csv"
    _save_summary_csv(summary_csv, rows)

    print("\n=== KD Ablation Summary ===")
    for row in rows:
        print(
            f"{row['name']}: CIDEr={row.get('CIDEr', '')}, "
            f"Bleu_4={row.get('Bleu_4', '')}, METEOR={row.get('METEOR', '')}"
        )
    print(f"Saved: {summary_json}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
