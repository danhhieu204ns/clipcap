import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _find_latest_checkpoint(directory: str, prefix: str) -> str:
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {directory}")

    latest_name = f"{prefix}_latest.pt"
    latest_path = dir_path / latest_name
    if latest_path.is_file():
        return str(latest_path)

    candidates = list(dir_path.glob(f"{prefix}-*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in {directory} for prefix '{prefix}'. "
            f"Expected {prefix}_latest.pt or {prefix}-XXX.pt"
        )

    def epoch_key(path: Path) -> int:
        match = re.search(r"-(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    candidates.sort(key=epoch_key)
    return str(candidates[-1])


def _resolve_checkpoint(explicit_path: str, directory: str, prefix: str) -> str:
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(f"Checkpoint not found: {explicit_path}")
        return explicit_path
    return _find_latest_checkpoint(directory=directory, prefix=prefix)


def _run_command(command: List[str], title: str) -> None:
    print(f"\n[RUN] {title}")
    print(" ".join(command))
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed ({title}) with exit code {completed.returncode}")


def _load_metrics(json_path: str) -> Dict[str, float]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("metrics", {})


def _save_summary_csv(path: str, summary: Dict[str, Dict[str, float]]) -> None:
    metric_names = [
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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", *metric_names])
        for mode, metrics in summary.items():
            row = [mode] + [metrics.get(name, "") for name in metric_names]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all 4 modes (cnn_rnn, mlp, transformer, finetune) on MSCOCO and aggregate metrics."
    )
    parser.add_argument("--python_exec", default=sys.executable)
    parser.add_argument("--evaluate_script", default="./evaluate.py")

    parser.add_argument("--data", default="./data/mscoco/mscoco_clip_ViT-B_32_val.pkl")
    parser.add_argument("--images_dir", default="./data/mscoco/val2017")
    parser.add_argument("--captions_file", default="./data/mscoco/results_val.csv")

    parser.add_argument("--out_dir", default="./checkpoints/mscoco_eval_all4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--decode", default="beam", choices=["beam", "nucleus"])
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--entry_length", type=int, default=67)
    parser.add_argument("--cnn_max_len", type=int, default=30)

    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--prefix_length_clip", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=8)

    parser.add_argument("--ckpt_cnn_rnn", default="")
    parser.add_argument("--ckpt_mlp", default="")
    parser.add_argument("--ckpt_transformer", default="")
    parser.add_argument("--ckpt_finetune", default="")

    parser.add_argument("--cnn_dir", default="./checkpoints/mscoco_cnn_rnn")
    parser.add_argument("--cnn_prefix", default="mscoco_cnn_rnn")
    parser.add_argument("--mlp_dir", default="./checkpoints/mscoco_mlp")
    parser.add_argument("--mlp_prefix", default="mscoco_mlp")
    parser.add_argument("--transformer_dir", default="./checkpoints/mscoco_transformer_frozen")
    parser.add_argument("--transformer_prefix", default="mscoco_transformer_frozen")
    parser.add_argument("--finetune_dir", default="./checkpoints/mscoco_transformer_finetune")
    parser.add_argument("--finetune_prefix", default="mscoco_transformer_finetune")
    args = parser.parse_args()

    if not os.path.isfile(args.evaluate_script):
        raise FileNotFoundError(f"evaluate script not found: {args.evaluate_script}")

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_cnn = _resolve_checkpoint(args.ckpt_cnn_rnn, args.cnn_dir, args.cnn_prefix)
    ckpt_mlp = _resolve_checkpoint(args.ckpt_mlp, args.mlp_dir, args.mlp_prefix)
    ckpt_tf = _resolve_checkpoint(args.ckpt_transformer, args.transformer_dir, args.transformer_prefix)
    ckpt_ft = _resolve_checkpoint(args.ckpt_finetune, args.finetune_dir, args.finetune_prefix)

    print("Using checkpoints:")
    print(f"- cnn_rnn: {ckpt_cnn}")
    print(f"- mlp: {ckpt_mlp}")
    print(f"- transformer: {ckpt_tf}")
    print(f"- finetune: {ckpt_ft}")

    jobs = [
        {
            "mode": "cnn_rnn",
            "args": [
                "--model_arch", "cnn_rnn",
                "--checkpoint", ckpt_cnn,
                "--images_dir", args.images_dir,
                "--captions_file", args.captions_file,
                "--device", args.device,
                "--max_samples", str(args.max_samples),
                "--temperature", str(args.temperature),
                "--cnn_max_len", str(args.cnn_max_len),
            ],
        },
        {
            "mode": "mlp",
            "args": [
                "--model_arch", "clipcap",
                "--data", args.data,
                "--checkpoint", ckpt_mlp,
                "--mapping_type", "mlp",
                "--only_prefix",
                "--prefix_length", str(args.prefix_length),
                "--prefix_length_clip", str(args.prefix_length_clip),
                "--num_layers", str(args.num_layers),
                "--decode", args.decode,
                "--beam_size", str(args.beam_size),
                "--top_p", str(args.top_p),
                "--temperature", str(args.temperature),
                "--entry_length", str(args.entry_length),
                "--device", args.device,
                "--max_samples", str(args.max_samples),
            ],
        },
        {
            "mode": "transformer",
            "args": [
                "--model_arch", "clipcap",
                "--data", args.data,
                "--checkpoint", ckpt_tf,
                "--mapping_type", "transformer",
                "--only_prefix",
                "--prefix_length", str(args.prefix_length),
                "--prefix_length_clip", str(args.prefix_length_clip),
                "--num_layers", str(args.num_layers),
                "--decode", args.decode,
                "--beam_size", str(args.beam_size),
                "--top_p", str(args.top_p),
                "--temperature", str(args.temperature),
                "--entry_length", str(args.entry_length),
                "--device", args.device,
                "--max_samples", str(args.max_samples),
            ],
        },
        {
            "mode": "finetune",
            "args": [
                "--model_arch", "clipcap",
                "--data", args.data,
                "--checkpoint", ckpt_ft,
                "--mapping_type", "transformer",
                "--prefix_length", str(args.prefix_length),
                "--prefix_length_clip", str(args.prefix_length_clip),
                "--num_layers", str(args.num_layers),
                "--decode", args.decode,
                "--beam_size", str(args.beam_size),
                "--top_p", str(args.top_p),
                "--temperature", str(args.temperature),
                "--entry_length", str(args.entry_length),
                "--device", args.device,
                "--max_samples", str(args.max_samples),
            ],
        },
    ]

    summary: Dict[str, Dict[str, float]] = {}

    for job in jobs:
        out_json = os.path.join(args.out_dir, f"eval_{job['mode']}.json")
        command = [
            args.python_exec,
            args.evaluate_script,
            *job["args"],
            "--save_predictions",
            out_json,
        ]
        _run_command(command, title=f"Evaluate {job['mode']}")
        summary[job["mode"]] = _load_metrics(out_json)

    summary_json = os.path.join(args.out_dir, "eval_summary_all4.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_csv = os.path.join(args.out_dir, "eval_summary_all4.csv")
    _save_summary_csv(summary_csv, summary)

    print("\n=== Summary (all 4 modes) ===")
    for mode, metrics in summary.items():
        cider = metrics.get("CIDEr")
        bleu4 = metrics.get("Bleu_4")
        print(f"{mode}: CIDEr={cider}, Bleu_4={bleu4}")

    print(f"Saved: {summary_json}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
