import argparse
import io
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

from flask import Flask, jsonify, render_template, request
from PIL import Image
import torch
from transformers import GPT2Tokenizer

from predict import (
    _normalize_text,
    build_clipcap_predict_model,
    build_cnn_rnn_predict_model,
    clip,
    generate2,
    generate_beam,
    generate_cnn_rnn_caption,
    transforms,
)

BASE_DIR = Path(__file__).resolve().parent

MODE_CONFIGS: Dict[str, Dict[str, object]] = {
    "cnn-rnn": {
        "label": "CNN-RNN",
        "model_arch": "cnn_rnn",
        "checkpoint": "checkpoints/cnn_rnn/flickr30k_cnn_rnn-014.pt",
        "temperature": 1.0,
        "cnn_max_len": 30,
    },
    "mlp": {
        "label": "ClipCap MLP (frozen)",
        "model_arch": "clipcap",
        "checkpoint": "checkpoints/flickr30k_mlp/flickr30k_mlp-009.pt",
        "mapping_type": "mlp",
        "only_prefix": True,
        "prefix_length": 10,
        "prefix_length_clip": 10,
        "num_layers": 8,
        "temperature": 1.0,
        "entry_length": 67,
        "decode": "beam",
    },
    "transformer": {
        "label": "ClipCap Transformer (frozen)",
        "model_arch": "clipcap",
        "checkpoint": "checkpoints/flickr30k_transformer_frozen/flickr30k_transformer_frozen-009.pt",
        "mapping_type": "transformer",
        "only_prefix": True,
        "prefix_length": 10,
        "prefix_length_clip": 10,
        "num_layers": 8,
        "temperature": 1.0,
        "entry_length": 67,
        "decode": "beam",
    },
    "finetune": {
        "label": "ClipCap Transformer (fine-tuned)",
        "model_arch": "clipcap",
        "checkpoint": "checkpoints/flickr30k_transformer_finetune/flickr30k_transformer_finetune-009.pt",
        "mapping_type": "transformer",
        "only_prefix": False,
        "prefix_length": 10,
        "prefix_length_clip": 10,
        "num_layers": 8,
        "temperature": 1.0,
        "entry_length": 67,
        "decode": "beam",
    },
}

MODE_ORDER = ["cnn-rnn", "mlp", "transformer", "finetune"]


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


class CaptionService:
    def __init__(self) -> None:
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(os.getenv("CAPTION_DEVICE", default_device))
        self.lock = threading.Lock()

        self._clip_runtime: Tuple[object, object, GPT2Tokenizer] | None = None
        self._clip_models: Dict[str, torch.nn.Module] = {}
        self._cnn_bundle: Tuple[torch.nn.Module, object] | None = None
        self._cnn_transform = None

    def _checkpoint_path(self, checkpoint_rel_path: str) -> str:
        path = BASE_DIR / checkpoint_rel_path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return str(path)

    def _load_clip_runtime(self) -> Tuple[object, object, GPT2Tokenizer]:
        with self.lock:
            if self._clip_runtime is not None:
                return self._clip_runtime
            if clip is None:
                raise RuntimeError("CLIP package is missing. Install with: pip install git+https://github.com/openai/CLIP.git")

            clip_model, preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self._clip_runtime = (clip_model, preprocess, tokenizer)
            return self._clip_runtime

    def _load_clipcap_model(self, mode: str) -> torch.nn.Module:
        with self.lock:
            if mode in self._clip_models:
                return self._clip_models[mode]

            config = MODE_CONFIGS[mode]
            args = argparse.Namespace(
                checkpoint=self._checkpoint_path(str(config["checkpoint"])),
                device=self.device,
                mapping_type=str(config.get("mapping_type", "mlp")),
                is_rn=False,
                only_prefix=_as_bool(config.get("only_prefix", True), default=True),
                prefix_length=int(config.get("prefix_length", 10)),
                prefix_length_clip=int(config.get("prefix_length_clip", 10)),
                num_layers=int(config.get("num_layers", 8)),
            )
            model = build_clipcap_predict_model(args).to(self.device).eval()
            self._clip_models[mode] = model
            return model

    def _load_cnn_bundle(self) -> Tuple[torch.nn.Module, object]:
        with self.lock:
            if self._cnn_bundle is not None:
                return self._cnn_bundle

            if transforms is None:
                raise RuntimeError("torchvision is required for cnn-rnn mode. Install with: pip install torchvision")

            args = argparse.Namespace(
                checkpoint=self._checkpoint_path(str(MODE_CONFIGS["cnn-rnn"]["checkpoint"])),
                device=self.device,
                embed_size=0,
                hidden_size=0,
                rnn_layers=0,
                dropout=0.0,
            )
            model, vocab = build_cnn_rnn_predict_model(args)
            model = model.to(self.device).eval()
            self._cnn_bundle = (model, vocab)
            return self._cnn_bundle

    def _load_cnn_transform(self):
        with self.lock:
            if self._cnn_transform is not None:
                return self._cnn_transform

            if transforms is None:
                raise RuntimeError("torchvision is required for cnn-rnn mode. Install with: pip install torchvision")

            self._cnn_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            return self._cnn_transform

    def _predict_cnn_rnn(self, image: Image.Image, config: Dict[str, object]) -> str:
        model, vocab = self._load_cnn_bundle()
        image_transform = self._load_cnn_transform()
        image_tensor = image_transform(image).unsqueeze(0)
        caption = generate_cnn_rnn_caption(
            model=model,
            vocab=vocab,
            image_tensor=image_tensor,
            device=self.device,
            max_len=int(config.get("cnn_max_len", 30)),
            temperature=float(config.get("temperature", 1.0)),
        )
        return _normalize_text(caption)

    def _predict_clipcap(self, image: Image.Image, mode: str, config: Dict[str, object]) -> str:
        clip_model, preprocess, tokenizer = self._load_clip_runtime()
        model = self._load_clipcap_model(mode)

        image_tensor = preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image_tensor).to(self.device, dtype=torch.float32)
            if _as_bool(config.get("normalize_prefix", False)):
                prefix = prefix / prefix.norm(2, dim=-1, keepdim=True)
            prefix_embed = model.clip_project(prefix).reshape(1, int(config.get("prefix_length", 10)), -1)

        decode = str(config.get("decode", "beam"))
        if decode == "beam":
            caption = generate_beam(
                model,
                tokenizer,
                beam_size=int(config.get("beam_size", 5)),
                embed=prefix_embed,
                entry_length=int(config.get("entry_length", 67)),
                temperature=float(config.get("temperature", 1.0)),
            )[0]
        else:
            caption = generate2(
                model,
                tokenizer,
                embed=prefix_embed,
                entry_length=int(config.get("entry_length", 67)),
                top_p=float(config.get("top_p", 0.8)),
                temperature=float(config.get("temperature", 1.0)),
            )
        return _normalize_text(caption)

    def _generate_caption_from_image(self, image: Image.Image, mode: str) -> Dict[str, object]:
        if mode not in MODE_CONFIGS:
            raise ValueError(f"Unsupported mode: {mode}")

        started = time.perf_counter()

        config = MODE_CONFIGS[mode]
        if config["model_arch"] == "cnn_rnn":
            caption = self._predict_cnn_rnn(image, config)
        else:
            caption = self._predict_clipcap(image, mode, config)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "mode": mode,
            "mode_label": config["label"],
            "caption": caption,
            "elapsed_ms": elapsed_ms,
        }

    def generate_caption(self, image_bytes: bytes, mode: str) -> Dict[str, object]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._generate_caption_from_image(image=image, mode=mode)

    def generate_all_captions(self, image_bytes: bytes, modes: List[str] | None = None) -> Dict[str, object]:
        selected_modes = modes or MODE_ORDER
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        started = time.perf_counter()
        results = [self._generate_caption_from_image(image=image, mode=mode) for mode in selected_modes]
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "mode": "all",
            "count": len(results),
            "elapsed_ms": elapsed_ms,
            "results": results,
        }


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
service = CaptionService()


@app.get("/")
def index():
    mode_labels = {key: str(cfg["label"]) for key, cfg in MODE_CONFIGS.items()}
    return render_template("index.html", mode_labels=mode_labels)


@app.get("/api/modes")
def list_modes():
    return jsonify({"ok": True, "modes": MODE_CONFIGS})


@app.post("/api/caption")
def caption_api():
    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"ok": False, "error": "Missing image file. Use multipart/form-data with field image."}), 400

    mode = (request.form.get("mode") or "all").strip().lower()
    if mode != "all" and mode not in MODE_CONFIGS:
        return jsonify({"ok": False, "error": "Invalid mode. Choose: all, cnn-rnn, mlp, transformer, finetune."}), 400

    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"ok": False, "error": "Uploaded image is empty."}), 400

    try:
        if mode == "all":
            result = service.generate_all_captions(image_bytes=image_bytes)
            return jsonify({"ok": True, **result})

        result = service.generate_caption(image_bytes=image_bytes, mode=mode)
        return jsonify({"ok": True, **result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False)
