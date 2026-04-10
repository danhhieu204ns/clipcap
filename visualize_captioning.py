import argparse
from contextlib import nullcontext
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from PIL import Image
from transformers import GPT2Tokenizer

from train import ClipCaptionModel, ClipCaptionPrefix, MappingType

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("matplotlib is required. Install with: pip install matplotlib") from exc

try:
    import clip
except ImportError:
    clip = None

CNN_RNN_IMPORT_ERROR: Optional[Exception] = None
try:
    from torchvision import transforms
    from train_cnn_rnn import CNNRNNCaptioner, END_TOKEN, PAD_TOKEN, START_TOKEN, UNK_TOKEN, Vocabulary
except ImportError as exc:
    transforms = None
    CNNRNNCaptioner = None
    Vocabulary = None
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    CNN_RNN_IMPORT_ERROR = exc


NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _safe_torch_load(path: str, map_location: torch.device, weights_only: bool):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _normalize_cam(cam: np.ndarray, invert: bool = False) -> np.ndarray:
    norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    if invert:
        norm = 1.0 - norm
    return norm


def _build_clipcap_model(args: argparse.Namespace, device: torch.device) -> ClipCaptionModel:
    mapping_type = {"mlp": MappingType.MLP, "transformer": MappingType.Transformer}[args.mapping_type]
    prefix_dim = 640 if args.is_rn else 512

    if args.only_prefix:
        model = ClipCaptionPrefix(
            args.prefix_length,
            clip_length=args.prefix_length_clip,
            prefix_size=prefix_dim,
            num_layers=args.num_layers,
            mapping_type=mapping_type,
        )
    else:
        model = ClipCaptionModel(
            args.prefix_length,
            clip_length=args.prefix_length_clip,
            prefix_size=prefix_dim,
            num_layers=args.num_layers,
            mapping_type=mapping_type,
        )

    state_dict = _safe_torch_load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model


def _decode_clipcap_caption(
    model: ClipCaptionModel,
    tokenizer: GPT2Tokenizer,
    prefix_embed: torch.Tensor,
    max_len: int,
    temperature: float,
    stop_token: str,
) -> Tuple[str, List[int]]:
    model.eval()
    stop_token_id = tokenizer.encode(stop_token)[0]
    generated = prefix_embed
    token_ids: List[int] = []

    with torch.no_grad():
        for _ in range(max_len):
            outputs = model.gpt(inputs_embeds=generated, use_cache=False, return_dict=True)
            logits = outputs.logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
            next_token = torch.argmax(logits, dim=-1)
            next_id = int(next_token.item())
            token_ids.append(next_id)

            next_embed = model.gpt.transformer.wte(next_token).unsqueeze(1)
            generated = torch.cat((generated, next_embed), dim=1)
            if next_id == stop_token_id:
                break

    caption = _normalize_text(tokenizer.decode(token_ids, skip_special_tokens=True))
    return caption, token_ids


def _overlay_on_raw_image(raw_image: Image.Image, cam_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = cam_map.shape
    image_np = np.asarray(raw_image.resize((w, h), Image.Resampling.BICUBIC)).astype(np.float32) / 255.0
    heatmap = plt.get_cmap("jet")(cam_map)[..., :3]
    overlay = np.clip(0.45 * image_np + 0.55 * heatmap, 0.0, 1.0)
    return image_np, overlay


def _run_clip_patch_cam(
    clip_model,
    preprocess,
    raw_image: Image.Image,
    target_text: str,
    device: torch.device,
    base_path: str,
    invert_cam: bool,
) -> str:
    if clip is None:
        return ""

    if not hasattr(clip_model, "visual") or not hasattr(clip_model.visual, "conv1"):
        raise RuntimeError("Unsupported CLIP visual backbone: expected a model with visual.conv1")

    image_tensor = preprocess(raw_image).unsqueeze(0).to(device)
    image_tensor.requires_grad_(True)

    activation_cache: Dict[str, torch.Tensor] = {}
    gradient_cache: Dict[str, torch.Tensor] = {}
    target_layer = clip_model.visual.conv1

    def _forward_hook(_module, _inputs, output):
        activation_cache["value"] = output

    def _backward_hook(_module, _grad_input, grad_output):
        gradient_cache["value"] = grad_output[0]

    handle_f = target_layer.register_forward_hook(_forward_hook)
    handle_b = target_layer.register_full_backward_hook(_backward_hook)

    try:
        clip_model.zero_grad(set_to_none=True)
        image_features = clip_model.encode_image(image_tensor)

        text_tokens = clip.tokenize([target_text], truncate=True).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        score = (image_features * text_features.detach().to(image_features.dtype)).sum()
        score.backward()
    finally:
        handle_f.remove()
        handle_b.remove()

    if "value" not in activation_cache or "value" not in gradient_cache:
        raise RuntimeError("Patch-CAM hooks did not capture activations/gradients")

    activations = activation_cache["value"].detach().float()
    gradients = gradient_cache["value"].detach().float()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = nnf.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = nnf.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
    cam_map = cam[0, 0].detach().cpu().numpy()
    cam_map = _normalize_cam(cam_map, invert=invert_cam)

    input_np, overlay = _overlay_on_raw_image(raw_image, cam_map)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(input_np)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(cam_map, cmap="jet")
    axes[1].set_title("CLIP Patch-CAM (high relevance = red)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    title_text = target_text if len(target_text) <= 72 else target_text[:69] + "..."
    fig.suptitle(f"CLIP spatial text: {title_text}", fontsize=11)
    fig.tight_layout()

    out_path = f"{base_path}_clip_patch_cam.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _run_clipcap_patchcam(args: argparse.Namespace, device: torch.device, base_path: str) -> Dict[str, str]:
    if clip is None:
        raise ImportError("clip package is required for clipcap mode. Install with: pip install git+https://github.com/openai/CLIP.git")

    model = _build_clipcap_model(args, device).to(device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    clip_model, preprocess = clip.load(args.clip_model_type, device=device, jit=False)
    raw_image = Image.open(args.image).convert("RGB")
    input_png = f"{base_path}_input.png"

    with torch.no_grad():
        image_tensor = preprocess(raw_image).unsqueeze(0).to(device)
        clip_prefix = clip_model.encode_image(image_tensor).to(device, dtype=torch.float32)
        if args.normalize_prefix:
            clip_prefix = clip_prefix / clip_prefix.norm(2, dim=-1, keepdim=True)
        prefix_embed = model.clip_project(clip_prefix).reshape(1, args.prefix_length, -1)

    caption, token_ids = _decode_clipcap_caption(
        model=model,
        tokenizer=tokenizer,
        prefix_embed=prefix_embed,
        max_len=args.entry_length,
        temperature=args.temperature,
        stop_token=args.stop_token,
    )

    target_text = _normalize_text(args.clip_spatial_text) if args.clip_spatial_text.strip() else caption
    if not target_text:
        target_text = caption

    clip_patch_cam_png = _run_clip_patch_cam(
        clip_model=clip_model,
        preprocess=preprocess,
        raw_image=raw_image,
        target_text=target_text,
        device=device,
        base_path=base_path,
        invert_cam=args.clip_patch_cam_invert,
    )

    meta = {
        "model_arch": "clipcap",
        "mapping_type": args.mapping_type,
        "only_prefix": bool(args.only_prefix),
        "caption": caption,
        "token_ids": token_ids,
        "clip_model_type": args.clip_model_type,
        "clip_spatial_text": target_text,
        "clip_patch_cam_invert": bool(args.clip_patch_cam_invert),
        "clip_patch_cam_png": clip_patch_cam_png,
        "input_png": input_png,
    }
    meta_path = f"{base_path}_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "caption": caption,
        "summary_json": meta_path,
        "clip_patch_cam_png": clip_patch_cam_png,
        "input_png": input_png,
    }


def _build_cnn_rnn_model(args: argparse.Namespace, device: torch.device) -> Tuple[object, object]:
    checkpoint = _safe_torch_load(args.checkpoint, map_location=device, weights_only=False)
    if "vocab" not in checkpoint:
        raise ValueError("Checkpoint does not include vocabulary. Re-train cnn_rnn with train.py --model_arch cnn_rnn")

    vocab_data = checkpoint["vocab"]
    vocab = Vocabulary(stoi=vocab_data["stoi"], itos=vocab_data["itos"])
    ckpt_args = checkpoint.get("args", {})

    embed_size = int(args.embed_size or ckpt_args.get("embed_size", 512))
    hidden_size = int(args.hidden_size or ckpt_args.get("hidden_size", 512))
    rnn_layers = int(args.rnn_layers or ckpt_args.get("rnn_layers", ckpt_args.get("num_layers", 1)))
    dropout = float(args.dropout if args.dropout > 0 else ckpt_args.get("dropout", 0.1))
    unfreeze_cnn = bool(ckpt_args.get("unfreeze_cnn", False))

    model = CNNRNNCaptioner(
        vocab_size=len(vocab.itos),
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=rnn_layers,
        freeze_backbone=not unfreeze_cnn,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, vocab


def _find_last_conv2d(module: nn.Module) -> nn.Conv2d:
    last_conv = None
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            last_conv = child
    if last_conv is None:
        raise RuntimeError("Could not find a Conv2d layer for Grad-CAM")
    return last_conv


def _generate_cnn_rnn_caption_and_target(
    model: object,
    vocab: object,
    image_tensor: torch.Tensor,
    device: torch.device,
    max_len: int,
    temperature: float,
) -> Tuple[str, List[int], torch.Tensor]:
    model.eval()

    start_idx = vocab.stoi.get(START_TOKEN)
    end_idx = vocab.stoi.get(END_TOKEN)
    special_ids = {
        vocab.stoi.get(PAD_TOKEN, -1),
        vocab.stoi.get(UNK_TOKEN, -1),
        start_idx,
        end_idx,
    }

    image_features = model.encoder(image_tensor.to(device))
    batch_size = image_features.size(0)

    h = model.decoder.init_h(image_features).view(batch_size, model.decoder.num_layers, model.decoder.hidden_size)
    c = model.decoder.init_c(image_features).view(batch_size, model.decoder.num_layers, model.decoder.hidden_size)
    h = h.transpose(0, 1).contiguous()
    c = c.transpose(0, 1).contiguous()

    token = torch.tensor([[start_idx]], dtype=torch.long, device=device)
    output_ids: List[int] = []
    score_terms: List[torch.Tensor] = []

    for _ in range(max_len):
        embedding = model.decoder.embed(token)
        out, (h, c) = model.decoder.lstm(embedding, (h, c))
        logits = model.decoder.fc(out[:, -1, :])
        if temperature > 0:
            logits = logits / temperature
        next_id = int(torch.argmax(logits, dim=-1).item())

        score_terms.append(logits[0, next_id])
        output_ids.append(next_id)

        if next_id == end_idx:
            break
        token = torch.tensor([[next_id]], dtype=torch.long, device=device)

    words = [vocab.itos[idx] for idx in output_ids if idx not in special_ids and 0 <= idx < len(vocab.itos)]
    caption = _normalize_text(" ".join(words))
    target_score = torch.stack(score_terms).sum() if score_terms else torch.tensor(0.0, device=device)
    return caption, output_ids, target_score


def _make_cnn_gradcam_overlay(image_tensor: torch.Tensor, cam_map: np.ndarray) -> np.ndarray:
    image = image_tensor.detach().cpu()[0]
    image = (image * NORM_STD + NORM_MEAN).clamp(0.0, 1.0)
    image_np = image.permute(1, 2, 0).numpy()

    heatmap = plt.get_cmap("jet")(cam_map)[..., :3]
    overlay = np.clip(0.45 * image_np + 0.55 * heatmap, 0.0, 1.0)
    return overlay


def _run_cnn_rnn_gradcam(args: argparse.Namespace, device: torch.device, base_path: str) -> Dict[str, str]:
    if CNN_RNN_IMPORT_ERROR is not None or transforms is None:
        raise ImportError(
            "cnn_rnn mode requires torchvision and PIL. Install with: pip install torchvision pillow"
        ) from CNN_RNN_IMPORT_ERROR

    model, vocab = _build_cnn_rnn_model(args, device)
    model = model.to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN.flatten().tolist(), std=NORM_STD.flatten().tolist()),
        ]
    )

    image = Image.open(args.image).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    image_tensor.requires_grad_(True)

    target_layer = _find_last_conv2d(model.encoder.backbone)
    activation_cache: Dict[str, torch.Tensor] = {}
    gradient_cache: Dict[str, torch.Tensor] = {}

    def _forward_hook(_module, _inputs, output):
        activation_cache["value"] = output

    def _backward_hook(_module, _grad_input, grad_output):
        gradient_cache["value"] = grad_output[0]

    handle_f = target_layer.register_forward_hook(_forward_hook)
    handle_b = target_layer.register_full_backward_hook(_backward_hook)

    try:
        grad_ctx = torch.backends.cudnn.flags(enabled=False) if device.type == "cuda" else nullcontext()
        with grad_ctx:
            model.zero_grad(set_to_none=True)
            caption, token_ids, target_score = _generate_cnn_rnn_caption_and_target(
                model=model,
                vocab=vocab,
                image_tensor=image_tensor,
                device=device,
                max_len=args.cnn_max_len,
                temperature=args.temperature,
            )
            target_score.backward()
    finally:
        handle_f.remove()
        handle_b.remove()

    if "value" not in activation_cache or "value" not in gradient_cache:
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

    activations = activation_cache["value"]
    gradients = gradient_cache["value"]
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = nnf.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = nnf.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
    cam_map = cam[0, 0].detach().cpu().numpy()
    cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)

    image_np = (image_tensor.detach().cpu()[0] * NORM_STD + NORM_MEAN).clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    overlay = _make_cnn_gradcam_overlay(image_tensor, cam_map)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(cam_map, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(f"CNN-RNN caption: {caption}", fontsize=11)
    fig.tight_layout()
    gradcam_path = f"{base_path}_gradcam.png"
    fig.savefig(gradcam_path, dpi=220)
    plt.close(fig)

    meta = {
        "model_arch": "cnn_rnn",
        "caption": caption,
        "token_ids": token_ids,
        "gradcam_png": gradcam_path,
    }
    meta_path = f"{base_path}_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "caption": caption,
        "summary_json": meta_path,
        "gradcam_png": gradcam_path,
    }


def _default_output_prefix(args: argparse.Namespace) -> str:
    stem = os.path.splitext(os.path.basename(args.image))[0]
    return f"{args.model_arch}_{stem}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize image captioning models: Grad-CAM (CNN-RNN) and Patch-CAM (ClipCap)."
    )
    parser.add_argument("--model_arch", required=True, choices=["clipcap", "cnn_rnn"])
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--out_dir", default="./visualizations", help="Output directory")
    parser.add_argument("--output_prefix", default="", help="Output file prefix")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--mapping_type", default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--only_prefix", action="store_true")
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--prefix_length_clip", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--is_rn", action="store_true")
    parser.add_argument("--normalize_prefix", action="store_true")
    parser.add_argument("--clip_model_type", default="ViT-B/32")
    parser.add_argument("--entry_length", type=int, default=67)
    parser.add_argument("--stop_token", default=".")
    parser.add_argument("--clip_spatial_text", default="", help="Optional text target used for ClipCap patch-CAM")
    parser.add_argument("--clip_patch_cam_invert", dest="clip_patch_cam_invert", action="store_true")
    parser.add_argument("--no_clip_patch_cam_invert", dest="clip_patch_cam_invert", action="store_false")
    parser.set_defaults(clip_patch_cam_invert=True)

    parser.add_argument("--cnn_max_len", type=int, default=30)
    parser.add_argument("--embed_size", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=0)
    parser.add_argument("--rnn_layers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = args.output_prefix.strip() or _default_output_prefix(args)
    base_path = os.path.join(args.out_dir, prefix)

    device = torch.device(args.device)
    if args.model_arch == "clipcap":
        outputs = _run_clipcap_patchcam(args, device, base_path)
    else:
        outputs = _run_cnn_rnn_gradcam(args, device, base_path)

    print("Visualization complete")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
