import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
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
except ImportError as exc:
    raise ImportError(
        "clip package is required. Install with: pip install git+https://github.com/openai/CLIP.git"
    ) from exc


def safe_torch_load(path: str, map_location: torch.device, weights_only: bool):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def sanitize_filename(text: str, fallback: str = "token") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    cleaned = cleaned.strip("_")
    return cleaned[:40] or fallback


def normalize_cam(cam: np.ndarray, invert: bool = True) -> np.ndarray:
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    if invert:
        cam = 1.0 - cam
    return cam


def overlay_on_raw_image(raw_image: Image.Image, cam_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = cam_map.shape
    image_np = np.asarray(raw_image.resize((w, h), Image.Resampling.BICUBIC)).astype(np.float32) / 255.0
    heatmap = plt.get_cmap("jet")(cam_map)[..., :3]
    overlay = np.clip(0.45 * image_np + 0.55 * heatmap, 0.0, 1.0)
    return image_np, overlay


def build_model(args: argparse.Namespace, device: torch.device):
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

    state_dict = safe_torch_load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def compute_patch_cam(
    clip_model,
    preprocess,
    raw_image: Image.Image,
    target_text: str,
    device: torch.device,
    invert_cam: bool,
) -> np.ndarray:
    image_tensor = preprocess(raw_image).unsqueeze(0).to(device)
    image_tensor.requires_grad_(True)

    activation_cache: Dict[str, torch.Tensor] = {}
    gradient_cache: Dict[str, torch.Tensor] = {}
    target_layer = clip_model.visual.conv1

    def forward_hook(_module, _inputs, output):
        activation_cache["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradient_cache["value"] = grad_output[0]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

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
    return normalize_cam(cam_map, invert=invert_cam)


def render_step_figure(
    raw_image: Image.Image,
    cam_map: np.ndarray,
    out_path: str,
    step_idx: int,
    token_text: str,
    context_before: str,
    caption_after: str,
) -> None:
    input_np, overlay = overlay_on_raw_image(raw_image, cam_map)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(input_np)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(cam_map, cmap="jet")
    axes[1].set_title("CLIP Patch-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    token_label = token_text if token_text else "<empty>"
    fig.suptitle(
        f"Step {step_idx}: next token = {token_label}\n"
        f"Before: {context_before or '<bos>'}\n"
        f"After: {caption_after or '<empty>'}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_timeline(
    raw_image: Image.Image,
    steps: List[Dict[str, object]],
    out_path: str,
    max_steps: int,
) -> None:
    shown_steps = steps[:max_steps]
    if not shown_steps:
        return

    cols = min(4, len(shown_steps))
    rows = int(math.ceil(len(shown_steps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes[len(shown_steps):]:
        ax.axis("off")

    for ax, step in zip(axes, shown_steps):
        cam_map = np.array(step["cam_map"])
        _, overlay = overlay_on_raw_image(raw_image, cam_map)
        ax.imshow(overlay)
        token_text = step["token_text"] or "<empty>"
        ax.set_title(f"{step['step']}: {token_text}", fontsize=10)
        ax.axis("off")

    fig.suptitle("Token-by-token CLIP spatial focus (proxy via Patch-CAM)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize token-by-token CLIP spatial focus for ClipCap checkpoints with transformer mapping. "
            "This uses CLIP Patch-CAM as a spatial proxy because ClipCap itself consumes a global CLIP embedding."
        )
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", required=True, help="Path to ClipCap checkpoint (.pt)")
    parser.add_argument("--out_dir", default="./visualizations/token_focus", help="Output directory")
    parser.add_argument("--output_prefix", default="", help="Prefix for saved files")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--mapping_type", default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--only_prefix", action="store_true", help="Use when GPT-2 was frozen during training")
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--prefix_length_clip", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--is_rn", action="store_true")
    parser.add_argument("--normalize_prefix", action="store_true")
    parser.add_argument("--clip_model_type", default="ViT-B/32")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum generated tokens to visualize")
    parser.add_argument("--stop_token", default=".")
    parser.add_argument("--clip_patch_cam_invert", dest="clip_patch_cam_invert", action="store_true")
    parser.add_argument("--no_clip_patch_cam_invert", dest="clip_patch_cam_invert", action="store_false")
    parser.set_defaults(clip_patch_cam_invert=True)
    args = parser.parse_args()

    if args.mapping_type != "transformer":
        print("Warning: this script is intended for transformer mapping, but it can still run with --mapping_type mlp.")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    model = build_model(args, device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    clip_model, preprocess = clip.load(args.clip_model_type, device=device, jit=False)
    raw_image = Image.open(args.image).convert("RGB")

    prefix = args.output_prefix.strip() or os.path.splitext(os.path.basename(args.image))[0]
    base_dir = os.path.join(args.out_dir, prefix)
    os.makedirs(base_dir, exist_ok=True)

    with torch.no_grad():
        image_tensor = preprocess(raw_image).unsqueeze(0).to(device)
        clip_prefix = clip_model.encode_image(image_tensor).to(device, dtype=torch.float32)
        if args.normalize_prefix:
            clip_prefix = clip_prefix / clip_prefix.norm(2, dim=-1, keepdim=True)
        generated = model.clip_project(clip_prefix).reshape(1, args.prefix_length, -1)

    stop_token_id = tokenizer.encode(args.stop_token)[0]
    generated_token_ids: List[int] = []
    steps: List[Dict[str, object]] = []

    for step_idx in range(1, args.max_steps + 1):
        with torch.no_grad():
            outputs = model.gpt(inputs_embeds=generated, use_cache=False, return_dict=True)
            logits = outputs.logits[:, -1, :]
            if args.temperature > 0:
                logits = logits / args.temperature
            next_token = torch.argmax(logits, dim=-1)
            next_id = int(next_token.item())

        context_before = normalize_text(tokenizer.decode(generated_token_ids, skip_special_tokens=True))
        generated_token_ids.append(next_id)
        caption_after = normalize_text(tokenizer.decode(generated_token_ids, skip_special_tokens=True))
        token_text = normalize_text(tokenizer.decode([next_id], skip_special_tokens=True))
        target_text = caption_after or token_text or context_before or args.stop_token

        cam_map = compute_patch_cam(
            clip_model=clip_model,
            preprocess=preprocess,
            raw_image=raw_image,
            target_text=target_text,
            device=device,
            invert_cam=args.clip_patch_cam_invert,
        )

        token_stub = sanitize_filename(token_text, fallback=f"token_{next_id}")
        step_path = os.path.join(base_dir, f"step_{step_idx:03d}_{token_stub}.png")
        render_step_figure(
            raw_image=raw_image,
            cam_map=cam_map,
            out_path=step_path,
            step_idx=step_idx,
            token_text=token_text,
            context_before=context_before,
            caption_after=caption_after,
        )

        steps.append(
            {
                "step": step_idx,
                "token_id": next_id,
                "token_text": token_text,
                "context_before": context_before,
                "caption_after": caption_after,
                "target_text_for_cam": target_text,
                "step_figure": step_path,
                "cam_map": cam_map.tolist(),
            }
        )

        next_embed = model.gpt.transformer.wte(next_token).unsqueeze(1)
        generated = torch.cat((generated, next_embed), dim=1)

        if next_id == stop_token_id:
            break

    timeline_path = os.path.join(base_dir, "timeline.png")
    build_timeline(raw_image=raw_image, steps=steps, out_path=timeline_path, max_steps=args.max_steps)

    summary = {
        "image": args.image,
        "checkpoint": args.checkpoint,
        "mapping_type": args.mapping_type,
        "only_prefix": bool(args.only_prefix),
        "prefix_length": args.prefix_length,
        "prefix_length_clip": args.prefix_length_clip,
        "num_layers": args.num_layers,
        "clip_model_type": args.clip_model_type,
        "temperature": args.temperature,
        "generated_caption": steps[-1]["caption_after"] if steps else "",
        "timeline_png": timeline_path,
        "note": (
            "Spatial maps are CLIP Patch-CAM proxies conditioned on generated text so far. "
            "ClipCap itself uses a global CLIP embedding and does not expose native patch-level attention during decoding."
        ),
        "steps": steps,
    }

    summary_path = os.path.join(base_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Visualization complete")
    print(f"Output directory: {base_dir}")
    print(f"Generated caption: {summary['generated_caption']}")
    print(f"Timeline: {timeline_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
