import argparse
import json
import os
import pickle
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import GPT2Tokenizer

from train import ClipCaptionModel, ClipCaptionPrefix, MappingType

CNN_RNN_IMPORT_ERROR = None
try:
    from train_cnn_rnn import (
        CNNRNNCaptioner,
        END_TOKEN,
        PAD_TOKEN,
        START_TOKEN,
        UNK_TOKEN,
        Vocabulary,
        parse_karpathy_json,
        parse_token_file,
    )
except ImportError as exc:
    CNN_RNN_IMPORT_ERROR = exc
    CNNRNNCaptioner = None
    Vocabulary = None
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    parse_karpathy_json = None
    parse_token_file = None


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def generate_beam(
    model: ClipCaptionModel,
    tokenizer: GPT2Tokenizer,
    beam_size: int = 5,
    embed: torch.Tensor = None,
    entry_length: int = 67,
    temperature: float = 1.0,
    stop_token: str = ".",
) -> str:
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)

    with torch.no_grad():
        generated = embed
        for _ in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                tokens = next_tokens
            else:
                logits[is_stopped] = -float("inf")
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]

            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break

    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)], skip_special_tokens=True)
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    return _normalize_text(output_texts[order[0]])


def generate_nucleus(
    model: ClipCaptionModel,
    tokenizer: GPT2Tokenizer,
    embed: torch.Tensor = None,
    entry_length: int = 67,
    top_p: float = 0.8,
    temperature: float = 1.0,
    stop_token: str = ".",
) -> str:
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("inf")
    tokens = None
    generated = embed

    with torch.no_grad():
        for _ in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            next_token_embed = model.gpt.transformer.wte(next_token)

            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            generated = torch.cat((generated, next_token_embed), dim=1)

            if stop_token_index == next_token.item():
                break

    output_text = tokenizer.decode(tokens.squeeze().cpu().numpy(), skip_special_tokens=True)
    return _normalize_text(output_text)


def load_eval_data(data_path: str) -> Tuple[torch.Tensor, List[str], Dict[str, List[str]], Dict[str, int]]:
    with open(data_path, "rb") as f:
        all_data = pickle.load(f)

    prefixes = all_data["clip_embedding"]
    captions_raw = all_data["captions"]

    image_id_to_refs: Dict[str, List[str]] = OrderedDict()
    image_id_to_embedding_idx: Dict[str, int] = OrderedDict()

    for item in captions_raw:
        image_id = str(item["image_id"])
        caption = _normalize_text(item["caption"])
        image_id_to_refs.setdefault(image_id, []).append(caption)
        if image_id not in image_id_to_embedding_idx:
            image_id_to_embedding_idx[image_id] = int(item["clip_embedding"])

    image_ids = list(image_id_to_refs.keys())
    return prefixes, image_ids, image_id_to_refs, image_id_to_embedding_idx


def build_model(args: argparse.Namespace) -> ClipCaptionModel:
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

    device = getattr(args, "device", "cpu")
    # Ensure checkpoint is loaded to the correct device and handle weights_only for PyTorch >=2.1
    try:
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    return model


def compute_bert_score(references: Dict[str, List[str]], predictions: Dict[str, List[str]]) -> Dict[str, float]:
    try:
        from bert_score import score
    except ImportError:
        print("Warning: bert_score not installed. Install with: pip install bert-score")
        return {}

    cands = []
    refs = []
    for image_id, preds in predictions.items():
        cands.append(preds[0])
        refs.append(references[image_id])

    print("Computing BERTScore...")
    P, R, F1 = score(cands, refs, lang="en", verbose=False)
    return {"BERTScore_P": float(P.mean().item()), "BERTScore_R": float(R.mean().item()), "BERTScore_F1": float(F1.mean().item())}


def compute_coco_metrics(references: Dict[str, List[str]], predictions: Dict[str, List[str]]) -> Dict[str, float]:
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.spice.spice import Spice
    except ImportError as exc:
        raise ImportError(
            "Missing COCO caption eval packages. Install with: "
            "pip install pycocoevalcap pycocotools"
        ) from exc

    scorers = []
    
    try:
        scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
    except Exception:
        pass
        
    try:
        scorers.append((Meteor(), ["METEOR"]))
    except Exception as exc:
        print(f"Warning: failed to initialize METEOR (Java may be missing): {exc}")
        
    try:
        scorers.append((Rouge(), ["ROUGE_L"]))
    except Exception:
        pass
        
    try:
        scorers.append((Cider(), ["CIDEr"]))
    except Exception:
        pass
        
    try:
        scorers.append((Spice(), ["SPICE"]))
    except Exception as exc:
        print(f"Warning: failed to initialize SPICE (Java may be missing): {exc}")

    metrics: Dict[str, float] = {}
    for scorer, method in scorers:
        try:
            score, _ = scorer.compute_score(references, predictions)
            if isinstance(method, list) and len(method) > 1:
                for name, value in zip(method, score):
                    metrics[name] = float(value)
            else:
                metrics[method[0]] = float(score)
        except Exception as exc:
            name = ",".join(method)
            print(f"Warning: failed to compute {name}: {exc}")

    return metrics


def load_cnn_rnn_eval_data(args: argparse.Namespace) -> Tuple[List[str], Dict[str, List[str]]]:
    if args.karpathy_json:
        captions_map = parse_karpathy_json(args.karpathy_json, split=args.split)
    elif args.captions_file:
        captions_map = parse_token_file(args.captions_file)
    else:
        raise ValueError("For cnn_rnn mode, provide --karpathy_json or --captions_file")

    references: Dict[str, List[str]] = OrderedDict()
    for image_id, captions in captions_map.items():
        image_path = os.path.join(args.images_dir, image_id)
        if not os.path.isfile(image_path):
            continue
        references[image_id] = [_normalize_text(caption) for caption in captions]

    image_ids = list(references.keys())
    if args.max_samples > 0:
        image_ids = image_ids[: args.max_samples]
        references = OrderedDict((image_id, references[image_id]) for image_id in image_ids)
    return image_ids, references


def build_cnn_rnn_model(args: argparse.Namespace) -> Tuple[CNNRNNCaptioner, Vocabulary]:
    try:
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)

    if "vocab" not in checkpoint:
        raise ValueError("Checkpoint does not include vocabulary. Re-train with cnn_rnn mode in train.py")

    vocab_data = checkpoint["vocab"]
    vocab = Vocabulary(stoi=vocab_data["stoi"], itos=vocab_data["itos"])
    ckpt_args = checkpoint.get("args", {})

    embed_size = int(getattr(args, "embed_size", 0) or ckpt_args.get("embed_size", 512))
    hidden_size = int(getattr(args, "hidden_size", 0) or ckpt_args.get("hidden_size", 512))
    rnn_layers = int(getattr(args, "rnn_layers", 0) or ckpt_args.get("rnn_layers", ckpt_args.get("num_layers", 1)))
    dropout = float(getattr(args, "dropout", 0.0) or ckpt_args.get("dropout", 0.1))
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


def generate_cnn_rnn_caption(
    model: CNNRNNCaptioner,
    vocab: Vocabulary,
    image_tensor: torch.Tensor,
    device: torch.device,
    max_len: int,
    temperature: float,
) -> str:
    model.eval()
    start_idx = vocab.stoi.get(START_TOKEN)
    end_idx = vocab.stoi.get(END_TOKEN)
    special_ids = {
        vocab.stoi.get(PAD_TOKEN, -1),
        vocab.stoi.get(UNK_TOKEN, -1),
        start_idx,
        end_idx,
    }

    with torch.no_grad():
        image_features = model.encoder(image_tensor.to(device))
        batch_size = image_features.size(0)

        h = model.decoder.init_h(image_features).view(batch_size, model.decoder.num_layers, model.decoder.hidden_size)
        c = model.decoder.init_c(image_features).view(batch_size, model.decoder.num_layers, model.decoder.hidden_size)
        h = h.transpose(0, 1).contiguous()
        c = c.transpose(0, 1).contiguous()

        token = torch.tensor([[start_idx]], dtype=torch.long, device=device)
        output_tokens: List[str] = []

        for _ in range(max_len):
            embedding = model.decoder.embed(token)
            out, (h, c) = model.decoder.lstm(embedding, (h, c))
            logits = model.decoder.fc(out[:, -1, :])
            if temperature > 0:
                logits = logits / temperature
            next_id = int(torch.argmax(logits, dim=-1).item())

            if next_id == end_idx:
                break
            if next_id not in special_ids and 0 <= next_id < len(vocab.itos):
                output_tokens.append(vocab.itos[next_id])

            token = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return _normalize_text(" ".join(output_tokens))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate image captioning models with COCO-style metrics.")
    parser.add_argument("--model_arch", default="clipcap", choices=["clipcap", "cnn_rnn"])
    parser.add_argument("--data", default="", help="Path to dataset .pkl generated by parse scripts (clipcap mode)")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--mapping_type", default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--only_prefix", action="store_true")
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--prefix_length_clip", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--is_rn", action="store_true")
    parser.add_argument("--normalize_prefix", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decode", default="beam", choices=["beam", "nucleus"])
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--entry_length", type=int, default=67)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means evaluate all images")
    parser.add_argument("--save_predictions", default="", help="Optional JSON output path")

    parser.add_argument("--images_dir", default="", help="Image folder (cnn_rnn mode)")
    parser.add_argument("--captions_file", default="", help="Caption token file/csv (cnn_rnn mode)")
    parser.add_argument("--karpathy_json", default="", help="Karpathy json annotation (cnn_rnn mode)")
    parser.add_argument("--split", default="test", choices=("all", "train", "val", "test"))
    parser.add_argument("--cnn_max_len", type=int, default=30)
    parser.add_argument("--embed_size", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=0)
    parser.add_argument("--rnn_layers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device(args.device)

    predictions: Dict[str, List[str]] = OrderedDict()
    eval_references: Dict[str, List[str]] = OrderedDict()

    if args.model_arch == "clipcap":
        if not args.data:
            raise ValueError("--data is required in clipcap mode")

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        print("Loading dataset...")
        prefixes, image_ids, references, image_to_embedding = load_eval_data(args.data)
        if args.max_samples > 0:
            image_ids = image_ids[: args.max_samples]

        print(f"Unique images for evaluation: {len(image_ids)}")
        print("Loading model checkpoint...")
        model = build_model(args).to(device).eval()

        for image_id in tqdm(image_ids, desc="Generating captions"):
            embedding_idx = image_to_embedding[image_id]
            prefix = prefixes[embedding_idx].unsqueeze(0).to(device, dtype=torch.float32)

            if args.normalize_prefix:
                prefix = prefix / prefix.norm(2, dim=-1, keepdim=True)

            with torch.no_grad():
                prefix_embed = model.clip_project(prefix).reshape(1, args.prefix_length, -1)

            if args.decode == "beam":
                pred_caption = generate_beam(
                    model=model,
                    tokenizer=tokenizer,
                    beam_size=args.beam_size,
                    embed=prefix_embed,
                    entry_length=args.entry_length,
                    temperature=args.temperature,
                )
            else:
                pred_caption = generate_nucleus(
                    model=model,
                    tokenizer=tokenizer,
                    embed=prefix_embed,
                    entry_length=args.entry_length,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )

            predictions[image_id] = [pred_caption]
            eval_references[image_id] = references[image_id]
    else:
        if CNN_RNN_IMPORT_ERROR is not None:
            raise ImportError(
                "cnn_rnn mode requires torchvision/PIL dependencies. Install with: pip install torchvision pillow tqdm"
            ) from CNN_RNN_IMPORT_ERROR

        if not args.images_dir:
            raise ValueError("--images_dir is required in cnn_rnn mode")

        print("Loading annotations...")
        image_ids, eval_references = load_cnn_rnn_eval_data(args)
        print(f"Unique images for evaluation: {len(image_ids)}")

        print("Loading model checkpoint...")
        model, vocab = build_cnn_rnn_model(args)
        model = model.to(device).eval()

        try:
            from torchvision import transforms as tv_transforms
        except ImportError as exc:
            raise ImportError("torchvision is required for cnn_rnn mode. Install with: pip install torchvision") from exc

        image_transform = tv_transforms.Compose(
            [
                tv_transforms.Resize(256),
                tv_transforms.CenterCrop(224),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        for image_id in tqdm(image_ids, desc="Generating captions"):
            image_path = os.path.join(args.images_dir, image_id)
            image = Image.open(image_path).convert("RGB")
            image_tensor = image_transform(image).unsqueeze(0)

            pred_caption = generate_cnn_rnn_caption(
                model=model,
                vocab=vocab,
                image_tensor=image_tensor,
                device=device,
                max_len=args.cnn_max_len,
                temperature=args.temperature,
            )
            predictions[image_id] = [pred_caption]

    print("Computing metrics...")
    metrics = compute_coco_metrics(eval_references, predictions)
    bert_metrics = compute_bert_score(eval_references, predictions)
    metrics.update(bert_metrics)

    print("\n=== Evaluation Results (ClipCap paper metrics) ===")
    metric_order = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE", "BERTScore_P", "BERTScore_R", "BERTScore_F1"]
    for key in metric_order:
        if key in metrics:
            print(f"{key}: {metrics[key]:.4f}")

    if args.save_predictions:
        out_obj = {
            "metrics": metrics,
            "predictions": predictions,
            "references": eval_references,
            "config": vars(args),
        }
        os.makedirs(os.path.dirname(args.save_predictions) or ".", exist_ok=True)
        with open(args.save_predictions, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(f"Saved predictions and metrics to {args.save_predictions}")


if __name__ == "__main__":
    main()
