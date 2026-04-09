import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple

import clip
import torch
from PIL import Image
from tqdm import tqdm


def add_period(caption: str) -> str:
    caption = caption.strip()
    if not caption:
        return caption
    if caption[-1] != '.':
        return f"{caption}."
    if len(caption) > 1 and caption[-2] == ' ':
        return caption[:-2] + '.'
    return caption


def parse_token_file(captions_file: str) -> Dict[str, List[str]]:
    """
    Parse Flickr30k token file formats:
    1) image_name#idx<TAB>caption
    2) image_name| idx | caption
    Always extract only the image filename (before | or #) for image lookup.
    """
    image_to_captions: Dict[str, List[str]] = {}
    with open(captions_file, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if '\t' in line:
                key, caption = line.split('\t', 1)
                # Support both # and | in key
                image_name = key.split('|', 1)[0].split('#', 1)[0].strip()
            elif '|' in line:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) < 3:
                    continue
                image_name = parts[0].split('#', 1)[0].strip()
                caption = parts[2]
            else:
                continue

            caption = add_period(caption)
            if not image_name or not caption:
                continue
            image_to_captions.setdefault(image_name, []).append(caption)
    return image_to_captions


def parse_karpathy_json(karpathy_json: str, split: str = 'all') -> Dict[str, List[str]]:
    """
    Parse Karpathy split JSON format (dataset_flickr30k.json).
    """
    with open(karpathy_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images = data.get('images', [])
    image_to_captions: Dict[str, List[str]] = {}
    for item in images:
        item_split = item.get('split', '')
        if split != 'all' and item_split != split:
            continue
        image_name = item.get('filename')
        if not image_name:
            continue
        sentences = item.get('sentences', [])
        for sentence in sentences:
            raw_caption = sentence.get('raw')
            if not raw_caption:
                continue
            caption = add_period(raw_caption)
            if caption:
                image_to_captions.setdefault(image_name, []).append(caption)
    return image_to_captions


def collect_samples(captions_map: Dict[str, List[str]], images_dir: str) -> List[Tuple[str, str]]:
    samples: List[Tuple[str, str]] = []
    missing_images = 0
    for image_name, captions in captions_map.items():
        image_path = os.path.join(images_dir, image_name)
        if not os.path.isfile(image_path):
            missing_images += 1
            continue
        for caption in captions:
            samples.append((image_name, caption))

    if missing_images:
        print(f"Warning: skipped {missing_images} images because files were not found in {images_dir}")
    return samples


def encode_images(samples: List[Tuple[str, str]], images_dir: str, clip_model_type: str, device: torch.device):
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    clip_model = clip_model.eval()

    unique_images = sorted({image_name for image_name, _ in samples})
    image_to_index: Dict[str, int] = {}
    all_embeddings: List[torch.Tensor] = []

    for idx, image_name in enumerate(tqdm(unique_images, desc='Encoding images')):
        image_path = os.path.join(images_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        all_embeddings.append(prefix)
        image_to_index[image_name] = idx

    captions = []
    for image_name, caption in samples:
        captions.append({
            'caption': caption,
            'clip_embedding': image_to_index[image_name],
            'image_id': image_name,
        })

    return torch.cat(all_embeddings, dim=0), captions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data/flickr30k')
    parser.add_argument('--images_dir', default='./data/flickr30k/flickr30k-images')
    parser.add_argument('--captions_file', default='./data/flickr30k/results_20130124.token')
    parser.add_argument('--karpathy_json', default='')
    parser.add_argument('--split', default='all', choices=('all', 'train', 'val', 'test'))
    parser.add_argument('--out_path', default='')
    parser.add_argument('--clip_model_type', default='ViT-B/32', choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.karpathy_json:
        captions_map = parse_karpathy_json(args.karpathy_json, split=args.split)
    else:
        if args.split != 'all':
            raise ValueError('--split requires --karpathy_json because token files do not include split labels')
        captions_map = parse_token_file(args.captions_file)

    samples = collect_samples(captions_map, args.images_dir)
    if not samples:
        raise RuntimeError('No valid image-caption samples were found. Check your dataset paths and annotation file.')

    device = torch.device(args.device)
    clip_embeddings, captions = encode_images(samples, args.images_dir, args.clip_model_type, device)

    clip_model_name = args.clip_model_type.replace('/', '_')
    if args.out_path:
        out_path = args.out_path
    elif args.split == 'all':
        out_path = os.path.join(args.data_root, f'flickr30k_clip_{clip_model_name}.pkl')
    else:
        out_path = os.path.join(args.data_root, f'flickr30k_clip_{clip_model_name}_{args.split}.pkl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'wb') as f:
        pickle.dump({'clip_embedding': clip_embeddings, 'captions': captions}, f)

    print(f'Done. Saved {len(captions)} captions and {clip_embeddings.shape[0]} image embeddings to {out_path}')


if __name__ == '__main__':
    main()
