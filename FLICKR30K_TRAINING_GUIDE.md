# Flickr30k Training Guide for ClipCap

This guide explains how to prepare Flickr30k, build CLIP embeddings, and train ClipCap in this repository.

## 1. Project Location

- Repo root: D:/WORKSPACE/clipcap/CLIP_prefix_caption

## 2. Expected Flickr30k Layout

Put your dataset under data/flickr30k:

- data/flickr30k/flickr30k-images/*.jpg
- data/flickr30k/results_20130124.token

Optional (Karpathy format):

- data/flickr30k/dataset_flickr30k.json

## 3. Download Flickr30k

Flickr30k is distributed with usage restrictions, so the easiest practical path is usually via Kaggle mirrors.

### Method A: Kaggle API (recommended)

1. Create a Kaggle account and accept the dataset terms on the Flickr30k dataset page.
2. Create an API token in Kaggle account settings and place kaggle.json at:
  - Windows: C:/Users/<your_user>/.kaggle/kaggle.json
3. Run:

```powershell
cd D:\WORKSPACE\clipcap\CLIP_prefix_caption
pip install kaggle
kaggle datasets download -d adityajn105/flickr30k -p .\data\flickr30k
Expand-Archive .\data\flickr30k\flickr30k.zip -DestinationPath .\data\flickr30k -Force
```

If your downloaded zip has a different name, replace flickr30k.zip with the actual filename.

### Method B: Download from browser

1. Open a Flickr30k dataset mirror on Kaggle.
2. Click Download.
3. Extract files into data/flickr30k.

After extraction, make sure these paths exist:

- data/flickr30k/flickr30k-images
- data/flickr30k/results_20130124.token

### Quick verification

```powershell
Get-ChildItem .\data\flickr30k\flickr30k-images -Filter *.jpg | Measure-Object
Test-Path .\data\flickr30k\results_20130124.token
```

Expected:
- Image count is around 31k files.
- Test-Path returns True.

## 4. Environment Setup

### Option A: Conda (same style as original repo)

```bash
conda env create -f environment.yml
conda activate clip_prefix_caption
```

### Option B: Python venv (Windows PowerShell)

```powershell
cd D:\WORKSPACE\clipcap\CLIP_prefix_caption
..\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/openai/CLIP.git
pip install transformers==4.11.3 tqdm scikit-image pillow
```

Notes:
- If you do not have CUDA, install CPU PyTorch wheels instead.
- If CLIP install fails, ensure Git is available in PATH.

## 5. Build Flickr30k CLIP Embeddings

### From token file format

```bash
python parse_flickr30k.py \
  --clip_model_type ViT-B/32 \
  --images_dir ./data/flickr30k/flickr30k-images \
  --captions_file ./data/flickr30k/results_20130124.token
```

### From Karpathy JSON format

```bash
python parse_flickr30k.py \
  --clip_model_type ViT-B/32 \
  --images_dir ./data/flickr30k/flickr30k-images \
  --karpathy_json ./data/flickr30k/dataset_flickr30k.json
```

Default output:

- data/flickr30k/flickr30k_clip_ViT-B_32.pkl

## 6. Train ClipCap on Flickr30k

### Full training (fine-tune GPT-2 + mapping)

```bash
python train.py \
  --data ./data/flickr30k/flickr30k_clip_ViT-B_32.pkl \
  --out_dir ./flickr30k_train/ \
  --prefix flickr30k_prefix \
  --device cuda:0
```

### Train only prefix mapping (lighter)

```bash
python train.py \
  --only_prefix \
  --data ./data/flickr30k/flickr30k_clip_ViT-B_32.pkl \
  --out_dir ./flickr30k_train/ \
  --prefix flickr30k_prefix \
  --mapping_type transformer \
  --num_layers 8 \
  --prefix_length 40 \
  --prefix_length_clip 40 \
  --device cuda:0
```

If you run on CPU:

```bash
python train.py --data ./data/flickr30k/flickr30k_clip_ViT-B_32.pkl --device cpu
```

## 7. Useful Training Arguments

- --epochs: number of epochs
- --bs: batch size
- --save_every: save checkpoint every N epochs
- --normalize_prefix: normalize CLIP prefix vectors
- --mapping_type: mlp or transformer

Example:

```bash
python train.py --data ./data/flickr30k/flickr30k_clip_ViT-B_32.pkl --epochs 20 --bs 32 --save_every 1
```

## 8. Output Files

Training checkpoints are written to out_dir:

- flickr30k_prefix-000.pt
- flickr30k_prefix-001.pt
- ...

## 9. Common Issues

1. Import errors (clip/torch/PIL not found)
- Cause: dependencies not installed in active environment.
- Fix: activate environment and reinstall required packages.

2. CUDA out of memory
- Cause: batch size or sequence settings too large.
- Fix: reduce --bs, use --only_prefix, or lower prefix_length.

3. No samples found during parsing
- Cause: wrong images_dir or annotation path.
- Fix: verify dataset paths and file names.

4. Missing images warning while parsing
- Cause: annotation references files not present in images_dir.
- Fix: sync image folder with annotations.

## 10. Minimal End-to-End Commands

```bash
python parse_flickr30k.py --clip_model_type ViT-B/32 --images_dir ./data/flickr30k/flickr30k-images --captions_file ./data/flickr30k/results_20130124.token
python train.py --data ./data/flickr30k/flickr30k_clip_ViT-B_32.pkl --out_dir ./flickr30k_train/ --prefix flickr30k_prefix --device cuda:0
```
