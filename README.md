# CLIP prefix captioning.

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>  
Inference Notebook: <a href="https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=20></a>  





## Official implementation for the paper ["ClipCap: CLIP Prefix for Image Captioning"](https://arxiv.org/abs/2111.09734)




## Description  
Image captioning is a complicated task, where usually a pretrained detection network is used, requires additional supervision in the form of object annotation. We present a new approach that does not requires additional information (i.e. requires only images and captions), thus can be applied to any data. In addition, our model's training time is much faster than similar methods while achieving comparable to state-of-the-art results, even for the Conceptual Captions dataset contains over 3M images. 

In our work, we use the [CLIP](https://github.com/openai/CLIP) model, which was already trained over an extremely large number of images, thus is capable of generating semantic encodings for arbitrary images without additional supervision. To produce meaningful sentences we fine-tune a pretrained language model, which has been proven to be successful for other natural language tasks. The key idea is to use the CLIP encoding as a prefix to the textual captions by employing a simple mapping network over the raw encoding, and then fine-tune our language model to generate a valid caption. In addition, we present another variant, where we utilize a transformer architecture for the mapping network and avoid the fine-tuning of GPT-2. Still, our light model achieve comaparable to state-of-the-art over nocaps dataset.

## COCO Examples

<table>
  <tr>
    <td><img src="Images/COCO_val2014_000000562207.jpg" ></td>
    <td><img src="Images/COCO_val2014_000000165547.jpg" ></td>
    <td><img src="Images/COCO_val2014_000000579664.jpg" ></td>
  </tr>
  <tr>
    <td>A couple of people standing next to an elephant. </td>
     <td>A wooden table sitting in front of a window.</td>
     <td>A bunch of bananas sitting on top of a table.</td>
  </tr>
 </table>
 
 <table>
  <tr>
    <td><img src="Images/COCO_val2014_000000060623.jpg" ></td>
    <td><img src="Images/COCO_val2014_000000386164.jpg" ></td>
    <td><img src="Images/COCO_val2014_000000354533.jpg" ></td>
  </tr>
  <tr>
    <td>A woman holding a plate with a piece of cake in front of her face. </td>
     <td>A wooden table topped with lots of wooden utensils.</td>
     <td>A red motorcycle parked on top of a dirt field.</td>
  </tr>
 </table>


## Conceptual Captions Examples

<table>
  <tr>
    <td><img src="Images/CONCEPTUAL_01.jpg" ></td>
    <td><img src="Images/CONCEPTUAL_02.jpg" ></td>
    <td><img src="Images/CONCEPTUAL_03.jpg" ></td>
  </tr>
  <tr>
    <td>3D render of a man holding a globe.</td>
     <td>Students enjoing the cherry blossoms</td>
     <td>Green leaf of lettuce on a white plate.</td>
  </tr>
 </table>
 
 <table>
  <tr>
    <td><img src="Images/CONCEPTUAL_04.jpg" ></td>
    <td><img src="Images/CONCEPTUAL_05.jpg" ></td>
    <td><img src="Images/CONCEPTUAL_06.jpg" ></td>
  </tr>
  <tr>
    <td>The hotel and casino on the waterfront. </td>
     <td>The triangle is a symbol of the soul.</td>
     <td>Cartoon boy in the bath.</td>
  </tr>
 </table>


## Inference Notebooks
To help visualize the results we provide a Colab notebook found in `notebooks/clip_prefix_captioning_inference.ipynb`.   
The notebook will download the pretrained models and run inference on a sample images or 
on images of your choosing. It is recommended to run this in [Google Colab](https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing).
Inference notebook for the **transformer mapping network (without fine-tune GPT-2)** can be found [here](https://colab.research.google.com/drive/180L3rMFmGujudwO1EJNF-lHIpAsAZ5xq?usp=sharing) for the COCO model (also in `notebooks/transformer_inference.ipynb`).



Both [COCO](https://drive.google.com/file/d/1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX/view?usp=sharing) and [Conceptual Captions](https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view?usp=sharing) pretrained models are available for mlp mapping network. For the transformer (without fine-tuning GPT-2) we provide [COCO](https://drive.google.com/file/d/1GYPToCqFREwi285wPLhuVExlz7DDUDfJ/view?usp=sharing) pretrained model.



## Inference GUI
1. Run it [in the browser](https://replicate.ai/rmokady/clip_prefix_caption) using replicate.ai UI.
2. Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/CLIP_prefix_captioning) (currently not supporting beam search)


## Training prerequisites

[comment]: <> (Dependencies can be found at the [Inference notebook](https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing) )
Clone, create environment and install dependencies:  
```
git clone https://github.com/rmokady/CLIP_prefix_caption && cd CLIP_prefix_caption
conda env create -f environment.yml
conda activate clip_prefix_caption
```

pip install transformers torch numpy pillow scikit-image git+https://github.com/openai/CLIP.git

python parse_flickr30k.py --clip_model_type ViT-B/32 --images_dir ./flickr30k_images/flickr30k_images --captions_file ./flickr30k_images/results.csv

Recommended for train/test split (Karpathy JSON):
python parse_flickr30k.py --clip_model_type ViT-B/32 --images_dir ./data/flickr30k/flickr30k-images --karpathy_json ./data/flickr30k/dataset_flickr30k.json --split train

## Training Modes

You can train the model in 3 different modes by adjusting the `--mapping_type` and `--only_prefix` arguments:

**1. Mode MLP (MLP mapper + GPT-2 not fine-tuned)**
Train only the MLP mapping network while keeping GPT-2 frozen:
```bash
python train.py --data ./data/flickr30k/flickr30k_clip_ViT-B_32_train.pkl --out_dir ./checkpoints/flickr30k_mlp --prefix flickr30k_mlp --mapping_type mlp --only_prefix
```

**2. Mode Transformer + GPT-2 frozen (Transformer mapper + GPT-2 not fine-tuned)**
Train only the Transformer mapping network while keeping GPT-2 frozen:
```bash
python train.py --data ./data/flickr30k/flickr30k_clip_ViT-B_32_train.pkl --out_dir ./checkpoints/flickr30k_transformer_frozen --prefix flickr30k_transformer_frozen --mapping_type transformer --only_prefix
```

**3. Mode Transformer + GPT-2 fine-tuned (Transformer mapper + GPT-2 fine-tuned)**
Train both the Transformer mapping network and fine-tune the GPT-2 language model:
```bash
python train.py --data ./data/flickr30k/flickr30k_clip_ViT-B_32_train.pkl --out_dir ./checkpoints/flickr30k_transformer_finetune --prefix flickr30k_transformer_finetune --mapping_type transformer
```

## COCO training

Download [train_captions](https://drive.google.com/file/d/1D3EzUK1d1lNhD2hAvRiKPThidiVbP2K_/view?usp=sharing) to `data/coco/annotations`.

Download [training images](http://images.cocodataset.org/zips/train2014.zip) and [validation images](http://images.cocodataset.org/zips/val2014.zip) and unzip (We use Karpathy et el. split).

Extract CLIP features using (output is `data/coco/oscar_split_ViT-B_32_train.pkl`):
```
python parse_coco.py --clip_model_type ViT-B/32
```
Train with fine-tuning of GPT2:
```
python train.py --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir ./coco_train/
```

Train only transformer mapping network:
```
python train.py --only_prefix --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir ./coco_train/ --mapping_type transformer  --num_layres 8 --prefix_length 40 --prefix_length_clip 40
```

**If you wish to use ResNet-based CLIP:** 

```
python parse_coco.py --clip_model_type RN50x4
```
```
python train.py --only_prefix --data ./data/coco/oscar_split_RN50x4_train.pkl --out_dir ./coco_train/ --mapping_type transformer  --num_layres 8 --prefix_length 40 --prefix_length_clip 40 --is_rn
```

## Conceptual training

Download the .TSV train/val files from [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/download) and place them under <data_root> directory.

Download the images and extract CLIP features using (outputs are `<data_root>/conceptual_clip_ViT-B_32_train.pkl` and  `<data_root>/conceptual_clip_ViT-B_32_val.pkl`):
```
python parse_conceptual.py --clip_model_type ViT-B/32 --data_root <data_root> --num_threads 16
```
Notice, downloading the images might take a few days.

Train with fine-tuning of GPT2:
```
python train.py --data <data_root>/conceptual_clip_ViT-B_32_train.pkl --out_dir ./conceptual_train/
```
Similarly to the COCO training, you can train a transformer mapping network, and / or parse the images using a ResNet-based CLIP. 

## Flickr30k training

Prepare Flickr30k images and annotations under `data/flickr30k`.

Common layout:
- Images: `data/flickr30k/flickr30k-images/*.jpg`
- Captions file: `data/flickr30k/results_20130124.token`

Extract CLIP features (output is `data/flickr30k/flickr30k_clip_ViT-B_32.pkl`):
```
python parse_flickr30k.py --clip_model_type ViT-B/32 --images_dir ./data/flickr30k/flickr30k-images --captions_file ./data/flickr30k/results_20130124.token
```

If you use Karpathy JSON (`dataset_flickr30k.json`), run:
```
python parse_flickr30k.py --clip_model_type ViT-B/32 --images_dir ./data/flickr30k/flickr30k-images --karpathy_json ./data/flickr30k/dataset_flickr30k.json
```

To create explicit train/val/test files (recommended for fair evaluation), run:
```
python parse_flickr30k.py --clip_model_type ViT-B/32 --images_dir ./data/flickr30k/flickr30k-images --karpathy_json ./data/flickr30k/dataset_flickr30k.json --split train
python parse_flickr30k.py --clip_model_type ViT-B/32 --images_dir ./data/flickr30k/flickr30k-images --karpathy_json ./data/flickr30k/dataset_flickr30k.json --split val
python parse_flickr30k.py --clip_model_type ViT-B/32 --images_dir ./data/flickr30k/flickr30k-images --karpathy_json ./data/flickr30k/dataset_flickr30k.json --split test
```

If you only have `results.csv` (no Karpathy JSON), use deterministic image-level split first:
```
python split_flickr30k_captions.py --captions_file ./flickr30k_images/results.csv --out_dir ./data/flickr30k --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --seed 42
```

Then parse each split to CLIP embeddings:
```
python parse_flickr30k.py --clip_model_type ViT-B/32 --images_dir ./flickr30k_images/flickr30k_images --captions_file ./data/flickr30k/results_train.csv --out_path ./data/flickr30k/flickr30k_clip_ViT-B_32_train.pkl
python parse_flickr30k.py --clip_model_type ViT-B/32 --images_dir ./flickr30k_images/flickr30k_images --captions_file ./data/flickr30k/results_val.csv --out_path ./data/flickr30k/flickr30k_clip_ViT-B_32_val.pkl
python parse_flickr30k.py --clip_model_type ViT-B/32 --images_dir ./flickr30k_images/flickr30k_images --captions_file ./data/flickr30k/results_test.csv --out_path ./data/flickr30k/flickr30k_clip_ViT-B_32_test.pkl
```

These commands generate:
- `data/flickr30k/flickr30k_clip_ViT-B_32_train.pkl`
- `data/flickr30k/flickr30k_clip_ViT-B_32_val.pkl`
- `data/flickr30k/flickr30k_clip_ViT-B_32_test.pkl`

### Cách split và train/test (khuyến nghị)

1. Tạo train/val/test bằng Karpathy split:
```
python parse_flickr30k.py --clip_model_type ViT-B/32 --images_dir ./data/flickr30k/flickr30k-images --karpathy_json ./data/flickr30k/dataset_flickr30k.json --split train
python parse_flickr30k.py --clip_model_type ViT-B/32 --images_dir ./data/flickr30k/flickr30k-images --karpathy_json ./data/flickr30k/dataset_flickr30k.json --split val
python parse_flickr30k.py --clip_model_type ViT-B/32 --images_dir ./data/flickr30k/flickr30k-images --karpathy_json ./data/flickr30k/dataset_flickr30k.json --split test
```

2. Train chỉ với train split:
```
python train.py --data ./data/flickr30k/flickr30k_clip_ViT-B_32_train.pkl --out_dir ./checkpoints/flickr30k_transformer_finetune --prefix flickr30k_transformer_finetune --mapping_type transformer
```

3. Dùng val split để chọn epoch (tùy chọn):
```
python evaluate.py --data ./data/flickr30k/flickr30k_clip_ViT-B_32_val.pkl --checkpoint ./checkpoints/flickr30k_transformer_finetune/flickr30k_transformer_finetune-009.pt --mapping_type transformer --prefix_length 10 --prefix_length_clip 10 --num_layers 8 --decode beam --beam_size 5
```

4. Báo cáo kết quả cuối trên test split:
```
python evaluate.py --data ./data/flickr30k/flickr30k_clip_ViT-B_32_test.pkl --checkpoint ./checkpoints/flickr30k_transformer_finetune/flickr30k_transformer_finetune-009.pt --mapping_type transformer --prefix_length 10 --prefix_length_clip 10 --num_layers 8 --decode beam --beam_size 5 --save_predictions ./checkpoints/flickr30k_transformer_finetune/eval_results.json
```

Train with fine-tuning of GPT2:
```
python train.py --data ./data/flickr30k/flickr30k_clip_ViT-B_32_train.pkl --out_dir ./flickr30k_train/ --prefix flickr30k_prefix
```

Train only transformer mapping network:
```
python train.py --only_prefix --data ./data/flickr30k/flickr30k_clip_ViT-B_32_train.pkl --out_dir ./flickr30k_train/ --prefix flickr30k_prefix --mapping_type transformer --num_layers 8 --prefix_length 40 --prefix_length_clip 40
```

## CNN-RNN Training (Flickr30k)

CNN-RNN is integrated directly into existing scripts (`train.py`, `predict.py`, `evaluate.py`) via `--model_arch cnn_rnn`.

Model:
- Encoder: ResNet-50 (torchvision)
- Decoder: LSTM language model

Install dependencies if needed:
```
pip install torchvision pillow tqdm
```

Train from token file (`results_20130124.token` or split CSV):
```
python train.py --model_arch cnn_rnn --images_dir ./data/flickr30k/flickr30k-images --captions_file ./data/flickr30k/results_20130124.token --out_dir ./checkpoints/cnn_rnn --prefix flickr30k_cnn_rnn --epochs 15 --batch_size 64
```

Train from Karpathy JSON split:
```
python train.py --model_arch cnn_rnn --images_dir ./data/flickr30k/flickr30k-images --karpathy_json ./data/flickr30k/dataset_flickr30k.json --split train --out_dir ./checkpoints/cnn_rnn --prefix flickr30k_cnn_rnn --epochs 15 --batch_size 64
```

Useful options:
- `--unfreeze_cnn`: fine-tune the ResNet backbone.
- `--min_word_freq`: minimum token frequency kept in vocabulary.
- `--max_tokens`: max caption length used for training.

Checkpoints are saved to `--out_dir` and include model weights, optimizer state, and vocabulary.

Predict 1 image with CNN-RNN:
```
python predict.py --model_arch cnn_rnn --image ./Images/COCO_val2014_000000562207.jpg --checkpoint ./checkpoints/cnn_rnn/flickr30k_cnn_rnn-014.pt
```

Evaluate CNN-RNN on Flickr30k annotations:
```
python evaluate.py --model_arch cnn_rnn --images_dir ./data/flickr30k/flickr30k-images --karpathy_json ./data/flickr30k/dataset_flickr30k.json --split test --checkpoint ./checkpoints/cnn_rnn/flickr30k_cnn_rnn-014.pt --save_predictions ./checkpoints/cnn_rnn/eval_results.json
```

ClipCap mode remains unchanged and is the default (`--model_arch clipcap`).

## Evaluation (paper metrics)

Evaluate a trained checkpoint with COCO-style captioning metrics used in the ClipCap paper:
- BLEU-1/2/3/4
- METEOR
- ROUGE-L
- CIDEr
- SPICE

Install evaluation dependencies:
```
pip install pycocoevalcap pycocotools
```

Run evaluation for transformer mapping + GPT-2 fine-tuned:
```
python evaluate.py --data ./data/flickr30k/flickr30k_clip_ViT-B_32_test.pkl --checkpoint ./checkpoints/flickr30k_transformer_finetune/flickr30k_transformer_finetune-009.pt --mapping_type transformer --prefix_length 10 --prefix_length_clip 10 --num_layers 8 --decode beam --beam_size 5 --save_predictions ./checkpoints/flickr30k_transformer_finetune/eval_results.json
```

Run evaluation for MLP mapping (GPT-2 frozen):
```
python evaluate.py --data ./data/flickr30k/flickr30k_clip_ViT-B_32_test.pkl --checkpoint ./checkpoints/flickr30k_mlp/flickr30k_mlp-009.pt --mapping_type mlp --only_prefix --prefix_length 10 --decode beam --beam_size 5 --save_predictions ./checkpoints/flickr30k_mlp/eval_results.json
```

Run evaluation for transformer mapping (GPT-2 frozen):
```
python evaluate.py --data ./data/flickr30k/flickr30k_clip_ViT-B_32_test.pkl --checkpoint ./checkpoints/flickr30k_transformer_frozen/flickr30k_transformer_frozen-009.pt --mapping_type transformer --only_prefix --prefix_length 10 --prefix_length_clip 10 --num_layers 8 --decode beam --beam_size 5 --save_predictions ./checkpoints/flickr30k_transformer_frozen/eval_results.json
```

Notes:
- Add `--normalize_prefix` if your model was trained with `--normalize_prefix`.
- Use `--max_samples N` for quick checks on a subset before full evaluation.
- You can switch decoding with `--decode nucleus --top_p 0.8`.

## Citation
If you use this code for your research, please cite:
```
@article{mokady2021clipcap,
  title={ClipCap: CLIP Prefix for Image Captioning},
  author={Mokady, Ron and Hertz, Amir and Bermano, Amit H},
  journal={arXiv preprint arXiv:2111.09734},
  year={2021}
}
```




## Acknowledgments
This repository is heavily based on [CLIP](https://github.com/openai/CLIP) and [Hugging-faces](https://github.com/huggingface/transformers) repositories.
For training we used the data of [COCO dataset](https://cocodataset.org/#home) and [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/).

## Contact
For any inquiry please contact us at our email addresses: ron.mokady@gmail.com or amirhertz@mail.tau.ac.il.


