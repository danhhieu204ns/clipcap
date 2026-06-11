import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
import random
from typing import Dict, Tuple, Optional, Union

CNN_RNN_IMPORT_ERROR = None
try:
    from torchvision import transforms
    from train_cnn_rnn import (
        CaptionCollator,
        CNNRNNCaptioner,
        FlickrCaptionDataset,
        build_vocab,
        load_samples,
        save_checkpoint as save_cnn_rnn_checkpoint,
        set_seed,
        train_one_epoch,
    )
except ImportError as exc:
    transforms = None
    CaptionCollator = None
    CNNRNNCaptioner = None
    FlickrCaptionDataset = None
    build_vocab = None
    load_samples = None
    save_cnn_rnn_checkpoint = None
    set_seed = None
    train_one_epoch = None
    CNN_RNN_IMPORT_ERROR = exc


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


def parse_mapping_type(mapping_type: Union[str, MappingType]) -> MappingType:
    if isinstance(mapping_type, MappingType):
        return mapping_type
    return {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[mapping_type]


def load_torch_state_dict(path: str, map_location: Union[str, torch.device] = 'cpu') -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def project_prefix(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.project_prefix(prefix)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP,
                 decoder_model: str = 'gpt2', mlp_hidden_scale: float = 0.5,
                 mlp_hidden_dim: Optional[int] = None):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.decoder_model = decoder_model
        self.gpt = GPT2LMHeadModel.from_pretrained(decoder_model)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        mapping_type = parse_mapping_type(mapping_type)
        if mapping_type == MappingType.MLP:
            output_dim = self.gpt_embedding_size * prefix_length
            hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else int(output_dim * mlp_hidden_scale)
            hidden_dim = max(1, hidden_dim)
            self.clip_project = MLP((prefix_size, hidden_dim, output_dim))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    mapping_type = parse_mapping_type(getattr(args, 'mapping_type', 'mlp'))
    decoder_model = getattr(args, 'decoder_model', 'gpt2')
    mlp_hidden_scale = getattr(args, 'mlp_hidden_scale', 0.5)
    mlp_hidden_dim = getattr(args, 'mlp_hidden_dim', None)
    prefix_dim = 640 if getattr(args, 'is_rn', False) else 512
    model_kwargs = dict(
        clip_length=getattr(args, 'prefix_length_clip', 10),
        prefix_size=prefix_dim,
        num_layers=getattr(args, 'num_layers', 8),
        mapping_type=mapping_type,
        decoder_model=decoder_model,
        mlp_hidden_scale=mlp_hidden_scale,
        mlp_hidden_dim=mlp_hidden_dim,
    )
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length, **model_kwargs)
    else:
        model = ClipCaptionModel(args.prefix_length, **model_kwargs)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(load_torch_state_dict(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser


def build_teacher_model(args, device: torch.device) -> Optional[ClipCaptionModel]:
    teacher_checkpoint = getattr(args, 'distill_teacher_checkpoint', '')
    logit_weight = getattr(args, 'distill_logit_weight', 0.0)
    prefix_weight = getattr(args, 'distill_prefix_weight', 0.0)
    if logit_weight <= 0 and prefix_weight <= 0:
        return None
    if not teacher_checkpoint:
        raise ValueError('--distill_teacher_checkpoint is required when a KD weight is > 0')
    if not os.path.isfile(teacher_checkpoint):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint}")

    teacher_prefix_length = getattr(args, 'distill_teacher_prefix_length', 0) or args.prefix_length
    teacher_clip_length = getattr(args, 'distill_teacher_prefix_length_clip', 0) or args.prefix_length_clip
    prefix_dim = 640 if getattr(args, 'distill_teacher_is_rn', False) else 512
    teacher_mapping_type = parse_mapping_type(getattr(args, 'distill_teacher_mapping_type', 'transformer'))
    teacher_kwargs = dict(
        clip_length=teacher_clip_length,
        prefix_size=prefix_dim,
        num_layers=getattr(args, 'distill_teacher_num_layers', 8),
        mapping_type=teacher_mapping_type,
        decoder_model=getattr(args, 'distill_teacher_decoder_model', 'gpt2'),
        mlp_hidden_scale=getattr(args, 'distill_teacher_mlp_hidden_scale', 0.5),
        mlp_hidden_dim=getattr(args, 'distill_teacher_mlp_hidden_dim', None),
    )
    if getattr(args, 'distill_teacher_only_prefix', False):
        teacher = ClipCaptionPrefix(teacher_prefix_length, **teacher_kwargs)
    else:
        teacher = ClipCaptionModel(teacher_prefix_length, **teacher_kwargs)

    teacher.load_state_dict(load_torch_state_dict(teacher_checkpoint, map_location=device))
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"Loaded teacher checkpoint: {teacher_checkpoint}")
    return teacher


def build_teacher_mask(mask: torch.Tensor, tokens: torch.Tensor, student_prefix_length: int,
                       teacher_prefix_length: int) -> torch.Tensor:
    if student_prefix_length == teacher_prefix_length:
        return mask
    token_mask = mask[:, student_prefix_length:]
    teacher_prefix_mask = torch.ones(tokens.shape[0], teacher_prefix_length, device=tokens.device)
    return torch.cat((teacher_prefix_mask, token_mask), dim=1)


def masked_logit_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                         tokens: torch.Tensor, temperature: float) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Logit KD shape mismatch: student={tuple(student_logits.shape)}, "
            f"teacher={tuple(teacher_logits.shape)}"
        )
    token_mask = tokens.ne(0).float()
    denom = token_mask.sum().clamp_min(1.0)
    student_log_probs = nnf.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = nnf.softmax(teacher_logits / temperature, dim=-1)
    kd_per_token = nnf.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)
    return (kd_per_token * token_mask).sum() / denom * (temperature ** 2)


def prefix_kd_loss(student_prefix: torch.Tensor, teacher_prefix: torch.Tensor, loss_type: str) -> torch.Tensor:
    if student_prefix.shape != teacher_prefix.shape:
        raise ValueError(
            f"Prefix KD shape mismatch: student={tuple(student_prefix.shape)}, "
            f"teacher={tuple(teacher_prefix.shape)}"
        )
    if loss_type == 'mse':
        return nnf.mse_loss(student_prefix, teacher_prefix)
    if loss_type == 'cosine':
        return 1 - nnf.cosine_similarity(student_prefix, teacher_prefix, dim=-1).mean()
    if loss_type == 'mse_cosine':
        mse = nnf.mse_loss(student_prefix, teacher_prefix)
        cosine = 1 - nnf.cosine_similarity(student_prefix, teacher_prefix, dim=-1).mean()
        return mse + cosine
    raise ValueError(f"Unsupported prefix KD loss: {loss_type}")


def train(dataset: ClipCocoDataset, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):

    device = torch.device(args.device)
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    teacher = build_teacher_model(args, device)
    distill_logit_weight = getattr(args, 'distill_logit_weight', 0.0)
    distill_prefix_weight = getattr(args, 'distill_prefix_weight', 0.0)
    distill_temperature = getattr(args, 'distill_temperature', 2.0)
    distill_prefix_loss_type = getattr(args, 'distill_prefix_loss', 'mse')
    use_logit_kd = teacher is not None and distill_logit_weight > 0
    use_prefix_kd = teacher is not None and distill_prefix_weight > 0
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    save_config(args)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1]
            ce_loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss = ce_loss
            postfix = {'loss': loss.item(), 'ce': ce_loss.item()}
            if teacher is not None:
                teacher_mask = build_teacher_mask(mask, tokens, model.prefix_length, teacher.prefix_length)
                with torch.no_grad():
                    teacher_outputs = teacher(tokens, prefix, teacher_mask)
                    teacher_logits = teacher_outputs.logits[:, teacher.prefix_length - 1: -1]
                    teacher_prefix = teacher.project_prefix(prefix) if use_prefix_kd else None
                if use_logit_kd:
                    logit_kd = masked_logit_kd_loss(logits, teacher_logits, tokens, distill_temperature)
                    loss = loss + distill_logit_weight * logit_kd
                    postfix['logit_kd'] = logit_kd.item()
                if use_prefix_kd:
                    student_prefix = model.project_prefix(prefix)
                    prefix_kd = prefix_kd_loss(student_prefix, teacher_prefix, distill_prefix_loss_type)
                    loss = loss + distill_prefix_weight * prefix_kd
                    postfix['prefix_kd'] = prefix_kd.item()
                postfix['loss'] = loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix(postfix)
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"{output_prefix}_latest.pt"),
        )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_arch', type=str, default='clipcap', choices=['clipcap', 'cnn_rnn'])
    parser.add_argument('--data', default='./data/coco/oscar_split_train.pkl')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=40)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--decoder_model', type=str, default='gpt2', help='HuggingFace GPT-2 style LM, e.g. gpt2 or distilgpt2')
    parser.add_argument('--mlp_hidden_scale', type=float, default=0.5, help='MLP mapper hidden dim as a fraction of prefix output dim')
    parser.add_argument('--mlp_hidden_dim', type=int, default=None, help='Optional absolute hidden dim for MLP mapper')
    parser.add_argument('--init_checkpoint', default='', help='Optional checkpoint used to initialize ClipCap weights before training')
    parser.add_argument('--clipcap_lr', type=float, default=2e-5)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--distill_teacher_checkpoint', default='')
    parser.add_argument('--distill_teacher_mapping_type', type=str, default='transformer', choices=['mlp', 'transformer'])
    parser.add_argument('--distill_teacher_decoder_model', type=str, default='gpt2')
    parser.add_argument('--distill_teacher_prefix_length', type=int, default=0)
    parser.add_argument('--distill_teacher_prefix_length_clip', type=int, default=0)
    parser.add_argument('--distill_teacher_num_layers', type=int, default=8)
    parser.add_argument('--distill_teacher_only_prefix', action='store_true')
    parser.add_argument('--distill_teacher_is_rn', action='store_true')
    parser.add_argument('--distill_teacher_mlp_hidden_scale', type=float, default=0.5)
    parser.add_argument('--distill_teacher_mlp_hidden_dim', type=int, default=None)
    parser.add_argument('--distill_logit_weight', type=float, default=0.0)
    parser.add_argument('--distill_prefix_weight', type=float, default=0.0)
    parser.add_argument('--distill_temperature', type=float, default=2.0)
    parser.add_argument('--distill_prefix_loss', type=str, default='mse', choices=['mse', 'cosine', 'mse_cosine'])
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')

    parser.add_argument('--images_dir', default='')
    parser.add_argument('--captions_file', default='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_tokens', type=int, default=40)
    parser.add_argument('--min_word_freq', type=int, default=5)
    parser.add_argument('--unfreeze_cnn', action='store_true', help='Train ResNet backbone weights (cnn_rnn mode)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.model_arch == 'clipcap':
        set_global_seed(args.seed)
        prefix_length = args.prefix_length
        dataset = ClipCocoDataset(args.data, prefix_length, gpt2_type=args.decoder_model,
                                  normalize_prefix=args.normalize_prefix)
        prefix_dim = 640 if args.is_rn else 512
        mapping_type = parse_mapping_type(args.mapping_type)
        model_kwargs = dict(
            clip_length=args.prefix_length_clip,
            prefix_size=prefix_dim,
            num_layers=args.num_layers,
            mapping_type=mapping_type,
            decoder_model=args.decoder_model,
            mlp_hidden_scale=args.mlp_hidden_scale,
            mlp_hidden_dim=args.mlp_hidden_dim,
        )
        if args.only_prefix:
            model = ClipCaptionPrefix(prefix_length, **model_kwargs)
            print("Train only prefix")
        else:
            model = ClipCaptionModel(prefix_length, **model_kwargs)
            print("Train both prefix and GPT")
            sys.stdout.flush()
        if args.init_checkpoint:
            if not os.path.isfile(args.init_checkpoint):
                raise FileNotFoundError(f"init_checkpoint not found: {args.init_checkpoint}")
            model.load_state_dict(load_torch_state_dict(args.init_checkpoint, map_location=torch.device('cpu')))
            print(f"Initialized ClipCap model from: {args.init_checkpoint}")
        train(dataset, model, args, lr=args.clipcap_lr, warmup_steps=args.warmup_steps,
              output_dir=args.out_dir, output_prefix=args.prefix)
    else:
        if CNN_RNN_IMPORT_ERROR is not None:
            raise ImportError(
                'cnn_rnn mode requires torchvision/PIL dependencies. Install with: pip install torchvision pillow tqdm'
            ) from CNN_RNN_IMPORT_ERROR

        if not args.images_dir:
            raise ValueError('--images_dir is required for cnn_rnn mode')

        set_seed(args.seed)
        device = torch.device(args.device)

        samples = load_samples(args)
        print(f"Loaded {len(samples)} image-caption pairs")

        captions_for_vocab = [caption for _, caption in samples]
        vocab = build_vocab(captions_for_vocab, min_freq=args.min_word_freq)
        print(f"Vocab size: {len(vocab.itos)}")

        image_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        dataset = FlickrCaptionDataset(
            samples=samples,
            images_dir=args.images_dir,
            vocab=vocab,
            image_transform=image_transform,
            max_tokens=args.max_tokens,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=CaptionCollator(pad_idx=vocab.pad_idx),
        )

        model = CNNRNNCaptioner(
            vocab_size=len(vocab.itos),
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            num_layers=args.rnn_layers,
            freeze_backbone=not args.unfreeze_cnn,
            dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        os.makedirs(args.out_dir, exist_ok=True)
        config_path = os.path.join(args.out_dir, f"{args.prefix}_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, indent=2)

        for epoch in range(args.epochs):
            avg_loss = train_one_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch_idx=epoch,
                total_epochs=args.epochs,
            )
            print(f"Epoch {epoch + 1}/{args.epochs} - avg_loss: {avg_loss:.4f}")

            if epoch % args.save_every == 0 or epoch == args.epochs - 1:
                ckpt_path = os.path.join(args.out_dir, f"{args.prefix}-{epoch:03d}.pt")
                save_cnn_rnn_checkpoint(
                    path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    avg_loss=avg_loss,
                    vocab=vocab,
                    args=args,
                )
                print(f"Saved checkpoint: {ckpt_path}")


if __name__ == '__main__':
    main()
