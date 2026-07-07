"""V16: V15 with BOUNDED depth correction.

Key fix for V15_L (DINOv2-L) failure:
- V12L's log_range is already very accurate (train loss ~0.012)
- V15's unbounded additive correction shifted range by 20%+ easily
  due to exponential scale in log-normalized space
  (0.02 in log-norm -> 0.2 in log-range -> exp(0.2)=1.22 multiplier)
- V16 clamps correction via tanh * max_correction_scale
- Also: lower lr_depth (5e-4), more freeze_main_epochs (2), higher patience (4)

Design: correction = tanh(head_out) * max_scale
  - max_scale=0.02 means max log-norm shift is +/- 0.02
  - This translates to +/- 22% max range multiplier
  - Safe for both ViT-B (benefits from larger corrections)
    and DINOv2-L (which needs small adjustments)

Philosophy unchanged from V15: CHANGE ONLY ONE THING - add zero-init depth head.
Loss remains pure V10: angle + rw * log_range_loss.

Removed from V13a/V14a (all shown to hurt):
- Stage 2 / relative_range_loss
- competition / bucket_mild weight schemes
- warmup / LR changes beyond warm-start-friendly values

Keep from V14a:
- Gated-residual design (main regressor unchanged, depth adds scalar to log_range)
- Zero-init depth_head last layer -> V15 init output IDENTICAL to V10
- Lower lr for depth components

Loss: angle_loss + range_loss_weight * log_range_loss  (exactly V10 recipe)
Weights: none  (uniform, exactly V10)

Phase 1 (freeze_main_epochs): Only depth head trains, main frozen.
  - Main branch cannot be hurt.
  - Depth head learns to contribute something useful on top of fixed main.
Phase 2: Unfreeze everything else (except backbone), train jointly with low LR.
"""

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from tqdm import tqdm

AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

LOG_RANGE_MAX = math.log1p(132.0)
LOG_RANGE_MIN = -LOG_RANGE_MAX

def range_to_log_normalized(r):
    log_r = torch.log1p(r.abs()) * r.sign()
    return (log_r - LOG_RANGE_MIN) / (LOG_RANGE_MAX - LOG_RANGE_MIN)

def log_normalized_to_range(norm_log_r):
    log_r = norm_log_r * (LOG_RANGE_MAX - LOG_RANGE_MIN) + LOG_RANGE_MIN
    return (torch.exp(log_r.abs()) - 1.0) * log_r.sign()

def log_range_loss(pred_log_norm, target_log_norm, beta=0.05):
    return F.smooth_l1_loss(pred_log_norm, target_log_norm, beta=beta, reduction='mean')

def safe_normalize(x, dim=-1, eps=1e-6):
    norm = x.norm(dim=dim, keepdim=True).clamp(min=eps)
    return x / norm

def angle_loss_cos_sin(pred_xy, target_vec, beta=0.1):
    pred_xy = safe_normalize(pred_xy, dim=-1)
    target_vec = safe_normalize(target_vec, dim=-1)
    return F.smooth_l1_loss(pred_xy, target_vec, beta=beta, reduction='mean')


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


parser = argparse.ArgumentParser(description='V15: Clean Depth Integration')
parser.add_argument('--batch_size', default=96, type=int)
parser.add_argument('--grad_accum_steps', default=2, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--lr_backbone', default=5e-8, type=float)
parser.add_argument('--lr_cross_attn', default=5e-5, type=float)
parser.add_argument('--lr_head', default=1e-3, type=float)
parser.add_argument('--lr_depth', default=5e-4, type=float,
                    help='Lower than V15 (2e-3) since tanh saturates with larger updates')
parser.add_argument('--wd', default=1e-4, type=float, dest='weight_decay')
parser.add_argument('--epochs', default=8, type=int)
parser.add_argument('--warmup_epochs', default=0, type=int)
parser.add_argument('--patience', default=4, type=int,
                    help='Higher patience for large model (oscillates more)')
parser.add_argument('--n_cross_layers', default=2, type=int)
parser.add_argument('--cross_heads', default=8, type=int)
parser.add_argument('--cross_dropout', default=0.1, type=float)
parser.add_argument('--freeze_main_epochs', default=2, type=int,
                    help='Freeze main for N epochs, only depth trains')
parser.add_argument('--dinov2_model', default='dinov2_vitb14', type=str,
                    choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'])
parser.add_argument('--train_image_dir', default='../pairUAV/train_tour', type=str)
parser.add_argument('--train_json_dir', default='../pairUAV/train', type=str)
parser.add_argument('--train_match_dir', default='./train_matches_data', type=str)
parser.add_argument('--test_image_dir', default='../pairUAV/test_tour', type=str)
parser.add_argument('--test_json_dir', default='../pairUAV/test', type=str)
parser.add_argument('--test_match_dir', default='./test_matches_data', type=str)
parser.add_argument('--test_output_txt', default='./test_predict_output_v16.txt', type=str)
parser.add_argument('--exp_name', default='v16_bounded', type=str)
parser.add_argument('--max_correction_scale', default=0.02, type=float,
                    help='Max |correction| in log-normalized space. 0.02 -> ~22% max range change')
parser.add_argument('--range_loss_weight', default=8.0, type=float)
parser.add_argument('--test_every', default=1, type=int, dest='eval_test_every',
                    help='Run test-set inference every N epochs (no metric'
                         ' computed; predictions saved as a checkpoint).')
parser.add_argument('--skip_final_test', action='store_true')
parser.add_argument('--depth_size', default=64, type=int)
parser.add_argument('--depth_compact_train', default='./depth_compact_train', type=str)
parser.add_argument('--depth_compact_test', default='./depth_compact_test', type=str)
parser.add_argument('--use_spatial', action='store_true')
parser.add_argument('--pretrained_ckpt', required=True, type=str)

import torchvision.transforms as T


class DualImageDatasetV15(Dataset):
    DINOV2_TRANSFORM = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, image_dir, json_dir, match_dir, depth_compact_path,
                 has_gt=True, force_image_ext=None, augment=False,
                 depth_size=64, use_spatial=False):
        super().__init__()
        self.image_dir = image_dir
        self.match_dir = match_dir
        self.has_gt = has_gt
        self.force_image_ext = force_image_ext
        self.augment = augment
        self.depth_size = depth_size
        self.use_spatial = use_spatial
        self.sx, self.sy = 224.0 / 640.0, 224.0 / 480.0
        self.json_paths = []
        for name in os.listdir(json_dir):
            sub_path = os.path.join(json_dir, name)
            if os.path.isdir(sub_path):
                for f in os.listdir(sub_path):
                    if f.endswith('.json'):
                        self.json_paths.append(os.path.join(sub_path, f))
            elif name.endswith('.json'):
                self.json_paths.append(sub_path)
        self.json_paths.sort(key=self._json_path_sort_key)

        self.depth_key2idx = None
        self.depth_maps_arr = None
        self.depth_stats_arr = None
        keys_path = f'{depth_compact_path}_keys.json'
        maps_path = f'{depth_compact_path}_maps.npy'
        stats_path = f'{depth_compact_path}_stats.npy'
        if not os.path.exists(keys_path):
            raise FileNotFoundError(f'Depth compact data not found: {depth_compact_path}_*')
        print(f'Loading compact depth: {depth_compact_path}_*', flush=True)
        with open(keys_path, 'r') as f:
            keys = json.load(f)
        self.depth_key2idx = {k: i for i, k in enumerate(keys)}
        self.depth_stats_arr = np.load(stats_path, mmap_mode='r')
        if use_spatial:
            self.depth_maps_arr = np.load(maps_path, mmap_mode='r')
        print(f'  {len(keys)} depth entries (spatial={use_spatial})', flush=True)

    @staticmethod
    def _extract_int(value):
        m = re.search(r'\d+', str(value))
        return int(m.group()) if m else float('inf')

    @classmethod
    def _json_path_sort_key(cls, json_path):
        p = Path(json_path)
        return (cls._extract_int(p.parent.name), p.parent.name, cls._extract_int(p.stem), p.stem)

    def _resolve_image_path(self, image_rel_path):
        image_rel_path = str(image_rel_path)
        image_name = Path(image_rel_path).name
        image_stem = Path(image_name).stem
        for c in [os.path.join(self.image_dir, image_rel_path),
                  os.path.join(self.image_dir, image_name)] + \
                 ([os.path.join(self.image_dir, image_stem + self.force_image_ext)] if self.force_image_ext else []):
            if os.path.exists(c):
                return c
        raise FileNotFoundError(f"Image not found for {image_rel_path}")

    def _load_depth(self, image_path):
        p = Path(image_path)
        stem = p.stem
        parent_name = p.parent.name
        for key_stem in [f'{parent_name}_{stem}', stem]:
            if key_stem in self.depth_key2idx:
                idx = self.depth_key2idx[key_stem]
                stats = self.depth_stats_arr[idx].copy()
                if self.use_spatial:
                    spatial = self.depth_maps_arr[idx].astype(np.float32)
                else:
                    spatial = np.zeros((self.depth_size, self.depth_size), dtype=np.float32)
                return spatial, stats
        return np.zeros((self.depth_size, self.depth_size), dtype=np.float32), \
               np.zeros(8, dtype=np.float32)

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, i):
        json_path = self.json_paths[i]
        p = Path(json_path)
        json_id = p.parent.name
        with open(json_path, "rb") as f:
            data = json.load(f)
        a_path = self._resolve_image_path(data["image_a"])
        b_path = self._resolve_image_path(data["image_b"])
        a_name = Path(a_path).stem
        b_name = Path(b_path).stem

        if self.has_gt:
            npz_candidates = [os.path.join(self.match_dir, json_id, f"{a_name}_{b_name}_matches.npz")]
        else:
            npz_candidates = [
                os.path.join(self.match_dir, a_name, f"{b_name}.npz"),
                os.path.join(self.match_dir, a_name, f"{b_name}_matches.npz"),
                os.path.join(self.match_dir, a_name, f"{a_name}_{b_name}_matches.npz"),
            ]
        npz_path = None
        for c in npz_candidates:
            if os.path.exists(c):
                npz_path = c
                break
        if npz_path is None:
            raise FileNotFoundError(f"Match npz not found: {npz_candidates}")

        with np.load(npz_path, allow_pickle=False) as z:
            k0, k1, m = z["keypoints0"], z["keypoints1"], z["matches"]
        valid = (m >= 0) & (m < len(k1))
        src, dst = k0[valid], k1[m[valid]]
        x_src, y_src = src[:, 0] * self.sx, src[:, 1] * self.sy
        x_dst, y_dst = dst[:, 0] * self.sx, dst[:, 1] * self.sy
        xi = np.clip(x_src, 0, 223.9999).astype(np.int32)
        yi = np.clip(y_src, 0, 223.9999).astype(np.int32)
        match_field = np.zeros((2, 224, 224), dtype=np.float32)
        match_field[0, yi, xi] = (x_dst - x_src).astype(np.float32)
        match_field[1, yi, xi] = (y_dst - y_src).astype(np.float32)
        disp_tensor = torch.from_numpy(match_field)

        num_matches = int(valid.sum())
        if num_matches >= 2:
            dx_raw = (dst[:, 0] - src[:, 0]) / 640.0
            dy_raw = (dst[:, 1] - src[:, 1]) / 480.0
            mag = np.sqrt(dx_raw**2 + dy_raw**2)
            match_stats = np.array([num_matches/1000.0, mag.mean(), dx_raw.mean(), dy_raw.mean(), dx_raw.std(), dy_raw.std()], dtype=np.float32)
        else:
            match_stats = np.zeros(6, dtype=np.float32)
        match_stats_tensor = torch.from_numpy(match_stats)

        depth_a, stats_a = self._load_depth(a_path)
        depth_b, stats_b = self._load_depth(b_path)
        depth_pair = np.stack([depth_a, depth_b], axis=0)
        depth_stats_arr = np.concatenate([stats_a, stats_b])
        depth_pair_tensor = torch.from_numpy(depth_pair)
        depth_stats_tensor = torch.from_numpy(depth_stats_arr)

        do_flip = self.augment and self.has_gt and (torch.rand(1).item() > 0.5)
        with Image.open(a_path) as im:
            im_a = im.convert("RGB")
            if do_flip: im_a = im_a.transpose(Image.FLIP_LEFT_RIGHT)
            img_a_t = self.DINOV2_TRANSFORM(im_a)
        with Image.open(b_path) as im:
            im_b = im.convert("RGB")
            if do_flip: im_b = im_b.transpose(Image.FLIP_LEFT_RIGHT)
            img_b_t = self.DINOV2_TRANSFORM(im_b)
        if do_flip:
            disp_tensor = torch.flip(disp_tensor, dims=[2])
            disp_tensor[0] = -disp_tensor[0]
            match_stats_tensor[2] = -match_stats_tensor[2]
            if self.use_spatial:
                depth_pair_tensor = torch.flip(depth_pair_tensor, dims=[2])

        if not self.has_gt:
            return img_a_t, img_b_t, disp_tensor, match_stats_tensor, depth_pair_tensor, depth_stats_tensor, json_path

        theta_deg = float(data["heading_num"])
        range_num = float(data['range_num'])
        if do_flip: theta_deg = -theta_deg
        theta_rad = math.radians(theta_deg)
        label_vec = torch.tensor([math.cos(theta_rad), math.sin(theta_rad)], dtype=torch.float32)
        label_deg = torch.tensor(theta_deg, dtype=torch.float32)
        range_tensor = torch.tensor(range_num, dtype=torch.float32)
        log_norm_range = range_to_log_normalized(range_tensor)
        return img_a_t, img_b_t, disp_tensor, match_stats_tensor, depth_pair_tensor, depth_stats_tensor, label_vec, label_deg, json_path, range_num, log_norm_range


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, d_model=768, nhead=8, dropout=0.1):
        super().__init__()
        self.cross_attn_a2b = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_a2b = nn.LayerNorm(d_model)
        self.ffn_a = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model), nn.Dropout(dropout))
        self.norm_ffn_a = nn.LayerNorm(d_model)
        self.cross_attn_b2a = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_b2a = nn.LayerNorm(d_model)
        self.ffn_b = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model), nn.Dropout(dropout))
        self.norm_ffn_b = nn.LayerNorm(d_model)

    def forward(self, feat_a, feat_b):
        a_cross, _ = self.cross_attn_a2b(feat_a, feat_b, feat_b)
        a_cross = torch.clamp(a_cross, -100, 100)
        feat_a = self.norm_a2b(feat_a + a_cross)
        feat_a = self.norm_ffn_a(feat_a + self.ffn_a(feat_a))
        b_cross, _ = self.cross_attn_b2a(feat_b, feat_a, feat_a)
        b_cross = torch.clamp(b_cross, -100, 100)
        feat_b = self.norm_b2a(feat_b + b_cross)
        feat_b = self.norm_ffn_b(feat_b + self.ffn_b(feat_b))
        return feat_a, feat_b


class DepthSpatialEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class DepthHead(nn.Module):
    """Bounded correction: output in [-max_scale, +max_scale] via tanh."""
    def __init__(self, stats_dim=16, spatial_dim=0, hidden=64, max_scale=0.02):
        super().__init__()
        self.use_spatial = spatial_dim > 0
        self.max_scale = max_scale
        in_dim = stats_dim + spatial_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, stats, spatial_feat=None):
        if self.use_spatial and spatial_feat is not None:
            x = torch.cat([stats, spatial_feat], dim=-1)
        else:
            x = stats
        # tanh bounds output to [-max_scale, +max_scale]
        return torch.tanh(self.net(x).squeeze(-1)) * self.max_scale


class CrossAttentionPoseNetV15(nn.Module):
    DINOV2_DIMS = {'dinov2_vits14': 384, 'dinov2_vitb14': 768, 'dinov2_vitl14': 1024}

    def __init__(self, dinov2_model='dinov2_vitb14', n_cross_layers=2, cross_heads=8,
                 cross_dropout=0.1, use_spatial=False, max_correction_scale=0.02):
        super().__init__()
        self.feat_dim = self.DINOV2_DIMS[dinov2_model]
        self.use_spatial = use_spatial
        self.max_correction_scale = max_correction_scale

        print(f'Loading {dinov2_model}...', flush=True)
        local_hub = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
        if os.path.isdir(local_hub):
            self.backbone = torch.hub.load(local_hub, dinov2_model, pretrained=True, source='local')
        else:
            self.backbone = torch.hub.load('facebookresearch/dinov2', dinov2_model, pretrained=True)
        for p in self.backbone.parameters(): p.requires_grad = False
        print(f'  {dinov2_model} loaded, dim={self.feat_dim}, frozen', flush=True)

        self.cross_attn_layers = nn.ModuleList([
            BidirectionalCrossAttention(self.feat_dim, cross_heads, cross_dropout)
            for _ in range(n_cross_layers)
        ])
        self.disp_encoder = nn.Sequential(
            nn.Conv2d(2, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.disp_fc = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        self.match_encoder = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        self.interaction_proj = nn.Sequential(nn.Linear(self.feat_dim * 2, self.feat_dim), nn.ReLU())

        regressor_input_dim = self.feat_dim * 3 + 192
        self.regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 3))

        spatial_dim = 64 if use_spatial else 0
        if use_spatial:
            self.depth_spatial = DepthSpatialEncoder(out_dim=spatial_dim)
        self.depth_head = DepthHead(stats_dim=16, spatial_dim=spatial_dim,
                                     hidden=64, max_scale=max_correction_scale)

    def freeze_main_branch(self):
        for name, p in self.named_parameters():
            if 'depth_head' in name or 'depth_spatial' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        print('  -> Main branch FROZEN, only depth trainable', flush=True)

    def unfreeze_main_branch(self):
        for name, p in self.named_parameters():
            if 'backbone' not in name:
                p.requires_grad = True
        print('  -> Main branch UNFROZEN (backbone still frozen)', flush=True)

    def forward(self, img_a, img_b, disp, match_stats, depth_pair, depth_stats):
        fa = self.backbone.forward_features(img_a)
        fb = self.backbone.forward_features(img_b)
        ta = torch.cat([fa['x_norm_clstoken'].unsqueeze(1), fa['x_norm_patchtokens']], dim=1)
        tb = torch.cat([fb['x_norm_clstoken'].unsqueeze(1), fb['x_norm_patchtokens']], dim=1)
        for layer in self.cross_attn_layers:
            ta, tb = layer(ta, tb)
        cls_a, cls_b = ta[:, 0], tb[:, 0]
        interaction = self.interaction_proj(torch.cat([cls_a - cls_b, cls_a * cls_b], dim=1))
        disp_feat = self.disp_fc(self.disp_encoder(disp).flatten(1))
        match_feat = self.match_encoder(match_stats)
        combined = torch.cat([cls_a, cls_b, interaction, disp_feat, match_feat], dim=1)

        main_out = self.regressor(combined)

        spatial_feat = None
        if self.use_spatial:
            spatial_feat = self.depth_spatial(depth_pair)
        correction = self.depth_head(depth_stats, spatial_feat)

        log_range_corrected = main_out[:, 2] + correction
        out = torch.stack([main_out[:, 0], main_out[:, 1], log_range_corrected], dim=-1)
        return torch.clamp(out, -50, 50)


def warm_start_from_v10(model, ckpt_path):
    print(f'Warm-starting from {ckpt_path}...', flush=True)
    ckpt = torch.load(ckpt_path, weights_only=True, map_location='cpu')
    model_dict = model.state_dict()
    loaded, skipped = [], []
    for k, v in ckpt.items():
        clean_k = k.replace('module.', '')
        if clean_k in model_dict and model_dict[clean_k].shape == v.shape:
            model_dict[clean_k] = v
            loaded.append(clean_k)
        else:
            skipped.append(k)
    model.load_state_dict(model_dict)
    print(f'  Loaded {len(loaded)}/{len(ckpt)} params', flush=True)
    last_w = model.depth_head.net[-1].weight.abs().sum().item()
    last_b = model.depth_head.net[-1].bias.abs().sum().item()
    print(f'  Depth head zero-check: |W|={last_w:.6f}, |b|={last_b:.6f}', flush=True)


def run_test_inference(loader, model, device, output_path=None):
    """Run inference on the test loader and save predictions to a text file.
    No metric is computed - this is a label-free test-set checkpointing."""
    model.train(mode=False)
    results = []
    with torch.no_grad():
        for img_a, img_b, disp, ms, dp, ds, jpaths in loader:
            img_a = img_a.to(device, non_blocking=True)
            img_b = img_b.to(device, non_blocking=True)
            disp = disp.to(device, non_blocking=True)
            ms = ms.to(device, non_blocking=True)
            dp = dp.to(device, non_blocking=True)
            ds = ds.to(device, non_blocking=True)
            with autocast(device_type='cuda', dtype=AMP_DTYPE):
                out = model(img_a, img_b, disp, ms, dp, ds)
            pxy = safe_normalize(out[:, :2].float(), dim=-1)
            pd = torch.rad2deg(torch.atan2(pxy[:, 1], pxy[:, 0]))
            pr = log_normalized_to_range(out[:, 2].float()).clamp(-132, 132)
            for jp, d, r in zip(jpaths, pd.cpu().tolist(), pr.cpu().tolist()):
                results.append((jp, float(d), float(r)))
    results.sort(key=lambda x: extract_group_json_sort_key(x[0]))
    if output_path:
        with open(output_path, 'w') as f:
            for _, h, r in results:
                f.write(f'{h:.6f} {r:.6f}\n')
    return results


def extract_group_json_sort_key(json_path):
    p = Path(json_path)
    gv, jv = p.parent.name, p.stem
    try:
        with open(json_path, 'r') as f:
            d = json.load(f)
        gv = d.get('group_id', gv)
        jv = d.get('json_id', jv)
    except Exception:
        pass
    return (DualImageDatasetV15._extract_int(gv), str(gv), DualImageDatasetV15._extract_int(jv), str(jv))


def run_test_and_save_txt(loader, model, device, out_path):
    print('Final inference...', flush=True)
    model.eval()
    results = []
    with torch.no_grad():
        for img_a, img_b, disp, ms, dp, ds, jpaths in tqdm(loader, desc='Inference'):
            img_a = img_a.to(device, non_blocking=True)
            img_b = img_b.to(device, non_blocking=True)
            disp = disp.to(device, non_blocking=True)
            ms = ms.to(device, non_blocking=True)
            dp = dp.to(device, non_blocking=True)
            ds = ds.to(device, non_blocking=True)
            with autocast(device_type='cuda', dtype=AMP_DTYPE):
                out = model(img_a, img_b, disp, ms, dp, ds)
            pxy = safe_normalize(out[:, :2].float(), dim=-1)
            pd = torch.rad2deg(torch.atan2(pxy[:, 1], pxy[:, 0]))
            pr = log_normalized_to_range(out[:, 2]).clamp(-132, 132)
            for jp, d, r in zip(jpaths, pd.cpu().tolist(), pr.cpu().tolist()):
                results.append((jp, float(d), float(r)))
    results.sort(key=lambda x: extract_group_json_sort_key(x[0]))
    with open(out_path, 'w') as f:
        for _, h, r in results:
            f.write(f'{h:.6f} {r:.6f}\n')
    print(f'Saved {out_path} ({len(results)})', flush=True)


class AverageMeter:
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, v, n=1):
        self.val = v; self.sum += v * n; self.count += n; self.avg = self.sum / self.count


def train_epoch(loader, model, optimizer, epoch, device, args, scaler):
    model.train()
    batch_time = AverageMeter()
    losses_total, losses_angle, losses_range = AverageMeter(), AverageMeter(), AverageMeter()
    t0 = time.time()
    end = time.time()
    accum = args.grad_accum_steps

    for i, (img_a, img_b, disp, ms, dp, ds, lv, ld, _, lr_gt, llnr) in enumerate(loader):
        img_a = img_a.to(device, non_blocking=True)
        img_b = img_b.to(device, non_blocking=True)
        disp = disp.to(device, non_blocking=True)
        ms = ms.to(device, non_blocking=True)
        dp = dp.to(device, non_blocking=True)
        ds = ds.to(device, non_blocking=True)
        lv = lv.to(device, non_blocking=True)
        llnr = llnr.to(device, non_blocking=True)

        with autocast(device_type='cuda', dtype=AMP_DTYPE):
            output = model(img_a, img_b, disp, ms, dp, ds)
            angle_l = angle_loss_cos_sin(output[:, :2], lv)
            range_l = log_range_loss(output[:, 2], llnr)
            loss = angle_l + args.range_loss_weight * range_l
            loss = loss / accum

        losses_total.update(loss.item() * accum, img_a.size(0))
        losses_angle.update(angle_l.item(), img_a.size(0))
        losses_range.update(range_l.item(), img_a.size(0))

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        if (i + 1) % accum == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            has_nan = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) for p in model.parameters())
            if has_nan:
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'E[{epoch}][{i+1:5d}/{len(loader)}] '
                  f'T {batch_time.val:.3f}({batch_time.avg:.3f}) '
                  f'L {losses_total.val:.4f}({losses_total.avg:.4f}) '
                  f'A {losses_angle.val:.4f}({losses_angle.avg:.4f}) '
                  f'R {losses_range.val:.4f}({losses_range.avg:.4f})')

    elapsed = time.time() - t0
    print(f'\n===== Epoch {epoch} ({elapsed:.0f}s) | L: {losses_total.avg:.4f} (a: {losses_angle.avg:.4f}, r: {losses_range.avg:.4f}) =====')
    return losses_angle.avg


def main():
    args = parser.parse_args()
    exp_name = args.exp_name
    open(f"output_{exp_name}.log", "w").close()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f'AMP dtype: {AMP_DTYPE}', flush=True)
    variant = 'V16b (stats+spatial)' if args.use_spatial else 'V16a (stats-only)'
    print(f'{variant}: range_w={args.range_loss_weight}, lr_head={args.lr_head}, lr_depth={args.lr_depth}, freeze_main={args.freeze_main_epochs}, max_correction={args.max_correction_scale}', flush=True)

    model = CrossAttentionPoseNetV15(
        dinov2_model=args.dinov2_model,
        n_cross_layers=args.n_cross_layers,
        cross_heads=args.cross_heads,
        cross_dropout=args.cross_dropout,
        use_spatial=args.use_spatial,
        max_correction_scale=args.max_correction_scale,
    )
    warm_start_from_v10(model, args.pretrained_ckpt)
    model.to(device)
    model = torch.nn.DataParallel(model)
    total_p, train_p = count_parameters(model)
    print(f'Model: {total_p:,} params ({train_p:,} trainable)', flush=True)

    full_train_dataset = DualImageDatasetV15(
        args.train_image_dir, args.train_json_dir, args.train_match_dir,
        depth_compact_path=args.depth_compact_train,
        has_gt=True, augment=True,
        depth_size=args.depth_size, use_spatial=args.use_spatial)
    test_dataset = DualImageDatasetV15(
        args.test_image_dir, args.test_json_dir, args.test_match_dir,
        depth_compact_path=args.depth_compact_test,
        has_gt=False, force_image_ext='.webp', augment=False,
        depth_size=args.depth_size, use_spatial=args.use_spatial)
    print(f'Datasets: train={len(full_train_dataset)}, test={len(test_dataset)}', flush=True)

    num_workers = min(args.num_workers, os.cpu_count() or 1)
    train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True,
                              prefetch_factor=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=num_workers, pin_memory=True, persistent_workers=True,
                             prefetch_factor=2)

    main_groups = [
        {'params': model.module.backbone.parameters(), 'lr': args.lr_backbone},
        {'params': model.module.cross_attn_layers.parameters(), 'lr': args.lr_cross_attn},
        {'params': model.module.interaction_proj.parameters(), 'lr': args.lr_head},
        {'params': model.module.disp_encoder.parameters(), 'lr': args.lr_head},
        {'params': model.module.disp_fc.parameters(), 'lr': args.lr_head},
        {'params': model.module.match_encoder.parameters(), 'lr': args.lr_head},
        {'params': model.module.regressor.parameters(), 'lr': args.lr_head},
    ]
    depth_params = list(model.module.depth_head.parameters())
    if args.use_spatial:
        depth_params += list(model.module.depth_spatial.parameters())
    depth_group = [{'params': depth_params, 'lr': args.lr_depth}]
    optimizer = torch.optim.AdamW(main_groups + depth_group, weight_decay=args.weight_decay)

    def lr_fn(ep):
        prog = float(ep) / float(max(1, args.epochs))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_fn] * 8)
    scaler = GradScaler()

    # Phase 1: freeze main, only depth trains
    if args.freeze_main_epochs > 0:
        model.module.freeze_main_branch()
        print(f'[Phase 1] Main frozen; only depth trains for {args.freeze_main_epochs} epochs', flush=True)

    best_train_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        if epoch == args.freeze_main_epochs:
            model.module.unfreeze_main_branch()
            print(f'[Phase 2] Main unfrozen from epoch {epoch}', flush=True)

        avg_angle_loss = train_epoch(train_loader, model, optimizer, epoch, device, args, scaler)
        scheduler.step()

        lrs = scheduler.get_last_lr()
        print(f'Epoch {epoch} LRs: bb={lrs[0]:.2e}, head={lrs[6]:.2e}, depth={lrs[7]:.2e}', flush=True)
        dh_last_w = model.module.depth_head.net[-1].weight.abs().sum().item()
        print(f'  depth_head |W|={dh_last_w:.4f}  train_loss={avg_angle_loss:.4f}', flush=True)

        with open(f'output_{exp_name}.log', 'a') as f:
            f.write(f'Epoch {epoch}: train_loss={avg_angle_loss:.4f} dh_w={dh_last_w:.4f}\n')

        # Best-model selection by train loss (label-free).
        if avg_angle_loss < best_train_loss:
            best_train_loss = avg_angle_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_{exp_name}.pth')
            print(f'  -> Saved best (train_loss={best_train_loss:.4f})', flush=True)
        else:
            patience_counter += 1
            print(f'  -> No train-loss improvement ({patience_counter}/{args.patience})', flush=True)
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch}', flush=True)
                break

        # Periodic test inference snapshot (label-free; just saves a txt).
        if (epoch + 1) % args.eval_test_every == 0:
            run_test_inference(test_loader, model, device,
                               output_path=f'test_predict_{exp_name}_e{epoch}.txt')

    if not args.skip_final_test:
        best_path = f'best_model_{exp_name}.pth'
        if os.path.exists(best_path):
            print('Loading best model for final inference...', flush=True)
            model.load_state_dict(torch.load(best_path, weights_only=True))
        run_test_inference(test_loader, model, device, output_path=args.test_output_txt)
        print(f'\nFINAL: predictions saved to {args.test_output_txt}', flush=True)


if __name__ == '__main__':
    main()
