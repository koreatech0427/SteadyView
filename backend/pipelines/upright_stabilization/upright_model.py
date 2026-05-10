import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from config import INPUT_SIZE, MIN_SIGMA_SQ, PATCH_CROP_RATIO


class GlobalLocalResidualFusionNet(nn.Module):
    def __init__(self, proj_dim=256, num_local=7):
        super().__init__()
        self.num_local = num_local

        self.backbone = models.efficientnet_b0(weights=None)
        feat_dim = self.backbone.classifier[1].in_features
        self.feature_dim = feat_dim
        self.backbone.classifier[1] = nn.Linear(feat_dim, 1)

        self.global_aux_head = nn.Linear(feat_dim, 1)
        self.local_aux_head = nn.Linear(feat_dim, 1)

        self.view_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
        )

        fusion_in_dim = proj_dim * (1 + num_local) + 2
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(256, 1),
        )

    def extract_features(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, global_img, local_imgs):
        batch_size = global_img.size(0)

        global_feat = self.extract_features(global_img)
        local_flat = local_imgs.view(batch_size * self.num_local, *local_imgs.shape[2:])
        local_feat = self.extract_features(local_flat).view(batch_size, self.num_local, self.feature_dim)

        global_pred = self.global_aux_head(global_feat)
        local_pred = self.local_aux_head(local_feat).squeeze(-1)

        all_feat = torch.cat([global_feat.unsqueeze(1), local_feat], dim=1)
        proj_flat = self.view_proj(all_feat).flatten(1)

        local_mean_pred = local_pred.mean(dim=1, keepdim=True)
        fusion_input = torch.cat([proj_flat, global_pred, local_mean_pred], dim=1)
        residual = self.fusion_head(fusion_input)

        return {
            'final_pred': global_pred + residual,
            'global_pred': global_pred,
            'local_pred': local_pred,
            'local_mean_pred': local_mean_pred,
            'residual_pred': residual,
        }


def center_crop_square(image, size=None):
    h, w = image.shape[:2]
    crop_size = min(h, w)
    sx = (w - crop_size) // 2
    sy = (h - crop_size) // 2
    cropped = image[sy:sy + crop_size, sx:sx + crop_size]
    if size is not None:
        cropped = cv2.resize(cropped, (size, size))
    return cropped


def get_patch_offsets(h, w, crop_size):
    offsets = [
        (0, 0),
        (0, w - crop_size),
        ((h - crop_size) // 2, 0),
        ((h - crop_size) // 2, (w - crop_size) // 2),
        ((h - crop_size) // 2, w - crop_size),
        (h - crop_size, 0),
        (h - crop_size, w - crop_size),
    ]
    return [
        (max(0, min(top, h - crop_size)), max(0, min(left, w - crop_size)))
        for top, left in offsets
    ]


def extract_7patches_from_view(image, size=INPUT_SIZE, crop_ratio=PATCH_CROP_RATIO):
    h, w = image.shape[:2]
    crop_size = max(2, int(min(h, w) * crop_ratio))
    patches = []
    for top, left in get_patch_offsets(h, w, crop_size):
        patch = image[top:top + crop_size, left:left + crop_size]
        patches.append(cv2.resize(patch, (size, size)))
    return patches


def make_eval_views(image, size=INPUT_SIZE):
    square_view = center_crop_square(image, size=None)
    return cv2.resize(square_view, (size, size)), extract_7patches_from_view(square_view, size=size)


def build_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def infer_frame_angle(model, frame_bgr, transform, device):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    global_view, local_patches = make_eval_views(img_rgb, size=INPUT_SIZE)

    global_tensor = transform(global_view).unsqueeze(0).to(device, non_blocking=True)
    local_tensor = torch.stack([transform(p) for p in local_patches]).unsqueeze(0).to(device, non_blocking=True)

    use_amp = device.type == 'cuda'
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
        out = model(global_tensor, local_tensor)
        final_pred = float(out['final_pred'].item())
        global_pred = float(out['global_pred'].item())
        local_mean_pred = float(out['local_mean_pred'].item())

    sigma_sq = max(
        0.5 * (final_pred - global_pred) ** 2 +
        0.5 * (final_pred - local_mean_pred) ** 2,
        MIN_SIGMA_SQ,
    )
    return final_pred, sigma_sq


def load_upright_model(model_path, device):
    model = GlobalLocalResidualFusionNet().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    return model

