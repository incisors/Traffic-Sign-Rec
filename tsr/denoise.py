"""Artifact A (noise) restoration methods. Each takes and returns a uint8 HxWx3 image."""
import functools

import bm3d
import numpy as np
import torch

from . import config
from .classifier import DEVICE
from .data import estimate_sigma

DNCNN_PRETRAINED = config.ROOT / "data" / "dncnn" / "dncnn_color_blind.pth"  # from cszn/KAIR
DNCNN_FINETUNED = config.ROOT / "data" / "dncnn" / "dncnn_finetuned.pth"     # tsr.dncnn_finetune


def bm3d_denoise(image, sigma=None, scale=1.0):
    """BM3D run at noise level scale * sigma. sigma=None is the blind, deployable version:
    sigma is estimated from the image."""
    if sigma is None:
        sigma = estimate_sigma(image)
    if sigma == 0:
        return image
    out = bm3d.bm3d_rgb(image.astype(np.float32) / 255, scale * sigma / 255)
    return np.clip(np.rint(out * 255), 0, 255).astype(np.uint8)


def build_dncnn():
    """Colour DnCNN-B: 20 conv layers; the output is the predicted noise."""
    layers = [torch.nn.Conv2d(3, 64, 3, padding=1), torch.nn.ReLU()]
    for _ in range(18):
        layers += [torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU()]
    layers.append(torch.nn.Conv2d(64, 3, 3, padding=1))
    return torch.nn.Sequential(*layers).to(DEVICE)


def load_dncnn_weights(net, weights):
    state = torch.load(weights, map_location=DEVICE)
    net.load_state_dict({k.removeprefix("model."): v for k, v in state.items()})
    return net


@functools.lru_cache
def _dncnn(weights):
    return load_dncnn_weights(build_dncnn(), weights).eval()


@torch.no_grad()
def dncnn_denoise(image, weights=DNCNN_PRETRAINED):
    """Blind DnCNN: the network predicts the noise, which is subtracted from the image."""
    x = torch.from_numpy(np.ascontiguousarray(image)).to(DEVICE).permute(2, 0, 1)[None].float() / 255
    out = x - _dncnn(weights)(x)
    return (out.clamp(0, 1) * 255).round().byte()[0].permute(1, 2, 0).cpu().numpy()
