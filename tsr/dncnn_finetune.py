"""Fine-tune the pretrained blind DnCNN on synthetic noisy/clean pairs from the LISA training
split (proposal Sec. 5.2). Sigma is drawn uniformly from [5, 55] per image so the model
stays blind, the loss is the residual MSE, and the checkpoint with the best validation
PSNR is kept. The training noise is rounded and clipped exactly like tsr.degrade.

    python -m tsr.dncnn_finetune
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .classifier import DEVICE, clean_windows
from .data import crop_window, load_clean_frame, load_manifest
from .degrade import degrade
from .denoise import DNCNN_FINETUNED, DNCNN_PRETRAINED, build_dncnn, load_dncnn_weights

CONTEXT = 160  # crop centred on each sign: the window plus 24 px, more than DnCNN's 20 px reach
PATCH, BATCH, STEPS, LR, EVAL_EVERY = 64, 64, 6000, 5e-5, 300
SIGMA_RANGE = (5, 55)


def noisy_windows(rows, sigma, size):
    out = np.empty((len(rows), size, size, 3), np.uint8)
    for i, r in enumerate(rows):
        out[i] = crop_window(degrade(load_clean_frame(r["file"]), r["file"], sigma), r["cx"], r["cy"], size)
    return torch.from_numpy(out).to(DEVICE)


@torch.no_grad()
def val_psnr(net, noisy, clean):
    """Mean PSNR of the denoised classification windows, per noise level."""
    off = (CONTEXT - config.WINDOW) // 2
    ref = clean.permute(0, 3, 1, 2).float()
    result = {}
    for sigma, x in noisy.items():
        values = []
        for i in range(0, len(x), 64):
            z = x[i:i + 64].permute(0, 3, 1, 2).float() / 255
            out = ((z - net(z)).clamp(0, 1) * 255).round()[:, :, off:off + config.WINDOW, off:off + config.WINDOW]
            mse = ((out - ref[i:i + 64]) ** 2).mean(dim=(1, 2, 3))
            values.append(10 * torch.log10(255 ** 2 / mse))
        result[sigma] = torch.cat(values).mean().item()
    return result


def main():
    torch.manual_seed(0)
    train, val = load_manifest("train"), load_manifest("val")
    clean = torch.from_numpy(clean_windows(train, CONTEXT)).to(DEVICE)
    val_clean = torch.from_numpy(clean_windows(val, config.WINDOW)).to(DEVICE)
    val_noisy = {s: noisy_windows(val, s, CONTEXT) for s in config.NOISE_SIGMAS}

    net = load_dncnn_weights(build_dncnn(), DNCNN_PRETRAINED)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

    def check(step, loss):
        scores = val_psnr(net, val_noisy, val_clean)
        mean = sum(scores.values()) / len(scores)
        print(f"step {step:5d}  loss {loss:.6f}  val PSNR " + "  ".join(f"s{s}: {v:.2f}" for s, v in scores.items())
              + f"  mean {mean:.3f}", flush=True)
        return mean

    best = check(0, float("nan"))
    torch.save(net.state_dict(), DNCNN_FINETUNED)
    running = 0.0
    for step in range(1, STEPS + 1):
        idx = torch.randint(len(clean), (BATCH,), device=DEVICE)
        oy, ox = torch.randint(CONTEXT - PATCH + 1, (2,)).tolist()
        x = clean[idx, oy:oy + PATCH, ox:ox + PATCH].permute(0, 3, 1, 2).float()
        if torch.rand(()) < 0.5:
            x = x.flip(3)
        if torch.rand(()) < 0.5:
            x = x.flip(2)
        sigma = torch.empty(BATCH, 1, 1, 1, device=DEVICE).uniform_(*SIGMA_RANGE)
        noisy = (x + sigma * torch.randn_like(x)).round().clamp(0, 255)
        loss = F.mse_loss(net(noisy / 255), (noisy - x) / 255)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()
        if step % EVAL_EVERY == 0:
            mean = check(step, running / EVAL_EVERY)
            running = 0.0
            if mean > best:
                best = mean
                torch.save(net.state_dict(), DNCNN_FINETUNED)
    print(f"best mean val PSNR {best:.3f} -> {DNCNN_FINETUNED}")


if __name__ == "__main__":
    main()
