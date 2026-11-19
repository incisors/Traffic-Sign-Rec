"""The shared classifier (proposal Sec. 5.1): ResNet-18 trained once on clean LISA training
windows, checkpoint chosen on the validation split, then frozen for every experiment.

    python -m tsr.classifier
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from . import config
from .data import crop_window, load_clean_frame, load_manifest

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1) * 255

CONTEXT = config.WINDOW * 10 // 7  # training crops are larger so augmentation has room
MAX_ROTATION = 10  # degrees
SCALE = (0.8, 1.2)
EPOCHS, BATCH, LR = 30, 64, 3e-4


def build_model(pretrained):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = torch.nn.Linear(model.fc.in_features, config.NUM_CLASSES)
    upsample = torch.nn.Upsample(scale_factor=config.UPSAMPLE, mode="bilinear")
    return torch.nn.Sequential(upsample, model).to(DEVICE)


def load_classifier():
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(config.CHECKPOINT, map_location=DEVICE))
    return model.eval()


def normalize(x):
    """uint8 N x H x W x 3 tensor -> ImageNet-normalised float N x 3 x H x W."""
    return (x.permute(0, 3, 1, 2).float() - MEAN) / STD


@torch.no_grad()
def logits(model, windows):
    """Class scores (N x NUM_CLASSES tensor) for a uint8 array of N x WINDOW x WINDOW x 3 windows."""
    out = []
    for i in range(0, len(windows), 256):
        x = torch.from_numpy(np.ascontiguousarray(windows[i:i + 256])).to(DEVICE)
        out.append(model(normalize(x)).cpu())
    return torch.cat(out)


def predict(model, windows):
    """Predicted labels for a uint8 array of N x WINDOW x WINDOW x 3 windows."""
    return logits(model, windows).argmax(1).numpy()


def clean_windows(rows, size):
    out = np.empty((len(rows), size, size, 3), np.uint8)
    file, frame = None, None
    for i, r in enumerate(rows):
        if r["file"] != file:
            file, frame = r["file"], load_clean_frame(r["file"])
        out[i] = crop_window(frame, r["cx"], r["cy"], size)
    return out


def augment(x):
    """Random rotation and scale about the centre, then the central window."""
    n = x.shape[0]
    angle = (torch.rand(n, device=x.device) * 2 - 1) * math.radians(MAX_ROTATION)
    zoom = (config.WINDOW / CONTEXT) / torch.empty(n, device=x.device).uniform_(*SCALE)
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = zoom * torch.cos(angle)
    theta[:, 0, 1] = -zoom * torch.sin(angle)
    theta[:, 1, 0] = zoom * torch.sin(angle)
    grid = F.affine_grid(theta, (n, 3, config.WINDOW, config.WINDOW), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)


def main():
    torch.manual_seed(0)
    train, val, test = (load_manifest(s) for s in ("train", "val", "test"))
    x_train = torch.from_numpy(clean_windows(train, CONTEXT)).to(DEVICE)
    y_train = torch.tensor([r["label"] for r in train], device=DEVICE)
    x_val, y_val = clean_windows(val, config.WINDOW), np.array([r["label"] for r in val])

    model = build_model(pretrained=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * math.ceil(len(train) / BATCH))
    best = 0.0
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(train), device=DEVICE)
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(model(augment(normalize(x_train[idx]))), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        acc = float((predict(model, x_val) == y_val).mean())
        saved = acc > best
        if saved:
            best = acc
            torch.save(model.state_dict(), config.CHECKPOINT)
        print(f"epoch {epoch + 1:2d}  loss {loss.item():.3f}  val top-1 {acc:.4f}{'  * saved' if saved else ''}")

    model = load_classifier()
    pred = predict(model, clean_windows(test, config.WINDOW))
    hit = pred == np.array([r["label"] for r in test])
    print(f"\nclean val top-1 {best:.4f} ({len(val)} samples)")
    print(f"clean test top-1 {hit.mean():.4f} ({len(test)} samples)")
    for cam in ("aiua", "vid"):
        m = np.array([r["camera"] == cam for r in test])
        print(f"  {cam}: {hit[m].mean():.4f} ({m.sum()})")
    classes = {r["label"]: r["cls"] for r in test}
    for k in sorted(classes):
        m = np.array([r["label"] == k for r in test])
        print(f"  {classes[k]:20s} {hit[m].mean():.3f} ({m.sum()})")


if __name__ == "__main__":
    main()
