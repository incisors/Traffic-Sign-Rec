"""Synthetic degradations (proposal Sec. 4.3), applied to the full clean frame.

Every random draw is seeded by the frame's file name, so a given frame always gets the
same noise field and the same blur angle, whoever runs it and whichever method follows.
"""
import zlib

import numpy as np


def _rng(file, stream):
    return np.random.default_rng([zlib.crc32(file.encode()), stream])


def add_noise(frame, sigma, file):
    """Additive Gaussian noise, independent per colour channel, clipped to [0, 255]."""
    noise = _rng(file, 0).standard_normal(frame.shape, dtype=np.float32)
    return np.clip(np.rint(frame + sigma * noise), 0, 255).astype(np.uint8)


def blur_angle(file):
    """The fixed random motion direction of this frame, in degrees [0, 180)."""
    return float(_rng(file, 1).uniform(0, 180))


def motion_blur(frame, length, angle):
    """Linear motion blur with a PSF of `length` pixels at `angle` degrees."""
    raise NotImplementedError("Artifact B: linear-PSF motion blur goes here")


def degrade(frame, file, sigma=0, blur_len=0):
    """Blur first, then noise: the physical order (sensor noise is added after optical blur)."""
    if blur_len:
        frame = motion_blur(frame, blur_len, blur_angle(file))
    if sigma:
        frame = add_noise(frame, sigma, file)
    return frame
