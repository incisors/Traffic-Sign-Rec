"""Shared settings. Both team members must use the same values."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LISA = ROOT / "data" / "LISA" / "signDatabasePublicFramesOnly"
MANIFEST = ROOT / "data" / "manifest.csv"
CHECKPOINT = ROOT / "data" / "classifier.pt"

# Sample selection (proposal Sec. 4.2)
MIN_BOX = 32            # keep signs whose bounding box is at least MIN_BOX x MIN_BOX pixels
SHARPNESS_DROP = 0.10   # drop this fraction of the least sharp crops
NUM_CLASSES = 10        # keep the most frequent classes
GROUP_GAP = 300         # tracks of one video closer than this many frames share a split
SPLIT = (0.70, 0.15, 0.15)

# The aiua* sequences are un-demosaiced grey mosaics with a 2x2 checkerboard pattern
# (sigma-hat ~12 on the "clean" frame). True = remove it when loading a clean frame.
DEMOSAIC_AIUA = True

# Classification window. The proposal (Sec. 5.1) used 224 with no resizing, but the median
# sign is only 43 px: that reached ~92% clean test accuracy (89-95% across training seeds),
# while a 112 window upsampled 2x reaches ~99%. Restoration still runs on the full frame.
# To go back to the proposal's design: WINDOW = 224, UPSAMPLE = 1, then retrain.
WINDOW = 112
UPSAMPLE = 2

# Degradation levels (Sec. 4.3)
NOISE_SIGMAS = (15, 25, 50)
BLUR_LENGTHS = (10, 20, 30)
COMBINED = {"blur_len": 20, "sigma": 25}
