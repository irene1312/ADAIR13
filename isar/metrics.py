"""FFDNet-ISAR amplitude convention: fixed training scale, clipping for metrics only."""
import math
import numpy as np
from skimage.metrics import structural_similarity


def metrics(reference, estimate, amplitude_scale):
    if reference.shape != estimate.shape or reference.ndim != 2:
        raise ValueError('Expected matching 2D complex frames')
    if not (np.isfinite(reference).all() and np.isfinite(estimate).all()):
        raise ValueError('Non-finite metric input')
    if not np.isfinite(amplitude_scale) or amplitude_scale <= 0:
        raise ValueError('Invalid amplitude scale')
    a, b = [np.clip(np.abs(z) / amplitude_scale, 0, 1) for z in (reference, estimate)]
    mse = float(np.mean((a - b) ** 2))
    psnr = math.inf if mse == 0 else -10 * math.log10(mse)
    ssim = float(structural_similarity(a, b, data_range=1.0, win_size=7,
                                      gaussian_weights=False, use_sample_covariance=True))
    energy = float(np.sum(np.abs(reference) ** 2))
    error = float(np.sum(np.abs(estimate - reference) ** 2))
    nmse = error / energy if energy > 0 else (0.0 if error == 0 else math.inf)
    db = -math.inf if nmse == 0 else 10 * math.log10(nmse)
    return dict(psnr=psnr, ssim=ssim, nmse=nmse, nmse_db=db)
