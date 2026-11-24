"""IDF f5ca181 input statistics and float32 encode/decode boundary."""
import hashlib
import numpy as np
from .data import channels, as_complex


def check_image(z):
    z = np.asarray(z)
    if z.ndim != 2 or not z.size or not np.iscomplexobj(z) or not np.isfinite(z).all():
        raise ValueError('Expected one finite nonempty complex 2D image')
    return z


def check_stats(stats):
    if (np.shape(stats['mean']) != (2,) or not np.isfinite(stats['mean']).all()
            or not np.isfinite(stats['std']) or stats['std'] <= 0
            or not np.isfinite(stats['amplitude_scale']) or stats['amplitude_scale'] <= 0):
        raise ValueError('Invalid separate means/shared std/training amplitude_scale')


def input_stats(z, training, mode='per_image'):
    """Fit float64 population statistics only to the actual, unpadded input."""
    check_image(z)
    check_stats(training)
    if mode == 'fixed_train':
        return dict(training)
    if mode != 'per_image':
        raise ValueError('normalization must be per_image or fixed_train')
    x = np.stack((z.real, z.imag)).astype(np.float64)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            mean = x.mean(axis=(1, 2), dtype=np.float64)
            centered = x - mean[:, None, None]
            std = float(np.sqrt(np.sum(centered ** 2, dtype=np.float64) / (2 * z.size)))
    except FloatingPointError as exc:
        raise ValueError('Non-finite per-image statistics') from exc
    result = dict(training, mean=mean.tolist(), std=std,
                  source='actual input only; population pooled within-channel variance')
    check_stats(result)
    return result


def encode(z, stats):
    check_image(z)
    check_stats(stats)
    # Retain IDF's initial channels(float32) conversion, then float64 arithmetic.
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        x = ((channels(z).astype(np.float64) - np.asarray(stats['mean'])[:, None, None])
             / stats['std']).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite encoded input')
    return x


def decode(x, stats):
    check_stats(stats)
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[0] != 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite [2,H,W] network output')
    x = x.astype(np.float64) * stats['std'] + np.asarray(stats['mean'])[:, None, None]
    z = as_complex(x).astype(np.complex128)
    check_image(z)
    return z


def array_sha(z):
    return hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()
