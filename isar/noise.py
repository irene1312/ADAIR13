import numpy as np


def add_noise(clean, snr_db, rng):
    clean = np.asarray(clean, dtype=np.complex128)
    if not np.isfinite(clean).all() or not np.isfinite(snr_db):
        raise ValueError('Non-finite signal or SNR')
    power = float(np.mean(np.abs(clean) ** 2))
    if not np.isfinite(power) or power <= 0:
        raise ValueError('Zero or invalid signal power; exclude and investigate the frame')
    sigma = np.sqrt(power / (2 * 10.0 ** (snr_db / 10)))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Invalid noise scale')
    noise = sigma * (rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape))
    actual = 10 * np.log10(power / np.mean(np.abs(noise) ** 2))
    return clean + noise, float(actual)
