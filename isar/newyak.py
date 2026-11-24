"""Read the two NewYak images without changing legacy target_zong access."""
from pathlib import Path

import numpy as np
import h5py
from scipy.io import loadmat


FILES = {'Yake1': 'yake1.mat', 'Yake2': 'yake2.mat'}


def read_newyak(path):
    path = Path(path)
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as f:
            d = f['data_test']['im_ori']
            if h5py.check_dtype(ref=d.dtype) is not None:
                if d.size != 1:
                    raise ValueError(f'Expected scalar data_test.im_ori reference: {path}')
                d = f[d[()].flat[0]]
            image = d[()].T
            if image.dtype.names:
                image = image['real'] + 1j * image['imag']
    else:
        item = loadmat(path, variable_names=['data_test'])['data_test']
        if item.shape != (1, 1) or not item.dtype.names or 'im_ori' not in item.dtype.names:
            raise ValueError(f'Expected scalar data_test.im_ori: {path}')
        image = item['im_ori'][0, 0]
    if (image.shape != (128, 128) or image.dtype != np.dtype('complex128')
            or not np.isfinite(image).all() or not np.any(image)):
        raise ValueError(f'Expected finite nonzero complex128 128x128 image: {path}')
    return image
