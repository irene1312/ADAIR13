"""Read-only MAT access. All public frame indices are MATLAB 1-based."""
import hashlib
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat, whosmat

TARGETS = ('Aqua', 'CALIPSO', 'Cloudsat', 'HST', 'Jason-3', 'OCO-2', 'TG', 'WV2', 'Beidou')


def data_root(value=None):
    value = value or os.environ.get('ISAR_DATA_ROOT')
    if not value:
        raise ValueError('Specify --data-root, config data_root, or ISAR_DATA_ROOT')
    return Path(value).expanduser().resolve()


def inspect_mat(path):
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as f:
            if 'target_zong' not in f:
                return None
            d = f['target_zong']
            if not isinstance(d, h5py.Dataset):
                raise ValueError(f'Unsupported target_zong group: {path}')
            # MATLAB v7.3 HDF5 dimensions are reversed relative to MATLAB.
            shape = tuple(reversed(d.shape))
            complex_data = d.dtype.kind == 'c' or set(d.dtype.names or ()) >= {'real', 'imag'}
            fmt = 'mat73'
    else:
        entries = {name: (shape, kind) for name, shape, kind in whosmat(path)}
        if 'target_zong' not in entries:
            return None
        shape, _ = entries['target_zong']
        complex_data = None  # whosmat does not expose the complex flag; check on read.
        fmt = 'scipy'
    if len(shape) == 2:
        shape = (*shape, 1)
    if len(shape) != 3:
        raise ValueError(f'Expected H,W,F: {path}: {shape}')
    return dict(format=fmt, shape=list(shape), complex=complex_data)


def discover(root):
    found = {}
    for path in sorted(root.rglob('*.mat')):
        info = inspect_mat(path)
        if info is None:
            continue  # RCS/scattering files are never read as images.
        names = [t for t in TARGETS if any(p.lower().startswith(t.lower() + '_')
                                          or p.lower() == t.lower() for p in path.relative_to(root).parts)]
        if len(names) != 1:
            raise ValueError(f'Cannot identify unique target: {path}')
        target = names[0]
        if target in found:
            raise ValueError(f'Multiple target_zong files for {target}; select a narrower root')
        found[target] = dict(path=path.relative_to(root).as_posix(), **info)
    return found


class MatReader:
    """v7.3 slices on demand; v5 compressed MAT requires one whole variable.

    Keep at most one v5 variable per reader, never all satellites in memory.
    No persistent HDF5 handles (safe for later worker processes).
    """
    def __init__(self):
        self.cached_path = None
        self.cached_array = None

    def frame(self, path, index):
        path = Path(path)
        if h5py.is_hdf5(path):
            with h5py.File(path, 'r') as f:
                d = f['target_zong']
                count = d.shape[0] if d.ndim == 3 else 1
                if not 1 <= index <= count:
                    raise IndexError(index)
                a = d[index - 1, :, :].T if d.ndim == 3 else d[:].T
                if a.dtype.names:
                    a = a['real'] + 1j * a['imag']
        else:
            if self.cached_path != path:
                self.cached_array = None  # release previous satellite before allocating another
                a = loadmat(path, variable_names=['target_zong'])['target_zong']
                self.cached_array = a[:, :, None] if a.ndim == 2 else a
                self.cached_path = path
            if not 1 <= index <= self.cached_array.shape[2]:
                raise IndexError(index)
            a = self.cached_array[:, :, index - 1]
        if a.ndim != 2 or a.dtype != np.dtype('complex128') or not np.isfinite(a).all():
            raise ValueError(f'Expected finite complex128 2D data: {path}, frame {index}, dtype={a.dtype}')
        return np.asarray(a, dtype=np.complex128)


def available_memory_bytes():
    """Conservative OS memory preflight, without adding a dependency."""
    if os.name == 'nt':
        import ctypes
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ('total', 'available', 'page_total', 'page_available',
                 'virtual_total', 'virtual_available', 'extended')]
        s = Status()
        s.length = ctypes.sizeof(s)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return s.available
    else:
        try:
            for line in Path('/proc/meminfo').read_text().splitlines():
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
        except OSError:
            pass
    return None


class PreloadedMatReader(MatReader):
    """Optional read-only v5 preloading; v7.3 retains on-demand slicing.

    Does not sample or transform data. Budget covers retained variables; OS
    preflight also reserves a decode-sized temporary plus 512 MiB headroom.
    Network/backward memory is additional: this is not a total-RAM guarantee.
    """
    def __init__(self, paths, max_mib=1536):
        super().__init__()
        self.arrays = {}
        sizes = {}
        for p in sorted({Path(p).resolve() for p in paths}):
            info = inspect_mat(p)
            if info is None:
                raise ValueError(f'Missing target_zong: {p}')
            if info['format'] == 'scipy':
                sizes[p] = int(np.prod(info['shape'])) * 16
        total = sum(sizes.values())
        if total > max_mib * 1024**2:
            raise MemoryError(f'Preload needs {total / 1024**2:.1f} MiB; budget={max_mib} MiB')
        available = available_memory_bytes()
        required = total + max(sizes.values(), default=0) + 512 * 1024**2
        if total and (available is None or available < required):
            raise MemoryError(f'Preload OS preflight failed: need {required}, available={available}')
        for p in sizes:
            a = loadmat(p, variable_names=['target_zong'])['target_zong']
            if a.ndim == 2:
                a = a[:, :, None]
            if a.ndim != 3 or not np.iscomplexobj(a):
                raise ValueError(f'Expected complex H,W,F: {p}')
            a.setflags(write=False)
            self.arrays[p] = a
        self.preloaded_bytes = sum(a.nbytes for a in self.arrays.values())

    def frame(self, path, index):
        a = self.arrays.get(Path(path).resolve())
        if a is None:
            return super().frame(path, index)
        if not 1 <= index <= a.shape[2]:
            raise IndexError(index)
        z = a[:, :, index - 1]
        if not np.isfinite(z).all():
            raise ValueError(f'Non-finite frame: {path}, {index}')
        return np.asarray(z, dtype=np.complex128)


def stable_seed(seed, target, frame, snr):
    # Independent of manifest ordering, selected subset, Python hash, and model.
    key = f'{int(seed)}|{target.lower()}|{int(frame)}|{float(snr):.8g}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')


def channels(a):
    return np.stack((a.real, a.imag)).astype(np.float32)


def as_complex(a):
    return a[0].astype(np.float64) + 1j * a[1].astype(np.float64)


def normalization(records, root):
    reader = MatReader()
    count, mean, m2, peak = 0, np.zeros(2), np.zeros(2), 0.0
    for r in records:
        if r['subset'] != 'train':
            continue
        z = reader.frame(root / r['mat_path'], r['frame'])
        x = np.stack((z.real, z.imag)).reshape(2, -1)
        n = x.shape[1]
        delta = x.mean(1) - mean
        new_count = count + n
        m2 += ((x - x.mean(1)[:, None]) ** 2).sum(1) + delta**2 * count * n / new_count
        mean += delta * n / new_count
        count = new_count
        peak = max(peak, float(np.abs(z).max()))
    if not count:
        raise ValueError('No training frames')
    std = float(np.sqrt(m2.sum() / (2 * count)))
    if not np.isfinite(std) or std <= 0 or peak <= 0:
        raise ValueError('Degenerate training statistics')
    return dict(mean=mean.tolist(), std=std, amplitude_scale=peak, pixels_per_channel=count,
                source='train only; population pooled within-channel variance')
