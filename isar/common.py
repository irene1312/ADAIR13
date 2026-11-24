import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import uuid

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else str(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value, exclusive=True):
    with Path(path).open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(safe_json(value), f, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to write')
    with Path(path).open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def append_csv(path, row):
    with Path(path).open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)
        f.flush()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def new_run(root, label):
    if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):
        raise ValueError('Run label must contain letters, numbers, underscore or hyphen')
    p = Path(root) / f'{label}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}'
    p.mkdir(parents=True, exist_ok=False)
    return p.resolve()


def environment():
    versions = {}
    for name in ('torch', 'numpy', 'scipy', 'h5py', 'einops', 'scikit-image', 'matplotlib', 'pytest'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'MISSING'
    try:
        revision = subprocess.check_output(['git', '-c', f'safe.directory={REPO.as_posix()}',
            '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-c', f'safe.directory={REPO.as_posix()}',
            '-C', str(REPO), 'status', '--porcelain'], text=True).strip())
    except (subprocess.SubprocessError, OSError):
        revision, dirty = 'unknown', None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions,
                cuda_available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda,
                cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                git_commit=revision, git_dirty=dirty)


def load_config(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def setup_device(device, threads=4):
    if threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    d = torch.device(device)
    if d.type not in ('cpu', 'cuda') or (d.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError(f'Unavailable device: {device}')
    return d
