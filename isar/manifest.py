"""Never discover a replacement split. Validate and use the IDF manifest verbatim."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from .common import REPO, digest
from .data import MatReader, inspect_mat, normalization

SEEN = ('Aqua', 'HST', 'Beidou')
UNSEEN = ('CALIPSO', 'WV2')
RANGES = {'train': (21, 240), 'val': (261, 300), 'test_seen': (321, 360)}
CANONICAL = REPO / 'manifests/isar_sim.json'
EXPECTED_SHA256 = 'a1f01317ad40d1895fe4393b1cf75aa44ca01de7c95d5057ab9589f6a0dbaa2c'


def manifest_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def load_manifest(path, smoke=False):
    m = json.loads(Path(path).read_text(encoding='utf-8'))
    if m.get('synthetic_smoke'):
        if not smoke:
            raise ValueError('Synthetic manifest requires explicit --smoke')
    elif manifest_hash(path) != EXPECTED_SHA256:
        raise ValueError('Formal manifest must match the unmodified IDF f5ca181 isar_sim.json')
    validate_manifest(m)
    return m, manifest_hash(path)


def validate_manifest(m):
    if m.get('frame_index_base') != 1:
        raise ValueError('Manifest frame numbering must be MATLAB 1-based')
    rows = m['records']
    keys = [(r['mat_path'], r['frame']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate or overlapping manifest frame assignments')
    synthetic = m.get('synthetic_smoke', False)
    for r in rows:
        p = Path(r['mat_path'])
        if p.is_absolute() or '..' in p.parts or ':' in r['mat_path']:
            raise ValueError('Manifest paths must stay within data root')
        if r['target'] not in m['files'] or r['mat_path'] != m['files'][r['target']]['path']:
            raise ValueError('Manifest target/file inconsistency')
        if not isinstance(r['frame'], int) or not 1 <= r['frame'] <= m['files'][r['target']]['shape'][2]:
            raise ValueError('Manifest frame out of range')
        target, frame, subset = r['target'], r['frame'], r['subset']
        if target in UNSEEN and subset != 'test_unseen':
            raise ValueError('Unseen satellite leakage')
        if subset in RANGES and target not in SEEN:
            raise ValueError('Non-seen target in train/validation/seen test')
        if not synthetic:
            expected = ('test_unseen' if target in UNSEEN else 'unused')
            if target in SEEN:
                expected = next((s for s, (lo, hi) in RANGES.items() if lo <= frame <= hi), 'guard')
            if subset != expected:
                raise ValueError(f'Manifest interval changed: {r}')
    if not synthetic:
        counts = Counter(r['subset'] for r in rows)
        for subset, n in dict(train=660, val=120, test_seen=120, test_unseen=720).items():
            if counts[subset] != n:
                raise ValueError(f'Incorrect {subset} count: {counts[subset]} != {n}')


def verify_data(m, root, subsets=('train', 'val', 'test_seen', 'test_unseen'), compute_stats=True):
    root = Path(root).resolve()
    records = [r for r in m['records'] if r['subset'] in subsets]
    targets = sorted({r['target'] for r in records})
    files = {}
    reader = MatReader()
    for target in targets:
        entry = m['files'][target]
        path = root / entry['path']
        if not path.is_file():
            raise FileNotFoundError(f'Manifest file missing for {target}: {path}; no automatic reassignment')
        info = inspect_mat(path)
        if info is None or info['shape'] != entry['shape']:
            raise ValueError(f'{target}: MAT shape differs from original manifest: {info} vs {entry}')
        frames = [r for r in records if r['target'] == target]
        print(f'Checking {target}: shape={info["shape"]}, {len(frames)} assigned frames', flush=True)
        for r in frames:
            reader.frame(path, r['frame'])
        files[target] = dict(**info, path=entry['path'], sha256=digest(path), size_bytes=path.stat().st_size)
    stats = None
    if compute_stats:
        if not all(r in records for r in m['records'] if r['subset'] == 'train'):
            raise ValueError('Statistics verification requires the entire training split')
        print('Computing statistics from all clean training frames only', flush=True)
        stats = normalization(m['records'], root)
        expected = m['normalization']
        for k in ('mean', 'std', 'amplitude_scale', 'pixels_per_channel'):
            if not np.allclose(stats[k], expected[k], rtol=1e-10, atol=1e-14):
                raise ValueError(f'Training data inconsistent with IDF manifest statistics: {k}: {stats[k]} != {expected[k]}')
    return dict(files=files, verified_counts=dict(Counter(r['subset'] for r in records)),
                computed_training_stats=stats, consistency='shape, finite complex128 frames, file SHA256, train statistics')
