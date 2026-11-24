"""Six explicit zero-shot test groups; test defaults to per-image input statistics."""
import argparse
from pathlib import Path
import numpy as np
from .checkpoint import load_checkpoint
from .common import REPO, digest, new_run, setup_device, environment, write_json
from .data import inspect_mat
from .evaluation import evaluate
from .manifest import load_manifest, verify_data
from .model import build_model
from .newyak import FILES, read_newyak

SUITES = ('satellite-seen', 'satellite-unseen', 'aircraft-raw', 'aircraft-noisy', 'newyak-raw', 'newyak-noisy')


def measured_records(root, newyak=False):
    root = Path(root).expanduser().resolve()
    names = FILES if newyak else {t: t+'_Test.mat' for t in ('An26', 'Yak', 'JZ')}
    if not root.is_dir():
        raise FileNotFoundError(root)
    records, files = [], {}
    for target, filename in names.items():
        matches = [p for p in root.rglob('*') if p.is_file() and p.name.lower()==filename.lower()]
        if len(matches)!=1:
            raise ValueError(f'Expected exactly one {filename}, found {len(matches)}; Train MAT is never selected')
        path = matches[0]
        if newyak:
            z = read_newyak(path)
            shape, count = list(z.shape)+[1], 1
        else:
            info = inspect_mat(path)
            if info is None or info['shape'][:2] != [128, 128]:
                raise ValueError(f'Expected target_zong 128x128xN in {path}')
            shape, count = info['shape'], info['shape'][2]
        relative = path.relative_to(root).as_posix()
        files[target] = dict(path=relative, shape=shape, sha256=digest(path), size_bytes=path.stat().st_size)
        records.extend(dict(target=target, mat_path=relative, frame=i, subset='measured_test',
                            reader='newyak' if newyak else 'target_zong') for i in range(1, count+1))
    return root, records, files


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=SUITES, required=True)
    p.add_argument('--checkpoint', required=True, help='Explicit trusted checkpoint .pt or latest/best/final.json pointer')
    p.add_argument('--manifest', default=str(REPO/'manifests/isar_sim.json'))
    p.add_argument('--data-root')
    p.add_argument('--output-root', default='results/isar')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--normalization', choices=('per_image', 'fixed_train'), default='per_image')
    p.add_argument('--snrs', nargs='+', type=float, default=[-5, 0, 5])
    p.add_argument('--save-complex', choices=('all', 'selected', 'none'), default='all')
    p.add_argument('--export-mat', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--limit', type=int, help='SMOKE ONLY: first N frames per target; never a formal full-set result')
    p.add_argument('--no-visuals', action='store_true', help='Explicit diagnostic runs only')
    return p


def execute(args):
    device = setup_device(args.device, args.threads)
    checkpoint_path, c = load_checkpoint(args.checkpoint)
    smoke = bool(args.smoke or c['config'].get('smoke', False))
    if args.limit is not None and (not args.smoke or args.limit < 1):
        raise ValueError('--limit requires --smoke and a positive number')
    if args.no_visuals and not args.smoke:
        raise ValueError('--no-visuals is a diagnostic option and requires --smoke')
    if not args.snrs or not np.isfinite(args.snrs).all() or len(set(args.snrs))!=len(args.snrs):
        raise ValueError('SNRs must be finite, nonempty and unique')
    satellite = args.suite.startswith('satellite')
    newyak = args.suite.startswith('newyak')
    raw = args.suite.endswith('raw')
    defaults = dict(satellite='/root/autodl-tmp/data_45', aircraft='/root/autodl-tmp/datasets', newyak='/root/autodl-tmp/newyak/newyak')
    domain = 'satellite' if satellite else 'newyak' if newyak else 'aircraft'
    root = Path(args.data_root or defaults[domain]).expanduser().resolve()
    manifest_sha = None
    if satellite:
        m, manifest_sha = load_manifest(args.manifest, args.smoke)
        if manifest_sha != c['manifest_sha256'] or m['normalization'] != c['stats']:
            raise ValueError('Checkpoint manifest/training statistics mismatch')
        subset = 'test_seen' if args.suite.endswith('seen') and not args.suite.endswith('unseen') else 'test_unseen'
        records = [r for r in m['records'] if r['subset']==subset]
        verified = verify_data(m, root, (subset,), compute_stats=False)
        files = verified['files']
    else:
        root, records, files = measured_records(root, newyak)
    full_count = len(records)
    if args.limit:
        counts = {}
        limited = []
        for r in records:
            counts[r['target']] = counts.get(r['target'], 0)+1
            if counts[r['target']] <= args.limit:
                limited.append(r)
        records = limited
    run = new_run(args.output_root, ('SMOKE_' if smoke else '')+args.suite)
    write_json(run/'run.json', dict(suite=args.suite, checkpoint=str(checkpoint_path),
        checkpoint_sha256=digest(checkpoint_path), checkpoint_step=c['step'], config=c['config'],
        training_stats=c['stats'], manifest_sha256=manifest_sha, source_manifest_sha256=c['manifest_sha256'],
        data_root=str(root), files=files, full_available_frames=full_count, evaluated_frames=len(records),
        normalization=args.normalization, snrs=[] if raw else args.snrs,
        raw_no_added_noise=raw, smoke=smoke, limit=args.limit, device=str(device), environment=environment(),
        save_complex=args.save_complex, export_mat=args.export_mat))
    model = build_model(c['config']['model']).to(device)
    model.load_state_dict(c['model'], strict=True)
    evaluate(model, records, root, c['stats'], args.snrs, c['config']['noise_seed'], device, run,
             args.normalization, raw, satellite, args.save_complex, args.export_mat,
             not args.no_visuals, smoke)
    write_json(run/'completed.json', dict(status='complete', evaluated_frames=len(records), smoke=smoke))
    print(f'Results saved: {run}', flush=True)
    return run


def main():
    execute(parser().parse_args())
