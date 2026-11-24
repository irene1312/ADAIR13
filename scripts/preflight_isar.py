import _bootstrap
import argparse
from pathlib import Path
from isar.common import environment, new_run, write_json, REPO
from isar.manifest import load_manifest, verify_data
from isar.testing import measured_records


def main():
    p = argparse.ArgumentParser(description='Read-only real-data/environment verification; no split creation')
    p.add_argument('--data-root', required=True)
    p.add_argument('--aircraft-root')
    p.add_argument('--newyak-root')
    p.add_argument('--manifest', default=str(REPO/'manifests/isar_sim.json'))
    p.add_argument('--output-root', default='local_validation')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    env = environment()
    print(env, flush=True)
    if any(v=='MISSING' for v in env['packages'].values()):
        raise RuntimeError('Required dependencies are missing; see printed environment')
    manifest, sha = load_manifest(args.manifest, args.smoke)
    report = dict(environment=env, manifest_sha256=sha,
                  satellite=verify_data(manifest, Path(args.data_root), compute_stats=True))
    if args.aircraft_root:
        _, rows, files = measured_records(args.aircraft_root)
        from isar.data import MatReader
        reader = MatReader()
        for r in rows:
            reader.frame(Path(args.aircraft_root)/r['mat_path'], r['frame'])
        report['aircraft'] = dict(count=len(rows), files=files)
    if args.newyak_root:
        _, rows, files = measured_records(args.newyak_root, True)
        report['newyak'] = dict(count=len(rows), files=files)
    run = new_run(args.output_root, 'preflight')
    write_json(run/'preflight.json', report)
    print(f'Preflight passed: {run}', flush=True)


if __name__ == '__main__':
    main()
