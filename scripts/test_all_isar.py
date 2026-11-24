import _bootstrap
import argparse
import subprocess
import sys
from pathlib import Path
from isar.testing import SUITES


def main():
    p = argparse.ArgumentParser(description='Run all six suites with one explicit checkpoint; stop on failure')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest')
    p.add_argument('--satellite-root', default='/root/autodl-tmp/data_45')
    p.add_argument('--aircraft-root', default='/root/autodl-tmp/datasets')
    p.add_argument('--newyak-root', default='/root/autodl-tmp/newyak/newyak')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output-root', default='results/isar')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--normalization', choices=('per_image', 'fixed_train'), default='per_image')
    p.add_argument('--save-complex', choices=('all', 'selected', 'none'), default='all')
    p.add_argument('--export-mat', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    for suite in SUITES:
        root = args.satellite_root if suite.startswith('satellite') else args.aircraft_root if suite.startswith('aircraft') else args.newyak_root
        cmd = [sys.executable, str(Path(__file__).with_name('test_isar.py')), '--suite', suite,
               '--checkpoint', args.checkpoint, '--data-root', root, '--device', args.device,
               '--threads', str(args.threads), '--output-root', args.output_root,
               '--normalization', args.normalization, '--save-complex', args.save_complex]
        if args.export_mat:
            cmd.append('--export-mat')
        if args.manifest:
            cmd.extend(['--manifest', args.manifest])
        if args.smoke:
            cmd.append('--smoke')
        if args.limit is not None:
            cmd.extend(['--limit', str(args.limit)])
        print(f'Running {suite}', flush=True)
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
