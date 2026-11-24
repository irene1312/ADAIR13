import _bootstrap
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from isar.common import new_run, write_json
from isar.data import stable_seed
from isar.metrics import metrics
from isar.noise import add_noise
from isar.normalization import input_stats, array_sha, encode


def audit(root, output_root):
    report = []
    for path in sorted(Path(root).iterdir()):
        if not path.is_dir() or not (path/'completed.json').exists():
            continue
        meta = json.loads((path/'run.json').read_text(encoding='utf-8'))
        logs = [json.loads(s) for s in (path/'normalization_stats.jsonl').read_text(encoding='utf-8').splitlines()]
        raw = meta['raw_no_added_noise']
        for row in logs:
            stem = f'{row["target"]}_f{row["frame"]:04d}_snr{row["snr_db"]}'
            with np.load(path/'complex'/f'{stem}.npz') as data:
                original = data['reference'] if meta['suite'].startswith('satellite') else data['original_measured']
                actual, restored = data['input'], data['restored']
                assert original.dtype==actual.dtype==restored.dtype==np.complex128
                assert actual.shape==restored.shape==tuple(row['input_shape'])
                assert array_sha(actual)==row['input_sha256'] and array_sha(original)==row['original_sha256']
                if raw:
                    np.testing.assert_array_equal(original, actual)
                    assert row['noise_seed'] is None
                else:
                    seed=stable_seed(meta['config']['noise_seed'],row['target'],row['frame'],row['snr_db'])
                    expected,_=add_noise(original,row['snr_db'],np.random.default_rng(seed))
                    np.testing.assert_array_equal(expected,actual)
                stats=input_stats(actual,meta['training_stats'],meta['normalization'])
                assert row['mean_real']==stats['mean'][0] and row['mean_imag']==stats['mean'][1] and row['shared_std']==stats['std']
                assert row['encoded_input_sha256']==array_sha(encode(actual,stats))
                np.testing.assert_array_equal(restored.real,data['restored_real'])
                np.testing.assert_array_equal(restored.imag,data['restored_imag'])
                np.testing.assert_array_equal(np.angle(restored),data['restored_phase'])
                mat=path/'complex'/f'{stem}.mat'
                if mat.exists():
                    np.testing.assert_array_equal(loadmat(mat)['restored'],restored)
                if meta['suite'].startswith('satellite'):
                    for prefix,z in (('noisy',actual),('restored',restored)):
                        actual_metrics=metrics(original,z,meta['training_stats']['amplitude_scale'])
                        for k,v in actual_metrics.items():
                            assert np.isclose(row[prefix+'_'+k],v,rtol=0,atol=0)
                else:
                    assert not any('psnr' in k or 'ssim' in k or 'nmse' in k for k in row)
        with (path/'per_image.csv').open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==len(logs)
        assert meta['normalization']=='per_image'
        record=dict(run=str(path.resolve()),suite=meta['suite'],inputs=len(logs),
                    png_count=len(list(path.rglob('*.png'))),npz_count=len(list(path.rglob('*.npz'))),
                    mat_count=len(list(path.rglob('*.mat'))),all_checks_passed=True)
        report.append(record)
        print(f'Audited {meta["suite"]}: {len(logs)} actual saved complex outputs',flush=True)
    if not report:
        raise ValueError('No completed result runs found')
    run=new_run(output_root,'output_audit')
    write_json(run/'audit.json',dict(runs=report,total_inputs=sum(r['inputs'] for r in report),
              total_png=sum(r['png_count'] for r in report),all_checks_passed=True))
    print(f'Audit saved: {run}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description='Replay normalization/noise/metrics against all NPZ/MAT outputs of completed all-save runs')
    p.add_argument('--results-root',required=True)
    p.add_argument('--output-root',default='local_validation')
    args=p.parse_args()
    audit(args.results_root,args.output_root)
