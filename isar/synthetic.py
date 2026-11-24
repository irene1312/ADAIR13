"""Explicitly labelled small synthetic MAT fixtures, never scientific ISAR data."""
from pathlib import Path
import numpy as np
from scipy.io import savemat
import h5py
from .common import new_run, write_json, load_config, REPO
from .data import normalization


def create_synthetic(root='local_validation'):
    run = new_run(root, 'SYNTHETIC_SMOKE_DATA')
    sim, aircraft, newyak = (run/name for name in ('satellite', 'aircraft', 'newyak'))
    for p in (sim, aircraft, newyak):
        p.mkdir()
    rng = np.random.default_rng(1234)
    records, files = [], {}
    for i, target in enumerate(('Aqua', 'HST', 'Beidou', 'CALIPSO', 'WV2')):
        yy, xx = np.mgrid[:128, :128]
        envelope = .01 + np.exp(-((yy-45-i*3)**2+(xx-60)**2)/40)
        a = (envelope[:,:,None] * (rng.standard_normal((128,128,4))+1j*rng.standard_normal((128,128,4)))).astype(np.complex128)
        filename = target+'.mat'
        if i%2:
            compound = np.empty(a.transpose(2,1,0).shape, dtype=[('real','<f8'),('imag','<f8')])
            compound['real'], compound['imag'] = a.transpose(2,1,0).real, a.transpose(2,1,0).imag
            with h5py.File(sim/filename, 'x') as f:
                f.create_dataset('target_zong', data=compound)
            fmt = 'mat73'
        else:
            with (sim/filename).open('xb') as f:
                savemat(f, {'target_zong': a}, do_compression=True)
            fmt = 'scipy'
        files[target] = dict(path=filename, shape=list(a.shape), format=fmt, complex=True)
        for frame in range(1,5):
            subset = ('train' if frame<=2 else 'val' if frame==3 else 'test_seen') if i<3 else 'test_unseen'
            records.append(dict(target=target,mat_path=filename,frame=frame,subset=subset))
    m = dict(synthetic_smoke=True, version=1, frame_index_base=1, records=records, files=files)
    m['normalization'] = normalization(records, sim)
    write_json(run/'manifest.json', m)
    for target in ('An26', 'Yak', 'JZ'):
        z = (rng.standard_normal((128,128,3))+1j*rng.standard_normal((128,128,3))).astype(np.complex128)
        with (aircraft/(target+'_Test.mat')).open('xb') as f:
            savemat(f, {'target_zong': z}, do_compression=True)
    for name in ('yake1', 'yake2'):
        z = (rng.standard_normal((128,128))+1j*rng.standard_normal((128,128))).astype(np.complex128)
        with (newyak/(name+'.mat')).open('xb') as f:
            savemat(f, {'data_test': {'im_ori': z}}, do_compression=True)
    config = load_config(REPO/'configs/isar/smoke.json')
    config['manifest'] = str(run/'manifest.json')
    config['data_root'] = str(sim)
    config['output_root'] = str(run/'experiments')
    config['train'].update(patch_size=32, batch_size=4, micro_batch_size=2)
    write_json(run/'config.json', config)
    return run
