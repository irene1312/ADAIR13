import argparse
from pathlib import Path
import numpy as np
import torch
import h5py
from isar.checkpoint import load_checkpoint
from isar.common import load_config
from isar.experiment import train
from isar.newyak import read_newyak


def test_checkpoint_resume_matches_uninterrupted(synthetic):
    base = dict(config=str(synthetic/'config.json'), data_root=None, manifest=None, output_root=None,
                device='cpu', threads=2, resume=None, steps=None, batch_size=None, micro_batch_size=None,
                patch_size=None, val_limit=None, smoke=True, label='integration', preload_train=False, preload_mib=1536)
    first = train(argparse.Namespace(**dict(base,steps=1)))
    resumed = train(argparse.Namespace(**dict(base,steps=2,resume=str(first/'latest.json'))))
    continuous = train(argparse.Namespace(**dict(base,steps=2)))
    _, a = load_checkpoint(resumed/'latest.json')
    _, b = load_checkpoint(continuous/'final.json')
    assert a['step']==b['step']==2 and a['best_score']==b['best_score']
    for key in a['model']: torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    for state_a,state_b in zip(a['optimizer']['state'].values(),b['optimizer']['state'].values()):
        for k in state_a:
            if torch.is_tensor(state_a[k]): torch.testing.assert_close(state_a[k],state_b[k],rtol=0,atol=0)
            else: assert state_a[k]==state_b[k]
    assert all(k in a for k in ('optimizer','config','stats','manifest_sha256','rng','data_identity','best_checkpoint'))
    assert (first/'step_000001.pt').exists() and (resumed/'step_000002.pt').exists()
    print(f'Retained integration runs: {first}, {resumed}, {continuous}')


def test_newyak_v73_struct_reference(retained_workspace):
    path = retained_workspace/'newyak_struct_v73.mat'
    rng=np.random.default_rng(32)
    z=(rng.normal(size=(128,128))+1j*rng.normal(size=(128,128))).astype(np.complex128)
    a=np.empty(z.T.shape,dtype=[('real','<f8'),('imag','<f8')])
    a['real'],a['imag']=z.T.real,z.T.imag
    with h5py.File(path,'x') as f:
        target=f.create_dataset('image',data=a)
        group=f.create_group('data_test')
        d=group.create_dataset('im_ori',shape=(1,1),dtype=h5py.ref_dtype)
        d[0,0]=target.ref
    np.testing.assert_array_equal(read_newyak(path),z)
