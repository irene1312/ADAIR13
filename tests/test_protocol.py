import ast
import copy
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest
import torch
from scipy.io import savemat, loadmat
import h5py

from isar.common import REPO, load_config, new_run
from isar.data import MatReader, stable_seed, normalization, channels, as_complex
from isar.evaluation import aggregate, evaluate
from isar.experiment import training_batch, optimizer_step
from isar.manifest import load_manifest, validate_manifest, verify_data
from isar.metrics import metrics
from isar.model import build_model, padded_forward
from isar.noise import add_noise
from isar.normalization import encode, decode, input_stats
from isar.testing import parser, measured_records, SUITES
from isar.synthetic import create_synthetic

torch.set_num_threads(2)
IDF = Path('D:/Research/IDF/presentation-v1')
REF = 'f5ca1811af03c4f7a3c37a5f13040a6b50d8eca6'


def ref_namespace(name):
    if not IDF.is_dir():
        pytest.skip('Exact local IDF reference worktree is unavailable')
    head = subprocess.check_output(['git','-c',f'safe.directory={IDF.as_posix()}','-C',str(IDF),'rev-parse','HEAD'], text=True).strip()
    assert head==REF
    namespace = {}
    exec(compile((IDF/name).read_text(encoding='utf-8'), str(IDF/name), 'exec'), namespace)
    return namespace


def reference_functions(names, namespace):
    source = ast.parse((IDF/'isar/experiment.py').read_text(encoding='utf-8'))
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<IDF-reference-functions>', 'exec'), namespace)
    return namespace


def test_manifest_exact_intervals_and_no_leakage():
    m, _ = load_manifest(REPO/'manifests/isar_sim.json')
    if IDF.exists():
        assert (REPO/'manifests/isar_sim.json').read_bytes().replace(b'\r\n',b'\n')==(IDF/'manifests/isar_sim.json').read_bytes().replace(b'\r\n',b'\n')
    for target in ('Aqua','HST','Beidou'):
        for subset, frames in (('train', range(21,241)), ('val',range(261,301)), ('test_seen',range(321,361))):
            assert [r['frame'] for r in m['records'] if r['target']==target and r['subset']==subset]==list(frames)
    for t in ('CALIPSO','WV2'):
        assert len([r for r in m['records'] if r['target']==t and r['subset']=='test_unseen'])==360
    bad = copy.deepcopy(m)
    bad['records'][0]['subset']='train'
    with pytest.raises(ValueError): validate_manifest(bad)
    bad = copy.deepcopy(m)
    bad['records'].append(copy.deepcopy(bad['records'][0]))
    with pytest.raises(ValueError): validate_manifest(bad)


@pytest.mark.parametrize('hdf', [False, True])
def test_mat_axis_frame_dtype_and_channels(retained_workspace, hdf):
    z = (np.arange(5*7*3).reshape(5,7,3)+1j*np.arange(5*7*3).reshape(5,7,3)*3).astype(np.complex128)
    path = retained_workspace/f'orientation_{hdf}.mat'
    if hdf:
        raw = z.transpose(2,1,0)
        a = np.empty(raw.shape, dtype=[('real','<f8'),('imag','<f8')])
        a['real'],a['imag']=raw.real,raw.imag
        with h5py.File(path,'x') as f: f.create_dataset('target_zong',data=a)
    else:
        with path.open('xb') as f: savemat(f,{'target_zong':z})
    reader = MatReader()
    for i in range(1,4):
        actual=reader.frame(path,i)
        np.testing.assert_array_equal(actual,z[:,:,i-1])
        np.testing.assert_array_equal(channels(actual)[0],actual.real.astype(np.float32))
        np.testing.assert_array_equal(channels(actual)[1],actual.imag.astype(np.float32))
    for i in (0,4):
        with pytest.raises(IndexError): reader.frame(path,i)


def test_invalid_mat_rejected(retained_workspace):
    for dtype in (np.float64, np.complex64):
        path=retained_workspace/f'invalid_{dtype.__name__}.mat'
        with path.open('xb') as f: savemat(f,{'target_zong':np.ones((8,8,2),dtype=dtype)})
        with pytest.raises(ValueError): MatReader().frame(path,1)


def test_noise_and_stable_seed_exact_idf():
    noise_ref=ref_namespace('isar/noise.py')['add_noise']
    data_ref=ref_namespace('isar/data.py')
    z=np.random.default_rng(42).normal(size=(31,29))+1j*np.random.default_rng(4).normal(size=(31,29))
    for snr in (-5,0,5):
        seed=stable_seed(20261027,'Aqua',321,snr)
        assert seed==data_ref['stable_seed'](20261027,'Aqua',321,snr)
        actual, sampled=add_noise(z,snr,np.random.default_rng(seed))
        expected,sampled_ref=noise_ref(z,snr,np.random.default_rng(seed))
        np.testing.assert_array_equal(actual,expected)
        assert sampled==sampled_ref and sampled!=snr
        rng=np.random.default_rng(seed)
        sigma=np.sqrt(np.mean(np.abs(z)**2)/(2*10**(snr/10)))
        np.testing.assert_array_equal(actual,z+sigma*(rng.standard_normal(z.shape)+1j*rng.standard_normal(z.shape)))


def test_per_image_stats_encode_decode_exact_idf():
    reference=ref_namespace('isar/test_normalization.py')
    boundary=reference_functions({'encode','decode'},{'np':np,'channels':channels,'as_complex':as_complex})
    training=dict(mean=[.3,-.8],std=1.4,amplitude_scale=7.0)
    a=np.random.default_rng(12).normal(size=(13,19))+1j*np.random.default_rng(13).normal(size=(13,19))
    for z in (a, 4*a+20-8j):
        stats=input_stats(z,training)
        assert stats['mean']==reference['input_stats'](z,training,'per_image')['mean']
        assert stats['std']==reference['input_stats'](z,training,'per_image')['std']
        assert stats['amplitude_scale']==7.0
        np.testing.assert_array_equal(encode(z,stats),boundary['encode'](z,stats))
        np.testing.assert_array_equal(decode(encode(z,stats),stats),boundary['decode'](encode(z,stats),stats))
        np.testing.assert_allclose(decode(encode(z,stats),stats),z,rtol=3e-6,atol=3e-6)
        assert encode(z,stats).dtype==np.float32
        assert decode(encode(z,stats),stats).dtype==np.complex128
    s1,s2=input_stats(a,training),input_stats(4*a+20-8j,training)
    assert s1['mean']!=s2['mean'] and s1['std']!=s2['std']
    assert input_stats(a,training,'fixed_train')==training
    noisy,_=add_noise(a,-5,np.random.default_rng(123))
    assert input_stats(noisy,training)['mean']!=input_stats(a,training)['mean']
    for bad in (np.ones((9,9),dtype=np.complex128),a*np.nan,np.zeros((2,3,4),dtype=np.complex128)):
        with pytest.raises(ValueError): input_stats(bad,training)


def test_train_statistics_and_sampling_exact_idf(synthetic):
    m,_=load_manifest(synthetic/'manifest.json',True)
    actual=normalization(m['records'],synthetic/'satellite')
    reference=ref_namespace('isar/data.py')['normalization'](m['records'],synthetic/'satellite')
    assert actual==reference==m['normalization']
    groups={t:[r for r in m['records'] if r['target']==t and r['subset']=='train'] for t in ('Aqua','HST','Beidou')}
    t=load_config(synthetic/'config.json')['train']
    ns=reference_functions({'training_batch','encode'},{'np':np,'channels':channels,'add_noise':add_noise})
    for step in (1,2,1000):
        a=training_batch(MatReader(),groups,synthetic/'satellite',10,step,t,actual)
        b=ns['training_batch'](MatReader(),groups,synthetic/'satellite',10,step,t,actual)
        for aa,bb in zip(a,b): np.testing.assert_array_equal(aa,bb)


def test_metrics_exact_and_db_aggregation():
    fn=ref_namespace('isar/metrics.py')['metrics']
    rng=np.random.default_rng(99)
    clean=rng.standard_normal((16,19))+1j*rng.standard_normal((16,19))
    noisy,_=add_noise(clean,0,rng)
    assert metrics(clean,noisy,3)==fn(clean,noisy,3)
    row=dict(target='Aqua',snr_db=0)
    r1={**row,**{'noisy_'+k:v for k,v in metrics(clean,noisy,3).items()},**{'restored_'+k:v for k,v in metrics(clean,noisy*.5,3).items()}}
    r2=copy.deepcopy(r1)
    r2['restored_nmse']*=10
    r2['restored_nmse_db']+=10
    avg=aggregate([r1,r2])[0]
    assert avg['restored_nmse_db']!=avg['restored_nmse_db_of_mean']


def test_shape_residual_and_fre_channels():
    from net.model import AdaIR
    cfg=load_config(REPO/'configs/isar/smoke.json')['model']
    model=build_model(cfg)
    assert model.patch_embed.proj.in_channels==2 and model.output.out_channels==2
    for name in ('fre1','fre2','fre3'):
        assert getattr(model,name).conv.in_channels==getattr(model,name).conv1.in_channels==2
    torch.nn.init.zeros_(model.output.weight)
    x=torch.randn(1,2,35,51)
    with torch.no_grad(): np.testing.assert_array_equal(padded_forward(model,x).numpy(),x.numpy())
    assert (padded_forward(model,x)<0).any()
    rgb_cfg=dict(cfg,inp_channels=3,out_channels=3)
    rgb=AdaIR(**rgb_cfg)
    assert rgb(torch.randn(1,3,32,40)).shape==(1,3,32,40)


def test_rgb_matches_unmodified_author_model():
    from net.model import AdaIR
    source=subprocess.check_output(['git','-c',f'safe.directory={REPO.as_posix()}','-C',str(REPO),'show','ccb8b98:net/model.py'],text=True,encoding='utf-8')
    namespace={}
    exec(compile(source,'<author-original>','exec'),namespace)
    cfg=load_config(REPO/'configs/isar/smoke.json')['model']
    cfg=dict(cfg,inp_channels=3,out_channels=3)
    torch.manual_seed(10)
    original=namespace['AdaIR'](**cfg)
    torch.manual_seed(10)
    current=AdaIR(**cfg)
    x=torch.randn(1,3,32,40)
    with torch.no_grad():
        torch.testing.assert_close(original(x),current(x),rtol=0,atol=0)


def test_gradient_accumulation_preserves_effective_batch():
    cfg=load_config(REPO/'configs/isar/smoke.json')
    a=build_model(cfg['model'])
    b=copy.deepcopy(a)
    rng=np.random.default_rng(14)
    x,y=[rng.standard_normal((3,2,16,16)).astype(np.float32) for _ in range(2)]
    t=dict(cfg['train'],micro_batch_size=3)
    oa=torch.optim.AdamW(a.parameters(),lr=1e-4,weight_decay=1e-4)
    ob=torch.optim.AdamW(b.parameters(),lr=1e-4,weight_decay=1e-4)
    stats=dict(mean=[0,0],std=1,amplitude_scale=3)
    la,_=optimizer_step(a,oa,x,y,t,stats,torch.device('cpu'))
    lb,_=optimizer_step(b,ob,x,y,dict(t,micro_batch_size=2),stats,torch.device('cpu'))
    assert abs(la-lb)<1e-6
    for pa,pb in zip(a.parameters(),b.parameters()): torch.testing.assert_close(pa,pb,atol=2e-6,rtol=1e-4)


def test_six_entrypoints_and_test_defaults():
    assert len(SUITES)==6
    for suite in SUITES:
        args=parser().parse_args(['--suite',suite,'--checkpoint','explicit.pt'])
        assert args.normalization=='per_image' and args.save_complex=='all' and args.limit is None


def test_measured_discovery_never_reads_train(synthetic):
    with (synthetic/'aircraft'/'An26_Train.mat').open('xb') as f:
        f.write(b'this must never be read')
    root,rows,files=measured_records(synthetic/'aircraft')
    assert len(rows)==9 and all(r['mat_path'].endswith('_Test.mat') for r in rows)
    root,rows,files=measured_records(synthetic/'newyak',True)
    assert len(rows)==2 and all(r['reader']=='newyak' for r in rows)


def test_raw_no_noise_per_image_log_and_complex_export(synthetic,retained_workspace,monkeypatch):
    import isar.evaluation as ev
    def forbidden(*args,**kwargs): raise AssertionError('raw must not call noise')
    monkeypatch.setattr(ev,'add_noise',forbidden)
    root,rows,_=measured_records(synthetic/'aircraft')
    rows=[r for r in rows if r['target']=='An26'][:2]
    model=build_model(load_config(REPO/'configs/isar/smoke.json')['model'])
    stats=dict(mean=[0,0],std=1,amplitude_scale=3)
    out=new_run(retained_workspace,'raw')
    a,_=evaluate(model,rows,root,stats,[-5,0,5],20261027,torch.device('cpu'),out,raw=True,satellite=False,
                 visuals=False,export_mat=True,smoke=True)
    assert len(a)==2 and not any('psnr' in k or 'nmse' in k or 'ssim' in k for k in a[0])
    logs=[json.loads(s) for s in (out/'normalization_stats.jsonl').read_text().splitlines()]
    assert all(r['original_sha256']==r['input_sha256'] and r['noise_seed'] is None for r in logs)
    assert logs[0]['shared_std']!=logs[1]['shared_std']
    npz=next((out/'complex').glob('*.npz'))
    with np.load(npz) as f:
        assert f['restored'].dtype==np.complex128
        np.testing.assert_array_equal(f['restored'].real,f['restored_real'])
        np.testing.assert_array_equal(f['restored'].imag,f['restored_imag'])
        np.testing.assert_array_equal(f['restored_phase'],np.angle(f['restored']))
        expected=f['restored'].copy()
    np.testing.assert_array_equal(loadmat(npz.with_suffix('.mat'))['restored'],expected)


def test_unique_run_never_overwrites(retained_workspace):
    a,b=new_run(retained_workspace,'same'),new_run(retained_workspace,'same')
    assert a!=b and a.is_dir() and b.is_dir()


def test_visualization_shared_physical_scales(retained_workspace,monkeypatch):
    import isar.visualization as vis
    seen=[]
    def capture(path,panels,scale,title): seen.append((Path(path),panels,scale,title))
    monkeypatch.setattr(vis,'draw_grid',capture)
    z=np.ones((9,9),dtype=np.complex128)*(2+1j)
    panels=[('Raw Input',z),('AdaIR Restored',z*.1)]
    vis.save_comparison(retained_workspace,'frame',panels,7,z,'SMOKE')
    assert [s[2] for s in seen]==[7,7,7,float(np.abs(z).max()),float(np.abs(z).max()),float(np.abs(z).max())]
    assert seen[0][1] is panels and seen[3][1] is panels


def test_manifest_shape_and_statistics_mismatch_errors(synthetic):
    m,_=load_manifest(synthetic/'manifest.json',True)
    bad=copy.deepcopy(m)
    bad['files']['Aqua']['shape'][0]=256
    with pytest.raises(ValueError, match='shape differs'): verify_data(bad,synthetic/'satellite',('train',),False)
    bad=copy.deepcopy(m)
    bad['normalization']['std']*=2
    with pytest.raises(ValueError, match='statistics'): verify_data(bad,synthetic/'satellite',('train',),True)
