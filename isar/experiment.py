"""Step-based satellite-only training, preserving IDF's batch sampling RNG order."""
import argparse
from collections import defaultdict
import copy
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint, restore_rng, save_checkpoint
from .common import REPO, append_csv, environment, load_config, new_run, setup_device, write_json
from .data import MatReader, PreloadedMatReader, data_root
from .evaluation import evaluate
from .manifest import load_manifest, verify_data
from .model import build_model, FULL_MODEL
from .noise import add_noise
from .normalization import encode
from .visualization import training_curves


def training_batch(reader, groups, root, seed, step, t, stats):
    """Copied IDF f5ca181 sampling sequence. One target pool per effective batch."""
    targets = sorted(groups)
    rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
    pool = groups[targets[int(rng.integers(len(targets)))]]
    xs, ys = [], []
    for _ in range(t['batch_size']):
        r = pool[int(rng.integers(len(pool)))]
        clean = reader.frame(root / r['mat_path'], r['frame'])
        patch = t['patch_size']
        h, w = clean.shape
        if patch > min(h, w):
            raise ValueError('Patch exceeds frame size')
        i, j = rng.integers(h - patch + 1), rng.integers(w - patch + 1)
        clean = np.rot90(clean[i:i+patch, j:j+patch], int(rng.integers(4)))
        if rng.integers(2):
            clean = np.flip(clean, axis=0)
        snr = float(rng.uniform(*t['snr_range']))
        noisy, _ = add_noise(clean, snr, rng)
        xs.append(encode(noisy, stats))
        ys.append(encode(clean, stats))
    return np.stack(xs), np.stack(ys)


def optimizer_step(model, optimizer, xs, ys, t, stats, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, loss_value = len(xs), 0.0
    micro = t['micro_batch_size']
    for i in range(0, total, micro):
        # Keep only one microbatch on CUDA; sample the original effective batch once.
        x = torch.from_numpy(xs[i:i+micro]).to(device)
        y = torch.from_numpy(ys[i:i+micro]).to(device)
        pred = model(x)
        loss = F.l1_loss(pred, y)
        if t['amplitude_l1_weight']:
            mu = torch.as_tensor(stats['mean'], device=device, dtype=pred.dtype)[None,:,None,None]
            offset = mu / stats['std']
            loss = loss+t['amplitude_l1_weight']*F.l1_loss(
                torch.linalg.vector_norm(pred+offset, dim=1), torch.linalg.vector_norm(y+offset, dim=1))
        if not torch.isfinite(loss):
            raise ValueError('Non-finite loss')
        weight = len(x)/total
        (loss*weight).backward()
        loss_value += loss.item()*weight
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t['grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    return loss_value, float(grad_norm)


def check_config(config, smoke):
    t = config['train']
    for field in ('steps', 'batch_size', 'micro_batch_size', 'patch_size', 'validate_every', 'save_every', 'log_every'):
        if not isinstance(t[field], int) or t[field] <= 0:
            raise ValueError(f'{field} must be a positive integer')
    if t['patch_size'] % 8:
        raise ValueError('Training patch_size must be a multiple of 8')
    if t['micro_batch_size'] > t['batch_size']:
        raise ValueError('micro_batch_size exceeds effective batch size')
    if t['noise_mode']!='uniform' or t['snr_range']!=[-5,5]:
        raise ValueError('Reproduction uses uniform SNR [-5,5]')
    if config['validation']['selection_metric']!='restored_psnr':
        raise ValueError('Select best checkpoint by mean validation restored_psnr')
    if not smoke:
        default = load_config(REPO/'configs/isar/default.json')
        numerical = dict(t)
        numerical['micro_batch_size'] = default['train']['micro_batch_size']
        if (config['model']!=FULL_MODEL or numerical!=default['train']
                or config['validation']!=default['validation']
                or config['seed']!=10 or config['noise_seed']!=20261027):
            raise ValueError('Non-default/limited configurations require explicit --smoke; formal defaults are frozen')


def resume_compatible(c, config, stats, manifest_sha, identity):
    if c['manifest_sha256']!=manifest_sha or c['stats']!=stats or c['data_identity']!=identity:
        raise ValueError('Resume manifest, training statistics or training/validation data changed')
    old, new = copy.deepcopy(c['config']), copy.deepcopy(config)
    for d in (old, new):
        for key in ('output_root', 'data_root', 'manifest', 'training_stats', 'io'):
            d.pop(key, None)
        for key in ('steps', 'log_every', 'save_every', 'micro_batch_size'):
            d['train'].pop(key, None)
    if old!=new:
        raise ValueError('Resume numerical/validation configuration mismatch')


def train(args):
    device = setup_device(args.device, args.threads)
    config = load_config(args.config)
    smoke = bool(args.smoke)
    if config.get('smoke') and not smoke:
        raise ValueError('Smoke configuration requires --smoke')
    config['smoke'] = smoke
    for field in ('steps', 'batch_size', 'patch_size', 'micro_batch_size'):
        value = getattr(args, field)
        if value is not None:
            config['train'][field] = value
    if args.val_limit is not None:
        if not smoke or args.val_limit < 1:
            raise ValueError('--val-limit requires --smoke and a positive number')
        config['validation']['limit'] = args.val_limit
    if args.output_root:
        config['output_root'] = args.output_root
    manifest_path = args.manifest or config['manifest']
    config['manifest'] = str(Path(manifest_path).resolve())
    root = data_root(args.data_root or config.get('data_root'))
    config['data_root'] = str(root)
    check_config(config, smoke)
    m, manifest_sha = load_manifest(manifest_path, smoke)
    # No measured files, test images or unseen images are read for training or model selection.
    verified = verify_data(m, root, ('train', 'val'), compute_stats=True)
    stats = m['normalization']
    config['training_stats'] = stats
    identity = {k: v['sha256'] for k,v in verified['files'].items()}
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=device.type=='cuda')
    model = build_model(config['model']).to(device)
    t = config['train']
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    start, best, best_path = 0, -math.inf, None
    if args.resume:
        resume_path, c = load_checkpoint(args.resume)
        resume_compatible(c, config, stats, manifest_sha, identity)
        model.load_state_dict(c['model'], strict=True)
        optimizer.load_state_dict(c['optimizer'])
        restore_rng(c['rng'])
        start, best, best_path = c['step'], c['best_score'], c['best_checkpoint']
        if best_path and not Path(best_path).is_file():
            candidate = resume_path.parent/Path(best_path.replace('\\', '/')).name
            if not candidate.is_file():
                raise FileNotFoundError('Resume best checkpoint missing after relocation; retain/copy the historical best step file')
            best_path = str(candidate)
    if start >= t['steps']:
        raise ValueError('Requested total steps must exceed completed checkpoint step')
    run = new_run(config['output_root'], ('SMOKE_' if smoke else '')+args.label)
    write_json(run/'config.json', config)
    write_json(run/'run.json', dict(config=config, training_stats=stats, manifest_sha256=manifest_sha,
               data_verification=verified, resume=args.resume, start_step=start, device=str(device),
               environment=environment(), smoke=smoke, parameters=sum(p.numel() for p in model.parameters()),
               protocol='fixed LR; complex Re/Im L1; fixed clean-training stats; no pretrained weights; satellite only',
               checkpoint_policy='append-only full step files; latest/best/final JSON references; no pruning'))
    groups = defaultdict(list)
    for r in m['records']:
        if r['subset']=='train':
            groups[r['target']].append(r)
    val = [r for r in m['records'] if r['subset']=='val']
    if config['validation']['limit'] is not None:
        val = val[:config['validation']['limit']]
    reader = (PreloadedMatReader([root/rr[0]['mat_path'] for rr in groups.values()], args.preload_mib)
              if args.preload_train else MatReader())
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    began = time.perf_counter()
    step_times = []
    last_log = began
    for step in range(start+1, t['steps']+1):
        tick = time.perf_counter()
        xs, ys = training_batch(reader, groups, root, seed, step, t, stats)
        loaded = time.perf_counter()
        loss, grad_norm = optimizer_step(model, optimizer, xs, ys, t, stats, device)
        if device.type=='cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter()-loaded
        step_times.append(elapsed)
        if step==start+1 or step%t['log_every']==0 or time.perf_counter()-last_log>=30:
            peak = torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None
            append_csv(run/'train.csv', dict(step=step, lr=optimizer.param_groups[0]['lr'], loss=loss,
                       grad_norm_before_clip=grad_norm, data_seconds=loaded-tick,
                       optimizer_step_seconds=elapsed, cuda_peak_allocated_mib=peak))
            print(f'step={step}/{t["steps"]} loss={loss:.7g} step_seconds={elapsed:.3f} cuda_peak_MiB={peak}', flush=True)
            last_log = time.perf_counter()
        improved = False
        if step%t['validate_every']==0 or step==t['steps']:
            valout = run/f'validation_{step:06d}'
            print(f'Validation step {step}: {len(val)} frames x {config["validation"]["snrs"]}, fixed_train', flush=True)
            rows, _ = evaluate(model, val, root, stats, config['validation']['snrs'], config['noise_seed'],
                               device, valout, normalization='fixed_train', save_mode='none', visuals=False, smoke=smoke)
            avg = {k: float(np.mean([r['restored_'+k] for r in rows])) for k in ('psnr','ssim','nmse','nmse_db')}
            append_csv(run/'validation.csv', dict(step=step, **{'restored_'+k:v for k,v in avg.items()}))
            improved = avg['psnr'] > best
            best = max(best, avg['psnr'])
            print(f'validation step={step} restored_psnr={avg["psnr"]:.7g} best={best:.7g}', flush=True)
        if improved or step%t['save_every']==0 or step==t['steps']:
            best_path = save_checkpoint(run, model, optimizer, step, best, best_path, config, stats,
                                       manifest_sha, identity, improved, step==t['steps'])
    training_curves(run)
    write_json(run/'completed.json', dict(status='complete', step=t['steps'], best_score=best,
               best_checkpoint=best_path, mean_optimizer_step_seconds=float(np.mean(step_times)),
               total_seconds=time.perf_counter()-began,
               cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None,
               cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2 if device.type=='cuda' else None))
    with (run/'SUMMARY.md').open('x', encoding='utf-8') as f:
        f.write(f'# {"SMOKE ONLY — " if smoke else ""}AdaIR ISAR training\n\n'
                f'Completed optimizer steps {start+1}..{t["steps"]}; effective batch {t["batch_size"]}; microbatch {t["micro_batch_size"]}.\n\n'
                f'Best validation mean restored PSNR: {best:.8g}. Training/validation normalization: fixed_train.\n\n'
                'latest.json, best.json and final.json point to full checkpoints. No old checkpoint is deleted.\n')
    print(f'Training saved: {run}', flush=True)
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(REPO/'configs/isar/default.json'))
    p.add_argument('--manifest')
    p.add_argument('--data-root')
    p.add_argument('--output-root')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--resume')
    p.add_argument('--steps', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--micro-batch-size', type=int)
    p.add_argument('--patch-size', type=int)
    p.add_argument('--val-limit', type=int)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--label', default='train')
    p.add_argument('--preload-train', action='store_true')
    p.add_argument('--preload-mib', type=int, default=1536)
    train(p.parse_args())
