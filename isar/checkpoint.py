"""Append-only full step checkpoints. Pointers update only inside the new active run."""
from pathlib import Path
import json
import random
import numpy as np
import torch
from .common import write_json

KIND = 'adair-isar-complex2ch-v1'


def load_checkpoint(path):
    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix.lower()=='.json':
        pointer = json.loads(path.read_text(encoding='utf-8'))
        target = Path(pointer['checkpoint'])
        path = (target if target.is_absolute() else path.parent/target).resolve(strict=True)
    # Only load explicitly selected, trusted local experiment artifacts.
    c = torch.load(path, map_location='cpu', weights_only=False)
    if c.get('kind') != KIND:
        raise ValueError('Expected an AdaIR ISAR checkpoint; RGB/other experiment weights are incompatible')
    return path, c


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(run, model, optimizer, step, best, best_path, config, stats, manifest_sha, data_identity, improved=False, final=False):
    run = Path(run)
    path = run/f'step_{step:06d}.pt'
    if improved:
        best_path = str(path.resolve())
    c = dict(kind=KIND, model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
             best_score=best, best_checkpoint=best_path, config=config, stats=stats,
             manifest_sha256=manifest_sha, data_identity=data_identity, rng=rng_state())
    with path.open('xb') as f:
        torch.save(c, f)
    write_json(run/'latest.json', dict(checkpoint=path.name, step=step), exclusive=False)
    if best_path:
        pointer = Path(best_path).name if Path(best_path).parent == run.resolve() else best_path
        write_json(run/'best.json', dict(checkpoint=pointer, score=best), exclusive=False)
    if final:
        write_json(run/'final.json', dict(checkpoint=path.name, step=step))
    return best_path
