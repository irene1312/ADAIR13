"""Physical amplitude, fixed shared norms. Display never feeds back into inference."""
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def selected_frames(records, count=3):
    groups = defaultdict(list)
    for r in records:
        groups[r['target']].append((r['mat_path'], r['frame']))
    chosen = set()
    for target, frames in groups.items():
        frames = sorted(set(frames))
        for i in np.linspace(0, len(frames)-1, min(count, len(frames)), dtype=int):
            chosen.add((target, *frames[i]))
    return chosen


def draw_grid(path, panels, scale, title):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid display scale')
    fig, axes = plt.subplots(1, len(panels), squeeze=False,
                             figsize=(3.3*len(panels), 3.9), layout='constrained')
    for ax, (label, z) in zip(axes[0], panels):
        im = ax.imshow(np.abs(z), cmap='hot', vmin=0, vmax=scale, interpolation='nearest')
        ax.set_title(label)
        ax.axis('off')
    fig.suptitle(title, fontsize=10)
    fig.colorbar(im, ax=list(axes.flat), label='Physical complex amplitude', shrink=.8)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_comparison(root, stem, panels, training_scale, actual_input, title):
    for mode, scale in (('training_scale', training_scale), ('input_peak_scale', float(np.abs(actual_input).max()))):
        folder = Path(root) / mode
        folder.mkdir(parents=True, exist_ok=True)
        caption = f'{title}\n{mode}; shared 0..{scale:.6g}'
        draw_grid(folder/f'{stem}_comparison.png', panels, scale, caption)
        for label, z in panels:
            slug = label.lower().replace(' ', '_')
            draw_grid(folder/f'{stem}_{slug}.png', [(label, z)], scale, caption)


def metric_curves(summary, root, satellite=True):
    folder = Path(root)
    folder.mkdir(parents=True, exist_ok=True)
    targets = sorted({r['target'] for r in summary})
    fields = ('psnr', 'ssim', 'nmse', 'nmse_db') if satellite else ('energy_ratio', 'input_rms', 'restored_rms')
    for field in fields:
        fig, axes = plt.subplots(1, len(targets), squeeze=False,
                                 figsize=(3.6*len(targets), 3.5), layout='constrained')
        for target, ax in zip(targets, axes[0]):
            rows = sorted((r for r in summary if r['target']==target), key=lambda r: -999 if r['snr_db']=='raw' else float(r['snr_db']))
            x = [i if r['snr_db']=='raw' else r['snr_db'] for i,r in enumerate(rows)]
            if satellite:
                for prefix, label in (('noisy', 'Noisy'), ('restored', 'AdaIR Restored')):
                    ax.plot(x, [r[f'{prefix}_{field}'] for r in rows], 'o-', label=label)
            else:
                ax.plot(x, [r[field] for r in rows], 'o-', label='Auxiliary diagnostic')
                if rows[0]['snr_db']=='raw':
                    ax.set_xticks(x, ['Raw'])
            ax.set_title(target)
            ax.set_xlabel('Input SNR (dB)' if rows[0]['snr_db']!='raw' else 'Measured input')
            ax.set_ylabel(field + (' (mean of per-image dB)' if field=='nmse_db' else ''))
            ax.grid(alpha=.3)
            ax.legend(fontsize=7)
        fig.savefig(folder/f'{field}_snr.png', dpi=120)
        plt.close(fig)


def training_curves(run):
    run = Path(run)
    for name, fields, xlabel in (('train', ('loss',), 'Optimizer step'),
                                 ('validation', ('restored_psnr', 'restored_ssim', 'restored_nmse', 'restored_nmse_db'), 'Optimizer step')):
        path = run/f'{name}.csv'
        if not path.exists():
            continue
        with path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        for field in fields:
            fig, ax = plt.subplots(figsize=(6, 3.5), layout='constrained')
            ax.plot([int(r['step']) for r in rows], [float(r[field]) for r in rows], 'o-')
            ax.set(xlabel=xlabel, ylabel=field)
            ax.grid(alpha=.3)
            fig.savefig(run/f'{name}_{field}_step.png', dpi=120)
            plt.close(fig)
