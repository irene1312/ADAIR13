from collections import defaultdict
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.io import savemat
from .common import safe_json, write_json, write_csv
from .data import MatReader, stable_seed
from .metrics import metrics
from .model import restore
from .newyak import read_newyak
from .noise import add_noise
from .normalization import input_stats, array_sha, encode
from .visualization import selected_frames, save_comparison, metric_curves


def aggregate(rows, satellite=True):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['target'], row['snr_db'])].append(row)
        groups[('ALL', row['snr_db'])].append(row)
    fields = ([p+'_'+m for p in ('noisy', 'restored') for m in ('psnr', 'ssim', 'nmse', 'nmse_db')]
              if satellite else ['energy_ratio', 'input_peak', 'restored_peak', 'input_rms', 'restored_rms'])
    result = []
    for (target, snr), group in groups.items():
        r = dict(target=target, snr_db=snr, count=len(group))
        r.update({k: float(np.mean([row[k] for row in group])) for k in fields})
        if satellite:
            for prefix in ('noisy', 'restored'):
                n = r[prefix+'_nmse']
                r[prefix+'_nmse_db_of_mean'] = -math.inf if n==0 else 10*math.log10(n)
        result.append(r)
    return result


def save_complex(path, original, actual_input, restored, export_mat=False, satellite=True):
    arrays = {('reference' if satellite else 'original_measured'): original.astype(np.complex128),
              'input': actual_input.astype(np.complex128), 'restored': restored.astype(np.complex128),
              'restored_real': restored.real, 'restored_imag': restored.imag, 'restored_phase': np.angle(restored)}
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **arrays)
    if export_mat:
        with Path(path).with_suffix('.mat').open('xb') as f:
            savemat(f, arrays, do_compression=True)


def evaluate(model, records, root, training_stats, snrs, noise_seed, device, output,
             normalization='per_image', raw=False, satellite=True, save_mode='all',
             export_mat=False, visuals=True, smoke=False, progress_every=10):
    if not records:
        raise ValueError('Empty evaluation subset')
    if raw and satellite:
        raise ValueError('Satellite evaluation requires declared artificial noise')
    if save_mode not in ('all', 'selected', 'none'):
        raise ValueError('Invalid complex save mode')
    model.eval()
    output, root = Path(output), Path(root)
    output.mkdir(parents=True, exist_ok=True)
    chosen = selected_frames(records)
    reader = MatReader()
    rows, count = [], 0
    snrs = ['raw'] if raw else snrs
    total = len(records)*len(snrs)
    began = time.perf_counter()
    last_progress = began
    with (output/'normalization_stats.jsonl').open('x', encoding='utf-8') as log:
        for record in records:
            original = (read_newyak(root/record['mat_path']) if record.get('reader')=='newyak'
                        else reader.frame(root/record['mat_path'], record['frame']))
            for snr in snrs:
                seed, actual = None, None
                if raw:
                    actual_input = original.copy()  # raw path never calls add_noise
                else:
                    seed = stable_seed(noise_seed, record['target'], record['frame'], snr)
                    actual_input, actual = add_noise(original, snr, np.random.default_rng(seed))
                stats = input_stats(actual_input, training_stats, normalization)
                restored = restore(model, actual_input, stats, device)
                row = dict(target=record['target'], mat_path=record['mat_path'], frame=record['frame'],
                           subset=record['subset'], snr_db=snr, noise_seed=seed, actual_snr_db=actual)
                energy_in = float(np.sum(np.abs(actual_input)**2))
                if energy_in <= 0 or not np.isfinite(energy_in):
                    raise ValueError('Invalid actual input energy')
                if satellite:
                    for prefix, z in (('noisy', actual_input), ('restored', restored)):
                        row.update({prefix+'_'+k: v for k,v in metrics(original, z, training_stats['amplitude_scale']).items()})
                else:
                    row.update(energy_ratio=float(np.sum(np.abs(restored)**2)/energy_in),
                               input_peak=float(np.abs(actual_input).max()), restored_peak=float(np.abs(restored).max()),
                               input_rms=float(np.sqrt(np.mean(np.abs(actual_input)**2))),
                               restored_rms=float(np.sqrt(np.mean(np.abs(restored)**2))))
                info = dict(row, normalization_mode=normalization, input_sha256=array_sha(actual_input),
                            original_sha256=array_sha(original), input_shape=list(actual_input.shape),
                            padded_shape=[int(math.ceil(s/8)*8) for s in actual_input.shape],
                            input_dtype=str(actual_input.dtype), output_dtype=str(restored.dtype),
                            encoded_input_sha256=array_sha(encode(actual_input, stats)),
                            mean_real=stats['mean'][0], mean_imag=stats['mean'][1], shared_std=stats['std'],
                            amplitude_scale=training_stats['amplitude_scale'], raw_no_added_noise=raw,
                            input_peak_display_scale=float(np.abs(actual_input).max()))
                log.write(json.dumps(safe_json(info), allow_nan=False)+'\n')
                log.flush()
                rows.append(row)
                selected = (record['target'], record['mat_path'], record['frame']) in chosen
                stem = f'{record["target"]}_f{record["frame"]:04d}_snr{snr}'
                if save_mode=='all' or (save_mode=='selected' and selected):
                    (output/'complex').mkdir(exist_ok=True)
                    save_complex(output/'complex'/f'{stem}.npz', original, actual_input, restored, export_mat, satellite)
                if visuals and selected:
                    panels = ([('Raw Input', actual_input), ('AdaIR Restored', restored)] if raw else
                              [('Reference' if satellite else 'Original Reference', original),
                               ('Noisy', actual_input), ('AdaIR Restored', restored)])
                    title = f'{record["target"]} frame {record["frame"]} | '+('raw measured' if raw else f'SNR {snr} dB')
                    if smoke:
                        title = 'SMOKE ONLY / NOT A FORMAL RESULT | '+title
                    save_comparison(output/'images', stem, panels, training_stats['amplitude_scale'], actual_input, title)
                count += 1
                if count==1 or count % progress_every==0 or count==total or time.perf_counter()-last_progress>=30:
                    print(f'evaluate {count}/{total} {record["target"]} frame={record["frame"]} snr={snr} elapsed={time.perf_counter()-began:.1f}s', flush=True)
                    last_progress = time.perf_counter()
    summary = aggregate(rows, satellite)
    write_csv(output/'per_image.csv', rows)
    write_csv(output/'summary.csv', summary)
    write_json(output/'results.json', dict(summary=summary, per_image=rows, smoke=smoke,
        metric_protocol=('physical complex outputs; amplitude PSNR/SSIM with frozen training scale and metric-only clipping; '
                         'arithmetic per-image NMSE(dB) and separate dB(mean NMSE)' if satellite else
                         'measured data have no ground truth: energy/peak/RMS diagnostics only'),
        visualization='hot, nearest; shared training scale and actual input peak scale; no per-panel fitting'))
    if visuals:
        metric_curves(summary, output/'curves', satellite)
    with (output/'SUMMARY.md').open('x', encoding='utf-8') as f:
        f.write(f'# {"SMOKE ONLY — " if smoke else ""}AdaIR ISAR evaluation\n\n'
                f'{len(records)} original frames; {len(rows)} inputs. Normalization: `{normalization}`.\n\n'
                f'{"Raw measured inputs, no artificial noise." if raw else "Independent complex Gaussian artificial noise, IDF stable_seed."}\n\n'
                f'{"Full-complex NMSE; amplitude PSNR/SSIM use the frozen training amplitude scale." if satellite else "No PSNR, SSIM or NMSE: measured originals are not noise-free truth. Energy ratio is auxiliary, not accuracy."}\n\n'
                'See per_image.csv, summary.csv, normalization_stats.jsonl, images/, curves/ and complex/.\n')
    return rows, summary
