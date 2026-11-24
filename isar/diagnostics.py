"""Observe the author's unmodified hard frequency split by real forwards/backwards."""
import argparse
import time
import numpy as np
import torch
from .common import new_run, write_json, environment, setup_device
from .model import build_model, FULL_MODEL, padded_forward


def grad_state(model):
    result = {}
    for name, p in model.named_parameters():
        if name.startswith(('fre1.', 'fre2.', 'fre3.')):
            result[name] = ('none' if p.grad is None else 'nonzero' if torch.count_nonzero(p.grad) else 'zero')
    return result


def diagnose(output_root='local_validation', device='cpu', threads=4):
    device = setup_device(device, threads)
    run = new_run(output_root, 'full_model_diagnostics')
    torch.manual_seed(10)
    model = build_model(FULL_MODEL).to(device)
    report = dict(environment=environment(), config=FULL_MODEL,
                  parameters=sum(p.numel() for p in model.parameters()), cases=[])
    observations, handles = {}, []
    for name in ('fre1', 'fre2', 'fre3'):
        module = getattr(model, name)
        def conv_hook(m, inp, out, key=name):
            observations.setdefault(key, {})['feature_shape'] = list(out.shape)
        handles.append(module.conv1.register_forward_hook(conv_hook))
        def rate_hook(m, inp, out, key=name):
            shape = observations[key]['feature_shape']
            h, w = shape[-2:]
            threshold = out.detach().sigmoid()
            radii = [((h // 128 * t[0]).int().item(), (w // 128 * t[1]).int().item()) for t in threshold]
            low_ratios = [4 * rh * rw / (h * w) for rh, rw in radii]
            observations[key].update(threshold=threshold.cpu().flatten().tolist(), radii=radii,
                                     low_mask_nonzero_fraction=low_ratios,
                                     high_mask_nonzero_fraction=[1 - p for p in low_ratios])
        handles.append(module.rate_conv.register_forward_hook(rate_hook))
        original = module.fft
        def fft_probe(x, n=128, func=original, key=name):
            high, low = func(x, n)
            observations[key].update(raw_high_norm=high.detach().norm().item(),
                                     raw_low_norm=low.detach().norm().item(),
                                     raw_low_nonzero=int(torch.count_nonzero(low.detach())))
            return high, low
        module.fft = fft_probe
        def cross_hook(m, inp, out, key=name):
            observations[key]['post_low_cross_attention_norm'] = out.detach().norm().item()
        handles.append(module.channel_cross_h.register_forward_hook(cross_hook))
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for h, w in ((128, 128), (256, 256), (133, 197), (512, 512), (1024, 1024)):
        # 512/1024 probe only AFLB. A full 1024 model backward is unnecessary here.
        observations.clear()
        began = time.perf_counter()
        if h <= 256:
            with torch.no_grad():
                x = torch.randn(1, 2, h, w, device=device)
                out = padded_forward(model, x)
                assert out.shape == x.shape and torch.isfinite(out).all()
        else:
            with torch.no_grad():
                for name, div in (('fre1', 8), ('fre2', 4), ('fre3', 2)):
                    m = getattr(model, name)
                    x = torch.randn(1, 2, h // div, w // div, device=device)
                    # Direct fft is sufficient to observe original mask at decoder feature sizes.
                    m.fft(x)
        case = dict(input_shape=[h, w], scope='full_model' if h <= 256 else 'fft_only_at_decoder_sizes',
                    seconds=time.perf_counter()-began,
                    modules={k: dict(v) for k, v in observations.items()})
        report['cases'].append(case)
        print(f'diagnostic forward {h}x{w}: {case["seconds"]:.3f}s', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    report['backward'] = []
    for stage in range(2):
        began = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(1, 2, 128, 128, device=device)
        y = torch.randn_like(x)
        pred = model(x)
        loss = torch.nn.functional.l1_loss(pred, y)
        loss.backward()
        row = dict(stage='initial_para1_zero' if stage == 0 else 'after_one_optimizer_step',
                   loss=loss.item(), gradients=grad_state(model),
                   modules={k: dict(v) for k,v in observations.items()})
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, error_if_nonfinite=True)
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        row['seconds'] = time.perf_counter() - began
        report['backward'].append(row)
        print(f'diagnostic backward {stage}: {row["seconds"]:.3f}s, loss={loss.item():.6g}', flush=True)
    # RGB default shape, parameter count and finite signed output.
    from net.model import AdaIR
    with torch.no_grad():
        rgb = AdaIR().to(device)
        z = rgb(torch.randn(1, 3, 32, 40, device=device))
        report['rgb'] = dict(parameters=sum(p.numel() for p in rgb.parameters()), shape=list(z.shape),
                             finite=bool(torch.isfinite(z).all()))
    report['cuda_peak_allocated_mib'] = torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None
    report['cuda_memory_note'] = 'CPU-only run; AutoDL CUDA peak memory remains to be measured' if device.type=='cpu' else 'measured'
    # Isolate nonempty-mask autograd from the zero gate and small image degeneracy.
    from net.model import FreModule
    probe = FreModule(16, 1, False, in_dim=2).to(device)
    probe_x = torch.randn(1, 2, 512, 512, device=device)
    high, low = probe.fft(probe_x)
    (high.mean()+low.mean()).backward()
    report['nonempty_mask_fft_backward'] = dict(feature_shape=[512,512], raw_low_norm=low.detach().norm().item(),
        rate_conv_gradients={n: 'none' if p.grad is None else 'nonzero' if torch.count_nonzero(p.grad) else 'zero'
                             for n,p in probe.rate_conv.named_parameters()},
        conv1_gradient_norm=probe.conv1.weight.grad.norm().item())
    for handle in handles:
        handle.remove()
    write_json(run/'diagnostics.json', report)
    print(f'Diagnostics saved: {run}', flush=True)
    return run, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', default='local_validation')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    diagnose(args.output_root, args.device, args.threads)
