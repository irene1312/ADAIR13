import numpy as np
import torch
import torch.nn.functional as F
from net.model import AdaIR
from .normalization import encode, decode

FULL_MODEL = dict(inp_channels=2, out_channels=2, dim=48, num_blocks=[4, 6, 6, 8],
                  num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                  bias=False, LayerNorm_type='WithBias', decoder=True)


def build_model(config):
    if config.get('inp_channels') != 2 or config.get('out_channels') != 2:
        raise ValueError('ISAR models require Re,Im input/output channels')
    if not config.get('decoder', True):
        raise ValueError('ISAR reproduction requires all three AFLBs')
    return AdaIR(**config).float()


def padded_forward(model, x):
    """Pad outside the model to multiple of 8. Never resize the ISAR image."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect' if h > ph and w > pw else 'replicate')
    return model(x)[..., :h, :w]


@torch.no_grad()
def restore(model, z, stats, device):
    x = torch.from_numpy(encode(z, stats)).unsqueeze(0).to(device)
    output = padded_forward(model, x)
    if output.dtype != torch.float32 or not torch.isfinite(output).all():
        raise ValueError('Expected finite float32 model result')
    result = decode(output[0].cpu().numpy(), stats)
    assert result.dtype == np.complex128 and result.shape == z.shape
    return result
