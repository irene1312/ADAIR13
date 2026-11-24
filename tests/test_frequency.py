import torch
from net.model import FreModule


def test_frequency_integer_mask_disconnect_and_cross_attention():
    torch.manual_seed(10)
    module=FreModule(16,1,False,in_dim=2)
    small=torch.randn(1,2,64,64)
    high,low=module.fft(small)
    assert torch.count_nonzero(low)==0 and torch.count_nonzero(high)>0
    y=torch.randn(1,16,64,64)
    assert torch.count_nonzero(module.channel_cross_h(low,y))>0
    # At a nonempty mask, hard indexing still disconnects the threshold network.
    high,low=module.fft(torch.randn(1,2,512,512))
    assert torch.count_nonzero(low)>0
    (high.mean()+low.mean()).backward()
    assert all(p.grad is None for p in module.rate_conv.parameters())
    assert module.conv.weight.grad is None and module.score_gen.weight.grad is None
    assert module.conv1.weight.grad is not None and module.conv1.weight.grad.norm()>0
