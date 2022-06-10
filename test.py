import torch
from torch_eff_distloss import eff_distloss, eff_distloss_native, flatten_eff_distloss


def original_distloss(w, m, interval):
    '''
    Original O(N^2) realization of distortion loss.
    There are B rays each with N sampled points.
    w:        Float tensor in shape [B,N]. Volume rendering weights of each point.
    m:        Float tensor in shape [B,N]. Midpoint distance to camera of each point.
    interval: Scalar or float tensor in shape [B,N]. The query interval of each point.
    '''
    loss_uni = (1/3) * (interval * w.pow(2)).sum(-1).mean()
    ww = w.unsqueeze(-1) * w.unsqueeze(-2)          # [B,N,N]
    mm = (m.unsqueeze(-1) - m.unsqueeze(-2)).abs()  # [B,N,N]
    loss_bi = (ww * mm).sum((-1,-2)).mean()
    return loss_uni + loss_bi


def check(func_name, my_forward_val, ans_forward, ans_backward, w):
    ret = 'PASS' if torch.isclose(ans_forward, my_forward_val) else 'FAIL'
    print(f'Test {func_name} forward:', ret)
    my_forward_val.backward()
    my_backward_grad = w.grad.clone()
    w.grad = None
    ret = 'PASS' if torch.isclose(ans_backward, my_backward_grad).all() else 'FAIL'
    print(f'Test {func_name} backward:', ret)


if __name__ == '__main__':
    # B rays N points
    B = 8192
    N = 128
    w = torch.rand(B, N).cuda()
    w = w / w.sum(-1, keepdim=True)
    w = w.clone().requires_grad_()
    s = torch.linspace(0, 1, N+1).cuda()
    m = (s[1:] + s[:-1]) * 0.5
    m = m[None].repeat(B,1)
    interval = 1/N

    # Compute forward & backward answer
    ans_forward = original_distloss(w, m, interval)
    ans_forward.backward()
    ans_backward = w.grad.clone()
    w.grad = None

    # scalar interval
    interval = 1/N
    check(
        'eff_distloss_native scalar interval',
        eff_distloss_native(w, m, interval),
        ans_forward, ans_backward, w)

    # array interval
    interval = torch.full_like(m, 1/N)
    check(
        'eff_distloss_native array interval',
        eff_distloss_native(w, m, interval),
        ans_forward, ans_backward, w)

    # scalar interval
    interval = 1/N
    check(
        'eff_distloss scalar interval',
        eff_distloss(w, m, interval),
        ans_forward, ans_backward, w)

    # array interval
    interval = torch.full_like(m, 1/N)
    check(
        'eff_distloss array interval',
        eff_distloss(w, m, interval),
        ans_forward, ans_backward, w)


    # irregular shape, scalar interval
    interval = 1/N
    ray_id = torch.arange(len(w))[:,None].repeat(1,N).cuda()
    check(
        'flatten_eff_distloss scalar interval',
        flatten_eff_distloss(w.flatten(), m.flatten(), interval, ray_id.flatten()),
        ans_forward, ans_backward, w)

    # irregular shape, array interval
    interval = torch.full_like(m, 1/N).flatten()
    check(
        'flatten_eff_distloss array interval',
        flatten_eff_distloss(w.flatten(), m.flatten(), interval, ray_id.flatten()),
        ans_forward, ans_backward, w)

