import time
import numpy as np

import torch
from torch_efficient_distloss import eff_distloss, eff_distloss_native, flatten_eff_distloss


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


def einsum_distloss(w, m, interval):
    '''
    Einsum realization of distortion loss.
    There are B rays each with N sampled points.
    w:        Float tensor in shape [B,N]. Volume rendering weights of each point.
    m:        Float tensor in shape [N]. Midpoint distance to camera of each point.
    interval: Scalar or float tensor in shape [B,N]. The query interval of each point.
    Note: 
    The first term of distortion could be experssed as `(w @ mm @ w.T).diagonal()`, which 
    could be further accelerated by einsum function `torch.einsum('bq, qp, bp->b', w, mm, w)`
    '''
    mm = (m.unsqueeze(-1) - m.unsqueeze(-2)).abs()  # [N,N]
    loss = torch.einsum('bq, qp, bp->b', w, mm, w)
    loss += (w*w*interval).sum(-1)/3.
    return loss.mean()

def gen_example(B, N):
    w = torch.rand(B, N).cuda()
    w = w / w.sum(-1, keepdim=True)
    w = w.clone().requires_grad_()
    s = torch.linspace(0, 1, N+1).cuda()
    m_ = (s[1:] + s[:-1]) * 0.5
    m = m_[None].repeat(B,1)
    interval = 1/N
    return w, m, interval, m_


def spec(f, NTIMES, *args):
    ts_forward = []
    ts_backward = []
    for i in range(1+NTIMES):
        torch.cuda.empty_cache()

        torch.cuda.synchronize()
        s_time = time.time()
        loss = f(*args)
        torch.cuda.synchronize()
        e_time = time.time()
        if i>0:
            ts_forward.append(e_time - s_time)

        torch.cuda.synchronize()
        s_time = time.time()
        loss.backward()
        torch.cuda.synchronize()
        e_time = time.time()
        if i>0:
            ts_backward.append(e_time - s_time)

        del loss

    #print(f'forward : {np.sum(ts_forward):.1f} sec.')
    #print(f'backward: {np.sum(ts_backward):.1f} sec.')
    print(f'total   : {np.sum(ts_forward) + np.sum(ts_backward):.1f} sec.')

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    loss = f(*args)
    torch.cuda.synchronize()
    mem_forward = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    loss.backward()
    torch.cuda.synchronize()
    mem_backward = torch.cuda.max_memory_allocated()

    del loss

    #print('forward :', mem_forward/1024/1024, 'MB.')
    #print('backward:', mem_backward/1024/1024, 'MB.')
    print(f'total    : {max(mem_forward,mem_backward)/1024/1024:.0f} MB.')


if __name__ == '__main__':
    # B rays N points
    B = 8192
    NTIMES = 100

    for N in [32, 64, 128, 256, 384, 512, 1024]:
        print(f' B={B}; N={N} '.center(50, '='))
        w, m, interval, m_ = gen_example(B, N)
        ray_id = torch.arange(len(w))[:,None].repeat(1,N).cuda()

        try:
           print(' original_distloss '.center(50, '.'))
           spec(original_distloss, NTIMES, w, m, interval)
        except RuntimeError as e:
           print(e)

        try:
            print(' eff_distloss_native '.center(50, '.'))
            spec(eff_distloss_native, NTIMES, w, m, interval)
        except RuntimeError as e:
            print(e)

        try:
            print(' eff_distloss '.center(50, '.'))
            spec(eff_distloss, NTIMES, w, m, interval)
        except RuntimeError as e:
            print(e)

        try:
            print(' flatten_eff_distloss '.center(50, '.'))
            spec(flatten_eff_distloss, NTIMES, w.flatten(), m.flatten(), interval, ray_id.flatten())
        except RuntimeError as e:
            print(e)
        
        try:
            print('einsum_distloss'.center(50, '.'))
            spec(einsum_distloss, NTIMES, w, m_, interval)
        except RuntimeError as e:
            print(e)
