# torch_efficient_distloss

Distortion loss is proposed by [mip-nerf-360](https://jonbarron.info/mipnerf360/), which encourages volume rendering weights to be compact and sparse and can alleviate `floater` and `background collapse` artifact.
In our DVGOv2 report (coming soon), we show that the distortion loss is also helpful to point-based query, which speeds up our training and gives us better quantitative results.

A pytorch pseudo-code for the distortion loss:
```python
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
```

Unfortunately, the straightforward implementation results in `O(N^2)` space time complexity for N sampled points on a ray. In this package, we provide our `O(N)` realization presnted in the DVGOv2 report.

Please cite mip-nerf-360 if you find this repo helpful. We will be happy if you also cite DVGOv2.
```
@inproceedings{BarronMVSH22,
  author    = {Jonathan T. Barron and
               Ben Mildenhall and
               Dor Verbin and
               Pratul P. Srinivasan and
               Peter Hedman},
  title     = {Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields},
  booktitle = {CVPR},
  year      = {2022},
}

@article{SunSC22_2,
  author    = {Cheng Sun and
               Min Sun and
               Hwann{-}Tzong Chen},
  title     = {Improved Direct Voxel Grid Optimization for Radiance Fields Reconstruction},
  journal   = {to be announced},
  year      = {2022},
}
```

## Installation
```
pip install torch_efficient_distloss
```
Assumed `Pytorch` and `numpy` are already installed.

## Documentation
All functions are runs in `O(N)` and are numerical equivalent to the distortion loss.
```python
from torch_efficient_distloss import eff_distloss, eff_distloss_native, flatten_eff_distloss

# A toy example
B = 8192  # number of rays
N = 128   # number of points on a ray
w = torch.rand(B, N).cuda()
w = w / w.sum(-1, keepdim=True)
w = w.clone().requires_grad_()
s = torch.linspace(0, 1, N+1).cuda()
m = (s[1:] + s[:-1]) * 0.5
m = m[None].repeat(B,1)
interval = 1/N

loss = eff_distloss(w, m, interval)
loss.backward()
print('Loss', loss)
print('Gradient', w.grad)
```
- `eff_distloss_native`:
    - Using built-in Pytorch operation to implement the `O(N)` distortion loss.
    - Input:
        - `w`: Float tensor in shape [B,N]. Volume rendering weights of each point.
        - `m`: Float tensor in shape [B,N]. Midpoint distance to camera of each point.
        - `interval`: Scalar or float tensor in shape [B,N]. The query interval of each point.
- `eff_distloss`:
    - The same as `eff_distloss_native`. Slightly faster and consume slightly more GPU memory.
- `flatten_eff_distloss`:
    - Support varied number of sampled points on each ray.
    - All input tensor should be flatten.
    - Should provide an additional flatten Long tensor `ray_id` to specify the ray index of each point. `ray_id` should be increasing (i.e., `ray_id[i-1]<=ray_id[i]`) and ranging from `0` to `N-1`.


## Testing
### Numerical equivalent
Run `python test.py`. All our implementation is numerical equivalent to the `O(N^2)` `original_distloss`.

### Speed and memeory benchmark
Run `python test_time_mem.py`. We use a batch of `B=8192` rays. Below is the results on my `RTX 2080Ti` GPU.
- Peak GPU memory (MB)
    | \# of pts `N` | 32 | 64 | 128 | 256 | 384 | 512 |
    |:------------:|:--:|:--:|:---:|:---:|:---:|:---:|
    |`original_distloss`   |102|396|1560|6192|OOM|OOM|
    |`eff_distloss_native` |12|24|48|96|144|192|
    |`eff_distloss`        |14|28|56|112|168|224|
    |`flatten_eff_distloss`|13|26|52|104|156|208|
- Run time accumulated over 100 runs (sec)
    | \# of pts `N` | 32 | 64 | 128 | 256 | 384 | 512 |
    |:------------:|:--:|:--:|:---:|:---:|:---:|:---:|
    |`original_distloss`   |0.2|0.8|2.4|17.9|OOM|OOM|
    |`eff_distloss_native` |0.1|0.1|0.1|0.2|0.3|0.3|
    |`eff_distloss`        |0.1|0.1|0.1|0.1|0.2|0.2|
    |`flatten_eff_distloss`|0.1|0.1|0.1|0.2|0.2|0.3|

