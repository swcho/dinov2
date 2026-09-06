# %% [markdown]
# # Efficient Stochastic Depth — "마스킹" 대신 "슬라이싱"
#
# **Stochastic depth**(Huang et al., 2016)는 residual 블록 $x \leftarrow x + f(x)$ 에서
# 샘플마다 확률 $d$ 로 residual $f(x)$ 를 통째로 버리는(=블록을 건너뛰는) 정규화 기법이다.
#
# - **기존 구현(masking)**: 배치 $B$ 개 전부에 대해 $f(x)$ 를 계산한 뒤, 버릴 샘플의 결과에 0을 곱한다.
#   → 버린 샘플의 연산·메모리가 그대로 낭비된다.
# - **DINOv2의 efficient 구현(slicing)**: 배치 차원에서 $B$ 개 샘플을 무작위로 섞고(`randperm`),
#   **앞쪽 $(1-d)\times B$ 개만 슬라이스**해서 $f$ 에 넣는다. 나머지는 $f$ 를 아예 계산하지 않는다.
#   → 연산·메모리가 대략 drop rate $d$ 에 비례해 절약된다. 논문은 $d = 40\%$ 라는 높은 값을 쓴다.
#
# 이 스크립트는 두 구현을 numpy 로 나란히 만들어 (1) 실제로 몇 개만 계산하는지,
# (2) 기대값이 같도록 스케일이 어떻게 보정되는지, (3) 절약되는 연산량을 확인한다.

# %%
# 필요 패키지: numpy, plotly, kaleido (그림 저장용)
import os
import numpy as np
import plotly.graph_objects as go


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. 장난감 residual 함수와 호출 횟수 계측
#
# 실제 ViT 블록의 $f$ 는 Attention/MLP 이지만, 여기서는 "몇 개의 샘플에 대해 계산했는가"만 세면 되므로
# 단순한 선형 변환으로 대체하고, 처리한 샘플 수를 누적한다.

# %%
D = 8  # 토큰 차원 (작게)
W = rng.normal(scale=0.1, size=(D, D))
calls = {"samples_processed": 0}


def residual_func(x_sub):
    """x_sub: (b_sub, N, D). 처리한 샘플 수를 기록한다."""
    calls["samples_processed"] += x_sub.shape[0]
    return np.tanh(x_sub @ W)


B, N = 10, 4  # 배치 10, 토큰 4
x = rng.normal(size=(B, N, D))
print("x.shape =", x.shape)
# 출력: x.shape = (10, 4, 8)

# %% [markdown]
# ## 2. 기존 방식: 전부 계산하고 마스킹
#
# 샘플별 keep 마스크 $m_i \sim \text{Bernoulli}(1-d)$ 를 뽑고
# $$x_i \leftarrow x_i + \frac{m_i}{1-d}\, f(x_i)$$
# 로 갱신한다. $1/(1-d)$ 는 기대값을 보존하기 위한 스케일이다. 하지만 $f$ 는 **B개 전부** 계산된다.

# %%
def masked_stochastic_depth(x, residual_func, d):
    b = x.shape[0]
    keep = (rng.random(b) > d).astype(x.dtype)          # (b,)
    residual = residual_func(x)                          # 전부 계산!
    return x + residual * (keep / (1 - d))[:, None, None]


d = 0.4
calls["samples_processed"] = 0
y_mask = masked_stochastic_depth(x, residual_func, d)
print(f"[masking]  d={d}: f 에 들어간 샘플 수 = {calls['samples_processed']} / {B}")
# 출력: [masking]  d=0.4: f 에 들어간 샘플 수 = 10 / 10

# %% [markdown]
# ## 3. DINOv2 방식: 섞고 → 앞쪽 $(1-d)B$ 개만 슬라이스 → index_add
#
# 1. `brange = randperm(B)[: int((1-d)·B)]` — 무작위 순열의 앞부분만 취한다.
# 2. `residual = f(x[brange])` — 살아남은 샘플에 대해서만 계산한다.
# 3. `x[brange] += residual · B / |brange|` — 결과를 원래 위치에 더하되,
#    스케일 $\frac{B}{(1-d)B} = \frac{1}{1-d}$ 로 기대값을 맞춘다.
#
# 이는 실제 저장소 `dinov2/layers/block.py` 의 `drop_add_residual_stochastic_depth` 와 동일한 흐름이다.

# %%
def efficient_stochastic_depth(x, residual_func, d):
    b = x.shape[0]
    sample_subset_size = max(int(b * (1 - d)), 1)
    brange = rng.permutation(b)[:sample_subset_size]     # 1) 섞고 앞쪽만 슬라이스
    residual = residual_func(x[brange])                  # 2) 부분집합에만 f 적용
    scale = b / sample_subset_size                       # = 1/(1-d)
    out = x.copy()
    np.add.at(out, brange, residual * scale)             # 3) torch.index_add 에 해당
    return out, brange


calls["samples_processed"] = 0
y_eff, brange = efficient_stochastic_depth(x, residual_func, d)
print(f"[slicing]  d={d}: 살아남은 인덱스 brange = {sorted(brange.tolist())}")
print(f"[slicing]  d={d}: f 에 들어간 샘플 수 = {calls['samples_processed']} / {B}  "
      f"(= int((1-d)*B) = {int((1 - d) * B)})")
print(f"[slicing]  residual 스케일 = B/|brange| = {B}/{len(brange)} = {B / len(brange):.3f} = 1/(1-d)")
# 출력: [slicing]  d=0.4: 살아남은 인덱스 brange = [0, 2, 4, 6, 8, 9]
# 출력: [slicing]  d=0.4: f 에 들어간 샘플 수 = 6 / 10  (= int((1-d)*B) = 6)
# 출력: [slicing]  residual 스케일 = B/|brange| = 10/6 = 1.667 = 1/(1-d)

# %%
# 버려진 샘플은 x 그대로(항등), 살아남은 샘플만 residual 이 더해졌는지 확인
dropped = np.setdiff1d(np.arange(B), brange)
print("버려진 샘플이 항등 통과했나?", np.allclose(y_eff[dropped], x[dropped]))
print("살아남은 샘플은 바뀌었나?   ", not np.allclose(y_eff[brange], x[brange]))
# 출력: 버려진 샘플이 항등 통과했나? True
# 출력: 살아남은 샘플은 바뀌었나?    True

# %% [markdown]
# ## 4. 두 방식의 기대값은 같다 (Monte Carlo 확인)
#
# 스케일 $1/(1-d)$ 덕분에 $\mathbb{E}[y] = x + f(x)$ 가 두 구현 모두에서 성립한다.
# 차이는 **"전부 계산 후 0 곱하기"** vs **"살아남은 것만 계산"** 이라는 연산 경로뿐이다.

# %%
T = 20000
acc_mask = np.zeros_like(x)
acc_eff = np.zeros_like(x)
for _ in range(T):
    acc_mask += masked_stochastic_depth(x, residual_func, d)
    acc_eff += efficient_stochastic_depth(x, residual_func, d)[0]
target = x + residual_func(x)
print("E[y]-(x+f(x)) 최대 오차 (masking):", np.abs(acc_mask / T - target).max().round(4))
print("E[y]-(x+f(x)) 최대 오차 (slicing):", np.abs(acc_eff / T - target).max().round(4))
# 출력: E[y]-(x+f(x)) 최대 오차 (masking): 0.0087
# 출력: E[y]-(x+f(x)) 최대 오차 (slicing): 0.0071
# (표본 수 T 에 따른 MC 잡음 수준이며, 두 방식 모두 x+f(x) 로 수렴)

# %% [markdown]
# ## 5. 절약되는 연산량: drop rate 에 비례
#
# residual 계산량은 슬라이스 크기 $\lfloor (1-d) B \rfloor$ 에 비례하므로,
# 절약 비율은 대략 $d$ 와 같다. $d=0.4$ 이면 블록 내부 Attention/MLP 연산과 activation 메모리의 약 40% 를 건너뛴다.
# (참고: 저장소 기본 설정 `ssl_default_config.yaml` 은 `drop_path_rate: 0.3`, `drop_path_uniform: true` —
# 모든 블록에 같은 $d$ 를 적용한다. 논문의 ViT-g 학습에서는 $d = 0.4$.)

# %%
ds = np.linspace(0.0, 0.9, 10)
B_big = 1024
compute_frac_mask = np.ones_like(ds)                              # 마스킹: 항상 100%
compute_frac_eff = np.floor((1 - ds) * B_big) / B_big             # 슬라이싱: (1-d)
for dd, m, e in zip(ds, compute_frac_mask, compute_frac_eff):
    print(f"d={dd:.1f}  masking 계산량={m:5.0%}   slicing 계산량={e:5.0%}   절약={1 - e:5.0%}")
# 출력: d=0.0  masking 계산량= 100%   slicing 계산량= 100%   절약=   0%
# 출력: d=0.1  masking 계산량= 100%   slicing 계산량=  90%   절약=  10%
# 출력: d=0.2  masking 계산량= 100%   slicing 계산량=  80%   절약=  20%
# 출력: d=0.3  masking 계산량= 100%   slicing 계산량=  70%   절약=  30%
# 출력: d=0.4  masking 계산량= 100%   slicing 계산량=  60%   절약=  40%
# 출력: d=0.5  masking 계산량= 100%   slicing 계산량=  50%   절약=  50%
# 출력: d=0.6  masking 계산량= 100%   slicing 계산량=  40%   절약=  60%
# 출력: d=0.7  masking 계산량= 100%   slicing 계산량=  30%   절약=  70%
# 출력: d=0.8  masking 계산량= 100%   slicing 계산량=  20%   절약=  80%
# 출력: d=0.9  masking 계산량= 100%   slicing 계산량=  10%   절약=  90%

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=ds, y=compute_frac_mask * 100, mode="lines+markers",
                         name="기존 stochastic depth (masking)", line=dict(dash="dash", color="#888")))
fig.add_trace(go.Scatter(x=ds, y=compute_frac_eff * 100, mode="lines+markers",
                         name="DINOv2 efficient (shuffle + slice)", line=dict(color="#1f77b4", width=3)))
fig.add_vline(x=0.4, line=dict(color="crimson", dash="dot"))
fig.add_annotation(x=0.4, y=60, text="논문 설정 d=40% → 블록 연산 60%만 수행",
                   showarrow=True, arrowhead=2, ax=120, ay=-40, font=dict(color="crimson"))
fig.update_layout(
    title="Residual 블록 f(x) 연산량 vs. drop rate d (배치 B=1024)",
    xaxis_title="drop rate d", yaxis_title="f(x) 를 계산한 샘플 비율 (%)",
    yaxis=dict(range=[0, 110]), legend=dict(x=0.02, y=0.15),
    width=800, height=480, template="plotly_white",
)
_show(fig)
fig.write_image(os.path.join(HERE, "expy.png"))
print("saved:", os.path.join(HERE, "expy.png"))
# 출력: saved: .../37d99f60-8fd4-487a-9dba-a82774be4a9f/expy.png

# %% [markdown]
# ## 정리
#
# | | 기존 stochastic depth | DINOv2 efficient stochastic depth |
# |---|---|---|
# | 무엇을 뽑나 | 샘플별 Bernoulli 마스크 | `randperm(B)[: (1-d)·B]` 인덱스 |
# | $f$ 를 계산하는 샘플 수 | $B$ (전부) | $(1-d)\,B$ (슬라이스만) |
# | 결과 반영 | 마스크 곱하기 | `index_add` 로 원위치에 더하기 |
# | 기대값 보정 | $\times \frac{1}{1-d}$ | $\times \frac{B}{(1-d)B} = \frac{1}{1-d}$ |
# | 연산·메모리 절약 | 없음 | $\approx d$ 비율 (논문 $d=40\%$) |
#
# 핵심: **"버릴 샘플의 residual 을 계산하고 0을 곱하는" 대신 "계산 자체를 건너뛴다."**
# 높은 $d=0.4$ 는 정규화·학습 안정성(NaN 회피, Table 1)과 함께 큰 모델(ViT-g)의 연산·메모리 절감을 동시에 준다.
