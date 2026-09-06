# %% [markdown]
# # Sequence packing과 block-diagonal mask
#
# DINOv2는 길이가 다른 crop 토큰 시퀀스(global 257, local 50)를 **하나의 긴 시퀀스로 이어붙여**(packing)
# 한 번에 forward한다. 시퀀스가 서로 섞이지 않게 하는 장치는 self-attention의 **block-diagonal mask**다.
#
# $$A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt d} + M\right)V,\qquad
# M_{ab} = \begin{cases}0 & a,b \text{ 같은 시퀀스}\\ -\infty & \text{다른 시퀀스}\end{cases}$$
#
# 이 스크립트는 numpy만으로 (1) 개별 forward, (2) packing + mask forward, (3) packing without mask를
# 비교해 "수학적으로 완전히 동일"함을 수치로 확인한다.

# %%
# 필요 패키지: numpy, plotly, kaleido (expy.png 저장용)
import os
import numpy as np

def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. 장난감 single-head attention
# 하나의 가중치 세트 $W_q, W_k, W_v$ 를 공유하는 attention 층. `mask`는 더해지는 additive bias
# (0 또는 $-\infty$).

# %%
D = 8  # 토큰 차원
Wq, Wk, Wv = (rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(3))

def softmax(s, axis=-1):
    s = s - s.max(axis=axis, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=axis, keepdims=True)

def attention(x, mask=None):
    """x: [N, D] -> (출력 [N, D], attention 확률 [N, N])"""
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    s = q @ k.T / np.sqrt(D)
    if mask is not None:
        s = s + mask
    p = softmax(s)
    return p @ v, p

# %% [markdown]
# ## 2. 길이가 다른 crop 시퀀스 4개
# 실제 DINOv2 비율(global 2개, local 8개)의 축소판: global 2개(길이 6), local 2개(길이 3).

# %%
seqlens = [6, 6, 3, 3]
crops = [rng.standard_normal((n, D)) for n in seqlens]
print("seqlens:", seqlens, " packed length:", sum(seqlens))
# 출력: seqlens: [6, 6, 3, 3]  packed length: 18

# %% [markdown]
# ## 3. 기준: 시퀀스를 하나씩 따로 forward (iBOT 방식)

# %%
outs_sep, attn_sep = zip(*(attention(c) for c in crops))
print([o.shape for o in outs_sep])
# 출력: [(6, 8), (6, 8), (3, 8), (3, 8)]

# %% [markdown]
# ## 4. packing + block-diagonal mask
# `xformers.ops.fmha.BlockDiagonalMask.from_seqlens(seqlens)` 가 하는 일을 numpy로 옮긴 것.
# 대각 블록은 0, 나머지는 $-\infty$.

# %%
def block_diagonal_mask(seqlens):
    N = sum(seqlens)
    m = np.full((N, N), -np.inf)
    start = 0
    for n in seqlens:
        m[start:start + n, start:start + n] = 0.0
        start += n
    return m

def split(packed, seqlens):
    return np.split(packed, np.cumsum(seqlens)[:-1], axis=0)

x_cat = np.concatenate(crops, axis=0)          # torch.cat(..., dim=1) 에 해당
mask = block_diagonal_mask(seqlens)
out_packed, attn_packed = attention(x_cat, mask)
outs_packed = split(out_packed, seqlens)      # attn_bias.split(x) 에 해당

allowed = np.isfinite(mask)
print("mask 허용 비율: %.3f  (= sum(n_i^2)/N^2 = %d/%d)"
      % (allowed.mean(), sum(n * n for n in seqlens), sum(seqlens) ** 2))
# 출력: mask 허용 비율: 0.278  (= sum(n_i^2)/N^2 = 90/324)

# %% [markdown]
# ## 5. 검증: 개별 forward와 완전히 동일한가?

# %%
max_err = max(np.abs(a - b).max() for a, b in zip(outs_sep, outs_packed))
print("packed(mask) vs separate  max |diff| =", max_err)
print("np.allclose:", all(np.allclose(a, b) for a, b in zip(outs_sep, outs_packed)))
# 출력: packed(mask) vs separate  max |diff| = 4.440892098500626e-16
# 출력: np.allclose: True

# attention 확률 행렬도 블록별로 일치하고, 블록 바깥은 정확히 0
off_block = attn_packed[~allowed]
print("블록 바깥 attention 최대값:", off_block.max())
print("대각 블록이 개별 attention과 일치:",
      all(np.allclose(attn_packed[s:s + n, s:s + n], a)
          for a, (s, n) in zip(attn_sep, zip(np.cumsum([0] + seqlens[:-1]), seqlens))))
# 출력: 블록 바깥 attention 최대값: 0.0
# 출력: 대각 블록이 개별 attention과 일치: True

# %% [markdown]
# ## 6. 반례: 마스크 없이 packing하면 cross-contamination
# 같은 concat이지만 mask=None. softmax 분모에 다른 crop의 key가 끼어들어 결과가 달라진다.

# %%
out_nomask, attn_nomask = attention(x_cat, None)
outs_nomask = split(out_nomask, seqlens)
max_err_nomask = max(np.abs(a - b).max() for a, b in zip(outs_sep, outs_nomask))
print("packed(no mask) vs separate  max |diff| =", round(float(max_err_nomask), 4))
print("다른 crop으로 새는 attention 질량(행 평균):",
      round(float(attn_nomask[~allowed].reshape(len(x_cat), -1).sum(1).mean()), 3))
# 출력: packed(no mask) vs separate  max |diff| = 2.5307
# 출력: 다른 crop으로 새는 attention 질량(행 평균): 0.775

# %% [markdown]
# ## 7. 왜 softmax가 $-\infty$ 를 "완벽히" 지우는가
# 행 $a$의 attention: $p_{ab} = \dfrac{e^{s_{ab}+M_{ab}}}{\sum_c e^{s_{ac}+M_{ac}}}$.
# $M_{ab}=-\infty$ 인 항은 $e^{-\infty}=0$ 이라 분자·분모 모두에서 사라지고,
# 남는 합은 자기 시퀀스 토큰만 — 즉 단독 forward의 softmax와 항등이다.
# 아래는 첫 crop의 첫 행을 두 방식으로 직접 계산해 비교.

# %%
q0 = (crops[0] @ Wq)[0]
k_own = crops[0] @ Wk
k_all = x_cat @ Wk
s_own = k_own @ q0 / np.sqrt(D)
s_all = k_all @ q0 / np.sqrt(D) + mask[0]
print("softmax(own)    :", np.round(softmax(s_own), 4))
print("softmax(packed) :", np.round(softmax(s_all)[:6], 4), " (나머지 12개는 0)")
# 출력: softmax(own)    : [0.0694 0.3369 0.115  0.0735 0.2394 0.1658]
# 출력: softmax(packed) : [0.0694 0.3369 0.115  0.0735 0.2394 0.1658]  (나머지 12개는 0)

# %% [markdown]
# ## 8. 실제 DINOv2 크기에서의 효율
# patch 14: global 224² → 16²+1 = 257 토큰, local 98² → 7²+1 = 50 토큰. 이미지 1장 = 2 global + 8 local.
# dense 마스크로 구현하면 N×N 중 유효 항목 비율이 낮다 → xFormers는 블록 바깥 타일을 **계산 자체를 건너뛴다**.
# 비교: 모두 257로 padding하는 대안은 토큰 수가 크게 늘어난다.

# %%
g, l = 16 * 16 + 1, 7 * 7 + 1
real_seqlens = [g] * 2 + [l] * 8
N = sum(real_seqlens)
useful = sum(n * n for n in real_seqlens)
print(f"packed N = {N}, dense N^2 = {N*N:,}, 유효 블록 항목 = {useful:,} ({useful/(N*N):.1%})")
print(f"padding 대안: 10 crops × {g} = {10*g} 토큰 vs packing {N} 토큰 → 토큰 {10*g/N:.2f}배")
print(f"padding 대안의 attention 항목: {10*g*g:,} vs 블록-희소 {useful:,} → {10*g*g/useful:.2f}배")
# 출력: packed N = 914, dense N^2 = 835,396, 유효 블록 항목 = 152,098 (18.2%)
# 출력: padding 대안: 10 crops × 257 = 2570 토큰 vs packing 914 토큰 → 토큰 2.81배
# 출력: padding 대안의 attention 항목: 660,490 vs 블록-희소 152,098 → 4.34배

# %% [markdown]
# ## 9. 시각화
# 왼쪽: block-diagonal mask (허용=1). 가운데: mask 적용 attention — 대각 블록 바깥이 0.
# 오른쪽: mask 없는 attention — 다른 crop으로 attention이 샌다.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.06,
                    subplot_titles=("block-diagonal mask (1=허용)",
                                    "attention with mask (개별 forward와 동일)",
                                    "attention without mask (섞임)"))
fig.add_trace(go.Heatmap(z=allowed.astype(float), colorscale="Greys", zmin=0, zmax=1,
                         showscale=False), row=1, col=1)
fig.add_trace(go.Heatmap(z=attn_packed, colorscale="Viridis", zmin=0, zmax=attn_packed.max(),
                         showscale=False), row=1, col=2)
fig.add_trace(go.Heatmap(z=attn_nomask, colorscale="Viridis", zmin=0, zmax=attn_packed.max(),
                         colorbar=dict(title="p", len=0.8)), row=1, col=3)
# 시퀀스 경계선
for c in (1, 2, 3):
    for b in np.cumsum(seqlens)[:-1]:
        fig.add_hline(y=b - 0.5, line=dict(color="red", width=1), row=1, col=c)
        fig.add_vline(x=b - 0.5, line=dict(color="red", width=1), row=1, col=c)
    fig.update_yaxes(autorange="reversed", scaleanchor=f"x{c if c > 1 else ''}", row=1, col=c)
fig.update_layout(width=1350, height=480, title_text=f"Sequence packing: seqlens={seqlens}, N={sum(seqlens)}",
                  margin=dict(t=90, b=30))
_show(fig)
png_path = os.path.join(HERE, "expy.png")
try:
    fig.write_image(png_path, scale=2)
    print("saved:", png_path)
except Exception as e:  # kaleido 미설치 등
    print("expy.png 저장 실패:", e)
# 출력: saved: /home/sungwoo/projects/swcho/dinov2/fm/papers/.fm/hints/752e5776-a44b-4bd1-b9ab-0ceb2c7b805c/expy.png
