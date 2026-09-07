# %% [markdown]
# # register 추가에 따른 계산 비용은 얼마나 늘어나는가?
#
# **카드 요약**
#
# > 파라미터 수 증가는 무시할 만한 수준이고 FLOP은 약간 증가한다.
# > register 16개면 FLOP이 최대 6% 늘지만, 주로 쓰는 4개에서는 2% 미만이다.
#
# 이 노트북은 그 숫자를 **말로 외우지 않고 직접 계산해서** 확인한다.
# 논문 *Vision Transformers Need Registers* (arXiv:2309.16588) 부록 B
# "Complexity Analysis" 및 **Fig. 12**(register 개수 대비 파라미터/FLOP 증가율 그림)에
# 해당하는 내용을 해석식으로 재현한다.
#
# 실행 환경 (의존성):
# ```
# /home/sungwoo/miniforge3/envs/trellis/bin/python
#   torch 2.4 / numpy 1.26 / plotly 6.9 / kaleido
# ```
# GPU 불필요 (전부 CPU).

# %%
import math

import numpy as np
import plotly.graph_objects as go


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


print("numpy", np.__version__)

# 출력: numpy 1.26.4


# %% [markdown]
# ## 1. ViT 한 블록의 FLOP을 토큰 수 $N$의 함수로 유도
#
# 표기: 토큰 수 $N$, 임베딩 차원 $d$, MLP ratio $r$ (보통 4), 블록 수 $L$.
# 아래는 **MAC(곱셈-누산) 개수**로 센다. FLOP으로 바꾸려면 전부 2를 곱하면 되고,
# 우리가 관심 있는 것은 **비율**이라 어느 쪽으로 세든 결과는 같다.
#
# ### MHSA
#
# | 연산 | 모양 | MAC |
# |---|---|---|
# | $Q,K,V$ projection | $(N{\times}d)\cdot(d{\times}3d)$ | $3Nd^2$ |
# | $QK^\top$ (attention 행렬) | $(N{\times}d)\cdot(d{\times}N)$ | $N^2 d$ |
# | $\mathrm{softmax}(\cdot)V$ | $(N{\times}N)\cdot(N{\times}d)$ | $N^2 d$ |
# | output projection | $(N{\times}d)\cdot(d{\times}d)$ | $Nd^2$ |
#
# $$\text{MHSA}(N) = 4Nd^2 + 2N^2 d$$
#
# head로 쪼개도 총합은 같다 ($h$개의 head가 각각 $d/h$ 차원을 맡으므로).
#
# ### MLP
#
# $d \to rd \to d$ 두 개의 Linear:
#
# $$\text{MLP}(N) = 2r\,Nd^2 \quad (r=4 \Rightarrow 8Nd^2)$$
#
# ### 블록 하나 / 모델 전체
#
# $$\text{MAC}_{\text{block}}(N) = (4 + 2r)\,Nd^2 + 2N^2 d \;\;\overset{r=4}{=}\;\; 12Nd^2 + 2N^2 d$$
#
# $$\boxed{\;\text{MAC}_{\text{model}}(N) = L\left(12Nd^2 + 2N^2 d\right) + \text{MAC}_{\text{patch-embed}}\;}$$
#
# LayerNorm·bias·softmax 같은 $O(Nd)$ 항은 위 항들보다 $d$배 작아서 무시한다.
#
# ### 핵심 관찰
#
# register $R$개를 붙이는 것은 **$N \to N + R$** 로 바꾸는 것뿐이다.
# 새 연산자도, 새 레이어도 없다. 그래서 오버헤드는
#
# $$\text{overhead}(R) = \frac{\text{MAC}(N+R)}{\text{MAC}(N)} - 1$$
#
# 이고, $Nd^2$ 항에서는 **선형**($\approx R/N$), $N^2d$ 항에서는 **2차**($\approx 2R/N$)로 는다.
# ViT-L/14 @224 에서는 $Nd^2$ 항이 압도적이라 사실상 $\approx R/N$ 이 된다.

# %%
def block_macs(N: int, d: int, r: float = 4.0) -> float:
    """ViT 블록 1개의 MAC 수 (MHSA + MLP)."""
    mhsa = 4 * N * d * d + 2 * N * N * d
    mlp = 2 * r * N * d * d
    return mhsa + mlp


def model_macs(N: int, d: int, depth: int, r: float = 4.0, patch_embed_macs: float = 0.0) -> float:
    return depth * block_macs(N, d, r) + patch_embed_macs


# ViT-L/14 (DINOv2 vit_large: embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4)
D, DEPTH, R_MLP = 1024, 24, 4.0
PATCH, IMG = 14, 224
N_PATCH = (IMG // PATCH) ** 2          # 16*16 = 256
N_BASE = 1 + N_PATCH                   # [CLS] + patch = 257
PE_MACS = N_PATCH * (PATCH * PATCH * 3) * D  # patch embedding (register와 무관)

print(f"patch 수 = {N_PATCH}, 기본 토큰 수 N = {N_BASE}")
print(f"블록 1개 MAC        = {block_macs(N_BASE, D):.4e}")
print(f"  ├─ Nd^2 항(12Nd^2) = {12*N_BASE*D*D:.4e}  ({12*N_BASE*D*D/block_macs(N_BASE,D):6.2%})")
print(f"  └─ N^2d 항(2N^2d)  = {2*N_BASE*N_BASE*D:.4e}  ({2*N_BASE*N_BASE*D/block_macs(N_BASE,D):6.2%})")
print(f"모델 전체 MAC       = {model_macs(N_BASE, D, DEPTH, R_MLP, PE_MACS):.4e}")
print(f"모델 전체 FLOP(=2x) = {2*model_macs(N_BASE, D, DEPTH, R_MLP, PE_MACS)/1e9:.1f} GFLOPs")

# 출력: patch 수 = 256, 기본 토큰 수 N = 257
# 출력: 블록 1개 MAC        = 3.3691e+09
# 출력:   ├─ Nd^2 항(12Nd^2) = 3.2338e+09  (95.99%)
# 출력:   └─ N^2d 항(2N^2d)  = 1.3527e+08  ( 4.01%)
# 출력: 모델 전체 MAC       = 8.1012e+10
# 출력: 모델 전체 FLOP(=2x) = 162.0 GFLOPs


# %% [markdown]
# ## 2. register 개수별 FLOP 증가율 — 논문과 대조
#
# 논문은 DINOv2 **ViT-L/14** 를 $R = 0, 1, 2, 4, 8, 16$ 으로 학습했다 (Fig. 8).
# 그 설정 그대로 오버헤드를 계산해서 Fig. 12의 주장과 맞춰 본다.

# %%
REGISTERS = [0, 1, 2, 4, 8, 16]

base = model_macs(N_BASE, D, DEPTH, R_MLP, PE_MACS)
rows = []
for R in REGISTERS:
    m = model_macs(N_BASE + R, D, DEPTH, R_MLP, PE_MACS)
    rows.append((R, N_BASE + R, m, m / base - 1.0, (N_BASE + R) / N_BASE - 1.0))

print(f"{'R':>3} {'N':>5} {'GFLOPs':>9} {'FLOP 증가율':>11} {'토큰수 증가율':>13}")
print("-" * 48)
for R, N, m, ov, tv in rows:
    print(f"{R:>3} {N:>5} {2*m/1e9:>9.2f} {ov:>11.2%} {tv:>13.2%}")

print()
print("논문(Fig. 12) 주장과 대조:")
ov4 = dict((r[0], r[3]) for r in rows)
print(f"  R=4  : {ov4[4]:.2%}  -> '2% 미만'  ? {'OK' if ov4[4] < 0.02 else 'NG'}")
print(f"  R=16 : {ov4[16]:.2%}  -> '최대 6% 정도' ? {'대략 일치' if 0.05 < ov4[16] < 0.07 else 'NG'}")

# 출력:   R     N    GFLOPs    FLOP 증가율       토큰수 증가율
# 출력: ------------------------------------------------
# 출력:   0   257    162.02       0.00%         0.00%
# 출력:   1   258    162.68       0.40%         0.39%
# 출력:   2   259    163.33       0.81%         0.78%
# 출력:   4   261    164.64       1.62%         1.56%
# 출력:   8   265    167.27       3.24%         3.11%
# 출력:  16   273    172.52       6.48%         6.23%
# 출력:
# 출력: 논문(Fig. 12) 주장과 대조:
# 출력:   R=4  : 1.62%  -> '2% 미만'  ? OK
# 출력:   R=16 : 6.48%  -> '최대 6% 정도' ? 대략 일치


# %% [markdown]
# ### 읽는 법
#
# * **FLOP 증가율은 토큰 수 증가율($R/N$)보다 아주 조금 크다.** 차이는 $N^2d$ 항 때문이다.
#   $R=16$이면 토큰이 6.23% 느는데 FLOP은 6.48% 는다.
# * $R=4$ → **1.6%**. 논문의 "below 2%" 와 일치한다.
# * $R=16$ → **6.5%**. 논문 Fig. 12의 "up to 6%" 와 사실상 같은 값이다.
#   해석식이 살짝 높게 나오는 이유: 실제 모델에는 register가 개수를 늘리지 않는
#   고정 비용(patch embedding, LayerNorm, head 등)이 더 있고, 논문 그림은
#   그 전체를 분모로 쓰기 때문이다. 여기서는 patch embedding만 넣었는데
#   그마저 전체의 0.2% 수준이라 거의 영향이 없다.
# * 결론의 방향은 동일하다: **연산 그래프는 그대로이고 시퀀스만 몇 토큰 길어진다.**

# %%
# 고정 비용(patch embed)을 분모에 넣고/빼고 비교 — 사실상 차이 없음
for R in (4, 16):
    with_pe = model_macs(N_BASE + R, D, DEPTH, R_MLP, PE_MACS) / model_macs(N_BASE, D, DEPTH, R_MLP, PE_MACS) - 1
    without_pe = model_macs(N_BASE + R, D, DEPTH, R_MLP, 0.0) / model_macs(N_BASE, D, DEPTH, R_MLP, 0.0) - 1
    print(f"R={R:>2}: patch-embed 포함 {with_pe:.3%} / 제외 {without_pe:.3%}")
print(f"patch embedding이 전체 MAC에서 차지하는 비중: {PE_MACS/base:.3%}")

# 출력: R= 4: patch-embed 포함 1.617% / 제외 1.620%
# 출력: R=16: patch-embed 포함 6.479% / 제외 6.491%
# 출력: patch embedding이 전체 MAC에서 차지하는 비중: 0.190%


# %% [markdown]
# ## 3. 파라미터 증가는 정말 "무시할 만한" 수준인가
#
# register 토큰은 `nn.Parameter(torch.zeros(1, R, d))` 하나다
# (DINOv2 구현: `dinov2/models/vision_transformer.py` 의 `self.register_tokens`).
# 즉 새로 생기는 학습 파라미터는 정확히
#
# $$\Delta P = R \times d$$
#
# 개뿐이다. positional embedding조차 늘지 않는다 — DINOv2는 pos_embed를 더한
# **뒤에** register를 concat 하므로 register에는 위치 임베딩이 붙지 않는다.
#
# ViT-L의 전체 파라미터를 항별로 세어 비교해 본다.

# %%
def vit_param_count(d: int, depth: int, r: float, patch: int, n_patch: int, R: int = 0, in_chans: int = 3) -> dict:
    per_block = (
        2 * d                       # LayerNorm 1 (weight+bias)
        + (3 * d * d + 3 * d)       # qkv
        + (d * d + d)               # attn proj
        + 2 * d                     # LayerNorm 2
        + (d * int(r * d) + int(r * d))  # mlp fc1
        + (int(r * d) * d + d)      # mlp fc2
        + 2 * d                     # LayerScale ls1, ls2
    )
    return {
        "patch_embed": patch * patch * in_chans * d + d,
        "cls_token": d,
        "pos_embed": (n_patch + 1) * d,
        "blocks": depth * per_block,
        "final_norm": 2 * d,
        "mask_token": d,
        "register_tokens": R * d,
    }


parts = vit_param_count(D, DEPTH, R_MLP, PATCH, N_PATCH, R=0)
total = sum(parts.values())
for k, v in parts.items():
    print(f"  {k:<16} {v:>12,}")
print(f"  {'TOTAL (R=0)':<16} {total:>12,}  ({total/1e6:.1f}M)")
print()
print(f"{'R':>3} {'추가 파라미터':>13} {'전체 대비':>12} {'ppm':>8}")
print("-" * 40)
for R in REGISTERS:
    extra = R * D
    print(f"{R:>3} {extra:>13,} {extra/total:>12.6%} {extra/total*1e6:>8.1f}")

# 출력:   patch_embed           603,136
# 출력:   cls_token               1,024
# 출력:   pos_embed             263,168
# 출력:   blocks            302,358,528
# 출력:   final_norm              2,048
# 출력:   mask_token              1,024
# 출력:   register_tokens             0
# 출력:   TOTAL (R=0)       303,228,928  (303.2M)
# 출력:
# 출력:   R       추가 파라미터        전체 대비      ppm
# 출력: ----------------------------------------
# 출력:   0             0    0.000000%      0.0
# 출력:   1         1,024    0.000338%      3.4
# 출력:   2         2,048    0.000675%      6.8
# 출력:   4         4,096    0.001351%     13.5
# 출력:   8         8,192    0.002702%     27.0
# 출력:  16        16,384    0.005403%     54.0


# %% [markdown]
# **R=4 → 4,096개, 전체 3억 파라미터의 13.5 ppm (0.0014%).**
# 논문이 "negligible change in number of parameters" 라고 쓴 이유가 이것이다.
# 비유하면 3억 개 중 4천 개 — 백만 분의 13이다. 반올림하면 사라진다.
#
# 반대로 FLOP은 왜 안 사라지나? 파라미터는 $R \times d$ 만 늘지만,
# **그 파라미터가 모든 블록의 모든 연산을 $R$ 토큰만큼 더 통과**하기 때문이다.
# 저장 비용은 $O(Rd)$, 계산 비용은 $O(L \cdot R d^2)$.

# %% [markdown]
# ## 4. 해상도(=토큰 수)에 따라 오버헤드는 어떻게 변하나
#
# $R$이 고정이면 오버헤드는 대략 $R/N$ 이므로 **$N$이 클수록 상대 오버헤드는 작아진다.**
# 정확히는
#
# $$\text{overhead}(R, N) = \frac{12(N{+}R)d^2 + 2(N{+}R)^2 d}{12Nd^2 + 2N^2d} - 1$$
#
# 고해상도 추론(DINOv2는 518px 등에서도 쓴다)에서는 register 오버헤드가 1% 아래로 떨어진다.
# 즉 **register는 해상도가 올라갈수록 더 공짜에 가까워진다.**

# %%
IMG_SIZES = [112, 154, 224, 280, 336, 392, 448, 518, 616, 700, 784, 896, 1036]
R_CURVES = [1, 2, 4, 8, 16]
PAL = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a", 8: "#eda100", 16: "#e87ba4"}

n_tokens = [1 + (s // PATCH) ** 2 for s in IMG_SIZES]
curves = {}
for R in R_CURVES:
    curves[R] = [
        (model_macs(N + R, D, DEPTH, R_MLP) / model_macs(N, D, DEPTH, R_MLP) - 1) * 100 for N in n_tokens
    ]

fig = go.Figure()
for R in R_CURVES:
    fig.add_trace(
        go.Scatter(
            x=n_tokens,
            y=curves[R],
            mode="lines+markers",
            name=f"R={R}",
            line=dict(color=PAL[R], width=2),
            marker=dict(size=7, color=PAL[R], line=dict(width=1.5, color="#fcfcfb")),
            hovertemplate="N=%{x} tokens<br>오버헤드 %{y:.2f}%<extra>R=" + str(R) + "</extra>",
        )
    )
    # 직접 라벨: 곡선이 서로 잘 벌어지는 왼쪽 끝에 붙인다
    # (팔레트 검증에서 contrast WARN -> 색만으로 식별되지 않게 라벨 필수)
    fig.add_annotation(
        x=math.log10(n_tokens[0]),
        y=math.log10(curves[R][0]),
        text=f"R={R}",
        showarrow=False,
        xanchor="right",
        xshift=-10,
        font=dict(size=13, color=PAL[R]),
    )

# 논문 설정(224px, N=257) 기준선
# 주의: log 축에서 add_shape 좌표는 "실제 데이터 값", add_annotation 좌표는 "log10 값"이다.
fig.add_shape(
    type="line", xref="x", yref="paper",
    x0=N_BASE, x1=N_BASE, y0=0, y1=1,
    line=dict(color="#a8a7a0", width=1, dash="dot"),
)
fig.add_annotation(
    x=math.log10(N_BASE), y=1.0, yref="paper",
    text="224px · N=257 (논문 설정)", showarrow=False,
    xanchor="left", xshift=6, yanchor="top",
    font=dict(size=11, color="#52514e"),
)
# "R=4에서 2% 미만" 기준선
fig.add_shape(
    type="line", xref="paper", yref="y",
    x0=0, x1=1, y0=2.0, y1=2.0,
    line=dict(color="#a8a7a0", width=1, dash="dash"),
)
fig.add_annotation(
    x=1.0, xref="paper", y=math.log10(2.0),
    text="논문 기준선 2%", showarrow=False,
    xanchor="right", yanchor="bottom", xshift=-4,
    font=dict(size=11, color="#52514e"),
)
# 논문이 실제로 인용하는 두 점을 강조
for R, label, ay in ((4, "R=4 · 1.6%", 30), (16, "R=16 · 6.5%", -28)):
    fig.add_annotation(
        x=math.log10(N_BASE),
        y=math.log10(curves[R][IMG_SIZES.index(224)]),
        text=label,
        showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="#8a8981",
        ax=52, ay=ay,
        font=dict(size=12, color="#0b0b0b"),
        bgcolor="#fcfcfb", bordercolor="#c9c8c2", borderwidth=1, borderpad=3,
    )

TICKS = [0.03, 0.1, 0.3, 1, 2, 3, 10, 30]
fig.update_layout(
    title=dict(
        text="register 오버헤드는 토큰이 많아질수록 줄어든다<br>"
        "<sup>ViT-L/14 (d=1024, L=24, r=4) · FLOP 증가율 = MAC(N+R)/MAC(N) − 1 · 양축 log</sup>",
        font=dict(size=17, color="#0b0b0b"),
        x=0.02, xanchor="left",
    ),
    xaxis=dict(
        title="시퀀스 길이 N = 1 + patch 수",
        type="log",
        range=[math.log10(40), math.log10(8000)],
        tickvals=[65, 122, 257, 577, 1370, 3137, 5477],
        ticktext=["65<br>112px", "122<br>154px", "257<br>224px", "577<br>336px",
                  "1370<br>518px", "3137<br>784px", "5477<br>1036px"],
        showgrid=True, gridcolor="#ececea", zeroline=False, linecolor="#c9c8c2",
    ),
    yaxis=dict(
        title="FLOP 증가율",
        type="log",
        range=[math.log10(0.025), math.log10(40)],
        tickvals=TICKS,
        ticktext=[f"{t:g}%" for t in TICKS],
        showgrid=True, gridcolor="#ececea", zeroline=False, linecolor="#c9c8c2",
    ),
    plot_bgcolor="#fcfcfb",
    paper_bgcolor="#fcfcfb",
    font=dict(family="DejaVu Sans, sans-serif", size=13, color="#0b0b0b"),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0, title=""),
    margin=dict(l=110, r=40, t=110, b=80),
    width=980,
    height=560,
    hovermode="x unified",
)

_show(fig)
fig.write_image("/home/sungwoo/projects/swcho/dinov2/fm/papers/.fm/hints/20948ea2-6003-476a-b0ed-b4654428a1de/expy.png", scale=2)
print("saved expy.png")

# 표 형태로도 (색만으로 읽지 않도록)
print(f"\n{'해상도':>7} {'N':>6} " + " ".join(f"{'R='+str(R):>7}" for R in R_CURVES))
print("-" * 55)
for s, N in zip(IMG_SIZES, n_tokens):
    vals = " ".join(f"{curves[R][n_tokens.index(N)]:>6.2f}%" for R in R_CURVES)
    print(f"{s:>5}px {N:>6} {vals}")

# 출력: saved expy.png
# 출력:
# 출력:     해상도      N     R=1     R=2     R=4     R=8    R=16
# 출력: -------------------------------------------------------
# 출력:   112px     65   1.55%   3.11%   6.22%  12.45%  24.94%
# 출력:   154px    122   0.84%   1.67%   3.34%   6.69%  13.40%
# 출력:   224px    257   0.40%   0.81%   1.62%   3.24%   6.49%
# 출력:   280px    401   0.26%   0.53%   1.06%   2.12%   4.24%
# 출력:   336px    577   0.19%   0.38%   0.75%   1.51%   3.02%
# 출력:   392px    785   0.14%   0.28%   0.57%   1.14%   2.27%
# 출력:   448px   1025   0.11%   0.22%   0.45%   0.89%   1.79%
# 출력:   518px   1370   0.09%   0.17%   0.35%   0.69%   1.38%
# 출력:   616px   1937   0.06%   0.13%   0.26%   0.51%   1.03%
# 출력:   700px   2501   0.05%   0.10%   0.21%   0.41%   0.83%
# 출력:   784px   3137   0.04%   0.09%   0.17%   0.34%   0.68%
# 출력:   896px   4097   0.03%   0.07%   0.14%   0.27%   0.55%
# 출력:  1036px   5477   0.03%   0.05%   0.11%   0.22%   0.43%


# %% [markdown]
# 저해상도일수록 register가 비싸다: 112px(N=65)에서 R=16이면 **24.9%** 나 는다.
# 반대로 518px(N=1370)에서는 R=16이어도 1.4%다.
# 논문이 쓰는 224px + R=4 조합은 딱 "거의 공짜" 구간에 있다.

# %% [markdown]
# ## 5. torch로 해석식 검증
#
# 위 식이 실제 코드와 맞는지 `torch.utils.flop_counter.FlopCounterMode` 로 확인한다.
# 작은 ViT 블록(d=64, heads=4, r=4)을 CPU에 만들고 실제 FLOP을 센다.
# FlopCounterMode는 matmul을 $2MNK$ FLOP으로 세므로 예측값은
# $2 \times (12Nd^2 + 2N^2d)$ 이다 (bias·LayerNorm·softmax는 세지 않음).

# %%
import torch
import torch.nn as nn


class TinyBlock(nn.Module):
    """attention을 explicit matmul로 쓴 최소 ViT 블록 (FLOP 카운팅이 명확해지도록)."""

    def __init__(self, d=64, heads=4, r=4):
        super().__init__()
        self.h = heads
        self.d = d
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, r * d), nn.Linear(r * d, d)

    def forward(self, x):
        B, N, d = x.shape
        h, hd = self.h, d // self.h
        y = self.n1(x)
        q, k, v = self.qkv(y).reshape(B, N, 3, h, hd).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) * hd**-0.5
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, N, d)
        x = x + self.proj(y)
        x = x + self.fc2(torch.nn.functional.gelu(self.fc1(self.n2(x))))
        return x


try:
    from torch.utils.flop_counter import FlopCounterMode

    d_small, r_small = 64, 4
    blk = TinyBlock(d_small, heads=4, r=r_small).eval()
    print(f"{'N':>5} {'측정 FLOP':>14} {'해석식 FLOP':>14} {'비율':>8}")
    print("-" * 46)
    meas = {}
    for N in (32, 64, 257, 273):
        x = torch.randn(1, N, d_small)
        fc = FlopCounterMode(display=False)
        with fc, torch.no_grad():
            blk(x)
        got = fc.get_total_flops()
        pred = 2 * block_macs(N, d_small, r_small)
        meas[N] = got
        print(f"{N:>5} {got:>14,} {int(pred):>14,} {got/pred:>8.4f}")

    # register 오버헤드도 실측으로 확인: N=257 -> 273 (R=16)
    print()
    print(f"실측 R=16 오버헤드 (N 257->273): {meas[273]/meas[257]-1:.2%}")
    print(f"해석식 R=16 오버헤드 (블록 기준): {block_macs(273, d_small, r_small)/block_macs(257, d_small, r_small)-1:.2%}")
except Exception as e:  # noqa: BLE001
    print("FlopCounterMode 검증 불가 -> 해석식만 사용:", repr(e))

# 출력:     N        측정 FLOP       해석식 FLOP       비율
# 출력: ----------------------------------------------
# 출력:    32      3,407,872      3,407,872   1.0000
# 출력:    64      7,340,032      7,340,032   1.0000
# 출력:   257     42,172,672     42,172,672   1.0000
# 출력:   273     45,916,416     45,916,416   1.0000
# 출력:
# 출력: 실측 R=16 오버헤드 (N 257->273): 8.88%
# 출력: 해석식 R=16 오버헤드 (블록 기준): 8.88%


# %% [markdown]
# 해석식이 실측과 **정확히 일치**한다 (비율 1.0000).
#
# 위 tiny 블록은 $d=64$ 로 작아서 $N^2d$ 항의 비중이 커지고, 그래서 오버헤드가
# 8.88%로 ViT-L의 6.49%보다 크게 나온다. $d$가 클수록 $12Nd^2$ 항이 지배해서
# 오버헤드가 $R/N$ 에 수렴한다 — 큰 모델일수록 register가 상대적으로 더 싸다.

# %%
# d를 키우면서 R=16 오버헤드가 R/N (=6.23%) 로 수렴하는지 확인
N = N_BASE
print(f"{'d':>6} {'R=16 오버헤드':>14}   (하한 R/N = {16/N:.2%})")
for d in (64, 128, 256, 512, 1024, 2048, 4096):
    ov = block_macs(N + 16, d) / block_macs(N, d) - 1
    print(f"{d:>6} {ov:>14.3%}")

# 출력:      d      R=16 오버헤드   (하한 R/N = 6.23%)
# 출력:     64         8.877%
# 출력:    128         7.884%
# 출력:    256         7.174%
# 출력:    512         6.736%
# 출력:   1024         6.491%
# 출력:   2048         6.361%
# 출력:   4096         6.294%


# %% [markdown]
# ## 정리
#
# | 항목 | R=4 (논문 기본값) | R=16 (최대) |
# |---|---|---|
# | 추가 파라미터 (ViT-L, d=1024) | 4,096개 = **13.5 ppm** | 16,384개 = **54 ppm** |
# | FLOP 증가 (224px, N=257) | **1.6%** (논문: <2%) | **6.5%** (논문: ~6%) |
# | FLOP 증가 (518px, N=1370) | 0.35% | 1.38% |
#
# * register는 **새로운 연산자가 아니라 시퀀스 길이 $N \to N+R$** 일 뿐이다.
# * 파라미터는 $R \times d$ 만 늘어 사실상 0에 수렴 → "negligible".
# * FLOP은 $\approx R/N$ 만큼 늘어 224px·R=4에서 2% 미만 → "slight".
# * 그 2% 미만의 비용으로 artifact 제거 + dense task 성능 향상 + 깔끔한 attention map을
#   얻으므로, 논문이 R=4를 기본값으로 채택한 것은 비용/효과 면에서 자명한 선택이다.
