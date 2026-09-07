# %% [markdown]
# # artifact patch의 정량적 판별 기준: output token norm과 150 cutoff
#
# 논문: *Vision Transformers Need Registers* (Darcet et al., ICLR 2024), §2.1
#
# **핵심 주장**
# - artifact patch와 정상 patch를 가르는 관측량은 **모델 출력단 patch token의 L2 norm**
#   $$\lVert x_i^{(L)} \rVert_2,\qquad x_i^{(L)}\in\mathbb{R}^d$$
# - DINOv2에서 이 norm의 분포는 **bimodal** → 두 봉우리 사이 "골짜기"에 cutoff를 찍으면 된다.
# - 논문이 고른 값이 **150**이고, 그때 outlier 비율은 **2.37%**.
# - 단, 150은 hand-picked 값이며 모델마다 달라진다.
#
# 여기서는 실제 DINOv2 가중치 없이 **합성 분포**로 (a) bimodality, (b) cutoff 선택 문제,
# (c) 임계값 민감도를 보인다.
#
# 필요 패키지: numpy, plotly, kaleido
# 실행 인터프리터: `/home/sungwoo/miniforge3/envs/trellis/bin/python`

# %%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


rng = np.random.default_rng(0)
print("numpy", np.__version__)
# 출력: numpy 1.26.4

# %% [markdown]
# ## 1. patch token norm 분포를 합성한다
#
# Fig. 3(오른쪽)의 모양을 그대로 흉내낸다.
#
# - **DINO (artifact 없음)**: 단봉. norm이 $0\sim50$에 몰려 있고 200 이상은 사실상 없다.
# - **DINOv2 (artifact 있음)**: 혼합 분포
#   $$p(n) = (1-w)\,p_{\text{normal}}(n) \;+\; w\,p_{\text{outlier}}(n)$$
#   - $p_{\text{normal}}$: lognormal, 대부분 $0\sim100$
#   - $p_{\text{outlier}}$: $400\!\sim\!500$ 근처의 넓고 낮은 봉우리
#   - 혼합 비율 $w$는 **norm > 150 비율이 2.37%** 가 되도록 맞춘다.

# %%
N_IMAGES, N_PATCHES = 400, 256  # 400장 x 16x16 patch
N = N_IMAGES * N_PATCHES

W_OUT = 0.0238  # outlier 성분의 혼합 비율 (논문 2.37%에 맞춰 조정)


def sample_normal_tokens(n, rng):
    """정상 patch token: norm이 0~100에 몰린 lognormal."""
    return np.exp(rng.normal(np.log(38.0), 0.32, size=n))


def sample_outlier_tokens(n, rng):
    """high-norm patch token: 400~500 근처의 넓은 봉우리 (아래로 긴 꼬리)."""
    x = rng.beta(4.0, 1.6, size=n)  # 0~1, 오른쪽으로 치우침
    return 60.0 + 520.0 * x  # 60~580, 최빈 ~450 (아래쪽으로 얇은 꼬리를 남겨 둔다)


is_out = rng.random(N) < W_OUT
norms_v2 = np.where(is_out, sample_outlier_tokens(N, rng), sample_normal_tokens(N, rng))
norms_dino = np.exp(rng.normal(np.log(22.0), 0.30, size=N))  # 단봉, artifact 없음

ratio_150 = (norms_v2 > 150).mean()
print(f"DINOv2  norm>150 비율 = {ratio_150 * 100:.2f}%  (논문: 2.37%)")
print(f"DINO    norm>150 비율 = {(norms_dino > 150).mean() * 100:.4f}%")
print(f"정상 토큰 median={np.median(norms_v2[~is_out]):.1f}, "
      f"outlier median={np.median(norms_v2[is_out]):.1f}, "
      f"배율={np.median(norms_v2[is_out]) / np.median(norms_v2[~is_out]):.1f}x")
# 출력: DINOv2  norm>150 비율 = 2.34%  (논문: 2.37%)
# 출력: DINO    norm>150 비율 = 0.0000%
# 출력: 정상 토큰 median=38.0, outlier median=443.2, 배율=11.7x
#   → 논문의 "roughly 10x higher norm", "around 2% of the sequence"와 같은 자릿수

# %% [markdown]
# ## 2. bimodality 확인: 골짜기(valley)가 어디인가
#
# 로그 y축 히스토그램에서 두 봉우리 사이의 **밀도 최소 구간**을 찾는다.
# cutoff는 이 골짜기 안 어디에 두어도 판정 결과가 거의 안 변한다.

# %%
counts, edges = np.histogram(norms_v2, bins=120, range=(0, 600))
centers = 0.5 * (edges[:-1] + edges[1:])

# 두 봉우리: 첫 봉우리(<150 영역)와 둘째 봉우리(>250 영역)
peak1 = centers[np.argmax(np.where(centers < 150, counts, -1))]
peak2 = centers[np.argmax(np.where(centers > 250, counts, -1))]

# 그 사이에서 밀도가 가장 낮은 지점 = valley
mid = (centers > peak1) & (centers < peak2)
valley_center = centers[mid][np.argmin(counts[mid])]

print(f"peak1 (정상 mode)    ≈ {peak1:.0f}  (bin count {counts[centers == peak1][0]})")
print(f"peak2 (outlier mode) ≈ {peak2:.0f}  (bin count {counts[centers == peak2][0]})")
print(f"valley 최저점        ≈ {valley_center:.0f}  (bin count {counts[mid].min()})")
n_valley = counts[(centers > 100) & (centers < 260)].sum()
print(f"100<norm<260 구간의 토큰 수 = {n_valley} / {counts.sum()} "
      f"(전체의 {n_valley / counts.sum() * 100:.3f}%)")
# 출력: peak1 (정상 mode)    ≈ 32  (bin count 16815)
# 출력: peak2 (outlier mode) ≈ 462  (bin count 66)
# 출력: valley 최저점        ≈ 182  (bin count 0)
# 출력: 100<norm<260 구간의 토큰 수 = 235 / 102400 (전체의 0.229%)
#   → 두 봉우리 사이는 사실상 비어 있다. 이 구간 어디에 선을 그어도 판정 결과가 거의 같다.

# %% [markdown]
# ## 3. cutoff 민감도: 임계값을 바꾸면 outlier 비율이 얼마나 흔들리나
#
# $$r(t) = \frac{1}{N}\sum_i \mathbb{1}\!\left[\lVert x_i^{(L)}\rVert_2 > t\right]$$
#
# 골짜기 안에서는 $r(t)$가 거의 평평하다(= cutoff 선택에 둔감).
# 봉우리 위에서는 $t$를 조금만 옮겨도 $r$이 급변한다(= 기준으로 못 씀).
# 민감도는 $\left|\dfrac{d\log r}{dt}\right|$ 로 본다.

# %%
ts = np.arange(20, 601, 2.0)
ratio = np.array([(norms_v2 > t).mean() for t in ts])
sens = np.abs(np.gradient(np.log(np.maximum(ratio, 1e-12)), ts))  # |d log r / dt|

for t in (40.0, 60.0, 100.0, 150.0, 200.0, 250.0, 400.0):
    r = (norms_v2 > t).mean()
    s = sens[np.argmin(np.abs(ts - t))]
    print(f"t={t:5.0f}  outlier 비율={r * 100:6.2f}%   |dlog r/dt|={s:.5f}")

# r(150)에서 5% 이내로만 벗어나는 t 대역 = "cutoff를 아무 데나 둬도 되는" 안전 구간
safe = ts[np.abs(ratio / ratio_150 - 1.0) <= 0.05]
safe_zone = (safe.min(), safe.max())
print(f"\n안전 구간(r(t)가 r(150)의 ±5% 이내) = [{safe_zone[0]:.0f}, {safe_zone[1]:.0f}]  폭 {safe_zone[1] - safe_zone[0]:.0f}")
print(f"논문 cutoff 150 이 이 안에 있는가? {safe_zone[0] <= 150 <= safe_zone[1]}")
# 출력: t=   40  outlier 비율= 45.02%   |dlog r/dt|=0.06633
# 출력: t=   60  outlier 비율=  9.86%   |dlog r/dt|=0.07371
# 출력: t=  100  outlier 비율=  2.47%   |dlog r/dt|=0.00415
# 출력: t=  150  outlier 비율=  2.34%   |dlog r/dt|=0.00010
# 출력: t=  200  outlier 비율=  2.33%   |dlog r/dt|=0.00021
# 출력: t=  250  outlier 비율=  2.26%   |dlog r/dt|=0.00119
# 출력: t=  400  outlier 비율=  1.56%   |dlog r/dt|=0.00612
# 출력:
# 출력: 안전 구간(r(t)가 r(150)의 ±5% 이내) = [102, 264]  폭 162
# 출력: 논문 cutoff 150 이 이 안에 있는가? True
#   → t를 102~264 어디로 옮겨도 outlier 집합이 5% 이내로만 바뀐다. 150은 이 평지 한가운데다.
#   → 반대로 t=40~60(첫 봉우리 옆구리)에서는 민감도가 2~3자릿수 크고, 비율이 45%까지 폭발한다.

# %% [markdown]
# ## 4. 반례: 150은 보편 상수가 아니다
#
# Fig. 7의 y축을 보면 DeiT-III는 **정상 토큰조차 norm이 300~500**이다.
# 같은 150을 들이대면 전 토큰이 "artifact"로 판정된다.
# → 외울 것은 숫자가 아니라 **"분포를 그리고 골짜기를 찾는다"는 절차**.

# %%
deit_normal = np.exp(rng.normal(np.log(400.0), 0.12, size=N))
deit_out = 700.0 + 700.0 * rng.beta(2.5, 2.0, size=int(N * 0.03))
norms_deit = np.concatenate([deit_normal, deit_out])

print(f"DeiT-III류 분포에 t=150 적용 → outlier 판정 비율 = {(norms_deit > 150).mean() * 100:.1f}%")
counts_d, edges_d = np.histogram(norms_deit, bins=150, range=(0, 1600))
c_d = 0.5 * (edges_d[:-1] + edges_d[1:])
mid_d = (c_d > 550) & (c_d < 900)
t_deit = c_d[mid_d][np.argmin(counts_d[mid_d])]
print(f"이 분포의 골짜기에서 다시 고른 cutoff ≈ {t_deit:.0f} "
      f"→ outlier 비율 = {(norms_deit > t_deit).mean() * 100:.1f}%")
# 출력: DeiT-III류 분포에 t=150 적용 → outlier 판정 비율 = 100.0%
# 출력: 이 분포의 골짜기에서 다시 고른 cutoff ≈ 645 → outlier 비율 = 2.9%

# %% [markdown]
# ## 5. 공간 맵으로 보기 (Fig. 3 왼쪽 재현)
#
# 16x16 patch grid에 norm을 배치. outlier는 논문 관찰대로 **가장자리/배경 쪽**에 몰리게 둔다
# (부록 Fig. 10: outlier는 중앙보다 경계 근처에 잘 나타난다).

# %%
G = 16
grid = sample_normal_tokens(G * G, rng).reshape(G, G)

yy, xx = np.mgrid[0:G, 0:G]
d_border = np.minimum.reduce([xx, yy, G - 1 - xx, G - 1 - yy])  # 경계까지 거리
w_pos = np.exp(-d_border / 1.5).ravel()
w_pos /= w_pos.sum()
idx = rng.choice(G * G, size=6, replace=False, p=w_pos)
grid.ravel()[idx] = sample_outlier_tokens(6, rng)

print(f"grid 내 norm>150 패치 수 = {(grid > 150).sum()} / {G * G}"
      f"  ({(grid > 150).mean() * 100:.1f}%)")
print(f"정상 패치 norm 범위 = [{grid[grid <= 150].min():.0f}, {grid[grid <= 150].max():.0f}]")
print(f"outlier 패치 norm   = {np.sort(grid[grid > 150])[::-1].round(0)}")
# 출력: grid 내 norm>150 패치 수 = 6 / 256  (2.3%)
# 출력: 정상 패치 norm 범위 = [16, 103]
# 출력: outlier 패치 norm   = [527. 469. 466. 426. 426. 345.]

# %% [markdown]
# ## 6. 종합 시각화

# %%
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "(a) output patch token norm 분포 — DINOv2는 bimodal",
        "(b) 16x16 patch norm map — 배경 쪽 소수가 튄다",
        "(c) cutoff t 에 따른 outlier 비율 r(t)",
        "(d) 민감도 |d log r / dt| — 골짜기에서 최소",
    ),
    specs=[[{"type": "xy"}, {"type": "heatmap"}], [{"type": "xy"}, {"type": "xy"}]],
    vertical_spacing=0.14,
    horizontal_spacing=0.11,
)

# (a) 히스토그램
fig.add_trace(
    go.Histogram(x=norms_dino, xbins=dict(start=0, end=600, size=5),
                 name="DINO (artifact 없음)", marker_color="#8FB8DE",
                 histnorm="probability", opacity=0.85),
    row=1, col=1,
)
fig.add_trace(
    go.Histogram(x=norms_v2, xbins=dict(start=0, end=600, size=5),
                 name="DINOv2 (artifact 있음)", marker_color="#2E5E8A",
                 histnorm="probability", opacity=0.85),
    row=1, col=1,
)
fig.add_vline(x=150, line=dict(color="#C0392B", width=2, dash="dash"), row=1, col=1)
fig.add_annotation(x=150, y=0.0, xref="x", yref="paper",
                   text=f"t=150<br>{ratio_150 * 100:.2f}%", showarrow=False,
                   xanchor="left", yanchor="bottom", font=dict(color="#C0392B", size=11),
                   row=1, col=1)

# (b) norm map
fig.add_trace(
    go.Heatmap(z=grid, colorscale="Viridis", zmin=0, zmax=100,
               colorbar=dict(title="norm", len=0.38, y=0.81, x=1.005),
               hovertemplate="norm=%{z:.0f}<extra></extra>"),
    row=1, col=2,
)

# (c) r(t)
fig.add_trace(
    go.Scatter(x=ts, y=ratio * 100, mode="lines", name="r(t)",
               line=dict(color="#2E5E8A", width=2.5), showlegend=False),
    row=2, col=1,
)
fig.add_vrect(x0=safe_zone[0], x1=safe_zone[1], fillcolor="#F4D06F",
              opacity=0.35, line_width=0, row=2, col=1)
fig.add_vline(x=150, line=dict(color="#C0392B", width=2, dash="dash"), row=2, col=1)
fig.add_trace(
    go.Scatter(x=[150], y=[ratio_150 * 100], mode="markers+text",
               marker=dict(color="#C0392B", size=10),
               text=[f" {ratio_150 * 100:.2f}%"], textposition="middle right",
               showlegend=False, textfont=dict(color="#C0392B", size=11)),
    row=2, col=1,
)

# (d) 민감도
fig.add_trace(
    go.Scatter(x=ts, y=sens, mode="lines", name="|d log r/dt|",
               line=dict(color="#7B5EA7", width=2.5), showlegend=False),
    row=2, col=2,
)
fig.add_vrect(x0=safe_zone[0], x1=safe_zone[1], fillcolor="#F4D06F",
              opacity=0.35, line_width=0, row=2, col=2)
fig.add_vline(x=150, line=dict(color="#C0392B", width=2, dash="dash"), row=2, col=2)

fig.update_yaxes(type="log", title_text="확률", row=1, col=1)
fig.update_xaxes(title_text="L2 norm", row=1, col=1)
fig.update_xaxes(showticklabels=False, row=1, col=2)
fig.update_yaxes(showticklabels=False, autorange="reversed", scaleanchor="x2", row=1, col=2)
fig.update_xaxes(title_text="cutoff t", row=2, col=1)
fig.update_yaxes(type="log", title_text="outlier 비율 (%)", row=2, col=1)
fig.update_xaxes(title_text="cutoff t", row=2, col=2)
fig.update_yaxes(type="log", title_text="|d log r / dt|", row=2, col=2)

fig.update_layout(
    title="artifact 판별 기준 = output token norm, cutoff는 bimodal 분포의 골짜기(합성 데이터)",
    barmode="overlay",
    template="plotly_white",
    width=1150,
    height=760,
    legend=dict(x=0.22, y=0.93, xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.75)", bordercolor="#CCCCCC",
                borderwidth=1, font=dict(size=11)),
    font=dict(size=12),
)

_show(fig)
fig.write_image("expy.png", scale=2)
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## 정리
#
# 1. 판별 관측량 = **출력단 patch token의 L2 norm** (attention 밝기도, [CLS]도, 중간 레이어도 아니다).
# 2. DINOv2에서 이 분포는 **bimodal** → 두 mode 사이 골짜기에 cutoff를 둔다. 논문 값 **150**, 그때 **2.37%**.
# 3. 골짜기 안에서는 cutoff를 꽤 옮겨도 판정이 거의 안 변한다(셀 3의 민감도).
#    이 **둔감함**이 "150이라는 임의의 숫자를 써도 되는" 근거다.
# 4. 하지만 150은 **hand-picked**다. DeiT-III처럼 스케일이 다른 모델에서는 완전히 무의미해진다(셀 4).
# 5. 이 기준으로 갈라놓은 뒤 논문은 outlier가 (i) 이웃과 유사한 중복 배경 패치에 생기고,
#    (ii) 위치·픽셀 정보를 덜 담고, (iii) 이미지 전역 정보를 더 담는다는 것을 보였고,
#    → 그 역할을 대신할 **register token**을 처방한다.
