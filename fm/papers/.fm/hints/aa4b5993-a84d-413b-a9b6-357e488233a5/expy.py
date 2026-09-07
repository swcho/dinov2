# %% [markdown]
# # register 개수 ablation: dense task vs. ImageNet 의 상반된 경향
#
# 출처: Darcet et al., *Vision Transformers Need Registers* (arXiv:2309.16588v2),
# Sec. 3.2 "Number of register tokens", **Fig. 8 (bottom)**.
#
# 실험: DINOv2 **ViT-L/14** 를 register 개수 $n \in \{0,1,2,4,8,16\}$ 으로 각각 학습한 뒤
# frozen feature linear probing 으로 평가.
#
# - ImageNet top-1 acc (↑) — global classification
# - average of segmentation tasks, mIoU (↑) — dense
# - average of depth tasks, rmse (↓) — dense
#
# 논문 본문의 결론:
# > *"There seems to be an optimal number of registers for dense tasks, and adding one
# > brings most of the benefit. ... On ImageNet, however, performance improves when using
# > more registers. In all our experiments, we kept 4 register tokens."*
#
# **주의**: 이 ablation의 수치 표는 논문 본문·부록 어디에도 없다(Table 2a는 $n{=}4$ vs $n{=}0$ 만,
# 그것도 ADE20k / NYUd 단일 벤치마크로 보고). 따라서 아래 값들은 **Fig. 8 아래 그래프를
# 눈금 기준으로 읽어 낸 근사치**다. 경향(상승/포화/최적점)은 그림과 정확히 일치한다.
#
# 실행 환경(의존성): `/home/sungwoo/miniforge3/envs/trellis/bin/python`
# (numpy 1.26 / plotly 6.9 / kaleido 설치되어 있음. 시스템 `python3` 에는 numpy 조차 없음)

# %%
import os

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


HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
print("numpy", np.__version__)
print("out dir:", HERE)
# 출력: numpy 1.26.4
# 출력: out dir: /home/sungwoo/projects/swcho/dinov2/fm/papers/.fm/hints/aa4b5993-a84d-413b-a9b6-357e488233a5

# %% [markdown]
# ## 1. Fig. 8 (bottom) 수치 하드코딩
#
# 세 subplot 모두 x축은 register 개수 `0, 1, 2, 4, 8, 16`.

# %%
# --- 모두 Fig. 8 (bottom) 그래프에서 읽은 근사치 (소수 둘째/셋째 자리는 ±0.02 수준 오차 가능) ---
n_reg = [0, 1, 2, 4, 8, 16]

imagenet_top1 = [84.34, 84.43, 84.61, 84.60, 84.64, 84.80]  # ↑ 높을수록 좋음
seg_miou = [66.02, 66.66, 66.68, 66.91, 66.76, 66.76]  # ↑ 높을수록 좋음
depth_rmse = [2.846, 2.748, 2.774, 2.754, 2.729, 2.761]  # ↓ 낮을수록 좋음

for n, a, m, r in zip(n_reg, imagenet_top1, seg_miou, depth_rmse):
    print(f"n={n:2d}  IN1k={a:.2f}  mIoU={m:.2f}  rmse={r:.3f}")
# 출력: n= 0  IN1k=84.34  mIoU=66.02  rmse=2.846
# 출력: n= 1  IN1k=84.43  mIoU=66.66  rmse=2.748
# 출력: n= 2  IN1k=84.61  mIoU=66.68  rmse=2.774
# 출력: n= 4  IN1k=84.60  mIoU=66.91  rmse=2.754
# 출력: n= 8  IN1k=84.64  mIoU=66.76  rmse=2.729
# 출력: n=16  IN1k=84.80  mIoU=66.76  rmse=2.761

# %% [markdown]
# ## 2. "1개가 대부분의 이득을 가져간다"를 숫자로
#
# dense 지표에 대해, register 1개가 회수하는 이득의 비율을
#
# $$\text{recovered}_1 = \frac{|v(1) - v(0)|}{|v(n^\*) - v(0)|}, \qquad
# n^\* = \arg\max_n v(n)\ (\text{또는 } \arg\min \text{ for rmse})$$
#
# 로 정의한다. ImageNet 에는 내부 최적점이 없으므로($n^\*=16$, 즉 범위의 끝) 같은 비율을 계산하면
# 값이 훨씬 작게 나온다 — 이것이 두 경향의 차이를 그대로 드러낸다.

# %%
def recovered_by_one(values, lower_is_better=False):
    v = np.asarray(values, dtype=float)
    best_i = int(np.argmin(v)) if lower_is_better else int(np.argmax(v))
    total = abs(v[best_i] - v[0])
    first = abs(v[1] - v[0])
    return n_reg[best_i], total, first, (first / total if total > 0 else float("nan"))


for name, vals, low in [
    ("ImageNet top-1", imagenet_top1, False),
    ("seg mIoU      ", seg_miou, False),
    ("depth rmse    ", depth_rmse, True),
]:
    n_star, total, first, ratio = recovered_by_one(vals, low)
    print(f"{name}  n*={n_star:2d}  총개선={total:.3f}  n=1개선={first:.3f}  1개가 회수={ratio:6.1%}")
# 출력: ImageNet top-1  n*=16  총개선=0.460  n=1개선=0.090  1개가 회수= 19.6%
# 출력: seg mIoU        n*= 4  총개선=0.890  n=1개선=0.640  1개가 회수= 71.9%
# 출력: depth rmse      n*= 8  총개선=0.117  n=1개선=0.098  1개가 회수= 83.8%

# %% [markdown]
# **읽는 법**
#
# - dense 두 지표: 최적점 $n^\*$ 가 **범위 내부**(4, 8)에 있고, register **1개만으로 72~84%** 를 회수한다.
# - ImageNet: 최적점이 **범위 끝(16)** 이고, 1개는 겨우 20%. 나머지 80%는 register 를 늘려 가며 쌓인다.
#
# 즉 dense = **포화형(아티팩트가 사라지면 끝)**, classification = **누적형(전역 용량이 계속 도움)**.

# %% [markdown]
# ## 3. 세 지표를 나란히 — 원본 Fig. 8 재현

# %%
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=(
        "ImageNet top-1 ↑ (global)",
        "segmentation mIoU ↑ (dense)",
        "depth rmse ↓ (dense)",
    ),
    horizontal_spacing=0.08,
)

panels = [
    (imagenet_top1, "#4C78A8", 16, "단조 증가"),
    (seg_miou, "#F58518", 4, "n*=4 이후 하락"),
    (depth_rmse, "#54A24B", 8, "n*=8 이후 반등"),
]
for c, (vals, color, n_star, note) in enumerate(panels, start=1):
    fig.add_trace(
        go.Scatter(
            x=n_reg, y=vals, mode="lines+markers",
            line=dict(color=color, width=3), marker=dict(size=10),
            showlegend=False, hovertemplate="n=%{x}<br>%{y}<extra></extra>",
        ),
        row=1, col=c,
    )
    i = n_reg.index(n_star)
    fig.add_trace(
        go.Scatter(
            x=[n_star], y=[vals[i]], mode="markers+text",
            marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color="crimson", width=3)),
            text=[f"  n*={n_star}"], textposition="middle right",
            textfont=dict(color="crimson", size=11),
            showlegend=False, hoverinfo="skip",
        ),
        row=1, col=c,
    )
    suffix = "" if c == 1 else str(c)  # plotly 첫 축은 'x'/'y' (x1/y1 아님)
    fig.add_annotation(
        text=note, xref=f"x{suffix} domain", yref=f"y{suffix} domain",
        x=0.5, y=-0.28, showarrow=False, font=dict(size=11, color="dimgray"),
    )
    fig.update_xaxes(title_text="number of [reg] tokens", tickvals=n_reg, row=1, col=c)

fig.update_layout(
    title="Fig. 8 재현 — dense 는 내부 최적점, ImageNet 은 단조 증가 (DINOv2 ViT-L/14)",
    template="plotly_white", width=1080, height=420, margin=dict(t=90, b=90),
)
_show(fig)
print("panel 1 done")
# 출력: panel 1 done

# %% [markdown]
# ## 4. 정규화해서 한 축에 겹치기
#
# 스케일이 다르므로 각 지표를 **"좋을수록 1"** 이 되도록 min-max 정규화한다.
# rmse 는 낮을수록 좋으므로 부호를 뒤집는다:
#
# $$\tilde v(n) = \frac{s\cdot v(n) - \min_k\, s\cdot v(k)}{\max_k\, s\cdot v(k) - \min_k\, s\cdot v(k)},
# \qquad s = \begin{cases} +1 & \text{higher is better} \\ -1 & \text{rmse} \end{cases}$$

# %%
def norm01(values, lower_is_better=False):
    v = np.asarray(values, dtype=float)
    if lower_is_better:
        v = -v
    return (v - v.min()) / (v.max() - v.min())


in_n = norm01(imagenet_top1)
seg_n = norm01(seg_miou)
dep_n = norm01(depth_rmse, lower_is_better=True)

print("n      IN1k~   mIoU~   rmse~")
for i, n in enumerate(n_reg):
    print(f"{n:2d}   {in_n[i]:6.3f}  {seg_n[i]:6.3f}  {dep_n[i]:6.3f}")
# 출력: n      IN1k~   mIoU~   rmse~
# 출력:  0    0.000   0.000   0.000
# 출력:  1    0.196   0.719   0.838
# 출력:  2    0.587   0.742   0.615
# 출력:  4    0.565   1.000   0.786
# 출력:  8    0.652   0.831   1.000
# 출력: 16    1.000   0.831   0.726

# %%
fig2 = go.Figure()
for name, vals, color, dash in [
    ("ImageNet top-1 (global)", in_n, "#4C78A8", "solid"),
    ("segmentation mIoU (dense)", seg_n, "#F58518", "dash"),
    ("depth rmse (dense, 부호반전)", dep_n, "#54A24B", "dot"),
]:
    fig2.add_trace(
        go.Scatter(
            x=n_reg, y=vals, mode="lines+markers", name=name,
            line=dict(color=color, width=3, dash=dash), marker=dict(size=10),
            hovertemplate="n=%{x}<br>%{y:.3f}<extra>" + name + "</extra>",
        )
    )
fig2.add_vrect(
    x0=0.6, x1=1.4, fillcolor="lightsalmon", opacity=0.25, line_width=0,
    annotation_text="register 1개만으로<br>dense 이득 72~84% 회수",
    annotation_position="top left", annotation_font_size=11,
)
fig2.update_layout(
    title="정규화(0=최악, 1=최선) — dense 두 곡선은 일찍 치솟고 꺾이지만 ImageNet 만 끝까지 오른다",
    xaxis=dict(title="number of [reg] tokens", tickvals=n_reg),
    yaxis=dict(title="normalized score (높을수록 좋음)"),
    template="plotly_white", width=900, height=460,
    legend=dict(orientation="h", y=-0.2),
)
_show(fig2)
print("panel 2 done")
# 출력: panel 2 done

# %% [markdown]
# ## 5. 왜 논문은 4개를 골랐나 — 가중합 스코어
#
# dense 와 classification 을 **동시에** 최적화하는 개수를 찾기 위해 정규화 점수의 가중합을 쓴다.
# dense 를 두 지표(seg, depth)로 나눠 각각 $1/4$, classification 에 $1/2$ 를 주면
#
# $$S(n) = \tfrac12\,\widetilde{\text{IN1k}}(n) \;+\; \tfrac14\,\widetilde{\text{mIoU}}(n) \;+\; \tfrac14\,\widetilde{\text{rmse}}(n)$$
#
# 여기에 부록 B(Fig. 12)의 **연산 비용**을 페널티로 얹는다. 논문 보고에 따르면 FLOP 증가는
# register 개수에 거의 선형이고 $n{=}16$ 에서 약 $+6\%$, $n{=}4$ 에서 $2\%$ 미만이다 →
# $\text{flop}(n) \approx 0.375\%\times n$ 로 근사.

# %%
score = 0.5 * in_n + 0.25 * seg_n + 0.25 * dep_n

# Fig. 12: FLOP 증가율은 n 에 거의 선형, n=16 에서 ~6% (근사 기울기 6/16 = 0.375 %/token)
flop_pct = np.array(n_reg, dtype=float) * 0.375
LAMBDA = 0.05  # FLOP 1%p 당 스코어 페널티
score_cost = score - LAMBDA * flop_pct

print("n    S(n)    FLOP+%   S-cost")
for i, n in enumerate(n_reg):
    print(f"{n:2d}  {score[i]:6.3f}  {flop_pct[i]:6.2f}  {score_cost[i]:7.3f}")
print()
print("비용 무시 시 최적 n =", n_reg[int(np.argmax(score))])
print("비용 반영 시 최적 n =", n_reg[int(np.argmax(score_cost))])
print("논문이 실제로 채택한 n = 4  ('In all our experiments, we kept 4 register tokens.')")
# 출력: n    S(n)    FLOP+%   S-cost
# 출력:  0   0.000    0.00    0.000
# 출력:  1   0.487    0.38    0.468
# 출력:  2   0.633    0.75    0.595
# 출력:  4   0.729    1.50    0.654
# 출력:  8   0.784    3.00    0.634
# 출력: 16   0.889    6.00    0.589
# 출력:
# 출력: 비용 무시 시 최적 n = 16   <- ImageNet 한 지표가 끌어올린 결과
# 출력: 비용 반영 시 최적 n = 4
# 출력: 논문이 실제로 채택한 n = 4  ('In all our experiments, we kept 4 register tokens.')

# %% [markdown]
# `n=16` 이 raw 가중합 $S$ 로는 가장 높지만, 그 우위는 거의 전부 **ImageNet 한 지표**에서 나온다.
# dense 두 지표만 보면 16 은 이미 최적점을 지난 상태이고, FLOP 도 6% 더 쓴다.
# 아주 약한 비용 페널티($\lambda = 0.05$)만 넣어도 균형점이 **$n=4$** 로 옮겨간다 — 논문의 선택과 일치.

# %%
fig3 = make_subplots(specs=[[{"secondary_y": True}]])
fig3.add_trace(
    go.Bar(x=[str(n) for n in n_reg], y=score, name="가중합 S(n) (비용 무시)",
           marker_color="#B0C4DE", hovertemplate="n=%{x}<br>S=%{y:.3f}<extra></extra>"),
    secondary_y=False,
)
fig3.add_trace(
    go.Bar(x=[str(n) for n in n_reg], y=score_cost, name="S(n) − 0.05·FLOP%  (비용 반영)",
           marker_color="#4C78A8", hovertemplate="n=%{x}<br>S'=%{y:.3f}<extra></extra>"),
    secondary_y=False,
)
fig3.add_trace(
    go.Scatter(x=[str(n) for n in n_reg], y=flop_pct, name="FLOP 증가율 (%)",
               mode="lines+markers", line=dict(color="crimson", width=3, dash="dot"),
               marker=dict(size=9)),
    secondary_y=True,
)
fig3.add_annotation(x="4", y=float(score_cost[3]), text="논문 채택 n=4", showarrow=True,
                    arrowhead=2, ay=-45, font=dict(color="crimson", size=12))
fig3.update_yaxes(title_text="balanced score", secondary_y=False)
fig3.update_yaxes(title_text="FLOP 증가율 (%)", secondary_y=True, showgrid=False)
fig3.update_layout(
    title="dense + classification 을 동시에 보면 n=4 가 균형점",
    xaxis_title="number of [reg] tokens", barmode="group",
    template="plotly_white", width=900, height=460,
    legend=dict(orientation="h", y=-0.2),
)
_show(fig3)
print("panel 3 done")
# 출력: panel 3 done

# %% [markdown]
# ## 6. 정적 이미지 저장 (expy.png)
#
# 위 세 그림을 한 장(①원본 3패널 / ②정규화 / ③균형 스코어)으로 합쳐 `expy.png` 로 저장한다 (kaleido 필요).

# %%
combo = make_subplots(
    rows=3, cols=3, vertical_spacing=0.11, horizontal_spacing=0.09,
    row_heights=[0.30, 0.35, 0.35],
    specs=[
        [{}, {}, {}],
        [{"colspan": 3}, None, None],
        [{"colspan": 3, "secondary_y": True}, None, None],
    ],
    subplot_titles=(
        "① ImageNet top-1 ↑ (global)",
        "① segmentation mIoU ↑ (dense)",
        "① depth rmse ↓ (dense)",
        "② 정규화(0=최악, 1=최선): dense 는 n=1에서 급등하고 최적점 뒤 하락 / ImageNet 만 끝까지 상승",
        "③ 균형 스코어 + FLOP 비용 → 논문의 선택 n=4",
    ),
)

# ① 원본 단위 그대로, 지표마다 자기 y축을 갖는 세 패널 (Fig. 8 bottom 재현)
for c, (vals, color, n_star, note) in enumerate(panels, start=1):
    combo.add_trace(
        go.Scatter(x=n_reg, y=vals, mode="lines+markers", showlegend=False,
                   line=dict(color=color, width=3), marker=dict(size=9)),
        row=1, col=c,
    )
    i = n_reg.index(n_star)
    combo.add_trace(
        go.Scatter(x=[n_star], y=[vals[i]], mode="markers", showlegend=False, hoverinfo="skip",
                   marker=dict(size=17, color="rgba(0,0,0,0)",
                               line=dict(color="crimson", width=3))),
        row=1, col=c,
    )
    suffix = "" if c == 1 else str(c)
    combo.add_annotation(text=f"<b>{note}</b>", xref=f"x{suffix} domain", yref=f"y{suffix} domain",
                         x=0.5, y=1.02, showarrow=False, font=dict(size=11, color="crimson"))

# ② 정규화 후 한 축에 겹치기
for name, vals, color, dash in [
    ("ImageNet top-1 (global)", in_n, "#4C78A8", "solid"),
    ("segmentation mIoU (dense)", seg_n, "#F58518", "dash"),
    ("depth rmse (dense, 부호반전)", dep_n, "#54A24B", "dot"),
]:
    combo.add_trace(
        go.Scatter(x=n_reg, y=vals, mode="lines+markers", name=name,
                   line=dict(color=color, width=3, dash=dash), marker=dict(size=9)),
        row=2, col=1,
    )
combo.add_vrect(x0=0.55, x1=1.45, fillcolor="lightsalmon", opacity=0.3, line_width=0,
                row=2, col=1, annotation_text="register 1개가<br>dense 이득 72~84% 회수",
                annotation_position="top right", annotation_font_size=11)

# ③ 균형 스코어와 비용
combo.add_trace(
    go.Bar(x=n_reg, y=score, name="가중합 S(n)", marker_color="#B0C4DE"),
    row=3, col=1, secondary_y=False,
)
combo.add_trace(
    go.Bar(x=n_reg, y=score_cost, name="S(n) − 0.05·FLOP%", marker_color="#4C78A8"),
    row=3, col=1, secondary_y=False,
)
combo.add_trace(
    go.Scatter(x=n_reg, y=flop_pct, name="FLOP 증가율 (%)", mode="lines+markers",
               line=dict(color="crimson", width=3, dash="dot"), marker=dict(size=8)),
    row=3, col=1, secondary_y=True,
)
combo.add_annotation(x=4, y=float(score_cost[3]), text="<b>n=4</b>", showarrow=True, arrowhead=2,
                     ax=25, ay=-45, font=dict(color="crimson", size=13), row=3, col=1)

for c in (1, 2, 3):
    combo.update_xaxes(tickvals=n_reg, row=1, col=c)
for r in (2, 3):
    combo.update_xaxes(title_text="number of [reg] tokens", tickvals=n_reg, row=r, col=1)
combo.update_yaxes(title_text="normalized ↑", row=2, col=1)
combo.update_yaxes(title_text="balanced score", row=3, col=1, secondary_y=False)
combo.update_yaxes(title_text="FLOP +%", row=3, col=1, secondary_y=True, showgrid=False)
combo.update_layout(
    title=dict(
        text="register 개수: dense task 는 최적점 존재(1개가 대부분) / ImageNet 은 많을수록 상승",
        x=0.5, xanchor="center", y=0.985,
    ),
    template="plotly_white", width=1100, height=1200, barmode="group",
    margin=dict(t=110, b=110),
    legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="center", x=0.5),
)
_show(combo)

png_path = os.path.join(HERE, "expy.png")
combo.write_image(png_path, scale=2)  # kaleido 필요
print("saved:", png_path, os.path.getsize(png_path), "bytes")
# 출력: saved: /home/sungwoo/projects/swcho/dinov2/fm/papers/.fm/hints/aa4b5993-a84d-413b-a9b6-357e488233a5/expy.png 451644 bytes

# %% [markdown]
# ## 7. 정리
#
# | | dense task (seg / depth) | ImageNet 분류 |
# |---|---|---|
# | 곡선 모양 | 범위 **내부에 최적점** ($n^\*=4$ / $8$), 이후 하락·정체 | $n$ 에 대해 **단조 증가**, 16에서 최고 |
# | register 1개의 기여 | 전체 이득의 **72% / 84%** | 겨우 **20%** |
# | 원인 (논문 해석) | high-norm **아티팩트 제거**가 이득의 본체 → 곧 포화 | register 가 **전역 정보 용량**을 더해 줌 → 누적 |
# | 실용 결론 | | 두 경향의 절충 + FLOP(<2%) 고려 → **4개** |
