# %% [markdown]
# # DINOv2 Table 1: iBOT → DINOv2 점진적 ablation 재현
#
# 논문(Oquab et al., 2023, arXiv:2304.07193) Table 1은 iBOT 베이스라인에 개선 요소를
# **하나씩 누적**시키며 ImageNet-1k 검증셋 Top-1 정확도(k-NN / linear probe)를 기록한다.
# 실험 설정: **ViT-Large, ImageNet-22k 사전학습**.
#
# 이 스크립트는 논문 수치를 그대로 옮겨 놓고
# 1. 각 단계의 변화량 $\Delta$ 을 다시 계산해 논문 표의 ↑/↓ 값과 일치하는지 확인하고,
# 2. 누적 곡선과 단계별 기여도를 plotly로 그린다.
#
# 필요 패키지: plotly, kaleido (expy.png 저장용)

# %%
# 필요 패키지: plotly kaleido
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# %% [markdown]
# ## 1. Table 1 원본 수치
#
# 각 행은 "이전 행의 설정 + 해당 요소"이다. 마지막 행(Untying heads)이 곧 DINOv2 최종 레시피.

# %%
# (단계 이름, k-NN Top-1, linear Top-1) — 논문 Table 1 그대로
TABLE1 = [
    ("iBOT (baseline)",              72.9, 82.3),
    ("+ our reproduction",           74.5, 83.2),
    ("+ LayerScale, Stoch. Depth",   75.4, 82.0),
    ("+ 128k prototypes",            76.6, 81.9),
    ("+ KoLeo",                      78.9, 82.5),
    ("+ SwiGLU FFN",                 78.7, 83.1),
    ("+ Patch size 14",              78.9, 83.5),
    ("+ Teacher momentum 0.994",     79.4, 83.6),
    ("+ Tweak warmup schedules",     80.5, 83.8),
    ("+ Batch size 3k",              81.7, 84.7),
    ("+ Sinkhorn-Knopp",             81.7, 84.7),
    ("+ Untying heads (= DINOv2)",   82.0, 84.5),
]

steps = [r[0] for r in TABLE1]
knn = [r[1] for r in TABLE1]
lin = [r[2] for r in TABLE1]

print(f"{'단계':<30}{'k-NN':>7}{'linear':>8}")
for s, k, l in TABLE1:
    print(f"{s:<30}{k:>7.1f}{l:>8.1f}")
# 출력:
# 단계                            k-NN  linear
# iBOT (baseline)                 72.9    82.3
# + our reproduction              74.5    83.2
# + LayerScale, Stoch. Depth      75.4    82.0
# + 128k prototypes               76.6    81.9
# + KoLeo                         78.9    82.5
# + SwiGLU FFN                    78.7    83.1
# + Patch size 14                 78.9    83.5
# + Teacher momentum 0.994        79.4    83.6
# + Tweak warmup schedules        80.5    83.8
# + Batch size 3k                 81.7    84.7
# + Sinkhorn-Knopp                81.7    84.7
# + Untying heads (= DINOv2)      82.0    84.5

# %% [markdown]
# ## 2. 단계별 변화량 $\Delta$ 재계산
#
# $\Delta_i = \text{acc}_i - \text{acc}_{i-1}$. 논문 표에 적힌 ↑/↓ 값과 일치해야 한다.
# 시작→끝 총 개선은
# $$\Delta_{\text{total}} = \sum_i \Delta_i = \text{acc}_{\text{DINOv2}} - \text{acc}_{\text{iBOT}}$$

# %%
def deltas(xs):
    return [None] + [round(b - a, 1) for a, b in zip(xs[:-1], xs[1:])]


d_knn = deltas(knn)
d_lin = deltas(lin)


def fmt(d):
    if d is None:
        return "   -"
    if d == 0:
        return "   ="
    return f"{'↑' if d > 0 else '↓'}{abs(d):.1f}"


print(f"{'단계':<30}{'Δk-NN':>7}{'Δlinear':>9}")
for s, dk, dl in zip(steps, d_knn, d_lin):
    print(f"{s:<30}{fmt(dk):>7}{fmt(dl):>9}")
print()
print(f"총 개선  k-NN: {knn[0]} → {knn[-1]}  (+{knn[-1]-knn[0]:.1f})")
print(f"총 개선  linear: {lin[0]} → {lin[-1]}  (+{lin[-1]-lin[0]:.1f})")
# 출력:
# 단계                           Δk-NN  Δlinear
# iBOT (baseline)                    -        -
# + our reproduction              ↑1.6     ↑0.9
# + LayerScale, Stoch. Depth      ↑0.9     ↓1.2
# + 128k prototypes               ↑1.2     ↓0.1
# + KoLeo                         ↑2.3     ↑0.6
# + SwiGLU FFN                    ↓0.2     ↑0.6
# + Patch size 14                 ↑0.2     ↑0.4
# + Teacher momentum 0.994        ↑0.5     ↑0.1
# + Tweak warmup schedules        ↑1.1     ↑0.2
# + Batch size 3k                 ↑1.2     ↑0.9
# + Sinkhorn-Knopp                   =        =
# + Untying heads (= DINOv2)      ↑0.3     ↓0.2
#
# 총 개선  k-NN: 72.9 → 82.0  (+9.1)
# 총 개선  linear: 82.3 → 84.5  (+2.2)

# %% [markdown]
# ## 3. 관찰 포인트
#
# - 논문은 **k-NN 성능을 최적화 목표**로 삼았다 (linear는 k-NN에 의해 하한이 정해진다는 경험 때문).
#   그래서 k-NN 총 개선(+9.1)이 linear(+2.2)보다 훨씬 크다.
# - linear가 **떨어진** 단계: LayerScale+Stochastic Depth(↓1.2), 128k prototypes(↓0.1), Untying heads(↓0.2).
#   LayerScale/SD는 학습 안정성(NaN 방지)을 위해 감수한 비용.
# - k-NN 최대 기여: **KoLeo(+2.3)**, 그 다음 our reproduction(+1.6), 128k prototypes/Batch 3k(+1.2).
# - Sinkhorn-Knopp는 k-NN/linear 모두 변화 없음(=).

# %%
biggest_knn = max((d, s) for s, d in zip(steps[1:], d_knn[1:]))
drops_lin = [(s, d) for s, d in zip(steps[1:], d_lin[1:]) if d < 0]
print("k-NN 최대 기여 단계:", biggest_knn[1], f"(+{biggest_knn[0]})")
print("linear가 하락한 단계:", drops_lin)
print("k-NN이 linear보다 항상 낮은가?", all(k < l for k, l in zip(knn, lin)))
# 출력:
# k-NN 최대 기여 단계: + KoLeo (+2.3)
# linear가 하락한 단계: [('+ LayerScale, Stoch. Depth', -1.2), ('+ 128k prototypes', -0.1), ('+ Untying heads (= DINOv2)', -0.2)]
# k-NN이 linear보다 항상 낮은가? True

# %% [markdown]
# ## 4. 시각화: 누적 곡선(위) + 단계별 기여도(아래)
#
# 위: 요소를 누적할 때의 k-NN / linear 정확도. 시작점(iBOT)과 끝점(DINOv2)에 값을 표기.
# 아래: 각 단계의 $\Delta$ (증가=파랑/주황, 감소=빨강 계열).

# %%
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    row_heights=[0.6, 0.4],
    subplot_titles=(
        "누적 정확도 (ImageNet-1k Top-1, ViT-L / IN-22k)",
        "단계별 변화량 Δ (percentage points)",
    ),
)

C_KNN, C_LIN = "#1f77b4", "#ff7f0e"
x = list(range(len(steps)))

# 위: 누적 곡선
fig.add_trace(go.Scatter(
    x=x, y=knn, mode="lines+markers", name="k-NN",
    line=dict(color=C_KNN, width=3, shape="hv"), marker=dict(size=8),
    hovertemplate="%{customdata}<br>k-NN %{y:.1f}<extra></extra>", customdata=steps,
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=x, y=lin, mode="lines+markers", name="linear",
    line=dict(color=C_LIN, width=3, shape="hv"), marker=dict(size=8),
    hovertemplate="%{customdata}<br>linear %{y:.1f}<extra></extra>", customdata=steps,
), row=1, col=1)

# 시작/끝 값 라벨
for series, color, dy in ((knn, C_KNN, -14), (lin, C_LIN, 14)):
    for i in (0, len(x) - 1):
        fig.add_annotation(
            x=i, y=series[i], text=f"<b>{series[i]:.1f}</b>", showarrow=False,
            yshift=dy, font=dict(color=color, size=13), row=1, col=1,
        )

# 아래: Δ 막대 (첫 행 iBOT는 Δ 없음 → 0)
dk = [0.0] + d_knn[1:]
dl = [0.0] + d_lin[1:]
fig.add_trace(go.Bar(
    x=x, y=dk, name="Δ k-NN", offsetgroup=0,
    marker_color=[C_KNN if v >= 0 else "#d62728" for v in dk],
    text=[fmt(v) if i > 0 else "" for i, v in enumerate(dk)], textposition="outside",
    showlegend=False,
), row=2, col=1)
fig.add_trace(go.Bar(
    x=x, y=dl, name="Δ linear", offsetgroup=1,
    marker_color=[C_LIN if v >= 0 else "#e377c2" for v in dl],
    text=[fmt(v) if i > 0 else "" for i, v in enumerate(dl)], textposition="outside",
    showlegend=False,
), row=2, col=1)
fig.add_hline(y=0, line_color="gray", line_width=1, row=2, col=1)

fig.update_xaxes(
    tickmode="array", tickvals=x, ticktext=steps, tickangle=-35, row=2, col=1,
)
fig.update_yaxes(title_text="Top-1 (%)", range=[71, 86], row=1, col=1)
fig.update_yaxes(title_text="Δ (pp)", range=[-1.8, 2.9], row=2, col=1)
fig.update_layout(
    title="DINOv2 Table 1 — iBOT 베이스라인에서 DINOv2까지 요소를 누적한 ablation",
    barmode="group", template="plotly_white", width=1000, height=780,
    legend=dict(orientation="h", y=1.02, x=0.7),
    margin=dict(l=70, r=30, t=90, b=170),
)

_show(fig)

out = Path(__file__).resolve().parent / "expy.png" if "__file__" in globals() else Path("expy.png")
try:
    fig.write_image(str(out), scale=2)
    print("saved:", out)
except Exception as e:  # kaleido/Chrome 미설치 등
    print("expy.png 저장 실패:", type(e).__name__, e)
# 출력:
# saved: .../279f364f-c6b8-4d72-8dd4-e812f650b9d9/expy.png
