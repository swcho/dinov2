# %% [markdown]
# # DINOv2 학습 하이퍼파라미터 스케줄 재현
#
# 논문 Appendix B Table 16의 문장 —
# "625k iterations, AdamW, LayerScale 1e-5, weight decay cosine 0.04→0.2,
# LR warmup 100k, teacher momentum cosine 0.994→1, float16 (DINO head gradient만 float32 reduce)" —
# 를 공식 저장소 `dinov2/utils/utils.py::CosineScheduler`와 `configs/train/vitg14.yaml`
# 값으로 그대로 함수화해 iteration $t$ 에 따른 값을 확인한다.
#
# 공식 구현의 스케줄은 warmup 구간(선형)과 cosine 구간의 연결이다:
#
# $$
# s(t)=\begin{cases}
# s_{\text{start}} + (s_{\text{base}}-s_{\text{start}})\dfrac{t}{T_w} & 0\le t<T_w \\[6pt]
# s_{\text{final}} + \tfrac12\,(s_{\text{base}}-s_{\text{final}})\Bigl(1+\cos\dfrac{\pi\,(t-T_w)}{T-T_w}\Bigr) & T_w\le t<T
# \end{cases}
# $$
#
# - LR: $T_w=100\text{k}$, $s_{\text{base}}=\text{lr}$, $s_{\text{final}}=10^{-6}$ (`min_lr`)
# - weight decay: $T_w=0$, $0.04\to0.2$
# - teacher momentum: $T_w=0$, $0.994\to1.0$
# - teacher temperature: warmup만 있음 — $0.04\to0.07$ 을 30 epoch(=37.5k iter) 동안 선형, 이후 고정

# %%
# 필요 패키지: numpy, plotly, kaleido
#   기본 python3에는 numpy가 없으므로 아래 인터프리터로 실행:
#   /home/sungwoo/miniforge3/envs/trellis/bin/python expy.py
import math
import os

import numpy as np


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# %% [markdown]
# ## 1. 공식 `CosineScheduler`를 그대로 옮긴 함수
#
# 저장소 코드는 길이 `total_iters`의 배열을 미리 만들어 두고 `schedule[it]`로 조회한다.
# `it >= total_iters`이면 `final_value`를 돌려준다. 아래는 그 동작을 벡터화한 순수 함수 버전.

# %%
def cosine_schedule(t, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0.0):
    """dinov2/utils/utils.py::CosineScheduler 와 동일한 값을 iteration t(배열 가능)에 대해 반환."""
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t)

    # (1) 선형 warmup: np.linspace(start, base, warmup_iters)[t]
    in_warm = t < warmup_iters
    if warmup_iters > 0:
        out[in_warm] = start_warmup_value + (base_value - start_warmup_value) * t[in_warm] / (warmup_iters - 1)

    # (2) cosine: final + 0.5*(base-final)*(1+cos(pi*i/n)),  i = t - warmup, n = total - warmup
    n = total_iters - warmup_iters
    i = t - warmup_iters
    in_cos = (~in_warm) & (t < total_iters)
    out[in_cos] = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * i[in_cos] / n))

    # (3) 학습 종료 이후
    out[t >= total_iters] = final_value
    return out


# %% [markdown]
# ## 2. 설정값 (configs/train/vitg14.yaml + ssl_default_config.yaml)
#
# | 항목 | config 키 | 값 |
# |---|---|---|
# | 총 iteration | `epochs(500) × OFFICIAL_EPOCH_LENGTH(1250)` | 625,000 |
# | LR warmup | `warmup_epochs(80) × 1250` | 100,000 |
# | base LR (batch 1024 기준) | `optim.base_lr` | 2e-4 |
# | LR 스케일링 규칙 | `scaling_rule: sqrt_wrt_1024` | $\text{lr}=\text{base\_lr}\cdot\sqrt{B/1024}$ |
# | 최종 LR | `min_lr` | 1e-6 |
# | weight decay | `weight_decay → weight_decay_end` | 0.04 → 0.2 |
# | teacher momentum | `momentum_teacher → final_momentum_teacher` | 0.994 → 1.0 |
# | teacher temp | `warmup_teacher_temp → teacher_temp`, 30 epoch | 0.04 → 0.07 |
# | LayerScale 초기값 | `student.layerscale` | 1e-5 |

# %%
EPOCH_LEN = 1250
EPOCHS = 500
T = EPOCHS * EPOCH_LEN  # 625_000
WARMUP = 80 * EPOCH_LEN  # 100_000

BATCH = 3072  # ViT-g/14 from scratch (Table 16)
BASE_LR = 2.0e-4  # batch 1024 기준
LR = BASE_LR * math.sqrt(BATCH / 1024.0)  # sqrt_wrt_1024 스케일링 규칙
MIN_LR = 1.0e-6

WD0, WD1 = 0.04, 0.2
M0, M1 = 0.994, 1.0
TEMP0, TEMP1 = 0.04, 0.07
TEMP_WARMUP = 30 * EPOCH_LEN  # 37_500

print(f"T = {T:,} iters, LR warmup = {WARMUP:,} iters ({WARMUP / T:.0%} of T)")
print(f"peak LR = {BASE_LR:g} * sqrt({BATCH}/1024) = {LR:.3e}  (논문 Table 16: 3.5e-4)")
# 출력:
# T = 625,000 iters, LR warmup = 100,000 iters (16% of T)
# peak LR = 0.0002 * sqrt(3072/1024) = 3.464e-04  (논문 Table 16: 3.5e-4)


def lr_at(t):
    return cosine_schedule(t, LR, MIN_LR, T, warmup_iters=WARMUP, start_warmup_value=0.0)


def wd_at(t):
    return cosine_schedule(t, WD0, WD1, T)


def momentum_at(t):
    return cosine_schedule(t, M0, M1, T)


def temp_at(t):
    # base=final=0.07, total=warmup=37.5k → warmup 후에는 final_value(0.07) 고정
    return cosine_schedule(t, TEMP1, TEMP1, TEMP_WARMUP, warmup_iters=TEMP_WARMUP, start_warmup_value=TEMP0)


# %% [markdown]
# ## 3. 대표 iteration에서의 값
#
# $t=0$(시작), $50\text{k}$(warmup 중간), $100\text{k}$(warmup 종료 = LR 피크),
# $312.5\text{k}$(전체의 절반 — cosine의 중점이 아님에 주의: cosine은 100k에서 시작하므로
# 중점은 362.5k), $625\text{k}$(종료).

# %%
checkpoints = [0, 50_000, 100_000, 312_500, 625_000]
print(f"{'iter':>9} | {'LR':>10} | {'weight decay':>12} | {'momentum':>9} | {'1/(1-m)':>8} | {'teacher T':>9}")
print("-" * 72)
for t in checkpoints:
    lr, wd, m, tp = (float(f(t)) for f in (lr_at, wd_at, momentum_at, temp_at))
    window = "inf" if m >= 1.0 else f"{1 / (1 - m):8.0f}"
    print(f"{t:>9,} | {lr:10.3e} | {wd:12.4f} | {m:9.6f} | {window:>8} | {tp:9.4f}")
# 출력:
#      iter |         LR | weight decay |  momentum |  1/(1-m) | teacher T
# ------------------------------------------------------------------------
#         0 |  0.000e+00 |       0.0400 |  0.994000 |      167 |    0.0400
#    50,000 |  1.732e-04 |       0.0425 |  0.994094 |      169 |    0.0700
#   100,000 |  3.464e-04 |       0.0499 |  0.994371 |      178 |    0.0700
#   312,500 |  2.246e-04 |       0.1200 |  0.997000 |      333 |    0.0700
#   625,000 |  1.000e-06 |       0.2000 |  1.000000 |      inf |    0.0700
# 관찰: 312.5k(전체의 절반)에서 WD와 momentum은 정확히 중간값(0.12, 0.997)이지만
#       LR은 cosine이 100k에서 시작하므로 아직 피크의 65% 수준이다.

# %% [markdown]
# ## 4. teacher momentum 0.994 의 의미 — EMA 유효 평균 창
#
# teacher 파라미터는 $\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s$ 로 갱신된다.
# 이는 student 파라미터 이력에 가중치 $(1-m)m^k$ 를 두는 지수 평균이며,
# 유효 평균 창(가중치 합이 $1-1/e$ 가 되는 길이)은 대략 $1/(1-m)$ 스텝이다.
#
# - $m=0.994 \Rightarrow 1/(1-m)\approx 167$ 스텝 (DINOv2 시작값)
# - $m=0.996 \Rightarrow 250$ 스텝 (DINO 기본값)
# - $m=0.999 \Rightarrow 1000$ 스텝
#
# 즉 0.994는 teacher가 student를 **더 빨리** 따라가게 하는 쪽이고,
# cosine으로 1.0에 가까워지면 teacher는 거의 고정된 안정 타깃이 된다.

# %%
for m in (0.994, 0.996, 0.999):
    window = 1 / (1 - m)
    # k 스텝 뒤 남아 있는 옛 가중치 비율 m^k: window 스텝 후 ≈ e^-1
    print(f"m={m}: 1/(1-m) = {window:7.1f} steps,  m^window = {m ** window:.3f} (≈ 1/e = {math.exp(-1):.3f})")

# 학습 후반, 예컨대 t=600k 에서의 창 크기
m_late = float(momentum_at(600_000))
print(f"t=600k: m = {m_late:.6f} → 1/(1-m) ≈ {1 / (1 - m_late):,.0f} steps")
# 출력:
# m=0.994: 1/(1-m) =   166.7 steps,  m^window = 0.367 (≈ 1/e = 0.368)
# m=0.996: 1/(1-m) =   250.0 steps,  m^window = 0.367 (≈ 1/e = 0.368)
# m=0.999: 1/(1-m) =  1000.0 steps,  m^window = 0.368 (≈ 1/e = 0.368)
# t=600k: m = 0.999976 → 1/(1-m) ≈ 42,273 steps

# %% [markdown]
# ## 5. 스케줄 시각화
#
# 4개 서브플롯(공유 x축): LR(log), weight decay, teacher momentum, teacher temperature.
# 점선은 LR warmup 종료(100k).

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

t = np.arange(0, T + 1, 500)
fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=(
        "Learning rate (log scale) — warmup 100k → cosine to 1e-6",
        "Weight decay — cosine 0.04 → 0.2",
        "Teacher EMA momentum — cosine 0.994 → 1.0",
        "Teacher temperature — linear warmup 0.04 → 0.07 (30 epochs = 37.5k)",
    ),
)
fig.add_trace(go.Scatter(x=t, y=lr_at(t), name="LR", line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=wd_at(t), name="weight decay", line=dict(color="#ff7f0e")), row=2, col=1)
fig.add_trace(go.Scatter(x=t, y=momentum_at(t), name="momentum", line=dict(color="#2ca02c")), row=3, col=1)
fig.add_trace(go.Scatter(x=t, y=temp_at(t), name="teacher temp", line=dict(color="#d62728")), row=4, col=1)

for r in range(1, 5):
    fig.add_vline(x=WARMUP, line=dict(color="gray", dash="dot"), row=r, col=1)
fig.add_annotation(
    x=WARMUP, y=math.log10(LR), xref="x", yref="y", text=f"warmup end (100k)<br>peak LR={LR:.2e}",
    showarrow=True, arrowhead=2, ax=80, ay=30,
)
fig.update_yaxes(type="log", row=1, col=1, title_text="LR")
fig.update_yaxes(row=2, col=1, title_text="wd")
fig.update_yaxes(row=3, col=1, title_text="m", range=[0.9935, 1.0005])
fig.update_yaxes(row=4, col=1, title_text="T")
fig.update_xaxes(row=4, col=1, title_text="iteration (total 625k)")
fig.update_layout(
    height=1000, width=900, showlegend=False,
    title_text="DINOv2 (ViT-g/14, batch 3072) training schedules — Table 16 + configs/train/vitg14.yaml",
)
_show(fig)

out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "expy.png")
fig.write_image(out_png, scale=1)
print("saved:", out_png)
# 출력:
# saved: .../.fm/hints/37ab055b-65fb-49ff-ae5e-4513192b51fd/expy.png
