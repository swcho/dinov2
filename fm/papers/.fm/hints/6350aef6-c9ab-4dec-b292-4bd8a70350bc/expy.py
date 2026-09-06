# %% [markdown]
# # Sinkhorn-Knopp centering 실험
#
# DINOv2 teacher 의 target 분포 $p_t$ 를 만드는 세 방법을 비교한다.
#
# 1. 단순 softmax: $p_t = \mathrm{softmax}(z/\tau)$
# 2. DINO centering: $p_t = \mathrm{softmax}((z - c)/\tau)$, $c$ 는 배치 평균의 EMA
# 3. Sinkhorn-Knopp (SwAV/DINOv2): $Q = \exp(z/\tau)^\top$ 를 행·열 교대 정규화 3회
#
# 특히 SK 가 배치 내 prototype 사용량을 정확히 $B/K$ 로 맞추는 것과,
# 3회 반복이 충분한 이유(기하급수적 수렴)를 확인한다.

# %%
# 필요 패키지: numpy, plotly, kaleido (png 저장)
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


HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. 붕괴 조짐이 있는 teacher 로짓 만들기
#
# $B = 256$ 장, $K = 8$ prototype, teacher 온도 $\tau_t = 0.07$ (DINOv2 최종값).
# 0번 prototype 에 bias(+1.0)를 주어 "거의 모든 이미지가 0번 서랍" 이라고 답하는 상황을 흉내낸다.
# (실제 DINO head 의 마지막 층은 weight-norm 된 prototype 과 정규화된 feature 의 내적이라 로짓이
# $[-1, 1]$ 범위에 있으므로, 이 정도 스케일이 현실적이다.)

# %%
B, K, TEMP = 256, 8, 0.07
NOISE, BIAS = 0.1, 1.0


def make_logits():
    z = rng.normal(0.0, NOISE, size=(B, K))
    z[:, 0] += BIAS  # 한 prototype 이 독점하는 편향
    return z


logits = make_logits()


def softmax(z, temp):
    z = z / temp
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def usage(p):
    """배치 내 prototype 사용 비율 = 각 열(prototype)의 평균 확률. 합은 1."""
    return p.mean(axis=0)


p_soft = softmax(logits, TEMP)
print("단순 softmax 사용 비율:", np.round(usage(p_soft), 4))
print("각 샘플 확률 합 (min,max):", p_soft.sum(1).min().round(6), p_soft.sum(1).max().round(6))
# 출력: 단순 softmax 사용 비율: [1. 0. 0. 0. 0. 0. 0. 0.]
# 출력: 각 샘플 확률 합 (min,max): 1.0 1.0
# → 온도 0.07 에서 0번 prototype 이 사실상 100% 를 차지. 완전 붕괴.

# %% [markdown]
# ## 2. DINO 의 EMA centering
#
# $$c \leftarrow m\,c + (1-m)\,\tfrac{1}{B}\sum_i z_i,\qquad p_t = \mathrm{softmax}\big((z-c)/\tau\big)$$
#
# `DINOLoss.softmax_center_teacher` / `update_center` (center_momentum=0.9) 에 해당한다.
# 실제 학습처럼 center 가 여러 스텝에 걸쳐 갱신된다고 가정하고 20 스텝 돌린다.

# %%
center = np.zeros((1, K))
m = 0.9
for step in range(20):
    center = m * center + (1 - m) * make_logits().mean(axis=0, keepdims=True)
p_center = softmax(logits - center, TEMP)
print("center 값:", np.round(center[0], 3))
print("EMA centering 사용 비율:", np.round(usage(p_center), 3))
# 출력: center 값: [ 0.879  0.     0.    -0.     0.001 -0.002  0.001  0.   ]
# 출력: EMA centering 사용 비율: [0.347 0.082 0.087 0.107 0.101 0.086 0.089 0.1  ]
# → 독점은 사라지지만 균등(0.125)에서 벗어나 있고(0번 0.347), 과거 통계(EMA)에 의존한다.

# %% [markdown]
# ## 3. Sinkhorn-Knopp centering (DINOv2 `sinkhorn_knopp_teacher` 를 numpy 로 옮김)
#
# $Q \in \mathbb{R}^{K\times B}$, 목표: 행 합 $=1/K$ (prototype 균등), 열 합 $=1/B$ (샘플마다 분포).
#
# $$Q^\ast = \mathrm{Diag}(u)\,\exp(Z/\tau)\,\mathrm{Diag}(v)$$
#
# 원 코드와 같이 로짓을 그대로 $\exp(z/\tau)$ 에 넣는다(shift 없음). 로짓이 유계이므로 오버플로 걱정이 없다.

# %%
def sinkhorn_knopp_teacher(teacher_output, teacher_temp, n_iterations=3, record=None):
    Q = np.exp(teacher_output / teacher_temp).T  # K x B (논문 표기와 동일)
    B_ = Q.shape[1]
    K_ = Q.shape[0]
    Q = Q / Q.sum()  # 전체 합 1  (원 코드: all_reduce 로 GPU 전체 합)
    if record is not None:
        record.append(Q.copy())
    for _ in range(n_iterations):
        Q /= Q.sum(axis=1, keepdims=True)  # 행: prototype 마다 합 1
        Q /= K_                             # → 1/K
        Q /= Q.sum(axis=0, keepdims=True)  # 열: 샘플 마다 합 1
        Q /= B_                             # → 1/B
        if record is not None:
            record.append(Q.copy())
    Q *= B_  # 열 합 = 1 → 각 샘플의 확률분포
    return Q.T  # B x K


p_sk = sinkhorn_knopp_teacher(logits, TEMP, n_iterations=3)
print("SK(3회) 사용 비율:", np.round(usage(p_sk), 4))
print("각 샘플 확률 합 (min,max):", p_sk.sum(1).min().round(6), p_sk.sum(1).max().round(6))
print("이상적 균등값 1/K =", 1 / K)
# 출력: SK(3회) 사용 비율: [0.1253 0.1254 0.1249 0.1245 0.1251 0.1255 0.1247 0.1247]
# 출력: 각 샘플 확률 합 (min,max): 1.0 1.0
# 출력: 이상적 균등값 1/K = 0.125
# → 이 배치 안에서 prototype 사용량이 (소수 셋째 자리까지) 균등, 각 샘플은 여전히 확률분포.

# %% [markdown]
# ## 4. 왜 3회면 충분한가 — 수렴 속도
#
# 각 반복 후(열 정규화 직후) 행 합이 $1/K$ 에서 얼마나 벗어나는지 최대 상대 오차
# $\max_k |K\cdot\text{rowsum}_k - 1|$ 를 기록한다. 마지막 단계가 열 정규화이므로
# 열 조건(샘플마다 합 1)은 항상 정확히 만족되고, 행 조건만 근사다.

# %%
hist = []
_ = sinkhorn_knopp_teacher(logits, TEMP, n_iterations=10, record=hist)
row_err = [np.abs(K * Q.sum(axis=1) - 1).max() for Q in hist]
for it, e in enumerate(row_err):
    print(f"반복 {it:2d}: 행 합 최대 상대 오차 = {e:.3e}")
ratios = np.array(row_err[2:]) / np.array(row_err[1:-1])
print("반복당 오차 감소 비율(2회 이후 평균): %.3f" % ratios.mean())
# 출력: 반복  0: 행 합 최대 상대 오차 = 7.000e+00
# 출력: 반복  1: 행 합 최대 상대 오차 = 8.903e-02
# 출력: 반복  2: 행 합 최대 상대 오차 = 1.795e-02
# 출력: 반복  3: 행 합 최대 상대 오차 = 3.704e-03
# 출력: 반복  4: 행 합 최대 상대 오차 = 8.852e-04
# 출력: 반복  5: 행 합 최대 상대 오차 = 2.159e-04
# 출력: 반복  6: 행 합 최대 상대 오차 = 5.356e-05
# 출력: 반복  7: 행 합 최대 상대 오차 = 1.348e-05
# 출력: 반복  8: 행 합 최대 상대 오차 = 3.431e-06
# 출력: 반복  9: 행 합 최대 상대 오차 = 8.814e-07
# 출력: 반복 10: 행 합 최대 상대 오차 = 2.280e-07
# 출력: 반복당 오차 감소 비율(2회 이후 평균): 0.240
# → 반복마다 약 1/4 로 줄어드는 등비 감소(선형 수렴). 3회면 ~4e-3 수준.

# %% [markdown]
# ## 5. SK 가 배정을 어떻게 바꾸는가
#
# 원래 점수로는 모두 0번 prototype 이 1등이지만, "다른 prototype 도 공평하게 써야 한다" 는
# 제약 때문에 상대적으로 다른 prototype 점수가 높았던 샘플들이 그쪽으로 배정된다.

# %%
argmax_soft = p_soft.argmax(1)
argmax_sk = p_sk.argmax(1)
print("softmax argmax 분포:", np.bincount(argmax_soft, minlength=K))
print("SK      argmax 분포:", np.bincount(argmax_sk, minlength=K))
# 각 샘플에서 0번 대비 다른 prototype 의 상대 점수가 큰 샘플이 다른 쪽으로 갔는지 확인
rel = logits[:, 1:] - logits[:, [0]]
print("SK 가 0번 이외로 보낸 샘플의 상대점수 평균:", rel.max(1)[argmax_sk != 0].mean().round(3))
print("SK 가 0번으로 남긴 샘플의 상대점수 평균:  ", rel.max(1)[argmax_sk == 0].mean().round(3))
# 출력: softmax argmax 분포: [256   0   0   0   0   0   0   0]
# 출력: SK      argmax 분포: [33 32 22 33 40 37 29 30]
# 출력: SK 가 0번 이외로 보낸 샘플의 상대점수 평균: -0.846
# 출력: SK 가 0번으로 남긴 샘플의 상대점수 평균:   -1.056
# → 0번 대비 손해가 적은(상대점수가 높은) 샘플들이 다른 prototype 으로 배정된다.

# %% [markdown]
# ## 6. 시각화

# %%
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        f"배치 내 prototype 사용 비율 (B={B}, K={K}, τ={TEMP})",
        "SK 반복 횟수에 따른 행 합 상대 오차",
    ),
)
x = [f"p{k}" for k in range(K)]
fig.add_trace(go.Bar(name="softmax", x=x, y=usage(p_soft), marker_color="#c0392b"), 1, 1)
fig.add_trace(go.Bar(name="EMA centering", x=x, y=usage(p_center), marker_color="#f39c12"), 1, 1)
fig.add_trace(go.Bar(name="Sinkhorn-Knopp ×3", x=x, y=usage(p_sk), marker_color="#2980b9"), 1, 1)
fig.add_hline(y=1 / K, line_dash="dash", line_color="gray", row=1, col=1,
              annotation_text="1/K", annotation_position="top right")

fig.add_trace(go.Scatter(x=list(range(len(row_err))), y=row_err, mode="lines+markers",
                         name="max |K·rowsum − 1|", marker_color="#2980b9"), 1, 2)
fig.add_vline(x=3, line_dash="dash", line_color="#27ae60", row=1, col=2,
              annotation_text="DINOv2: 3회", annotation_position="top right")
fig.update_yaxes(type="log", row=1, col=2, title_text="상대 오차 (log)")
fig.update_xaxes(title_text="반복 횟수", row=1, col=2)
fig.update_yaxes(title_text="사용 비율", row=1, col=1)
fig.update_layout(barmode="group", width=1100, height=450,
                  title_text="Sinkhorn-Knopp centering: 균등 분배 강제와 3회 수렴",
                  legend=dict(orientation="h", y=-0.2))
_show(fig)
out_png = os.path.join(HERE, "expy.png")
fig.write_image(out_png, scale=2)
print("saved:", out_png)
# 출력: saved: .../expy.png
