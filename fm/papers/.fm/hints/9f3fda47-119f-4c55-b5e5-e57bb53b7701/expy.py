# %% [markdown]
# # KoLeo regularizer 실험
#
# DINOv2가 쓰는 **KoLeo(Kozachenko-Leonenko) regularizer**를 numpy만으로 구현하고,
# 이 항이 실제로 배치 안의 특징 벡터를 "고르게 퍼뜨리는지" 단계적으로 확인한다.
#
# $$\mathcal{L}_{koleo} = -\frac{1}{n}\sum_{i=1}^{n}\log(d_{n,i}),\qquad
# d_{n,i} = \min_{j\neq i}\|x_i - x_j\|$$
#
# 실험 순서
# 1. KoLeo loss 구현 (논문 정의 그대로)
# 2. 뭉친 점 vs 퍼진 점의 loss 비교
# 3. DINOv2 코드의 "내적 최대 = 거리 최소" 트릭 검증
# 4. KoLeo만으로 경사하강 → 점들이 원 위에 등간격으로 퍼지는지 관찰
# 5. Kozachenko-Leonenko 미분 엔트로피 추정량과의 관계 확인
# 6. 시각화 → `expy.png`

# %%
# 필요 패키지: numpy, scipy, plotly, kaleido (모두 trellis env에 설치됨)
import os

import numpy as np
from scipy.special import digamma, gammaln


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
# ## 1. KoLeo loss 구현
#
# 논문 정의를 그대로 옮긴다. DINOv2는 loss를 계산하기 전에 특징을 $\ell_2$ 정규화하므로
# `normalize=True`가 기본값이다.

# %%
def l2_normalize(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def nn_distances(x):
    """각 점 x_i 에 대해 d_{n,i} = min_{j != i} ||x_i - x_j||  (n,) 반환."""
    diff = x[:, None, :] - x[None, :, :]          # (n, n, D)
    dist = np.linalg.norm(diff, axis=-1)          # (n, n)
    np.fill_diagonal(dist, np.inf)                # j == i 제외
    return dist.min(axis=1)


def koleo_loss(x, normalize=True, eps=1e-8):
    if normalize:
        x = l2_normalize(x)
    d = nn_distances(x)
    return -np.mean(np.log(d + eps))


x_demo = rng.normal(size=(5, 3))
print("d_{n,i} =", np.round(nn_distances(l2_normalize(x_demo)), 4))
print("KoLeo   =", round(koleo_loss(x_demo), 4))
# 출력: d_{n,i} = [0.7434 0.7434 1.6442 0.6158 0.6158]
# 출력: KoLeo   = 0.2131

# %% [markdown]
# ## 2. 뭉친 점 vs 퍼진 점
#
# 단위원 위의 점 $n=8$개를 두 가지로 놓는다.
# - **뭉침**: 각도가 $[0, 0.3]$ rad 안에 몰려 있음 (feature collapse 상황)
# - **등간격**: $2\pi/8$ 간격으로 고르게 배치
#
# 최근접 거리 $d_{n,i}$가 작을수록 $-\log d$ 는 커지므로, 뭉친 쪽 loss가 훨씬 커야 한다.

# %%
def on_circle(theta):
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


n = 8
x_clumped = on_circle(np.linspace(0, 0.3, n))
x_uniform = on_circle(np.linspace(0, 2 * np.pi, n, endpoint=False))

print("뭉친 점   KoLeo =", round(koleo_loss(x_clumped), 4),
      " (평균 d =", round(nn_distances(x_clumped).mean(), 4), ")")
print("등간격 점 KoLeo =", round(koleo_loss(x_uniform), 4),
      " (평균 d =", round(nn_distances(x_uniform).mean(), 4), ")")
# 출력: 뭉친 점   KoLeo = 3.15  (평균 d = 0.0429 )
# 출력: 등간격 점 KoLeo = 0.2674  (평균 d = 0.7654 )

# %% [markdown]
# 등간격일 때 $d = 2\sin(\pi/8) \approx 0.7654$ 이고 $-\log 0.7654 \approx 0.2674$ 로 손계산과 일치한다.
# 뭉친 경우 loss가 10배 이상 크다.

# %% [markdown]
# ## 3. DINOv2 코드의 트릭: "내적 최대 = 거리 최소"
#
# `dinov2/loss/koleo_loss.py` 는 거리 행렬을 직접 만들지 않고 `x @ x.T` 의 **행별 최댓값 인덱스**로
# 최근접 이웃을 찾는다. $\ell_2$ 정규화된 벡터에서는
# $$\|x_i - x_j\|^2 = 2 - 2\,x_i^{\top}x_j$$
# 이므로 내적이 최대인 $j$ 가 곧 거리가 최소인 $j$ 다. 대각선은 $-1$로 채워 자기 자신을 제외한다.

# %%
def pairwise_nn_inner(x):
    """DINOv2 KoLeoLoss.pairwise_NNs_inner 의 numpy 버전."""
    dots = x @ x.T
    np.fill_diagonal(dots, -1.0)
    return dots.argmax(axis=1)


x = l2_normalize(rng.normal(size=(64, 16)))
I_inner = pairwise_nn_inner(x)

dist = np.linalg.norm(x[:, None] - x[None], axis=-1)
np.fill_diagonal(dist, np.inf)
I_dist = dist.argmin(axis=1)

print("두 방법의 최근접 이웃 인덱스 일치:", np.array_equal(I_inner, I_dist))
print("코드식 loss:", round(-np.log(np.linalg.norm(x - x[I_inner], axis=1) + 1e-8).mean(), 6))
print("정의식 loss:", round(koleo_loss(x), 6))
# 출력: 두 방법의 최근접 이웃 인덱스 일치: True
# 출력: 코드식 loss: 0.077202
# 출력: 정의식 loss: 0.077202

# %% [markdown]
# ## 4. KoLeo만으로 경사하강 — 점들이 스스로 퍼지는가?
#
# $-\log d_{n,i}$ 를 $x_i$ 로 미분하면
# $$\frac{\partial}{\partial x_i}\bigl(-\log\|x_i - x_j\|\bigr) = -\frac{x_i - x_j}{\|x_i - x_j\|^2}$$
# 즉 **가장 가까운 이웃에서 멀어지는 방향**으로, 거리가 가까울수록 더 세게 밀린다 (쿨롱 반발과 같은 꼴).
# 뭉친 초기 상태에서 KoLeo만 최소화하면서 매 스텝 단위원으로 다시 투영해 본다
# (DINOv2가 $\ell_2$ 정규화된 특징에 KoLeo를 걸기 때문에, 단위 구 위에서의 움직임만 본다).

# %%
def koleo_grad(x):
    """L2 정규화된 x 에 대한 d(KoLeo)/dx (최근접 이웃 인덱스는 상수로 취급)."""
    n = len(x)
    I = pairwise_nn_inner(x)
    diff = x - x[I]                                   # (n, D)
    d2 = (diff ** 2).sum(axis=1, keepdims=True) + 1e-8
    grad = -diff / d2 / n                             # x_i 에 대한 항
    np.add.at(grad, I, diff / d2 / n)                 # x_j = x_I 에 대한 대칭 항
    return grad


def angular_gaps(x):
    th = np.sort(np.arctan2(x[:, 1], x[:, 0]))
    return np.diff(np.concatenate([th, [th[0] + 2 * np.pi]]))


n = 12
theta0 = rng.uniform(0, 0.8, size=n)                 # 0.8 rad 안에 뭉친 12개 점
x0 = on_circle(theta0)
x = x0.copy()
steps = 1000
history = [koleo_loss(x)]
for step in range(steps):
    lr = 0.05 * (1 - step / steps) + 0.001              # 선형 감소 학습률 (NN 인덱스가 바뀌며 생기는 진동 억제)
    x = x - lr * koleo_grad(x)
    x = l2_normalize(x)
    history.append(koleo_loss(x))
x_final = x

print(f"초기 loss = {history[0]:.4f}, 최종 loss = {history[-1]:.4f}")
print(f"이론 최솟값(등간격, d=2sin(pi/{n})) = {-np.log(2*np.sin(np.pi/n)):.4f}")
print("최종 각도 간격(rad):", np.round(angular_gaps(x_final), 3))
print(f"등간격 2pi/{n} = {2*np.pi/n:.3f}")
# 출력: 초기 loss = 4.6116, 최종 loss = 0.6593
# 출력: 이론 최솟값(등간격, d=2sin(pi/12)) = 0.6585
# 출력: 최종 각도 간격(rad): [0.526 0.526 0.525 0.524 0.521 0.521 0.521 0.521 0.522 0.524 0.525 0.526]
# 출력: 등간격 2pi/12 = 0.524

# %% [markdown]
# KoLeo 항 하나만으로도 점들이 원 위에 거의 완벽한 **등간격**으로 퍼졌고 loss는 이론 최솟값에 도달했다.
# 이것이 논문이 말하는 "uniform span of the features within a batch"다.

# %% [markdown]
# ## 5. Kozachenko-Leonenko 엔트로피 추정량과의 관계
#
# $d$차원 표본 $x_1,\dots,x_n$ 의 미분 엔트로피 $H(p) = -\int p\log p$ 는 최근접 거리 $\varepsilon_i$ 만으로
# 추정할 수 있다 (Kozachenko & Leonenko, 1987):
# $$\hat H_{KL} = \frac{d}{n}\sum_{i=1}^{n}\log \varepsilon_i + \log V_d + \psi(n) - \psi(1),
# \qquad V_d = \frac{\pi^{d/2}}{\Gamma(d/2+1)}$$
# 표본에 의존하는 부분은 $\frac{d}{n}\sum\log\varepsilon_i = -d\cdot\mathcal{L}_{koleo}$ **뿐**이다. 따라서
# $$\mathcal{L}_{koleo} = -\frac{1}{d}\hat H_{KL} + \text{const}$$
# 즉 KoLeo를 최소화하는 것은 배치 특징 분포의 **엔트로피를 최대화**하는 것이고,
# 유계 영역(단위 구)에서 엔트로피가 최대인 분포는 **균등분포**다.

# %%
def kl_entropy(x):
    n, d = x.shape
    eps_i = nn_distances(x)
    log_Vd = (d / 2) * np.log(np.pi) - gammaln(d / 2 + 1)
    return d * np.mean(np.log(eps_i)) + log_Vd + digamma(n) - digamma(1)


n = 4000
# 1D 균등분포 U(0,1): 진짜 H = 0
u = rng.uniform(0, 1, size=(n, 1))
# 1D 정규분포 N(0,1): 진짜 H = 0.5 log(2 pi e) ~ 1.4189
g = rng.normal(0, 1, size=(n, 1))
# 2D 균등분포 U(0,1)^2: 진짜 H = 0
u2 = rng.uniform(0, 1, size=(n, 2))

for name, s, true_H in [("U(0,1)   ", u, 0.0),
                        ("N(0,1)   ", g, 0.5 * np.log(2 * np.pi * np.e)),
                        ("U(0,1)^2 ", u2, 0.0)]:
    print(f"{name} KL 추정 H = {kl_entropy(s):7.4f}  (진짜 H = {true_H:.4f}),  KoLeo = {koleo_loss(s, normalize=False):.4f}")
# 출력: U(0,1)    KL 추정 H =  0.0280  (진짜 H = 0.0000),  KoLeo = 9.5359
# 출력: N(0,1)    KL 추정 H =  1.4365  (진짜 H = 1.4189),  KoLeo = 8.1272
# 출력: U(0,1)^2  KL 추정 H = -0.0260  (진짜 H = 0.0000),  KoLeo = 5.0209

# %% [markdown]
# 추정량이 진짜 엔트로피를 잘 맞추고, 같은 $d$ 안에서는 엔트로피가 큰 분포(N(0,1))가
# KoLeo 값이 작다 — 부호가 반대인 선형 관계임을 확인했다.
#
# (주의: KoLeo의 절대값은 $n$, $d$에 따라 달라지므로 서로 다른 설정 사이의 비교는 의미가 없다.
# 학습에서는 같은 배치 크기·차원 안에서 상대적으로만 쓰인다.)

# %% [markdown]
# ## 6. 시각화
#
# 왼쪽: 뭉친 초기 점(회색)과 KoLeo 경사하강 후(파랑). 가운데: 학습 곡선과 이론 최솟값.
# 오른쪽: 점별 최근접 거리 $d_{n,i}$ before/after (로그 축).

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("단위원 위 12개 점: before → after",
                    "KoLeo loss (경사하강)",
                    "점별 최근접 거리 d_{n,i}"),
    column_widths=[0.36, 0.34, 0.30],
)

circ = np.linspace(0, 2 * np.pi, 200)
fig.add_trace(go.Scatter(x=np.cos(circ), y=np.sin(circ), mode="lines",
                         line=dict(color="lightgray", width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=x0[:, 0], y=x0[:, 1], mode="markers", name="before (뭉침)",
                         marker=dict(color="gray", size=10, symbol="circle-open")), row=1, col=1)
fig.add_trace(go.Scatter(x=x_final[:, 0], y=x_final[:, 1], mode="markers", name="after (KoLeo)",
                         marker=dict(color="royalblue", size=10)), row=1, col=1)
for a, b in zip(x0, x_final):
    fig.add_trace(go.Scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode="lines",
                             line=dict(color="rgba(65,105,225,0.25)", width=1), showlegend=False), row=1, col=1)

fig.add_trace(go.Scatter(y=history, mode="lines", name="KoLeo loss", line=dict(color="royalblue")), row=1, col=2)
fig.add_hline(y=-np.log(2 * np.sin(np.pi / 12)),
              line=dict(color="crimson", dash="dash"), row=1, col=2,
              annotation_text="이론 최솟값 -log(2 sin(pi/12))", annotation_position="top right")

idx = np.arange(1, 13)
fig.add_trace(go.Scatter(x=idx, y=nn_distances(x0), mode="markers", name="d before",
                         marker=dict(color="gray", size=9, symbol="circle-open")), row=1, col=3)
fig.add_trace(go.Scatter(x=idx, y=nn_distances(x_final), mode="markers", name="d after",
                         marker=dict(color="royalblue", size=9)), row=1, col=3)
fig.add_hline(y=2 * np.sin(np.pi / 12), line=dict(color="crimson", dash="dash"), row=1, col=3,
              annotation_text="등간격 d = 2 sin(pi/12)", annotation_position="bottom right")

fig.update_xaxes(scaleanchor="y", scaleratio=1, range=[-1.2, 1.2], row=1, col=1)
fig.update_yaxes(range=[-1.2, 1.2], row=1, col=1)
fig.update_xaxes(title_text="step", row=1, col=2)
fig.update_yaxes(title_text="loss", row=1, col=2)
fig.update_xaxes(title_text="점 번호 i", row=1, col=3)
fig.update_yaxes(title_text="d_{n,i} (log)", type="log", row=1, col=3)
fig.update_layout(title="KoLeo regularizer: 최근접 이웃을 밀어내어 배치 특징을 고르게 퍼뜨린다",
                  width=1400, height=480, template="plotly_white",
                  legend=dict(orientation="h", y=-0.15))

_show(fig)
png_path = os.path.join(HERE, "expy.png")
fig.write_image(png_path, scale=2)
print("saved:", png_path)
# 출력: saved: /home/sungwoo/projects/swcho/dinov2/fm/papers/.fm/hints/9f3fda47-119f-4c55-b5e5-e57bb53b7701/expy.png
