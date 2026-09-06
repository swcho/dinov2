# %% [markdown]
# # DINO image-level objective 를 numpy 로 따라가기
#
# 목표: 논문 §4 의 image-level objective
#
# $$\mathcal{L}_{DINO} = -\sum_k p_t^{(k)} \log p_s^{(k)},\qquad
# p_s = \mathrm{softmax}(s/\tau_s),\quad p_t = \mathrm{softmax}((t-c)/\tau_t)$$
#
# 를 (1) softmax/온도, (2) cross-entropy, (3) centering·sharpening, (4) Sinkhorn-Knopp,
# (5) 작은 학습 시뮬레이션(collapse 재현) 순서로 확인한다. torch 없이 numpy 만 사용.
#
# 필요 패키지: numpy, plotly, kaleido (expy.png 저장용)

# %%
import os
import numpy as np

np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


def softmax(z, tau=1.0, axis=-1):
    z = z / tau
    z = z - z.max(axis=axis, keepdims=True)  # 수치 안정화
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(z, tau=1.0, axis=-1):
    z = z / tau
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def entropy(p, axis=-1):
    return -(p * np.log(p + 1e-12)).sum(axis=axis)


# %% [markdown]
# ## 1. softmax 와 온도 $\tau$
#
# 같은 점수 벡터에 $\tau$ 만 바꿔 보면, $\tau$ 가 작을수록 분포가 뾰족(sharp)해진다.
# DINO 는 student 에 $\tau_s=0.1$, teacher 에 $\tau_t=0.04\sim0.07$ 을 준다.

# %%
scores = np.array([0.9, 0.7, 0.5, 0.1, -0.3])  # cos 유사도 같은 prototype score 예시
for tau in [1.0, 0.1, 0.04]:
    p = softmax(scores, tau)
    print(f"tau={tau:<5} p={np.round(p, 4)}  entropy={entropy(p):.3f}")
# 출력:
# tau=1.0   p=[0.3087 0.2527 0.2069 0.1387 0.093 ]  entropy=1.531
# tau=0.1   p=[8.666e-01 1.173e-01 1.590e-02 3.000e-04 0.000e+00]  entropy=0.444
# tau=0.04  p=[0.9933 0.0067 0.     0.     0.    ]  entropy=0.041

# %% [markdown]
# ## 2. cross-entropy 와 그 최소값
#
# $H(p_t,p_s) = -\sum_k p_t^{(k)}\log p_s^{(k)} = H(p_t) + \mathrm{KL}(p_t\|p_s)$ 이므로
# $p_s=p_t$ 일 때 최소이고 그 값은 $H(p_t)$ 이다. 무작위 $p_s$ 로 수치 확인.

# %%
def cross_entropy(p_t, log_p_s):
    return -(p_t * log_p_s).sum(-1)


p_t = softmax(scores, 0.07)
ce_self = cross_entropy(p_t, np.log(p_t))
print(f"H(p_t)            = {entropy(p_t):.4f}")
print(f"CE(p_t, p_s=p_t)  = {ce_self:.4f}")
for _ in range(3):
    s_rand = np.random.randn(5)
    ce = cross_entropy(p_t, log_softmax(s_rand, 0.1))
    print(f"CE(p_t, random p_s) = {ce:.4f}  (>= {ce_self:.4f}: {ce >= ce_self})")
# 출력:
# H(p_t)            = 0.2316
# CE(p_t, p_s=p_t)  = 0.2316
# CE(p_t, random p_s) = 5.5631  (>= 0.2316: True)
# CE(p_t, random p_s) = 18.2089  (>= 0.2316: True)
# CE(p_t, random p_s) = 12.3747  (>= 0.2316: True)

# %% [markdown]
# ## 3. teacher 의 centering + sharpening
#
# 논문/코드: `p_t = softmax((t - c) / tau_t)`, `c <- m*c + (1-m)*mean_batch(t)`, $m=0.9$.
#
# 시나리오: 배치 $B=6$, prototype $K=5$. 모든 샘플의 0번 prototype 점수에 공통 편향 $+3$ 이 있어
# (collapse 초기 증상) 그대로 softmax 하면 모두 0번으로 몰린다. centering 이 그 편향을 제거하는지 본다.

# %%
B, K = 6, 5
t = np.random.randn(B, K) * 0.5
t[:, 0] += 3.0  # 공통 편향: 모든 샘플이 prototype 0 을 선호

center = np.zeros((1, K))
momentum = 0.9
for _ in range(50):  # 여러 iteration 동안 같은 통계의 배치를 봤다고 가정
    center = momentum * center + (1 - momentum) * t.mean(0, keepdims=True)
print("EMA center      :", np.round(center[0], 3))

p_raw = softmax(t, 0.07)                 # centering 없음, sharpening 만
p_center = softmax(t - center, 1.0)      # centering 만, sharpening 없음
p_both = softmax(t - center, 0.07)       # DINO teacher
for name, p in [("sharpen only ", p_raw), ("center only  ", p_center), ("center+sharpen", p_both)]:
    print(f"{name}: argmax={p.argmax(1)}  mean per-sample H={entropy(p).mean():.3f}"
          f"  H(batch-mean p)={entropy(p.mean(0)):.3f}")
# 출력:
# EMA center      : [ 2.619  0.197 -0.076  0.057  0.143]
# sharpen only : argmax=[0 0 0 0 0 0]  mean per-sample H=0.000  H(batch-mean p)=0.000
# center only  : argmax=[1 4 3 0 2 3]  mean per-sample H=1.478  H(batch-mean p)=1.608
# center+sharpen: argmax=[1 4 3 0 2 3]  mean per-sample H=0.239  H(batch-mean p)=1.533

# %% [markdown]
# 해석
# * **sharpen only**: 전부 prototype 0 → 한 차원 지배형 collapse. 배치 평균 분포의 엔트로피 0.
# * **center only**: 샘플별로 다른 prototype 을 가리키지만 각 분포가 흐릿함(H 큼) → 균등분포형 collapse 위험.
# * **center+sharpen**: 각 샘플은 뾰족(H≈0)하면서 배치 전체로는 여러 prototype 에 퍼짐(H(batch-mean) 큼). DINO 가 원하는 상태.

# %% [markdown]
# ## 4. Sinkhorn-Knopp centering (DINOv2 옵션)
#
# `DINOLoss.sinkhorn_knopp_teacher` 를 그대로 옮김: $Q=\exp(t/\tau_t)^\top$ ($K\times B$) 를
# 행 합 $1/K$, 열 합 $1/B$ 가 되도록 3회 번갈아 정규화. 반복할수록 prototype 별 점유율(열 합)이 $B/K$ 로 균등해진다
# (3회는 완전 수렴 전이지만 EMA centering 보다 훨씬 균등하다).

# %%
def sinkhorn_knopp_teacher(teacher_output, teacher_temp, n_iterations=3):
    Q = np.exp(teacher_output / teacher_temp).T  # K x B
    Bn = Q.shape[1]
    Kn = Q.shape[0]
    Q /= Q.sum()
    for _ in range(n_iterations):
        Q /= Q.sum(1, keepdims=True); Q /= Kn   # 행: prototype 별 총 질량 1/K
        Q /= Q.sum(0, keepdims=True); Q /= Bn   # 열: 샘플 별 총 질량 1/B
    Q *= Bn                                     # 열 합 = 1 (각 샘플이 확률분포)
    return Q.T


p_sk = sinkhorn_knopp_teacher(t, 0.07)
print("row sums (per sample)   :", np.round(p_sk.sum(1), 4))
print("col sums, 3 iters (DINOv2):", np.round(p_sk.sum(0), 3), " target B/K =", B / K)
print("col sums, 20 iters       :", np.round(sinkhorn_knopp_teacher(t, 0.07, 20).sum(0), 3))
print("argmax:", p_sk.argmax(1))
print("EMA-centered p_t 의 prototype 점유율:", np.round(p_both.sum(0), 3))
# 출력:
# row sums (per sample)   : [1. 1. 1. 1. 1. 1.]
# col sums, 3 iters (DINOv2): [1.23  0.967 0.916 1.798 1.089]  target B/K = 1.2
# col sums, 20 iters       : [1.145 1.177 1.203 1.253 1.222]
# argmax: [1 4 3 0 2 3]
# EMA-centered p_t 의 prototype 점유율: [1.534 0.529 0.937 1.844 1.156]

# %% [markdown]
# ## 5. 작은 학습 시뮬레이션: centering 유무에 따른 collapse
#
# 진짜 ViT 대신 "class token" 을 3개 군집에서 뽑은 8차원 벡터 $x$ 로 대체하고,
# DINO head 를 선형층 $s = W x + b$ 로 단순화한다.
# 같은 이미지의 두 crop 은 $x$ 에 서로 다른 잡음을 더한 두 벡터로 흉내낸다.
#
# * student: $p_s=\mathrm{softmax}(W_s x_1/\tau_s)$
# * teacher: $p_t=\mathrm{softmax}((W_t x_2 - c)/\tau_t)$, $W_t \leftarrow \lambda W_t+(1-\lambda)W_s$
# * 손실 $-\sum p_t\log p_s$ 의 $W_s$ 에 대한 기울기: $\partial\mathcal{L}/\partial s = (p_s - p_t)/\tau_s$
#   (softmax + cross-entropy 의 표준 결과), 따라서 $\partial\mathcal{L}/\partial W_s = \frac{1}{\tau_s}(p_s-p_t)\,x_1^\top$.

# %%
def make_batch(n_per_cluster=16, dim=8, n_clusters=3, noise=0.3, rng=np.random):
    centers = rng.randn(n_clusters, dim) * 2.0
    xs, labels = [], []
    for ci, cen in enumerate(centers):
        xs.append(cen + rng.randn(n_per_cluster, dim) * 0.5)
        labels += [ci] * n_per_cluster
    x = np.concatenate(xs)
    x1 = x + rng.randn(*x.shape) * noise   # crop 1 (student 가 봄)
    x2 = x + rng.randn(*x.shape) * noise   # crop 2 (teacher 가 봄)
    return x1, x2, np.array(labels)


def train_dino_toy(use_centering, steps=400, K=8, tau_s=0.1, tau_t=0.05, lr=0.5,
                   ema=0.9, center_m=0.9, seed=1):
    rng = np.random.RandomState(seed)
    x1, x2, labels = make_batch(rng=rng)
    x1 = np.hstack([x1, np.ones((len(x1), 1))])  # bias 항
    x2 = np.hstack([x2, np.ones((len(x2), 1))])
    W_s = rng.randn(K, x1.shape[1]) * 0.1
    W_s[0, -1] += 2.0                           # prototype 0 을 선호하는 초기 편향 (collapse 씨앗)
    W_t = W_s.copy()
    c = np.zeros((1, K))
    losses = []
    for _ in range(steps):
        t_out = x2 @ W_t.T                      # teacher prototype scores (no grad)
        if use_centering:
            p_t = softmax(t_out - c, tau_t)
            c = center_m * c + (1 - center_m) * t_out.mean(0, keepdims=True)
        else:
            p_t = softmax(t_out, tau_t)
        s_out = x1 @ W_s.T
        log_p_s = log_softmax(s_out, tau_s)
        loss = cross_entropy(p_t, log_p_s).mean()
        losses.append(loss)
        grad_s = (np.exp(log_p_s) - p_t) / tau_s          # dL/ds
        grad_W = grad_s.T @ x1 / len(x1)                   # dL/dW_s
        W_s -= lr * grad_W
        W_t = ema * W_t + (1 - ema) * W_s                  # teacher EMA
    p_final = softmax(x1 @ W_s.T, tau_s)
    return np.array(losses), p_final, labels


for use_c in [False, True]:
    losses, p_final, labels = train_dino_toy(use_c)
    assign = p_final.argmax(1)
    usage = np.bincount(assign, minlength=8)
    print(f"centering={use_c!s:<5} final loss={losses[-1]:.4f}  prototype usage={usage}"
          f"  #used={np.count_nonzero(usage)}  H(mean p)={entropy(p_final.mean(0)):.3f}")
    for ci in range(3):
        print(f"   cluster {ci}: assigned prototypes -> {np.bincount(assign[labels == ci], minlength=8)}")
# 출력:
# centering=False final loss=0.0000  prototype usage=[48  0  0  0  0  0  0  0]  #used=1  H(mean p)=0.000
#    cluster 0: assigned prototypes -> [16  0  0  0  0  0  0  0]
#    cluster 1: assigned prototypes -> [16  0  0  0  0  0  0  0]
#    cluster 2: assigned prototypes -> [16  0  0  0  0  0  0  0]
# centering=True  final loss=0.0000  prototype usage=[ 0  0 16  0  0  0 16 16]  #used=3  H(mean p)=1.099
#    cluster 0: assigned prototypes -> [ 0  0  0  0  0  0 16  0]
#    cluster 1: assigned prototypes -> [ 0  0  0  0  0  0  0 16]
#    cluster 2: assigned prototypes -> [ 0  0 16  0  0  0  0  0]

# %% [markdown]
# 두 경우 모두 손실은 0 에 수렴하지만 결과는 전혀 다르다. centering 이 없으면 모든 샘플(3 군집 48개)이
# prototype 0 하나에 배정되는 **collapse** — 손실 값만으로는 구분할 수 없는 자명해다.
# centering 이 있으면 군집마다 서로 다른 prototype(2, 6, 7)을 쓰고, 배치 평균 분포의 엔트로피는 $\log 3\approx1.099$ 로 최대.
# 실제 DINOv2 는 $K=65{,}536$, ViT backbone, 3-layer MLP head 로 같은 원리를 대규모로 수행한다.

# %% [markdown]
# ## 6. 시각화 → `expy.png`
#
# (왼쪽) 학습 곡선, (가운데/오른쪽) 학습 후 student 의 $p_s$ 히트맵 (행=샘플, 열=prototype).

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

res = {use_c: train_dino_toy(use_c) for use_c in [False, True]}
fig = make_subplots(
    rows=1, cols=3, column_widths=[0.34, 0.33, 0.33],
    subplot_titles=("DINO loss (toy)", "p_s without centering (collapse)", "p_s with centering"),
)
for use_c, name in [(False, "no centering"), (True, "centering")]:
    fig.add_trace(go.Scatter(y=res[use_c][0], mode="lines", name=name), row=1, col=1)
for col, use_c in [(2, False), (3, True)]:
    fig.add_trace(go.Heatmap(z=res[use_c][1], colorscale="Blues", zmin=0, zmax=1,
                             showscale=(col == 3), colorbar=dict(title="p_s")), row=1, col=col)
    fig.update_xaxes(title_text="prototype k", row=1, col=col)
    fig.update_yaxes(title_text="sample (3 clusters x 16)", row=1, col=col)
fig.update_xaxes(title_text="step", row=1, col=1)
fig.update_yaxes(title_text="-sum p_t log p_s", row=1, col=1)
fig.update_layout(width=1200, height=420, template="plotly_white",
                  title_text="DINO image-level objective: centering 이 collapse 를 막는다",
                  legend=dict(x=0.02, y=0.98))
_show(fig)
png_path = os.path.join(HERE, "expy.png")
try:
    fig.write_image(png_path, scale=2)
    print("saved:", png_path)
except Exception as e:  # kaleido 미설치 등
    print("write_image 실패:", e)
# 출력:
# saved: /home/sungwoo/projects/swcho/dinov2/fm/papers/.fm/hints/10bf9f4b-5c35-4dc9-a71a-f6a28629b67e/expy.png
