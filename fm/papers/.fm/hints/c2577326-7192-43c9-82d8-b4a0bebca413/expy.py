# %% [markdown]
# # 이미지 유사도 함수 $m(s,r)$ = 코사인 유사도 — 실험으로 이해하기
#
# DINOv2 논문(Appendix A.2)은 두 이미지 $s, r$의 유사도를 다음으로 정의한다.
#
# $$m(s,r)=\text{cosine-similarity}\big(f(s),f(r)\big)=\frac{f(s)\cdot f(r)}{\|f(s)\|_2\,\|f(r)\|_2}$$
#
# $f$는 이미지를 특징 벡터로 바꾸는 모델(검색에는 ViT-H/16, 중복 제거에는 Pizzi et al. 2022 임베딩).
# 이 스크립트는 $f$ 대신 numpy 벡터를 직접 만들어 다음을 확인한다.
#
# 1. 코사인 유사도를 직접 구현하고, 각도가 알려진 2차원 벡터로 검증
# 2. 크기 불변성: $m(a,3a)=1$, 유클리드 거리와의 비교
# 3. $\ell_2$ 정규화 후 $\|\hat a-\hat b\|^2 = 2-2\cos\theta$ 수치 검증
# 4. 장난감 "이미지 임베딩" 집합에서 같은 군집/다른 군집 유사도 분포와 논문 임계값 0.6 / 0.45
# 5. plotly 시각화 → `expy.png`

# %%
# 필요 패키지: numpy, plotly, kaleido
# 기본 python3에는 numpy가 없으므로 다음으로 실행:
#   /home/sungwoo/miniforge3/envs/trellis/bin/python expy.py
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

np.set_printoptions(precision=4, suppress=True)
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# %% [markdown]
# ## 1. 코사인 유사도 직접 구현 & 2차원 검증
#
# 고교 기하: $\vec a\cdot\vec b=|\vec a||\vec b|\cos\theta$ 이므로
# $\cos\theta=\dfrac{\vec a\cdot\vec b}{|\vec a||\vec b|}$.
# 성분 정의 $\vec a\cdot\vec b=\sum_i a_ib_i$, $\|\vec a\|_2=\sqrt{\sum_i a_i^2}$ 로 그대로 구현한다.

# %%
def cosine_similarity(a, b):
    """m(s, r) = f(s)·f(r) / (||f(s)||_2 ||f(r)||_2) — 성분 합으로 직접 계산"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    dot = np.sum(a * b)
    norm_a = np.sqrt(np.sum(a * a))
    norm_b = np.sqrt(np.sum(b * b))
    return dot / (norm_a * norm_b)


# 각도를 알고 있는 2차원 벡터로 검증: a=(1,0), b=(cosθ, sinθ)·(길이 임의)
a = np.array([1.0, 0.0])
for deg in [0, 30, 45, 60, 90, 120, 180]:
    th = np.deg2rad(deg)
    b = 2.5 * np.array([np.cos(th), np.sin(th)])  # 길이 2.5 — 크기는 결과에 영향 없어야 함
    m = cosine_similarity(a, b)
    print(f"θ={deg:3d}°  m(a,b)={m:+.4f}  cos(θ)={np.cos(th):+.4f}  일치={np.isclose(m, np.cos(th))}")
# 출력:
# θ=  0°  m(a,b)=+1.0000  cos(θ)=+1.0000  일치=True
# θ= 30°  m(a,b)=+0.8660  cos(θ)=+0.8660  일치=True
# θ= 45°  m(a,b)=+0.7071  cos(θ)=+0.7071  일치=True
# θ= 60°  m(a,b)=+0.5000  cos(θ)=+0.5000  일치=True
# θ= 90°  m(a,b)=+0.0000  cos(θ)=+0.0000  일치=True
# θ=120°  m(a,b)=-0.5000  cos(θ)=-0.5000  일치=True
# θ=180°  m(a,b)=-1.0000  cos(θ)=-1.0000  일치=True

# %%
# 3차원 예시 (hi.md의 예): a=(1,2,2), b=(2,1,2) → 내적 8, 노름 3·3 → 8/9
print("3차원 m =", cosine_similarity([1, 2, 2], [2, 1, 2]), " 기대값 8/9 =", 8 / 9)
# 출력: 3차원 m = 0.8888888888888888  기대값 8/9 = 0.8888888888888888

# %% [markdown]
# ## 2. 크기 불변성 — 유클리드 거리와의 비교
#
# 특징 벡터의 "길이"는 내용과 무관한 요인(밝기·스케일)에 흔들릴 수 있다.
# 코사인은 방향만 보므로 $m(a, 3a)=1$ 이지만, 유클리드 거리 $\|a-3a\|=2\|a\|$는 크게 벌어진다.

# %%
rng = np.random.default_rng(0)
a = rng.normal(size=64)                # 64차원 "임베딩"
for k in [1, 3, 10, 0.01]:
    b = k * a
    print(f"b = {k:>5}·a : m(a,b) = {cosine_similarity(a, b):.4f}, "
          f"유클리드 ||a-b|| = {np.linalg.norm(a - b):8.4f}")
# 출력:
# b =     1·a : m(a,b) = 1.0000, 유클리드 ||a-b|| =   0.0000
# b =     3·a : m(a,b) = 1.0000, 유클리드 ||a-b|| =  14.6307
# b =    10·a : m(a,b) = 1.0000, 유클리드 ||a-b|| =  65.8381
# b =  0.01·a : m(a,b) = 1.0000, 유클리드 ||a-b|| =   7.2422

# %% [markdown]
# ## 3. $\ell_2$ 정규화 후: $\|\hat a-\hat b\|^2 = 2-2\cos\theta$
#
# 단위벡터 $\hat a=a/\|a\|$, $\hat b=b/\|b\|$에 대해
# $\|\hat a-\hat b\|^2=\|\hat a\|^2-2\hat a\cdot\hat b+\|\hat b\|^2=2-2\cos\theta$.
# 정규화하면 유클리드 거리와 코사인 유사도가 단조 관계 → 최근접 이웃 순위가 같다.

# %%
def l2_normalize(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


max_err = 0.0
for _ in range(1000):
    D = rng.integers(2, 512)
    a, b = rng.normal(size=D), rng.normal(size=D)
    lhs = np.sum((l2_normalize(a) - l2_normalize(b)) ** 2)
    rhs = 2 - 2 * cosine_similarity(a, b)
    max_err = max(max_err, abs(lhs - rhs))
print(f"1000개 무작위 쌍(D=2~511)에서 |LHS-RHS| 최대 오차 = {max_err:.2e}")
# 출력: 1000개 무작위 쌍(D=2~511)에서 |LHS-RHS| 최대 오차 = 1.33e-15

# 정규화 후 내적 = 코사인 유사도 (Faiss 내적 인덱스로 코사인 검색이 가능한 이유)
a, b = rng.normal(size=64), rng.normal(size=64)
print("정규화 후 내적 =", np.dot(l2_normalize(a), l2_normalize(b)),
      " / 코사인 =", cosine_similarity(a, b))
# 출력: 정규화 후 내적 = -0.10885924081567655  / 코사인 = -0.10885924081567658

# %% [markdown]
# ## 4. 장난감 이미지 임베딩: 같은 군집 vs 다른 군집
#
# - "원본 이미지" 20장 → 각각 무작위 단위벡터 ($D=64$)
# - 각 원본의 "근사 중복(near-duplicate)" 7장 → 원본 + 작은 잡음 (리사이즈·크롭·압축된 사본 흉내)
# - 쌍별 코사인 행렬을 계산해, 같은 군집 쌍과 다른 군집 쌍의 유사도 분포를 비교한다.
# - 논문 임계값: self-dedup $m>0.6$, relative dedup $m>0.45$

# %%
D, n_base, n_dup, noise = 64, 20, 7, 0.55
bases = rng.normal(size=(n_base, D))
bases /= np.linalg.norm(bases, axis=1, keepdims=True)

emb, label = [], []
for c in range(n_base):
    for _ in range(n_dup):
        v = bases[c] + noise * rng.normal(size=D) / np.sqrt(D)  # 잡음 크기 ≈ noise
        emb.append(v)
        label.append(c)
emb, label = np.array(emb), np.array(label)
emb_n = emb / np.linalg.norm(emb, axis=1, keepdims=True)  # ℓ2 정규화

S = emb_n @ emb_n.T                       # 쌍별 코사인 유사도 행렬 (정규화 후 내적)
N = len(emb)
iu = np.triu_indices(N, k=1)              # 상삼각(중복 없이 쌍만)
same = (label[:, None] == label[None, :])[iu]
sims = S[iu]
same_sims, diff_sims = sims[same], sims[~same]

print(f"임베딩 {N}개, 쌍 {len(sims)}개 (같은 군집 {same.sum()}, 다른 군집 {(~same).sum()})")
print(f"같은 군집: 평균 {same_sims.mean():.3f}, 최소 {same_sims.min():.3f}, 최대 {same_sims.max():.3f}")
print(f"다른 군집: 평균 {diff_sims.mean():.3f}, 최소 {diff_sims.min():.3f}, 최대 {diff_sims.max():.3f}")
for thr, name in [(0.6, "self-dedup"), (0.45, "relative dedup")]:
    print(f"임계값 {thr} ({name}): 같은 군집 중 {np.mean(same_sims > thr):.1%} 통과, "
          f"다른 군집 중 {np.mean(diff_sims > thr):.1%} 오탐")
# 출력:
# 임베딩 140개, 쌍 9730개 (같은 군집 420, 다른 군집 9310)
# 같은 군집: 평균 0.770, 최소 0.606, 최대 0.868
# 다른 군집: 평균 -0.011, 최소 -0.473, 최대 0.477
# 임계값 0.6 (self-dedup): 같은 군집 중 100.0% 통과, 다른 군집 중 0.0% 오탐
# 임계값 0.45 (relative dedup): 같은 군집 중 100.0% 통과, 다른 군집 중 0.0% 오탐

# %% [markdown]
# 다른 군집(무관한 이미지)은 $D=64$에서 대체로 $0$ 근처에 몰리고, 같은 군집(근사 중복)은 높은 값에 몰린다.
# 논문은 self-dedup에 $0.6$을, 평가셋 유출 방지용 relative dedup에는 더 엄격한(낮은) $0.45$를 쓴다 —
# 문턱을 낮출수록 "같은 군집"을 놓치는 일은 줄고 대신 무관한 이미지가 걸릴 위험이 늘어난다.
# 위 실험에서는 두 분포가 충분히 떨어져 있어 $0.45$에서도 오탐이 없다.

# %% [markdown]
# ## 5. 시각화 — 2차원 벡터 다이어그램 + 유사도 히스토그램

# %%
fig = make_subplots(
    rows=1, cols=2, column_widths=[0.4, 0.6],
    subplot_titles=("2D: cos θ = a·b / (|a||b|)  (b와 3b는 방향이 같아 m(a,b)=m(a,3b))",
                    f"쌍별 코사인 유사도 분포 (D={D}, 단위벡터 군집 시뮬레이션)"),
)

# (좌) 2차원 벡터 다이어그램
a2 = np.array([1.0, 0.0])
th = np.deg2rad(45)
b2 = 0.8 * np.array([np.cos(th), np.sin(th)])
b2_scaled = 3 * b2
for v, name, color, dash in [(a2, "a=(1,0)", "#1f77b4", "solid"),
                             (b2, f"b (m={cosine_similarity(a2, b2):.3f})", "#d62728", "solid"),
                             (b2_scaled, f"3b (m={cosine_similarity(a2, b2_scaled):.3f})", "#d62728", "dot")]:
    fig.add_trace(go.Scatter(x=[0, v[0]], y=[0, v[1]], mode="lines+markers", name=name,
                             line=dict(color=color, width=3, dash=dash),
                             marker=dict(size=[0, 10], symbol="arrow", angleref="previous")),
                  row=1, col=1)
arc_t = np.linspace(0, th, 30)
fig.add_trace(go.Scatter(x=0.5 * np.cos(arc_t), y=0.5 * np.sin(arc_t), mode="lines",
                         line=dict(color="gray", width=1), showlegend=False), row=1, col=1)
fig.add_annotation(x=0.6, y=0.18, text="θ=45°", showarrow=False, row=1, col=1)
fig.update_xaxes(range=[-0.2, 2.6], scaleanchor="y", row=1, col=1)
fig.update_yaxes(range=[-0.2, 2.6], row=1, col=1)

# (우) 히스토그램 + 임계값
fig.add_trace(go.Histogram(x=diff_sims, name="다른 군집 (무관한 이미지)", nbinsx=60,
                           marker_color="#1f77b4", opacity=0.75), row=1, col=2)
fig.add_trace(go.Histogram(x=same_sims, name="같은 군집 (근사 중복)", nbinsx=30,
                           marker_color="#ff7f0e", opacity=0.75), row=1, col=2)
for thr, name, color, pos in [(0.6, "self-dedup >0.6", "#d62728", "top right"),
                              (0.45, "relative dedup >0.45", "#2ca02c", "top left")]:
    fig.add_vline(x=thr, line=dict(color=color, width=2, dash="dash"), row=1, col=2,
                  annotation_text=name, annotation_position=pos,
                  annotation_font=dict(color=color))
fig.update_xaxes(title_text="m(s, r) = cosine similarity", range=[-0.6, 1.0], row=1, col=2)
fig.update_yaxes(title_text="쌍 개수 (log)", type="log", row=1, col=2)
fig.update_layout(barmode="overlay", width=1200, height=520,
                  title_text="DINOv2 이미지 유사도 m(s,r): 코사인 유사도와 중복 제거 임계값",
                  legend=dict(orientation="h", y=-0.15))

_show(fig)
png_path = os.path.join(HERE, "expy.png")
fig.write_image(png_path, scale=2)
print("저장:", png_path)
# 출력: 저장: .../c2577326-7192-43c9-82d8-b4a0bebca413/expy.png
