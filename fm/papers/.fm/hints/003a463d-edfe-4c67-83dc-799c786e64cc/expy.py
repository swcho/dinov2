# %% [markdown]
# # DINOv2 retrieval에서 N=4를 고른 이유 — collision 트레이드오프 실험
#
# DINOv2는 curated 이미지(query) 하나마다 uncurated pool에서 **최근접 이웃 $N$개**를 가져와
# 데이터셋을 키운다(sample-based retrieval). 논문은 $N$을 훨씬 크게 해도 검색 결과가
# *눈으로 보면* 괜찮았지만, **collision**(여러 query의 이웃으로 중복 검색되는 pool 이미지)이
# 늘어나서 $N=4$를 절충점으로 택했다고 말한다.
#
# 이 스크립트는 합성 임베딩으로 그 트레이드오프를 재현한다.
#
# - **품질 프록시**: $N$번째 이웃의 코사인 유사도 (멀어질수록 관련성 하락)
# - **collision 비율**: $1 - \dfrac{|\text{unique retrieved}|}{N \cdot |Q|}$
#
# 필요 패키지: numpy, scipy, plotly, kaleido (정적 PNG 저장용)

# %%
import os
import numpy as np
from scipy.spatial import cKDTree

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
# ## 1. 합성 임베딩 공간 만들기
#
# 실제 논문은 ViT-H/16 임베딩 + 코사인 유사도를 쓴다. 여기서는 $d=32$ 차원 공간에
# **개념 클러스터** $C$개를 두고, curated query와 uncurated pool을 같은 클러스터 주변에서
# 샘플링한다. 단위 벡터로 정규화하면 유클리드 최근접 = 코사인 최근접이 된다:
#
# $$\|u-v\|_2^2 = 2 - 2\,u\cdot v \quad (\|u\|=\|v\|=1)$$

# %%
D, C = 32, 50            # 임베딩 차원, 개념(클러스터) 수
Q_PER_C, P_PER_C = 8, 400  # 클러스터당 query 수 / pool 수


def make_points(n_per_cluster, centers, spread):
    pts = np.concatenate(
        [c + spread * rng.standard_normal((n_per_cluster, D)) for c in centers]
    )
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


centers = rng.standard_normal((C, D))
centers /= np.linalg.norm(centers, axis=1, keepdims=True)

queries = make_points(Q_PER_C, centers, spread=0.25)   # curated (예: ImageNet-22k)
pool = make_points(P_PER_C, centers, spread=0.35)      # uncurated (예: 웹 크롤)

print(f"queries: {queries.shape}, pool: {pool.shape}")
# 출력: queries: (400, 32), pool: (20000, 32)

# %% [markdown]
# ## 2. $N$을 바꾸며 sample-based retrieval 수행
#
# query마다 $N$개의 최근접 pool 이미지를 가져온 뒤,
#
# - 총 검색 건수 $N\cdot|Q|$ (Table 15의 "Retrieved" 열이 정확히 이 값이다),
# - 실제로 **서로 다른** 이미지 수 (collision을 제거한 진짜 증강량),
# - $N$번째 이웃의 평균 코사인 유사도 (품질 프록시)
#
# 를 기록한다.

# %%
tree = cKDTree(pool)
N_LIST = [1, 2, 4, 8, 16, 32, 64]
MAX_N = max(N_LIST)

dist, idx = tree.query(queries, k=MAX_N)          # 거리 오름차순
cos = 1 - dist**2 / 2                             # 유클리드 -> 코사인 유사도

rows = []
for N in N_LIST:
    retrieved = idx[:, :N]
    total = retrieved.size
    unique = np.unique(retrieved).size
    rows.append(
        dict(
            N=N,
            total=total,
            unique=unique,
            collision_rate=1 - unique / total,
            nth_cos=cos[:, N - 1].mean(),
            mean_cos=cos[:, :N].mean(),
        )
    )

print(f"{'N':>3} {'total':>6} {'unique':>6} {'collision':>9} {'N-th cos':>8} {'mean cos':>8}")
for r in rows:
    print(f"{r['N']:>3} {r['total']:>6} {r['unique']:>6} {r['collision_rate']:>9.1%} "
          f"{r['nth_cos']:>8.3f} {r['mean_cos']:>8.3f}")
# 출력:
#   N  total unique collision N-th cos mean cos
#   1    400    393      1.7%    0.669    0.669
#   2    800    760      5.0%    0.639    0.654
#   4   1600   1482      7.4%    0.614    0.637
#   8   3200   2836     11.4%    0.585    0.616
#  16   6400   5239     18.1%    0.556    0.592
#  32  12800   9035     29.4%    0.523    0.564
#  64  25600  13857     45.9%    0.488    0.534

# %% [markdown]
# 관찰:
#
# - $N$이 커져도 $N$번째 이웃의 유사도는 **완만하게** 떨어진다 → "눈으로 보면 여전히 괜찮다".
# - 반면 collision 비율은 $N$이 커질수록 계속 늘어 $N=32$에서 약 30%, $N=64$에서 약 46%에 이른다. 같은 개념의 query 8개가
#   같은 pool 영역을 파고 들어가면서 서로의 이웃을 중복으로 가져오기 때문이다.
# - $N=4$는 유사도 손실이 작으면서(0.669→0.614) collision이 한 자리수 %(7.4%)에 머무는 지점이다.

# %% [markdown]
# ## 3. collision의 정체: pool 이미지가 몇 개의 query에 잡혔는가
#
# collision은 "pool 이미지 하나가 여러 query의 이웃"인 상황이다.
# 그 **다중도(multiplicity)** 분포를 $N=4$와 $N=32$에서 비교한다.

# %%
def multiplicity(N):
    counts = np.bincount(idx[:, :N].ravel(), minlength=len(pool))
    return counts[counts > 0]


for N in (4, 32):
    m = multiplicity(N)
    hist = np.bincount(m)[1:]
    print(f"N={N:>2}: 1회 {hist[0]:>5}, 2회 {hist[1] if len(hist) > 1 else 0:>4}, "
          f"3회+ {hist[2:].sum() if len(hist) > 2 else 0:>4}, 최대 {m.max()}회")
# 출력:
# N= 4: 1회  1376, 2회   96, 3회+   10, 최대 4회
# N=32: 1회  6234, 2회 2073, 3회+  728, 최대 8회

# %% [markdown]
# ## 4. 논문 Table 15와 대조: Retrieved 열은 정확히 $N \times |Q|$
#
# 논문 부록 Table 15의 sample-based 행은 collision을 세지 않은 **명목** 검색 수를 적는다.
# 즉 실제 고유 이미지 수는 이보다 적을 수 있고, $N$이 클수록 그 괴리(collision)가 커진다.

# %%
table15 = {
    "ImageNet-22k (N=4)": (14_197_086, 4, 56_788_344),
    "ImageNet-1k (N=32)": (1_281_167, 32, 40_997_344),
    "Google Landmarks v2 (N=4)": (1_580_470, 4, 6_321_880),
}
for name, (q, n, retrieved) in table15.items():
    print(f"{name:<28} {q:>11,} x {n:>2} = {q*n:>11,}  (논문: {retrieved:>11,}) "
          f"{'OK' if q*n == retrieved else 'MISMATCH'}")
# 출력:
# ImageNet-22k (N=4)            14,197,086 x  4 =  56,788,344  (논문:  56,788,344) OK
# ImageNet-1k (N=32)             1,281,167 x 32 =  40,997,344  (논문:  40,997,344) OK
# Google Landmarks v2 (N=4)      1,580,470 x  4 =   6,321,880  (논문:   6,321,880) OK

# %% [markdown]
# ## 5. 시각화: 품질 vs collision 트레이드오프

# %%
Ns = [r["N"] for r in rows]
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "품질 프록시 vs collision 비율",
        "고유 검색 이미지 수 (명목 N·|Q| 대비)",
    ),
    specs=[[{"secondary_y": True}, {}]],
)

fig.add_trace(
    go.Scatter(x=Ns, y=[r["nth_cos"] for r in rows], name="N번째 이웃 코사인 유사도",
               mode="lines+markers", line=dict(color="#1f77b4")),
    row=1, col=1, secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=Ns, y=[100 * r["collision_rate"] for r in rows], name="collision 비율 (%)",
               mode="lines+markers", line=dict(color="#d62728")),
    row=1, col=1, secondary_y=True,
)
fig.add_vline(x=4, line_dash="dash", line_color="gray", row=1, col=1,
              annotation_text="N=4", annotation_position="top left")

fig.add_trace(
    go.Scatter(x=Ns, y=[r["total"] for r in rows], name="명목 N·|Q|",
               mode="lines+markers", line=dict(color="gray", dash="dot")),
    row=1, col=2,
)
fig.add_trace(
    go.Scatter(x=Ns, y=[r["unique"] for r in rows], name="고유 이미지 수",
               mode="lines+markers", line=dict(color="#2ca02c")),
    row=1, col=2,
)

fig.update_xaxes(type="log", title_text="N (query당 최근접 이웃 수)", tickvals=Ns)
fig.update_yaxes(title_text="코사인 유사도", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="collision (%)", row=1, col=1, secondary_y=True)
fig.update_yaxes(title_text="이미지 수", row=1, col=2)
fig.update_layout(
    title="DINOv2 sample-based retrieval: N을 키우면 품질은 완만히, collision은 급격히",
    width=1100, height=450, legend=dict(orientation="h", y=-0.25),
)
_show(fig)

png_path = os.path.join(HERE, "expy.png")
try:
    fig.write_image(png_path, scale=2)
    print("saved:", png_path)
except Exception as e:  # kaleido 미설치 등
    print("PNG 저장 실패:", e)
# 출력: saved: /home/sungwoo/projects/swcho/dinov2/fm/papers/.fm/hints/003a463d-edfe-4c67-83dc-799c786e64cc/expy.png

# %% [markdown]
# ## 정리
#
# | 관점 | $N$ 작음 (1~4) | $N$ 큼 (32~64) |
# |---|---|---|
# | $N$번째 이웃 유사도 | 높음 | 완만히 하락 — 시각적으로는 여전히 그럴듯 |
# | collision | 수 % | 수십 % — 여러 query가 같은 이미지를 중복 검색 |
# | 실효 증강량 | $\approx N\cdot|Q|$ | 명목치보다 크게 작음 |
#
# 논문의 선택 $N=4$: "훨씬 큰 $N$도 시각적 품질은 좋아 보였지만 collision이 늘어난다"는
# 관찰과 정확히 일치한다. 큰 데이터셋(ImageNet-22k, GLDv2)에는 $N=4$,
# 핵심 축으로 삼은 ImageNet-1k에만 예외적으로 $N=32$를 썼다(부록 A.4).
