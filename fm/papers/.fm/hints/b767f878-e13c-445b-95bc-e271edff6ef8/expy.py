# %% [markdown]
# # DINOv2 cluster-based retrieval 축소 모형 (2-D toy)
#
# DINOv2(2304.07193) Appendix A.4의 **cluster-based retrieval** 절차를 2차원 점으로 흉내 낸다.
#
# | 논문 (실제 규모) | 이 toy |
# |---|---|
# | uncurated 소스 7.44억 장 (dedup 후) | pool 점 20,000개 |
# | 분산 $k$-means, $K = 100{,}000$ 클러스터 | sklearn KMeans, $K = 200$ |
# | 클러스터당 평균 $\approx 7{,}440$ 장 | 클러스터당 평균 $100$ 점 |
# | 검색 대상 데이터셋 이미지가 **3장 넘게**(> 3) 속한 클러스터 선택 | 동일하게 **> 3** |
# | 선택 클러스터마다 $M = 10{,}000$ 장 추출 | 클러스터마다 $m = 100$ 점 |
# | 데이터셋당 최대 $1{,}000{,}000$ 장 | 데이터셋당 최대 $300$ 점 |
# | sample-based: 쿼리당 $k = 4$ 최근접 | 동일하게 $k = 4$ |
#
# 비교 포인트: 작은 쿼리 집합(60점)에 대해 sample-based는 최대 $4 \times 60 = 240$ 점만 가져오지만,
# cluster-based는 "쿼리가 몰린 개념 영역(클러스터)" 전체를 가져오므로 훨씬 크게 확장된다.

# %%
# 필요 패키지: numpy, scipy, scikit-learn, plotly, kaleido
# 실행 환경: /home/sungwoo/miniforge3/envs/trellis/bin/python (기본 python3에는 numpy가 없음)
import os
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
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
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."

# %% [markdown]
# ## 0. Uncurated pool 만들기 — 소수의 지배적 모드 + 다수의 희소 모드
#
# 웹 크롤 이미지 분포를 흉내 낸다: 몇 개의 **지배적 모드**(예: 셀카, 스크린샷 같은 흔한 이미지)에 점이 몰려 있고,
# 그 밖에 수십 개의 **희소 모드**(특정 새 종, 특정 항공기 기종 같은 개념)가 흩어져 있다.
# §3.3의 "a few dominant modes에 과적합하지 않도록 rebalance"가 필요한 상황이다.

# %%
N_POOL = 20_000
n_dense_modes, n_sparse_modes = 6, 80
dense_centers = rng.uniform(-8, 8, size=(n_dense_modes, 2))
sparse_centers = rng.uniform(-10, 10, size=(n_sparse_modes, 2))

n_dense_each = 2_000                           # 6 × 2000 = 12,000
n_sparse_each = (N_POOL - n_dense_each * n_dense_modes) // n_sparse_modes   # 100
pool_parts, pool_mode = [], []
for i, c in enumerate(dense_centers):
    pool_parts.append(c + rng.normal(0, 0.9, size=(n_dense_each, 2)))
    pool_mode.append(np.full(n_dense_each, i))
for j, c in enumerate(sparse_centers):
    pool_parts.append(c + rng.normal(0, 0.35, size=(n_sparse_each, 2)))
    pool_mode.append(np.full(n_sparse_each, n_dense_modes + j))
pool = np.vstack(pool_parts)
pool_mode = np.concatenate(pool_mode)
print("pool size:", pool.shape, "| dense pts:", n_dense_each * n_dense_modes, "| sparse pts:", n_sparse_each * n_sparse_modes)
# 출력: pool size: (20000, 2) | dense pts: 12000 | sparse pts: 8000

# %% [markdown]
# ## 1. 작은 curated 쿼리 데이터셋 두 개
#
# - **데이터셋 A** (60점): 희소 모드 3곳에 18점씩 몰려 있고(개념이 뚜렷), 나머지 6점은 pool 곳곳에 혼자 떨어진 잡음.
#   → 잡음 점이 걸린 클러스터는 쿼리가 1개뿐이므로 "> 3" 조건에서 걸러져야 한다.
# - **데이터셋 B** (20점): 희소 모드 1곳에 몰려 있음. A보다 작아 균형(cap) 효과를 대비해서 볼 수 있다.
#
# 논문에서는 A, B 자리에 Flowers-102(1,020장), CUB-200(5,994장), AmsterTime(1,231장) 같은 소규모 fine-grained 데이터셋이 들어간다.

# %%
sparse_idx = rng.choice(n_sparse_modes, size=4, replace=False)
A_centers, B_center = sparse_centers[sparse_idx[:3]], sparse_centers[sparse_idx[3]]
A_query = np.vstack([c + rng.normal(0, 0.3, size=(18, 2)) for c in A_centers]
                    + [rng.uniform(-10, 10, size=(6, 2))])        # 54 + 6 = 60
B_query = B_center + rng.normal(0, 0.3, size=(20, 2))
datasets = {"A": A_query, "B": B_query}
print({k: v.shape for k, v in datasets.items()})
# 출력: {'A': (60, 2), 'B': (20, 2)}

# %% [markdown]
# ## 2. Step 1 — uncurated pool을 $K$개 클러스터로 k-means
#
# 논문은 7.44억 장 임베딩(ViT-H/16 self-supervised feature, cosine similarity)을 분산 k-means로 $K = 100{,}000$개로 나눈다.
# "각 클러스터가 서로 다른 이미지 개념/내용을 담아야 한다"는 것이 전제.
# 여기서는 $K = 200$ → 클러스터당 평균 $20{,}000 / 200 = 100$점 (논문: $744\text{M}/100\text{k} \approx 7{,}440$장).

# %%
K = 200
km = KMeans(n_clusters=K, n_init=1, random_state=0).fit(pool)
pool_label = km.labels_
sizes = np.bincount(pool_label, minlength=K)
print(f"K={K}, cluster size mean={sizes.mean():.1f}, min={sizes.min()}, max={sizes.max()}")
# 출력: K=200, cluster size mean=100.0, min=28, max=222

# %% [markdown]
# ## 3. Step 2 — 쿼리를 클러스터에 배정하고, 쿼리가 **3개 넘게** 속한 클러스터만 선택
#
# 클러스터 $c$에 배정된 쿼리 수를 $n_c$라 하면 선택 집합은 $S = \{c : n_c > 3\}$.
# 혼자 떨어진 잡음 쿼리($n_c = 1$)가 걸린 클러스터는 배제된다 — 이것이 "> 3" 조건의 역할이다.

# %%
def select_clusters(query, km, min_count=3):
    q_label = km.predict(query)
    counts = np.bincount(q_label, minlength=km.n_clusters)
    selected = np.flatnonzero(counts > min_count)
    return q_label, counts, selected

sel = {}
for name, q in datasets.items():
    q_label, counts, selected = select_clusters(q, km)
    sel[name] = selected
    touched = np.flatnonzero(counts)
    print(f"[{name}] 쿼리 {len(q)}점이 닿은 클러스터 {len(touched)}개 → 선택(>3) {len(selected)}개;",
          "선택된 클러스터의 쿼리 수:", counts[selected].tolist(),
          "| 배제된 클러스터의 쿼리 수:", counts[np.setdiff1d(touched, selected)].tolist())
# 출력: [A] 쿼리 60점이 닿은 클러스터 13개 → 선택(>3) 4개; 선택된 클러스터의 쿼리 수: [9, 17, 18, 5] | 배제된 클러스터의 쿼리 수: [1, 1, 3, 1, 1, 1, 1, 1, 1]
# 출력: [B] 쿼리 20점이 닿은 클러스터 2개 → 선택(>3) 2개; 선택된 클러스터의 쿼리 수: [9, 11] | 배제된 클러스터의 쿼리 수: []
# 관찰: A의 한 클러스터는 쿼리가 정확히 3개인데도 배제됐다 — 조건이 '3장 이상'이 아니라 '3장 넘게(> 3)'이기 때문.
#       18점씩 넣은 한 영역이 k-means 경계에 걸려 두 클러스터로 갈라진 것(17 + 1, 9 + 3 + ...)도 보인다.

# %% [markdown]
# ## 4. Step 3 — 선택 클러스터마다 $m$점 추출, 데이터셋당 cap으로 rebalancing
#
# 논문: 클러스터마다 $M = 10{,}000$장(평균 크기 7,440보다 커서 사실상 클러스터 대부분을 가져옴),
# 데이터셋당 최대 $1\text{M}$장으로 잘라 LVD-142M 안에서 데이터셋 간 균형을 유지.
# Table 15를 보면 Food-101은 21.67M장, Met은 62.86M장이 검색됐지만 모두 1M으로 잘렸다.
# 여기서는 $m = 100$, cap $= 300$ (A에는 cap이 걸리고 B에는 안 걸리게 잡았다).

# %%
M_PER_CLUSTER, CAP = 100, 300

def cluster_based(selected, pool_label, m, cap, rng):
    picked = []
    for c in selected:
        members = np.flatnonzero(pool_label == c)
        take = min(m, len(members))
        picked.append(rng.choice(members, size=take, replace=False))
    picked = np.concatenate(picked) if picked else np.empty(0, int)
    before_cap = len(picked)
    if before_cap > cap:
        picked = rng.choice(picked, size=cap, replace=False)
    return picked, before_cap

cluster_ret = {}
for name in datasets:
    idx, before = cluster_based(sel[name], pool_label, M_PER_CLUSTER, CAP, rng)
    cluster_ret[name] = (idx, before)
    print(f"[{name}] cluster-based: {len(sel[name])}개 클러스터 × ≤{M_PER_CLUSTER} = {before}점 검색 → cap {CAP} 적용 후 {len(idx)}점")
# 출력: [A] cluster-based: 4개 클러스터 × ≤100 = 384점 검색 → cap 300 적용 후 300점
# 출력: [B] cluster-based: 2개 클러스터 × ≤100 = 148점 검색 → cap 300 적용 후 148점

# %% [markdown]
# ## 5. Step 4 — 비교: sample-based $k$-NN 검색 ($k = 4$)
#
# sample-based는 쿼리 하나마다 최근접 $k = 4$장을 가져와 데이터셋을 $k$배로 늘리려는 방식(ImageNet-22k, GLv2처럼 1M 초과 데이터셋용).
# 쿼리가 60개면 상한이 $4 \times 60 = 240$이고, 서로 다른 쿼리가 같은 이웃을 뽑는 **collision**이 있으면 그보다도 적다.
# 반면 cluster-based는 쿼리 수가 아니라 "쿼리가 점유한 개념 영역의 크기"에 비례해 확장되므로 소규모 데이터셋에 유리하다.

# %%
tree = cKDTree(pool)
KNN = 4
knn_ret = {}
for name, q in datasets.items():
    _, nn = tree.query(q, k=KNN)
    flat = nn.ravel()
    uniq = np.unique(flat)
    knn_ret[name] = uniq
    print(f"[{name}] sample-based k={KNN}: {len(q)}×{KNN} = {len(flat)}개 검색, collision 제거 후 unique {len(uniq)}개")
# 출력: [A] sample-based k=4: 60×4 = 240개 검색, collision 제거 후 unique 196개
# 출력: [B] sample-based k=4: 20×4 = 80개 검색, collision 제거 후 unique 58개

# %%
print(f"{'dataset':<8}{'#query':>7}{'sample(k=4)':>13}{'cluster(pre-cap)':>18}{'cluster(post-cap)':>19}{'확장배율(cluster/sample)':>26}")
for name, q in datasets.items():
    s = len(knn_ret[name]); c_idx, c_before = cluster_ret[name]
    print(f"{name:<8}{len(q):>7}{s:>13}{c_before:>18}{len(c_idx):>19}{len(c_idx)/s:>26.2f}")
# 출력: dataset  #query  sample(k=4)  cluster(pre-cap)  cluster(post-cap)   확장배율(cluster/sample)
# 출력: A            60          196               384                300                      1.53
# 출력: B            20           58               148                148                      2.55

# %% [markdown]
# ## 6. 시각화
#
# 왼쪽: 회색 = uncurated pool, 진한 색 = 쿼리(A 파랑, B 주황), 연한 색 = cluster-based로 가져온 점, 초록 테두리 = sample-based(k-NN)로 가져온 점.
# A의 잡음 쿼리 6점 주변에는 연한 점이 없다(">3" 조건으로 배제됨). 오른쪽: 데이터셋별 검색 수 비교.

# %%
C_POOL, C_A, C_B, C_A_LIGHT, C_B_LIGHT, C_KNN = "#c3c2b7", "#2a78d6", "#eb6834", "#86b6ef", "#f5b28f", "#1baf7a"
fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                    subplot_titles=("2-D toy: cluster-based vs sample-based retrieval",
                                    f"검색된 점 수 (cap={CAP})"))
fig.add_trace(go.Scattergl(x=pool[:, 0], y=pool[:, 1], mode="markers", name="uncurated pool",
                           marker=dict(size=3, color=C_POOL, opacity=0.5)), row=1, col=1)
for name, light in (("A", C_A_LIGHT), ("B", C_B_LIGHT)):
    idx = cluster_ret[name][0]
    fig.add_trace(go.Scattergl(x=pool[idx, 0], y=pool[idx, 1], mode="markers", name=f"cluster-based 검색 ({name})",
                               marker=dict(size=5, color=light)), row=1, col=1)
for name, dark in (("A", C_A), ("B", C_B)):
    idx = knn_ret[name]
    fig.add_trace(go.Scattergl(x=pool[idx, 0], y=pool[idx, 1], mode="markers", name=f"sample-based k=4 검색 ({name})",
                               marker=dict(size=7, color="rgba(0,0,0,0)", line=dict(width=1.5, color=C_KNN))), row=1, col=1)
    q = datasets[name]
    fig.add_trace(go.Scattergl(x=q[:, 0], y=q[:, 1], mode="markers", name=f"쿼리 데이터셋 {name} ({len(q)}점)",
                               marker=dict(size=8, color=dark, line=dict(width=1, color="#fcfcfb"))), row=1, col=1)

methods = ["sample-based (k=4)", "cluster-based (pre-cap)", "cluster-based (post-cap)"]
for name, color in (("A", C_A), ("B", C_B)):
    vals = [len(knn_ret[name]), cluster_ret[name][1], len(cluster_ret[name][0])]
    fig.add_trace(go.Bar(x=methods, y=vals, name=f"데이터셋 {name}", marker_color=color,
                         text=vals, textposition="outside"), row=1, col=2)
fig.add_hline(y=CAP, line=dict(color="#898781", dash="dot", width=1), row=1, col=2,
              annotation_text=f"cap = {CAP}", annotation_position="top left")
fig.update_layout(width=1400, height=620, template="plotly_white", barmode="group",
                  paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
                  legend=dict(orientation="h", y=-0.12, x=0), margin=dict(t=60, b=120))
fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=1)
fig.update_yaxes(showgrid=False, zeroline=False, scaleanchor="x", row=1, col=1)
fig.update_yaxes(title_text="검색된 점 수", gridcolor="#e1e0d9", row=1, col=2)
_show(fig)
out_png = os.path.join(HERE, "expy.png")
fig.write_image(out_png, scale=2)
print("saved:", out_png)
# 출력: saved: /home/sungwoo/projects/swcho/dinov2/fm/papers/.fm/hints/b767f878-e13c-445b-95bc-e271edff6ef8/expy.png

# %% [markdown]
# ## 정리
#
# 1. **k-means**: pool 전체를 $K$개 개념 클러스터로 (논문 $K = 100\text{k}$, 분산 구현).
# 2. **선택**: 검색 대상 데이터셋의 이미지가 3장 **넘게** 속한 클러스터만 (잡음 배제).
# 3. **추출**: 선택 클러스터마다 $M$장 (논문 10,000장).
# 4. **cap**: 데이터셋당 최대 1M장으로 잘라 LVD-142M 내 균형 유지.
#
# 쿼리가 수천 장 수준인 fine-grained 데이터셋에서 k-NN($k=4$)은 기껏 수만 장으로만 늘어나지만,
# cluster-based는 개념 영역 단위로 가져와 1M 규모까지 확장할 수 있다 — 그래서 논문은 1M 미만 데이터셋에 cluster-based를 쓴다.
