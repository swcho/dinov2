# %% [markdown]
# # Self-deduplication 재현 실험: k-NN 그래프 → 임계값 → connected components
#
# DINOv2(Oquab et al., 2023) Appendix A.3의 self-deduplication 절차를 장난감 데이터로 재현한다.
#
# 1. 각 이미지 임베딩(여기서는 합성 벡터)에 대해 코사인 유사도로 $k=64$ 최근접 이웃을 검색
# 2. 유사도 $>0.6$ 인 이웃 간선만 남김
# 3. k-NN 그래프의 connected component(연결 요소)를 추출 → component마다 대표 1장만 유지
# 4. 임계값을 0.3…0.9로 바꾸며 "왜 0.6인가"를 살펴보고, 전이적(transitive) 병합도 확인
#
# 코사인 유사도는 논문 A.2의 정의를 그대로 쓴다:
# $$m(s,r)=\frac{f(s)\cdot f(r)}{\|f(s)\|_2\,\|f(r)\|_2}$$
#
# 필요 패키지: numpy, scipy, plotly, kaleido
# 실행: /home/sungwoo/miniforge3/envs/trellis/bin/python expy.py  (기본 python3에는 numpy가 없음)

# %%
import os
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
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

# %% [markdown]
# ## 0. 합성 "임베딩" 생성
#
# 실제 논문은 SSCD(Pizzi et al., 2022) 임베딩을 쓴다. 여기서는 D=32 단위 벡터로 흉내 낸다.
# - 중복 그룹(ground truth): 그룹 크기 1~8, 총 N≈400
# - 그룹마다 기준 벡터 $b$를 뽑고, 멤버는 $\mathrm{normalize}(b+\varepsilon\,\hat g)$ ($\hat g$: 무작위 단위 방향)
# - 두 멤버의 코사인 유사도는 대략 $1/(1+\varepsilon^2)$. $\varepsilon\in[0.2,0.9]$ 이면 유사도 ≈ 0.96 … 0.55
#   → 일부 "느슨한" 그룹은 임계값 0.6 근처에 걸치게 된다(현실의 강한 편집본 near-duplicate에 해당).

# %%
D = 32
N_TARGET = 400


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


sizes, labels_gt, X = [], [], []
gid = 0
while sum(sizes) < N_TARGET:
    s = int(rng.integers(1, 9))  # 그룹 크기 1..8
    eps = rng.uniform(0.2, 0.9)  # 그룹 내 흩어짐 정도
    base = unit(rng.normal(size=D))
    g = unit(rng.normal(size=(s, D)))
    members = unit(base[None, :] + eps * g)
    X.append(members)
    labels_gt += [gid] * s
    sizes.append(s)
    gid += 1

X = np.vstack(X)
labels_gt = np.array(labels_gt)
N, G = X.shape[0], len(sizes)
print(f"N = {N} 개 벡터, ground-truth 그룹 수 G = {G}, 그룹 크기 분포 = {np.bincount(sizes)[1:]}")
# 출력: N = 401 개 벡터, ground-truth 그룹 수 G = 85, 그룹 크기 분포 = [10  8  8 11 11 15 14  8]

# %% [markdown]
# ## 1. 브루트포스 코사인 k-NN ($k=64$)
#
# 단위 벡터이므로 내적이 곧 코사인 유사도. $S = XX^\top$ 를 계산해 각 행에서 자기 자신을 제외한 상위 $k$개를 고른다.
# 논문에서는 13억 장이라 이 $N\times N$ 행렬을 만들 수 없고, Faiss의 GPU IVF-PQ 인덱스로 근사 k-NN을 배치 검색한다.

# %%
K = 64
S = X @ X.T
np.fill_diagonal(S, -np.inf)  # 자기 자신 제외
nn_idx = np.argsort(-S, axis=1)[:, :K]  # (N, K) 이웃 인덱스
nn_sim = np.take_along_axis(S, nn_idx, axis=1)  # (N, K) 이웃 유사도

rows = np.repeat(np.arange(N), K)
cols = nn_idx.ravel()
sims = nn_sim.ravel()
same_group = labels_gt[rows] == labels_gt[cols]
print(f"k-NN 간선 수 = {len(sims)} (= N*k), 그 중 같은 그룹 간선 = {same_group.sum()}")
print(f"같은 그룹 간선 유사도: 중앙값 {np.median(sims[same_group]):.3f}, 최소 {sims[same_group].min():.3f}")
print(f"다른 그룹 간선 유사도: 중앙값 {np.median(sims[~same_group]):.3f}, 최대 {sims[~same_group].max():.3f}")
# 출력: k-NN 간선 수 = 25664 (= N*k), 그 중 같은 그룹 간선 = 1902
# 출력: 같은 그룹 간선 유사도: 중앙값 0.774, 최소 0.284
# 출력: 다른 그룹 간선 유사도: 중앙값 0.250, 최대 0.657

# %% [markdown]
# ## 2~3. 임계값 $>0.6$ 으로 간선 필터 → 희소 인접행렬 → connected components
#
# 논문은 "scalable disjoint set(union-find)"으로 component를 구했다. 여기서는 동일한 결과를 주는
# `scipy.sparse.csgraph.connected_components` 를 쓴다. 각 component에서 **대표 1장**(가장 작은 인덱스)만 남긴다.


# %%
def dedup(threshold, rows=rows, cols=cols, sims=sims, n=N):
    keep = sims > threshold
    A = coo_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(n, n))
    n_comp, comp = connected_components(A, directed=False)  # directed=False → A+Aᵀ 로 대칭화
    # 대표 1장: component별 첫 인덱스
    _, rep = np.unique(comp, return_index=True)
    return n_comp, comp, rep


THR = 0.6
n_comp, comp, rep = dedup(THR)
print(f"임계값 {THR}: {N} 장 → {n_comp} 장 (제거 {N - n_comp} 장, {100 * (N - n_comp) / N:.1f}%)")
print(f"ground-truth 그룹 수 = {G}")

# component 품질: 한 component에 서로 다른 GT 그룹이 섞였는지(과병합), 한 GT 그룹이 여러 component로 갈라졌는지(과분할)
mixed = sum(len(set(labels_gt[comp == c])) > 1 for c in range(n_comp))
split = sum(len(set(comp[labels_gt == g])) > 1 for g in range(G))
print(f"여러 GT 그룹이 섞인 component = {mixed} 개, 여러 component로 갈라진 GT 그룹 = {split} 개")
# 출력: 임계값 0.6: 401 장 → 97 장 (제거 304 장, 75.8%)
# 출력: ground-truth 그룹 수 = 85
# 출력: 여러 GT 그룹이 섞인 component = 3 개, 여러 component로 갈라진 GT 그룹 = 8 개

# %% [markdown]
# 논문 규모로 환산하면 13억 → 11억(약 15% 제거)이었다. 웹 크롤 데이터는 대부분 unique이고 소수만 중복이므로
# 제거율이 낮다. 위 장난감 데이터는 의도적으로 중복을 많이 넣어 제거율이 높다.
#
# ## 4. 전이적 병합: A–B, B–C 는 연결되지만 A–C 는 임계값 미만
#
# 쌍별 비교(pairwise)라면 A와 C는 "다른 이미지"다. 하지만 connected component는 경로만 있으면 한 묶음으로 본다.
# 2차원 벡터로 각도 0°, 46°, 92° 를 잡으면 $\cos46^\circ\approx0.695>0.6$, $\cos92^\circ\approx-0.035<0.6$.

# %%
ang = np.deg2rad([0.0, 46.0, 92.0])
ABC = np.stack([np.cos(ang), np.sin(ang)], axis=1)
S3 = ABC @ ABC.T
print("sim(A,B) = %.3f, sim(B,C) = %.3f, sim(A,C) = %.3f" % (S3[0, 1], S3[1, 2], S3[0, 2]))
A3 = coo_matrix((S3 > 0.6).astype(float) - np.eye(3))
n3, c3 = connected_components(A3, directed=False)
print(f"쌍별 판정: A~C 중복? {S3[0, 2] > 0.6}   |   component 수 = {n3}, 라벨 = {c3.tolist()} → A,B,C 한 묶음")
# 출력: sim(A,B) = 0.695, sim(B,C) = 0.695, sim(A,C) = -0.035
# 출력: 쌍별 판정: A~C 중복? False   |   component 수 = 1, 라벨 = [0, 0, 0] → A,B,C 한 묶음

# %% [markdown]
# 합성 데이터에서도 같은 일이 일어난다: component 안에서 실제로 유사도 $>0.6$ 인 쌍의 비율을 보면 100%가 아니다.

# %%
S_sym = X @ X.T
big = [c for c in range(n_comp) if (comp == c).sum() >= 3]
pairs_total = pairs_direct = 0
for c in big:
    idx = np.where(comp == c)[0]
    sub = S_sym[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), 1)
    pairs_total += len(iu[0])
    pairs_direct += (sub[iu] > THR).sum()
print(f"크기≥3 component 내부 쌍 {pairs_total}개 중 직접 유사도>0.6 인 쌍 = {pairs_direct}개 "
      f"({100 * pairs_direct / pairs_total:.1f}%) → 나머지는 전이적으로 묶임")
# 출력: 크기≥3 component 내부 쌍 947개 중 직접 유사도>0.6 인 쌍 = 787개 (83.1%) → 나머지는 전이적으로 묶임

# %% [markdown]
# ## 5. 임계값 sweep: 왜 0.6인가
#
# 임계값 $t$ 를 바꾸며
# - dedup 후 남는 장수(= component 수)와 GT 그룹 수 비교
# - k-NN 간선 중 $>t$ 로 살아남은 간선의 정밀도(같은 그룹 비율)/재현율(같은 그룹 간선 중 살아남은 비율)
#
# 임계값이 낮으면 다른 그룹 간선이 새어 들어와 component가 과병합되고(정밀도↓),
# 높으면 느슨한 그룹이 갈라진다(재현율↓). 0.6은 이 사이의 타협점이다.
# 논문의 relative dedup은 더 엄격한 0.45를 쓴다 — 평가셋 오염은 놓치는 것이 더 큰 문제이므로 재현율을 우선한 것이다.

# %%
ths = np.round(np.arange(0.30, 0.91, 0.05), 2)
n_comps, precs, recs = [], [], []
for t in ths:
    nc, _, _ = dedup(t)
    keep = sims > t
    tp = (keep & same_group).sum()
    precs.append(tp / max(keep.sum(), 1))
    recs.append(tp / same_group.sum())
    n_comps.append(nc)
print(" t    #comp  prec   rec")
for t, nc, p, r in zip(ths, n_comps, precs, recs):
    print(f"{t:.2f}  {nc:5d}  {p:.3f}  {r:.3f}")
# 출력:  t    #comp  prec   rec
# 출력: 0.30      1  0.226  0.999
# 출력: 0.35      1  0.369  0.999
# 출력: 0.40      1  0.575  0.991
# 출력: 0.45      9  0.799  0.976
# 출력: 0.50     36  0.921  0.941
# 출력: 0.55     77  0.979  0.900
# 출력: 0.60     97  0.996  0.837
# 출력: 0.65    117  0.999  0.755
# 출력: 0.70    150  1.000  0.664
# 출력: 0.75    187  1.000  0.553
# 출력: 0.80    235  1.000  0.442
# 출력: 0.85    271  1.000  0.323
# 출력: 0.90    324  1.000  0.181

# %% [markdown]
# ## 6. 시각화
#
# 왼쪽: k-NN 간선 유사도 분포(같은 그룹 vs 다른 그룹)와 임계값 0.6 / 0.45 선.
# 가운데: 임계값별 dedup 후 장수(점선 = GT 그룹 수). 오른쪽: 간선 정밀도/재현율.

# %%
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
fig = make_subplots(
    rows=1, cols=3, horizontal_spacing=0.07,
    subplot_titles=("k-NN 간선 유사도 분포 (k=64)", "임계값별 dedup 후 장수", "임계값별 간선 정밀도 / 재현율"),
)
bins = dict(start=-0.4, end=1.0, size=0.02)
fig.add_trace(go.Histogram(x=sims[same_group], name="같은 그룹 간선", marker_color=BLUE, xbins=bins,
                           opacity=0.85, marker_line_width=0), row=1, col=1)
fig.add_trace(go.Histogram(x=sims[~same_group], name="다른 그룹 간선", marker_color=ORANGE, xbins=bins,
                           opacity=0.85, marker_line_width=0), row=1, col=1)
fig.update_layout(barmode="overlay")
fig.update_yaxes(type="log", title_text="간선 수 (log)", row=1, col=1)
fig.update_xaxes(title_text="코사인 유사도", row=1, col=1)
for t, lab, pos in [(0.6, "self 0.6", "top right"), (0.45, "relative 0.45", "top left")]:
    fig.add_vline(x=t, line_dash="dash", line_color=INK, line_width=1.5, row=1, col=1,
                  annotation_text=lab, annotation_position=pos, annotation_font_color=INK,
                  annotation_yshift=-18)

fig.add_trace(go.Scatter(x=ths, y=n_comps, mode="lines+markers", name="dedup 후 장수 (#component)",
                         line=dict(color=VIOLET, width=2), marker=dict(size=8)), row=1, col=2)
fig.add_hline(y=G, line_dash="dot", line_color=MUTED, row=1, col=2,
              annotation_text=f"GT 그룹 수 = {G}", annotation_position="bottom right", annotation_font_color=MUTED)
fig.add_vline(x=0.6, line_dash="dash", line_color=INK, line_width=1.5, row=1, col=2)
fig.update_xaxes(title_text="임계값 t", row=1, col=2)
fig.update_yaxes(title_text="장수", row=1, col=2)

fig.add_trace(go.Scatter(x=ths, y=precs, mode="lines+markers", name="정밀도 (살아남은 간선 중 같은 그룹)",
                         line=dict(color=AQUA, width=2), marker=dict(size=8)), row=1, col=3)
fig.add_trace(go.Scatter(x=ths, y=recs, mode="lines+markers", name="재현율 (같은 그룹 간선 중 생존)",
                         line=dict(color=YELLOW, width=2), marker=dict(size=8)), row=1, col=3)
fig.add_vline(x=0.6, line_dash="dash", line_color=INK, line_width=1.5, row=1, col=3)
fig.update_xaxes(title_text="임계값 t", row=1, col=3)
fig.update_yaxes(title_text="비율", range=[0.3, 1.02], row=1, col=3)

fig.update_layout(
    title=dict(text=f"Self-deduplication 장난감 재현: N={N}, k={K}, 임계값 0.6 → {n_comp}장", font=dict(color=INK)),
    template="plotly_white", paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=MUTED, size=12),
    legend=dict(orientation="h", y=-0.22, x=0), width=1400, height=520, margin=dict(t=90, b=110),
)
fig.update_xaxes(showgrid=False, zeroline=False)
fig.update_yaxes(gridcolor="#e6e5e1", zeroline=False)
_show(fig)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "expy.png")
fig.write_image(out, scale=2)
print("saved:", out)
# 출력: saved: <hint dir>/expy.png
