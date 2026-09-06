# %% [markdown]
# # DINOv2 patch matching 파이프라인 toy 재현
#
# 논문(§7.5 "Patch matching")의 4단계를 합성 데이터로 그대로 밟아 본다.
#
# 1. **전경 검출** — 패치 특징 PCA의 첫 성분(PC1) 부호로 전경/배경 분리
# 2. **유클리드 거리 행렬** — 이미지 A, B의 전경 패치 특징 사이 $d_{ij}=\lVert f_i^A - f_j^B\rVert_2$
# 3. **assignment problem** — 헝가리안 알고리즘(`linear_sum_assignment`)으로 일대일 매칭.
#    최근접 이웃(NN) 매칭과 비교해 "다대일 충돌"이 사라지는 것을 확인
# 4. **non-maximum suppression** — 거리가 작은(강한) 매칭부터 남기고, 반경 $r$ 안의 이웃 매칭은 억제
#
# 합성 데이터: 12×12 패치 격자 두 장. 객체는 서로 다른 위치·모양으로 놓이고,
# 객체 패치의 $D$차원 특징은 **4개 part 프로토타입 + 노이즈**, 배경은 **별도 프로토타입 + 노이즈**.
#
# 필요 패키지: numpy, scipy, plotly, kaleido
# (실행 환경: /home/sungwoo/miniforge3/envs/trellis/bin/python — 기본 python3에는 numpy가 없음)

# %%
import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import plotly.graph_objects as go


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


rng = np.random.default_rng(0)
G = 12          # 격자 한 변 패치 수
D = 16          # 특징 차원
N_PARTS = 4     # 객체 part 수 (예: 머리/몸통/앞다리/뒷다리)
SIGMA = 0.35    # 패치 특징 노이즈 표준편차

# %% [markdown]
# ## 0. 합성 이미지 A, B 생성
#
# 각 패치에 **part id**를 부여한다 (0 = 배경, 1..4 = 객체 part).
# 객체의 바운딩박스를 4분면으로 나눠 part 1~4로 라벨링하므로,
# A와 B에서 객체가 다른 위치/크기에 있어도 "같은 part"는 같은 의미를 가진다
# (논문에서 비행기 날개 ↔ 새 날개가 매칭되는 상황의 축소판).

# %%
def make_part_map(r0, r1, c0, c1):
    """(r0..r1, c0..c1) 범위를 객체로 두고 4분면 part id를 채운 G×G 맵을 반환."""
    m = np.zeros((G, G), dtype=int)
    rm, cm = (r0 + r1) // 2, (c0 + c1) // 2
    for r in range(r0, r1):
        for c in range(c0, c1):
            m[r, c] = 1 + (2 if r >= rm else 0) + (1 if c >= cm else 0)
    return m


part_A = make_part_map(2, 10, 1, 7)    # 객체가 왼쪽에 세로로 긴 모양
part_B = make_part_map(4, 10, 4, 12)   # 객체가 오른쪽 아래, 가로로 넓은 모양

# 프로토타입: 배경 1개 + part 4개. 서로 충분히 떨어진 랜덤 벡터.
protos = rng.normal(size=(N_PARTS + 1, D))
protos /= np.linalg.norm(protos, axis=1, keepdims=True)
protos *= 3.0
protos[0] *= 1.5  # 배경 프로토타입은 조금 더 멀리 (PC1이 전경/배경 축이 되게)


def make_features(part_map):
    ids = part_map.ravel()
    return protos[ids] + SIGMA * rng.normal(size=(G * G, D))


feat_A = make_features(part_A)   # (144, D)
feat_B = make_features(part_B)
coords = np.array([(r, c) for r in range(G) for c in range(G)])  # 패치 인덱스 → (row, col)

print("A 객체 패치 수:", (part_A > 0).sum(), "/ B 객체 패치 수:", (part_B > 0).sum())
print("A part 분포:", np.bincount(part_A.ravel(), minlength=5))
print("B part 분포:", np.bincount(part_B.ravel(), minlength=5))
# 출력: A 객체 패치 수: 48 / B 객체 패치 수: 48
# 출력: A part 분포: [96 12 12 12 12]
# 출력: B part 분포: [96 12 12 12 12]

# %% [markdown]
# ## 1. PCA PC1 부호로 전경 검출
#
# 논문: "We keep only patches with a positive value after we threshold the first component."
#
# 두 이미지의 패치를 한데 모아 중심화하고 SVD로 첫 주성분 $u_1$을 구한 뒤
# 각 패치의 사영 $s_i = (f_i-\bar f)\cdot u_1$ 부호로 전경/배경을 나눈다.
# 배경 vs. 객체가 가장 큰 분산 방향이므로 PC1이 그 축을 잡는다.
# (PC 부호는 임의이므로, 다수인 배경이 음수가 되도록 부호를 맞춘다.)

# %%
feats = np.vstack([feat_A, feat_B])
mean = feats.mean(axis=0)
_, S, Vt = np.linalg.svd(feats - mean, full_matrices=False)
pc1 = Vt[0]
score = (feats - mean) @ pc1
if np.median(score) > 0:          # 배경(다수)이 음수가 되도록 부호 정렬
    score = -score
fg_mask = score > 0
fg_A, fg_B = fg_mask[:G * G], fg_mask[G * G:]

explained = S**2 / (S**2).sum()
print(f"PC1 설명 분산 비율: {explained[0]:.3f}, PC2: {explained[1]:.3f}")
print("전경 검출 정확도 A: %.3f  B: %.3f" % (
    (fg_A == (part_A.ravel() > 0)).mean(), (fg_B == (part_B.ravel() > 0)).mean()))
print("검출된 전경 패치 수 A:", fg_A.sum(), " B:", fg_B.sum())
# 출력: PC1 설명 분산 비율: 0.547, PC2: 0.129
# 출력: 전경 검출 정확도 A: 1.000  B: 1.000
# 출력: 검출된 전경 패치 수 A: 48  B: 48

# %% [markdown]
# ## 2. 전경 패치 간 유클리드 거리 행렬
#
# 배경을 제거했으므로 거리 행렬은 $n_A \times n_B$ (여기선 48×48).
# 배경을 빼지 않으면 96개의 배경 패치끼리 "거리 0에 가까운" 가짜 매칭이 대량 생겨
# 전경의 의미 있는 매칭을 가려 버린다.

# %%
idx_A = np.flatnonzero(fg_A)
idx_B = np.flatnonzero(fg_B)
FA, FB = feat_A[idx_A], feat_B[idx_B]
Dmat = cdist(FA, FB, metric="euclidean")   # (nA, nB)
pa = part_A.ravel()[idx_A]                 # 각 전경 패치의 정답 part id
pb = part_B.ravel()[idx_B]
same = pa[:, None] == pb[None, :]
print("거리 행렬 shape:", Dmat.shape)
print(f"같은 part 쌍 평균 거리: {Dmat[same].mean():.2f} / 다른 part 쌍 평균 거리: {Dmat[~same].mean():.2f}")
# 출력: 거리 행렬 shape: (48, 48)
# 출력: 같은 part 쌍 평균 거리: 1.92 / 다른 part 쌍 평균 거리: 4.77

# %% [markdown]
# ## 3. Assignment problem vs. 최근접 이웃(NN)
#
# **NN**: 각 A 패치가 독립적으로 가장 가까운 B 패치를 고른다 → 여러 A 패치가 같은 B 패치로 몰리는
# **다대일 충돌**이 생긴다.
#
# **Assignment (헝가리안 / 선형 할당)**: 순열 $\pi$ 를 골라 총 비용을 최소화한다.
# $$\min_{\pi}\ \sum_i d_{i,\pi(i)} \quad \text{s.t. } \pi \text{는 일대일}$$
# 한 B 패치는 최대 한 번만 쓰이므로, 전체를 보며 "누가 어디로 가야 총합이 작아지는가"를 결정한다.
# `scipy.optimize.linear_sum_assignment` 가 $O(n^3)$ 헝가리안 계열 알고리즘으로 정확히 푼다.

# %%
# --- 최근접 이웃 (독립적 argmin) ---
nn_j = Dmat.argmin(axis=1)
nn_unique = len(np.unique(nn_j))
nn_acc = (pa == pb[nn_j]).mean()
print(f"[NN]        매칭 수 {len(nn_j)}, 서로 다른 B 타깃 수 {nn_unique} "
      f"→ {len(nn_j) - nn_unique}개의 A 패치가 다른 A 패치와 B 타깃을 공유(충돌)")
print(f"[NN]        part 일치율 {nn_acc:.3f}, 총 거리 {Dmat[np.arange(len(nn_j)), nn_j].sum():.2f}")

# --- 헝가리안 (일대일) ---
row, col = linear_sum_assignment(Dmat)
hung_acc = (pa[row] == pb[col]).mean()
print(f"[Hungarian] 매칭 수 {len(row)}, 서로 다른 B 타깃 수 {len(np.unique(col))} (일대일)")
print(f"[Hungarian] part 일치율 {hung_acc:.3f}, 총 거리 {Dmat[row, col].sum():.2f}")
# 출력: [NN]        매칭 수 48, 서로 다른 B 타깃 수 23 → 25개의 A 패치가 다른 A 패치와 B 타깃을 공유(충돌)
# 출력: [NN]        part 일치율 1.000, 총 거리 68.64
# 출력: [Hungarian] 매칭 수 48, 서로 다른 B 타깃 수 48 (일대일)
# 출력: [Hungarian] part 일치율 1.000, 총 거리 75.00

# %% [markdown]
# NN의 총 거리가 더 작은 것은 당연하다 — 제약이 없으니 각자 최솟값을 고른다.
# 대신 B의 48개 패치 중 25개는 아무에게도 선택되지 않고, 어떤 B 패치는 여럿에게 중복 선택된다.
# 헝가리안은 약간의 총 거리를 희생해 **모든 패치를 빠짐없이, 겹침 없이** 대응시킨다.
# 이것이 그림에서 "한 점에서 여러 선이 뻗어나오는" 지저분한 매칭을 원천 차단하는 이유다.

# %% [markdown]
# ## 4. Non-maximum suppression (NMS)
#
# 48개의 일대일 매칭을 전부 그리면 선이 너무 많다. 논문은 "salient한 것만 남기기 위해" NMS를 적용한다.
#
# 여기서의 NMS는 검출기의 박스 NMS와 같은 논리다:
# 1. 매칭을 **거리 오름차순**(= 매칭 강도 내림차순)으로 정렬
# 2. 앞에서부터 하나씩 보며, **이미 남긴 매칭 중 A 격자에서 반경 $r$ 이내**에 있는 것이 있으면 억제
# 3. 없으면 채택
#
# 결과: 공간적으로 겹치는 매칭 묶음마다 가장 강한 하나만 살아남아, 객체 위에 대략 $r$ 간격으로 퍼진
# 대표 매칭들만 남는다.

# %%
def nms_matches(rows, cols, dist, coords_A, radius):
    order = np.argsort(dist[rows, cols])          # 거리 작은(강한) 매칭부터
    kept = []
    for k in order:
        p = coords_A[rows[k]]
        if all(np.linalg.norm(p - coords_A[rows[m]]) > radius for m in kept):
            kept.append(k)
    return np.array(kept)


R = 2.0
kept = nms_matches(row, col, Dmat, coords[idx_A], radius=R)
kept_rows, kept_cols = row[kept], col[kept]
print(f"NMS(r={R}) 후 매칭 수: {len(kept)} / {len(row)}")
print(f"남은 매칭 part 일치율: {(pa[kept_rows] == pb[kept_cols]).mean():.3f}")
print("남은 매칭 (A(row,col) -> B(row,col), part, 거리):")
for k in kept[np.argsort(Dmat[kept_rows, kept_cols])]:
    a, b = coords[idx_A[row[k]]], coords[idx_B[col[k]]]
    print(f"  A{tuple(a)} -> B{tuple(b)}  part {pa[row[k]]}  d={Dmat[row[k], col[k]]:.2f}")
# 출력: NMS(r=2.0) 후 매칭 수: 10 / 48
# 출력: 남은 매칭 part 일치율: 1.000
# 출력: 남은 매칭 (A(row,col) -> B(row,col), part, 거리):
# 출력:   A(2, 2) -> B(4, 5)  part 1  d=0.82
# 출력:   A(7, 5) -> B(8, 8)  part 4  d=0.91
# 출력:   A(3, 4) -> B(5, 11)  part 2  d=1.06
# 출력:   A(4, 1) -> B(4, 4)  part 1  d=1.14
# 출력:   A(2, 6) -> B(4, 9)  part 2  d=1.32
# 출력:   A(9, 2) -> B(8, 4)  part 3  d=1.40
# 출력:   A(6, 3) -> B(7, 7)  part 3  d=1.55
# 출력:   A(5, 6) -> B(4, 8)  part 2  d=1.64
# 출력:   A(7, 1) -> B(8, 7)  part 3  d=1.78
# 출력:   A(9, 6) -> B(8, 9)  part 4  d=2.07

# %% [markdown]
# ## 5. 시각화
#
# 왼쪽이 이미지 A, 오른쪽이 이미지 B. 사각형 색 = part id(회색 = 배경, PCA로 걸러진 패치),
# 선 = NMS 후 남은 매칭. 선이 **같은 색 영역끼리** 이어지는지(=part 일치) 확인한다.
# 논문 Figure 10에서 날개↔날개, 코↔코 선이 그려지는 것과 같은 구조다.

# %%
PART_COLORS = {0: "#d9d9d9", 1: "#4c72b0", 2: "#dd8452", 3: "#55a868", 4: "#c44e52"}
PART_NAMES = {0: "배경", 1: "part 1 (좌상)", 2: "part 2 (우상)", 3: "part 3 (좌하)", 4: "part 4 (우하)"}
OFFSET = G + 3   # B 격자를 오른쪽으로 밀어 놓는 x 오프셋


def grid_traces(part_map, fg, x_off, name_suffix):
    traces = []
    for pid in range(N_PARTS + 1):
        sel = np.flatnonzero(part_map.ravel() == pid)
        traces.append(go.Scatter(
            x=coords[sel, 1] + x_off, y=coords[sel, 0],
            mode="markers",
            marker=dict(symbol="square", size=22, color=PART_COLORS[pid],
                        line=dict(width=1, color="white")),
            name=PART_NAMES[pid], legendgroup=str(pid), showlegend=(x_off == 0),
            hovertext=[f"{name_suffix} ({r},{c}) part={pid} fg={fg[i]}"
                       for i, (r, c) in zip(sel, coords[sel])], hoverinfo="text"))
    return traces


fig = go.Figure()
fig.add_traces(grid_traces(part_A, fg_A, 0, "A"))
fig.add_traces(grid_traces(part_B, fg_B, OFFSET, "B"))
for k in kept:
    a, b = coords[idx_A[row[k]]], coords[idx_B[col[k]]]
    fig.add_trace(go.Scatter(
        x=[a[1], b[1] + OFFSET], y=[a[0], b[0]], mode="lines",
        line=dict(width=2.5, color=PART_COLORS[pa[row[k]]]),
        showlegend=False, hoverinfo="skip"))
fig.update_layout(
    title=f"PCA 전경 검출 → 유클리드 거리 → 헝가리안 일대일 매칭 → NMS(r={R}): "
          f"{len(kept)}/{len(row)} 매칭, part 일치율 {(pa[kept_rows] == pb[kept_cols]).mean():.0%}",
    xaxis=dict(visible=False, range=[-1, OFFSET + G]),
    yaxis=dict(visible=False, autorange="reversed", scaleanchor="x"),
    width=1000, height=520, template="plotly_white",
    legend=dict(orientation="h", y=-0.02),
    annotations=[
        dict(x=(G - 1) / 2, y=-0.9, text="이미지 A", showarrow=False, font=dict(size=15)),
        dict(x=OFFSET + (G - 1) / 2, y=-0.9, text="이미지 B", showarrow=False, font=dict(size=15)),
    ],
)
_show(fig)

out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".",
                       "expy.png")
fig.write_image(out_png, scale=2)
print("saved:", out_png)
# 출력: saved: <hint dir>/expy.png
