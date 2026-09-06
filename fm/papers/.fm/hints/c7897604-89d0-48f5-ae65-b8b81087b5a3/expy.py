# %% [markdown]
# # DINOv2 패치 특징 PCA 시각화 — 2단계 PCA 절차를 numpy로 재현
#
# DINOv2 논문(Fig. 1, Fig. 9)의 절차:
#
# 1. 이미지들의 **모든 패치 특징**에 대해 PCA → **첫 주성분(PC1) 점수가 양수인 패치만** 남긴다
#    (전경 / 배경 분리).
# 2. 남은(전경) 패치들에 대해, **같은 카테고리 이미지 3장**을 한꺼번에 **두 번째 PCA** →
#    첫 3개 성분을 각각 **R, G, B 채널**에 매핑해 그린다.
#
# 여기서는 실제 ViT 대신 **합성 패치 특징**(부위 프로토타입 + 잡음)을 만들어
# 같은 절차가 왜 동작하는지 단계별로 확인한다.
#
# 필요 패키지: numpy, plotly, kaleido (검증 환경: `/home/sungwoo/miniforge3/envs/trellis/bin/python`)

# %%
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


rng = np.random.default_rng(0)
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# %% [markdown]
# ## 1. 합성 데이터: 16×16 패치 그리드 3장
#
# 각 패치는 $D=64$차원 특징 벡터를 가진다(실제 DINOv2 ViT-L은 1024차원).
# 패치 종류는 4가지 — **배경(0)**, **머리(1)**, **몸통(2)**, **다리(3)** — 이고
# 각 종류마다 프로토타입 벡터 $\mu_k$가 있다. 패치 특징은
#
# $$ x = \mu_{k} + \epsilon, \qquad \epsilon \sim \mathcal N(0, \sigma^2 I) $$
#
# 핵심 설계: 전경 부위 프로토타입들은 공통 "객체 벡터" $o$를 공유하고
# ($\mu_{\text{part}} = o + \delta_{\text{part}}$), 배경은 $o$와 반대쪽에 있다.
# 즉 **전경 vs 배경의 차이가 부위 간 차이보다 훨씬 크다** — 실제 DINOv2 특징에서
# 관찰되는 성질을 모사한 것이다.

# %%
G, D = 16, 64          # 그리드 크기, 특징 차원
NOISE = 0.45

o = rng.normal(size=D); o = 4.0 * o / np.linalg.norm(o)          # 객체 공통 방향(크게)
deltas = rng.normal(size=(3, D)); deltas = 1.5 * deltas / np.linalg.norm(deltas, axis=1, keepdims=True)
MU = np.zeros((4, D))
MU[0] = -o                       # 배경
MU[1:] = o + deltas              # 머리/몸통/다리 = 공통 객체 벡터 + 부위별 편차

PART_NAMES = ["배경", "머리", "몸통", "다리"]


def make_label_map(x0, y0, w, h, legs_split=True, flip=False):
    """(x0,y0)에서 시작하는 w×h 상자 안에 머리/몸통/다리를 배치한 라벨 맵."""
    lab = np.zeros((G, G), dtype=int)
    head_h = max(1, h // 4); legs_h = max(1, h // 3); body_h = h - head_h - legs_h
    # 머리: 상단, 폭의 절반, 좌/우 치우침(flip)으로 자세 변화
    hx = x0 + (w // 2 if flip else 0)
    lab[y0:y0 + head_h, hx:hx + max(1, w // 2)] = 1
    # 몸통: 가운데, 전체 폭
    lab[y0 + head_h:y0 + head_h + body_h, x0:x0 + w] = 2
    # 다리: 하단, 두 갈래 또는 통짜
    ly = y0 + head_h + body_h
    if legs_split:
        lab[ly:ly + legs_h, x0:x0 + max(1, w // 3)] = 3
        lab[ly:ly + legs_h, x0 + w - max(1, w // 3):x0 + w] = 3
    else:
        lab[ly:ly + legs_h, x0:x0 + w] = 3
    return lab


labels = np.stack([
    make_label_map(2, 2, 8, 12),                         # 이미지 A: 왼쪽 위, 정면
    make_label_map(6, 3, 6, 9, flip=True),               # 이미지 B: 오른쪽, 작고 머리가 오른쪽
    make_label_map(3, 5, 10, 8, legs_split=False),       # 이미지 C: 넓고 낮음, 다리 통짜
])                                                       # (3, G, G)

feats = MU[labels] + NOISE * rng.normal(size=(3, G, G, D))   # (3, G, G, D)
X_all = feats.reshape(-1, D)                                 # (3*G*G, D) = 모든 패치
fg_true = (labels > 0).reshape(-1)
print("패치 총수:", X_all.shape[0], "| 전경 패치 수(정답):", fg_true.sum())
print("이미지별 부위 패치 수:", [np.bincount(l.ravel(), minlength=4).tolist() for l in labels])
# 출력: 패치 총수: 768 | 전경 패치 수(정답): 180
# 출력: 이미지별 부위 패치 수: [[188, 12, 40, 16], [214, 6, 24, 12], [186, 10, 40, 20]]

# %% [markdown]
# ## 2. PCA를 SVD로 구현
#
# 데이터 행렬 $X \in \mathbb R^{N\times D}$를 열 평균으로 중심화한 $X_c$에 대해
#
# $$ X_c = U \Sigma V^\top $$
#
# 를 취하면 $V$의 열이 **주성분 방향**(공분산 행렬의 고유벡터), $\sigma_i^2/(N-1)$이 각 성분의 분산이다.
# $k$번째 주성분 **점수**(score)는 $s_k = X_c v_k$ — 각 패치를 그 방향으로 사영한 값이다.
#
# PC1은 정의상 **데이터 분산이 가장 큰 방향**이다. 패치 집합에서 가장 큰 변이는
# "객체인가 배경인가"이므로 PC1 점수의 부호가 전경/배경을 가른다.

# %%
def pca(X, k):
    """반환: scores (N,k), components (k,D), explained_variance_ratio (k,)"""
    mean = X.mean(0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (X.shape[0] - 1)
    return Xc @ Vt[:k].T, Vt[:k], var[:k] / var.sum()


# %% [markdown]
# ## 3. 1단계 PCA: 모든 패치 → PC1 thresholding → 전경 마스크
#
# 주의: 주성분 벡터 $v$와 $-v$는 동일한 축이므로 **부호는 임의**다. 논문은 "양수인 패치를 남긴다"고
# 했지만 어느 쪽이 객체인지는 관례/확인이 필요하다. 여기서는 *이미지 테두리 패치는 대부분 배경*이라는
# 가정으로 테두리 패치의 평균 점수가 음수가 되도록 부호를 맞춘다.

# %%
s1, comps1, evr1 = pca(X_all, k=3)
pc1 = s1[:, 0]

border = np.zeros((G, G), dtype=bool); border[[0, -1], :] = True; border[:, [0, -1]] = True
border = np.tile(border.reshape(-1), 3)
if pc1[border].mean() > 0:
    pc1 = -pc1

fg_mask = pc1 > 0
acc = (fg_mask == fg_true).mean()
print(f"1단계 PCA 분산 설명 비율(PC1~3): {np.round(evr1, 3)}")
print(f"PC1>0 마스크 vs 정답 전경 일치율: {acc:.3f}")
print(f"PC1 점수 범위 — 배경: [{pc1[~fg_true].min():.1f}, {pc1[~fg_true].max():.1f}], "
      f"전경: [{pc1[fg_true].min():.1f}, {pc1[fg_true].max():.1f}]")
# 출력: 1단계 PCA 분산 설명 비율(PC1~3): [0.466 0.019 0.015]
# 출력: PC1>0 마스크 vs 정답 전경 일치율: 1.000
# 출력: PC1 점수 범위 — 배경: [-3.2, -0.5], 전경: [4.8, 7.1]

# %% [markdown]
# PC1 하나가 전체 분산의 거의 절반(47%)을 설명하고, 배경/전경 점수가 0을 사이에 두고 완전히 갈린다.
#
# 반면 1단계 PCA의 PC2, PC3는 분산 비율이 2% 미만에 불과하다 — 전경/배경 대비가 워낙 커서
# 부위 간 차이가 뒤로 밀려나고 잡음과 섞이기 때문이다. **그래서 배경을 제거한 뒤 PCA를 다시 한다.**

# %% [markdown]
# ## 4. 2단계 PCA: 전경 패치만, 3장을 **한꺼번에** → PC1~3 → RGB
#
# - 배경을 빼면 남은 분산은 **객체 내부의 부위 변이**뿐이므로 상위 성분이 "부위" 축이 된다.
# - 3장의 전경 패치를 **하나의 행렬로 합쳐** PCA하면 세 이미지가 **같은 주성분 좌표계**를 공유한다.
#   따라서 이미지 A의 머리와 이미지 B의 머리는 같은 좌표 → 같은 색이 된다.
#   (이미지마다 따로 PCA하면 축의 순서·부호가 달라져 색이 대응되지 않는다.)
# - 성분 $k$의 점수를 min-max 정규화해 $[0,1]$로 만든 뒤 $k=1,2,3$을 R, G, B에 넣는다.

# %%
X_fg = X_all[fg_mask]
s2, comps2, evr2 = pca(X_fg, k=3)
print(f"2단계 PCA 분산 설명 비율(PC1~3): {np.round(evr2, 3)}")

rgb_fg = (s2 - s2.min(0)) / (s2.max(0) - s2.min(0) + 1e-9)      # (N_fg, 3) in [0,1]
rgb = np.zeros((3 * G * G, 3)); rgb[fg_mask] = rgb_fg            # 배경은 검정
rgb_imgs = (rgb.reshape(3, G, G, 3) * 255).astype(np.uint8)

# 같은 부위 → 같은 색인지 수치로 확인: 부위별 평균 RGB(이미지별)
lab_flat = labels.reshape(3, -1)
for name, k in zip(PART_NAMES[1:], [1, 2, 3]):
    means = [rgb.reshape(3, -1, 3)[i][lab_flat[i] == k].mean(0).round(2).tolist() for i in range(3)]
    print(f"{name} 평균 RGB  A={means[0]}  B={means[1]}  C={means[2]}")
# 출력: 2단계 PCA 분산 설명 비율(PC1~3): [0.089 0.057 0.034]
# 출력: 머리 평균 RGB  A=[0.65, 0.22, 0.47]  B=[0.64, 0.16, 0.47]  C=[0.69, 0.13, 0.47]
# 출력: 몸통 평균 RGB  A=[0.28, 0.58, 0.41]  B=[0.32, 0.63, 0.49]  C=[0.3, 0.58, 0.5]
# 출력: 다리 평균 RGB  A=[0.75, 0.7, 0.47]  B=[0.79, 0.81, 0.44]  C=[0.75, 0.69, 0.41]

# %% [markdown]
# 같은 부위의 평균 RGB가 세 이미지에서 거의 동일하다(머리≈자홍, 몸통≈녹색, 다리≈노랑) — 위치·크기·자세가
# 달라도 **색이 곧 부위 라벨**로 기능한다. 부위가 3종이므로 PC1, PC2 두 축만으로 충분히 분리되고
# PC3(3.4%)는 잡음에 가깝다. 배경을 제거했기 때문에 2단계에서는 부위 축이 상위 성분으로 올라왔다
# (1단계에서는 부위 변이가 PC2·PC3 자리에도 못 올라오고 잡음에 묻혀 있었다). 실제 DINOv2에서는 부위가 연속적으로 변해 색이 부드럽게 이어진다.

# %% [markdown]
# ## 5. 시각화: 정답 부위 / 1단계 PC1 점수(마스크) / 2단계 RGB

# %%
titles = []
for row in ["정답 부위 라벨", "1단계 PC1 점수 (>0 = 전경)", "2단계 PCA → RGB"]:
    titles += [f"{row} — 이미지 {c}" for c in "ABC"]
fig = make_subplots(rows=3, cols=3, subplot_titles=titles, horizontal_spacing=0.04, vertical_spacing=0.08)

# 라벨 0..3 을 zmin=-0.5, zmax=3.5 로 그리면 각 라벨은 [0,.25),[.25,.5),[.5,.75),[.75,1] 구간에 놓인다
part_colors = [[0, "#111111"], [0.25, "#111111"], [0.25, "#e76f51"], [0.5, "#e76f51"],
               [0.5, "#2a9d8f"], [0.75, "#2a9d8f"], [0.75, "#e9c46a"], [1, "#e9c46a"]]
pc1_imgs = pc1.reshape(3, G, G)
vmax = np.abs(pc1).max()
for c in range(3):
    fig.add_trace(go.Heatmap(z=labels[c], zmin=-0.5, zmax=3.5, colorscale=part_colors, showscale=False,
                             hovertemplate="부위=%{z}<extra></extra>"), row=1, col=c + 1)
    fig.add_trace(go.Heatmap(z=pc1_imgs[c], zmin=-vmax, zmax=vmax, colorscale="RdBu", reversescale=True,
                             showscale=(c == 2), colorbar=dict(title="PC1", len=0.28, y=0.5),
                             hovertemplate="PC1=%{z:.2f}<extra></extra>"), row=2, col=c + 1)
    fig.add_trace(go.Image(z=rgb_imgs[c], hovertemplate="RGB=%{color}<extra></extra>"), row=3, col=c + 1)

for r in range(1, 4):
    for c in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=r, col=c)
        fig.update_yaxes(showticklabels=False, autorange="reversed" if r < 3 else None,
                         scaleanchor=f"x{(r - 1) * 3 + c if (r - 1) * 3 + c > 1 else ''}", row=r, col=c)

fig.update_layout(
    title="2단계 PCA 재현: PC1 thresholding으로 배경 제거 → 전경만 3장 공동 PCA → PC1~3 = RGB",
    width=1000, height=1000, template="plotly_white", font=dict(size=11),
)
fig.update_annotations(font_size=11)
_show(fig)
out_png = os.path.join(HERE, "expy.png")
fig.write_image(out_png, scale=2)
print("저장:", out_png)
# 출력: 저장: /home/sungwoo/projects/swcho/dinov2/fm/papers/.fm/hints/c7897604-89d0-48f5-ae65-b8b81087b5a3/expy.png

# %% [markdown]
# ## 6. 정리
#
# | 단계 | 입력 | 하는 일 | 왜 |
# |---|---|---|---|
# | PCA ① | 모든 패치 (3장) | PC1 점수 부호로 threshold | 최대 분산 방향 = 객체 vs 배경 대비 |
# | 마스크 | PC1 > 0 | 전경 패치만 남김 | 배경 분산을 제거해 부위 변이가 상위 성분으로 올라오게 |
# | PCA ② | 전경 패치 (3장 합쳐서) | PC1~3 추출 | 3장이 같은 좌표계 → 같은 부위 = 같은 색 |
# | 색 | PC1,2,3 → min-max → R,G,B | 시각화 | 3차원 임베딩을 사람이 볼 수 있게 |
