# %% [markdown]
# # high-norm token은 어디서 생기는가 — "이웃 4개와의 코사인 유사도" 재현
#
# *Vision Transformers Need Registers* (arXiv:2309.16588) §2는 high-norm(outlier) token이
# **patch embedding 직후 기준으로 상하좌우 이웃 4개와 코사인 유사도가 1에 가까운 patch**,
# 즉 **중복된 정보만 담은 patch**에서 나타난다는 것을 Fig. 5a로 보인다.
#
# 여기서는 DINOv2 가중치 없이(네트워크/시간 비용 회피) **원본 픽셀 패치 수준**에서
# 같은 측정을 재현한다. patch embedding은 결국 patch 픽셀에 대한 선형 사상이므로,
# "이웃과 거의 같은 patch"라는 성질은 픽셀 단계에서 이미 드러난다.
#
# 필요 패키지: numpy, pillow, plotly, kaleido
# 실행 인터프리터: `/home/sungwoo/miniforge3/envs/trellis/bin/python`

# %%
# 필요 패키지: numpy, pillow, plotly, kaleido
# 실행: /home/sungwoo/miniforge3/envs/trellis/bin/python expy.py
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
np.random.seed(0)
print("numpy", np.__version__)
# 출력: numpy 1.26.4

# %% [markdown]
# ## 1. 입력 이미지 준비
#
# 논문 Fig. 2의 입력 이미지 중 하나(벽 앞의 고양이)를 asset 이미지에서 잘라 쓴다.
# **읽기 전용**으로만 접근한다. 파일이 없으면 "하늘 + 물체" 합성 이미지로 대체한다.
#
# 이미지는 $224 \times 224$, patch 크기는 DINOv2와 같은 $14 \times 14$ → $16 \times 16 = 256$개 patch.

# %%
ASSET = (
    "/home/sungwoo/projects/swcho/dinov2/fm/papers/.fm/assets/"
    "2309.16588v2/_page_1_Figure_1.jpeg"
)
IMG_SIZE, PATCH = 224, 14
G = IMG_SIZE // PATCH  # 격자 한 변 = 16


def load_image():
    """논문 Fig.2의 고양이 입력 이미지를 crop. 실패 시 합성 이미지로 fallback."""
    try:
        from PIL import Image

        im = Image.open(ASSET).convert("RGB")
        # Fig.2 왼쪽 "Input" 열, 두 번째 행(고양이)의 위치
        im = im.crop((37, 202, 175, 341)).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        return np.asarray(im, dtype=np.float32) / 255.0, "논문 Fig.2 입력 이미지 (고양이)"
    except Exception as e:  # pragma: no cover
        print("fallback to synthetic:", e)
        yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float32)
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.float32)
        img[..., 0] = 0.45 + 0.15 * yy / IMG_SIZE  # 균일한 하늘 그라디언트
        img[..., 1] = 0.65 + 0.10 * yy / IMG_SIZE
        img[..., 2] = 0.95
        blob = ((xx - 150) ** 2 + (yy - 150) ** 2) < 45**2
        tex = np.random.rand(IMG_SIZE, IMG_SIZE, 1).astype(np.float32)  # 질감 있는 물체
        img[blob] = (0.25 + 0.6 * tex)[blob]
        return np.clip(img, 0, 1), "합성 이미지 (하늘 + 물체)"


img, img_label = load_image()
print(img_label, img.shape, "grid =", G, "x", G)
# 출력: 논문 Fig.2 입력 이미지 (고양이) (224, 224, 3) grid = 16 x 16

# %% [markdown]
# ## 2. patch로 자르고 flatten
#
# 각 patch는 $14 \times 14 \times 3 = 588$차원 벡터가 된다.
# 이것이 patch embedding layer가 선형 사상을 걸기 직전의 raw 입력이다.
#
# 원본 RGB 값은 전부 양수라 어떤 두 patch를 골라도 코사인 유사도가 0.9 이상으로 뭉개진다.
# 그래서 **전체 patch의 평균 벡터를 빼서(mean-centering)** 비교한다.
# 실제 ViT의 patch embedding도 학습된 projection + bias 때문에 사실상 원점이 옮겨진
# 좌표계에서 동작하므로, centering이 논문 세팅에 더 가깝다.

# %%
patches = (
    img.reshape(G, PATCH, G, PATCH, 3)  # (16,14,16,14,3)
    .transpose(0, 2, 1, 3, 4)  # (16,16,14,14,3)
    .reshape(G * G, PATCH * PATCH * 3)  # (256, 588)
)
P_raw = patches
P = patches - patches.mean(axis=0, keepdims=True)  # mean-centering
print("patches:", P.shape)
# 출력: patches: (256, 588)

# %% [markdown]
# ## 3. 코사인 유사도
#
# 두 patch 벡터 $p_i,\ p_j$ 사이의 코사인 유사도는
#
# $$\mathrm{sim}(p_i, p_j) = \frac{\langle p_i,\, p_j\rangle}{\lVert p_i\rVert\,\lVert p_j\rVert}
#   = \langle \hat p_i,\, \hat p_j \rangle, \qquad \hat p = \frac{p}{\lVert p\rVert}$$
#
# 격자 좌표 $(r,c)$의 patch에 대해 상하좌우 이웃 집합을
# $\mathcal{N}(r,c) = \{(r{\pm}1,c),\,(r,c{\pm}1)\}$ (격자 밖은 제외) 라 할 때,
# 논문이 재는 값은 이웃 4개와의 평균 유사도
#
# $$s_{r,c} = \frac{1}{|\mathcal{N}(r,c)|}\sum_{(u,v)\in\mathcal{N}(r,c)}
#             \mathrm{sim}\bigl(p_{r,c},\, p_{u,v}\bigr)$$
#
# 이고, artifact(high-norm) patch에서는 $s_{r,c} \approx 1$ 이다.

# %%
def neighbor_cos(P, G):
    """각 patch와 상하좌우 4-이웃의 평균 코사인 유사도 -> (G, G)"""
    Ph = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    Ph = Ph.reshape(G, G, -1)
    acc = np.zeros((G, G), np.float64)
    cnt = np.zeros((G, G), np.float64)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r0, r1 = max(0, dr), G + min(0, dr)
        c0, c1 = max(0, dc), G + min(0, dc)
        a = Ph[r0:r1, c0:c1]
        b = Ph[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
        acc[r0:r1, c0:c1] += (a * b).sum(-1)
        cnt[r0:r1, c0:c1] += 1
    return acc / cnt


S = neighbor_cos(P, G)  # mean-centered
S_raw = neighbor_cos(P_raw, G)  # centering 없이 (비교용)
print(f"centered : min={S.min():.3f}  max={S.max():.3f}  mean={S.mean():.3f}")
print(f"raw RGB  : min={S_raw.min():.3f}  max={S_raw.max():.3f}  mean={S_raw.mean():.3f}")
# 출력: centered : min=-0.222  max=1.000  mean=0.683
# 출력: raw RGB  : min=0.651  max=1.000  mean=0.945

# %% [markdown]
# raw RGB로 재면 전 구간이 0.65~1.00(평균 0.95)으로 뭉개져 배경과 물체가 구분되지 않는다.
# centering 후에는 −0.22 ~ 1.00으로 퍼져서, "이웃과 중복인 patch"가 뚜렷이 분리된다.
#
# ## 4. 상위 2% = 논문의 outlier 비율
#
# 논문은 DINOv2 ViT-g에서 norm > 150인 token 비율을 **2.37%** 로 측정했다.
# 여기서도 유사도 상위 2%만 골라 "outlier 후보"로 표시한다.

# %%
OUTLIER_FRAC = 0.02
k = max(1, int(round(G * G * OUTLIER_FRAC)))
thr = np.sort(S.ravel())[-k]
mask = S >= thr
rc = np.argwhere(mask)
print(f"상위 {OUTLIER_FRAC:.0%} = {k}개 / {G*G}개, 임계값 = {thr:.4f}")
print("좌표(row, col):", [tuple(int(v) for v in p) for p in rc])
print(f"상위 2% 평균 유사도 = {S[mask].mean():.4f}")
print(f"나머지 평균 유사도  = {S[~mask].mean():.4f}")
# 출력: 상위 2% = 5개 / 256개, 임계값 = 0.9996
# 출력: 좌표(row, col): [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
# 출력: 상위 2% 평균 유사도 = 0.9998
# 출력: 나머지 평균 유사도  = 0.6770

# %% [markdown]
# 뽑힌 좌표가 모두 **col 0 (row 0~4)**, 즉 이미지 **왼쪽 위 가장자리의 균일한 벽**이다.
# 고양이 털·눈·윤곽에서는 한 칸만 옮겨도 픽셀 패턴이 바뀌므로 유사도가 낮다.
# 이는 논문 부록 Fig. 10(오른쪽)에서 outlier가 feature map **테두리/모서리**에 몰린다는
# 관찰과도 방향이 같다 — object-centric 사진에서 가장자리는 대개 배경이기 때문.
#
# ## 5. 시각화

# %%
def to_uint8(a):
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        f"(a) 입력 — {img_label}",
        "(b) 4-이웃 평균 코사인 유사도 (mean-centered)",
        "(c) 유사도 상위 2% = outlier 후보 위치",
        "(d) 유사도 분포: 상위 2% vs 나머지",
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.09,
)

# (a) 원본 + patch 격자
fig.add_trace(go.Image(z=to_uint8(img)), row=1, col=1)
for i in range(1, G):
    fig.add_shape(
        type="line", x0=i * PATCH, x1=i * PATCH, y0=0, y1=IMG_SIZE,
        line=dict(color="rgba(255,255,255,0.22)", width=1), row=1, col=1,
    )
    fig.add_shape(
        type="line", x0=0, x1=IMG_SIZE, y0=i * PATCH, y1=i * PATCH,
        line=dict(color="rgba(255,255,255,0.22)", width=1), row=1, col=1,
    )

# (b) 유사도 히트맵
fig.add_trace(
    go.Heatmap(
        z=S, colorscale="Magma", zmin=float(S.min()), zmax=1.0,
        colorbar=dict(title="cos", len=0.38, y=0.80, x=1.005, thickness=12),
        hovertemplate="row %{y}, col %{x}<br>cos=%{z:.3f}<extra></extra>",
    ),
    row=1, col=2,
)
fig.update_yaxes(autorange="reversed", row=1, col=2, scaleanchor="x2", scaleratio=1)

# (c) 흑백 원본 위에 상위 2% 오버레이
gray = img.mean(-1)
overlay = np.stack([gray * 0.55] * 3, -1)
for r, c in rc:
    overlay[r * PATCH : (r + 1) * PATCH, c * PATCH : (c + 1) * PATCH] = [1.0, 0.85, 0.1]
fig.add_trace(go.Image(z=to_uint8(overlay)), row=2, col=1)

# (d) 분포 (논문 Fig.5a에 대응) — 두 trace가 같은 bin 격자를 쓰도록 명시
lo = float(np.floor(S.min() * 20) / 20)
bins = dict(start=lo, end=1.0, size=(1.0 - lo) / 40)
fig.add_trace(
    go.Histogram(
        x=S[~mask], xbins=bins, autobinx=False, name="나머지 98% (normal)",
        marker_color="#4C78A8", opacity=0.85,
    ),
    row=2, col=2,
)
fig.add_trace(
    go.Histogram(
        x=S[mask], xbins=bins, autobinx=False, name="상위 2% (artifact 후보)",
        marker_color="#F58518", opacity=1.0,
        marker_line=dict(color="#7a3d00", width=1),
    ),
    row=2, col=2,
)
fig.add_vline(
    x=thr, line=dict(color="crimson", dash="dash", width=1.5),
    annotation_text=f"상위 2% 임계 {thr:.3f}", annotation_position="top left",
    row=2, col=2,
)

for r, c in [(1, 1), (2, 1)]:
    fig.update_xaxes(visible=False, row=r, col=c)
    fig.update_yaxes(visible=False, row=r, col=c)
fig.update_xaxes(title_text="4-이웃 평균 코사인 유사도", row=2, col=2)
fig.update_yaxes(title_text="patch 수", type="log", row=2, col=2)
fig.update_layout(
    title=dict(
        text="high-norm token은 '이웃과 중복된' 균일 배경 patch에서 나타난다"
        f"  ({G}×{G} patches, {PATCH}×{PATCH} px)",
        x=0.5,
    ),
    width=1100, height=880, bargap=0.05, barmode="overlay",
    template="plotly_white",
    legend=dict(orientation="h", x=0.55, y=-0.06, xanchor="center"),
)
_show(fig)

out = os.path.join(HERE, "expy.png")
fig.write_image(out, scale=2)
print("saved:", out)
# 출력: saved: /home/sungwoo/projects/swcho/dinov2/fm/papers/.fm/hints/7090ca8b-3f02-4264-bd9d-c9a2e44a40ea/expy.png

# %% [markdown]
# ## 6. 정리
#
# | 관찰 | 이 예제 | 논문 |
# |---|---|---|
# | 측정 시점 | 원본 픽셀 patch | patch embedding **직후** |
# | 측정량 | 상하좌우 **4-이웃** 평균 코사인 유사도 | 동일 |
# | outlier 비율 | 상위 2% | norm > 150 인 token = **2.37%** |
# | outlier 유사도 | ≈ 0.9998 (나머지 0.68) | 1.0 근처에 뾰족한 스파이크 (Fig. 5a) |
# | 위치 | 왼쪽 위 **균일한 벽** | **균일한 배경**, feature map **가장자리** (Fig. 10) |
#
# 주의: 이 예제는 **필요조건 쪽**만 보인다. "이웃과 유사한 patch"가 전부 high-norm이
# 되지는 않는다 — 배경 patch는 훨씬 많지만 실제 outlier는 2%뿐이고,
# 그마저도 ViT-Large 이상 크기 + 충분한 학습 이후 layer 15 부근에서야 나타난다.
# 논문의 해석은 "모델이 **버려도 되는** token을 골라 global 정보 저장용으로 재활용한다"이고,
# 처방이 바로 **register token** 추가다.
