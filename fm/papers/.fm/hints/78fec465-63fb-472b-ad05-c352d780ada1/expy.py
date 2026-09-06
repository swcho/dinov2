# %% [markdown]
# # DINOv2 semantic segmentation "Linear" 설정 — numpy 토이 재현
#
# 논문(Sec. 7.4)의 **Linear** 설정:
#
# 1. frozen ViT 백본이 512×512 입력을 패치 크기 $p=16$으로 쪼개 $32\times32$ 개의 패치 토큰 $\mathbf{x}_{ij}\in\mathbb{R}^D$ 를 낸다.
# 2. 패치마다 **같은** 선형 층 $W\in\mathbb{R}^{C\times D}$ 를 적용해 저해상도 logit map $\mathbf{z}\in\mathbb{R}^{32\times32\times C}$ 를 만든다.
#    $$\mathbf{z}_{ij} = W\mathbf{x}_{ij} + \mathbf{b}$$
# 3. logit map을 512×512로 **bilinear 업샘플링**한 뒤 argmax → 분할 마스크.
#
# 이 스크립트는 백본을 학습하지 않고, "패치 토큰이 클래스 정보를 선형적으로 담고 있다"는 가정만
# 합성 데이터로 흉내 내서 **왜 경계가 뭉개지는지**를 mIoU 숫자와 그림으로 확인한다.

# %%
# 필요 패키지: numpy, plotly, kaleido (scipy는 있으면 사용, 없으면 numpy bilinear 구현으로 대체)
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


rng = np.random.default_rng(0)
H = W = 512          # 입력/출력 해상도
P = 16               # 패치 크기
G = H // P           # 패치 격자 한 변 = 32
C = 3                # 클래스 수 (배경 / 물체A / 물체B)
D = 64               # 토큰 차원 (실제 ViT-g는 1536)
print(f"패치 격자: {G}x{G} = {G*G} 토큰, 각 토큰 {D}차원, 클래스 {C}개")
# 출력: 패치 격자: 32x32 = 1024 토큰, 각 토큰 64차원, 클래스 3개

# %% [markdown]
# ## 1. 곡선 경계를 가진 512×512 ground-truth 마스크
#
# 클래스 1은 기울어진 타원, 클래스 2는 얇은 곡선 띠(가늘고 굽은 물체 — 저해상도에서 특히 취약한 형태).

# %%
yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
u = (xx - 230) / 150.0
v = (yy - 260) / 95.0
theta = np.deg2rad(25)
ur = u * np.cos(theta) - v * np.sin(theta)
vr = u * np.sin(theta) + v * np.cos(theta)
ellipse = (ur**2 + vr**2) <= 1.0

band_center = 330 + 60 * np.sin(yy / 70.0)          # x 위치가 y에 따라 흔들리는 띠
band = np.abs(xx - band_center) < 5                 # 폭 10px < 패치 한 변(16px)

gt = np.zeros((H, W), dtype=np.int64)
gt[ellipse] = 1
gt[band] = 2                                        # 띠가 타원 위를 지나면 띠가 우선
for c in range(C):
    print(f"class {c}: {(gt == c).mean()*100:5.1f}% 픽셀")
# 출력: class 0:  81.6% 픽셀
# 출력: class 1:  16.5% 픽셀
# 출력: class 2:   2.0% 픽셀

# %% [markdown]
# ## 2. 패치 토큰 시뮬레이션 (frozen 백본 흉내)
#
# 각 패치의 토큰은 그 패치 안 픽셀들의 **클래스 비율** $\pi_{ij}\in\Delta^{C}$ 에 따라 클래스 프로토타입
# $\mu_c\in\mathbb{R}^D$ 를 섞은 벡터 + 노이즈로 만든다:
# $$\mathbf{x}_{ij} = \sum_c \pi_{ij,c}\,\mu_c + \epsilon,\qquad \epsilon\sim\mathcal N(0,\sigma^2 I)$$
# 핵심 가정: **한 패치 = 한 토큰**. 경계가 패치 안을 지나가면 그 정보는 토큰 하나에 뭉개져 들어간다.

# %%
# (32,32,16,16) 블록으로 재배열 → 패치별 클래스 히스토그램
gt_blocks = gt.reshape(G, P, G, P).transpose(0, 2, 1, 3).reshape(G, G, P * P)
pi = np.stack([(gt_blocks == c).mean(-1) for c in range(C)], axis=-1)   # (32,32,C)

mu = rng.normal(size=(C, D)) * 1.0                # 클래스 프로토타입
sigma = 0.6
tokens = pi @ mu + rng.normal(size=(G, G, D)) * sigma   # (32,32,D)
print("tokens.shape =", tokens.shape)
pure = (pi.max(-1) == 1.0).mean()
print(f"단일 클래스만 담은 '순수' 패치 비율: {pure*100:.1f}%  → 나머지는 경계가 지나는 혼합 패치")
# 출력: tokens.shape = (32, 32, 64)
# 출력: 단일 클래스만 담은 '순수' 패치 비율: 88.4%  → 나머지는 경계가 지나는 혼합 패치

# %% [markdown]
# ## 3. 선형 층 $D\to C$ 학습 (패치 단위 분류)
#
# 논문처럼 타깃은 **패치 해상도**로 내린 GT(여기선 패치의 다수 클래스)이고, 손실은 cross-entropy.
# 파라미터는 $W\in\mathbb{R}^{C\times D},\ \mathbf b\in\mathbb{R}^C$ 만 학습한다 — 백본(=tokens)은 frozen.

# %%
X = tokens.reshape(-1, D)                         # (1024, D)
y = pi.reshape(-1, C).argmax(-1)                  # 패치의 다수 클래스 (1024,)
Y = np.eye(C)[y]

Wl = np.zeros((D, C)); bl = np.zeros(C)
lr = 0.5
for it in range(300):
    logits = X @ Wl + bl
    prob = np.exp(logits - logits.max(1, keepdims=True)); prob /= prob.sum(1, keepdims=True)
    grad = (prob - Y) / len(X)
    Wl -= lr * X.T @ grad
    bl -= lr * grad.sum(0)
loss = -np.log(prob[np.arange(len(y)), y] + 1e-12).mean()
patch_acc = ((X @ Wl + bl).argmax(-1) == y).mean()
print(f"학습 후 CE loss = {loss:.3f}, 패치 단위 정확도 = {patch_acc*100:.1f}%")
print(f"학습 파라미터 수 = {Wl.size + bl.size}  (D*C + C)")
# 출력: 학습 후 CE loss = 0.003, 패치 단위 정확도 = 100.0%
# 출력: 학습 파라미터 수 = 195  (D*C + C)

# %% [markdown]
# ## 4. 32×32 logit map → 512×512 bilinear 업샘플링 → argmax
#
# 업샘플링은 **logit** 에 대해 수행한 뒤 argmax 한다(논문 순서). 픽셀 $(r,s)$ 의 값은 주변 4개 패치 중심의 logit을
# 거리 가중 평균한 것이므로, 경계는 최대 패치 한 변(16px) 폭에 걸쳐 부드럽게 섞인다.

# %%
logit_map = (tokens @ Wl + bl)                    # (32,32,C) 저해상도 logit map
pred_lowres = logit_map.argmax(-1)                # 32×32 예측


def bilinear_upsample(z, out_h, out_w):
    """z: (h,w,C) → (out_h,out_w,C). align_corners=False 방식(패치 중심을 픽셀 중심에 맞춤)."""
    h, w, _ = z.shape
    ys = (np.arange(out_h) + 0.5) * h / out_h - 0.5
    xs = (np.arange(out_w) + 0.5) * w / out_w - 0.5
    ys = np.clip(ys, 0, h - 1); xs = np.clip(xs, 0, w - 1)
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, h - 1); x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None, None]; wx = (xs - x0)[None, :, None]
    return ((1 - wy) * (1 - wx) * z[y0][:, x0] + (1 - wy) * wx * z[y0][:, x1]
            + wy * (1 - wx) * z[y1][:, x0] + wy * wx * z[y1][:, x1])


try:
    from scipy.ndimage import zoom
    logit_up = np.stack([zoom(logit_map[..., c], P, order=1, grid_mode=True, mode="nearest")
                         for c in range(C)], -1)
    how = "scipy.ndimage.zoom(order=1)"
except ImportError:
    logit_up = bilinear_upsample(logit_map, H, W)
    how = "numpy bilinear"
pred_linear = logit_up.argmax(-1)                 # 최종 512×512 분할
pred_nearest = np.repeat(np.repeat(pred_lowres, P, 0), P, 1)   # 비교용: 그냥 블록 복제
print(f"업샘플링 방식: {how}, logit_up.shape = {logit_up.shape}")
# 출력: 업샘플링 방식: scipy.ndimage.zoom(order=1), logit_up.shape = (512, 512, 3)

# %% [markdown]
# ## 5. mIoU 평가와 "경계 오류" 분해
#
# $$\mathrm{IoU}_c=\frac{|P_c\cap G_c|}{|P_c\cup G_c|},\qquad \mathrm{mIoU}=\frac1C\sum_c \mathrm{IoU}_c$$
# 패치 단위 정확도는 100%인데 픽셀 mIoU는 왜 떨어질까? 오류 픽셀이 **어디** 있는지 세어 보면 답이 나온다.

# %%
def miou(pred, gt, C):
    ious = []
    for c in range(C):
        inter = ((pred == c) & (gt == c)).sum(); union = ((pred == c) | (gt == c)).sum()
        ious.append(inter / union if union else np.nan)
    return np.nanmean(ious), ious


for name, pr in [("nearest(블록 복제)", pred_nearest), ("bilinear(논문 Linear)", pred_linear)]:
    m, ious = miou(pr, gt, C)
    print(f"{name:22s} pixel acc={np.mean(pr == gt)*100:5.1f}%  mIoU={m*100:5.1f}  "
          + "  ".join(f"IoU{c}={v*100:4.1f}" for c, v in enumerate(ious)))
# 출력: nearest(블록 복제)         pixel acc= 97.1%  mIoU= 74.0  IoU0=97.2  IoU1=90.5  IoU2=34.5
# 출력: bilinear(논문 Linear)    pixel acc= 98.0%  mIoU= 69.6  IoU0=98.2  IoU1=94.9  IoU2=15.8

# GT 경계에서 8px(패치 반 변) 이내에 있는 픽셀 = "경계 지대"
gy = np.abs(np.diff(gt, axis=0, prepend=gt[:1])) > 0
gx = np.abs(np.diff(gt, axis=1, prepend=gt[:, :1])) > 0
edge = gy | gx
try:
    from scipy.ndimage import binary_dilation
    border = binary_dilation(edge, iterations=P // 2)
except ImportError:
    border = edge.copy()
    for _ in range(P // 2):
        border = border | np.roll(border, 1, 0) | np.roll(border, -1, 0) | np.roll(border, 1, 1) | np.roll(border, -1, 1)
err = pred_linear != gt
print(f"경계 지대(GT 경계 ±{P//2}px) 픽셀 비율: {border.mean()*100:.1f}%")
print(f"오류 픽셀 중 경계 지대에 있는 비율: {(err & border).sum() / err.sum() * 100:.1f}%")
print(f"얇은 띠(class 2, 폭 10px) 재현율: {((pred_linear == 2) & (gt == 2)).sum() / (gt == 2).sum() * 100:.1f}%")
# 출력: 경계 지대(GT 경계 ±8px) 픽셀 비율: 9.7%
# 출력: 오류 픽셀 중 경계 지대에 있는 비율: 100.0%
# 출력: 얇은 띠(class 2, 폭 10px) 재현율: 16.4%

# %% [markdown]
# 결론: 패치 분류는 완벽해도 **오류가 전부 경계 ±8px 지대에 몰려** 있고, 패치 폭보다 얇은 구조(class 2, 10px)는
# IoU가 15.8%로 무너진다(bilinear는 넓은 영역 경계는 다듬어 주지만, 얇은 구조의 logit은 이웃 패치와 평균되어 오히려 사라진다 — nearest 34.5% → bilinear 15.8%). 논문이 "extremely simple but cannot easily produce high-resolution segmentations"
# 라고 쓴 이유가 그대로 재현된다. `+ms`(4개 층 concat, 640 해상도, multiscale TTA)나
# Mask2Former 헤드가 하는 일은 결국 이 경계 정보를 되살리는 것이다.

# %% [markdown]
# ## 6. 시각화: GT / 32×32 logit argmax / 업샘플 예측 / 오류 위치

# %%
colors = [[0, "#1f2833"], [0.5, "#f28e2b"], [1, "#4e79a7"]]
fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.03,
                    subplot_titles=("GT 512×512", f"{G}×{G} logit argmax (패치 단위)",
                                    "bilinear 업샘플 → argmax", "오류 픽셀 (경계에 몰림)"))
hm = dict(colorscale=colors, zmin=0, zmax=C - 1, showscale=False)
fig.add_trace(go.Heatmap(z=gt[::-1], **hm), 1, 1)
fig.add_trace(go.Heatmap(z=pred_lowres[::-1], **hm), 1, 2)
fig.add_trace(go.Heatmap(z=pred_linear[::-1], **hm), 1, 3)
fig.add_trace(go.Heatmap(z=err[::-1].astype(int), colorscale=[[0, "#f4f4f4"], [1, "#d62728"]],
                         zmin=0, zmax=1, showscale=False), 1, 4)
# 두 번째 패널은 32×32 격자이므로 픽셀 축과 비율만 맞춘다
for c in range(1, 5):
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=c)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, scaleanchor=f"x{c if c > 1 else ''}", row=1, col=c)
m_lin, _ = miou(pred_linear, gt, C)
fig.update_layout(width=1500, height=440, template="plotly_white",
                  title=f"DINOv2 'Linear' 분할 설정 토이: 패치 {P}, 격자 {G}×{G}, D={D}→C={C}, "
                        f"픽셀 mIoU={m_lin*100:.1f} (패치 정확도 {patch_acc*100:.0f}%)")
_show(fig)
out = Path(__file__).resolve().parent / "expy.png" if "__file__" in globals() else Path("expy.png")
fig.write_image(str(out), scale=1)
print("saved", out)
# 출력: saved .../expy.png
