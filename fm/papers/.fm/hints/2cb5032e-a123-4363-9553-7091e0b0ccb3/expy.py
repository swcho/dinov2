# %% [markdown]
# # 부록 A 재현: bicubic 보간의 aliasing이 만드는 수직 줄무늬
#
# **논문**: Darcet et al., *Vision Transformers Need Registers* (arXiv:2309.16588), Appendix A
#
# 논문의 주장은 이렇다.
#
# > 원래 DINOv2 구현은 학습 중 position embedding을 $16\times16$ 맵에서 $7\times7$ 맵으로
# > **antialiasing 없이** 보간했다. 이런 보간 함수(bicubic resize)에 **단위 gradient**를
# > 흘려보내면 Fig. 11 같은 gradient가 나오고, 이는 outlier 위치 분포(Fig. 10 left)에서
# > 관찰된 수직 줄무늬 패턴과 닮았다.
#
# 이 노트북은 그 실험을 torch로 그대로 재현한다.
#
# - 왜 $16\to7$인가: DINOv2 학습 설정에서 global crop은 $224/14 = 16$ 패치,
#   local crop은 $98/14 = 7$ 패치다. pos_embed는 $16\times16$ 격자로 학습되므로
#   local crop마다 $7\times7$로 다운샘플된다.
# - 왜 "단위 gradient"인가: 출력 $7\times7$ 전부에 대해 $\partial L/\partial y = 1$을 주면
#   (`y.sum().backward()`), 입력 $16\times16$이 받는 gradient는
#   $\;g_{ij} = \sum_{pq} w_{pq,ij}\;$ 즉 **각 입력 셀이 출력에 기여하는 총 가중치**가 된다.
#   학습 내내 이 비율로 pos_embed가 갱신되므로, $g$가 균일하지 않으면
#   특정 위치의 위치 임베딩만 계속 크게/작게 학습된다.
#
# 실행: `/home/sungwoo/miniforge3/envs/trellis/bin/python expy.py`
# (의존성: torch 2.4 / numpy 1.26 / plotly 6.9 / kaleido — 모두 이 인터프리터에 있음. CPU만으로 동작)

# %%
import os

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots

torch.manual_seed(0)
DEVICE = "cpu"
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


print("torch", torch.__version__, "| device", DEVICE)
# 출력: torch 2.4.0+cu121 | device cpu

# %% [markdown]
# ## 1. 실제 DINOv2 코드에서 `antialias`가 들어가는 자리
#
# `dinov2/models/vision_transformer.py`의 `interpolate_pos_encoding` (발췌):
#
# ```python
# M = int(math.sqrt(N))            # pos_embed 격자 한 변 = 16
# kwargs = {}
# if self.interpolate_offset:
#     # Historical kludge: add a small number to avoid floating point error ...
#     sx = float(w0 + self.interpolate_offset) / M      # (7 + 0.1) / 16
#     sy = float(h0 + self.interpolate_offset) / M
#     kwargs["scale_factor"] = (sx, sy)
# else:
#     kwargs["size"] = (w0, h0)                          # (7, 7)
# patch_pos_embed = nn.functional.interpolate(
#     patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
#     mode="bicubic",
#     antialias=self.interpolate_antialias,              # <-- 여기!
#     **kwargs,
# )
# ```
#
# 원조 DINOv2 체크포인트는 `interpolate_antialias=False`, `interpolate_offset=0.1`로 학습됐다
# (`dinov2/hub/classifiers.py`의 기본값이 정확히 그것). registers 논문 이후 공개된 모델은
# `interpolate_antialias=True, interpolate_offset=0.0`을 쓴다.

# %%
# 논문 Appendix A 실험 그대로: 16x16 단일 채널 -> 7x7 bicubic 다운샘플, 단위 gradient 역전파.
IN, OUT = 16, 7


def unit_grad(antialias: bool, **kwargs) -> np.ndarray:
    """16x16 입력을 (기본) 7x7로 bicubic 다운샘플한 뒤 출력 전체에 1의 gradient를 흘린다."""
    x = torch.zeros(1, 1, IN, IN, device=DEVICE, requires_grad=True)
    y = F.interpolate(x, mode="bicubic", antialias=antialias, **kwargs)
    assert y.shape[-2:] == (OUT, OUT), y.shape
    y.sum().backward()  # 단위 gradient
    return x.grad[0, 0].detach().numpy()


g_off = unit_grad(antialias=False, size=(OUT, OUT))
g_on = unit_grad(antialias=True, size=(OUT, OUT))

np.set_printoptions(precision=2, suppress=True, linewidth=200)
print("antialias=False, 첫 4행:")
print(g_off[:4])
# 출력: antialias=False, 첫 4행:
# 출력: [[ 0.12  0.26 -0.02  0.34 -0.05  0.31  0.04  0.19  0.19  0.04  0.31 -0.05  0.34 -0.02  0.26  0.12]
# 출력:  [ 0.26  0.59 -0.04  0.76 -0.11  0.7   0.09  0.43  0.43  0.09  0.7  -0.11  0.76 -0.04  0.59  0.26]
# 출력:  [-0.02 -0.04  0.   -0.05  0.01 -0.05 -0.01 -0.03 -0.03 -0.01 -0.05  0.01 -0.05  0.   -0.04 -0.02]
# 출력:  [ 0.34  0.76 -0.05  0.98 -0.14  0.9   0.12  0.56  0.56  0.12  0.9  -0.14  0.98 -0.05  0.76  0.34]]

print("\nantialias=True, 첫 4행:")
print(g_on[:4])
# 출력: antialias=True, 첫 4행:
# 출력: [[0.13 0.18 0.17 0.16 0.16 0.16 0.16 0.16 0.16 0.16 0.16 0.16 0.16 0.17 0.18 0.13]
# 출력:  [0.18 0.24 0.23 0.21 0.21 0.22 0.22 0.21 0.21 0.22 0.22 0.21 0.21 0.23 0.24 0.18]
# 출력:  [0.17 0.23 0.22 0.2  0.2  0.21 0.21 0.21 0.21 0.21 0.21 0.2  0.2  0.22 0.23 0.17]
# 출력:  [0.16 0.21 0.2  0.19 0.19 0.19 0.19 0.19 0.19 0.19 0.19 0.19 0.19 0.2  0.21 0.16]]

# %% [markdown]
# 눈으로도 바로 보인다.
#
# - `antialias=False`: 열 인덱스 0,1,3,5,7,8,10,12,14,15는 큰 양수, 열 2,4,11,13은 **0 근처거나 음수**.
#   같은 패턴이 행 방향에도 있어 격자무늬(수직 줄무늬 + 수평 줄무늬)가 된다.
# - `antialias=True`: 전부 0.13~0.24 사이. 사실상 균일.
#
# 총합은 두 경우 모두 같다 — bicubic 커널은 정규화돼 있으므로
# $\sum_{ij} g_{ij} = 7\times7 = 49$. 즉 문제는 "총량"이 아니라 **분배의 불균형**이다.

# %%
print("합계 (antialias=False):", round(float(g_off.sum()), 4))
print("합계 (antialias=True) :", round(float(g_on.sum()), 4))
print("출력 픽셀 수 7x7      :", OUT * OUT)
# 출력: 합계 (antialias=False): 49.0
# 출력: 합계 (antialias=True) : 49.0
# 출력: 출력 픽셀 수 7x7      : 49

# %% [markdown]
# ## 2. 정량 비교: 줄무늬의 세기
#
# 입력 격자의 **열별 gradient 합** $c_j = \sum_i g_{ij}$ 로 1D 단면을 만들어 본다.
# 균일하다면 $c_j$는 모두 $49/16 = 3.0625$ 여야 한다.

# %%
col_off, col_on = g_off.sum(axis=0), g_on.sum(axis=0)
ideal = OUT * OUT / IN

print("이상적인 열 합:", round(ideal, 4))
print("antialias=False 열 합:", np.round(col_off, 2))
print("antialias=True  열 합:", np.round(col_on, 2))
# 출력: 이상적인 열 합: 3.0625
# 출력: antialias=False 열 합: [ 2.39  5.36 -0.35  6.92 -1.02  6.36  0.86  3.97  3.97  0.86  6.36 -1.02  6.92 -0.35  5.36  2.39]
# 출력: antialias=True  열 합: [2.53 3.44 3.29 3.03 3.03 3.06 3.06 3.06 3.06 3.06 3.06 3.03 3.03 3.29 3.44 2.53]


def stats(name, g, c):
    print(
        f"{name:16s} | cell std {g.std():.4f} | cell max-min {g.max() - g.min():.4f} "
        f"| col std {c.std():.4f} | col max/min {c.max() / c.min():+.2f}"
    )


stats("antialias=False", g_off, col_off)
stats("antialias=True", g_on, col_on)
print(f"\n열 합 표준편차 비율: {col_off.std() / col_on.std():.1f}배")
# 출력: antialias=False  | cell std 0.3031 | cell max-min 1.1219 | col std 2.8612 | col max/min -6.80
# 출력: antialias=True   | cell std 0.0218 | cell max-min 0.1115 | col std 0.2458 | col max/min +1.36
# 출력:
# 출력: 열 합 표준편차 비율: 11.6배

# %% [markdown]
# `antialias=False`는 열별 gradient가 **부호까지 뒤집힌다** (max/min 비가 음수).
# 어떤 위치의 position embedding은 매 step 강하게 밀리고, 바로 옆 위치는 거의 갱신되지 않거나
# **반대 방향으로** 갱신된다. 학습 내내 누적되면 특정 열의 위치 임베딩만 이상한 값이 되고,
# 그 위치의 토큰이 high-norm outlier가 되기 쉬워진다 — 이것이 Fig. 10 left의 수직 줄무늬다.

# %% [markdown]
# ## 3. 왜 이런 일이 벌어지나: 커널 폭과 배율
#
# `F.interpolate`의 bicubic은 출력 픽셀 $p$의 중심을 입력 좌표로 되돌린 뒤
# 그 주변 **고정된 4탭** 창에 Keys의 bicubic 커널 $k(t)$ ($a=-0.75$)를 씌운다:
#
# $$ y_p \;=\; \sum_{m=-1}^{2} k\!\left(t_p - m\right)\, x_{\lfloor u_p \rfloor + m},
# \qquad u_p = \frac{p + 0.5}{s} - 0.5,\quad t_p = u_p - \lfloor u_p \rfloor $$
#
# 여기서 $s = 7/16 = 0.4375$. 즉 출력 픽셀 간격이 입력 격자 기준 $1/s \approx 2.29$칸인데
# 커널은 여전히 폭 4칸짜리다. 그래서:
#
# - 어떤 입력 열은 단 하나의 출력 픽셀에만, 그것도 bicubic 커널의 **음의 lobe**
#   ($|t|>1$ 구간에서 $k(t)<0$)에만 걸린다 → gradient가 **음수**가 된다
# - 어떤 열은 출력 픽셀 중심 바로 위에 놓여 $k(0)\approx 1$ 을 통째로 받는다 → gradient ≈ 1
# - 출력 픽셀이 입력 격자를 $1/s \approx 2.29$칸 간격으로 훑으므로 이 운/불운이 **주기적으로** 반복된다
#
#
# `antialias=True`는 다운샘플 시 커널을 배율만큼 늘려($\text{support} \propto 1/s$, 여기선 폭 ~9칸)
# **모든 입력 열이 어떤 출력 픽셀엔가 유의미하게 기여**하게 만든다. 그래서 gradient가 고르게 퍼진다.
#
# 아래에서 실제 가중치 행렬 $W \in \mathbb{R}^{7\times16}$ 를 뽑아 확인한다
# (1D로 줄여도 성질은 같다: $g = \mathbf{1}^\top W$).

# %%
def weight_matrix(antialias: bool) -> np.ndarray:
    """1D 16 -> 7 bicubic의 명시적 가중치 행렬 W (행=출력, 열=입력). 입력 basis에 각각 통과."""
    eye = torch.eye(IN, device=DEVICE).reshape(IN, 1, 1, IN)  # 16개의 1x16 신호
    out = F.interpolate(eye, size=(1, OUT), mode="bicubic", antialias=antialias)
    return out.reshape(IN, OUT).T.numpy()  # (7, 16)


W_off, W_on = weight_matrix(False), weight_matrix(True)

print("antialias=False, 출력 열 p가 참조하는 입력 열 (|w|>1e-3):")
for p in range(OUT):
    idx = np.where(np.abs(W_off[p]) > 1e-3)[0]
    print(f"  p={p}: {idx.tolist()}  (탭 {len(idx)}개)")
# 출력: antialias=False, 출력 열 p가 참조하는 입력 열 (|w|>1e-3):
# 출력:   p=0: [0, 1, 2]  (탭 3개)
# 출력:   p=1: [1, 2, 3, 4]  (탭 4개)
# 출력:   p=2: [4, 5, 6, 7]  (탭 4개)
# 출력:   p=3: [6, 7, 8, 9]  (탭 4개)
# 출력:   p=4: [8, 9, 10, 11]  (탭 4개)
# 출력:   p=5: [11, 12, 13, 14]  (탭 4개)
# 출력:   p=6: [13, 14, 15]  (탭 3개)

print("\nantialias=True, 같은 기준:")
for p in range(OUT):
    idx = np.where(np.abs(W_on[p]) > 1e-3)[0]
    print(f"  p={p}: {idx.tolist()}  (탭 {len(idx)}개)")
# 출력: antialias=True, 같은 기준:
# 출력:   p=0: [0, 1, 2, 3, 4, 5]  (탭 6개)
# 출력:   p=1: [0, 1, 2, 3, 4, 5, 6, 7]  (탭 8개)
# 출력:   p=2: [1, 2, 3, 4, 5, 6, 7, 8, 9]  (탭 9개)
# 출력:   p=3: [4, 5, 6, 7, 8, 9, 10, 11]  (탭 8개)
# 출력:   p=4: [6, 7, 8, 9, 10, 11, 12, 13, 14]  (탭 9개)
# 출력:   p=5: [8, 9, 10, 11, 12, 13, 14, 15]  (탭 8개)
# 출력:   p=6: [10, 11, 12, 13, 14, 15]  (탭 6개)

# %%
# antialias=False에서 입력 열 5는 p=2 하나만 커버하고, 그 가중치도 음의 lobe에 걸린다.
for j in (2, 4, 5, 3):
    w = W_off[:, j]
    hit = np.where(np.abs(w) > 1e-3)[0]
    print(f"입력 열 {j:2d}: 열 합 {w.sum():+.3f}, 걸리는 출력 {hit.tolist()}, 값 {np.round(w[hit], 3).tolist()}")
# 출력: 입력 열  2: 열 합 -0.050, 걸리는 출력 [0, 1], 값 [-0.111, 0.061]
# 출력: 입력 열  4: 열 합 -0.145, 걸리는 출력 [1, 2], 값 [-0.046, -0.099]
# 출력: 입력 열  5: 열 합 +0.909, 걸리는 출력 [2], 값 [0.909]
# 출력: 입력 열  3: 열 합 +0.989, 걸리는 출력 [1], 값 [0.989]

# %% [markdown]
# 입력 열 4는 **음수 가중치만** 받아 순합이 $-0.145$, 열 2도 순합이 음수다.
# 반면 열 3은 출력 $p{=}1$ 하나에서 $+0.989$, 열 5는 출력 $p{=}2$ 하나에서 $+0.909$를 받는다.
# 이웃한 열끼리 gradient가 $-0.15$ 와 $+0.99$ 로 갈리는 것 — 이게 줄무늬의 정체다.
#
# ## 4. 논문 Fig. 11과의 대조: `interpolate_offset` 켠 버전
#
# 실제 DINOv2 학습은 `size=(7,7)`이 아니라 `scale_factor=(7.1/16, 7.1/16)`를 썼다
# (float 오차 회피용 historical kludge). 그러면 좌우 대칭이 깨지면서
# 논문 Fig. 11의 비대칭 패턴(오른쪽 끝 열/아래 행이 흐릿한 모습)이 나온다.

# %%
s = (7 + 0.1) / 16
g_paper = unit_grad(antialias=False, scale_factor=(s, s))
col_paper = g_paper.sum(axis=0)
print("scale_factor =", round(s, 5))
print("열 합:", np.round(col_paper, 2))
print("행 합:", np.round(g_paper.sum(axis=1), 2))
# 출력: scale_factor = 0.44375
# 출력: 열 합: [ 2.51  5.19 -0.01  6.79 -1.01  6.74  0.11  5.06  2.67  2.35  5.32 -0.11  6.83 -1.01  6.68  0.88]
# 출력: 행 합: [ 2.51  5.19 -0.01  6.79 -1.01  6.74  0.11  5.06  2.67  2.35  5.32 -0.11  6.83 -1.01  6.68  0.88]

# %% [markdown]
# 열 2, 6, 11이 ~0, 열 4와 13이 음수, 마지막 열 15가 눈에 띄게 작다 —
# 논문 Fig. 11의 회색/파란 줄 위치와 일치한다.

# %%
# --- 시각화: 히트맵 2장(+논문 재현판) + 열별 gradient 막대 ---
zmax = float(np.abs(np.concatenate([g_off, g_on])).max())

fig = make_subplots(
    rows=2,
    cols=3,
    row_heights=[0.62, 0.38],
    vertical_spacing=0.13,
    horizontal_spacing=0.07,
    subplot_titles=(
        "antialias=False (size=7)",
        "antialias=False (offset 0.1, 논문 Fig.11)",
        "antialias=True (size=7)",
        "열별 gradient 합 — 이상값 3.06 대비",
        None,
        None,
    ),
    specs=[
        [{}, {}, {}],
        [{"colspan": 3}, None, None],
    ],
)

for c, (g, name) in enumerate(
    [(g_off, "off"), (g_paper, "paper"), (g_on, "on")], start=1
):
    fig.add_trace(
        go.Heatmap(
            z=g,
            zmin=-zmax,
            zmax=zmax,
            colorscale="RdBu_r",  # 발산형: 따뜻/차가운 두 색 + 중립 midpoint(0)
            showscale=(c == 3),
            colorbar=dict(len=0.5, y=0.75, thickness=10, title="grad"),
            hovertemplate="row %{y}, col %{x}<br>grad %{z:.3f}<extra></extra>",
            name=name,
        ),
        row=1,
        col=c,
    )
    fig.update_yaxes(autorange="reversed", scaleanchor=f"x{'' if c == 1 else c}", row=1, col=c)

x = list(range(IN))
fig.add_trace(
    go.Bar(
        x=x,
        y=col_off,
        name="antialias=False",
        marker_color="#c2453a",
        marker_line_width=0,
        hovertemplate="입력 열 %{x}<br>합 %{y:.2f}<extra>antialias=False</extra>",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=x,
        y=col_on,
        name="antialias=True",
        marker_color="#2f6f9f",
        marker_line_width=0,
        hovertemplate="입력 열 %{x}<br>합 %{y:.2f}<extra>antialias=True</extra>",
    ),
    row=2,
    col=1,
)
fig.add_hline(
    y=ideal,
    line=dict(color="#7a7a7a", width=2, dash="dot"),
    annotation_text=f"균일할 때 {ideal:.2f}",
    annotation_position="top right",
    row=2,
    col=1,
)
fig.add_hline(y=0, line=dict(color="#b0b0b0", width=1), row=2, col=1)

fig.update_xaxes(title_text="입력 격자 열 인덱스 (0-15)", dtick=1, row=2, col=1)
fig.update_yaxes(title_text="gradient 합", row=2, col=1)
fig.update_layout(
    title="16×16 → 7×7 bicubic에 단위 gradient를 흘렸을 때 (Registers 논문 부록 A 재현)",
    template="plotly_white",
    bargap=0.25,
    width=1150,
    height=760,
    legend=dict(orientation="h", y=-0.13, x=0.5, xanchor="center"),
    font=dict(size=12),
)

_show(fig)

png_path = os.path.join(HERE, "expy.png")
fig.write_image(png_path, scale=2)  # kaleido 필요
print("saved:", png_path)
# 출력: saved: /home/sungwoo/projects/swcho/dinov2/fm/papers/.fm/hints/2cb5032e-a123-4363-9553-7091e0b0ccb3/expy.png

# %% [markdown]
# ## 5. 정리
#
# | | antialias=False (원조 DINOv2) | antialias=True (registers 논문) |
# |---|---|---|
# | 셀 gradient 범위 | $-0.14 \sim 0.98$ (부호 반전) | $0.13 \sim 0.24$ |
# | 열별 합 표준편차 | 2.86 | 0.25 (**11.6배** 차이) |
# | 패턴 | 배율 $16/7\approx2.29$ 주기의 줄무늬 | 균일 |
#
# **답**: 부록 A의 outlier 위치 분포에 보이는 수직 줄무늬는 모델이 학습한 "의미 있는" 구조가 아니라,
# 원래 DINOv2 구현이 position embedding을 $16\times16 \to 7\times7$로 **antialiasing 없이** bicubic
# 보간한 데서 온 **구현 아티팩트**다. 그 보간에 단위 gradient를 흘려보내면 똑같은 줄무늬가 재현된다.
# 논문 저자들은 이후 실험(Tables 2a, 3)에서 항상 antialiasing을 켰고, 그러자 줄무늬가 사라진
# outlier 분포(Fig. 10 right)를 얻었다 — 남은 건 "테두리 쪽에 outlier가 몰린다"는 진짜 현상뿐이다.
#
# 주의: antialiasing은 **줄무늬**만 없앤다. high-norm outlier 자체는 그대로 남고,
# 그걸 없애는 것이 이 논문의 본론인 register token이다.
