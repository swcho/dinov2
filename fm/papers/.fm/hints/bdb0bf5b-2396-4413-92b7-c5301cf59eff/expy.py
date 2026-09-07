# %% [markdown]
# # outlier token이 global 정보를 담고 있음을 보이는 실험 (합성 데이터 재현)
#
# 논문: *Vision Transformers Need Registers* (Darcet et al., ICLR 2024), Sec. 2.1 "Artifacts hold global information", Table 1 / Appendix G Table 6.
#
# **논문의 프로토콜**
#
# 1. 분류 데이터셋의 각 이미지를 DINOv2-g에 통과시켜 patch embedding을 뽑는다.
# 2. 그 중 **단 하나의 토큰**을 무작위로 고른다. 이때 `norm > 150`이면 high-norm(outlier), 아니면 normal.
# 3. 그 토큰 하나를 **이미지 표현**으로 삼아 로지스틱 회귀 분류기를 학습하고 정확도를 잰다.
# 4. 비교군으로 `[CLS]` 토큰으로도 같은 분류기를 학습한다.
#
# 결과(Aircraft): normal **17.1** / outlier **79.1** / `[CLS]` **87.3**.
# 단일 패치 토큰인데도 outlier 쪽이 `[CLS]`에 육박한다 → outlier 토큰이 이미지 전역(global) 정보를 들고 있다.
#
# 이 스크립트는 실제 DINOv2 가중치 없이, 위 논리 구조만 갖는 **토이 데이터셋**으로 같은 패턴을 재현한다.
#
# 의존성 / 실행 인터프리터:
# `/home/sungwoo/miniforge3/envs/trellis/bin/python`  (numpy 1.26, scikit-learn 1.7, plotly 6.9, kaleido)

# %%
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
print("numpy", np.__version__)
# 출력: numpy 1.26.4

# %% [markdown]
# ## 1. 토이 생성 모델
#
# 클래스 $c$마다 단위 벡터 global 신호 $g_c \in \mathbb{R}^D$ 를 하나씩 둔다.
# 이미지 하나는 $P$개의 patch 토큰으로 이루어지고, 각 토큰은
#
# $$t_i = s_i\,\bigl(\kappa_i\, g_c + \eta_i\bigr),\qquad \eta_i \sim \mathcal{N}(0,\, I_D)$$
#
# 로 만든다. $\eta_i$는 패치별 국소 정보/noise(기대 norm $\approx \sqrt{D} = 8$)이고,
# $\kappa_i$는 그 토큰에 **global 신호가 얼마나 실려 있는지**를 정한다.
#
# | 토큰 종류 | $\kappa$ | 스케일 $s$ | 의미 |
# |---|---|---|---|
# | normal patch | $1.05$ | $1$ | 국소 정보 위주, 클래스 정보 희박 |
# | outlier patch | $3.10$ | $10$ | 국소 정보를 버리고 global 정보를 적재, norm이 ~10배 |
# | `[CLS]` | $3.60$ | $1$ | 설계상 global 요약 |
#
# 논문 Fig. 3의 "대부분 norm 0~100, 2.37%만 매우 큰 norm" 을 흉내내기 위해
# outlier 비율은 $\approx 2.4\%$, 스케일은 10배로 잡았다.
# 주의: 로지스틱 회귀는 (정규화만 맞추면) 스케일에 거의 불변하므로,
# **정확도 차이를 만드는 것은 norm 자체가 아니라 $\kappa$ = global 신호 대 국소 noise 비(SNR)** 다.

# %%
D = 64  # 토큰 차원
N_CLASSES = 20  # 클래스 수 (chance = 5%)
P = 128  # 이미지당 patch 토큰 수
OUTLIER_RATIO = 0.024  # 논문의 2.37%와 비슷하게

KAPPA = {"normal": 1.05, "outlier": 3.10, "cls": 3.60}
SCALE = {"normal": 1.0, "outlier": 10.0, "cls": 1.0}

N_TRAIN, N_TEST = 3000, 1500


def make_class_signals(rng):
    g = rng.normal(size=(N_CLASSES, D))
    return g / np.linalg.norm(g, axis=1, keepdims=True)  # 클래스별 단위 global 벡터


def make_tokens(kind, labels, g, rng):
    """labels(=클래스 인덱스) 각각에 대해 지정한 종류의 토큰 1개씩 생성."""
    n = len(labels)
    noise = rng.normal(size=(n, D))
    return SCALE[kind] * (KAPPA[kind] * g[labels] + noise)


rng = np.random.default_rng(0)
g = make_class_signals(rng)
labels = rng.integers(0, N_CLASSES, size=8)
print(np.round(np.linalg.norm(make_tokens("normal", labels, g, rng), axis=1), 2))
print(np.round(np.linalg.norm(make_tokens("outlier", labels, g, rng), axis=1), 2))
# 출력: [8.39 8.83 8.09 6.59 8.03 9.13 8.89 7.59]
# 출력: [83.18 86.37 81.22 93.77 85.04 86.95 88.87 84.24]

# %% [markdown]
# ## 2. norm 분포가 bimodal 한지 확인 (논문 Fig. 3 오른쪽)
#
# 이미지 몇 장 분량의 토큰을 전부 만들어 $L_2$ norm 히스토그램을 그려 본다.
# 실제 DINOv2처럼 "대부분 낮은 norm + 소수의 매우 큰 norm" 두 봉우리가 나오고,
# 그 사이 어딘가에 컷오프(논문에선 150)를 그으면 outlier를 기계적으로 골라낼 수 있다.

# %%
def make_image_bank(n_images, rng):
    """이미지별 전체 토큰과 outlier 마스크를 생성."""
    labels = rng.integers(0, N_CLASSES, size=n_images)
    n_out = max(1, int(round(P * OUTLIER_RATIO)))
    tokens = np.empty((n_images, P, D), dtype=np.float64)
    is_out = np.zeros((n_images, P), dtype=bool)
    for i, c in enumerate(labels):
        idx = rng.choice(P, size=n_out, replace=False)
        is_out[i, idx] = True
        lab = np.full(P, c)
        tok = make_tokens("normal", lab, g, rng)
        tok[idx] = make_tokens("outlier", lab[idx], g, rng)
        tokens[i] = tok
    return tokens, is_out, labels


bank_tokens, bank_is_out, _ = make_image_bank(200, np.random.default_rng(1))
norms = np.linalg.norm(bank_tokens, axis=-1).ravel()
flag = bank_is_out.ravel()
CUTOFF = 30.0  # 두 봉우리 사이 (논문의 150에 해당하는 hand-picked 값)
print(f"outlier 비율: {flag.mean():.4f}")
print(f"norm > {CUTOFF} 비율: {(norms > CUTOFF).mean():.4f}")
print(f"normal  norm 평균 {norms[~flag].mean():.2f} / outlier norm 평균 {norms[flag].mean():.2f}")
# 출력: outlier 비율: 0.0234
# 출력: norm > 30.0 비율: 0.0234
# 출력: normal  norm 평균 8.03 / outlier norm 평균 85.03

# %% [markdown]
# ## 3. 핵심 실험: 토큰 1개 → 로지스틱 회귀
#
# 이미지마다 **랜덤 토큰 1개**를 뽑아 그것만으로 20-way 분류기를 학습한다.
# 세 조건을 비교한다.
#
# - `normal`: 무작위 normal patch 토큰 1개
# - `outlier`: 무작위 outlier patch 토큰 1개
# - `[CLS]`: global 신호 그 자체에 해당하는 토큰
#
# 랜덤 토큰 선택 자체가 분산을 만들기 때문에(논문 Appendix G),
# 시드를 바꿔 가며 여러 번 반복하고 평균 ± 표준편차를 본다.

# %%
def run_once(seed):
    r = np.random.default_rng(seed)
    y_tr = r.integers(0, N_CLASSES, size=N_TRAIN)
    y_te = r.integers(0, N_CLASSES, size=N_TEST)
    accs = {}
    for kind in ("normal", "outlier", "cls"):
        X_tr = make_tokens(kind, y_tr, g, r)
        X_te = make_tokens(kind, y_te, g, r)
        # 스케일 차이가 정규화 세기를 바꾸지 않도록 표준화(실제 linear probing과 동일한 관행)
        mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit((X_tr - mu) / sd, y_tr)
        accs[kind] = clf.score((X_te - mu) / sd, y_te) * 100
    return accs


SEEDS = [0, 1, 2, 3, 4]
runs = [run_once(s) for s in SEEDS]
res = {k: np.array([r[k] for r in runs]) for k in ("normal", "outlier", "cls")}
for k, v in res.items():
    print(f"{k:8s} {v.mean():5.1f} ± {v.std():.1f}")
# 출력: normal    18.8 ± 1.2
# 출력: outlier   78.1 ± 1.0
# 출력: cls       88.0 ± 1.0

# %% [markdown]
# 논문 Table 1(Aircraft 열)의 **normal 17.1 / outlier 79.1 / [CLS] 87.3** 과 같은 패턴이 나온다.
#
# - normal 토큰: chance(5%)보다는 낫지만 형편없다 → 국소 정보만 들고 있다.
# - outlier 토큰: **단일 패치인데도** `[CLS]`에 근접 → 이미지 전역 정보를 담고 있다.
# - 이 격차가 "artifacts hold global information" 주장의 근거다.

# %% [markdown]
# ## 4. 무엇이 정확도를 만드는가: norm이 아니라 SNR
#
# 흔한 오해는 "norm이 크니까 분류가 잘 되는 것 아니냐" 인데, 그렇지 않다.
# outlier 토큰의 스케일 $s$만 바꾸고 $\kappa$를 고정하면 정확도는 거의 변하지 않는다.
# 반대로 $\kappa$(global 신호 비중)를 바꾸면 정확도가 확 움직인다.

# %%
_orig_scale = SCALE["outlier"]
for s in (1.0, 10.0, 100.0):
    SCALE["outlier"] = s
    print(f"scale={s:6.1f}  acc={run_once(0)['outlier']:.1f}")
SCALE["outlier"] = _orig_scale
# 출력: scale=   1.0  acc=79.9
# 출력: scale=  10.0  acc=79.9
# 출력: scale= 100.0  acc=79.9

_orig_kappa = KAPPA["outlier"]
for kap in (1.05, 2.0, 3.10, 5.0):
    KAPPA["outlier"] = kap
    print(f"kappa={kap:4.2f}  acc={run_once(0)['outlier']:.1f}")
KAPPA["outlier"] = _orig_kappa
# 출력: kappa=1.05  acc=17.4
# 출력: kappa=2.00  acc=48.1
# 출력: kappa=3.10  acc=79.9
# 출력: kappa=5.00  acc=99.5

# %% [markdown]
# 즉 큰 norm은 **표지(marker)** 일 뿐이고, 실제로 정확도를 올리는 것은
# 그 토큰에 얼마나 많은 global 정보가 실렸는가다.
# 논문이 norm > 150 컷오프를 쓰는 이유도 "norm이 원인" 이라서가 아니라
# **outlier를 값싸게 식별하는 기준** 이기 때문이다.

# %% [markdown]
# ## 5. 뒷면: outlier는 국소 정보를 잃었다 (논문 Fig. 5b)
#
# 같은 토이 모델에 patch 위치 $p_i$ 성분을 넣어 보면, global 정보를 많이 실은 토큰일수록
# 위치 예측(=국소 정보) 정확도는 떨어진다. 논문의 position prediction top-1
# (normal 41.7 vs outlier 22.8)과 같은 방향이다.

# %%
POS = 16  # 위치 클래스 수 (4x4 격자라고 생각)
pos_basis = np.random.default_rng(7).normal(size=(POS, D))
pos_basis /= np.linalg.norm(pos_basis, axis=1, keepdims=True)

# 토큰 종류별로 "국소(위치) 성분을 얼마나 남겨 두는가". outlier는 이걸 버린 상태다.
LOCAL_W = {"normal": 1.60, "outlier": 1.05}


def run_position_probe(kind, seed=0):
    r = np.random.default_rng(seed)
    out = {}
    for split, n in (("tr", N_TRAIN), ("te", N_TEST)):
        y_cls = r.integers(0, N_CLASSES, size=n)
        y_pos = r.integers(0, POS, size=n)
        X = KAPPA[kind] * g[y_cls] + LOCAL_W[kind] * pos_basis[y_pos] + r.normal(size=(n, D))
        out[split] = (X, y_pos)
    clf = LogisticRegression(max_iter=2000).fit(*out["tr"])
    return clf.score(*out["te"]) * 100


for kind in ("normal", "outlier"):
    print(f"{kind:8s} position top-1 = {run_position_probe(kind):.1f}")
# 출력: normal   position top-1 = 41.9
# 출력: outlier  position top-1 = 21.7

# %% [markdown]
# ## 6. 시각화
#
# 왼쪽: 토큰 norm 분포(bimodal, 로그 y축) — outlier를 컷오프로 분리할 수 있다.
# 오른쪽: 토큰 1개짜리 linear probing 정확도(5 시드 평균 ± 표준편차)와 논문 Table 1(Aircraft) 값.

# %%
COLORS = {"normal": "#5B8FF9", "outlier": "#E8684A", "cls": "#5AD8A6"}

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "토큰 L2 norm 분포 (논문 Fig. 3 재현)",
        "토큰 1개 linear probing 정확도 (논문 Table 1 재현)",
    ),
)

fig.add_trace(
    go.Histogram(
        x=norms[~flag], name="normal", marker_color=COLORS["normal"], nbinsx=60, opacity=0.85
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Histogram(
        x=norms[flag], name="outlier", marker_color=COLORS["outlier"], nbinsx=60, opacity=0.85
    ),
    row=1,
    col=1,
)
fig.add_vline(
    x=CUTOFF,
    line_dash="dash",
    line_color="#888",
    annotation_text=f"cutoff={CUTOFF}",
    row=1,
    col=1,
)

names = ["normal", "outlier", "[CLS]"]
keys = ["normal", "outlier", "cls"]
means = [res[k].mean() for k in keys]
stds = [res[k].std() for k in keys]
paper = [17.1, 79.1, 87.3]  # 논문 Table 1 / Table 6, Aircraft

fig.add_trace(
    go.Bar(
        x=names,
        y=means,
        error_y=dict(type="data", array=stds),
        name="toy 재현",
        marker_color=[COLORS[k] for k in keys],
        width=0.55,
        text=[f"{m:.1f}" for m in means],
        textposition="inside",
        insidetextanchor="start",
        textfont=dict(color="white", size=14),
        showlegend=False,
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=names,
        y=paper,
        mode="markers",
        name="논문 값 (Aircraft)",
        marker=dict(symbol="diamond", size=13, color="#262626"),
    ),
    row=1,
    col=2,
)
fig.add_hline(
    y=100 / N_CLASSES, line_dash="dot", line_color="#aaa", annotation_text="chance", row=1, col=2
)

fig.update_xaxes(title_text="L2 norm", row=1, col=1)
fig.update_yaxes(title_text="count (log)", type="log", row=1, col=1)
fig.update_yaxes(title_text="top-1 accuracy (%)", range=[0, 105], row=1, col=2)
fig.update_layout(
    title="outlier token은 global 정보를 담는다: 단일 토큰 linear probing",
    barmode="overlay",
    bargap=0.05,
    template="plotly_white",
    width=1000,
    height=430,
    legend=dict(orientation="h", y=-0.18),
)

_show(fig)
fig.write_image(os.path.join(HERE, "expy.png"), scale=2)
print("saved:", os.path.join(HERE, "expy.png"))
# 출력: saved: /home/sungwoo/projects/swcho/dinov2/fm/papers/.fm/hints/bdb0bf5b-2396-4413-92b7-c5301cf59eff/expy.png

# %% [markdown]
# ## 정리
#
# - 실험 설계: DINOv2-g → patch embedding → **norm 기준으로 high-norm/normal 분류** →
#   이미지당 **랜덤 토큰 1개**를 표현으로 삼아 **로지스틱 회귀** 학습 → 정확도 비교.
# - 결과: outlier ≫ normal, 그리고 outlier ≈ `[CLS]`.
#   단일 패치 토큰이 이미지 전체를 알아맞힌다는 것은 그 토큰이 **global 정보를 적재**하고 있다는 뜻.
# - 반대편 증거(Fig. 5b): outlier는 위치 예측/픽셀 복원 성능이 낮다 → **국소 정보는 버렸다**.
# - 종합 해석: 큰 모델은 정보량이 적은(이웃과 중복된) 패치를 골라 그 토큰을 재활용해
#   global 정보를 저장·처리하는 **register**로 쓴다. 이것이 register 토큰을 명시적으로
#   추가하자는 제안의 근거가 된다.
