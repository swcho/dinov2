# %% [markdown]
# # Instance recognition을 non-parametric 방식으로 평가하기 — 코사인 유사도 랭킹과 mAP
#
# DINOv2 논문 7.3절의 instance recognition 평가 절차를 장난감(toy) 문제로 재현한다.
#
# 1. 쿼리 이미지·데이터베이스 이미지의 **frozen feature**(여기서는 난수로 합성)를 준비
# 2. 특징을 L2 정규화 → **코사인 유사도** = 내적
# 3. 유사도 내림차순으로 데이터베이스를 **랭킹** (학습 파라미터 없음 = non-parametric)
# 4. 랭킹의 정답 위치로부터 **AP**(average precision)를 계산, 쿼리 평균 → **mAP**
# 5. `sklearn.metrics.average_precision_score`와 대조, PR 곡선을 plotly로 시각화
#
# 필요 패키지: numpy, plotly, kaleido (expy.png 저장용), scikit-learn (검증용, 선택)

# %%
import numpy as np


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(4)

# %% [markdown]
# ## 1. 장난감 데이터베이스 만들기
#
# 3개의 랜드마크(인스턴스) `A, B, C`가 있다고 하자. 각 인스턴스는 특징 공간에서 하나의
# 원형(prototype) 방향을 갖고, 같은 인스턴스를 찍은 사진들은 그 방향 근처에 노이즈를 섞어 놓인다.
# 데이터베이스에는 인스턴스별 양성(positive) 이미지와, 어떤 쿼리에도 해당하지 않는 distractor를 넣는다.
#
# 특징 차원 $d=16$, 데이터베이스 크기 $N=14$ (A 4장, B 3장, C 2장, distractor 5장).

# %%
D = 16
instances = ["A", "B", "C"]
n_pos = {"A": 4, "B": 3, "C": 2}
n_distractor = 5


def make_split(noise, rng):
    """쿼리 3개 + 데이터베이스 특징/라벨 생성. noise = 원형(단위 벡터) 대비 노이즈 벡터의 기대 크기(클수록 특징 품질이 나쁨)."""
    proto = rng.normal(size=(len(instances), D))
    proto /= np.linalg.norm(proto, axis=1, keepdims=True)
    db_feats, db_labels = [], []
    for i, name in enumerate(instances):
        for _ in range(n_pos[name]):
            db_feats.append(proto[i] + noise * rng.normal(size=D) / np.sqrt(D))
            db_labels.append(name)
    for _ in range(n_distractor):
        db_feats.append(rng.normal(size=D))
        db_labels.append("-")
    q_feats = proto + noise * rng.normal(size=proto.shape) / np.sqrt(D)  # 쿼리도 같은 인스턴스의 다른 사진
    return np.array(q_feats), np.array(db_feats), np.array(db_labels)


q_raw, db_raw, db_labels = make_split(noise=1.2, rng=rng)
print("query shape:", q_raw.shape, " database shape:", db_raw.shape)
print("database labels:", " ".join(db_labels))
# 출력:
# query shape: (3, 16)  database shape: (14, 16)
# database labels: A A A A B B B C C - - - - -

# %% [markdown]
# ## 2. L2 정규화 → 코사인 유사도
#
# 코사인 유사도는
# $$\cos(q, x) = \frac{q \cdot x}{\|q\|\,\|x\|}$$
# 이므로, 미리 $\hat q = q/\|q\|$, $\hat x = x/\|x\|$로 정규화해 두면 **행렬곱 하나**로 모든
# (쿼리, DB) 쌍의 유사도를 얻는다. 이 단계에 학습되는 파라미터는 전혀 없다 — 특징 추출기(backbone)
# 위에 분류기·투영층을 얹지 않으므로, 점수는 **특징 자체가 얼마나 좋은지**를 그대로 반영한다.


# %%
def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


q, db = l2n(q_raw), l2n(db_raw)
S = q @ db.T  # (3 queries, 14 db)
print("cosine similarity matrix (rows=query A,B,C / cols=db):")
print(S)
# 출력:
# cosine similarity matrix (rows=query A,B,C / cols=db):
# [[ 0.323  0.521  0.198  0.507  0.1   -0.148 -0.271 -0.022  0.469  0.03   0.083  0.075  0.02   0.383]
#  [-0.427 -0.063  0.065 -0.062  0.398  0.327  0.514  0.177 -0.086  0.165 -0.248 -0.06  -0.085 -0.079]
#  [-0.602 -0.299 -0.07  -0.382  0.175  0.287  0.377  0.466  0.039 -0.156 -0.469  0.036 -0.234 -0.04 ]]
# (열 0~3 = A, 4~6 = B, 7~8 = C, 9~13 = distractor. 대각 블록이 크지만 쿼리 A가 C 이미지(열 8)와도 0.469로 꽤 비슷함)

# %% [markdown]
# ## 3. 랭킹 — 쿼리 A의 검색 결과
#
# 유사도 내림차순으로 데이터베이스를 정렬한다. 각 순위 $k$에서
# $$\text{precision@}k = \frac{\#\{\text{상위 } k \text{ 중 정답}\}}{k}, \qquad
# \text{recall@}k = \frac{\#\{\text{상위 } k \text{ 중 정답}\}}{\#\{\text{전체 정답}\}}$$


# %%
def rank_table(sims, labels, target):
    order = np.argsort(-sims)
    rel = (labels[order] == target).astype(int)
    hits = np.cumsum(rel)
    k = np.arange(1, len(order) + 1)
    prec = hits / k
    rec = hits / rel.sum()
    return order, rel, prec, rec


order, rel, prec, rec = rank_table(S[0], db_labels, "A")
print(" rank  db_idx  label  sim    rel  prec@k  recall@k")
for k in range(len(order)):
    j = order[k]
    print(f"{k+1:5d}  {j:6d}  {db_labels[j]:>5s}  {S[0, j]:.3f}   {rel[k]}   {prec[k]:.3f}   {rec[k]:.3f}")
# 출력:
#  rank  db_idx  label  sim    rel  prec@k  recall@k
#     1       1      A  0.521   1   1.000   0.250
#     2       3      A  0.507   1   1.000   0.500
#     3       8      C  0.469   0   0.667   0.500   <- 오답(C)이 3위에 끼어듦
#     4      13      -  0.383   0   0.500   0.500   <- distractor
#     5       0      A  0.323   1   0.600   0.750
#     6       2      A  0.198   1   0.667   1.000   <- 마지막 정답, recall 100%
#     7       4      B  0.100   0   0.571   1.000
#     8      10      -  0.083   0   0.500   1.000
#     9      11      -  0.075   0   0.444   1.000
#    10       9      -  0.030   0   0.400   1.000
#    11      12      -  0.020   0   0.364   1.000
#    12       7      C  -0.022   0   0.333   1.000
#    13       5      B  -0.148   0   0.308   1.000
#    14       6      B  -0.271   0   0.286   1.000

# %% [markdown]
# ## 4. AP(average precision) 정의와 계산
#
# 정답 개수를 $R$, $k$번째 결과가 정답이면 $\mathrm{rel}(k)=1$이라 할 때
# $$\mathrm{AP} = \frac{1}{R}\sum_{k=1}^{N} \mathrm{rel}(k)\,\text{precision@}k$$
# 즉 **정답이 등장하는 순위에서의 precision을 평균**한 값이다.
# 이는 PR 곡선 아래 면적 $\sum_k (\mathrm{recall}_k - \mathrm{recall}_{k-1})\,\text{precision}_k$과 동일하다
# (정답 하나가 recall을 $1/R$씩 올리므로).
#
# - 정답이 모두 맨 위에 오면 AP = 1.0
# - 정답이 뒤로 밀릴수록 그 순위의 precision이 낮아져 AP가 떨어진다.
#
# $Q$개 쿼리의 AP를 평균한 것이 **mAP** $= \frac{1}{Q}\sum_q \mathrm{AP}_q$.


# %%
def average_precision(sims, labels, target):
    order, rel, prec, rec = rank_table(sims, labels, target)
    ap_sum = (rel * prec).sum() / rel.sum()                 # 정의 1: 정답 위치 precision 평균
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    ap_area = ((rec - rec_prev) * prec).sum()               # 정의 2: PR 곡선 아래 면적
    assert np.isclose(ap_sum, ap_area)
    return ap_sum


aps = {name: average_precision(S[i], db_labels, name) for i, name in enumerate(instances)}
for name, ap in aps.items():
    print(f"AP(query {name}) = {ap:.4f}")
print(f"mAP = {np.mean(list(aps.values())):.4f}")

try:
    from sklearn.metrics import average_precision_score
    sk = [average_precision_score((db_labels == n).astype(int), S[i]) for i, n in enumerate(instances)]
    print("sklearn AP:", np.round(sk, 4), "-> match:", np.allclose(sk, list(aps.values())))
except ImportError:
    print("sklearn 없음 - 검증 생략")
# 출력:
# AP(query A) = 0.8167    <- (1.000 + 1.000 + 0.600 + 0.667) / 4 : 정답 순위 1,2,5,6의 precision 평균
# AP(query B) = 1.0000
# AP(query C) = 0.7000
# mAP = 0.8389
# sklearn AP: [0.817 1.    0.7  ] -> match: True

# %% [markdown]
# ## 5. 정답이 뒤로 밀리는 경우 — AP가 어떻게 깎이는지 손으로 확인
#
# 랭킹을 직접 지정해서 AP를 계산해 본다. 정답 3개($R=3$), DB 6개.

# %%
def ap_from_ranking(rel):
    rel = np.asarray(rel, dtype=float)
    prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
    return (rel * prec).sum() / rel.sum(), prec


for rel in ([1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]):
    ap, prec = ap_from_ranking(rel)
    hit_prec = [f"{p:.2f}" for p, r in zip(prec, rel) if r]
    print(f"ranking {rel} -> precision at hits {hit_prec} -> AP = {ap:.3f}")
# 출력:
# ranking [1, 1, 1, 0, 0, 0] -> precision at hits ['1.00', '1.00', '1.00'] -> AP = 1.000
# ranking [1, 0, 1, 0, 1, 0] -> precision at hits ['1.00', '0.67', '0.60'] -> AP = 0.756
# ranking [0, 0, 0, 1, 1, 1] -> precision at hits ['0.25', '0.40', '0.50'] -> AP = 0.383

# %% [markdown]
# ## 6. 특징 품질 vs mAP — "Table 9"를 흉내 내기
#
# 같은 데이터베이스 구성에서 **노이즈(특징 품질)** 만 바꾸면서 mAP를 재면, non-parametric 평가가
# 왜 특징 자체의 품질을 드러내는지 볼 수 있다. 노이즈가 커질수록(= 같은 인스턴스의 사진들이
# 특징 공간에서 덜 뭉칠수록) 코사인 랭킹이 흐트러져 mAP가 떨어진다.
# 논문에서 DINOv2가 Oxford-Hard mAP를 SSL 대비 +41%p, OpenCLIP 대비 +34%p 앞선 것은,
# 랭킹만으로 평가했을 때 특징이 인스턴스를 그만큼 잘 분리한다는 뜻이다.


# %%
def mean_ap(noise, trials=200, seed=1):
    r = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        qf, dbf, lab = make_split(noise, r)
        qf, dbf = l2n(qf), l2n(dbf)
        Sm = qf @ dbf.T
        out.append(np.mean([average_precision(Sm[i], lab, n) for i, n in enumerate(instances)]))
    return float(np.mean(out))


noise_levels = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
map_by_noise = {nz: mean_ap(nz) for nz in noise_levels}
print(" noise   mAP")
for nz, m in map_by_noise.items():
    print(f"  {nz:4.1f}  {m:.3f}")
# 출력:
#  noise   mAP
#   0.2  1.000
#   0.4  1.000
#   0.6  0.995
#   0.8  0.963
#   1.0  0.883
#   1.5  0.674
#   2.0  0.541

# %% [markdown]
# ## 7. 시각화 — 쿼리별 PR 곡선과 노이즈-mAP 곡선
#
# 왼쪽: 노이즈를 크게 준(=품질 나쁜) 특징으로 다시 검색한 세 쿼리의 PR 곡선. 곡선 아래 면적이 AP.
# 오른쪽: 특징 노이즈에 따른 mAP.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

q2_raw, db2_raw, lab2 = make_split(noise=1.2, rng=np.random.default_rng(7))
S2 = l2n(q2_raw) @ l2n(db2_raw).T

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("쿼리별 Precision-Recall 곡선 (noise=1.2)", "특징 노이즈 vs mAP (200회 평균)"),
)
colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
for i, name in enumerate(instances):
    _, rel, prec, rec = rank_table(S2[i], lab2, name)
    ap = average_precision(S2[i], lab2, name)
    fig.add_trace(
        go.Scatter(x=np.concatenate([[0], rec]), y=np.concatenate([[1], prec]),
                   mode="lines+markers", line_shape="hv", name=f"query {name} (AP={ap:.2f})",
                   line=dict(color=colors[name])),
        row=1, col=1,
    )
fig.add_trace(
    go.Scatter(x=list(map_by_noise), y=list(map_by_noise.values()), mode="lines+markers",
               name="mAP", line=dict(color="#d62728")),
    row=1, col=2,
)
fig.update_xaxes(title_text="recall", range=[0, 1.02], row=1, col=1)
fig.update_yaxes(title_text="precision", range=[0, 1.05], row=1, col=1)
fig.update_xaxes(title_text="feature noise σ", row=1, col=2)
fig.update_yaxes(title_text="mAP", range=[0, 1.05], row=1, col=2)
fig.update_layout(
    title="Non-parametric instance retrieval: 코사인 랭킹 → AP/mAP",
    width=1100, height=450, template="plotly_white", legend=dict(orientation="h", y=-0.2),
)
_show(fig)

import os
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "expy.png")
fig.write_image(_out, scale=2)
print("saved:", _out)
# 출력:
# saved: /home/sungwoo/projects/swcho/dinov2/fm/papers/.fm/hints/bbff54aa-85c7-491c-a021-e2afb88f1a05/expy.png
