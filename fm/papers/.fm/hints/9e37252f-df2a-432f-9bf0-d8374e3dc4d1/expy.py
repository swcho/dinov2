# %% [markdown]
# # DINOv2 ViT-g 백본의 파라미터 수 — 직접 세어 보기
#
# **카드 요지**: DINOv2 ViT-g/14 백본은 **약 11억(1.1B) 파라미터**이며,
# embedding dim **1536**, **24 heads**(64 dim/head), **40 blocks**, **SwiGLU FFN** 구성이다
# (논문 5장 "Fast and memory-efficient attention" 절, 부록 Table 17).
#
# 이 스크립트는 논문 Table 17의 숫자와 이 저장소의 실제 구현
# (`dinov2/models/vision_transformer.py`의 `vit_giant2`, `dinov2/layers/swiglu_ffn.py`)을
# 근거로 파라미터 수를 **numpy 없이 산술만으로** 재현한다. torch 불필요.
#
# 필요 패키지: plotly, kaleido (그래프 저장용; 없으면 계산 부분만 실행됨)

# %%
import os

def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# %% [markdown]
# ## 1. 블록 하나의 파라미터 수
#
# ViT 블록 하나는 다음으로 구성된다 (DINOv2 구현 기준, bias 포함):
#
# - **Attention**: $W_{qkv} \in \mathbb{R}^{D \times 3D}$, $W_{proj} \in \mathbb{R}^{D \times D}$
#   $\Rightarrow 4D^2 + 4D$
# - **LayerNorm × 2**: 각 $2D$ $\Rightarrow 4D$
# - **LayerScale × 2**: 각 $D$ $\Rightarrow 2D$
# - **FFN**: 아래 두 종류 중 하나
#   - 일반 MLP (hidden $H = 4D$): $W_1 \in \mathbb{R}^{D \times H}$, $W_2 \in \mathbb{R}^{H \times D}$
#     $\Rightarrow 2DH + H + D$
#   - **SwiGLU** (DINOv2 `SwiGLUFFNFused`): $\text{SwiGLU}(x) = (\text{SiLU}(xW_1) \odot xW_2)\,W_3$.
#     게이트 때문에 행렬이 3개라, 파라미터 수를 MLP와 맞추기 위해 hidden을
#     $H' = \lfloor \tfrac{2}{3}\cdot 4D \rfloor$ (8의 배수로 정렬)로 줄인다.
#     $\Rightarrow 2DH' + 2H' + H'D + D = 3DH' + 2H' + D$
#
# head 수는 파라미터 수에 **영향을 주지 않는다** — $D$를 몇 조각으로 나눠 attention을 계산하느냐의 문제일 뿐,
# $W_{qkv}$의 크기는 $D \times 3D$로 동일하다.

# %%
def attn_params(D):
    return (D * 3 * D + 3 * D) + (D * D + D)          # qkv + proj (bias 포함)

def mlp_params(D, mlp_ratio=4):
    H = int(mlp_ratio * D)
    return (D * H + H) + (H * D + D)                    # fc1 + fc2

def swiglu_hidden(D, mlp_ratio=4):
    # dinov2/layers/swiglu_ffn.py SwiGLUFFNFused: hidden = (int(hidden*2/3) + 7) // 8 * 8
    H = int(mlp_ratio * D)
    return (int(H * 2 / 3) + 7) // 8 * 8

def swiglu_params(D, mlp_ratio=4):
    Hp = swiglu_hidden(D, mlp_ratio)
    w12 = D * (2 * Hp) + 2 * Hp                         # W1, W2를 하나의 Linear(D, 2H')로 융합
    w3 = Hp * D + D
    return w12 + w3

def block_params(D, ffn="swiglu", mlp_ratio=4):
    norms = 2 * (2 * D)          # LayerNorm(weight, bias) × 2
    layerscale = 2 * D           # LayerScale gamma × 2
    ffn_p = swiglu_params(D, mlp_ratio) if ffn == "swiglu" else mlp_params(D, mlp_ratio)
    return attn_params(D) + norms + layerscale + ffn_p

D = 1536
print("ViT-g  D =", D, " heads = 24  → dim/head =", D // 24)
print("SwiGLU hidden H' =", swiglu_hidden(D), "(MLP였다면 4D =", 4 * D, ")")
print(f"attention      : {attn_params(D):>13,}")
print(f"SwiGLU FFN     : {swiglu_params(D):>13,}")
print(f"(참고) MLP FFN : {mlp_params(D):>13,}")
print(f"block (SwiGLU) : {block_params(D):>13,}")
# 출력:
# ViT-g  D = 1536  heads = 24  → dim/head = 64
# SwiGLU hidden H' = 4096 (MLP였다면 4D = 6144 )
# attention      :     9,443,328
# SwiGLU FFN     :    18,884,096
# (참고) MLP FFN :    18,882,048
# block (SwiGLU) :    28,336,640

# %% [markdown]
# SwiGLU와 MLP의 FFN 파라미터 수가 거의 같다는 점에 주목 — hidden을 $\tfrac{2}{3}$로 줄이는 것이
# 바로 "파라미터 수를 유지하면서 게이트를 넣는" 트릭이다 (Shazeer, 2020).
#
# ## 2. 전체 백본 = 40 blocks + 임베딩/토큰 + 최종 norm
#
# - Patch embedding (conv $14\times14$, 3채널 → $D$): $3\cdot14^2 D + D$
# - cls token $D$, mask token $D$
# - positional embedding: 학습 해상도 518px, 패치 14 → $37^2 + 1 = 1370$ 토큰 $\Rightarrow 1370 D$
# - 최종 LayerNorm: $2D$

# %%
def backbone_params(D, depth, ffn="swiglu", patch=14, img=518, mlp_ratio=4):
    n_tok = (img // patch) ** 2 + 1
    patch_embed = 3 * patch * patch * D + D
    tokens = D + D + n_tok * D                         # cls + mask + pos_embed
    final_norm = 2 * D
    return depth * block_params(D, ffn, mlp_ratio) + patch_embed + tokens + final_norm

total_g = backbone_params(1536, 40)
print(f"ViT-g/14 (DINOv2) 총 파라미터: {total_g:,}  ≈ {total_g/1e9:.2f}B")
print(f"  그 중 40 blocks           : {40*block_params(1536):,}  ({40*block_params(1536)/total_g:.1%})")
# 출력:
# ViT-g/14 (DINOv2) 총 파라미터: 1,136,480,768  ≈ 1.14B
#   그 중 40 blocks           : 1,133,465,600  (99.7%)

# %% [markdown]
# 논문이 말하는 **"1.1B parameters"**가 그대로 나온다. 파라미터의 99.7%는 40개 블록에 있고,
# 그 안에서 FFN이 약 2/3, attention이 약 1/3을 차지한다.
#
# ## 3. Zhai et al.(2022)의 원래 ViT-g와 비교
#
# 원래 ViT-g는 $D=1408$, 16 heads(88 dim/head), 40 blocks, MLP hidden 6144다.
# DINOv2는 FlashAttention 효율을 위해 **dim/head가 64의 배수, $D$가 256의 배수**가 되도록
# $D=1536$, 24 heads로 바꿨다. 그 대가로 파라미터가 약 12% 늘어난다.

# %%
def zhai_vitg_params(patch=14, img=224):
    D, depth, H = 1408, 40, 6144
    attn = attn_params(D)
    ffn = (D * H + H) + (H * D + D)
    blk = attn + 4 * D + ffn                          # LayerScale 없음(원본 ViT)
    n_tok = (img // patch) ** 2 + 1
    return depth * blk + (3 * patch * patch * D + D) + D + n_tok * D + 2 * D

zhai = zhai_vitg_params()
print(f"Zhai ViT-g/14 (1408, 16 heads, MLP 6144): {zhai:,} ≈ {zhai/1e9:.2f}B")
print(f"DINOv2 ViT-g/14 (1536, 24 heads, SwiGLU): {total_g:,} ≈ {total_g/1e9:.2f}B")
print(f"증가율: {(total_g/zhai - 1):.1%}")
print("dim/head: Zhai 1408/16 =", 1408 / 16, "(64의 배수 아님) vs DINOv2 1536/24 =", 1536 / 24)
print("D mod 256: Zhai", 1408 % 256, "vs DINOv2", 1536 % 256)
# 출력:
# Zhai ViT-g/14 (1408, 16 heads, MLP 6144): 1,011,202,432 ≈ 1.01B
# DINOv2 ViT-g/14 (1536, 24 heads, SwiGLU): 1,136,480,768 ≈ 1.14B
# 증가율: 12.4%
# dim/head: Zhai 1408/16 = 88.0 (64의 배수 아님) vs DINOv2 1536/24 = 64.0
# D mod 256: Zhai 128 vs DINOv2 0

# %% [markdown]
# ## 4. DINOv2 모델 패밀리 전체 (Table 17)
#
# | Arch | Embed dim | Heads | Blocks | FFN |
# |---|---|---|---|---|
# | ViT-S/14 | 384 | 6 | 12 | MLP |
# | ViT-B/14 | 768 | 12 | 18 | MLP |
# | ViT-L/14 | 1024 | 16 | 24 | MLP (distilled) / SwiGLU (scratch) |
# | ViT-g/14 | 1536 | 24 | 40 | SwiGLU |
#
# 파라미터 수는 대략 $\text{depth}\times 12D^2$로 스케일한다 (attention $4D^2$ + FFN $8D^2$).

# %%
family = [
    ("ViT-S/14", 384, 12, "mlp"),
    ("ViT-B/14", 768, 18, "mlp"),
    ("ViT-L/14", 1024, 24, "mlp"),
    ("ViT-g/14", 1536, 40, "swiglu"),
]
rows = []
for name, d, depth, ffn in family:
    p = backbone_params(d, depth, ffn)
    approx = depth * 12 * d * d
    rows.append((name, d, depth, ffn, p))
    print(f"{name}: D={d:>5} depth={depth:>2} {ffn:6s} → {p/1e6:8.1f}M  (근사 12·depth·D² = {approx/1e6:7.1f}M)")
# 출력:
# ViT-S/14: D=  384 depth=12 mlp    →     22.1M  (근사 12·depth·D² =    21.2M)
# ViT-B/14: D=  768 depth=18 mlp    →    129.1M  (근사 12·depth·D² =   127.4M)
# ViT-L/14: D= 1024 depth=24 mlp    →    304.4M  (근사 12·depth·D² =   302.0M)
# ViT-g/14: D= 1536 depth=40 swiglu →   1136.5M  (근사 12·depth·D² =  1132.5M)

# %% [markdown]
# 공개된 체크포인트 크기(S 21M / B 86M / L 300M / g 1.1B)와 일치한다
# (B는 논문 Table 17의 18 blocks 기준이면 129M, 공개 hub 모델은 12 blocks라 86M).
#
# ## 5. 시각화 — 모델별 파라미터 구성(attention / FFN / 기타)

# %%
try:
    import plotly.graph_objects as go

    names = [r[0] for r in rows]
    attn_tot = [r[2] * attn_params(r[1]) / 1e6 for r in rows]
    ffn_tot = [r[2] * (swiglu_params(r[1]) if r[3] == "swiglu" else mlp_params(r[1])) / 1e6 for r in rows]
    other = [r[4] / 1e6 - a - f for r, a, f in zip(rows, attn_tot, ffn_tot)]

    fig = go.Figure()
    fig.add_bar(name="Attention (4D²)", x=names, y=attn_tot, marker_color="#4C78A8")
    fig.add_bar(name="FFN (≈8D²)", x=names, y=ffn_tot, marker_color="#F58518")
    fig.add_bar(name="기타 (embed/norm/token)", x=names, y=other, marker_color="#B0B0B0")
    for r in rows:
        fig.add_annotation(x=r[0], y=r[4] / 1e6, text=f"{r[4]/1e6:,.0f}M", showarrow=False, yshift=12)
    fig.update_layout(
        barmode="stack",
        title="DINOv2 백본 파라미터 수 (Table 17 구성으로 계산) — ViT-g ≈ 1.1B",
        yaxis_title="파라미터 수 (M)",
        xaxis_title="D / heads / blocks:  S 384/6/12 · B 768/12/18 · L 1024/16/24 · g 1536/24/40",
        legend=dict(orientation="h", y=1.08),
        width=820, height=520,
    )
    _show(fig)
    out = os.path.join(HERE, "expy.png")
    fig.write_image(out, scale=2)
    print("saved:", out)
except ImportError as e:
    print("plotly/kaleido 없음 — 시각화 생략:", e)
# 출력:
# saved: /home/sungwoo/projects/swcho/dinov2/fm/papers/.fm/hints/9e37252f-df2a-432f-9bf0-d8374e3dc4d1/expy.png

# %% [markdown]
# ## 정리
#
# - DINOv2 ViT-g/14: $D=1536$, 24 heads (64 dim/head), 40 blocks, SwiGLU FFN → **≈1.14B ≈ "1.1B"**
# - head 수는 파라미터 수와 무관; $D$와 depth가 결정한다 ($\approx 12\,\text{depth}\,D^2$)
# - 1408→1536으로 키운 이유는 정확도가 아니라 **FlashAttention 하드웨어 효율**(64 dim/head, $D$가 256의 배수)
# - SwiGLU는 hidden을 $\tfrac{2}{3}$로 줄여 MLP와 파라미터 수를 거의 같게 맞춘다
