# %% [markdown]
# # register 토큰은 시퀀스 어디에 어떻게 들어가는가 — 최소 ViT로 직접 확인
#
# *Vision Transformers Need Registers* (arXiv:2309.16588) §2.2:
#
# > We add these tokens **after the patch embedding layer**, with a **learnable value**,
# > **similarly to the [CLS] token**. At the end of the vision transformer,
# > these tokens are **discarded**, and the [CLS] token and patch tokens are used
# > as image representations, as usual.
#
# 이 스크립트는 torch로 아주 작은 ViT를 직접 만들어 위 문장의 네 조각을 하나씩 shape으로 확인한다.
#
# 1. patch embedding → `[CLS] ‖ patch` 시퀀스와 pos_embed
# 2. `nn.Parameter`로 register를 만들어 `torch.cat` (→ 길이 $1+R+N$)
# 3. 블록 통과 후 register slice를 **버리는** forward
# 4. attention 행렬에서 register가 차지하는 행/열 시각화
# 5. register 개수에 따른 FLOP 오버헤드
#
# 실행 인터프리터 (torch 2.4 / numpy 1.26 / plotly 6.9 / kaleido):
# `/home/sungwoo/miniforge3/envs/trellis/bin/python`  — CPU만으로 동작한다.

# %%
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

torch.manual_seed(0)
DEVICE = torch.device("cpu")


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
# ## 1. patch embedding → `[CLS] ‖ patch`, 그리고 pos_embed
#
# 이미지 $x \in \mathbb{R}^{B\times 3\times H\times W}$ 를 patch size $P$ 의 conv로 자르면
# $N = (H/P)\cdot(W/P)$ 개의 토큰이 나온다. 여기에 학습 파라미터 `[CLS]` 하나를 앞에 붙이고
# 길이 $1+N$ 짜리 `pos_embed`를 더한다.
#
# 아래 크기는 전부 장난감 값이다: $H=W=16$, $P=4$ 이므로 $N=16$, $D=32$.

# %%
B, C, H, W, P, D = 2, 3, 16, 16, 4, 32
N = (H // P) * (W // P)

patch_embed = nn.Conv2d(C, D, kernel_size=P, stride=P)
cls_token = nn.Parameter(torch.zeros(1, 1, D))
pos_embed = nn.Parameter(torch.zeros(1, 1 + N, D))  # ← 길이에 register 몫이 없다
nn.init.normal_(cls_token, std=1e-6)  # DINOv2 init_weights 와 동일한 std
nn.init.trunc_normal_(pos_embed, std=0.02)

img = torch.randn(B, C, H, W)
x = patch_embed(img).flatten(2).transpose(1, 2)  # (B, N, D)
print("patch_embed 직후      :", tuple(x.shape))

x = torch.cat((cls_token.expand(B, -1, -1), x), dim=1)  # (B, 1+N, D)
print("[CLS] concat 후       :", tuple(x.shape))

x = x + pos_embed  # pos_embed는 CLS+patch 에만 더해진다
print("pos_embed 더한 후     :", tuple(x.shape), "| pos_embed:", tuple(pos_embed.shape))
# 출력: patch_embed 직후      : (2, 16, 32)
# 출력: [CLS] concat 후       : (2, 17, 32)
# 출력: pos_embed 더한 후     : (2, 17, 32) | pos_embed: (1, 17, 32)


# %% [markdown]
# ## 2. register 삽입 — `nn.Parameter` + `expand` + `cat`
#
# register는 **이미지에서 유도되지 않는다**. `nn.Parameter(torch.zeros(1, R, D))` 하나가
# 배치 차원으로 broadcast 될 뿐이라, 같은 배치의 두 이미지에 대해 입력 시점의 register 값은
# 완전히 동일하다. 그리고 pos_embed를 **이미 더한 뒤에** 붙이므로 위치 정보를 받지 않는다.
#
# DINOv2 `prepare_tokens_with_masks` 원본:
#
# ```python
# if self.register_tokens is not None:
#     x = torch.cat(
#         (
#             x[:, :1],                                          # [CLS]
#             self.register_tokens.expand(x.shape[0], -1, -1),   # [REG] x R
#             x[:, 1:],                                          # patches
#         ),
#         dim=1,
#     )
# ```

# %%
R = 4
register_tokens = nn.Parameter(torch.zeros(1, R, D))
nn.init.normal_(register_tokens, std=1e-6)  # [CLS]와 똑같은 초기화

x_reg = torch.cat(
    (
        x[:, :1],                             # [CLS]
        register_tokens.expand(B, -1, -1),    # [REG] x R
        x[:, 1:],                             # patches
    ),
    dim=1,
)
print(f"1 + R + N = 1 + {R} + {N} = {1 + R + N}")
print("register concat 후    :", tuple(x_reg.shape))
print("register 파라미터 수  :", register_tokens.numel())
same = torch.allclose(x_reg[0, 1 : 1 + R], x_reg[1, 1 : 1 + R])
print("배치 0/1의 register 입력이 동일한가 :", same)
# 출력: 1 + R + N = 1 + 4 + 16 = 21
# 출력: register concat 후    : (2, 21, 32)
# 출력: register 파라미터 수  : 128
# 출력: 배치 0/1의 register 입력이 동일한가 : True


# %% [markdown]
# ## 3. 블록 통과 후 register를 **버리는** forward
#
# self-attention은 시퀀스 전체를 보므로 register도 다른 토큰과 정보를 주고받는다.
# 하지만 마지막 slice에서 잘라내기 때문에 **반환 shape은 $R$과 무관**하다.
#
# $$\text{out}[:,0]=\texttt{[CLS]},\quad
#   \underbrace{\text{out}[:,1:1+R]}_{\text{폐기}},\quad
#   \text{out}[:,1+R:]=\texttt{patch}\times N$$

# %%
class TinyBlock(nn.Module):
    """LN → MHSA → residual → LN → MLP → residual (ViT 블록 최소 구현)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, need_weights=False):
        h = self.norm1(x)
        a, w = self.attn(h, h, h, need_weights=need_weights, average_attn_weights=True)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, w


class TinyViT(nn.Module):
    def __init__(self, num_register_tokens=0, dim=D, heads=4, depth=2):
        super().__init__()
        self.num_register_tokens = num_register_tokens
        self.patch_embed = nn.Conv2d(C, dim, kernel_size=P, stride=P)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + N, dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, dim)) if num_register_tokens else None
        )
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        self.blocks = nn.ModuleList([TinyBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def prepare_tokens(self, img):
        b = img.shape[0]
        t = self.patch_embed(img).flatten(2).transpose(1, 2)
        t = torch.cat((self.cls_token.expand(b, -1, -1), t), dim=1)
        t = t + self.pos_embed
        if self.register_tokens is not None:
            t = torch.cat((t[:, :1], self.register_tokens.expand(b, -1, -1), t[:, 1:]), dim=1)
        return t

    def forward_features(self, img, keep_attn=False):
        t = self.prepare_tokens(img)
        seq_in = t.shape[1]
        attn = None
        for i, blk in enumerate(self.blocks):
            last = i == len(self.blocks) - 1
            t, w = blk(t, need_weights=keep_attn and last)
            if last:
                attn = w
        t = self.norm(t)
        r = self.num_register_tokens
        return {
            "seq_len_inside": seq_in,
            "x_norm_clstoken": t[:, 0],           # 이미지 표현
            "x_norm_regtokens": t[:, 1 : r + 1],  # ← 분석용으로만 노출, 표현에는 안 씀
            "x_norm_patchtokens": t[:, r + 1 :],  # 이미지 표현
            "attn": attn,
        }


for r in (0, 4, 16):
    out = TinyViT(num_register_tokens=r).forward_features(img)
    print(
        f"R={r:2d} | 블록 내부 seq_len={out['seq_len_inside']:3d}"
        f" | CLS {tuple(out['x_norm_clstoken'].shape)}"
        f" | patch {tuple(out['x_norm_patchtokens'].shape)}"
        f" | 폐기된 reg {tuple(out['x_norm_regtokens'].shape)}"
    )
# 출력: R= 0 | 블록 내부 seq_len= 17 | CLS (2, 32) | patch (2, 16, 32) | 폐기된 reg (2, 0, 32)
# 출력: R= 4 | 블록 내부 seq_len= 21 | CLS (2, 32) | patch (2, 16, 32) | 폐기된 reg (2, 4, 32)
# 출력: R=16 | 블록 내부 seq_len= 33 | CLS (2, 32) | patch (2, 16, 32) | 폐기된 reg (2, 16, 32)


# %% [markdown]
# 블록 내부 시퀀스 길이는 $R$에 따라 17 → 21 → 33으로 늘어나지만,
# **밖으로 나오는 표현의 shape은 항상 `CLS (B,D)` + `patch (B,N,D)`로 동일하다.**
# 이것이 "at the end of the vision transformer, these tokens are discarded"의 정확한 의미다.
#
# DINOv2도 dense feature 경로에서 같은 슬라이스를 쓴다
# (`_get_intermediate_layers_not_chunked`):
#
# ```python
# outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
# ```

# %%
# 버려진다고 해서 계산에 영향이 없는 건 아니다 — register가 attention을 통해 CLS/patch를 바꾼다.
m = TinyViT(num_register_tokens=4)
with torch.no_grad():
    base = m.forward_features(img)["x_norm_patchtokens"].clone()
    m.register_tokens.add_(torch.randn_like(m.register_tokens) * 0.5)  # register 값만 변경
    after = m.forward_features(img)["x_norm_patchtokens"]
print("register 값만 바꿨을 때 patch 표현 변화(L2):", f"{(after - base).norm().item():.4f}")
# 출력: register 값만 바꿨을 때 patch 표현 변화(L2): 4.7566


# %% [markdown]
# ## 4. attention 행렬에서 register는 어디에 있나
#
# 마지막 블록의 attention 가중치 $A \in \mathbb{R}^{L\times L}$, $L = 1+R+N$ 을 그린다.
# 인덱스 규약은 DINOv2와 같다.
#
# $$\underbrace{0}_{\texttt{[CLS]}}\;\;
#   \underbrace{1 \dots R}_{\texttt{[REG]}}\;\;
#   \underbrace{R+1 \dots R+N}_{\texttt{patch}}$$
#
# register가 차지하는 **행**(register가 다른 토큰을 읽는다)과 **열**(다른 토큰이 register를
# 읽는다)이 모두 존재한다는 점이 핵심이다. register는 write-only 버퍼가 아니라 양방향이다.

# %%
R_VIS = 3
mv = TinyViT(num_register_tokens=R_VIS)
with torch.no_grad():
    A = mv.forward_features(img, keep_attn=True)["attn"][0]  # (L, L), head 평균
L = A.shape[0]
labels = ["[CLS]"] + [f"[REG{i+1}]" for i in range(R_VIS)] + [f"p{i}" for i in range(N)]
print("attention shape:", tuple(A.shape), "| L = 1 +", R_VIS, "+", N)
reg_rows = A[1 : 1 + R_VIS].sum().item()
reg_cols = A[:, 1 : 1 + R_VIS].sum().item()
print(f"register 행 총합 {reg_rows:.3f} (= R, 각 행이 softmax라 1씩)")
print(f"register 열 총합 {reg_cols:.3f} (다른 토큰들이 register에 준 총 attention)")
# 출력: attention shape: (20, 20) | L = 1 + 3 + 16
# 출력: register 행 총합 3.000 (= R, 각 행이 softmax라 1씩)
# 출력: register 열 총합 2.737 (다른 토큰들이 register에 준 총 attention)


# %% [markdown]
# ## 5. FLOP 오버헤드 — 왜 R=4가 "공짜"인가
#
# 토큰 $L$개, 폭 $D$인 ViT 블록 하나의 MAC 수는
#
# $$\text{MAC}(L) = \underbrace{12\,L D^2}_{\text{qkv·proj·MLP}} \;+\; \underbrace{2\,L^2 D}_{\text{attention}}$$
#
# 앞항은 $L$에 **선형**, 뒷항만 제곱이다. ViT-L/14 · 224px 기준 $N=256$이라
# $L=257 \to 261$ (R=4)은 1.6% 남짓 증가에 그친다. 논문 Fig. 12와 같은 결론.

# %%
def vit_block_macs(seq_len, dim):
    return 12 * seq_len * dim * dim + 2 * seq_len * seq_len * dim


DIM_L, DEPTH_L, N_L = 1024, 24, 256  # ViT-L/14 @ 224px
PARAMS_L = 304_000_000

base_macs = DEPTH_L * vit_block_macs(1 + N_L, DIM_L)
regs = [0, 1, 2, 4, 8, 16]
flop_pct, param_pct = [], []
for r in regs:
    macs = DEPTH_L * vit_block_macs(1 + r + N_L, DIM_L)
    flop_pct.append(100.0 * (macs / base_macs - 1.0))
    param_pct.append(100.0 * (r * DIM_L) / PARAMS_L)
    print(
        f"R={r:2d} | L={1+r+N_L:3d} | {macs/1e9:7.2f} GMAC"
        f" | FLOP +{flop_pct[-1]:.2f}% | params +{param_pct[-1]:.5f}% ({r*DIM_L:,})"
    )
# 출력: R= 0 | L=257 |   80.86 GMAC | FLOP +0.00% | params +0.00000% (0)
# 출력: R= 1 | L=258 |   81.19 GMAC | FLOP +0.40% | params +0.00034% (1,024)
# 출력: R= 2 | L=259 |   81.51 GMAC | FLOP +0.81% | params +0.00067% (2,048)
# 출력: R= 4 | L=261 |   82.17 GMAC | FLOP +1.62% | params +0.00135% (4,096)
# 출력: R= 8 | L=265 |   83.48 GMAC | FLOP +3.24% | params +0.00269% (8,192)
# 출력: R=16 | L=273 |   86.11 GMAC | FLOP +6.49% | params +0.00539% (16,384)


# %% [markdown]
# ## 6. 시각화
#
# 왼쪽: attention 행렬에서 `[CLS]`/`[REG]`/`patch` 밴드.
# 오른쪽: register 개수에 따른 FLOP 증가율(논문 채택값 R=4 강조).

# %%
CLS_C, REG_C, PATCH_C = "#2E5EAA", "#E8A33D", "#8C8C8C"

fig = make_subplots(
    rows=1,
    cols=2,
    column_widths=[0.55, 0.45],
    subplot_titles=(
        f"마지막 블록 attention (L = 1 + {R_VIS} + {N})",
        "register 개수별 FLOP 증가 (ViT-L/14 @224)",
    ),
)

fig.add_trace(
    go.Heatmap(
        z=A.numpy(),
        x=labels,
        y=labels,
        colorscale="Blues",
        showscale=False,  # 절대값이 아니라 "어느 행/열이 register인가"가 요점
        hovertemplate="query %{y} → key %{x}<br>attn %{z:.3f}<extra></extra>",
    ),
    row=1,
    col=1,
)
fig.update_yaxes(autorange="reversed", row=1, col=1)

# register 밴드(행/열) 표시
for lo, hi, color in [(0.5, 0.5 + R_VIS, REG_C)]:
    fig.add_shape(  # 행 밴드
        type="rect", x0=-0.5, x1=L - 0.5, y0=lo, y1=hi,
        line=dict(color=color, width=2), fillcolor="rgba(0,0,0,0)", row=1, col=1,
    )
    fig.add_shape(  # 열 밴드
        type="rect", x0=lo, x1=hi, y0=-0.5, y1=L - 0.5,
        line=dict(color=color, width=2, dash="dot"), fillcolor="rgba(0,0,0,0)", row=1, col=1,
    )
fig.add_annotation(
    x=L - 0.5, y=0.5 + R_VIS / 2, text="REG 행: 다른 토큰을 읽음",
    showarrow=False, xanchor="right", font=dict(color=REG_C, size=11), row=1, col=1,
)
fig.add_annotation(
    x=0.5 + R_VIS / 2, y=-0.6, text="REG 열: 다른 토큰이 읽어감",
    showarrow=False, yanchor="bottom", font=dict(color=REG_C, size=11), row=1, col=1,
)

fig.add_trace(
    go.Bar(
        x=[str(r) for r in regs],
        y=flop_pct,
        marker_color=[REG_C if r == 4 else PATCH_C for r in regs],
        text=[f"+{v:.2f}%" for v in flop_pct],
        textposition="outside",
        hovertemplate="R=%{x}<br>FLOP +%{y:.2f}%<extra></extra>",
        showlegend=False,
    ),
    row=1,
    col=2,
)
fig.add_annotation(
    x="4", y=flop_pct[regs.index(4)], text="논문 채택값", showarrow=True, arrowhead=2,
    ay=-40, font=dict(color=CLS_C, size=11), row=1, col=2,
)
fig.update_xaxes(title_text="register 개수 R", row=1, col=2)
fig.update_yaxes(title_text="FLOP 증가율 (%)", range=[0, 8], row=1, col=2)

fig.update_layout(
    title="register 토큰: 시퀀스 안에서는 동등하게 계산되고, 출력에서만 잘려나간다",
    template="plotly_white",
    height=560,
    width=1200,
    margin=dict(l=70, r=30, t=90, b=60),
)

_show(fig)
fig.write_image("expy.png", scale=2)  # kaleido 필요
print("saved expy.png")
# 출력: saved expy.png


# %% [markdown]
# ## 정리
#
# | 질문 | 코드로 확인한 답 |
# |---|---|
# | 어디에? | `patch_embed` → `[CLS] cat` → `pos_embed` **다음**에 `torch.cat` |
# | 어떻게? | `nn.Parameter(torch.zeros(1, R, D))`, `[CLS]`와 동일한 `normal_(std=1e-6)` |
# | 위치 정보? | 없음 — `pos_embed` 길이가 $1+N$ 이라 register 몫이 없다 |
# | 계산 참여? | 예 — attention 행/열 모두 존재, register만 바꿔도 patch 표현이 달라진다 |
# | 출력? | `x[:, 1:1+R]`은 폐기, `[CLS]`와 `x[:, 1+R:]`만 이미지 표현 |
# | 비용? | R=4에서 FLOP +1.6%, 파라미터 +0.0014% |
