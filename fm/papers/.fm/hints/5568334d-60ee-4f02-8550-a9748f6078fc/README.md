# DINOv2가 직접 구현한 FlashAttention의 특징

> **Q.** DINOv2가 직접 구현한 FlashAttention의 특징은?
>
> **A.** self-attention 층의 메모리 사용량과 속도를 개선한 자체 버전으로, 모든 케이스에서 원본과 동등하거나 더 나으면서 더 많은 use-case와 하드웨어를 커버한다. GPU 특성상 head당 embedding 차원이 64의 배수일 때 효율이 가장 좋다.

출처: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* (arXiv 2304.07193), **Section 5 "Efficient implementation" — "Fast and memory-efficient attention"** 단락.

---

## 1. 어디에 나오는 이야기인가

논문 5장은 "대규모로 학습하기 위한 엔지니어링 개선"을 모아 둔 장이다. A100 GPU + PyTorch 2.0 환경에서, **같은 하드웨어 기준으로 iBOT 구현 대비 약 2배 빠르고 메모리는 1/3만 사용**한다고 요약하며, 그 근거로 네 가지 기법을 든다.

| 기법 | 한 줄 요약 |
|---|---|
| **Fast and memory-efficient attention** (이 카드) | FlashAttention을 자체 구현해 self-attention의 메모리·속도 개선 |
| Sequence packing | 224 크롭과 98 크롭의 서로 다른 길이 토큰 시퀀스를 하나로 이어 붙이고 block-diagonal mask로 분리 |
| Efficient stochastic depth | drop된 residual을 마스킹하는 대신 계산 자체를 건너뜀 (d = 40%) |
| FSDP | student/teacher/optimizer 모멘트 4개 복제본(ViT-g 기준 16 GB)을 GPU 간 샤딩, 통신은 float16 |

이 카드는 그 첫 번째 항목이다.

## 2. 원문이 말하는 세 가지 특징

논문 원문(요지)은 다음과 같다.

> We implemented our own version of FlashAttention (Dao et al., 2022) to improve memory usage and speed on the self-attention layers. Our version is on par with or better than the original on all cases considered, while covering more use-cases and hardware. Due to the GPU hardware specifics, the efficiency is best when the embedding dimension per head is a multiple of 64, and the matrix operations are even better when the full embedding dimension is a multiple of 256.

여기서 카드의 답을 세 조각으로 나눠 읽을 수 있다.

### (1) 목적 — self-attention 층의 메모리와 속도

표준 attention은 `softmax(QKᵀ/√d)V`를 계산할 때 **N×N 크기의 attention 행렬을 GPU 메모리(HBM)에 실제로 만들어 저장**한다. 토큰 수 N이 커질수록 메모리가 O(N²)로 늘고, 이 큰 행렬을 HBM에 쓰고 다시 읽는 I/O가 병목이 된다.

FlashAttention(Dao et al., NeurIPS 2022)의 핵심 아이디어는 이 행렬을 **통째로 만들지 않는 것**이다. Q, K, V를 블록 단위로 잘라 GPU의 빠른 on-chip SRAM 안에서 부분 softmax를 계산하고(online softmax / tiling), 결과만 누적해 나간다. 수학적으로는 **정확히 같은 결과(exact attention)** 를 내지만 HBM 읽기·쓰기 횟수가 크게 줄어 메모리는 O(N)으로, 속도도 실제 wall-clock 기준으로 빨라진다. DINOv2는 이런 "IO-aware" attention을 자체 커널로 구현해 ViT의 모든 self-attention 층에 적용했다.

### (2) 성능 — 원본과 동등하거나 더 나음 + 더 넓은 커버리지

논문은 "고려한 모든 케이스에서 원본 FlashAttention과 동등하거나 더 낫다(on par with or better)"고 말하면서, 동시에 **더 많은 use-case와 하드웨어를 커버**한다고 강조한다.

당시 공개 FlashAttention은 지원 범위가 제한적이었다(특정 GPU 세대, 특정 head 차원, 제한된 attention bias/mask 형태 등). DINOv2의 학습 파이프라인은 바로 뒤에 나오는 **sequence packing** 때문에 attention 행렬에 **block-diagonal mask**를 얹어야 하는데, 이런 임의의 마스크/bias를 받아 주는 것이 "더 많은 use-case"의 대표적인 예다. 이 저수준 구성 요소들은 Meta의 **xFormers** 라이브러리(Lefaudeux et al., 2022)로 공개되어 있다고 논문이 명시한다.

실제 코드(`dinov2/layers/attention.py`)를 보면 이 자체 구현이 어떻게 연결되는지 드러난다.

```python
from xformers.ops import memory_efficient_attention, unbind

class MemEffAttention(Attention):
    def forward(self, x, attn_bias=None):
        ...
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = unbind(qkv, 2)
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)   # ← FlashAttention류 커널
        ...
```

`attn_bias`로 block-diagonal 마스크를 넘길 수 있고, xFormers가 없으면 일반 `Attention`으로 폴백한다(단, nested tensor/packing은 xFormers 필수).

### (3) 하드웨어 제약 — head당 차원은 64의 배수, 전체 차원은 256의 배수

FlashAttention류 커널은 head 차원 `d`를 GPU의 텐서 코어 타일 크기와 SRAM 블록에 맞춰 처리하기 때문에, **`d`가 64의 배수**일 때 낭비 없이 꽉 채워 계산할 수 있다. 또 QKV projection, 출력 projection, FFN 같은 큰 행렬곱은 **전체 embedding 차원이 256의 배수**일 때 가장 효율적이다.

이 제약이 실제 **모델 아키텍처 결정**에까지 영향을 준 것이 이 단락의 재미있는 점이다.

| | Zhai et al. (2022)의 ViT-g | DINOv2의 ViT-g/14 |
|---|---|---|
| Embedding dim | 1408 | **1536** (= 6 × 256) |
| Heads | 16 | **24** |
| dim / head | 88 (64의 배수 아님) | **64** |
| Blocks | 40 | 40 |
| 파라미터 | 약 1.0B | 약 1.1B |

즉 DINOv2는 원래 ViT-g 설계를 그대로 쓰지 않고, **head당 64차원 × 24 heads = 1536**이 되도록 살짝 바꿨다. 논문은 이 변경이 최종 정확도에 유의미한 차이를 만들지 않았다고 보고한다. 부록 Table 17에도 ViT-g/14가 embed dim 1536, heads 24, blocks 40, SwiGLU FFN으로 기재되어 있다. 코드에도 그 의도가 docstring으로 남아 있다.

```python
def vit_giant2(...):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
```

참고로 다른 크기도 같은 규칙을 만족한다. ViT-S(384/6 heads), ViT-B(768/12), ViT-L(1024/16) 모두 head당 64차원이다.

## 3. 기억용 요약

- **무엇**: FlashAttention(Dao et al., 2022)의 **자체 구현** — attention 행렬을 HBM에 통째로 만들지 않는 IO-aware exact attention.
- **효과**: self-attention의 **메모리 사용량 ↓, 속도 ↑**. 5장 전체 기법을 합치면 iBOT 대비 약 2배 빠름, 메모리 1/3.
- **원본 대비**: 모든 케이스에서 **동등 이상**, 그러면서 **더 많은 use-case(예: block-diagonal mask)와 하드웨어**를 지원. xFormers로 공개.
- **하드웨어 제약**: **head당 차원 64의 배수**가 최적, 전체 차원은 256의 배수면 더 좋음.
- **파급 효과**: 그래서 ViT-g를 1408/16 heads(88 dim/head) 대신 **1536/24 heads(64 dim/head)** 로 설계. 정확도 차이 없음, 1.1B 파라미터.

## 4. 함께 헷갈리기 쉬운 것

- FlashAttention은 **근사(approximate) attention이 아니다.** 결과는 표준 attention과 수학적으로 동일하고, 달라지는 것은 메모리 접근 패턴과 속도다.
- "2배 빠르고 메모리 1/3"은 FlashAttention **하나만의 효과가 아니라** 5장의 네 기법(attention, sequence packing, stochastic depth, FSDP)을 합친 결과다.
- 64의 배수 제약은 논문이 "GPU hardware specifics" 때문이라고 설명하는 **효율성** 조건이지, 그 외 차원에서 동작하지 않는다는 뜻은 아니다.
