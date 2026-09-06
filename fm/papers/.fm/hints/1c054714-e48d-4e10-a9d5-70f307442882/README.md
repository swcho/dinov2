# DINOv2의 ViT-g가 원논문(Zhai et al., 2022) ViT-g와 다른 점

> **Q.** ViT-g 아키텍처가 원논문(Zhai et al., 2022)과 다른 점과 이유는?
> **A.** 1408 dim/16 heads(88 dim/head) 대신 1536 dim/24 heads(64 dim/head)를 쓴다. head당 차원을 64의 배수, 전체 차원을 256의 배수로 맞춰 행렬 연산 효율을 극대화하기 위함이며, 최종 정확도 차이는 유의미하지 않았다.

## 1. 출처: 논문 5절 "Efficient implementation" — Fast and memory-efficient attention

DINOv2 논문(Oquab et al., 2023, arXiv 2304.07193) 5절은 "더 큰 모델을 더 큰 데이터로 학습시키기 위한 구현상의 개선"을 모아 둔 절이다. 그 첫 항목이 **자체 구현한 FlashAttention**인데, 여기서 ViT-g 아키텍처 변경이 함께 언급된다. 원문 요지:

- FlashAttention(Dao et al., 2022)을 직접 재구현해 self-attention의 메모리와 속도를 개선했다.
- **GPU 하드웨어 특성상, head당 임베딩 차원이 64의 배수일 때 효율이 가장 좋고, 전체 임베딩 차원이 256의 배수이면 행렬 연산이 더욱 좋아진다.**
- 그래서 Zhai et al. (2022)이 제안한 ViT-g와 **조금 다르게**, 1408 dim / 16 heads(88 dim/head)가 아니라 **1536 dim / 24 heads(64 dim/head)**를 사용한다.
- 실험상 최종 정확도에 유의미한 차이는 없었고, DINOv2의 ViT-g 백본은 **1.1B 파라미터**다.

같은 절에서 "동일 하드웨어에서 iBOT 구현 대비 약 2배 빠르고 메모리는 1/3만 사용"한다고 밝히는데, 이 아키텍처 조정도 그 효율 향상에 기여하는 요소 중 하나다.

## 2. 두 ViT-g 비교

Zhai et al., "Scaling Vision Transformers" (CVPR 2022)는 ViT-B/L/H 위에 ViT-g(약 1B)와 ViT-G(약 1.8B)를 새로 정의했다. DINOv2는 이름은 ViT-g를 그대로 쓰지만 폭(width)과 head 구성을 바꿨다.

| 항목 | Zhai et al. (2022) ViT-g | DINOv2 ViT-g/14 |
|---|---|---|
| 임베딩 차원 (width) | 1408 | **1536** |
| head 수 | 16 | **24** |
| head당 차원 | 1408 / 16 = **88** | 1536 / 24 = **64** |
| 깊이 (blocks) | 40 | 40 (동일) |
| FFN | MLP (hidden 6144) | **SwiGLU** (from-scratch 학습용) |
| 패치 크기 | 14 | 14 (동일) |
| 파라미터 | 약 1.0B | 약 1.1B |

깊이는 그대로 두고 폭만 1408 → 1536으로 약 9% 키웠다. 파라미터가 1.0B → 1.1B로 조금 늘어난 것은 이 폭 증가 때문이다(attention 4·d², FFN 항이 d에 비례하므로 폭이 커지면 블록당 파라미터도 커진다).

논문 부록 Table 17("Architecture details")에도 DINOv2가 사용한 전체 모델 계열이 정리되어 있다.

| Arch. | Embed dim | Heads | Blocks | FFN layer |
|---|---|---|---|---|
| ViT-S/14 (distilled) | 384 | 6 | 12 | MLP |
| ViT-B/14 (distilled) | 768 | 12 | 18 | MLP |
| ViT-L/14 (distilled) | 1024 | 16 | 24 | MLP |
| ViT-L/14 (from scratch) | 1024 | 16 | 24 | SwiGLU |
| **ViT-g/14 (from scratch)** | **1536** | **24** | **40** | **SwiGLU** |

S/B/L은 모두 head당 64 차원(384/6, 768/12, 1024/16)이므로, ViT-g를 1536/24로 잡으면 **전 계열이 head당 64 차원으로 통일**된다는 점도 확인할 수 있다.

## 3. 왜 64의 배수 / 256의 배수인가

### head당 차원 = 64의 배수
- FlashAttention 계열 커널은 Q·Kᵀ와 P·V 곱을 SRAM에 올라가는 **타일 단위**로 처리하며, 이 타일은 head 차원 방향으로 64 또는 128 같은 2의 거듭제곱 크기에 맞춰 짜여 있다.
- head당 88 차원이면 어떤 타일 크기에도 딱 맞지 않아 **128로 패딩**하거나 여러 조각으로 나눠 처리해야 한다. 이는 유효 연산량의 약 30% 이상을 0을 곱하는 데 쓰는 셈이 되고, 메모리 접근도 비정렬(unaligned)이 된다.
- A100의 Tensor Core(fp16/bf16)는 8 또는 16의 배수 차원에서 정상 동작하고, 64·128 단위에서 파이프라인 효율이 최대가 된다. 88은 8의 배수라 "돌아가긴" 하지만 최적은 아니다.

### 전체 임베딩 차원 = 256의 배수
- QKV projection(d → 3d), 출력 projection(d → d), FFN(d → hidden → d)은 모두 큰 GEMM이다. cuBLAS/CUTLASS 커널은 출력 행렬을 128×128, 256×128 같은 **타일로 쪼개** 처리하므로, 차원이 256의 배수면 타일 경계에 남는 조각(tail)이 없어 낭비가 사라진다.
- 1408 = 256 × 5.5 이므로 항상 반쪽 타일이 남는다. 1536 = 256 × 6 이라 정확히 맞는다.
- 정리하면 "**수학적으로는 아무 차원이든 되지만, 하드웨어가 잘 먹는 숫자로 맞춰 준다**"는 실용적 선택이다. 이는 LLM(예: LLaMA의 4096/8192, head 128)에서도 똑같이 보이는 관행이다.

### 왜 정확도는 안 바뀌는가
- 폭·head 구성의 미세 변경은 모델 용량(~1B)과 깊이(40)를 거의 그대로 두므로 표현력 차이가 작다. 논문은 "유의미한 차이가 없었다"고만 짧게 언급하며, 이 변경은 성능 개선이 아니라 **순수히 학습 비용 절감**을 위한 것임을 분명히 한다.

## 4. 코드에서 확인

이 저장소의 `dinov2/models/vision_transformer.py`에는 해당 아키텍처가 `vit_giant2`라는 이름으로 정의되어 있고, docstring이 논문 문장을 그대로 반영한다.

```python
def vit_giant2(patch_size=16, num_register_tokens=0, ...):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        ...
    )
```

- 이름이 `vit_giant`가 아니라 **`vit_giant2`**인 이유가 바로 "Zhai의 ViT-g와 다른, 두 번째 버전의 giant"라는 뜻이다.
- `attn_class=MemEffAttention`(xFormers memory-efficient attention)을 쓰므로 head당 64 차원이 커널 효율과 직결된다.
- `dinov2/configs/train/vitg14.yaml`은 `arch: vit_giant2`, `ffn_layer: swiglufused`로 설정되어 Table 17의 "ViT-g/14 (from scratch), SwiGLU"와 일치한다. torch.hub의 `dinov2_vitg14`도 같은 조합을 사용한다.

## 5. 한 줄 정리

**"1408/16(88 per head) → 1536/24(64 per head)": 정확도를 위해서가 아니라, head 차원을 64의 배수·전체 차원을 256의 배수로 맞춰 FlashAttention과 GEMM 커널이 최고 효율로 돌게 하기 위한 하드웨어 친화적 조정. 결과는 1.1B 파라미터, 정확도는 동등.**

## 참고
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", arXiv 2304.07193, Sec. 5 및 Appendix Table 17.
- Zhai, Kolesnikov, Houlsby, Beyer, "Scaling Vision Transformers", CVPR 2022 — ViT-g(1408/16/40), ViT-G(1664/16/48) 정의.
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022.
