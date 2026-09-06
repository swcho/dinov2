# PyTorch FSDP의 두 번째 이점: float16 통신으로 통신 비용 절감

> **Q.** PyTorch FSDP가 통신 비용 측면에서 주는 두 번째 이점은?
>
> **A.** weight shard는 optimizer 요구대로 float32로 저장하되, 백본의 weight broadcast와 gradient reduce는 float16으로 수행한다. DDP의 float32 all-reduce 대비 통신 비용이 약 50% 줄어든다.

출처: DINOv2 논문(arXiv 2304.07193v2) §5 "Efficient implementation" 중 **Fully-Sharded Data Parallel (FSDP)** 단락.

---

## 1. 배경: 왜 FSDP를 썼는가 (첫 번째 이점 = 메모리)

논문은 FSDP의 이점을 두 가지로 나누어 서술한다. 카드가 묻는 것은 **두 번째**지만, 첫 번째를 알아야 문맥이 잡힌다.

- DINOv2 목표 함수를 **AdamW**로 최적화하려면 float32 정밀도의 모델 복제본이 **4개** 필요하다.
  1. student 가중치
  2. teacher 가중치 (EMA)
  3. optimizer 1차 모멘트 (m)
  4. optimizer 2차 모멘트 (v)
- ViT-g처럼 파라미터가 약 10억 개인 모델이면 이것만 **16 GB**(1B × 4 byte × 4 복제본).
- 이를 GPU 하나에 다 올리지 않고 **GPU들에 걸쳐 샤딩(shard)** 하는 것이 FSDP. 그 결과 모델 크기의 상한이 "GPU 1장의 메모리"가 아니라 "클러스터 전체 GPU 메모리 합"이 된다. → **첫 번째 이점: 메모리 절감**

## 2. 두 번째 이점: 통신 비용 절감 (카드의 핵심)

FSDP는 각 GPU가 파라미터의 일부(shard)만 들고 있으므로, 매 step마다

- **forward/backward 전에 weight를 모아야 하고** (all-gather / broadcast),
- **backward 후 gradient를 합쳐 다시 shard 주인에게 돌려줘야 한다** (reduce-scatter / reduce).

이 통신 자체가 곧 비용인데, PyTorch FSDP의 mixed-precision 기능 덕분에 다음처럼 **저장 정밀도와 통신 정밀도를 분리**할 수 있다.

| 항목 | 정밀도 | 이유 |
|---|---|---|
| weight shard **저장** (master weights, optimizer state) | **float32** | AdamW 등 optimizer가 정밀한 누적 갱신을 요구 |
| 백본(backbone) weight **broadcast** | **float16** | 통신량 절반 |
| 백본 gradient **reduce** | **float16** | 통신량 절반 |
| **MLP head(DINO head) gradient reduce** | **float32** (예외) | 학습 불안정 방지 |

- float32(4 byte) → float16(2 byte)이므로 주고받는 바이트 수가 절반. 그래서 논문은 DDP가 쓰는 **float32 gradient all-reduce** 대비 **"약 50% 통신 비용 감소"** 라고 말한다.
- 여기서 비교 대상인 DDP(DistributedDataParallel)는 DINO(Caron et al., 2021), iBOT(Zhou et al., 2022a) 같은 기존 자기지도 사전학습 방법들이 사용한 방식이다.

### 왜 "DDP + float16 autocast"와도 다른가?

autocast는 **연산(matmul 등)** 만 float16으로 내리고, DDP의 gradient all-reduce는 여전히 파라미터 dtype(float32) 버킷으로 이루어진다. 즉 계산은 빨라지지만 **통신은 그대로 float32**다. 반면 FSDP mixed-precision은 통신 버퍼 자체를 float16으로 지정할 수 있어(`MixedPrecision(param_dtype=fp16, reduce_dtype=fp16, ...)` 계열 설정) 통신량이 줄어든다. 논문 결론:

- GPU 노드를 늘릴 때 **FSDP 쪽이 DDP + float16 autocast보다 더 효율적으로 스케일**한다.
- "사실상 우리가 마주친 거의 모든 경우에서 PyTorch-FSDP mixed-precision이 DDP with autocast보다 우수했다."

### 예외 조항 기억하기: DINO head는 float32로 reduce

부록 B Table 16의 각주도 같은 내용을 반복한다. "모든 모델을 float16으로 학습하되, **DINO head의 gradient만 float32로 reduce**". 헤드는 프로토타입 차원(65536)이 큰 선형층이고 softmax-centering 기반 손실에 직접 연결되어 있어 정밀도 손실이 불안정으로 이어지기 쉽다는 실무적 판단이다. 카드 답에서 "백본의" 라는 한정어가 붙는 이유가 바로 이것.

## 3. 한눈에 정리

```
FSDP 이점 ① 메모리 : 4개 fp32 복제본(16GB for ViT-g)을 GPU들에 샤딩
FSDP 이점 ② 통신   : 저장은 fp32, 통신(backbone broadcast/reduce)은 fp16
                     → DDP fp32 all-reduce 대비 통신량 ≈ 50% 감소
                     (단, MLP head gradient는 fp32 reduce — 안정성)
```

## 4. 기억 팁

- "**저장은 32, 통신은 16, 헤드만 다시 32**" 한 줄로 암기.
- 50%라는 숫자는 단순히 4 byte → 2 byte에서 나온다. 계산 정밀도(autocast)가 아니라 **통신 정밀도**가 절반이 되었다는 점이 DDP와의 차이.
- 같은 §5의 다른 효율화 기법(FlashAttention 계열 커스텀 attention, sequence packing, efficient stochastic depth)과 묶어서 "DINOv2 효율 구현 4종"으로 떠올리면 문맥이 잘 살아난다.
