# FSDP를 쓰면 모델 크기 한계가 어떻게 바뀌나?

> **한 줄 답**: 모델 복제본(student, teacher, AdamW 1차·2차 모멘트)을 GPU들에 **sharding**하므로, 모델 크기가 **단일 GPU 메모리**가 아니라 **컴퓨트 노드 전체의 GPU 메모리 총합**에 의해 제한된다. ViT-g 기준 16 GB짜리 상태를 여러 GPU에 나눠 담는 방식이다.

출처: DINOv2 논문(arXiv 2304.07193) §5 *Efficient implementation* — "Fully-Sharded Data Parallel (FSDP)" 단락.

---

## 1. 왜 16 GB인가 — 학습 상태의 구성

DINOv2는 AdamW로 목적함수를 최소화한다. 이때 GPU에 **float32로 상주해야 하는 모델 크기의 텐서**가 4벌 필요하다.

| # | 복제본 | 역할 |
|---|--------|------|
| 1 | **student** 가중치 | 역전파로 실제로 갱신되는 네트워크 |
| 2 | **teacher** 가중치 | student의 EMA(momentum 0.994→1). 역전파는 없지만 가중치 한 벌을 통째로 들고 있어야 함 |
| 3 | AdamW **1차 모멘트** (m) | 파라미터마다 하나씩, 그래디언트의 이동평균 |
| 4 | AdamW **2차 모멘트** (v) | 파라미터마다 하나씩, 그래디언트 제곱의 이동평균 |

ViT-g/14는 임베딩 1536·24 헤드·40 블록·SwiGLU FFN(부록 Table 17)으로 **약 1.1B 파라미터**다.

- 1.1B 파라미터 × 4 byte(float32) ≈ **4.4 GB / 복제본**
- 4 복제본 ⇒ **≈ 16 GB**

논문은 이 값을 "16 GB of memory for a billion-parameter model"로 요약한다. 여기에 그래디언트, 활성값(activation), 큰 배치까지 더해지면 A100-40GB 한 장에는 사실상 들어가지 않는다.

> 참고: student·teacher를 fp16으로 굴리더라도 **optimizer가 요구하는 마스터 가중치·모멘트는 float32**여야 하므로, 위 4벌은 정밀도를 낮춰 지우기 어렵다. 그래서 "정밀도 낮추기"가 아니라 "쪼개서 나누기"가 필요하다.

## 2. DDP vs FSDP — 한계가 바뀌는 지점

| | DistributedDataParallel (DDP) | Fully-Sharded Data Parallel (FSDP) |
|---|---|---|
| GPU마다 들고 있는 것 | **모델·옵티마이저 상태 전체**(복제) | **자기 몫의 shard**만 (1/N) |
| 데이터 | GPU마다 다른 미니배치 | GPU마다 다른 미니배치 (동일) |
| 모델 크기 상한 | **GPU 한 장 메모리** | **노드(들) 전체 GPU 메모리 합** |
| 통신 | 그래디언트 all-reduce (DINOv2 이전 SSL 방법들이 사용) | forward/backward 직전에 shard를 all-gather, backward 후 reduce-scatter |

DDP는 이름 그대로 "데이터"만 나누고 모델은 **모든 GPU에 복제**한다. 그래서 16 GB의 상태가 GPU마다 그대로 올라가야 하고, 모델이 한 장에 안 들어가면 어떤 수를 써도 학습이 불가능하다.

FSDP는 파라미터·그래디언트·옵티마이저 상태를 **N개 GPU에 1/N씩 쪼개서 저장**한다. 예를 들어 8-GPU 노드에서는 16 GB가 GPU당 **2 GB**로 줄어든다. 어떤 레이어의 연산이 필요해질 때만 그 레이어의 shard를 잠시 모아(all-gather) 계산하고, 끝나면 다시 버린다(reshard). 그 결과 **모델 전체가 어느 한 장에 다 들어갈 필요가 없어지고**, 상한은 "참여하는 GPU 메모리를 전부 합친 값"이 된다. 이것이 카드가 묻는 "한계가 바뀌는" 핵심이다.

## 3. 덤으로 얻는 두 번째 이점 — 통신량 절반

PyTorch FSDP의 `MixedPrecision` 설정은 저장 정밀도와 통신 정밀도를 분리한다.

- **weight shard 저장**: float32 (옵티마이저 요구)
- **가중치 broadcast·그래디언트 reduce**: backbone은 **float16**
- **MLP 헤드(DINO/iBOT head) 그래디언트 reduce**: 학습 불안정을 피하기 위해 **float32** 유지

이로써 DDP의 float32 all-reduce에 비해 **통신 비용이 약 50% 감소**하고, GPU 노드 수를 늘릴 때 DDP + float16 autocast보다 더 효율적으로 확장된다. 논문은 "사실상 모든 경우에 Pytorch-FSDP mixed-precision이 DDP with autocast보다 우월했다"고 결론짓는다.

§5 서두의 요약: 같은 하드웨어에서 iBOT 구현 대비 DINOv2 코드는 **약 2× 빠르고 메모리는 1/3**만 쓴다. FSDP는 FlashAttention, sequence packing, efficient stochastic depth와 함께 이를 만든 네 요소 중 하나다.

## 4. 코드에서 확인하기 (facebookresearch/dinov2)

`dinov2/fsdp/__init__.py`의 `get_fsdp_wrapper`가 위 내용을 그대로 구현한다.

```python
mixed_precision_config = MixedPrecision(
    param_dtype=...,   # 통신·연산용 가중치 dtype (기본 fp16)
    reduce_dtype=...,  # 그래디언트 reduce dtype
    buffer_dtype=...,  # 버퍼 dtype (fp32)
)
fsdp_wrapper = partial(
    FSDP,
    sharding_strategy=...,          # NO_SHARD / SHARD_GRAD_OP / FULL_SHARD
    mixed_precision=mixed_precision_config,
    use_orig_params=True,
    auto_wrap_policy=ModuleWrapPolicy(modules_to_wrap),  # BlockChunk 단위로 감쌈
)
```

`dinov2/configs/ssl_default_config.yaml`의 기본값이 논문 서술과 정확히 대응한다.

| 모듈 | sharding_strategy | param_dtype | reduce_dtype |
|---|---|---|---|
| student.backbone | SHARD_GRAD_OP | fp16 | **fp16** |
| student.dino_head / ibot_head | SHARD_GRAD_OP | fp16 | **fp32** ← "MLP heads gradients are reduced in float32" |
| teacher.* | SHARD_GRAD_OP | fp16 | fp16 (teacher는 역전파 없음) |

`ssl_meta_arch.py`에서는 student·teacher 각각을 `get_fsdp_wrapper(..., modules_to_wrap={BlockChunk})`로 감싸고, teacher는 EMA 갱신 후 `reshard_fsdp_model(self.teacher)`로 shard 상태로 되돌려 메모리를 회수한다. 옵티마이저는 `ShardedGradScaler`로 fp16 손실 스케일링을 처리한다.

## 5. 기억 포인트

1. **4 복제본 × fp32** (student, teacher, Adam m, Adam v) ⇒ 1.1B 모델에 **16 GB**.
2. DDP는 이 16 GB를 **GPU마다 복제** → 한계 = GPU 한 장.
3. FSDP는 16 GB를 **GPU들에 shard** → 한계 = **노드 전체 GPU 메모리 합**.
4. 부가 효과: fp32 저장 + fp16 통신으로 **통신량 ~50% 절감**, 헤드 그래디언트만 fp32.
