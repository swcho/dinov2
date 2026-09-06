# Efficient stochastic depth — 마스킹 대신 계산을 건너뛴다

> **Q.** efficient stochastic depth가 기존 구현과 다른 점은?
> **A.** 드롭된 residual의 결과를 마스킹하는 대신 계산 자체를 건너뛴다. 전용 fused kernel 덕분에 drop rate에 거의 비례해 메모리와 연산을 절약한다.

출처: DINOv2 논문(arXiv 2304.07193) §5 *Efficient implementation* → "Efficient stochastic depth" 단락. 이 카드는 그림이 아니라 구현 세부를 다루므로, 논문 그림 대신 실제 코드(이 저장소의 `dinov2/layers/`)를 근거로 설명한다.

---

## 1. 먼저: stochastic depth(DropPath)란?

Stochastic depth(Huang et al., 2016)는 학습 중 **residual block 전체를 샘플 단위로 무작위로 끈다**. 블록의 출력 `x + f(x)`에서 `f(x)`(residual branch)를 확률 `d`로 0으로 만들고, 살아남은 샘플은 `1/(1-d)`로 스케일해 기대값을 보존한다. 깊은 네트워크의 정규화 기법이며, ViT 계열에서는 "drop path"라는 이름으로 쓰인다.

DINOv2는 이 비율을 상당히 높게(**d = 0.4**, 40%) 잡았다. Table 1의 ablation에 따르면 LayerScale과 높은 stochastic depth는 linear probe 성능을 약간 떨어뜨리지만, 학습 중 loss가 NaN으로 발산하는 것을 막아 **대형 모델(ViT-g, 1.1B)의 학습 안정성**을 크게 높인다. 즉 DINOv2에서 stochastic depth는 "있으면 좋은 옵션"이 아니라 대규모 학습을 가능하게 하는 필수 요소이고, 그래서 40%짜리 drop을 **공짜로** 만드는 것이 중요해진다.

## 2. 기존 구현: 계산은 다 하고, 결과를 마스킹

timm 등 널리 쓰이는 `DropPath`는 이 저장소의 `dinov2/layers/drop_path.py`에도 그대로 들어 있다.

```python
def drop_path(x, drop_prob, training):
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)          # (B, 1, 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    random_tensor.div_(keep_prob)                         # 살아남은 샘플은 1/(1-d) 배
    return x * random_tensor                              # 드롭된 샘플은 0 곱하기
```

블록에서는 `x = x + drop_path(f(x))` 형태로 쓰인다. 핵심 문제는 **`f(x)`가 배치 전체 B개에 대해 먼저 계산된 뒤** 그 결과에 0/1 마스크를 곱한다는 점이다.

- 연산: attention + MLP가 B개 샘플 전부에 대해 forward/backward 된다. 드롭될 40%도 계산한 뒤 버린다.
- 메모리: backward를 위해 저장되는 activation(attention 행렬, MLP 중간 hidden 등)도 B개 분량 그대로다.
- 결국 drop rate가 얼마든 비용은 d = 0인 경우와 **동일**하다. 정규화 효과만 얻고 효율 이득은 전혀 없다.

## 3. DINOv2 구현: 살릴 샘플만 골라 계산하고, 제자리에 더한다

논문 표현을 그대로 옮기면: *"randomly shuffling the B samples over the batch dimension, and slicing the first (1 − d) × B samples for the computations in the block."* — 배치를 무작위로 섞고 앞쪽 (1−d)·B개만 잘라서 블록 계산에 넣는다.

`dinov2/layers/block.py`의 `drop_add_residual_stochastic_depth`가 이것이다.

```python
def drop_add_residual_stochastic_depth(x, residual_func, sample_drop_ratio):
    # 1) 순열로 부분집합 추출
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = torch.randperm(b, device=x.device)[:sample_subset_size]   # 살릴 인덱스
    x_subset = x[brange]                                                # (0.6B, N, D)

    # 2) 부분집합에만 residual branch 적용  ← 여기서 계산량이 0.6배가 됨
    residual = residual_func(x_subset)

    # 3) 원래 자리에 스케일해서 더하기 (index_add = fused scatter-add)
    residual_scale_factor = b / sample_subset_size                      # = 1/(1-d)
    x_plus_residual = torch.index_add(x.flatten(1), 0, brange,
                                      residual.flatten(1), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)
```

세 단계로 정리하면:

| 단계 | 기존 DropPath | Efficient stochastic depth |
|---|---|---|
| 어떤 샘플을 드롭할지 결정 | 샘플별 Bernoulli(1−d) 마스크 | `randperm(B)` 후 앞 (1−d)·B개 인덱스(`brange`) 선택 |
| residual branch `f(·)` 계산 | **B개 전부** 계산 | **`x[brange]` — (1−d)·B개만** 계산 |
| 결과 합치기 | `x + mask * f(x) / (1−d)` (elementwise) | `index_add(x, brange, f(x_subset), alpha = B/|brange|)` — 선택된 행에만 스케일된 residual을 scatter-add |

수학적으로는 두 방식이 같은 것을 계산한다(살아남은 샘플: `x + f(x)/(1−d)`, 드롭된 샘플: `x` 그대로). 차이는 오직 **드롭될 샘플에 대해 `f`를 아예 호출하지 않는다**는 점이다. Bernoulli 대신 순열-슬라이싱을 쓰는 이유도 여기 있다: 매 스텝 살아남는 샘플 수를 정확히 (1−d)·B로 고정해야 텐서 크기가 결정적이 되어 커널을 효율적으로 돌릴 수 있다.

## 4. "fused kernel" 이 뭘 가리키는가

논문은 "thanks to specific fused kernels"라고만 쓰는데, 코드에서 두 곳이 해당한다.

1. **gather → 계산 → scatter-add 경로 자체.** `x[brange]`(index_select)로 모으고, `torch.index_add(..., alpha=...)`로 스케일과 덧셈을 한 번에 원위치에 뿌린다. 마스크 텐서를 만들고 곱하고 더하는 별도 elementwise 패스가 없다.
2. **xFormers의 `scaled_index_add`.** LayerScale(`ls1.gamma`, `ls2.gamma`)이 있는 경로(`add_residual`, `NestedTensorBlock`)에서는 `x[brange] += alpha * gamma * residual`을 **한 커널**로 처리한다. LayerScale 곱, drop 스케일, residual 덧셈, 인덱스 scatter를 각각 따로 하면 중간 텐서가 세 번 생기지만 이를 하나로 융합했다. 같은 파일의 `index_select_cat`은 여러 crop 시퀀스(global/local crops)를 살릴 인덱스만 골라 한 번에 이어붙이는 fused gather로, §5의 "sequence packing"과 이 기법을 함께 쓰기 위한 것이다.

이 커널들은 논문 각주에 언급된 xFormers 라이브러리(`xformers.ops`)에 들어 있다. xFormers가 없으면 코드는 `torch.index_add` 경로로 fallback한다.

## 5. 절약이 "drop rate에 거의 비례"하는 이유와 한계

- residual branch(attention + MLP)가 블록 연산의 거의 전부이므로, 그 부분을 (1−d)배만 계산하면 블록 FLOPs와 activation 메모리가 대략 (1−d)배가 된다. d = 0.4면 블록 비용의 약 40%가 사라진다. 이것이 논문이 말하는 "in proportion approximately equal to the drop rate".
- "거의(approximately)"인 이유: `randperm`, gather, scatter-add 같은 오버헤드가 추가되고, LayerNorm 입력 슬라이싱·residual stream 자체(`x`)는 여전히 B개 크기다. 그래서 코드에는 다음 조건이 있다.

```python
if self.training and self.sample_drop_ratio > 0.1:
    # the overhead is compensated only for a drop path rate larger than 0.1
    x = drop_add_residual_stochastic_depth(...)
elif self.training and self.sample_drop_ratio > 0.0:
    x = x + self.drop_path1(attn_residual_func(x))    # 낮은 비율이면 기존 마스킹 방식
```

  즉 drop rate가 0.1 이하일 때는 오버헤드가 이득을 상쇄해 기존 DropPath를 그대로 쓴다. DINOv2처럼 d = 0.4로 높게 잡았을 때 비로소 "drastic improvement in compute efficiency and memory usage"가 된다.
- 추론(eval)이나 distillation 단계에서는 stochastic depth를 끄므로 이 경로는 학습 전용이다(§5 "Model distillation": 소형 모델 증류 시 masking과 stochastic depth를 제거).

## 6. 큰 그림에서의 위치

§5의 효율화 항목은 (1) 자체 FlashAttention, (2) crop 시퀀스 패킹(block-diagonal attention mask), (3) efficient stochastic depth, (4) FSDP, (5) distillation이다. 이들을 합쳐 DINOv2 코드는 같은 하드웨어에서 iBOT 구현보다 **약 2배 빠르고 메모리는 1/3**만 쓴다. 그중 stochastic depth 항목은 "정규화를 위해 어차피 40%를 버릴 거라면, 버릴 것은 계산하지도 말자"는 단순한 원칙을 GPU 커널 수준까지 밀어붙인 사례다.

## 한 줄 요약

기존 DropPath = **다 계산하고 0을 곱한다**. DINOv2 = **배치를 섞어 앞 60%만 잘라 계산하고, fused `index_add`/`scaled_index_add`로 제자리에 더한다**. 결과는 같지만 비용은 (1−d)배.
