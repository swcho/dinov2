# Linear probing 그리드 서치가 비싸지 않은 이유

**질문**: linear probing 그리드 서치가 비싸지 않은 이유는?

**답**: 매 iteration마다 백본 추론을 한 번만 수행하고 그 출력을 모든 linear classifier에 먹이기 때문이다. 각 classifier는 행렬 곱 한 번만 수행한다.

---

## 1. 출처: DINOv2 논문 Appendix B.3

논문(2304.07193v2) Appendix B.3 "Linear probing evaluation"은 linear probing에 3개의 평가 하이퍼파라미터를 두고 SGD로 12,500 iteration 학습하며 다음 그리드를 탐색한다고 밝힌다.

| 축 | 값 | 개수 |
|---|---|---|
| learning rate | {0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5} | 13 |
| 사용하는 출력 레이어 수 (마지막 n개 블록) | {1, 4} | 2 |
| average-pooled patch token을 class token에 concat | {yes, no} | 2 |

곱하면 **13 × 2 × 2 = 52개 조합**이다. 그리고 validation set에서 가장 높은 정확도를 보고한다. 이어지는 문장이 이 카드의 원문이다.

> Note that this grid search is not expensive, because at each iteration we perform inference on the backbone only once, then feed the output to all linear classifiers (each performing a single matrix multiplication).

즉 52개 조합을 52번 따로 학습하는 것이 아니라, **한 번의 학습 루프 안에서 52개 선형 분류기를 동시에** 학습한다. 비용의 거의 전부인 백본 forward는 iteration당 1회만 일어난다.

## 2. 비용 구조를 정량적으로 보기

### 2.1 백본 forward 1회 (ViT-g/14, 224×224)

- 임베딩 차원 D = 1536, 24 heads, 40 blocks, 파라미터 약 **1.1B** (논문 Sec. 5)
- 토큰 수 N = (224/14)² + 1(cls) = 16² + 1 = **257**
- Transformer forward FLOPs 어림: 파라미터 하나가 토큰 하나당 1 MAC = 2 FLOPs를 소비하므로
  `2 × 1.1×10⁹ × 257 ≈ 5.7×10¹¹ FLOPs ≈ 0.57 TFLOPs / image`
- 블록 단위로 세어도 비슷하다: 블록당 토큰당 약 24·D² FLOPs(QKV 3D² + out-proj D² + MLP 8D², MAC 기준 12D²) = 24 × 1536² ≈ 5.7×10⁷, × 257 토큰 × 40 블록 ≈ 5.8×10¹¹. attention score(QKᵀ, AV)는 4·N·D per token으로 블록당 0.4 GFLOPs 정도라 여기서는 무시할 수 있는 수준.

→ **이미지 1장당 약 0.6 TFLOPs.** 배치 128이면 iteration당 약 75 TFLOPs.

### 2.2 선형 분류기 52개 (ImageNet-1k, C = 1000)

각 분류기는 `D_in × C` 행렬 곱 한 번이다. D_in은 조합에 따라 다르다.

| n blocks | avgpool | D_in | 계산 |
|---|---|---|---|
| 1 | no | 1,536 | cls 1개 |
| 1 | yes | 3,072 | cls + patch mean |
| 4 | no | 6,144 | cls 4개 concat |
| 4 | yes | 7,680 | cls 4개 + 마지막 블록 patch mean |

LR 13개마다 위 4종이 하나씩 있으므로, 이미지 1장에 대한 전체 forward MAC은

`13 × (1536 + 3072 + 6144 + 7680) × 1000 = 13 × 18,432 × 1000 ≈ 2.4×10⁸ MAC ≈ 0.48 GFLOPs`

backward는 가중치 gradient(입력 벡터와 출력 gradient의 outer product)만 필요하고, 입력은 백본에서 detach된 상수라 입력 gradient를 계산할 필요가 없다. 따라서 학습 시 총 비용은 forward의 약 2배, **≈ 1 GFLOPs / image**.

### 2.3 비율

```
52개 head 학습 비용   ≈ 1.0 × 10⁹ FLOPs
ViT-g forward 1회     ≈ 5.7 × 10¹¹ FLOPs
─────────────────────────────────────────
비율                  ≈ 0.17 %  (약 1/570)
```

가장 큰 head 하나(7680×1000 ≈ 7.7M MAC)조차 백본의 0.003% 수준이다. 따라서

**그리드 서치 전체 비용 ≈ 백본 forward 1회 × 12,500 iteration ≈ "분류기 1개만 학습할 때"와 거의 같다.**

52개를 따로 돌렸다면 백본 forward가 52배 반복되어 약 52배의 비용이 든다. 반면 이 설계에서 52배 늘어나는 것은 전체의 0.17%인 head 부분뿐이다.

또 하나 중요한 절약: 백본이 frozen이므로 **백본 backward가 없다**. fine-tuning이라면 forward의 약 2배인 backward가 추가되어 이미지당 약 1.7 TFLOPs가 되는데, linear probing은 0.57 TFLOPs로 끝난다. 백본은 `autocast(fp16/bf16)`로 돌리고 head만 fp32로 학습하면(`output.float()`) 메모리·속도도 더 아낀다.

### 2.4 메모리 관점(주의점)

계산량은 무시할 수준이지만 파라미터 수는 그렇지 않다. head 전체 파라미터는 13 × 18,432 × 1000 ≈ **2.4×10⁸개**(fp32 약 1 GB, SGD momentum 포함 약 2 GB). ViT-g 가중치(fp16 약 2.2 GB)와 비슷한 규모라 GPU 한 장에 충분히 올라가지만 "공짜"는 아니다. 비싸지 않은 이유는 **연산량이 백본에 비해 무시할 만하다**는 것이지 head가 작다는 뜻은 아니다.

## 3. "모든 classifier에 같은 출력을 먹인다"의 구현 방식

핵심 아이디어를 의사코드로 쓰면 다음과 같다.

```python
for images, labels in loader:                      # random-resized-crop 적용된 배치
    with torch.inference_mode(), autocast():
        feats = backbone.get_intermediate_layers(images, n=4, return_class_token=True)
        # feats: 마지막 4개 블록의 (patch tokens, cls token) 튜플 리스트, grad 없음

    losses = []
    for head in heads:                               # 52개
        x = build_input(feats, head.n_blocks, head.avgpool)   # concat/mean만, 행렬곱 없음
        logits = head.linear(x)                      # D_in × C 행렬 곱 1회
        losses.append(CE(logits, labels))
    sum(losses).backward()                           # head들끼리 파라미터가 분리돼 있어
    optimizer.step()                                 # 각 head는 자기 loss의 gradient만 받음
```

포인트:

1. **백본은 배치당 1회**만 돌고, 가장 많은 블록을 요구하는 조합(n=4)에 맞춰 마지막 4개 블록 출력을 한꺼번에 뽑아 둔다. n=1 조합은 그중 마지막 1개만 쓴다.
2. 52개 head의 loss를 **단순 합**해 한 번 `backward()` 한다. head들은 파라미터를 공유하지 않으므로 합의 gradient는 각 head에 대해 자기 loss의 gradient와 정확히 같다. 즉 수학적으로 52개를 독립적으로 학습한 것과 동일하다.
3. LR은 optimizer의 **param group**으로 head마다 다르게 준다. 하나의 SGD 인스턴스가 52개 그룹을 가진다.
4. 데이터 증강(random-resized-crop)이 매 iteration 다른 crop을 만들기 때문에 특징을 미리 한 번 뽑아 디스크에 캐시하는 방식은 쓸 수 없다. 그래서 "백본을 매 iteration 돌리되 1회만" 이라는 절충이 나온다.

## 4. 가능한 전제

- **백본이 frozen**이다. 학습되는 파라미터가 백본에 없으니 백본을 `inference_mode`로 돌려도 되고, 그 출력은 gradient가 필요 없는 상수 텐서가 된다.
- 따라서 **같은 특징을 여러 소비자가 재사용**할 수 있다. 백본을 학습시키는 fine-tuning이라면 각 hyperparameter 조합마다 백본 가중치가 달라지므로 forward를 공유할 수 없고, 이 트릭은 성립하지 않는다.
- 각 head가 선형이라 입력 전처리(concat, mean)를 제외하면 연산이 행렬 곱 1회뿐이며, head 간 상호작용이 없다.

## 5. 공개 코드 `dinov2/eval/linear.py`와 대조

실제 구현은 위 의사코드와 거의 1:1로 대응한다.

**특징 추출: `ModelWithIntermediateLayers` (`dinov2/eval/utils.py`)**
생성자에서 `feature_model.eval()`을 호출하고, `forward`는 `torch.inference_mode()` + autocast 안에서 `get_intermediate_layers(images, n_last_blocks, return_class_token=True)`를 호출한다. `run_eval_linear`에서 `n_last_blocks_list = [1, 4]`, `n_last_blocks = max(...) = 4`로 만들어지므로 백본은 항상 마지막 4개 블록의 출력을 한 번에 반환한다.

**입력 조립: `create_linear_input(x_tokens_list, use_n_blocks, use_avgpool)`**
`x_tokens_list[-use_n_blocks:]`로 필요한 블록만 잘라 cls token들을 `torch.cat`하고, `use_avgpool`이면 마지막 블록의 patch token 평균을 뒤에 붙인다. 마지막에 `.float()`로 fp32 변환. 행렬 곱은 없다.

**분류기 1개: `LinearClassifier`**
`nn.Linear(out_dim, num_classes)` 하나를 가지며, `forward`는 `create_linear_input(...)` 뒤 `self.linear(output)` — 논문의 "single matrix multiplication"이 그대로다. 가중치 초기화는 N(0, 0.01), bias 0.

**52개 묶음: `AllClassifiers`**
`nn.ModuleDict`를 감싸고, `forward(inputs)`가 `{k: v.forward(inputs) for k, v in self.classifiers_dict.items()}`를 반환한다. **같은 `inputs`(백본 출력)를 모든 head에 그대로 넘기는** 것이 이 클래스의 전부다.

**그리드 생성: `setup_linear_classifiers(sample_output, n_last_blocks_list, learning_rates, batch_size, num_classes)`**
```
for n in n_last_blocks_list:          # [1, 4]
    for avgpool in [False, True]:
        for _lr in learning_rates:    # 13개
```
3중 루프로 정확히 2×2×13 = 52개 `LinearClassifier`를 만들고, 이름을 `classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr}`로 붙여 dict에 넣는다. 동시에 `optim_param_groups.append({"params": ..., "lr": lr})`로 head별 param group을 쌓는다. LR은 `scale_lr`로 `lr × (batch_size × world_size) / 256` 스케일링한다. 마지막에 `AllClassifiers`로 감싸고 분산 환경이면 DDP로 래핑한다(백본이 아니라 head 묶음만 DDP).

**학습 루프: `eval_linear`**
```python
features = feature_model(data)          # 백본 1회
outputs = linear_classifiers(features)  # 52개 head 동시 forward
losses = {f"loss_{k}": nn.CrossEntropyLoss()(v, labels) for k, v in outputs.items()}
loss = sum(losses.values())
optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
```
optimizer는 `run_eval_linear`에서 `torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)` 하나로 만들고 cosine schedule을 건다.

**평가와 선택: `evaluate_linear_classifiers`**
`@torch.no_grad()`로 validation을 돌리며 head별 `LinearPostprocessor`와 metric을 만들어 한 번의 pass로 52개 정확도를 모두 얻고, top-1이 가장 높은 `best_classifier`를 기록한다. 논문의 "report the highest accuracy value obtained on the validation set"에 해당한다. 이후 `test_on_datasets`는 val에서 고른 head 이름을 넘겨 다른 테스트셋(ImageNet-V2, ReaL 등)에서는 그 head만 보고한다.

**논문과 코드의 작은 차이**
코드 기본값 `learning_rates=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]`은 논문 표의 {1e-4 ... 0.5}보다 한 단계 낮은 범위지만 개수는 동일하게 13개다. 코드는 배치 크기에 따라 LR을 스케일링하므로 실효 LR 범위는 배치 설정에 따라 달라진다.

## 6. 왜 이 설계가 "많은 벤치마크 × 많은 모델" 평가를 감당하게 해주는가

DINOv2 논문은 linear probing을 ImageNet-1k뿐 아니라 ImageNet-V2/ReaL, 12개 fine-grained 분류 벤치마크, 비디오 분류, ADE20k/Cityscapes 세그멘테이션(linear head), 깊이 추정(linear head) 등에 걸쳐 수행한다. 모델도 ViT-S/B/L/g 4종에 더해 OpenCLIP, EVA-CLIP, iBOT, DINO, MAE 등의 경쟁 모델을 **자기 코드로 다시 linear eval** 한다(Sec. 7.1: "For all models, we run the linear evaluation using our code").

- 그리드 서치를 하이퍼파라미터 조합 수만큼 반복하면 (벤치마크 수) × (모델 수) × 52 번의 백본 학습 루프가 필요하다.
- 이 설계에서는 (벤치마크 수) × (모델 수) × **1** 번이면 된다. 52라는 배수가 통째로 사라진다.
- 그러면서도 매 모델·벤치마크에 대해 LR/레이어 수/pooling을 공정하게 튠한 값을 보고할 수 있다. "하이퍼파라미터를 상대 모델에 불리하게 고정했다"는 비판을 피하면서 비교 표(Table 4 등)를 채울 수 있는 것이다.
- 백본 backward가 없고 head가 가벼우므로 가장 큰 ViT-g조차 평가 비용은 사전학습 대비 무시할 수준이다. 이것이 논문이 "frozen features가 out-of-the-box로 좋다"는 주장을 다수 벤치마크에서 일관되게 검증할 수 있었던 실무적 기반이다.

## 한 줄 정리

백본(≈0.6 TFLOPs/image)은 iteration마다 한 번만 돌리고, 그 frozen 출력을 52개 선형 head(합쳐도 ≈1 GFLOPs/image, 백본의 0.2%)에 공유해서 loss를 합산·동시 학습하므로, 52개 조합의 그리드 서치 비용이 분류기 1개 학습 비용과 거의 같아진다.
