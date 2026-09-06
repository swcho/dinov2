# Linear probing 평가에서 그리드 서치하는 3개 파라미터

> **Q.** linear probing 평가에서 그리드 서치하는 3개 파라미터는?
>
> **A.** learning rate(0.0001~0.5 중 13개 값), 사용할 output layer 수({1, 4}), average-pooled patch token을 class token과 concat할지 여부({yes, no})다. SGD로 12500 iteration, random-resized-crop 증강을 쓴다.

출처: DINOv2 논문(arXiv 2304.07193v2) **Appendix B.3 "Linear probing evaluation"**, 그리고 §7.1 "ImageNet Classification"의 linear 평가 문단.

---

## 1. Linear probing이 무엇을 측정하나

- 백본(ViT)의 가중치를 **완전히 고정(frozen)** 한 채, 그 출력 특징 위에 **선형 분류기 하나**(`nn.Linear`)만 학습한다. 백본은 finetuning하지 않는다.
- 측정 대상은 "특징이 클래스별로 **선형 분리 가능한가**"이다. 분류기 표현력이 최소이므로, 정확도가 높으면 그것은 분류기가 아니라 **특징 자체의 품질** 덕이다.
- 논문 §7.1은 "클래스가 실제로 선형 분리 가능하지 않을 수도 있지만, 단순성과 **재현 가능성**을 위해 선형 모델을 쓴다"고 명시한다. 비교 대상(iBOT ViT-L/16 on IN-22k 등)도 모두 같은 코드로 다시 평가해 공정성을 맞춘다.
- 결과: DINOv2 ViT-g/14는 ImageNet-1k linear 평가에서 이전 SOTA 대비 **+4.2%**, 그리고 ImageNet-ReaL/-V2로의 일반화 폭도 더 크다(Table 4). 반면 finetuning을 해도 향상은 "modest"(Table 5) — frozen 특징만으로 이미 잘 된다는 메시지.

## 2. 그리드 서치의 세 축

논문 원문(B.3):

> For linear probing we define 3 evaluation parameters: the learning rate, how many output layers we use, whether we concatenate the average-pooled patch token features with the class token (or use only the class token). We train our linear layer with SGD for 12500 iterations, using random-resized-crop data augmentation, and perform the following grid search:
> - learning rate in {0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5}
> - output layers in {1, 4}
> - concatenate average-pooled tokens in {yes, no}

### 2-1. Learning rate — 13개 값 (0.0001 ~ 0.5)

- 값들은 대략 **로그 간격**(1-2-5 패턴: 1e-4, 2e-4, 5e-4, 1e-3, …, 0.1, 0.2, 0.3, 0.5)으로 **3.5 자릿수(decade)** 를 훑는다.
- 왜 이렇게 넓어야 하나: 선형 분류기 하나라도 최적 LR은 크게 흔들린다.
  - **특징 스케일**: 백본마다(그리고 layer 수/avgpool 옵션마다) 입력 벡터의 norm이 다르다 → 같은 LR이라도 실질 step 크기가 다르다.
  - **입력 차원**: {1층, CLS만} vs {4층 + avgpool concat}은 차원이 4~5배 차이 → gradient 크기도 달라진다.
  - **클래스 수·데이터 크기**: ImageNet-1k(1000 클래스)와 소규모 fine-grained 데이터셋에서 최적 LR이 다르다. 같은 코드로 여러 데이터셋을 평가하므로 범위가 넓어야 한다.
- LR을 잘못 잡으면 "특징이 나쁘다"와 "분류기가 덜 수렴했다"를 구분할 수 없으므로, 특징 품질 측정을 위해서는 LR 축을 반드시 서치해야 한다.

### 2-2. Output layer 수 — {1, 4}

- **1**: 마지막 transformer block의 출력만 사용.
- **4**: 마지막 4개 block의 출력을 **concat**해서 사용(단순 평균이 아님 → 차원이 4배).
- 의미: SSL 목표(DINO/iBOT loss)는 마지막 층에서 정의되므로 마지막 층은 그 목표에 특화될 수 있다. 저층(마지막에서 2~4번째)은 좀 더 일반적·국소적인 정보를 보완해 주어, 선형 분류에 유리할 수 있다. 어느 쪽이 좋은지는 모델·데이터셋에 따라 달라서 두 경우를 다 본다.

### 2-3. Average-pooled patch token concat — {yes, no}

- **no**: class(CLS) token만 분류기 입력으로 사용.
- **yes**: CLS token 뒤에 **patch token들의 평균(average pooling)** 을 이어붙인다.
- 의미: CLS token은 attention으로 모은 **전역 요약**, patch 평균은 공간 위치들에 걸친 **국소 정보의 균일한 집계**다. 두 표현은 서로 다른 방식으로 이미지를 요약하므로, 합치면 선형 분리에 도움이 될 수 있다. DINO 계열 평가에서 오래 써 온 관행(DINO v1 linear 평가에서도 `avgpool` 옵션 존재).

### 2-4. 조합 수

13 (LR) × 2 (layers) × 2 (avgpool) = **52개 선형 분류기**를 동시에 학습한다.

## 3. 학습 설정

- **옵티마이저**: SGD (공개 코드 기준 momentum 0.9, weight decay 0)
- **iteration**: 12,500 (코드 기본값: `epochs=10 × epoch_length=1250 = 12500`)
- **스케줄**: cosine annealing → 0 (코드 기준)
- **증강**: random-resized-crop (코드는 `RandomResizedCrop(224, bicubic)` + 수평 뒤집기 0.5 + ImageNet 정규화). 크기·위치가 매번 바뀌는 crop을 쓰므로 분류기가 특정 프레이밍에 과적합하지 않는다.
- **손실**: 52개 분류기 각각의 CrossEntropy를 **합산**하여 한 번에 backward. 분류기들은 서로 독립(파라미터 공유 없음)이므로 합산 손실의 gradient는 각 분류기에 자기 손실의 gradient만 전달된다 — 즉 52개를 따로 학습한 것과 수학적으로 동일하다.

## 4. 왜 이 그리드 서치가 "비싸지 않은가"

논문 B.3 마지막 문장:

> Note that this grid search is not expensive, because at each iteration we perform inference on the backbone only once, then feed the output to all linear classifiers (each performing a single matrix multiplication).

- 비용의 거의 전부는 **frozen 백본의 forward**(ViT-g/14 등 대형 모델)이다. 이건 배치당 **1번**만 수행한다.
- 백본 출력(마지막 4개 block의 CLS + patch token)을 52개 분류기가 **공유**한다. 각 분류기는 `nn.Linear` 한 번 = 행렬곱 1회 → 52개 합쳐도 백본 forward에 비하면 무시할 수준.
- 백본은 `inference_mode` + autocast로 돌려 gradient도 저장하지 않는다.
- 결론: 52-조합 그리드 서치의 비용 ≈ 분류기 1개 학습 비용. (이 부분은 다음 카드의 주제.)

## 5. 결과 보고 관행 — validation 최고값

> We then report the highest accuracy value obtained on the validation set as is common practice.

- 52개 분류기 중 **validation top-1이 가장 높은 것**을 골라 그 값을 보고한다.
- 별도 held-out 세트로 하이퍼파라미터를 고르지 않으므로 엄밀히는 validation에 약간 낙관적으로 편향된 수치다. 그러나 SSL 커뮤니티의 표준 관행이고, 논문은 이를 보완하기 위해 **ImageNet-ReaL, ImageNet-V2** 같은 대체 테스트 세트에서도 정확도를 함께 보고한다(§7.1). 대체 세트에서는 validation에서 뽑은 best classifier를 그대로 적용한다.

## 6. 공개 코드(`dinov2/eval/linear.py`)와의 대조

논문의 세 축이 코드에 그대로 드러난다.

| 논문 | 코드 |
|---|---|
| output layers ∈ {1, 4} | `run_eval_linear`: `n_last_blocks_list = [1, 4]` (하드코딩). 백본은 `ModelWithIntermediateLayers(model, n_last_blocks=4)`로 마지막 4개 block의 `(patch_tokens, class_token)`을 한 번에 뽑는다. |
| avgpool concat ∈ {yes, no} | `setup_linear_classifiers`: `for avgpool in [False, True]`. `create_linear_input`은 마지막 `use_n_blocks`개 block의 **CLS를 concat**하고, avgpool이면 **마지막 block의 patch token 평균**(`torch.mean(..., dim=1)`)을 뒤에 이어붙인다. |
| LR 13개 | `--learning-rates` 인자. 코드 기본값은 `[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]` (역시 13개, 1-2-5 로그 간격). |
| SGD, 12500 iter | `torch.optim.SGD(momentum=0.9, weight_decay=0)`, `CosineAnnealingLR`, `epochs=10 × epoch_length=1250`. |
| random-resized-crop | `make_classification_train_transform()` = `RandomResizedCrop(224)` + `RandomHorizontalFlip(0.5)` + normalize. |
| validation 최고값 | `evaluate_linear_classifiers`가 모든 분류기의 top-1을 계산해 `best_classifier`를 고른다. 테스트 세트에서는 `best_classifier_on_val`로 그 이름을 고정해 사용. |

### LR 목록이 논문과 다른 이유 — `scale_lr`

```python
def scale_lr(learning_rates, batch_size):
    return learning_rates * (batch_size * distributed.get_global_size()) / 256.0
```

- 코드의 LR은 **batch 256 기준 base LR**이고, 실제 적용 LR은 전체 batch 크기에 비례해 스케일된다(linear scaling rule).
- 기본 `batch_size=128`(GPU당)이므로 예를 들어 8 GPU면 전체 batch 1024 → ×4 스케일. 코드 기본 목록 `1e-5 … 0.1`에 ×4를 하면 `4e-5 … 0.4`가 되어 논문의 `1e-4 … 0.5`와 거의 같은 범위가 된다. 즉 **논문의 숫자는 실제 적용 LR, 코드의 숫자는 base LR**이라고 이해하면 일치한다.
- 분류기 이름도 `classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}` 형태로 세 축을 그대로 인코딩해, 결과 JSON에서 어떤 조합이 이겼는지 바로 읽을 수 있다.

### 비용 공유 구조(코드 관점)

```
features = feature_model(data)          # 백본 forward 1회 (inference_mode, autocast)
outputs  = linear_classifiers(features) # AllClassifiers: 52개 nn.Linear에 같은 features 전달
loss     = sum(CE(v, labels) for v in outputs.values())
```

`AllClassifiers`는 `ModuleDict`를 돌며 같은 입력을 모든 분류기에 넣고 dict로 반환한다. 백본 계산은 그래프 밖(`inference_mode`)이라 backward는 52개 선형 층에서만 일어난다.

## 7. 한 줄 정리

**frozen 특징 위 선형 분류기**의 정확도로 특징 품질을 재되, 분류기 쪽 우연을 없애기 위해 **LR(13개, 로그 간격) × 출력층 수({1,4}) × patch 평균 concat({yes,no}) = 52개**를 **SGD 12,500 iter + random-resized-crop**으로 동시에 학습하고, **validation 최고값**을 보고한다. 백본 forward를 1회만 하고 모든 분류기가 공유하므로 비용은 거의 분류기 1개 수준이다.
