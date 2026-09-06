# LayerScale과 높은 Stochastic Depth를 쓰는데도 유지하는 이유는?

> **답:** linear probe 성능은 -1.2 떨어지지만 학습 중 NaN loss를 피해 학습 안정성을 크게 높인다. 이 안정성이 이후 개선들을 추가할 수 있는 토대가 되었다.

출처: DINOv2 논문 (Oquab et al., 2023, arXiv 2304.07193) §6.1 "Improved Training Recipe", Table 1 및 캡션.

---

## 1. 질문의 맥락 — Table 1 ablation

DINOv2는 iBOT을 베이스라인으로 두고 여러 컴포넌트를 **하나씩 누적**해 가며 ImageNet-1k에서 k-NN과 linear probe Top-1을 측정했다(ViT-L, ImageNet-22k로 사전학습). 저자들은 "linear probe 성능은 k-NN 성능에 의해 하한이 정해진다"는 경험을 근거로 **k-NN 성능을 최적화 목표**로 삼았다.

| 단계 | INet-1k k-NN | Δ | INet-1k linear | Δ |
|---|---|---|---|---|
| iBOT | 72.9 | | 82.3 | |
| + our reproduction | 74.5 | ↑1.6 | 83.2 | ↑0.9 |
| **+ LayerScale, Stochastic Depth** | **75.4** | **↑0.9** | **82.0** | **↓1.2** |
| + 128k prototypes | 76.6 | ↑1.2 | 81.9 | ↓0.1 |
| + KoLeo | 78.9 | ↑2.3 | 82.5 | ↑0.6 |
| + SwiGLU FFN | 78.7 | ↓0.2 | 83.1 | ↑0.6 |
| + Patch size 14 | 78.9 | ↑0.2 | 83.5 | ↑0.4 |
| + Teacher momentum 0.994 | 79.4 | ↑0.5 | 83.6 | ↑0.1 |
| + Tweak warmup schedules | 80.5 | ↑1.1 | 83.8 | ↑0.2 |
| + Batch size 3k | 81.7 | ↑1.2 | 84.7 | ↑0.9 |
| + Sinkhorn-Knopp | 81.7 | = | 84.7 | = |
| + Untying heads = DINOv2 | 82.0 | ↑0.3 | 84.5 | ↓0.2 |

표에서 거의 모든 행이 k-NN 또는 linear 중 하나 이상을 올리는데, **LayerScale + Stochastic Depth 행만 linear probe가 눈에 띄게 떨어진다(83.2 → 82.0, -1.2)**. k-NN은 0.9 오른다. 그런데도 저자들은 이 두 기법을 버리지 않았다. 캡션의 문장이 그 이유다:

> "Some modifications, like LayerScale and a high Stochastic Depth (rate=0.4), incur a decrease in linear probe performance, but have the benefits of increasing the stability of training by avoiding NaN loss values during training (Touvron et al., 2022). **Overall, these modifications allowed for the next set of improvements to be added.**"

본문(§6.1)에서도 같은 말을 반복한다: "Only LayerScale and Stochastic Depth incur a performance drop in linear probing but significantly improve the training stability in our experience."

즉 답의 핵심은 두 가지다.
1. **즉각적 이득**: NaN loss(학습 발산)를 피해 대규모 ViT-L/ViT-g 학습이 끝까지 돌아가게 한다.
2. **간접적 이득**: 학습이 안정적이어야 그 위에 128k prototypes, KoLeo, SwiGLU, 배치 3k 같은 공격적인 변경을 얹을 수 있다. Table 1이 누적(cumulative) ablation이라는 점을 떠올리면, 이 행 이후의 모든 향상(k-NN 75.4 → 82.0, linear 82.0 → 84.5)이 이 안정화 위에서 얻어진 것이다.

---

## 2. LayerScale이 무엇인가 (CaiT, Touvron et al., 2021)

**LayerScale**은 Touvron 등이 CaiT("Going deeper with Image Transformers", ICCV 2021)에서 제안한 기법으로, Transformer 블록의 **각 residual branch 출력에 채널별(learnable per-channel) 스케일 벡터를 곱해** 더한다.

```
x  ←  x + diag(λ_1, …, λ_d) · SA(LN(x))      # self-attention branch
x  ←  x + diag(λ'_1, …, λ'_d) · FFN(LN(x))   # feed-forward branch
```

- `λ`, `λ'`는 학습되는 파라미터로, **아주 작은 값(예: 1e-5 ~ 0.1)으로 초기화**한다. DINOv2는 Table 16에 따르면 **초기 LayerScale 값 1e-5**를 쓴다.
- 직관: 학습 초기에 각 블록의 residual branch가 거의 0에 가까운 기여만 하도록 만들어, 네트워크가 사실상 항등 함수(identity)에서 출발한다. 깊은 ViT(24층 ViT-L, 40층 ViT-g)에서 residual stream의 크기가 층을 거치며 폭발하는 것을 막고, 각 층이 "얼마나 기여할지"를 스스로 서서히 키우게 한다.
- CaiT 논문에서 이 기법은 **깊이(depth)를 늘릴 때 생기는 수렴 실패를 해결**하기 위해 도입되었고, 이후 DeiT III(Touvron et al., 2022 — DINOv2 캡션이 인용하는 논문)에서도 대형 ViT 학습 레시피의 표준 구성으로 쓰였다.

DINOv2의 ViT는 24~40개 블록에 SwiGLU FFN, float16 혼합정밀도(Table 16: "we train in float16 precision in all cases")까지 쓰기 때문에, 초기 activation 폭발 → overflow → NaN으로 이어지는 경로가 열려 있다. LayerScale은 이 경로를 원천 차단하는 장치다.

---

## 3. Stochastic Depth가 무엇인가 (Huang et al., 2016)

**Stochastic Depth**(drop path)는 Huang 등이 ResNet에 대해 제안한 정규화 기법으로, 학습 중 **각 residual 블록 전체를 확률 `d`로 통째로 건너뛰고(identity만 남김)** 테스트 시에는 모든 블록을 사용한다.

```
x  ←  x + m · f(x),   m ~ Bernoulli(1 − d)   (샘플 단위)
```

- Dropout이 뉴런을 끄는 것이라면, stochastic depth는 **층을 끈다**. 결과적으로 학습 시에는 기대 깊이가 얕은 여러 서브네트워크의 앙상블을 훈련하는 효과가 있다.
- DINOv2는 **drop rate 0.4(40%)**라는 상당히 높은 값을 쓴다(캡션의 "high Stochastic Depth (rate=0.4)", Table 16의 Drop-rate 0.4). 참고로 distilled된 S/B/L 모델은 0을 쓰고, **from-scratch로 학습하는 ViT-L/g에만 0.4**를 적용한다 — 안정성이 문제 되는 것은 대형 모델의 처음부터 학습이기 때문이다. §5 Model distillation에서도 distillation 시에는 "remove the masking and stochastic depth"라고 명시한다.
- DINOv2는 이 기법을 **효율적으로 재구현**했다(§5 "Efficient stochastic depth"): 드롭된 residual을 계산한 뒤 마스킹하는 대신, 배치 차원에서 샘플을 랜덤 셔플하고 앞의 `(1−d)·B`개만 잘라 블록을 통과시킨다. 그래서 40% 드롭이면 그만큼 연산과 메모리가 거의 비례해서 절약된다 — 높은 drop rate가 "정규화 + 안정성"뿐 아니라 "속도/메모리"에서도 이득이 되도록 설계한 것이다.

---

## 4. 왜 두 기법이 학습 안정성을 높이는가

두 기법은 모두 **residual branch의 기여를 줄이는 방향**으로 작동하며, 이는 깊은 ViT에서 학습 발산(NaN)을 막는 데 직접적으로 유효하다.

| 기법 | 안정화 메커니즘 |
|---|---|
| LayerScale | 초기 `λ ≈ 1e-5`로 각 블록 출력을 거의 0으로 억제 → residual stream 크기가 층을 거치며 지수적으로 커지는 것을 방지 → float16에서 overflow/NaN 예방. 각 층의 기여도를 학습이 스스로 조절하게 하므로 학습률에 덜 민감해진다. |
| Stochastic Depth (0.4) | 매 스텝 40%의 블록을 통째로 건너뛰므로 실제 forward/backward 경로가 짧아짐 → gradient가 통과하는 층 수가 줄어 폭발 확률 감소. 동시에 강한 정규화로 대형 모델의 과적합·붕괴(collapse)를 완화. |

DINOv2가 안정성에 특히 신경 쓰는 이유는 학습 설정이 전반적으로 공격적이기 때문이다.
- 자기지도 학습(DINO + iBOT loss)은 teacher/student EMA 구조와 centering/sharpening으로 인해 이미 collapse에 민감하다.
- ViT-g는 1.1B 파라미터, 40 블록, SwiGLU FFN, 배치 3072, 625k 반복, float16 학습이다. FSDP 절에서도 "MLP heads gradients are reduced in float32 to avoid training instabilities"라고 언급할 정도로 정밀도 관련 불안정을 의식하고 있다.
- 이런 조건에서 한 번 NaN이 나면 수천 GPU-시간이 날아간다. **linear probe -1.2는 한 번의 발산 비용에 비하면 훨씬 싸다.**

또한 저자들이 "linear probe는 k-NN에 의해 하한이 정해진다"고 보고 k-NN을 최적화했다는 점도 기억할 만하다. 이 행에서 k-NN은 오히려 +0.9 상승했으므로, 저자들의 기준으로는 손실만 있는 변경이 아니었다.

---

## 5. "이후 개선들의 토대"라는 맥락

Table 1은 위에서 아래로 컴포넌트를 **누적**하는 구조다. LayerScale + Stochastic Depth 행 다음에 오는 것들은
- **+128k prototypes** (DINO head의 프로토타입 수를 크게 늘림),
- **+KoLeo** (feature를 균일하게 퍼뜨리는 정규화 항),
- **+SwiGLU FFN**, **+Patch size 14**, **+Teacher momentum 0.994**,
- **+Tweak warmup schedules**, **+Batch size 3k**, **+Sinkhorn-Knopp**, **+Untying heads**

이며, 이 중 다수(프로토타입 확대, 배치 확대, warmup 조정, SwiGLU)는 그 자체가 학습을 불안정하게 만들 수 있는 변경이다. 저자들의 표현 "these modifications allowed for the next set of improvements to be added"는 곧, **안정화 장치를 먼저 깔아두지 않았다면 이후 행의 실험들이 NaN으로 실패해 애초에 측정조차 못 했을 것**이라는 의미다. 최종 DINOv2는 이 행(k-NN 75.4 / linear 82.0)에서 출발해 k-NN 82.0 / linear 84.5까지 올라갔고, 결론(§8)에서도 "an improved training recipe with better hyperparameters and regularization (Table 1)"을 성능의 첫 번째 요인으로 꼽는다.

---

## 요약 카드

- **무엇을**: LayerScale(CaiT; residual branch에 학습 가능한 채널별 스케일, 초기값 1e-5) + Stochastic Depth 0.4(블록 단위 drop path, 효율적 구현으로 연산까지 절약).
- **비용**: Table 1에서 linear probe 83.2 → 82.0 (**-1.2**). k-NN은 74.5 → 75.4 (+0.9).
- **이득**: 대형 ViT-L/g의 float16 자기지도 학습에서 **NaN loss 회피 → 학습 안정성**.
- **왜 유지**: 이 안정성 위에서 128k prototypes, KoLeo, SwiGLU, 배치 3k 등 후속 개선을 쌓을 수 있었기 때문. 누적 ablation의 "토대" 역할.
- **적용 범위**: from-scratch 학습(ViT-L/g)에만 drop rate 0.4; distilled S/B/L은 0, distillation 시 stochastic depth 제거.
