# 여러 register 토큰의 attention 패턴은 서로 같은가?

**한 줄 답**: 완전히 같지 않다. 일부 register는 장면 속 서로 다른 객체에 주목하는 흥미로운 패턴을 보이며, 이는 slot attention과 닮았다. 다만 이 분화는 아무도 강제하지 않았고 학습 중 자연히 창발한 것이다.

출처: *Vision Transformers Need Registers* (Darcet et al., ICLR 2024, arXiv:2309.16588) §3.4 "Qualitative evaluation of registers" 및 부록 G.

---

## 1. 저자들이 던진 질문

Register는 "모델이 forward pass 도중 정보를 저장·회수하는 빈 칸"으로 도입됐다. 출력이 어디에도 쓰이지 않으므로, 학습 신호 상으로는 **register끼리 서로 달라야 할 이유가 전혀 없다**. 그래서 §3.4의 질문은 이것이다.

> "We want to verify if they all exhibit similar attention patterns or whether a differentiation automatically emerges."
> (모두 비슷한 attention 패턴을 보이는가, 아니면 분화가 자동으로 나타나는가?)

검증 방법은 단순하다. 마지막 층에서 **[CLS] 토큰과 각 register 토큰이 patch 토큰들에게 주는 attention map**을 나란히 그려 본다.

## 2. Fig. 9 — 개별 이미지에서의 attention map

![Fig. 9: [CLS]와 register 토큰들의 attention map 비교](fig-1.jpeg)

*Figure 9: Comparison of the attention maps of the [CLS] and register tokens. Register tokens sometimes attend to different parts of the feature map, similarly to slot attention (Locatello et al., 2020). This behaviour was never required from the model, and emerged naturally from training.*

**그림을 실제로 뜯어본 관찰** (입력은 접시 위 커피잔 + 티스푼, 접시 위와 테이블 위에 놓인 각설탕들):

| 토큰 | 어디에 주목하는가 |
|---|---|
| `[CLS]` | 잔 안의 **커피 표면(원형 액체면)**이 가장 밝고, 각설탕들은 약하게만 반응. 즉 "이미지의 주제"에 해당하는 영역 |
| `[reg₀]` | 특정 물체에 붙지 않고 **접시 테두리·잔 윤곽선을 따라 넓게 흩뿌려진** 산발적 패턴. 객체보다는 배경·경계 쪽 |
| `[reg₆]` | **접시 위의 각설탕 하나**에 매우 밝고 좁게 집중. 다른 영역은 거의 죽어 있음 |
| `[reg₈]` | **티스푼**의 대각선 손잡이를 따라 길쭉하게 밝은 선. 명백히 "숟가락"이라는 하나의 물체를 잡고 있음 |
| `[reg₁₂]` | 다시 **커피 표면**에 집중 — [CLS]와 비슷하지만 동일하지는 않고 더 뭉쳐 있음 |

여기서 두 가지가 동시에 드러난다.

1. **분화는 실재한다.** reg₆(각설탕), reg₈(숟가락), reg₁₂(커피 면)는 서로 다른 객체를 잡는다. 4개 map이 서로 다른 그림이다.
2. **분화가 깔끔한 분할은 아니다.** reg₀처럼 특정 객체에 대응하지 않는 register도 있고, reg₁₂와 [CLS]처럼 영역이 겹치는 경우도 있다. 논문 표현이 정확히 이 뉘앙스다 — "registers do **not** have a **completely** aligned behavior", "**some selected** registers exhibit interesting attention patterns."

> 참고: 그림의 인덱스가 0, 6, 8, 12인 것으로 보아 register를 16개까지 늘린 모델(Fig. 8의 개수 ablation 계열)에서 **보기 좋은 것들을 골라** 실은 것이다. 본 실험의 기본 설정은 register 4개다. 즉 "모든 register가 각자 객체 하나씩을 담당한다"는 주장이 아니라, "그런 패턴이 **때때로**(sometimes) 나타난다"는 관찰에 가깝다.

## 3. Fig. 16 (부록 G) — 평균 attention map으로 본 positional focus 차이

한 장의 이미지는 우연일 수 있으므로, 저자들은 **ImageNet-22k의 임의 부분집합**에 모델을 돌려 마지막 층의 attention map을 **평균**했다. 개별 객체 정보는 씻겨 나가고, 각 토큰의 **위치적 편향(positional focus)** 만 남는다.

![Fig. 16: register와 [CLS]의 평균 attention map](fig-2.jpeg)

*Figure 16: Average attention map of registers and [CLS] token. There is a variability observed, with register 3 of this model focusing more on border areas. We also include the average attention map of a patch for comparison. The patch has a much more focused average attention.*

**그림을 실제로 뜯어본 관찰** (DINOv2+reg, register 4개):

- **(a) [CLS]**: 중앙에 크고 부드러운 밝은 blob. 가장자리로 갈수록 매끄럽게 어두워진다.
- **(b) reg₀**: [CLS]와 거의 판박이인 중앙 blob. 즉 **어떤 register는 [CLS]와 사실상 같은 위치 성향**을 갖는다.
- **(c) reg₁**: 중앙에 밝지만 blob이 훨씬 좁고 날카로우며, 상하좌우로 뻗은 **십자 모양** 구조에 네 모서리는 뚜렷이 어둡다.
- **(d) reg₂**: 밝은 부분이 중앙에서 **위쪽으로 치우쳐** 있고 세로로 길다 → 본문의 "register 2 tends to focus slightly more on the upper areas".
- **(e) reg₃**: **완전히 반전된 패턴**. 중앙이 검고 **네 변의 테두리(특히 좌·우·하단 가장자리)가 밝다** → "register 3 tends to focus on border areas".
- **(f) patch (비교군)**: 자기 위치 한 점만 극단적으로 밝은, 비교 불가할 만큼 **훨씬 뾰족한** 분포. register/[CLS]의 넓은 blob과 성격이 다르다는 대조군이다.

주의할 해석 포인트:
- ImageNet-22k는 장면 사진보다 **객체 중심(object-centric)** 이미지가 많다. 그래서 평균이 대체로 "가운데 blob"으로 나오는 것은 데이터 편향의 결과지, 모델이 늘 중앙만 본다는 뜻이 아니다.
- 그럼에도 reg₃만 테두리로 튄다는 사실은 **데이터 편향으로 설명되지 않는 개별 register의 역할 분화**를 보여준다. 저자들도 이를 Fig. 9와 "consistent"하다고, 즉 "some level of specialization"의 증거로 읽는다.

## 4. Slot Attention과의 비교 — 무엇이 닮았고 무엇이 다른가

### Slot Attention이란 (Locatello et al., NeurIPS 2020)

객체 중심 표현(object-centric representation)을 라벨 없이 뽑기 위한 모듈이다. 핵심 구조:

1. **슬롯 초기화**: K개의 latent 벡터("슬롯")를 공통 Gaussian에서 랜덤 샘플링한다. 슬롯들은 **교환 가능(exchangeable)** 하다 — 특정 객체 클래스에 미리 묶여 있지 않다.
2. **경쟁을 만드는 softmax 축 뒤집기**: 일반 attention은 각 query가 key들에 대해 softmax를 취한다(= key 축 정규화). Slot Attention은 슬롯을 query로 두되 **softmax를 슬롯 축에 대해** 취한다. 그러면 각 입력 위치에 대한 슬롯들의 가중치 합이 1이 되어, **하나의 입력 픽셀/특징을 어느 슬롯이 설명할지 슬롯들끼리 경쟁**하게 된다. 이 축 뒤집기가 "슬롯 간 분할"을 강제하는 장치다.
3. **반복 갱신**: 그 attention으로 입력을 가중 평균해 각 슬롯의 업데이트를 만들고, **GRU(+ residual MLP)** 로 슬롯을 갱신하는 과정을 T회(보통 3회) 반복한다. 반복할수록 슬롯이 특정 객체에 "달라붙는(bind)" 방향으로 수렴한다.
4. 이렇게 얻은 슬롯들을 디코딩해 비지도 객체 발견·속성 예측을 수행한다.

### 닮은 점

- **소수의 학습 가능한 추가 latent 토큰**이 patch/특징 집합을 attention으로 읽어들인다는 구조가 같다.
- 그 결과로 **latent마다 장면의 서로 다른 객체·영역이 배당**된다. Fig. 9의 reg₆(각설탕) / reg₈(숟가락) / reg₁₂(커피)는 슬롯이 객체에 binding된 그림과 시각적으로 매우 흡사하다. 논문이 "similarly to slot attention"이라 부른 이유가 이것이다.

### 결정적으로 다른 점

| | Slot Attention | Register |
|---|---|---|
| 분할의 근원 | **명시적 설계**: softmax 축을 슬롯 쪽으로 뒤집어 슬롯 간 경쟁을 강제 | **아무 제약 없음**. 일반 ViT self-attention 그대로, softmax는 key 축 정규화 |
| 갱신 방식 | 슬롯 단위 **반복 갱신(GRU + MLP, T회)** 으로 binding을 수렴시킴 | ViT 층을 한 번 통과할 뿐, 슬롯 전용 반복 루프 없음 |
| 목적 함수 | 객체 발견/재구성 등 **객체 중심 목표**를 직접 최적화 | 목표는 그냥 DINOv2/DeiT/CLIP의 원래 학습 목표. 객체 분할과 무관 |
| 출력 사용 | 슬롯 출력이 **모델의 산출물**(디코딩 대상) | register 출력은 **어디에도 쓰이지 않고 버려짐** |
| 분화의 성격 | 설계상 **보장**되는 것 | **창발(emergent)**. "nothing enforced this behavior", "emerged naturally from training" |
| 분화의 품질 | 슬롯 = 객체가 비교적 일관 | 부분적·비일관적. 어떤 register는 객체를 잡고, 어떤 것(reg₀, reg₃)은 경계/배경 성향 |

즉 **"결과가 닮았을 뿐, 메커니즘은 전혀 다르다"** 가 이 비교의 핵심이다. Slot attention은 객체 분할을 얻기 위해 아키텍처를 뜯어고쳤고, register는 artifact 제거라는 다른 목적으로 빈 토큰을 넣었을 뿐인데 부수적으로 비슷한 다양성이 나타났다.

## 5. 저자들이 남긴 숙제

§3.4의 마지막 문장은 이렇게 끝난다.

> "While nothing enforced this behavior, their activations had some natural diversity. **We leave the study of the regularization of registers for future work.**"

읽는 법:

- 현재의 분화는 **통제되지 않은 자연 발생물**이다. 어떤 register가 무엇을 담당할지 예측할 수 없고, 모델/시드/데이터가 바뀌면 달라질 수 있다(부록도 "register 3 **of this model**"이라고 한정한다).
- 따라서 "register를 정규화한다"는 것은 예컨대 register 간 **직교성·다양성 손실**을 걸거나, slot attention처럼 **경쟁 메커니즘을 명시적으로 부여**해서 이 창발적 특화를 의도적·재현적으로 만드는 방향을 뜻한다. 그러면 register가 artifact 흡수라는 위생적 역할을 넘어 **비지도 객체 분할 슬롯**으로 승격될 수 있다.
- 다만 이 논문의 범위는 거기까지가 아니다. 이 논문은 어디까지나 "artifact를 없애는 간단한 수정"이 주제이고, 객체 분화는 **관찰로만 보고된 부산물**이다. 카드의 답을 외울 때도 "흥미로운 패턴을 보인다"까지가 논문의 주장이고, "register가 객체 분할을 한다"로 과장하지 않는 것이 정확하다.

## 6. 시험에 나올 만한 한 줄 정리

- Q: 모든 register가 같은 attention을 보이는가? → **아니다(not completely aligned).**
- 근거 1(Fig. 9): 개별 이미지에서 register마다 각설탕·숟가락·커피 표면 등 **서로 다른 객체**를 잡는다.
- 근거 2(Fig. 16): ImageNet-22k 평균 attention에서 **reg₃는 테두리**, **reg₂는 약간 위쪽**, reg₀는 [CLS]와 유사한 중앙 — **positional focus가 register별로 다르다**.
- 비유: **slot attention**과 닮았다. 단, slot attention은 슬롯 축 softmax + 반복 갱신으로 경쟁을 **설계**한 것이고, register의 분화는 **아무 제약 없이 창발**한 것이다.
- 후속 과제: **register의 regularization**은 future work로 남겨졌다.

---

### Sources

- Darcet, Oquab, Mairal, Bojanowski. *Vision Transformers Need Registers*, arXiv:2309.16588 (§3.4, 부록 G / Fig. 9, Fig. 16)
- [Object-Centric Learning with Slot Attention (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html)
- [Object-Centric Learning with Slot Attention | MPI-IS](https://is.mpg.de/en/publications/locatelloetal20c)
- [slot-attention-pytorch (구현 참고: 슬롯 축 softmax, GRU 반복 갱신)](https://github.com/evelinehong/slot-attention-pytorch)
