# 깊이를 배운 적이 없는데 linear probe가 깊이를 읽어낸다는 것의 의미

**질문**: 깊이 정보로 학습하지 않았는데 linear probe가 깊이를 예측할 수 있다는 사실의 의미는?

**답**: DINOv2(및 OpenCLIP) 특징이 깊이 같은 복잡한 정보를 **선형적으로 분리 가능한 형태로** 담고 있다는 뜻이다. 즉 필요한 정보가 "readily available"(바로 꺼내 쓸 수 있는) 상태로 인코딩되어 있다.

---

## 1. 논문의 해당 문장

DINOv2 논문(Oquab et al., 2023) 7.4절 "Qualitative Results"의 Figure 7 설명 부분:

> These results highlight that our features, as well as the features extracted from OpenCLIP, are able to **linearly separate complex information such as depth, even though neither was trained with this type of information**. However, our features lead to a much smoother depth estimation, with less artifacts.

그리고 결론(8절)에서 같은 관찰을 일반화한다:

> A few properties emerge from these models, such as an understanding of object parts and **scene geometry** regardless of the image domains. [...] This paper also demonstrates that these visual features are compatible with classifiers as simple as linear layers — meaning the underlying information is **readily available**.

두 문장을 이으면 이 카드의 핵심이 된다: "선형 층 하나로 읽힌다 = 정보가 이미 정리된 형태로 특징 안에 있다 = 학습 목표에 없던 성질이 **창발**했다".

---

## 2. 왜 "linear probe로 예측 가능"이 "정보가 정리되어 있다"는 증거인가

표현학습(representation learning)에서 특징의 품질을 평가하는 표준 절차가 **linear probing**이다.

1. 백본(ViT)을 **완전히 고정(frozen)** 한다. 가중치를 한 톨도 바꾸지 않는다.
2. 그 위에 **선형 층 하나**(행렬 곱 + 편향, 즉 $y = Wx + b$)만 얹어 목표 과제로 학습한다.
3. 그 성능으로 특징을 평가한다.

선형 층은 표현력이 극단적으로 제한적이다. 입력 특징 벡터의 좌표들을 **가중합**하는 것 이상은 못 한다. 곱하거나, 조건을 따지거나, 두 좌표의 조합을 새로 만들 수 없다. 그래서 다음 논리가 성립한다.

- 선형 층이 어떤 정보 $z$(예: 깊이)를 잘 맞힌다
- ⇒ 특징 공간 안에 "$z$가 커지는 방향" $w$가 이미 존재해서, $w \cdot x + b$만 계산하면 $z$가 나온다
- ⇒ 백본이 이미 $z$를 **분리해서(disentangled) 특정 방향에 정렬해** 놓았다
- ⇒ 정보를 "추출"하는 일은 백본이 다 했고, probe는 그저 "읽기"만 한 것이다

반대로 정보가 특징 안에 **얽혀** 있다면(예: 여러 좌표의 비선형 조합으로만 복원 가능), 선형 층은 그것을 풀어낼 수 없어 성능이 낮게 나온다. 즉 linear probe 성능은 "정보가 있는가"가 아니라 "정보가 **얼마나 꺼내 쓰기 쉬운 형태로** 있는가"를 잰다. 논문이 ImageNet 분류에서도 "classes may not be linearly separable"임을 알면서도 굳이 선형 모델을 쓰는 이유(7.1절)가 바로 이것이다 — 재현성이 높고, 특징 자체의 품질만 분리해 측정할 수 있기 때문이다.

---

## 3. 왜 놀라운가: 깊이는 사전학습 목적함수 어디에도 없다

DINOv2의 학습 신호(4절 "Discriminative Self-supervised Pre-training")를 보면 깊이·3D·기하에 관한 것이 **전혀** 없다.

| 손실 항 | 내용 | 깊이와의 관계 |
|---|---|---|
| Image-level (DINO) | 같은 이미지의 **다른 crop** 두 장에 대해 student/teacher의 [CLS] 토큰 출력 분포가 일치하도록 cross-entropy | 없음 |
| Patch-level (iBOT) | student 입력의 일부 패치를 **마스킹**하고, teacher가 본 그 패치의 토큰을 예측 | 없음 |
| KoLeo | 배치 내 특징이 공간에 골고루 퍼지도록 하는 엔트로피 정규화 | 없음 |
| Sinkhorn-Knopp centering | teacher 출력 분포의 균형 | 없음 |

레이블도 없고, 깊이 센서 데이터도 없고, 텍스트 캡션도 없다(OpenCLIP은 캡션은 있지만 역시 깊이는 없다). 그런데 이 특징 위에 선형 층 하나만 얹으면 "이 픽셀은 카메라에서 몇 미터 떨어졌는가"가 나온다. 이것이 논문이 "emerge"(창발)라는 단어를 쓰는 이유다.

---

## 4. 어떤 학습 신호가 깊이를 부산물로 만들어내는가

논문이 직접 인과를 증명하진 않지만, 목적함수의 구조에서 합리적으로 추론할 수 있다.

**(a) 다중 crop 일관성(multi-crop consistency)**
같은 장면의 서로 다른 크기·위치의 crop이 같은 표현을 내야 한다. 물체가 크게 잡히든 작게 잡히든, 가운데 있든 구석에 있든 "같은 물체"로 인식하려면 모델은 **크기·원근·가림(occlusion)** 에 불변인 요소와 그에 따라 변하는 요소를 내부적으로 구분해야 한다. 이 과정에서 "얼마나 가까이/멀리 있는가"에 해당하는 축이 특징 안에 자연스럽게 생긴다.

**(b) 패치 수준 예측(iBOT masked image modeling)**
가려진 패치의 내용을 주변 패치로부터 맞히려면, 장면의 **국소 구조**(바닥은 아래로 갈수록 가까워진다, 벽은 평면이다, 물체는 경계에서 깊이가 불연속이다)를 알아야 한다. Table 3b에서 이 항을 빼면 dense task 성능이 약 3% 떨어지는 것이 그 방증이다. 깊이 추정도 dense task다.

**(c) 대규모·다양한 데이터(LVD-142M)**
1억 4200만 장의 큐레이션된 이미지는 실내·실외·다양한 카메라 거리를 폭넓게 덮는다. 데이터가 다양할수록 "이 장면들을 모두 일관되게 설명하는 가장 경제적인 내부 변수"로 기하 정보가 선택될 압력이 커진다. Table 2에서 LVD-142M이 ImageNet-22k보다 ImageNet 밖 도메인에서 일관되게 낫다는 것과 맥락이 같다.

**(d) 참고: 캡션 기반 학습의 한계**
Table 11에서 OpenCLIP ViT-G(캡션 학습)가 더 작은 iBOT ViT-L(SSL)에도 밀린다. 논문은 이를 "caption-based feature learning fails to learn subtle patterns like this one"이라고 해석한다. 텍스트는 "의자"라고는 말하지만 "의자가 2.3 m 앞에 있다"고는 거의 말하지 않기 때문이다. 그런데도 OpenCLIP 역시 선형으로 깊이가 어느 정도 읽힌다는 점이 Figure 7의 두 번째 포인트다 — 대규모 시각 사전학습 자체가 기하를 부산물로 만든다.

---

## 5. 정성 결과: Figure 7

![Figure 7. OpenCLIP-G vs DINOv2-g, frozen 특징 + 선형 층으로 얻은 분할(1행)과 깊이(2~4행)](fig-1.jpeg)

그림에서 실제로 볼 수 있는 것:

- **2행(NYUd)·3행(SUN-RGBd)**: 밝을수록 가깝고 어두울수록 멀다. OpenCLIP-G 열은 얼룩·노이즈가 심하고 물체 경계가 흐릿하다. DINOv2-g 열은 소파·테이블·의자의 윤곽이 깊이 맵에서 깨끗하게 살아 있고, 방 안쪽으로 갈수록 어두워지는 **연속적 깊이 기울기**가 뚜렷하다.
- **3행 오른쪽(SUN-RGBd, 흔들의자)**: 논문이 명시적으로 지적하는 예. OpenCLIP은 의자를 완전히 놓치지만("completely ignored"), DINOv2는 의자를 배경보다 가깝게 정확히 위치시킨다.
- **4행(KITTI, 실외)**: 도로가 아래(가까움, 밝음)에서 위(멀리, 어두움)로 자연스럽게 이어지고, 도로 위 자동차가 주변 도로면과 구별된다. NYUd 실내 데이터로 학습한 선형 층이 실외에서도 동작하는 것(SUN-RGBd 제로샷 전이 포함)이 "장면 기하를 도메인 무관하게 이해한다"는 결론 문장의 근거다.
- **핵심**: 두 열 모두 **동일한 선형 층 하나**다. 차이는 온전히 그 아래 frozen 특징이 깊이를 얼마나 "정리된 형태로" 담고 있느냐에서 온다.

---

## 6. 정량 결과: Table 11 (RMSE, 낮을수록 좋음)

세 가지 디코더를 비교한다.

- **lin. 1**: 마지막 층 패치 토큰 + [CLS] 토큰을 이어붙이고, 4배 upsample 후 **선형 층 하나**로 256개 깊이 구간 분류
- **lin. 4**: 같은 방식이되 4개 층(ViT-g는 $l=\{10,20,30,40\}$)의 토큰을 이어붙임 — 여전히 선형
- **DPT**: Dense Prediction Transformer 디코더(Ranftl et al., 2021), **비선형·다층**, 회귀

| Method | Arch. | NYUd lin.1 | lin.4 | DPT | KITTI lin.1 | lin.4 | DPT | NYUd→SUN lin.1 | lin.4 | DPT |
|---|---|---|---|---|---|---|---|---|---|---|
| (SOTA, Li et al. 2022b) | | | | 0.330 | | | 2.10 | | | 0.421 |
| OpenCLIP | ViT-G/14 | 0.541 | 0.510 | 0.414 | 3.57 | 3.21 | 2.56 | 0.537 | 0.476 | 0.408 |
| MAE | ViT-H/14 | 0.517 | 0.483 | 0.415 | 3.66 | 3.26 | 2.59 | 0.545 | 0.523 | 0.506 |
| DINO | ViT-B/8 | 0.555 | 0.539 | 0.492 | 3.81 | 3.56 | 2.74 | 0.553 | 0.541 | 0.520 |
| iBOT | ViT-L/16 | 0.417 | 0.387 | 0.358 | 3.31 | 3.07 | 2.55 | 0.447 | 0.435 | 0.426 |
| DINOv2 | ViT-S/14 | 0.449 | 0.417 | 0.356 | 3.10 | 2.86 | 2.34 | 0.477 | 0.431 | 0.409 |
| DINOv2 | ViT-B/14 | 0.399 | 0.362 | 0.317 | 2.90 | 2.59 | 2.23 | 0.448 | 0.400 | 0.377 |
| DINOv2 | ViT-L/14 | 0.384 | 0.333 | 0.293 | 2.78 | 2.50 | 2.14 | 0.429 | 0.396 | 0.360 |
| DINOv2 | ViT-g/14 | **0.344** | **0.298** | **0.279** | **2.62** | **2.35** | **2.11** | **0.402** | **0.362** | **0.338** |

읽는 법:

1. **lin. 1 열만 봐도 DINOv2-g(0.344)는 OpenCLIP-G(0.541)보다 훨씬 좋다.** 둘 다 선형 층 하나이므로 이 차이는 전적으로 특징 품질이다. 심지어 DINOv2-g의 **선형** 결과(0.344)가 OpenCLIP-G의 **DPT** 결과(0.414)보다 낫다 — 비선형 디코더로 애써 파내는 것보다 좋은 특징을 선형으로 읽는 편이 낫다는 뜻.
2. **DINOv2-g + DPT(0.279, 2.11, 0.338)는 전용 SOTA(0.330, 2.10, 0.421)를 따라잡거나 넘어선다**, 백본이 frozen인 상태로.
3. **NYUd → SUN-RGBd 제로샷**: 실내에서 학습한 헤드가 다른 데이터셋에서도 lin.1 기준 0.402로 동작한다. 기하 정보가 특정 데이터셋의 통계가 아닌 일반적 성질로 인코딩되어 있다는 증거다.

---

## 7. lin.1 vs DPT 격차 = "선형적으로 얼마나 readily available인가"의 척도

이 표에서 가장 개념적으로 중요한 것은 **같은 백본 안에서 lin.1 → lin.4 → DPT로 갈 때 성능이 얼마나 더 좋아지느냐**다.

- DPT는 비선형 디코더라서, 특징 안에 **얽힌 채로** 들어 있는 깊이 정보까지 풀어서 쓸 수 있다. 즉 DPT 성능 ≈ "특징에 깊이 정보가 **존재**하는 총량"의 근사치.
- lin.1 성능 ≈ "그중 **선형으로 바로 읽히는** 양".
- 따라서 **(lin.1 − DPT) 격차가 작을수록**, 깊이 정보가 이미 잘 정리되어 있어 비선형 디코더가 추가로 해 줄 일이 적다는 뜻이다.

NYUd에서 계산해 보면:

| 백본 | lin.1 | DPT | 격차 (lin.1 − DPT) | 상대 격차 |
|---|---|---|---|---|
| OpenCLIP ViT-G | 0.541 | 0.414 | 0.127 | 31% |
| MAE ViT-H | 0.517 | 0.415 | 0.102 | 25% |
| iBOT ViT-L | 0.417 | 0.358 | 0.059 | 16% |
| DINOv2 ViT-g | 0.344 | 0.279 | 0.065 | 23% |

OpenCLIP은 DPT를 붙이면 31%나 개선된다 — 정보가 있긴 하지만 선형으로는 잘 안 읽히는, 즉 "얽혀 있는" 상태다. DINOv2는 절대 성능이 높으면서 격차도 상대적으로 작다. 그리고 lin.4(중간 층까지 이어붙임)가 lin.1보다 항상 낫다는 것은, 깊이의 일부는 마지막 층보다 **중간 층**에 더 선형적으로 놓여 있음을 시사한다 — 마지막 층은 의미(semantic) 쪽으로 더 압축되기 때문이다.

이렇게 보면 "linear probe로 깊이 예측 가능"이라는 관찰은 이진적 사실이 아니라 **정도의 문제**이고, Table 11은 그 정도를 여러 백본에 대해 재는 자(ruler)다.

---

## 8. 논문 결론과의 연결: 창발적 성질

결론의 두 문장을 이 카드의 언어로 다시 쓰면:

- "understanding of **scene geometry** regardless of the image domains **emerge**" → 깊이·기하는 학습 목표가 아니었지만, 대규모 SSL의 부산물로 특징 안에 생겨났다. Figure 7의 NYUd/SUN-RGBd/KITTI 전이가 "regardless of domains"의 근거다.
- "compatible with classifiers as simple as linear layers — meaning the underlying information is **readily available**" → 생겨난 정보가 얽힌 채 숨어 있는 것이 아니라, 선형 층 하나로 읽힐 만큼 정돈된 좌표축에 놓여 있다.

논문은 이어서 이 성질을 LLM의 instruction emergence에 비유하며, 모델·데이터 규모를 더 키우면 더 많은 성질이 창발할 것으로 기대한다. 또 "readily available"이라는 특성이 후속 계획 — 시각 특징을 **단어 토큰처럼** 언어 모델에 그대로 먹이는 것 — 의 전제가 된다. 선형으로 읽히는 특징이라야 다른 시스템이 복잡한 디코더 없이 필요한 정보를 꺼내 쓸 수 있기 때문이다.

---

## 9. 한 줄 정리

> 사전학습 손실에 깊이가 한 번도 등장하지 않았는데 frozen 특징 위 **선형 층 하나**가 깊이를 맞힌다면, 그 깊이 정보는 백본이 이미 **분리해 정렬해 놓은** 상태다. DINOv2는 이 "readily available" 정도가 가장 높고(Table 11 lin.1 최저 RMSE, Figure 7의 깨끗한 깊이 맵), 이는 논문이 말하는 "장면 기하 이해의 창발"의 구체적 증거다.
