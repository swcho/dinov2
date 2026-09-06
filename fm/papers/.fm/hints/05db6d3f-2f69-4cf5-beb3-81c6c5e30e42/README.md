# MIM(masked image modeling) loss ablation — DINOv2 Table 3b

> **Q.** MIM(masked image modeling) loss ablation(Table 3b)의 결과는?
> **A.** ADE-20k 분할이 44.2 → 47.1로 약 3% 향상되어, dense prediction 태스크에 결정적임을 보인다. ImageNet-1k도 85.3 → 85.8로 소폭 오른다.

출처: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* (arXiv 2304.07193v2), Section 6.4 "Loss Components", Table 3.

---

## 1. Table 3b 원표

최고 성능 모델(ViT-g/14, LVD-142M)에서 iBOT의 MIM(패치 수준) 손실 항만 **빼 보는** 실험이다. 각 모델은 최종 런보다 짧은, 동일한 반복 횟수로 학습됐다.

| MIM 손실 | INet-1k (linear, acc %) | Im-A (acc %) | ADE-20k (linear seg., mIoU) | Oxford-M (retrieval, mAP) |
|:---:|:---:|:---:|:---:|:---:|
| ✕ 없음 | 85.3 | 72.0 | **44.2** | 64.3 |
| ✓ 있음 | 85.8 | 72.8 | **47.1** | 63.9 |
| 차이 | +0.5 | +0.8 | **+2.9** | −0.4 |

핵심 숫자는 **ADE-20k 44.2 → 47.1 (+2.9 mIoU, "almost 3%")**. 논문 본문 문장은 다음과 같다.

> "In Table 3b, we show the impact of using the masked image modeling term from iBOT. **This term is critical for dense prediction tasks**, leading to almost 3% performance improvement."

암기 포인트:
- **분할(dense)** = 큰 폭 상승(+2.9)
- **분류(image-level)** = 소폭 상승(+0.5 / +0.8)
- **검색(retrieval)** = 거의 변화 없음(−0.4, 노이즈 수준)

---

## 2. MIM / iBOT 손실이 패치 수준에서 무엇을 학습시키는가

DINOv2의 학습 목표는 두 개의 self-distillation 손실을 합친 것이다(Section 4).

| | Image-level objective (DINO) | Patch-level objective (iBOT, MIM) |
|---|---|---|
| 손실이 걸리는 토큰 | `[CLS]` 토큰 1개 | **마스킹된 패치 토큰들** (여러 개) |
| 학생 입력 | 서로 다른 crop | 일부 패치를 **마스킹**한 이미지 |
| 교사 입력 | 서로 다른 crop | 마스킹 **없는** 원본 이미지 |
| 손실식 | $\mathcal{L}_{DINO} = -\sum p_t \log p_s$ | $\mathcal{L}_{iBOT} = -\sum_i p_{ti}\log p_{si}$, $i$ = 마스킹된 패치 인덱스 |
| 학습되는 것 | 이미지 전체의 "무엇인가"(전역 의미) | **각 위치의 패치가 무엇인가**(국소 의미) |

동작 순서를 풀어 쓰면:

1. 학생 네트워크에 들어가는 이미지의 일부 패치를 무작위로 마스킹한다(교사에는 마스킹하지 않음).
2. 학생의 마스크 토큰 출력을 **iBOT head**(MLP)에 통과시켜 프로토타입 점수 → softmax → $p_{si}$.
3. 교사의 **같은 위치**(학생이 가린 자리)의 보이는 패치 토큰을 교사 iBOT head에 통과시켜 softmax + centering(Sinkhorn-Knopp) → $p_{ti}$.
4. 두 분포의 cross-entropy를 마스킹된 패치마다 계산해 합산한다.

즉 학생은 "가려진 위치에 교사가 봤을 때 어떤 의미의 패치가 있을지"를 **주변 문맥으로부터 예측**해야 한다. MAE처럼 픽셀을 복원하는 것이 아니라, **온라인 토크나이저(교사)가 부여한 의미적 시각 토큰**을 맞추는 것이다("Image BERT pre-training with Online Tokenizer" = iBOT). 결과적으로 백본은 다음을 강제로 학습한다.

- **패치별로 의미가 정의되는 표현**: 각 패치 토큰이 독립적으로 "이 위치는 창문/침대/하늘" 같은 의미 정보를 가져야 예측 대상이 될 수 있다.
- **공간적 문맥 추론**: 가려진 곳을 채우려면 인접 패치 간 관계(경계, 부품, 연속성)를 이해해야 한다.
- **위치 정합(spatial correspondence)**: 학생 위치 $i$ ↔ 교사 위치 $i$가 일대일로 묶여 있으므로, 패치 토큰이 "어디"의 정보를 잃지 않고 유지된다.

DINOv2는 이 head를 DINO head와 **분리(untying)** 해서 쓴다. 원 iBOT 논문은 두 head를 공유하는 것이 낫다고 했지만, 대규모에서는 반대였기 때문이다(Table 1 마지막 행).

---

## 3. 왜 dense prediction(분할)에는 결정적이고, 분류에는 영향이 작은가

### 3-1. 평가 방식이 "무엇을 읽어내는지"를 결정한다

Table 3의 세 평가는 서로 다른 토큰을 소비한다.

| 평가 | 사용하는 특징 | 필요한 성질 |
|---|---|---|
| ImageNet-1k / Im-A linear probe | `[CLS]`(전역) 토큰 위의 선형 분류기 | 이미지 전체를 잘 요약한 벡터 하나 |
| ADE-20k linear segmentation | **frozen 패치 토큰마다** 선형 분류기 → 32×32 저해상도 logit map → 업샘플 | **모든 패치 토큰이 각자 정확한 클래스 정보**를 가질 것 |
| Oxford-M retrieval | `[CLS]` 벡터 간 최근접 이웃 | 인스턴스별로 잘 퍼진 전역 임베딩 |

ADE-20k 선형 분할은 백본을 건드리지 않고 **패치 토큰 하나하나에 선형 레이어만** 얹는다. 따라서 패치 토큰 자체에 클래스가 선형 분리 가능하게 담겨 있어야 하는데, 이를 직접 훈련하는 항이 바로 MIM/iBOT 손실이다. 이 항을 빼면 패치 토큰은 `[CLS]`를 만들기 위한 중간 계산 결과일 뿐 자기 위치의 의미를 갖도록 최적화되지 않으므로, mIoU가 2.9포인트 떨어진다.

### 3-2. 분류에는 왜 소폭인가

- DINO 손실만으로도 `[CLS]` 토큰의 전역 표현은 충분히 좋아진다. 분류는 이미 image-level 목표가 직접 최적화하는 대상이다.
- MIM은 간접적으로만 돕는다: 패치 표현이 좋아지면 attention을 통해 `[CLS]`도 약간 더 풍부한 정보를 모으고(+0.5), 특히 국소 텍스처·부품 단서가 중요한 자연 적대 샘플(Im-A, +0.8)에서 조금 더 이득이 보인다.
- 검색(Oxford-M)은 오히려 −0.4로 아무 도움이 없다. 검색은 전역 임베딩의 분포(퍼짐)가 관건이고, 그것은 다음 절의 KoLeo가 담당한다.

### 3-3. 그림으로 보는 "패치 토큰 위 선형 분류기"

![Figure 7: frozen 특징 + 선형 분류기로 얻은 분할·깊이](fig-1.jpeg)

Figure 7(논문 부록)은 Table 3b의 ADE-20k 평가와 **같은 프로토콜**(frozen 백본 + 선형 레이어)로 얻은 결과다. 첫 행(ADE20K)에서 DINOv2-g 열의 분할 맵을 보면 하늘/건물/풀, 침대/베개/창문 등 영역 경계가 OpenCLIP-G 열보다 훨씬 깨끗하고 노이즈 조각이 적다. NYUd·SUN-RGBd 깊이 맵도 OpenCLIP-G는 얼룩덜룩한 반면 DINOv2-g는 소파·탁자·의자 형태가 매끈하게 드러난다. 이는 패치 토큰 각각이 자기 위치의 의미·기하 정보를 선형적으로 읽어낼 수 있게 담고 있다는 뜻이며, 그 성질을 학습 단계에서 직접 강제하는 항이 MIM 손실이다. OpenCLIP은 이미지-텍스트 대비학습(image-level 목표)만 쓰므로 패치 수준 표현이 상대적으로 약하다는 점과 대비된다.

---

## 4. Table 3a(KoLeo)와의 역할 분담

Section 6.4는 두 손실 항을 같은 방식으로 나란히 ablate한다. 두 표를 붙여서 보면 역할이 깔끔하게 나뉜다.

| 항 | 대상 | 목적 | 큰 효과가 나는 지표 | 영향이 없는 지표 |
|---|---|---|---|---|
| **KoLeo** (Table 3a) | `[CLS]` 특징(전역), 배치 내 | 최근접 이웃 거리 $d_{n,i}$의 로그를 최대화 → 특징을 **균일하게 펼침** | **Oxford-M 검색 55.6 → 63.9 (+8.3)** | ADE-20k 47.2 → 47.1 (변화 없음) |
| **MIM / iBOT** (Table 3b) | 마스킹된 **패치** 토큰(국소) | 가려진 패치의 의미 예측 → **패치별 의미 표현** 학습 | **ADE-20k 분할 44.2 → 47.1 (+2.9)** | Oxford-M 64.3 → 63.9 (변화 없음) |

Table 3a 원표:

| KoLeo | INet-1k | Im-A | ADE-20k | Oxford-M |
|:---:|:---:|:---:|:---:|:---:|
| ✕ | 85.3 | 70.6 | 47.2 | 55.6 |
| ✓ | 85.8 | 72.8 | 47.1 | 63.9 |

정리하면:

- **KoLeo는 전역 임베딩 공간의 "배치"를 다룬다.** 배치 안에서 서로 가장 가까운 두 이미지 특징을 밀어내어 특징이 뭉치지 않게 하므로, 인스턴스 단위로 구분해야 하는 최근접 이웃 검색(Oxford-M)에서 +8.3 mAP라는 큰 이득이 나온다. 반면 패치 토큰은 건드리지 않으니 분할은 그대로다.
- **MIM은 패치 토큰의 "내용"을 다룬다.** 각 위치가 무엇인지를 맞히게 하므로 분할 같은 dense task에 +2.9 mIoU가 나오고, 전역 임베딩의 분포에는 관여하지 않으니 검색은 그대로다.
- 두 항 모두 ImageNet-1k 분류는 85.3 → 85.8로 똑같이 +0.5만 오른다. 즉 **분류는 DINO 손실이 이미 담당**하고 있고, 두 보조 항은 각각 "검색"과 "분할"이라는 서로 다른 축을 보완하는 **직교적 역할**을 한다. 논문 캡션의 요약 문장 그대로다:

> "The KoLeo loss term improves nearest-neighbor search tasks (e.g. retrieval), and the MIM loss improves patch-level tasks (e.g. segmentation)."

이 역할 분담은 DINOv2가 "이미지 수준과 패치 수준 특징을 모두 잘 뽑는 범용 백본"을 목표로 한 이유이자, 하나의 백본으로 분류·검색·분할·깊이 추정을 frozen 상태로 처리할 수 있게 한 설계 근거다.

---

## 5. 한 줄 요약

**MIM(iBOT) 손실을 빼면 ADE-20k 선형 분할이 47.1 → 44.2로 약 3% 떨어지고 분류는 0.5%만 움직인다** — 패치마다 의미를 예측하게 하는 이 항이 dense prediction의 핵심이며, 검색을 책임지는 KoLeo(Table 3a, +8% mAP)와 역할이 나뉜다.
