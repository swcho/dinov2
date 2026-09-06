# DINOv2가 남성 이미지에 "Possibly-Human"을 자주 예측하는 이유

> **Q.** DINOv2가 남성 이미지에 Possibly-Human을 자주 예측하는 이유는?
>
> **A.** Possibly-Human 범주가 Scarf, Glasses, Beard처럼 사람과 자주 연관되는 ImageNet-22k 객체들로 구성되어 있는데, Beard 클래스의 빈도가 높기 때문이다. 논문은 더 철저한 편향 평가가 결함을 드러낼 수 있음도 인정한다.

출처: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* (arXiv 2304.07193v2), **8.2 Gender, Skintones and Age** 절과 **Table 13**.

---

## 1. 이 실험이 무엇을 측정하는가 — 평가 프로토콜

DINOv2 논문 8장은 두 가지 공정성 평가를 수행한다. 8.1은 Dollar Street로 지리·소득 편향을, 8.2는 **유해 라벨 연관(harmful label association)** 을 본다. 이 카드는 8.2에 해당한다. 프로토콜은 Goyal et al. (2022b, *Fairness Indicators for Systematic Assessments of Visual Feature Extractors*, FAccT)을 따르되 한 가지를 바꿨다.

| 항목 | 내용 |
|---|---|
| 백본 | DINOv2 ViT-g/14 (가장 큰 모델), 비교 대상 SEERv2 RG-10B |
| 분류기 | ImageNet-22k 중 **619개 클래스** 부분집합에 대한 다중 클래스 분류기 |
| 메타 범주 | 619 클래스를 **Human / Possibly-Human / Non-Human / Crime** 4개로 묶음. Non-Human과 Crime을 "유해(harmful)"로 간주 |
| 평가 데이터 | Casual Conversations (Hazirbas et al., 2021)에서 **2,955장**. 성별·피부톤·연령은 자기보고(self-reported) |
| 라벨 부여 규칙 | top-5 중 확률 **0.1 이상**인 라벨을 모두 채택 → 한 이미지에 여러 메타 범주가 동시에 붙을 수 있음 |
| 원 프로토콜과의 차이 | Goyal et al.은 백본을 파인튜닝했지만, DINOv2는 **백본을 frozen 상태로 두고 선형 분류기만 학습** |

핵심은 마지막 두 줄이다. 라벨이 다중으로 붙기 때문에 각 열의 수치는 "그 그룹 이미지 중 해당 메타 범주 라벨이 하나라도 붙은 비율(%)"이며, 열의 합이 100이 되지 않는다. 또 frozen 선형 평가이므로 결과는 "DINOv2 표현 공간에서 선형으로 읽어낼 수 있는 연관"을 말한다.

---

## 2. Table 13 재구성 — 숫자로 보는 남녀 차이

### DINOv2 ViT-g/14 (단위: %, 해당 라벨이 붙은 이미지 비율)

| 메타 범주 | 여성·어두운 피부 | 여성·밝은 피부 | 남성·어두운 피부 | 남성·밝은 피부 | 18–30 | 30–45 | 45–70 | 70+ |
|---|---|---|---|---|---|---|---|---|
| Non-Human (유해) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Crime (유해) | 0.0 | 0.0 | **0.2** | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 |
| Human | 97.3 | 97.7 | 86.1 | 84.0 | 91.2 | 90.2 | 93.2 | 88.7 |
| **Possibly-Human** | **15.8** | **17.2** | **52.2** | **48.1** | 35.3 | 37.3 | 23.0 | 9.7 |

### 비교: SEERv2 RG-10B

| 메타 범주 | 여성·어두운 피부 | 여성·밝은 피부 | 남성·어두운 피부 | 남성·밝은 피부 | 18–30 | 30–45 | 45–70 | 70+ |
|---|---|---|---|---|---|---|---|---|
| Non-Human | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Crime | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Human | 94.9 | 95.8 | 86.6 | 79.0 | 90.5 | 88.3 | 91.9 | 82.3 |
| Possibly-Human | 13.6 | 6.7 | 65.0 | 60.2 | 32.8 | 37.2 | 29.4 | 6.5 |

### 표에서 읽어야 할 것

1. **성별 격차가 압도적이다.** DINOv2의 Possibly-Human 비율은 여성 그룹 약 16–17%, 남성 그룹 약 48–52%로 **약 3배** 차이가 난다. 카드가 묻는 현상이 정확히 이 숫자다.
2. **같은 성별 안에서는 피부톤 차이가 작다.** 여성 15.8 vs 17.2, 남성 52.2 vs 48.1. 즉 격차를 만드는 변수는 피부톤이 아니라 성별이다. 논문도 "피부톤에 따른 큰 편차 없이 모든 그룹을 Human으로 분류한다"고 쓴다.
3. **Human 비율이 남성에서 더 낮다** (여성 97%대 vs 남성 84–86%). Possibly-Human이 top-5의 확률 질량을 나눠 가져가면서 Human 라벨이 0.1 문턱을 넘지 못하는 경우가 생긴 것으로 보인다. 두 행은 같은 현상의 양면이다.
4. **유해 범주는 사실상 0.** Crime 0.2% (남성·어두운 피부)와 0.1% (30–45)는 논문이 언급한 "배경에 감옥 창살처럼 보이는 막대가 있던 두 장"에 해당한다. 그룹의 속성이 아니라 배경 소품이 원인이라는 것이 저자들의 해석이다.
5. **SEERv2와의 비교.** SEERv2는 남성에서 60–65%로 격차가 더 컸다(여성 6.7–13.6%와 비교하면 5–9배). DINOv2는 남성 비율을 낮추고 여성 비율을 조금 올려 격차를 줄였지만 여전히 뚜렷하다.
6. **연령 패턴.** 70+ 그룹은 9.7%로 크게 낮다. 논문은 이를 따로 설명하지 않는다. Casual Conversations에서 70+ 피험자 수가 적다는 점, 그리고 Beard가 젊은 남성 그룹에서 더 자주 나타날 가능성 정도를 추정할 수 있지만 근거 수치는 없다.

---

## 3. 왜 남성에게 Possibly-Human이 붙는가 — 논문의 설명

논문 원문:

> We see that our model triggers the Possibly-Human classes often. This class is constructed from objects in ImageNet-22k that are often related to Humans, such as Scarf, Glasses, or Beard. Our model often predicts the Possibly-Human class for men because of the prevalence of the Beard class.

두 층으로 나눠 이해하면 된다.

### (a) 메타 범주의 구성 문제 — "사람과 함께 등장하는 물건"의 모음

Possibly-Human은 "사람일 수도 있는 것"이 아니라, **ImageNet-22k에서 사람과 자주 공존하는 객체 클래스**(스카프, 안경, 수염 등)를 모아 놓은 범주다. ImageNet-22k는 WordNet 명사 계층을 따르므로 `beard`, `scarf`, `spectacles` 같은 것이 독립 클래스로 존재한다. 이 클래스들은 사진의 주체가 사람이든 아니든 "그 물건이 화면에 있으면" 켜지는 것이 정상 동작이다. 분류기가 얼굴 클로즈업 이미지에서 안경·수염을 감지하는 것은 오작동이 아니라 **잘 작동하는 것**이다.

### (b) 실제 외형 상관 — 남성 이미지에 수염이 많다

Possibly-Human 안에서도 **Beard** 클래스가 특히 자주 발화하고, 수염은 남성 이미지에서 훨씬 자주 관찰되는 외형 속성이다. 따라서 "남성 → Possibly-Human 높음"은 모델이 남성을 덜 인간적으로 본다는 뜻이 아니라, "남성 → 수염 있음 → Beard 클래스 발화 → 이 클래스가 Possibly-Human 메타 범주에 속함"이라는 **설명 가능한 인과 사슬**의 결과다. 여성 그룹의 15–17%는 Scarf·Glasses 같은 나머지 클래스가 만든 기저선으로 볼 수 있다.

이 때문에 논문은 이 결과를 다음과 같이 정리한다.

> No clear pattern indicates a bias against a particular group in this study.

즉 저자들의 판단은 "이 현상은 유해 편향이 아니라 평가 설계(메타 범주 구성)와 데이터의 실제 외형 분포에서 나오는 부산물"이라는 것이다. 유해로 정의된 Non-Human·Crime이 사실상 0이라는 점이 이 판단을 뒷받침한다.

---

## 4. 그럼에도 왜 "더 철저한 평가가 결함을 드러낼 수 있다"고 유보하는가

> While this is encouraging, we also acknowledge that a more thorough evaluation of biases may reveal flaws in our model.

이 문장은 단순한 겸양이 아니다. 이 평가 프로토콜 자체가 편향을 놓치기 쉬운 구조이기 때문이다.

### 프로토콜의 해상도가 거칠다
- **619 클래스 → 4 메타 범주**라는 구획은 매우 거칠다. "Possibly-Human"으로 묶인 순간 그 안에서 *어떤* 클래스가 발화했는지(Beard인지, 혹은 덜 무해한 클래스인지)가 표에서 사라진다. 논문이 "Beard 때문"이라고 말하는 근거는 표가 아니라 별도 관찰이며, 클래스별 분해 수치는 제시되지 않는다.
- **top-5, 확률 ≥ 0.1**이라는 임계값은 임의적이다. 임계값을 바꾸면 Possibly-Human 비율과 Human 비율이 함께 움직인다.
- **유해 여부의 정의가 이진적이다.** Non-Human·Crime만 유해로 본다. 특정 그룹에 Human 대신 부속 물건 라벨이 훨씬 자주 붙는 것 자체가 문제인지 아닌지는 이 프로토콜이 판단하지 않는다.

### 사용된 속성이 제한적이다
- 성별·피부톤·연령은 **자기보고** 속성이며, 성별은 실질적으로 이진 비교(female/male)로만 제시된다. 피부톤도 darker/lighter 두 구간으로만 나뉜다. 교차 그룹(성별×피부톤×연령)은 보이지 않는다.
- Casual Conversations는 동의를 받은 피험자가 카메라를 향해 말하는 실내 영상에서 뽑은 프레임이다. 조명·구도·배경이 균질하다. 실제 배포 환경(거리 사진, 저조도, 군중)에서 나타날 편향을 대표하지 못한다.
- **2,955장**은 8개 그룹으로 나누면 그룹당 수백 장 수준이다. 0.1–0.2% 차이는 이미지 1–2장에 해당하므로 통계적 결론에는 약하다.

### frozen linear 평가라는 선택
- 원 프로토콜(Goyal et al., 2022b)은 백본을 파인튜닝하지만 DINOv2는 frozen + 선형 분류기만 학습했다. 이는 "표현 자체가 무엇을 선형으로 인코딩하는가"를 보기에는 적절하지만, 파인튜닝 시 드러날 수 있는 편향(혹은 반대로 파인튜닝이 완화할 편향)은 측정하지 못한다. SEERv2와의 비교도 프로토콜이 달라져 엄밀히 동일 조건이 아니다.
- 선형 분류기는 ImageNet-22k 라벨을 학습하므로, 결과에는 DINOv2 표현의 편향과 **ImageNet-22k 라벨 체계의 편향**이 섞여 있다. Beard가 하나의 클래스로 존재한다는 것 자체가 WordNet 분류 체계의 산물이다.

### 다른 절에서 이미 편향이 확인되었다
바로 앞 8.1절(Table 12)에서 DINOv2는 유럽 대비 **아프리카에서 −25.7%**, 고소득 대비 **저소득 가구에서 −31.7%** 성능 하락을 보였다. 저자들은 "우리 모델은 여전히 서구 부유 가구 쪽으로 편향되어 있다"고 명시한다. 같은 모델이 8.1에서는 뚜렷한 편향을 보였으니, 8.2에서 편향이 보이지 않는 것을 "편향이 없다"로 읽기보다 "이 특정 프로토콜이 잡아내지 못했다"로 읽는 것이 일관된 태도다.

### 학습 데이터의 출처
LVD-142M은 ImageNet 등 큐레이션된 시드 데이터셋과 유사한 이미지를 웹 풀에서 검색해 모은 데이터다. 시드의 인구 구성 편향이 큐레이션을 통해 그대로 이어질 수 있으나, 논문은 LVD-142M의 인구 통계 분석을 제시하지 않는다.

---

## 5. 한 줄 요약과 기억 포인트

- **현상**: Table 13에서 Possibly-Human 비율이 여성 15.8/17.2% vs 남성 52.2/48.1% (약 3배). 피부톤 간 차이는 작다.
- **원인**: Possibly-Human = 사람과 공존하는 ImageNet-22k 객체 클래스(Scarf, Glasses, Beard) 모음. 남성 이미지에 수염이 많아 **Beard** 클래스가 자주 발화 → 유해 편향이 아니라 설명 가능한 부산물.
- **뒷받침**: 유해 범주(Non-Human, Crime)는 사실상 0%. 유일한 Crime 발화는 배경의 창살 모양 막대 2장.
- **유보**: 4 메타 범주라는 거친 구획, 임의 임계값, 자기보고 이진 속성, 균질한 데이터셋 2,955장, frozen linear 평가, ImageNet-22k 라벨 체계의 개입, 그리고 8.1에서 이미 확인된 지리·소득 편향 때문에 "더 철저한 평가가 결함을 드러낼 수 있다".

## 참고
- Oquab et al., 2023. DINOv2. arXiv:2304.07193. Sec. 8.2, Table 13.
- Goyal, Romero Soriano, Hazirbas, Sagun, Usunier, 2022b. *Fairness Indicators for Systematic Assessments of Visual Feature Extractors.* FAccT. (619 클래스·4 메타 범주 프로토콜의 원출처)
- Hazirbas et al., 2021. *Towards Measuring Fairness in AI: the Casual Conversations Dataset.* IEEE TBIOM.
- Goyal et al., 2022a. *Vision models are more robust and fair when pretrained on uncurated images without supervision.* (SEERv2)
