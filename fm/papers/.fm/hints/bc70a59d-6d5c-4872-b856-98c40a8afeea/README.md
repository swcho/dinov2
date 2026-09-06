# 레이블 연관 공정성 평가(Label Association Fairness) 프로토콜

> **Q.** 레이블 연관 공정성 평가의 프로토콜은?
> **A.** ImageNet-22k의 619개 클래스 부분집합으로 다중 클래스 분류기를 학습하고, 이를 Human / Possibly Human / Non-Human / Crime 네 범주로 묶는다. Casual Conversations의 2955장에 대해 top-5 중 확률 0.1 이상인 레이블을 모두 취한다.

출처: DINOv2 논문(Oquab et al., 2023, arXiv:2304.07193) §8.2 "Gender, Skintones and Age" 및 Table 13.

---

## 1. 이 평가가 재는 것 — "유해 레이블 연관(harmful label association)"

DINOv2 논문 §8 "Fairness and Bias Analysis"는 가장 큰 모델 ViT-g/14를 대상으로 두 가지 공정성 평가를 수행한다.

| 절 | 평가 | 벤치마크 | 재는 것 |
|---|---|---|---|
| §8.1 | 지리적 공정성 | Dollar Street (Table 12) | 소득 구간·대륙별 인식 정확도 격차 |
| **§8.2** | **레이블 연관 공정성** | **Casual Conversations (Table 13)** | **사람 사진에 모델이 어떤 레이블을 붙이는가** |

이 카드는 두 번째, §8.2에 관한 것이다. 핵심 아이디어는 단순하다. **사람이 찍힌 사진**을 분류기에 넣었을 때, 모델이 사람과 무관한 사물·동물 클래스(Non-Human)나 범죄 관련 클래스(Crime)를 붙인다면 그것은 유해한 연관(harmful association)이다. 이 유해 레이블이 특정 성별·피부톤·연령 집단에 **더 자주** 붙는지 보면 모델의 편향을 드러낼 수 있다.

두 평가 모두 **Goyal et al. (2022b), "Fairness Indicators for Systematic Assessments of Visual Feature Extractors" (ACM FAccT 2022)** 의 프로토콜을 따른다. 이 논문은 비전 백본(feature extractor)의 공정성을 체계적으로 점검하기 위한 지표 묶음을 제안했고, 그중 하나가 바로 "harmful label association" 지표다. DINOv2는 이 기성 프로토콜을 거의 그대로 가져와 자기 모델을 점검한 것이다. (SEERv2를 소개한 Goyal et al. 2022a 역시 같은 저자 그룹의 논문으로 같은 지표로 SEER를 평가했기 때문에, Table 13의 비교 상대가 SEERv2인 것은 자연스럽다.)

---

## 2. 프로토콜 단계별 정리

### (1) ImageNet-22k에서 619개 클래스만 골라 분류기 학습

- ImageNet-22k(정확히는 21,841개 synset)에서 **사람과 관련 있거나, 사람 사진에 잘못 붙을 때 유해할 수 있는 클래스 619개**를 선별한다. 이 선별 목록은 Goyal et al. (2022b)가 정의한 것을 그대로 사용한다.
- 이 619개 클래스를 타깃으로 하는 **다중 클래스(multiclass) 분류기**를 학습한다. 나머지 2만여 개 클래스는 애초에 출력 공간에 없다.
- 왜 22k 전체가 아니라 619개만 쓰나? 전체를 쓰면 확률 질량이 수만 개 클래스로 흩어져 신호가 묻히고, 분석 대상인 "사람 관련 vs 유해" 구분에 집중할 수 없기 때문이다.

### (2) 619개 클래스를 4개 메타범주로 묶기

| 메타범주 | 성격 | 예시 클래스(ImageNet-22k synset) | 유해 여부 |
|---|---|---|---|
| **Human** | 사람 자체를 가리키는 클래스 | person, people, adult, face, man, woman 등 사람·신체 관련 synset | 무해 |
| **Possibly Human** | 사람 자체는 아니지만 사람과 자주 함께 등장하는 사물 | **Scarf, Glasses, Beard** (논문이 직접 든 예), 그 외 의류·장신구류 | 무해(단, 빈도 편차는 관찰 대상) |
| **Non-Human** | 사람 사진에 붙으면 모욕적인 비(非)인간 클래스 | 동물류(원류 등) 등 사람 아닌 생물·사물 | **유해** |
| **Crime** | 범죄·수감 관련 클래스 | 범죄자·수감자·죄수 계열 synset | **유해** |

- 논문 원문: *"Non-Human and Crime are considered harmful."* 즉 네 범주 중 **Non-Human과 Crime 두 개가 '유해 레이블'** 이고, 평가의 핵심은 이 두 범주가 사람 사진에 얼마나 붙는지다.
- Possibly-Human의 예시(Scarf, Glasses, Beard)는 DINOv2 논문 §8.2 결과 논의에서 직접 언급된 것이다. 나머지 범주의 구체적 클래스 목록은 Goyal et al. (2022b) 및 그 공개 코드(VISSL 기반)에 정의되어 있으며, 위 표의 예시는 그 성격을 설명하기 위한 대표 예다.

### (3) Casual Conversations 2955장에 추론

- **Casual Conversations** (Hazirbas et al., 2021, Meta AI)는 공정성 측정을 위해 만들어진 데이터셋이다. 약 3천 명의 참가자가 **유료로 동의하고** 촬영한 대화 영상으로 구성되며, 각 참가자에 대해 **연령·성별(자기보고)** 과 **피부톤(Fitzpatrick 척도)**, 조명 조건이 주석되어 있다.
  - DINOv2 논문은 "gender, skin tone, and age (all self-reported)"라 적었는데, 원 데이터셋 논문 기준으로 연령·성별은 자기보고, 피부톤은 훈련된 주석자가 Fitzpatrick 척도로 라벨링한 것이다. 카드 답에는 영향이 없지만 세부 출처 차이로 알아두면 좋다.
- 이 데이터셋을 쓰는 이유: (a) **사람만 찍힌 사진**이므로 어떤 비인간/범죄 레이블도 명백한 오류이자 유해 연관이고, (b) 인구통계 속성이 **참가자 동의 하에 신뢰성 있게** 제공되어 집단별로 결과를 쪼갤 수 있다.
- **2955장**은 참가자별 대표 프레임 1장씩을 뽑은 것으로, Goyal et al. (2022b)의 평가 세트를 그대로 따른 수치다.

### (4) 레이블 채택 규칙: "top-5 중 확률 ≥ 0.1"

- 각 이미지에 대해 619-way softmax 출력을 얻고, **확률 상위 5개 클래스** 중 **확률이 0.1 이상**인 것을 **모두** 채택한다.
- 이 규칙의 의미
  - **다중 레이블 허용**: top-1만 보면 "Human"이 1등일 때 2등으로 "Crime"이 0.3 확률로 붙어 있어도 놓친다. 유해 연관은 1등이 아니어도 문제이므로 여러 레이블을 동시에 취한다. 논문: *"Because of that, we can assign multiple classes to each image."*
  - **저확률 잡음 배제**: top-5를 무조건 다 세면 확률 0.01짜리 잡음까지 유해 레이블로 집계된다. 0.1 하한선은 "모델이 어느 정도 확신하는" 레이블만 남긴다.
  - **결과 해석**: 이렇게 하면 한 이미지가 Human과 Possibly-Human에 동시에 집계될 수 있어, Table 13의 열 합계는 100%가 되지 않는다(예: female darker에서 Human 97.3 + Possibly-Human 15.8 > 100).
- 집단별 수치는 "해당 집단 이미지 중 그 메타범주의 레이블이 하나 이상 채택된 비율(%)"이다.

### (5) DINOv2의 유일한 변경점: 백본 frozen

- 원 프로토콜(Goyal et al. 2022b)은 백본을 **파인튜닝**하면서 619-way 분류기를 학습한다.
- DINOv2는 *"we do not backpropagate gradients to the backbone and keep it frozen"* — 즉 **백본을 고정하고 그 위에 linear classifier만** 학습한다(Table 13 캡션: *"Instead of finetuning the backbone, we simply learn a linear classifier"*).
- 이유는 논문 전체의 평가 철학과 일치한다. DINOv2는 "파인튜닝 없이 바로 쓸 수 있는 범용 특징"을 주장하므로, 공정성 역시 **사전학습된 특징 그 자체**가 어떤 연관을 담고 있는지 보는 것이 목적에 맞다. 파인튜닝하면 특징이 바뀌어 사전학습의 편향과 파인튜닝의 효과가 섞인다.

---

## 3. Table 13 결과 재구성

집단별 각 메타범주 레이블 채택 비율(%). SEERv2는 RegNet-10B(논문 표기 RG-10B), DINOv2는 ViT-g/14.

| 모델 | 메타범주 | 여성·어두운 피부 | 여성·밝은 피부 | 남성·어두운 피부 | 남성·밝은 피부 | 18–30 | 30–45 | 45–70 | 70+ |
|---|---|---|---|---|---|---|---|---|---|
| SEERv2 RG-10B | Non-Human | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| | Crime | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| | Human | 94.9 | 95.8 | 86.6 | 79.0 | 90.5 | 88.3 | 91.9 | 82.3 |
| | Possibly-Human | 13.6 | 6.7 | 65.0 | 60.2 | 32.8 | 37.2 | 29.4 | 6.5 |
| DINOv2 ViT-g/14 | Non-Human | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| | Crime | 0.0 | 0.0 | **0.2** | 0.0 | 0.0 | **0.1** | 0.0 | 0.0 |
| | Human | 97.3 | 97.7 | 86.1 | 84.0 | 91.2 | 90.2 | 93.2 | 88.7 |
| | Possibly-Human | 15.8 | 17.2 | 52.2 | 48.1 | 35.3 | 37.3 | 23.0 | 9.7 |

### 읽는 법과 논문의 해석

1. **유해 레이블(Non-Human, Crime)은 사실상 0.** 두 모델 모두 Non-Human은 전 집단 0.0%. Crime은 DINOv2에서 "남성·어두운 피부" 0.2%, "30–45세" 0.1%에만 나타나는데, 논문은 이것이 **배경에 감옥 창살처럼 보이는 막대가 있던 두 장의 이미지** 때문이라 설명한다(동일한 두 이미지가 성별·피부톤 축과 연령 축에 각각 집계된 것). 2955장 중 2장이므로 0.2%·0.1%라는 수치와도 맞는다.
2. **Human은 모든 집단에서 높고 피부톤 간 차이가 작다.** DINOv2는 여성 97%대, 남성 84–86%, 연령대 88–93%. 피부톤(darker vs lighter)에 따른 편차가 크지 않다는 것이 논문이 강조하는 점이다. SEERv2 대비 대부분의 집단에서 Human 비율이 더 높다(예: 70+ 82.3 → 88.7).
3. **Possibly-Human이 자주 발화되며, 특히 남성에서 높다.** DINOv2는 남성에서 48–52%, 여성에서 16–17%. 논문은 이것이 **Beard(수염) 클래스**의 빈도 때문이라 설명한다 — 유해하지는 않지만 성별에 따라 다른 사물 레이블이 붙는 것 자체는 관찰할 가치가 있는 편차다. SEERv2는 이 격차가 더 컸다(남 60–65% vs 여 7–14%).
4. **결론**: *"No clear pattern indicates a bias against a particular group in this study."* 단, 논문은 곧바로 이 평가가 제한적이며 더 철저한 조사에서는 결함이 드러날 수 있다고 덧붙인다.

---

## 4. 한 줄 암기 요약

**Goyal 2022b 프로토콜** → ImageNet-22k **619 클래스**로 분류기(DINOv2는 **frozen 백본 + linear**) → **Human / Possibly-Human / Non-Human / Crime** 4 범주(뒤 둘이 유해) → **Casual Conversations 2955장** → **top-5 중 p ≥ 0.1 모두 채택**(다중 레이블) → 결과: 유해 레이블 ≈ 0, Possibly-Human은 Beard 때문에 남성에서 높음.

---

## 참고 문헌

- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, arXiv:2304.07193, §8.2, Table 13.
- Goyal, Romero Soriano, Hazirbas, Sagun, Usunier, *Fairness Indicators for Systematic Assessments of Visual Feature Extractors*, ACM FAccT 2022 (논문 내 인용 Goyal et al. 2022b).
- Hazirbas et al., *Towards Measuring Fairness in AI: the Casual Conversations Dataset*, IEEE T-BIOM 2021.
- Goyal et al., *Vision Models Are More Robust and Fair When Pretrained on Uncurated Images Without Supervision*, arXiv:2202.08360, 2022a (SEERv2).
