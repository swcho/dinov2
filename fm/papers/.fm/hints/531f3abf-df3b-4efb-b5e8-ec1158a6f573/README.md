# DINOv2의 Robustness(domain generalization) 평가 — iBOT 대비 개선폭

> **Q.** Robustness(domain generalization) 평가에서 DINOv2의 iBOT 대비 개선폭은?
> **A.** ImageNet-A에서 +29.6%, ImageNet-R에서 +22.1%, Sketch에서 +23.0% 개선했다. 다만 R과 Sketch에서는 최고 weakly-supervised 모델(OpenCLIP ViT-G/14)에 뒤진다.

출처: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* (arXiv 2304.07193v2), §7.1 "Robustness analysis" 및 Table 6.

---

## 1. 실험 설정 — 무엇을 어떻게 측정했나

- **평가 대상**: ImageNet-1k 위에서 학습한 **linear classifier**(frozen backbone + 선형 헤드, 해상도 224).
- **방법**: Table 4의 linear probing에서 가장 좋았던 선형 헤드를 **그대로 가져와** domain generalization 벤치마크에 *추론만* 수행. 벤치마크별 재학습·파인튜닝 없음.
- **주의점(논문 명시)**: 문헌의 대부분 robustness 수치는 ImageNet-1k에서 **end-to-end 파인튜닝**한 모델의 결과다. DINOv2는 frozen feature + linear head라는 훨씬 불리한 조건에서 비교하고 있다.
- **비교군**: SSL 계열(MAE, DINO, iBOT)과 weakly-supervised(텍스트 지도) 계열의 대표인 OpenCLIP ViT-G/14(LAION-2B).

## 2. Table 6 재구성 — Domain Generalization with a linear probe (res. 224)

Im-C만 낮을수록 좋고(↓, mCE), 나머지는 높을수록 좋다(Top-1 %).

| Method | Arch | Data | Im-A ↑ | Im-R ↑ | Im-C ↓ | Sketch ↑ |
|---|---|---|---|---|---|---|
| OpenCLIP | ViT-G/14 | LAION-2B | 63.8 | **87.8** | 45.3 | **66.4** |
| MAE | ViT-H/14 | INet-1k | 10.2 | 34.4 | 61.4 | 21.9 |
| DINO | ViT-B/8 | INet-1k | 23.9 | 37.0 | 56.6 | 25.5 |
| iBOT | ViT-L/16 | INet-22k | 41.5 | 51.0 | 43.9 | 38.5 |
| DINOv2 | ViT-S/14 | LVD-142M | 33.5 | 53.7 | 54.4 | 41.2 |
| DINOv2 | ViT-B/14 | LVD-142M | 55.1 | 63.3 | 42.7 | 50.6 |
| DINOv2 | ViT-L/14 | LVD-142M | 71.3 | 74.4 | 31.5 | 59.3 |
| DINOv2 | ViT-g/14 | LVD-142M | **75.9** | 78.8 | **28.2** | 62.5 |

### iBOT(ViT-L/16, INet-22k) 대비 DINOv2 차이 (표 수치로 직접 계산)

| DINOv2 모델 | Im-A | Im-R | Im-C (낮을수록 좋음) | Sketch |
|---|---|---|---|---|
| ViT-L/14 | +29.8 | +23.4 | −12.4 | +20.8 |
| ViT-g/14 | +34.4 | +27.8 | −15.7 | +24.0 |

> **수치 메모**: 본문이 인용한 "+29.6 / +22.1 / +23.0"은 Table 6의 어느 한 행을 빼서도 정확히 재현되지 않는다(ViT-L 기준 +29.8/+23.4/+20.8, ViT-g 기준 +34.4/+27.8/+24.0). 초고(v1) 시점의 수치가 본문에 남은 것으로 보이며, 카드 답안은 **논문 본문의 문장**을 그대로 따른 것이다. 어느 쪽으로 보아도 "A·R·Sketch 모두 20~30%p 이상, 압도적 개선"이라는 결론은 동일하다. 본문에서 언급하지 않았지만 Im-C도 43.9 → 28.2로 크게 개선되었다.

### OpenCLIP(최고 weakly-supervised) 대비

| 벤치마크 | OpenCLIP ViT-G/14 | DINOv2 ViT-g/14 | 판정 |
|---|---|---|---|
| Im-A | 63.8 | **75.9** | DINOv2 우위 (+12.1) |
| Im-R | **87.8** | 78.8 | OpenCLIP 우위 (−9.0) |
| Im-C ↓ | 45.3 | **28.2** | DINOv2 우위 |
| Sketch | **66.4** | 62.5 | OpenCLIP 우위 (−3.9) |

→ 카드 답안의 "R과 Sketch에서는 최고 weakly-supervised 모델에 뒤진다"가 바로 이 두 칸이다. (A와 C에서는 오히려 앞선다.)

## 3. 각 벤치마크가 측정하는 것

모두 ImageNet-1k 클래스 공간을 공유하므로, ImageNet-1k용 선형 헤드를 그대로 얹어 평가할 수 있다.

| 벤치마크 | 논문 인용 | 무엇을 측정하나 |
|---|---|---|
| **ImageNet-A** (Adversarial) | Hendrycks et al., 2021b "Natural adversarial examples" | 합성 노이즈가 아닌 **자연 사진**인데 기존 ResNet 계열 분류기가 *일관되게 틀리는* 어려운 예제 7,500장(200클래스). 배경·자세·가림·질감 편향 등 "분포 내이지만 어려운" 사례에 대한 강건성. |
| **ImageNet-R** (Rendition) | Hendrycks et al., 2021a "The many faces of robustness" | 같은 200클래스를 **그림·만화·낙서·자수·그래피티·조각·장난감·비디오게임** 등으로 표현한 3만 장. "사진 → 다른 표현 매체"로의 **스타일/도메인 전이** 강건성. |
| **ImageNet-Sketch** | Wang et al., 2019 | 1000클래스를 Google 이미지 검색 "sketch of ___"로 모은 **흑백 스케치** ~5만 장. 질감·색을 제거하고 **형태(shape) 단서**만으로 인식하는 능력. |
| **ImageNet-C** (Corruption) | Hendrycks & Dietterich, 2019 | ImageNet val에 노이즈·블러·날씨·디지털 등 **15종 손상 × 5단계 강도**를 합성 적용. 저수준 화질 저하에 대한 강건성. 지표는 mCE라 **낮을수록 좋다**. |

## 4. 왜 SSL(iBOT) 대비 이렇게 큰 개선이 나왔나

1. **사전학습 데이터의 규모와 다양성 (LVD-142M vs INet-22k)**
   - iBOT은 ImageNet-22k(14M)에서, DINOv2는 자동 큐레이션한 LVD-142M(142M)에서 학습했다. §6.2 Table 2의 ablation에서 같은 ViT-g·같은 반복 수로 데이터만 바꿔도 **Im-A가 73.5(INet-22k) → 73.9(LVD-142M)**로 유지·개선되고, 무큐레이션 랜덤 142M은 59.4로 크게 떨어진다. 즉 "많이"만이 아니라 "**다양하되 큐레이션된**" 데이터가 어려운 예제 강건성에 결정적이다.
   - LVD-142M은 웹의 1.2B 이미지에서 큐레이션 데이터셋과 *시각적으로 가까운* 이미지를 검색해 모은 것이라, ImageNet 사진 분포를 벗어난 조명·구도·촬영 조건·부분적 표현 매체까지 자연스럽게 포괄한다.

2. **모델 스케일과 데이터 스케일의 상호작용 (Figure 4)**

   ![Figure 4: 모델 크기(L/H/g)에 따른 INet-22k(주황) vs LVD-142M(파랑) 성능](fig-1.jpeg)

   위 그림의 세 번째 패널 **ImageNet-Sketch**를 보면, INet-22k로 학습한 모델(주황)은 ViT-L→g로 키워도 약 48→54 정도로 완만하게 오르는 반면, LVD-142M(파랑)은 약 50→61로 **모델이 커질수록 격차가 벌어진다**. ImageNet-1k 패널에서는 두 선이 ViT-g에서 거의 만나므로, 데이터 다양성의 이득은 *in-domain 정확도가 아니라 out-of-domain(Sketch, AmsterTime, Oxford-H)* 에서 두드러진다. iBOT(ViT-L/16, INet-22k)에서 DINOv2(ViT-g/14, LVD-142M)로의 도약이 바로 이 "모델 × 데이터" 스케일링 효과다.

3. **학습 레시피 개선 (§5, Table 1)**
   - iBOT를 출발점으로 KoLeo 정규화, Sinkhorn-Knopp centering, untied head, 큰 배치(3k), 고해상도 적응 등을 추가했다. Table 3(a)에서 **KoLeo만 켜도 Im-A 70.6 → 72.8**로 오른다. 특징 공간이 균일하게 퍼지면 어려운 예제가 소수 클래스에 뭉개지지 않는 효과로 해석할 수 있다.
   - 결과적으로 Table 4의 ImageNet-1k linear에서 iBOT 대비 +4.2%인데, robustness 벤치마크에서는 그 5~8배인 +20~30%p가 난다. 이는 "in-domain 정확도가 조금 오른 것"이 아니라 **feature의 일반화 성질 자체가 달라졌다**는 논문의 주장(Table 4 설명의 "stronger generalization")과 일치한다.

4. **Frozen feature라는 조건에서의 의미**
   - end-to-end 파인튜닝은 ImageNet-1k 사진 분포에 backbone을 재적합시켜 OOD에서 오히려 손해를 볼 수 있다. DINOv2는 backbone을 건드리지 않고 선형 헤드만 얹었기에 사전학습이 만든 일반적 표현이 그대로 살아 있고, 그것이 OOD 강건성으로 드러난다.

## 5. 왜 R·Sketch에서는 OpenCLIP(텍스트 지도)에 뒤지나

- **텍스트 지도가 '표현 매체' 개념을 직접 주입한다.** LAION-2B 캡션에는 "a *cartoon* of a dog", "*sketch* of a cat", "*sculpture* of an elephant" 같은 문구가 풍부하다. CLIP류는 이런 캡션과 이미지를 정렬하며 **"개"라는 개념이 사진·만화·조각·스케치 어느 매체로 그려져도 같은 것**임을 명시적으로 배운다. ImageNet-R과 Sketch가 측정하는 것이 정확히 이 매체 불변성이다.
- **SSL은 시각적 유사성만으로 묶는다.** DINOv2는 이미지 간 시각적 유사도(augmentation 불변성, 패치 복원)만으로 표현을 배운다. 사진 속 개와 그래피티 개는 픽셀 통계가 크게 달라 텍스트 없이 같은 클래스로 묶을 강한 신호가 없다. 게다가 LVD-142M은 ImageNet 등 **사진 중심** 큐레이션 셋과 가까운 이미지를 검색해 모은 데이터라 렌디션·스케치의 비율이 LAION-2B보다 낮을 가능성이 크다.
- **반면 Im-A와 Im-C는 사진 도메인 안의 어려움이다.** Im-A(어려운 자연 사진)와 Im-C(손상된 사진)는 표현 매체가 바뀌지 않는다. 여기서는 텍스트가 주는 개념적 도움보다 **사진에 대한 조밀하고 견고한 시각 표현**이 더 중요하고, DINOv2가 OpenCLIP을 각각 +12.1, 17.1 mCE 차로 앞선다. 논문 전체 주장 —"캡션 기반 학습은 미세한 시각 패턴을 놓친다(§7.3 depth 실험 언급)" — 과 일관된 그림이다.
- 정리: **DINOv2가 뒤지는 축은 '의미적 스타일 불변성', 앞서는 축은 '시각적 견고함'**이다. 이 구분이 Table 6을 한 줄로 요약하는 열쇠다.

## 6. 기억용 요약

- 평가: ImageNet-1k linear head를 frozen backbone에 얹어 **추론만** (문헌은 대개 파인튜닝).
- iBOT 대비: **A +29.6 / R +22.1 / Sketch +23.0** (본문 표기; 표 기준 ViT-g는 +34.4/+27.8/+24.0), Im-C도 43.9→28.2.
- OpenCLIP 대비: **A·C는 앞서고, R·Sketch는 뒤진다**.
- 이유: 개선은 *LVD-142M의 다양성 × 모델 스케일 × 레시피*; 열세는 *텍스트 지도가 주는 렌디션/스타일 개념의 부재*.
