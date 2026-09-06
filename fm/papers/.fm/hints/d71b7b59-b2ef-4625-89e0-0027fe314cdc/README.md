# 기존 자기지도학습(SSL)이 대규모로 확장될 때 겪은 문제

> **Q.** 기존 자기지도학습 연구가 대규모로 확장될 때 겪은 문제는?
>
> **A.** 대부분의 진전은 ImageNet-1k 같은 작은 큐레이션 데이터셋 위에서 이뤄졌고, 그 너머로 확장한 시도들은 uncurated 데이터를 써서 특징 품질이 크게 떨어졌다. 데이터 품질과 다양성을 통제하지 못한 것이 원인이다.

출처: DINOv2 논문(Oquab et al., 2023, arXiv 2304.07193) §1 Introduction, §2 Related Work, §6.2 Pretraining Data Source.

---

## 1. 배경: SSL은 왜 "ImageNet-1k 안"에 머물렀나

DINOv2 이전의 자기지도학습(SimCLR, MoCo, BYOL, SwAV, DINO, iBOT 등)은 레이블 없이 이미지만으로 특징을 배우는 데 성공했고, 이미지 수준(분류)과 픽셀 수준(분할)의 정보를 모두 담을 수 있다는 점에서 텍스트 감독(CLIP류)보다 원리적으로 유리하다고 여겨졌다.

그런데 논문이 지적하는 핵심 아이러니는 이것이다:

> "most of the advances in self-supervised learning were made in the context of pretraining on a **small curated dataset, ImageNet-1k**."

- ImageNet-1k(약 128만 장, 1000 클래스)는 사람이 클래스별로 이미지를 고르고 정리한 **큐레이션된** 데이터셋이다. 레이블을 직접 쓰지 않아도, "어떤 이미지가 데이터셋에 들어가는가" 자체가 이미 사람의 선별을 거친 결과다.
- 즉 기존 SSL의 성능은 사실상 **"작지만 깨끗하고 균형 잡힌 데이터"**라는 조건 위에서 얻어진 것이었다. 이 조건이 무너지면 어떻게 되는지는 검증되지 않았다.

## 2. 확장 시도와 그 실패 양상

ImageNet-1k 밖으로 데이터를 키운 연구들도 있었다. 논문이 인용하는 것은 Caron et al. (2019, DeeperCluster), Goyal et al. (2019, 2021 SEER, 2022a) 계열로, 수억~수십억 장의 **크롤링/인스타그램 등 uncurated 이미지**로 SSL을 학습했다.

> "Some efforts on scaling these approaches beyond ImageNet-1k have been attempted, but they focused on **uncurated datasets**, which typically lead to a **significant drop in the quality of the features**."

§2 Related Work의 "Scaling self-supervised pretraining" 단락은 좀 더 구체적으로 실패의 형태를 말한다:

- 이 연구들은 "판별적(discriminative) SSL이 데이터 양에 따라 스케일링된다"는 증거를 보여주긴 했다.
- 하지만 **사전학습 데이터 품질이 나빠서**, 대부분의 결과는 **특징을 그대로 쓰지 못하고 fine-tuning을 해야** 얻어졌다. 즉 "얼리는(frozen) 범용 특징"이라는 파운데이션 모델의 목표에는 도달하지 못했다.
- 이 계열은 "SSL이 *어떤* 데이터에서도 작동하는가"라는 질문에 집중했고, DINOv2는 반대로 "*최고의* 사전학습 인코더를 만드는 것"에 집중한다는 점에서 목표가 다르다.

## 3. 원인: 데이터 품질과 다양성의 통제 부재

> "This is explained by the **lack of control over the data quality and diversity**, which are essential to produce good features."

웹에서 긁어온 이미지는 두 가지 문제를 동시에 갖는다.

| 문제 | 구체적 양상 | 특징 학습에 미치는 영향 |
|---|---|---|
| **품질(quality)** | 근접 중복(near-duplicate), 저해상도, 밈/스크린샷/광고 배너 같은 노이즈 | 중복이 많으면 같은 이미지를 반복 학습하는 셈이 되어 사실상 데이터 양이 줄고, 노이즈는 표현을 흐린다 |
| **다양성(diversity) / 균형** | 소수의 지배적 모드(예: 특정 인기 피사체, 텍스트 이미지)에 분포가 심하게 치우침 | 논문 표현으로 "a few dominant modes에 과적합" — 롱테일 개념(희귀 동식물, 랜드마크 등)은 거의 학습되지 않는다 |

레이블이 있는 지도학습이라면 클래스 균형을 맞추는 식으로 통제할 수 있지만, SSL은 레이블이 없으므로 **데이터 분포를 조절할 도구 자체가 없었다**. 이것이 "통제하지 못했다"는 말의 실제 의미다.

## 4. DINOv2의 대응: 자동 큐레이션 파이프라인

이 문제 진단이 DINOv2의 데이터 기여로 직결된다. 사람 손으로 큐레이션한 작은 데이터셋(ImageNet-22k, ImageNet-1k train, Google Landmarks, 세밀 분류 데이터셋 등)을 **"씨앗"**으로 삼아, 12억 장의 uncurated 웹 이미지 풀에서 그와 시각적으로 유사한 이미지만 자동으로 골라 **LVD-142M**(1억 4200만 장)을 구성했다.

![Fig.3 DINOv2 데이터 처리 파이프라인](fig-1.jpeg)

그림에서 관찰되는 흐름을 위의 두 문제와 대응시키면:

1. **Embedding** — 위쪽 회색 실린더(Uncurated Data: 파이 사진 여러 장, 자동차, 접시 등 잡다한 웹 이미지)와 아래쪽 주황 실린더(Curated Data: 파이 사진 한 장)를 모두 ImageNet-22k로 자기지도 학습된 ViT-H/16 임베딩으로 변환한다. 메타데이터·텍스트·레이블은 전혀 쓰지 않는다.
2. **Deduplication** — 그림의 빨간 X 표시가 uncurated 쪽에서 중복된 파이 사진들을 제거하는 모습이다. 이는 **품질** 문제(근접 중복, 벤치마크 테스트셋 유출)에 대한 대응이다.
3. **Retrieval** — 큐레이션된 파이 이미지가 쿼리가 되어, uncurated 풀에서 가까운 이웃(비슷한 파이 사진)만 끌어온다. 무관한 자동차 이미지는 X로 걸러진다. 큰 쿼리 데이터셋은 이미지당 N=4개 최근접 이웃을, 작은 데이터셋은 k-means 클러스터에서 M장을 샘플링해 **다양성과 균형**을 통제한다.
4. **Augmented Curated Data** — 결과적으로 오른쪽 실린더처럼 "큐레이션된 씨앗 + 그와 닮은 웹 이미지"가 합쳐진, 크지만 분포가 통제된 데이터셋이 된다.

NLP에서 CCNet(Wenzek et al., 2020)이 위키피디아로 학습한 언어모델로 웹 텍스트를 점수화해 걸러낸 것과 같은 발상을 이미지에 적용한 것이다.

## 5. 실험적 확인: 큐레이션 유무가 실제로 특징 품질을 바꾸는가 (Table 2)

논문 §6.2는 이 카드의 주장을 통제 실험으로 검증한다. 같은 웹 소스에서 **무작위로 1억 4200만 장**을 뽑은 "Uncurated data"와 LVD-142M을 동일한 ViT-g/14, 동일한 반복 횟수로 학습해 비교했다.

| 학습 데이터 | INet-1k | Im-A | ADE-20k | Oxford-M | iNat2018 | iNat2021 | Places205 |
|---|---|---|---|---|---|---|---|
| INet-22k | 85.9 | 73.5 | 46.6 | 62.5 | 81.1 | 85.6 | 67.0 |
| **Uncurated data (142M)** | 83.3 | **59.4** | 48.5 | **54.3** | **68.0** | **76.4** | 67.2 |
| **LVD-142M** | 85.8 | **73.9** | 47.7 | **64.6** | **82.3** | **86.4** | 67.6 |

- 같은 양(142M)이라도 uncurated 데이터는 ImageNet-A에서 −14.5점, iNaturalist 2018에서 −14.3점, Oxford 검색에서 −10.3점 등 **크게 뒤진다**. 이것이 "특징 품질이 크게 떨어졌다"는 문장의 정량적 근거다.
- 반면 큐레이션된 LVD-142M은 ImageNet-1k 성능을 유지하면서, 큐레이션에 사용되지도 않은 도메인(iNaturalist, Places205)까지 개선한다 → **다양성 통제가 미지 도메인에도 이득**이 된다는 증거.
- 흥미롭게도 ADE-20k(분할)만은 uncurated가 약간 높다. 분할처럼 픽셀 수준의 지역적 통계가 중요한 과제에서는 잡다한 장면 다양성이 조금 도움이 될 수 있음을 시사하지만, 논문은 이를 예외로 두고 "대부분의 벤치마크에서 큐레이션이 낫다"고 결론짓는다.

## 6. 한 줄 정리

기존 SSL은 **작은 큐레이션 데이터(ImageNet-1k)에서만 검증**되었고, **크게 키우려던 시도는 uncurated 웹 데이터를 써서 품질·다양성을 통제하지 못해 특징 품질이 떨어졌다**(fine-tuning 없이는 못 쓸 정도). DINOv2는 이 진단으로부터 "큐레이션을 사람 대신 자기지도 임베딩 검색으로 자동화하자"는 LVD-142M 파이프라인을 도출했다.

## 기억 팁

- 세 키워드: **작은 큐레이션(ImageNet-1k) → uncurated로 확장 → 품질 하락**.
- 원인 두 단어: **품질(quality)** 과 **다양성(diversity)** 의 통제 실패.
- 수치 하나: 같은 142M이라도 uncurated vs curated에서 iNat2018 **68.0 vs 82.3**.
