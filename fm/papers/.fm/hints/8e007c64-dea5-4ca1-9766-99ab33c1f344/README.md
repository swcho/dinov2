# Wild 이미지 데이터의 난점과 DINOv2의 해법 — 개념 리밸런싱과 클러스터링

**Q.** 이미지 데이터를 wild에서 다룰 때 가장 큰 난점과 DINOv2의 해법은?

**A.** 개념을 리밸런싱해 소수의 지배적 모드에 과적합되지 않게 하는 것이 어렵다. 논문은 단순한 clustering 접근만으로도 이 문제가 충분히 잘 해결됨을 보였다.

---

## 1. 출처 문장 (Introduction, §1)

> "A major difficulty when dealing with images **in the wild** is to **rebalance concepts** and avoid **overfitting on a few dominant modes**. In this work, a **naive clustering approach** works reasonably well to resolve this issue."

이 카드는 위 두 문장을 그대로 묻는 것이다. 핵심 키워드 세 개를 기억하면 된다.

| 키워드 | 의미 |
|---|---|
| **images in the wild** | 웹 크롤링으로 모은, 아무 필터링·라벨·메타데이터가 없는 원시(uncurated) 이미지 |
| **rebalance concepts / dominant modes** | 웹 이미지의 분포는 극단적으로 편향되어 있어 소수 개념(모드)이 대부분을 차지함 → 이를 균형 있게 다시 맞춰야 함 |
| **naive clustering** | 임베딩 공간에서 k-means로 묶고, 클러스터 단위로 샘플링하는 단순한 방법 |

---

## 2. 왜 "리밸런싱"이 wild 데이터의 최대 난점인가

### 2-1. 웹 이미지는 롱테일(long-tail) 분포
인터넷에서 긁어온 이미지는 특정 모드가 압도적으로 많다(예: 셀피, 상품 사진, 스크린샷, 밈, 텍스트 이미지 등). 반대로 희귀 동식물, 특정 랜드마크, 의료·위성 등 특정 도메인 이미지는 거의 없다. 이런 분포에서 self-supervised 학습을 하면 모델은 **자주 등장하는 소수 모드에만 표현력을 쏟고**, 나머지 개념은 제대로 학습하지 못한다. 논문은 이를 "a few dominant modes에 과적합(overfitting)"이라고 표현한다.

### 2-2. 기존 SSL 스케일링 시도가 실패한 이유
논문 §1은 기존 연구(Caron et al., 2019; Goyal et al., 2021; 2022a)가 ImageNet-1k 밖으로 SSL을 확장하려 했지만 **uncurated 데이터를 그대로 썼기 때문에 feature 품질이 크게 떨어졌다**고 지적한다. 원인은 "데이터의 **품질(quality)과 다양성(diversity)**에 대한 통제 부재"이다. 즉 데이터를 많이 모으는 것보다 **개념의 균형**을 맞추는 것이 어렵고 중요하다는 것이 이 카드의 요지다.

### 2-3. NLP 파이프라인에서 영감
DINOv2의 데이터 파이프라인은 NLP의 CCNet(Wenzek et al., 2020)에서 영감을 받았다. CCNet은 Wikipedia로 학습한 언어 모델로 웹 텍스트를 점수화해 필터링한다. DINOv2는 이를 이미지로 옮겨, **외부 메타데이터·텍스트·수동 라벨 없이 이미지 간 시각적 유사도(data similarity)만으로** 필터링·리밸런싱한다.

---

## 3. DINOv2의 해법: 임베딩 → 중복 제거 → 검색(retrieval), 그리고 클러스터링

![Fig.3 DINOv2 데이터 처리 파이프라인](fig-1.jpeg)

Fig. 3은 파이프라인 전체를 보여준다. 그림에서 관찰되는 요소를 따라가면:

1. **Uncurated Data(회색 원통)와 Curated Data(주황 원통)**: 왼쪽 위 회색 원통이 웹에서 모은 원시 이미지(피자, 자동차, 접시 등 잡다한 이미지), 아래 주황 원통이 ImageNet-22k 등 정제된 시드(seed) 데이터셋이다.
2. **Embedding**: 두 소스의 모든 이미지를 self-supervised ViT-H/16(ImageNet-22k로 사전학습)으로 임베딩한다. 그림에서 각 이미지 옆에 붙은 작은 격자 막대가 임베딩 벡터를 뜻한다.
3. **Deduplication**: 회색 원통 안의 비슷한 피자 이미지 여러 장에 붉은 X가 그려져 있다 — 근접 중복(near-duplicate)을 제거해 다양성을 높이는 단계다(Pizzi et al., 2022의 copy-detection 사용; 1.3B → 1.1B → 벤치마크 test/val과의 상대 중복까지 제거하면 744M).
4. **Retrieval**: 주황 원통의 큐레이션 이미지(굵은 테두리의 피자)를 **질의(query)**로 삼아, 중복 제거된 uncurated 풀에서 가까운 이미지를 끌어온다. 그림에서 자동차는 X가 되어 버려지고, 피자와 유사한 접시 이미지만 살아남아 오른쪽으로 흘러간다.
5. **Augmented Curated Data**: 결과적으로 큐레이션 시드 + 검색으로 끌어온 wild 이미지가 합쳐진 **LVD-142M**이 만들어진다.

### 3-1. 리밸런싱을 실제로 수행하는 부분 — 클러스터 기반 검색
카드의 "clustering"은 §3과 부록 A.4에 구체적으로 나온다. 검색은 두 방식으로 이루어진다.

| 방식 | 적용 대상 | 절차 |
|---|---|---|
| **sample-based** | 1M장 이상의 큰 시드(ImageNet-22k, Google Landmarks v2) | 각 시드 이미지마다 최근접 N=4장(GLv2는 32장)을 가져와 데이터셋을 N배로 확장 |
| **cluster-based** | 작은 시드(Flowers-102, Food-101, ADE20K, KITTI 등 수천~수만 장) | uncurated 풀 전체를 분산 k-means로 **100,000개 클러스터**로 나눈 뒤, 시드 이미지가 3장 이상 속한 클러스터마다 **10,000장**씩 샘플링. 단 데이터셋별 최대 1M장으로 제한 |

이 절차가 곧 **리밸런싱**이다:
- 100k개 클러스터는 각각 "서로 다른 이미지 개념/내용"을 대표하도록 의도되었다.
- 웹에서 흔한 모드가 아무리 많아도 **한 클러스터에서 10,000장까지만** 뽑고, 한 시드 데이터셋에 대해 **최대 1M장**으로 자르므로, 지배적 모드가 데이터셋을 삼키지 못한다.
- 반대로 시드가 겨우 1,020장인 Flowers-102 같은 희귀 개념도 클러스터를 통해 약 1M장까지 끌어올려진다(Table 15). 즉 소수 개념이 **위로**, 지배 개념이 **아래로** 재조정된다.
- N을 4보다 크게 하면 여러 질의가 같은 이미지를 가져오는 **collision**이 늘어난다는 관찰도, 특정 모드로 몰리는 것을 피하려는 같은 맥락이다.

논문이 "naive"라고 부른 이유: 복잡한 밀도 추정이나 학습된 재가중치 없이, **k-means + 클러스터별 고정 개수 샘플링 + 상한**이라는 단순한 규칙만으로도 균형이 맞았기 때문이다.

### 3-2. 구현 규모
Faiss 라이브러리(GPU 가속 IVF + product quantization)로 8×V100 노드 20대에서 **2일 미만**에 LVD-142M을 생성했다.

---

## 4. 해법이 "충분히 잘" 작동했다는 증거 — Table 2 (§6.2)

같은 ViT-g/14를 같은 반복 횟수로, 데이터만 바꿔 학습한 결과(선형 프로브/mIoU/mAP):

| 학습 데이터 | INet-1k | Im-A | ADE-20k | Oxford-M | iNat2018 | iNat2021 | Places205 |
|---|---|---|---|---|---|---|---|
| INet-22k | 85.9 | 73.5 | 46.6 | 62.5 | 81.1 | 85.6 | 67.0 |
| **Uncurated (같은 소스에서 142M 무작위)** | 83.3 | **59.4** | 48.5 | 54.3 | **68.0** | **76.4** | 67.2 |
| **LVD-142M (curated)** | 85.8 | **73.9** | 47.7 | **64.6** | **82.3** | **86.4** | **67.6** |

- 같은 웹 소스에서 **무작위로 142M장을 뽑기만 한** Uncurated 행은 Im-A(59.4)와 iNaturalist(68.0/76.4)에서 크게 무너진다 — 지배 모드에 과적합되어 희귀·세밀한 개념(동식물 종 분류)을 놓친 전형적인 증상이다.
- 클러스터링 기반 리밸런싱을 거친 LVD-142M은 ImageNet-1k 성능을 유지하면서 나머지 모든 벤치마크에서 우위이며, **큐레이션 과정에 사용하지 않은 도메인**(iNat2018/2021, Places205)에서도 향상됐다. 논문은 이를 "our dataset provides a good balance of different types of images"로 요약한다.

---

## 5. 한 줄 정리

> **난점**: wild 이미지는 소수 지배 모드에 편향되어 있어, 개념 균형(리밸런싱)을 맞추지 못하면 SSL이 그 모드에만 과적합된다.
> **해법**: 이미지를 self-supervised 임베딩 → 중복 제거 → uncurated 풀을 k-means 100k 클러스터로 나누고, 큐레이션 시드가 속한 클러스터에서 클러스터당 고정 개수(10k)만 샘플링(상한 1M) — 이 "naive clustering"만으로 균형 잡힌 LVD-142M을 얻어 uncurated 학습 대비 Im-A +14.5, iNat2018 +14.3 포인트를 회복했다.

---

### 참고
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, arXiv:2304.07193 — §1 Introduction, §3 Data Processing, §6.2 Pretraining Data Source, Appendix A.3–A.4, Table 2, Table 15.
- Wenzek et al., *CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data*, 2020 — 파이프라인의 NLP 측 원형.
