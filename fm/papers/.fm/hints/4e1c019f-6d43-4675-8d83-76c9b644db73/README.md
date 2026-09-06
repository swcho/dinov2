# Relative deduplication — 무엇이며 임계값은 얼마인가?

> **핵심 한 줄**: 평가(벤치마크) 데이터셋의 **train/test split과 너무 닮은 이미지**를 학습용 uncurated 풀에서 걷어내는 단계. 코사인 유사도 **> 0.45**(self-dedup의 0.6보다 *더 엄격*)를 쓰고, reference 이미지가 속한 **duplicate component를 통째로 버려** 11억 장 → **7억 4400만 장**이 남는다.

출처: DINOv2 논문(arXiv 2304.07193) §3.2 "Deduplication" 및 Appendix A.3 "Deduplication".

---

## 1. 데이터 파이프라인 안에서의 위치

![DINOv2 데이터 처리 파이프라인(Figure 3)](fig-1.jpeg)

Figure 3은 LVD-142M을 만드는 4단계 흐름 — **Embedding → Deduplication → Retrieval → Augmented Curated Data** — 를 보여준다. 위쪽 회색 실린더가 웹에서 긁어온 *uncurated* 이미지, 아래쪽 주황 실린더가 ImageNet 같은 *curated* 이미지다. 그림의 **Deduplication** 열에서 빨간 X가 찍힌 이미지들이 바로 이 카드가 다루는 단계에서 제거되는 이미지다. 그림에는 한 열로 뭉쳐 있지만, 실제로는 두 개의 dedup이 순서대로 돌아간다.

| 단계 | 대상 | 기준(코사인 유사도) | 남기는 방식 | 결과 |
|---|---|---|---|---|
| **Self-deduplication** | uncurated 풀 내부의 서로 닮은 이미지 | **> 0.6** | 각 duplicate component에서 **대표 1장만 남김** | 1.3B(§3.2 기준 1.2B) → **1.1B** |
| **Relative deduplication** | 평가 데이터셋(train+test split)과 닮은 이미지 | **> 0.45** (더 엄격) | reference가 속한 component를 **통째로 버림** | 1.1B → **744M** |

두 단계 모두 같은 도구를 쓴다: Pizzi et al. (2022)의 copy-detection(SSCD) 임베딩, Faiss GPU 인덱스로 각 이미지의 $k=64$ 최근접 이웃 검색, 임계값 이상인 간선만 남긴 $k$-NN 그래프에서 disjoint-set(union-find)으로 **connected component** 추출. 유사도 함수는 Appendix A.2의 코사인 유사도다:

$$m(s,r)=\frac{f(s)\cdot f(r)}{\|f(s)\|_2\,\|f(r)\|_2}$$

---

## 2. Relative dedup의 목적 — 벤치마크 오염 방지

논문 원문(A.3): *"To reduce redundancy and also **properly evaluate the performance of our features**, we discard remaining images ... that are too similar to train and test splits of our evaluation datasets."*

- DINOv2는 웹 크롤 데이터(수십억 장)로 학습한다. 웹에는 ImageNet, Oxford/Paris, ADE20K 등 공개 벤치마크의 이미지가 그대로, 혹은 리사이즈·크롭·워터마크만 붙어 널려 있다.
- 이런 이미지가 학습 데이터에 섞이면 모델은 **test 이미지를 사실상 "본 채로"** 평가를 받게 된다 → **test-set contamination / data leakage**. 그 결과 linear probe, kNN, retrieval 점수가 진짜 일반화 능력보다 **과대평가**된다.
- 그래서 "우리 feature가 정말 좋은가"를 공정하게 말하려면, 학습 풀에서 벤치마크와 닮은 것들을 미리 제거해야 한다. §3.2에서도 *"We also remove near-duplicates of images contained in the test or validation set of any benchmark used in this work"*라고 명시한다.

### Self-dedup과 목적이 다르다
- **Self-dedup(0.6)**: 목적은 **중복 제거로 다양성 확보**. 같은 밈 이미지가 10만 장 있어도 학습에 도움이 안 되고 오히려 분포를 왜곡하니 1장만 남긴다. 그래서 "대표 1장 유지".
- **Relative dedup(0.45)**: 목적은 **평가 공정성**. 벤치마크와 관련 있는 건 하나도 남기면 안 되니 대표조차 남기지 않고 **component 전체 삭제**.

---

## 3. 왜 0.45가 "더 엄격한" 임계값인가

헷갈리기 쉬운 부분이다. 숫자가 *작아졌는데* 왜 *더 엄격*한가?

- 임계값은 "**이 값을 넘으면 중복으로 간주해 제거**"하는 컷이다.
- 코사인 유사도 0.6을 넘어야 제거되는 것보다, **0.45만 넘어도 제거**되는 쪽이 훨씬 많은 이미지를 걸러낸다. 즉 "근사 복제(near-duplicate)"의 범위를 넓게 잡아 **느슨하게 닮은 이미지까지 버리는** 것이다.
- 비유: 공항 보안 검색에서 "금속 탐지 민감도"를 높이면 벨트 버클까지 울린다. 오탐(false positive)이 늘지만, 한 건도 놓치면 안 되는 상황에선 그게 옳다. 벤치마크 유출은 "한 장도 놓치면 안 되는" 문제라 recall을 우선한 것.
- 대가: 벤치마크와 무관한 이미지도 일부 억울하게 제거된다. 하지만 풀이 11억 장이므로 3억 5천만 장을 버려도 충분히 남는다는 계산이다.

---

## 4. "Component를 통째로 버린다"의 의미 — 전이적 제거

Self-dedup은 component에서 **대표 1장을 남기지만**, relative dedup은 *"identifying the duplicate components (if any) to which each reference image belong and **discarding it entirely**"* — reference(평가셋) 이미지가 들어 있는 component를 **전부** 버린다.

이게 왜 중요한가? Connected component는 **전이적(transitive)**이다.

```
  test 이미지 T ── 0.5 ── 웹 이미지 A ── 0.5 ── 웹 이미지 B ── 0.5 ── 웹 이미지 C
```

- T와 A는 직접 닮았다(0.5 > 0.45). T와 C는 직접 비교하면 0.45를 못 넘을 수도 있다.
- 하지만 A–B–C가 사슬로 연결돼 있으면 셋 다 T와 같은 component에 속하고, **C도 삭제된다**.
- 즉 "test 이미지의 변형(크롭)의 변형(필터 적용)의 변형" 같은 **간접 근사 복제**까지 한꺼번에 걷어내는 효과. 임계값을 낮춘 것과 함께 recall을 극대화하는 두 번째 장치다.
- reference 이미지 자체는 uncurated 풀에 속하지 않으므로 그래프에 "질의"로만 들어가고, 걸린 component만 풀에서 지운다.

---

## 5. 감소 규모: 1.1B → 744M

- Self-dedup: 1.3B → 1.1B (약 15% 감소).
- Relative dedup: 1.1B → 744M — **약 3억 5600만 장(≈32%)** 이 추가로 사라진다.
- 벤치마크 유사 이미지가 정말로 3억 장이나 웹에 있는 건 아니다. **0.45라는 낮은 임계값 + component 전체 삭제**의 결합이 크게 걷어낸 결과다. 이것이 논문이 "stricter"라고 부른 이유이자, 평가 공정성을 위해 데이터 양을 기꺼이 희생한 지점이다.
- 이 744M 풀에서 이후 **Retrieval** 단계(Figure 3의 세 번째 열)로 curated 이미지와 가까운 것들만 뽑아 **LVD-142M**을 만든다.

> 참고: §3.2 본문은 uncurated 풀 크기를 "1.2B unique images"(PCA-hash dedup 후)라고 쓰고, Appendix A.3은 "1.3B"라고 쓴다. 논문 내부 표기 차이이며, 카드가 묻는 relative dedup 결과 **744M**은 두 곳에서 일치한다.

---

## 6. 어떤 데이터셋을 reference로 썼는가 — train *과* test 모두

- Appendix A.3: *"too similar to **train and test splits** of our evaluation datasets"*. test/validation만이 아니라 **train split도** reference에 포함한다.
- 왜 train까지? DINOv2의 평가는 backbone을 고정하고 그 위에 linear/kNN을 **평가셋의 train split으로** 학습시킨다. 사전학습 풀에 train split의 복제본이 들어 있으면 "사전학습에서 이미 해당 도메인을 외운 상태"가 되어 few-shot/저데이터 평가(예: ImageNet 1%/10% 실험)가 왜곡된다. 또한 Table 15의 curated source들(ImageNet-22k, ImageNet-1k train, Google Landmarks, Caltech-101, CUB, DTD, FGVC-Aircraft, Flowers-102 등)이 retrieval 질의로 쓰이므로, 그와 *거의 같은* 웹 복제본이 풀에 있으면 retrieval이 그냥 복제본을 되찾아오는 무의미한 결과가 된다.
- 논문 전체에서 쓰인 벤치마크(ImageNet-1k/V2/ReaL/A/R/Sketch, iNaturalist, Places205, 12개 fine-grained, Oxford/Paris/Met/AmsterTime retrieval, ADE20K/Cityscapes/VOC segmentation, NYUd/KITTI/SUN-RGBD depth 등)의 split이 reference 집합에 해당한다.

---

## 7. 이 처리가 없으면 벌어지는 일

1. **과대평가**: 모델이 test 이미지(또는 그 크롭/리사이즈)를 사전학습에서 봤다면, kNN 평가는 거의 "암기 조회"가 되고 linear probe 점수도 부풀려진다. "SSL이 supervised를 이겼다"는 주장이 무효가 된다.
2. **retrieval 벤치마크 붕괴**: Oxford/Paris 같은 instance retrieval은 정확히 같은 장소 사진을 찾는 과제라, 사전학습 풀에 정답 이미지가 있으면 사실상 답을 미리 본 것과 같다.
3. **비교의 불공정**: CLIP·OpenCLIP 등 웹 데이터 학습 모델과 비교할 때, 상대는 오염됐을 수 있어도 DINOv2는 오염을 제거했다고 말할 수 있어야 논문의 데이터 큐레이션 주장이 설득력을 갖는다.
4. **Retrieval 단계의 무의미화**: 위 6절처럼 curated 질의에 대해 웹 복제본만 되돌아온다.

이 문제는 LLM 쪽에서도 동일하게 다뤄진다(n-gram 기반 contamination check 등). DINOv2는 그 시각적 버전으로 **copy-detection 임베딩 + 낮은 임계값 + component 전체 삭제**를 택한 것이다.

---

## 암기 포인트

- **무엇**: 평가 데이터셋(train+test split)과 닮은 uncurated 이미지 제거 → 벤치마크 오염 방지.
- **임계값**: 코사인 유사도 **> 0.45** (self-dedup 0.6보다 낮음 = **더 엄격**, 더 많이 버림).
- **방식**: reference가 속한 duplicate component를 **통째로**(대표도 남기지 않고, 전이적으로) 삭제.
- **결과**: 1.1B → **744M**. 이후 retrieval로 LVD-142M.
- 기억법: "**0.6은 다양성, 0.45는 공정성** — 낮을수록 엄격."
