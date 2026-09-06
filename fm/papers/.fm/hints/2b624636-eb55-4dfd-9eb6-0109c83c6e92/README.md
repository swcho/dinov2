# 지리적 공정성 평가에서 드러난 DINOv2의 편향

> **Q.** 지리적 공정성 평가에서 드러난 DINOv2의 편향은?
> **A.** Europe 대비 Africa에서 성능이 25.7% 낮아 서구권 편향이 남아 있다. 고소득 가구 대비 저소득 가구 성능 차이도 31.7%로 크다.

출처: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* (arXiv 2304.07193v2), §8.1 "Geographical Fairness", Table 12.

---

## 1. 무엇을 측정했나 — Dollar Street 벤치마크

**Dollar Street**는 원래 스웨덴 Gapminder 재단의 사진 프로젝트다. 세계 각국 가정을 방문해 "칫솔", "침대", "부엌", "신발 보관 장소" 같은 일상 개념을 찍고, 가구를 **월 소득 순으로 한 줄(street)에 세운다**는 컨셉이다. 같은 "칫솔"이라도 저소득 가정에서는 나뭇가지일 수 있고, 고소득 가정에서는 전동 칫솔일 수 있다.

컴퓨터 비전 벤치마크로는 두 계열의 정리본이 있다.

| 정리본 | 내용 |
|---|---|
| De Vries et al. 2019, *Does object recognition work for everyone?* | 상용 인식 API가 저소득·비서구 가정 사진에서 10~20% 성능이 떨어짐을 처음 보임. DINOv2 논문이 인용한 출처 |
| Rojas et al. 2022 (NeurIPS Datasets & Benchmarks) | 데이터셋을 공개 정리한 버전. 논문 §8.1이 밝힌 규모: **54개국 289가구, 16,073장, 94개 개념** |

평가 프로토콜은 Goyal et al. 2022b (*Fairness Indicators for Systematic Assessments of Visual Feature Extractors*)를 따른다. 백본을 **얼린 채(frozen)** 94개 개념을 분류하는 선형 분류기를 학습하고, 정확도를 **소득 구간(low / medium / high)**과 **지역(Africa / Asia / Americas / Europe)**별로 쪼개서 본다. 이 평가는 ViT-g/14(가장 큰 모델)로만 수행했다.

### 왜 소득·지역별 성능 격차가 "편향 지표"가 되는가

- 94개 개념은 **가구의 소득과 위치에 따라 시각적 형태가 달라지도록** 골라진 것이다. 즉 "개념은 같은데 생김새만 다른" 문제다.
- 모델이 개념을 정말로 이해했다면 나뭇가지 칫솔도 전동 칫솔도 "칫솔"로 봐야 한다. 특정 소득/지역에서만 정확도가 떨어진다면, 모델은 개념이 아니라 **학습 데이터에 많이 등장한 특정 지역·계층의 외형**을 학습한 것이다.
- 따라서 그룹 간 정확도 격차 = 사전학습 데이터의 분포가 표현(feature)에 새겨진 정도를 읽는 프록시가 된다. 격차가 0에 가까울수록 공정하다.

---

## 2. Table 12 재구성 — DINOv2 vs SEERv2

논문 Table 12 (정확도 %, 높을수록 좋음). 표의 데이터 열 "RT-10B / IR-1B"는 OCR 오탈자로, 실제로는 **RegNet-10B / IG-1B**(Instagram 10억 장, 지리적으로 다양한 비큐레이션 데이터)다.

| Method | Arch. | Data | low | medium | high | Africa | Asia | Americas | Europe |
|---|---|---|---|---|---|---|---|---|---|
| SEERv2 | RegNet-10B | IG-1B | 59.7 | 78.5 | 86.6 | 65.9 | 76.3 | 81.1 | 85.6 |
| **DINOv2** | ViT-g/14 | LVD-142M | **67.4** | **83.3** | **90.5** | **74.0** | **81.6** | **86.2** | **89.7** |

### 격차 계산 (표 숫자 기준)

| 비교 축 | SEERv2 | DINOv2 |
|---|---|---|
| Europe − Africa (절대 pt) | 85.6 − 65.9 = **19.7 pt** | 89.7 − 74.0 = **15.7 pt** |
| high − low (절대 pt) | 86.6 − 59.7 = **26.9 pt** | 90.5 − 67.4 = **23.1 pt** |
| 그룹 순위 | low < Africa < Asia < medium < Americas < Europe < high | 동일한 순서 |

읽는 법:

1. **모든 열에서 DINOv2가 SEERv2보다 높다** (Africa +8.1, low +7.7 등). 절대 성능이 좋다.
2. **격차도 조금 줄었다** (지역 19.7→15.7 pt, 소득 26.9→23.1 pt). 논문이 "slightly fairer"라고 쓴 근거. Goyal 2022a에서 보고된 ImageNet 지도학습 베이스라인보다는 훨씬 낫다.
3. 그러나 **격차의 방향과 구조는 그대로다.** 저소득·Africa가 항상 꼴찌, 고소득·Europe이 항상 1등. 소득 축의 격차(23 pt)가 지역 축(16 pt)보다 더 크다는 점도 두 모델이 똑같다.

### 25.7%·31.7%는 어디서 나오나

본문의 문장은 다음 둘이다.

- "in Africa, where our model performance drops by **25.7%** compared to Europe" → 지역 격차
- "performs significantly better on high-income households than low-income ones, with a difference of **31.7%**" → 소득 격차

주의: 이 두 숫자는 Table 12의 반올림된 지역/소득 평균값으로 단순 재현되지 않는다(절대차 15.7/23.1 pt, Europe·high 기준 상대차 17.5%/25.5%). 논문이 계산식을 명시하지 않았으므로 **가구/국가 단위로 집계한 상대 격차**로 추정되지만, 암기용으로는 "논문이 본문에서 밝힌 헤드라인 수치 = 지역 25.7%, 소득 31.7%"로 기억하면 된다. 핵심 메시지는 어느 계산법을 쓰든 같다: **Africa와 저소득 가구에서 성능이 크게 떨어지며, 소득 격차가 지역 격차보다 더 크다.**

---

## 3. SEERv2를 이겼는데도 편향이 남은 이유 — 데이터 큐레이션이 편향의 원천

SEERv2는 "지리적으로 다양한 비큐레이션 Instagram 이미지로 학습하면 더 공정해진다"는 것을 주장한 모델이다(Goyal et al. 2022a). DINOv2가 그 SEERv2보다 모든 그룹에서 높고 격차도 조금 줄였지만, 격차 자체가 사라지지 않은 이유는 **LVD-142M이 만들어진 방식**에 있다.

![DINOv2 데이터 처리 파이프라인 (Figure 3)](fig-1.jpeg)

Figure 3의 흐름을 보면, 왼쪽에 **Curated data**(큐레이션된 시드 데이터셋)와 **Uncurated data**(웹 크롤링 12억 장)가 있고, 둘을 임베딩한 뒤 비큐레이션 이미지를 중복 제거하고 **큐레이션 이미지와 가까운 것만 retrieval로 골라** 최종 Augmented curated data(LVD-142M)를 만든다. 즉 12억 장의 웹 이미지 중 무엇을 남길지는 전적으로 **시드 데이터셋이 어떤 분포를 가졌는지**가 결정한다.

그 시드가 무엇인지는 Table 15에 있다. 규모가 큰 것만 뽑으면:

| 시드 데이터셋 | 포함 방식 | 최종 장수 | 성격 |
|---|---|---|---|
| ImageNet-22k | as is + sample retrieval | 14.2M + 56.8M | 영어 WordNet 명사 기반, Flickr 등 서구 웹 수집 |
| ImageNet-1k train | sample retrieval | 41.0M | 동일 계열 |
| Google Landmarks v2 | as is + sample retrieval | 1.6M + 6.3M | 유명 관광지 랜드마크, 서구·관광 중심 |
| Mapillary SLS | as is | 1.4M | 도시 스트리트뷰 |
| 나머지 fine-grained / seg / depth / retrieval 셋 | cluster retrieval | 각 ~1M | Caltech, CUB, Food-101, Cityscapes, Oxford/Paris 등 |

LVD-142M의 **약 80%(≈112M장)가 ImageNet 계열 시드에서 retrieval된 이미지**이고, 나머지 대부분도 서구 학계 벤치마크에서 나왔다. ImageNet의 지리적 편향(북미·서유럽 이미지 과대표집)은 Shankar et al. 2017 등에서 잘 알려진 사실이다. 따라서:

1. 시드가 서구 중산층 이상의 물건·풍경을 담고 있으면,
2. retrieval은 그와 **닮은** 웹 이미지만 끌어오므로 12억 장 중 아프리카 저소득 가정 사진 같은 것은 시드에 없는 만큼 걸러지고,
3. 결과적으로 큐레이션 데이터셋은 시드의 분포 편향을 **그대로 증폭**하며,
4. 자기지도 학습은 라벨 없이 이미지 분포 자체를 학습하므로 이 편향이 특징(feature)에 그대로 전이된다.

큐레이션이 성능을 끌어올린 바로 그 메커니즘(시드와 닮은 것만 남기기)이 동시에 편향을 굳히는 메커니즘이기도 하다는 점이 핵심이다. 논문의 Table 2가 "큐레이션이 비큐레이션보다 대부분 벤치마크에서 낫다"고 보여주는 것과 Table 12의 잔존 편향은 같은 설계의 두 얼굴이다.

### 왜 그래도 SEERv2보다는 나았나

- 모델 크기(ViT-g 1.1B)와 학습 레시피 차이로 **절대 성능이 전반적으로 높아져** 저소득/Africa 그룹도 함께 올라갔다.
- 시드 자체는 서구 중심이더라도 12억 장 웹 풀에서 retrieval을 했기 때문에 ImageNet-22k만 쓸 때보다는 다양성이 늘었을 것이다.
- 하지만 격차의 **구조**(소득 > 지역, low·Africa 최하)는 두 모델이 똑같다. "더 좋은 모델"이 "더 공정한 모델"과 같지 않다는 것을 보여준다.

---

## 4. 함의 — 데이터 큐레이션이 곧 편향의 원천

- DINOv2 논문은 "라벨 없이도 데이터를 잘 고르면 강한 범용 특징이 나온다"는 것을 증명한 논문이다. 그런데 **"잘 고른다"의 기준(시드 데이터셋)이 곧 모델의 세계관**이 된다. 자기지도 학습은 라벨 편향은 없지만 **분포 편향**은 그대로 물려받는다.
- 논문도 결론에서 이를 인정한다: "Despite improvements, we observe significant biases in our models toward wealthy households from Western countries."
- 실무적으로: DINOv2를 프리징해서 하위 태스크에 쓰면 이 편향이 하위 모델에도 상속된다. 비서구·저소득 환경(농촌 의료, 개발도상국 소매, 재난 대응 등)에 적용할 때는 Dollar Street 류의 그룹별 평가를 별도로 돌려봐야 한다.
- 대책 방향: 시드에 지리적으로 다양한 데이터셋(Dollar Street 자체, GeoDE 등)을 추가하거나, retrieval 시 지역/소득별 균형을 강제하는 것이 자연스러운 다음 단계다. 큐레이션 파이프라인을 손대지 않으면 모델을 키워도 격차는 남는다.

---

## 5. 한 줄 정리

Dollar Street(54개국·289가구·16,073장·94개념)에서 DINOv2 ViT-g는 SEERv2를 모든 그룹에서 이기고 격차도 조금 줄였지만(Table 12), **Africa는 Europe보다 25.7%, 저소득 가구는 고소득 가구보다 31.7% 낮아** 서구·부유층 편향이 남았다. 원인은 LVD-142M의 시드(ImageNet-22k/1k ≈ 80%, Google Landmarks 등)가 서구 중심이고, retrieval 기반 큐레이션이 그 분포를 그대로 증폭해 특징에 전이시켰기 때문이다.

## 관련 카드 힌트

- §8.2 라벨 연관 공정성(Casual Conversations, Table 13): 성별·피부색·나이별 Non-Human/Crime 라벨 예측은 없었으나 남성에게 Possibly-Human(Beard 등)이 자주 붙음.
- §4 데이터 파이프라인: 큐레이션 시드 → 임베딩 → 중복 제거 → retrieval(sample-based / cluster-based) → LVD-142M.
