# 자동 데이터 큐레이션에서 DINOv2가 기존 연구와 다른 점

**질문:** 자동 데이터 큐레이션에서 DINOv2가 기존 연구와 다른 점은?

**답:** 사전학습된 인코더, 메타데이터(해시태그 등), 어떤 지도 신호도 쓰지 않고 오직 **이미지 간 시각적 유사도**만으로 필터링한다. LAION 등이 pretrained encoder(CLIP)로 필터링한 것과 대비된다.

---

## 1. 논문의 원문 주장 (Related Work, "Automatic data curation")

DINOv2 논문(Oquab et al., 2023) 2장 관련 연구에서 저자들은 기존의 "웹 데이터 필터링" 접근들을 세 갈래로 정리한 뒤 자신들과의 차이를 한 문장으로 못박는다.

> Similarly, others have used **hashtags or other metadata** (Mahajan et al., 2018; Radford et al., 2021) or **pretrained vision encoders** (Schuhmann et al., 2021; 2022) to filter uncurated datasets. **Unlike these works, we use no pretrained encoders, metadata nor supervision to filter images and leverage visual similarity between images.**

즉, 기존 연구가 "이미지 밖의 신호"에 의존해 좋은 이미지를 골라냈다면, DINOv2는 "이미지 자체가 서로 얼마나 닮았는가"만을 기준으로 삼는다.

| 접근 | 대표 연구 | 필터링에 사용한 신호 | 필요한 외부 정보 |
|---|---|---|---|
| 해시태그·메타데이터 | Mahajan et al. 2018 (Instagram 3.5B), CLIP(Radford et al. 2021) | 해시태그, 캡션/검색어 등 텍스트 | 사람이 붙인 텍스트 라벨 |
| 사전학습 인코더 점수 | LAION-400M / LAION-5B (Schuhmann et al. 2021; 2022) | CLIP 이미지-텍스트 유사도 점수로 걸러냄 | 텍스트 지도로 학습된 CLIP 모델 + alt-text |
| **DINOv2 (LVD-142M)** | Oquab et al. 2023 | **자기지도 임베딩 간 코사인 유사도** | 없음(큐레이션된 시드 이미지셋만) |

### "pretrained encoder를 안 쓴다"의 정확한 의미

주의할 점이 하나 있다. DINOv2도 이미지를 임베딩으로 바꾸기 위해 **ImageNet-22k에서 자기지도로 학습한 ViT-H/16**을 사용한다(3장 "Self-supervised image retrieval"). 그러면 "사전학습 인코더를 안 쓴다"는 말과 모순 아닌가?

핵심은 **어떤 종류의 인코더인가**이다.

- LAION은 **텍스트 지도(이미지-캡션 쌍)로 학습된 CLIP**을 써서 "이미지가 캡션과 잘 맞는가"를 점수화한다. 필터링 기준 자체가 텍스트 지도에서 나온다.
- DINOv2는 **라벨 없이 자기지도로 학습된 인코더**를 쓰고, 텍스트나 라벨과의 정합성이 아니라 **이미지-이미지 시각적 거리**만 계산한다. 따라서 파이프라인 전체에 지도 신호·메타데이터·텍스트가 한 번도 들어가지 않는다.

논문 3장 첫 단락도 같은 점을 강조한다: *"Our pipeline does not require any metadata or text and directly works with images."* 비유하자면, 저자들은 텍스트 큐레이션 파이프라인 CCNet(Wenzek et al., 2020)에서 영감을 받았다고 밝힌다. CCNet은 Wikipedia로 학습한 언어모델로 크롤링 텍스트의 "위키피디아 같음"을 점수화해 골라내는데, DINOv2는 이를 "큐레이션된 이미지셋과 시각적으로 가까움"으로 옮겨온 것이다.

---

## 2. 실제 파이프라인 — 시각적 유사도만으로 어떻게 필터링하나

![Fig.3 DINOv2 데이터 처리 파이프라인](fig-1.jpeg)

위 그림(논문 Fig. 3)은 파이프라인 전체를 한눈에 보여준다. 그림에서 관찰되는 요소를 따라가 보자.

1. **두 개의 데이터 원통** — 위쪽 회색 "Uncurated Data"(웹 크롤링 원시 이미지 12억 장)와 아래쪽 주황색 "Curated Data"(ImageNet-22k, ImageNet-1k train, Google Landmarks, 세밀 분류 데이터셋 등). 그림에서 회색 원통에는 타르트·자동차·요리 등 잡다한 이미지가, 주황색 원통에는 타르트 이미지 하나가 들어 있다.
2. **Embedding** — 양쪽 이미지가 모두 같은 자기지도 ViT-H/16을 거쳐 벡터(그림의 작은 격자 막대)로 바뀐다. 텍스트나 태그는 어디에도 등장하지 않는다는 점이 이 그림의 요지다.
3. **Deduplication** — 비큐레이션 쪽에서 서로 거의 같은 이미지(그림에서 빨간 X가 찍힌 두 타르트)가 제거된다. Pizzi et al. (2022) 복제 탐지 임베딩으로 k=64 최근접 이웃을 찾고, 유사도 > 0.6인 연결 요소마다 대표 하나만 남긴다(1.3B → 1.1B). 이어서 평가 벤치마크의 train/test와 유사도 > 0.45인 이미지도 통째로 버린다(→ 744M).
4. **Retrieval** — 큐레이션된 시드 이미지(그림의 굵은 테두리 타르트)를 질의로 삼아, 비큐레이션 쪽에서 코사인 유사도가 높은 이웃을 끌어온다. 그림에서 자동차는 흐릿하게 사라지고(관련 없음), 요리 이미지가 살아남아 위로 올라간다.
5. **Augmented Curated Data** — 시드 데이터셋에 가져온 이미지가 합쳐져 최종 LVD-142M이 된다.

검색(retrieval) 단계의 두 가지 모드(부록 A.4):

- **샘플 기반**: 시드 데이터셋이 크면(1M 장 이상) 각 시드 이미지마다 최근접 이웃 N개를 가져온다. 기본 N=4(ImageNet-22k, Google Landmarks), ImageNet-1k는 N=32로 크게 잡아 핵심 축으로 삼는다. N을 더 키우면 여러 질의가 같은 이미지를 가져오는 충돌(collision)이 늘어나 4를 절충값으로 택했다.
- **클러스터 기반**: 시드가 작으면 비큐레이션 데이터를 k-means로 100,000개 클러스터로 나눈 뒤, 시드 이미지가 3장 이상 속한 클러스터에서 10,000장씩 샘플링한다(데이터셋당 최대 1M으로 균형 유지).

전 과정은 Faiss(GPU IVF-PQ 인덱스)로 구현되어 V100-32GB 8장 × 20노드 클러스터에서 2일 이내에 끝난다.

---

## 3. 왜 이 차이가 중요한가

**(a) 지도 신호의 편향을 물려받지 않는다.** 해시태그·alt-text는 사람이 쓴 텍스트라서 언어·문화·플랫폼 편향을 그대로 반영한다. CLIP 점수로 걸러낸 LAION은 CLIP이 잘 아는 개념 위주로 데이터가 쏠린다. 시각적 유사도 기반 큐레이션은 "어떤 이미지가 좋은가"를 텍스트가 아니라 시드 이미지셋의 분포로 정의하므로, 시드를 바꾸면 도메인을 바꿀 수 있다(예: 논문 6.2절에서 iNaturalist, Places205처럼 큐레이션에 쓰지 않은 도메인도 함께 향상됨).

**(b) 자기지도 학습의 철학과 일관된다.** DINOv2의 목표는 라벨 없이 범용 시각 특징을 얻는 것이다. 데이터 단계에서 텍스트 지도(CLIP)를 몰래 끌어오면 "자기지도"라는 주장이 반쪽이 된다. 파이프라인 전부를 이미지만으로 닫음으로써 전체 시스템이 진짜 self-supervised가 된다.

**(c) 큐레이션 효과는 실험으로 검증됐다.** 6.2절 Table 2에서 같은 웹 소스에서 무작위로 1.42억 장을 뽑은 "Uncurated data"와 LVD-142M을 같은 ViT-g/14, 같은 반복 수로 비교했다.

| 학습 데이터 | INet-1k | Im-A | ADE-20k | Oxford-M | iNat2018 | iNat2021 | Places205 |
|---|---|---|---|---|---|---|---|
| INet-22k | 85.9 | 73.5 | 46.6 | 62.5 | 81.1 | 85.6 | 67.0 |
| Uncurated data (142M) | 83.3 | 59.4 | **48.5** | 54.3 | 68.0 | 76.4 | 67.2 |
| **LVD-142M** | 85.8 | **73.9** | 47.7 | **64.6** | **82.3** | **86.4** | **67.6** |

무작위 웹 이미지는 같은 양이어도 ImageNet-A(59.4 vs 73.9), iNat2018(68.0 vs 82.3)처럼 크게 뒤진다. 시각적 유사도 기반 필터링만으로도 "지도 없는 큐레이션이 자기지도 사전학습에 실질적으로 도움이 된다"는 것을 보인 셈이다.

---

## 4. 한 줄 정리

기존 연구(Instagram 해시태그, CLIP, LAION)는 **텍스트·태그·지도 학습된 인코더**라는 "이미지 밖의 신호"로 웹 데이터를 걸렀지만, DINOv2는 **자기지도 임베딩 사이의 코사인 유사도**라는 "이미지 안의 신호"만으로 큐레이션된 시드셋과 닮은 이미지를 검색해 LVD-142M을 만들었다 — 파이프라인 어디에도 메타데이터·텍스트·라벨이 개입하지 않는다.

## 참고

- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, arXiv:2304.07193 — 2장 "Automatic data curation", 3장 "Data Processing", 6.2절, 부록 A.
- Schuhmann et al., LAION-400M (2021), LAION-5B (2022) — CLIP 유사도 점수 필터링.
- Mahajan et al. 2018 — Instagram 해시태그 약지도 사전학습; Radford et al. 2021 — CLIP.
- Wenzek et al. 2020 — CCNet, DINOv2 큐레이션이 영감을 받은 텍스트 파이프라인.
