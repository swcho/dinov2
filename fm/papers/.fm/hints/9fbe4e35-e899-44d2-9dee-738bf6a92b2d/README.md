# DINO 알고리즘이 특별히 주목받은 이유는?

## 한 줄 정답

**마지막 attention layer가 이미지의 의미적으로 일관된(semantically consistent) 부분에 자연스럽게 집중해 해석 가능한 attention map을 만들었기 때문**이다. 이 성질 덕분에 LOST 같은 비지도 object discovery 알고리즘이 DINO 위에 구축될 수 있었다.

논문 원문(Vision Transformers Need Registers, §1 Introduction):

> In particular, the DINO algorithm is shown to produce models that contain explicit information about the semantic layout of an image. Indeed, qualitative results show that **the last attention layer naturally focuses on semantically consistent parts of images and often produces interpretable attention maps**. Exploiting these properties, object discovery algorithms such as LOST (Siméoni et al., 2021) build on top of DINO.

---

## 1. 배경: DINO는 무엇을 했나

DINO(**DI**stillation with **NO** labels, Caron et al., ICCV 2021)는 라벨 없이 student/teacher 두 네트워크를 self-distillation으로 학습시키는 self-supervised 방법이다. 논문 제목이 "**Emerging** Properties in Self-Supervised Vision Transformers"인 이유가 핵심인데, 저자들이 의도적으로 설계한 게 아니라 **학습 결과로 저절로 튀어나온(emergent)** 성질이 있었다는 뜻이다.

그 성질이 바로 이것이다.

- 마지막 layer에서 `[CLS]` 토큰이 patch 토큰들을 향해 쏘는 attention을 이미지 격자로 되돌려 그려보면(= attention map), **객체의 실루엣이 그대로 드러난다.**
- segmentation mask, bounding box, pixel label을 **단 한 번도 본 적이 없는데도** 그렇다.
- head마다 서로 다른 의미 영역에 붙는 경향까지 있었다.
- 이 성질은 "self-supervised **+ ViT**"의 조합에서 두드러졌고, supervised ViT나 convnet에서는 그만큼 깨끗하게 나오지 않았다.

즉 DINO가 주목받은 포인트는 "ImageNet linear probe 점수가 몇 % 올랐다"가 아니라, **표현(feature) 안에 이미지의 semantic layout이 명시적으로(explicitly) 들어 있다는 사실이 눈으로 확인됐다**는 데 있다.

## 2. 그림으로 보기 — DINO만 예외적으로 깨끗한 attention map

![Fig 2: 여러 ViT의 마지막 layer attention map 비교 (DINO만 깨끗)](fig-1.jpeg)

*(Registers 논문 Figure 2. 왼쪽부터 입력 / DeiT-III-B / DeiT-III-L / OpenCLIP-B / OpenCLIP-L / DINO-B / DINOv2-g)*

이 그림이 카드 내용을 가장 직접적으로 보여준다. 각 열은 서로 다른 감독 방식으로 학습된 ViT의 `[CLS]→patch` attention map이다.

- **DINO-B 열(오른쪽에서 두 번째)**: 1행 사람+물보라, 2행 고양이, 4행 도넛 4개 — **밝은 영역이 실제 객체 형태를 따라 뭉쳐 있다.** 특히 4행에서는 도넛 4개가 각각 별개의 밝은 덩어리로 분리돼 보인다. 이것이 "의미적으로 일관된 부분에 집중한다 / 해석 가능하다"는 말의 실체다.
- **나머지 모든 열(DeiT-III, OpenCLIP, DINOv2-g)**: 객체 형태는 희미하고, 대신 **배경 여기저기에 노란 점(peaky outlier)이 튄다.** 사람이 보고 "여기가 객체구나" 하고 읽어낼 수 없다.

Registers 논문의 출발점이 바로 이 대비다. label supervision(DeiT-III), text supervision(OpenCLIP), self-supervision(DINOv2) — **감독 방식과 무관하게 전부 artifact가 있고, DINO만 예외**다. 논문은 이를 두고 "DINO가 사실 예외이고, DINOv2는 vision transformer의 baseline 거동에 부합한다"고 정리한다.

> 참고: 이 artifact의 정체는 이후 §2에서 밝혀진다. 출력 norm이 주변보다 약 **10배** 큰 high-norm 토큰이, 정보량이 적은 **배경 패치**에서 전체 시퀀스의 약 **2%** 비율로 나타나 모델의 내부 연산용 스크래치패드로 **재활용(repurpose)** 되는 현상이다. Registers는 이 역할을 대신 맡을 학습 가능한 토큰을 명시적으로 붙여주는 해법이다.

## 3. 왜 이게 중요한가 — LOST가 DINO 위에 세워졌다

해석 가능한 attention map/feature는 단순한 "보기 좋은 시각화"가 아니라 **downstream 알고리즘의 입력**이 되었다.

**LOST**(Siméoni et al., 2021, *Localizing Objects with Self-Supervised Transformers and no Labels*)의 동작:

1. DINO의 마지막 layer feature(key 등)로 patch 간 유사도 **Gram matrix**를 만든다.
2. 유사도가 양수인 patch끼리 연결한 그래프에서, **degree가 가장 낮은 patch를 seed**로 고른다. (배경은 서로 비슷해 degree가 높고, 객체 patch는 상대적으로 낮다는 직관)
3. seed와 양의 상관을 가지면서 degree가 낮은 patch들을 순서대로 붙여 **seed expansion**을 한다.
4. 확장된 seed 집합과 평균적으로 양의 상관을 갖는 patch를 모아 마스크/박스를 만든다.

여기서 결정적인 건 **라벨도, 학습도, 탐지 head도 전혀 없다**는 점이다. LOST는 DINO feature의 기하 구조를 읽기만 한다. 그래서 DINO의 feature map이 **매끄럽고(smooth) 객체 단위로 일관**해야만 성립한다. 논문 표현대로 이런 방법들은 "attention map에 담긴 정보를 모아 supervision 없이 객체를 탐지"하며 "컴퓨터 비전의 새로운 지평을 열었다".

## 4. 반증으로 확인하기 — DINOv2는 왜 LOST와 안 맞았나

이 카드의 주장이 진짜라는 것은 **성질이 깨졌을 때 무슨 일이 벌어지는지**를 보면 확실해진다.

DINOv2는 dense prediction(단안 depth, semantic segmentation)에서 frozen backbone + linear head만으로 강력한 성능을 내는데도, **LOST와는 놀랍도록 궁합이 나빴다.** 지도학습 backbone 수준의 실망스러운 성능만 나왔다. 원인이 바로 Fig 2에서 보이는 artifact다.

![Fig 13: LOST 중간 계산 결과 (registers 유무 비교)](fig-2.jpeg)

*(Registers 논문 Figure 13. 행: LOST score / seed와의 dot product / seed expansion)*

- **DeiT-III w/o REG, DINOv2 w/o REG 열**: LOST score와 seed expansion 맵에 **배경의 노란 outlier 점들이 그대로 박혀 있다.** 특히 DeiT-III w/o REG는 이미지 가장자리를 따라 밝은 점이 줄지어 있어, "degree가 낮은 patch를 seed로 고른다"는 LOST의 전제가 그대로 오염된다. 즉 **artifact patch가 seed로 뽑혀 엉뚱한 곳을 객체로 지목**하게 된다.
- **w/ REG 열**: 점들이 사라지고 새 형상 위로 매끄러운 덩어리가 형성된다. 중간 단계(dot prod. w/ seed)에서도 seed가 객체 위에 제대로 얹힌 게 보인다.

수치로도 그대로 나타난다 (Table 3, corloc):

| backbone | VOC 2007 | VOC 2012 | COCO 20k |
|---|---|---|---|
| DeiT-III | 11.7 | 13.1 | 10.7 |
| DeiT-III + reg | **27.1** | **32.7** | **25.1** |
| OpenCLIP | 38.8 | 44.3 | 31.0 |
| OpenCLIP + reg | 37.1 | 42.0 | 27.9 |
| DINOv2 | 35.3 | 40.2 | 26.9 |
| DINOv2 + reg | **55.4** | **60.0** | **42.0** |

DINOv2는 register 추가만으로 VOC2007에서 35.3 → 55.4 (**+20.1 corloc**)로 뛴다. 다만 원조 **DINO의 61.9 corloc**(Siméoni et al. 보고치)에는 아직 못 미친다 — DINO가 얼마나 특이한 위치에 있었는지를 보여주는 숫자다.

> OpenCLIP은 register를 넣어도 LOST 점수가 오히려 살짝 내려간다. 논문 §C의 분석에 따르면 OpenCLIP 실험에서는 feature 대신 **value**를 쓰는데, value projection이 outlier를 걸러내는(outlier가 value projection의 null space에 사는 것으로 보이는) 현상 때문에 register 없이도 이미 매끄러웠다.

## 5. 정리 — 왜 하필 이 카드인가

이 카드는 Registers 논문의 **문제 제기 자체**다. 논리 흐름을 한 줄로 이으면:

```
DINO: 마지막 attention layer가 의미 단위에 자연히 집중 → 해석 가능한 attention map
  → LOST 등 비지도 object discovery가 그 위에 성립 (SOTA를 크게 앞섬)
  → 그런데 후속작 DINOv2는 LOST와 안 맞음 (artifact 때문)
  → 조사해 보니 DeiT-III, OpenCLIP 등 대부분의 ViT도 마찬가지 = DINO가 오히려 예외
  → 원인: 배경 패치를 내부 연산용으로 재활용하는 high-norm 토큰
  → 해법: register 토큰을 명시적으로 추가 → artifact 소멸, attention map이 다시 DINO처럼 해석 가능해짐
```

**암기 포인트 3개**
1. 주체는 **마지막 attention layer**의 `[CLS]→patch` attention map이다. (feature 일반이 아니라)
2. 핵심 단어는 **의미적으로 일관된(semantically consistent)** + **해석 가능한(interpretable)**, 그리고 라벨 없이 **저절로 생긴(emergent)** 성질이라는 것.
3. 실용적 귀결이 **LOST(Siméoni et al., 2021) 같은 비지도 object discovery가 DINO 위에 세워졌다**는 것.

---

## 참고

- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), ICCV 2021 — https://arxiv.org/abs/2104.14294
- Siméoni et al., *Localizing Objects with Self-Supervised Transformers and no Labels* (LOST), BMVC 2021 — https://arxiv.org/abs/2109.14279 / 코드 https://github.com/valeoai/LOST
- Darcet et al., *Vision Transformers Need Registers*, ICLR 2024 — https://arxiv.org/abs/2309.16588 (이 카드의 출처, §1 / Fig. 2 / §3.3 / Fig. 13 / Table 3)
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023 — https://arxiv.org/abs/2304.07193
