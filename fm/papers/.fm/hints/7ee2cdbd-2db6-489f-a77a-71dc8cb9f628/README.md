# depth estimation에서 iBOT ViT-L이 OpenCLIP ViT-G를 이긴 것의 의미

**질문**: depth estimation 결과에서 iBOT ViT-L이 OpenCLIP ViT-G를 이긴 것의 의미는?

**답**: caption 기반 특징 학습이 깊이처럼 미묘한 패턴을 학습하지 못한다는 직관을 뒷받침한다. DINOv2 ViT-g/14는 NYUd DPT 0.279로 최상위다.

---

## 1. 어디에 나오는 이야기인가 — Table 11 (frozen feature depth estimation)

DINOv2 논문(Oquab et al., 2023) 7.4절 "Dense Recognition Tasks"의 **Depth estimation** 실험이다. 백본을 **완전히 얼린(frozen)** 상태에서 patch feature 위에 얕은 헤드만 학습해 단안 깊이 추정을 수행한다. 세 가지 세팅이 있다.

| 세팅 | 방법 |
|---|---|
| **lin. 1** | 마지막 층 patch 토큰에 [CLS]를 concat → 4배 bilinear upsample → 깊이를 256개 bin으로 나눠 분류하는 **선형층 하나** 학습 |
| **lin. 4** | 같은 프로토콜이지만 4개 층의 토큰을 concat (ViT-L은 l={5,12,18,24}, ViT-g는 l={10,20,30,40}) |
| **DPT** | Ranftl et al.(2021)의 DPT 디코더를 frozen 백본 위에 붙여 회귀 |

지표는 **RMSE(낮을수록 좋음)**. 벤치마크는 NYUd(실내), KITTI(실외 주행), 그리고 NYUd로 학습한 헤드를 SUN RGB-D에 그대로 적용하는 zero-shot 전이.

### Table 11 핵심 수치 (RMSE, 낮을수록 좋음)

| Method | Arch. | NYUd lin.1 | NYUd lin.4 | NYUd DPT | KITTI lin.1 | KITTI lin.4 | KITTI DPT | NYUd→SUN lin.1 | lin.4 | DPT |
|---|---|---|---|---|---|---|---|---|---|---|
| OpenCLIP | ViT-**G**/14 | 0.541 | 0.510 | 0.414 | 3.57 | 3.21 | 2.56 | 0.537 | 0.476 | 0.408 |
| MAE | ViT-H/14 | 0.517 | 0.483 | 0.415 | 3.66 | 3.26 | 2.59 | 0.545 | 0.523 | 0.506 |
| DINO | ViT-B/8 | 0.555 | 0.539 | 0.492 | 3.81 | 3.56 | 2.74 | 0.553 | 0.541 | 0.520 |
| **iBOT** | ViT-**L**/16 | **0.417** | **0.387** | **0.358** | **3.31** | **3.07** | **2.55** | **0.447** | **0.435** | 0.426 |
| DINOv2 | ViT-S/14 | 0.449 | 0.417 | 0.356 | 3.10 | 2.86 | 2.34 | 0.477 | 0.431 | 0.409 |
| DINOv2 | ViT-B/14 | 0.399 | 0.362 | 0.317 | 2.90 | 2.59 | 2.23 | 0.448 | 0.400 | 0.377 |
| DINOv2 | ViT-L/14 | 0.384 | 0.333 | 0.293 | 2.78 | 2.50 | 2.14 | 0.429 | 0.396 | 0.360 |
| **DINOv2** | ViT-**g**/14 | **0.344** | **0.298** | **0.279** | **2.62** | **2.35** | **2.11** | **0.402** | **0.362** | **0.338** |
| (참고) SOTA, Li et al. 2022b | 풀 파인튜닝 | | | 0.330 | | | 2.10 | | | 0.421 |

굵은 글씨 두 줄이 이 카드의 핵심이다.

## 2. "역전"이 왜 놀라운가 — 파라미터 수 차이

- **OpenCLIP ViT-G/14**: 비전 타워만 약 **1.8B** 파라미터(폭 1664, 48층). LAION-2B 이미지-텍스트 쌍으로 학습된, 당시 공개된 최강의 약지도(WSL) 범용 특징.
- **iBOT ViT-L/16**: 약 **0.3B** 파라미터(폭 1024, 24층). ImageNet-22k만으로 자기지도 학습.

즉 **6배 가까이 작은 모델**이, **더 작은 데이터셋**(INet-22k vs LAION-2B)으로 학습됐음에도 깊이 추정에서 **거의 모든 열에서 이겼다**:

- NYUd lin.1: 0.417 vs 0.541 (RMSE 23% 낮음)
- NYUd DPT: 0.358 vs 0.414
- KITTI lin.1: 3.31 vs 3.57
- NYUd→SUN lin.1: 0.447 vs 0.537

유일하게 OpenCLIP이 앞선 곳은 SUN RGB-D DPT(0.408 vs 0.426) 한 칸뿐이다. 반면 ImageNet 분류 같은 **의미(semantic) 중심 벤치마크**에서는 OpenCLIP ViT-G가 iBOT ViT-L을 크게 앞선다(Table 4: linear 86.2 vs 82.3). 즉 이 역전은 "OpenCLIP이 나쁜 모델"이라는 뜻이 아니라, **깊이라는 특정 정보에 대해 학습 신호의 종류가 모델 크기보다 중요하다**는 뜻이다.

## 3. 왜 캡션 지도는 깊이를 놓치는가

논문 서론의 주장을 그대로 인용하면:

> "This form of text-guided pretraining limits the information that can be retained about the image since **captions only approximate the rich information in images**, and **complex pixel-level information may not surface with this supervision**."

풀어서 설명하면:

1. **캡션은 이미지의 손실 압축이다.** "a wooden table with chairs in a dining room"이라는 문장에는 어떤 물체가 있는지는 있지만, 의자가 테이블보다 카메라에서 몇 미터 뒤에 있는지, 바닥이 어떻게 기울어져 멀어지는지 같은 **기하 정보는 거의 없다**. 사람은 사진을 설명할 때 깊이를 거의 언어화하지 않는다.
2. **contrastive 목표는 캡션을 맞추는 데 필요한 정보만 보존한다.** CLIP류 loss는 이미지 임베딩 하나([CLS] 수준)와 텍스트 임베딩을 정렬한다. 텍스트를 구별하는 데 도움이 안 되는 정보(픽셀 단위 기하, 표면 방향, 상대 거리)는 학습 압력이 없으므로 특징에서 **버려지거나 흐릿하게 남는다**. 논문이 "subtle patterns like this one"이라고 부르는 것이 바로 이런, **언어로 잘 기술되지 않는 정보**다.
3. **감독이 이미지 전체 수준에서만 들어온다.** 패치 하나하나가 무엇을 표현해야 하는지에 대한 직접 신호가 없어 dense prediction에 필요한 국소 정보가 약하다.

Figure 7의 정성 결과가 이를 그대로 보여준다.

![Figure 7: frozen OpenCLIP-G vs DINOv2-g 위 선형 프로브의 분할·깊이 결과](fig-1.jpeg)

- **NYUd·SUN-RGBd 행**(2~3번째 줄)을 보면 OpenCLIP-G의 깊이 맵은 전체적으로 **노이즈가 심하고 얼룩진 텍스처**처럼 보인다 — 물체 경계가 거의 살아 있지 않다. DINOv2-g는 소파, 식탁·의자, 탁자 위 물체가 **매끈한 깊이 면으로 분리**되어 나온다.
- 논문 본문이 지적한 사례: SUN RGB-D 오른쪽 이미지의 **흔들의자(chair)**는 OpenCLIP에서 완전히 사라지지만 DINOv2에서는 올바른 깊이에 자리 잡는다.
- **KITTI 행**(마지막 줄)은 둘 다 대략 도로가 가까움/하늘이 멂이라는 구조는 잡지만, DINOv2 쪽이 차량과 나무의 형태가 더 선명하다.
- 첫 줄 ADE20K 분할에서도 OpenCLIP은 "many artifacts and disconnected components"가 보인다 — 패치 수준 정보 부족이 깊이뿐 아니라 분할에도 같은 방식으로 드러난다.

논문은 공정하게도 OpenCLIP 특징 역시 깊이를 **선형 분리는 할 수 있다**고 인정한다(둘 다 깊이 감독 없이 학습됐음에도). 차이는 "할 수 있냐"가 아니라 **얼마나 정밀하고 매끈하게 담겨 있느냐**이다.

## 4. 왜 자기지도(특히 iBOT)는 깊이를 보존하는가

iBOT는 두 가지 loss를 함께 쓴다.

- **DINO loss (image-level)**: 같은 이미지의 다른 crop에서 나온 [CLS] 토큰이 같은 prototype 분포를 갖게 한다 → 전체 의미.
- **iBOT loss (patch-level, Masked Image Modeling)**: student 입력 패치 일부를 마스킹하고, 그 위치의 student 출력이 teacher의 **같은 위치 패치 토큰** 분포를 맞추게 한다:

  $$\mathcal{L}_{iBOT} = -\sum_i p_{ti}\log p_{si}$$

이 패치 수준 목표가 핵심이다. 가려진 패치가 무엇이어야 하는지를 **주변 패치의 맥락**으로부터 예측해야 하므로, 특징에는 "이 영역이 어떤 표면이고, 이웃과 어떻게 이어지며, 어떤 방향으로 기울어 있는가" 같은 **국소 기하·연속성 정보**가 자연스럽게 인코딩된다. 텍스트라는 병목이 없으므로 "언어로 안 표현되는" 정보도 버릴 이유가 없다. 서론의 표현으로는 SSL이 "can capture information at the **image and pixel level**"이라는 것.

논문 자체의 ablation도 이를 뒷받침한다. Table 3(b)에서 iBOT MIM loss를 제거하면 ImageNet 분류는 거의 변하지 않지만 **ADE20k 분할 등 dense task 성능이 눈에 띄게 떨어진다** — 패치 수준 loss가 정확히 dense 정보를 책임진다는 증거다.

참고로 같은 표에서 **MAE ViT-H/14**(0.517 / 0.415)가 iBOT ViT-L보다 못한 것도 흥미롭다. MAE 역시 패치 수준 SSL이지만 픽셀 재구성 목표라 특징 자체가 frozen 상태에선 선형 분리가 잘 안 되기로 알려져 있다(fine-tuning에 강함). 즉 "패치 수준 + discriminative(prototype 분포 매칭)"라는 iBOT 조합이 frozen dense feature에 특히 유리하다.

## 5. 서론 주장과의 연결 — 논문 전체 논리에서 이 관찰의 역할

DINOv2 논문의 큰 주장은 다음 두 문장으로 요약된다.

1. **텍스트 지도 없이 이미지만으로도 범용 특징을 만들 수 있다**(캡션은 정보를 제한하고 정렬된 이미지-텍스트 코퍼스를 요구한다).
2. 그러려면 **큐레이션된 대규모 데이터 + iBOT 계열 discriminative SSL을 안정적으로 스케일링**해야 한다.

Table 11의 iBOT vs OpenCLIP 역전은 주장 1의 **"캡션은 픽셀 수준 정보를 놓친다"는 가설의 실증**이다. DINOv2 자체의 성과가 아니라 **기존 공개 모델(iBOT) 두 개를 비교한 것**이기 때문에 더 설득력이 있다 — "우리가 잘 만들어서"가 아니라 "학습 신호의 성격 때문에" 생기는 차이라는 뜻이다. 그런 다음 DINOv2 결과가 주장 2를 보인다: 같은 SSL 계열을 제대로 스케일하면 iBOT보다도 훨씬 좋아진다.

## 6. DINOv2 ViT-g/14 0.279의 위치

- NYUd DPT **0.279**는 Table 11 전체에서 **가장 낮은 RMSE**다. 표 상단에 참고로 적힌 **SOTA 0.330(Li et al., 2022b, 풀 파인튜닝)** 보다도 낮다 — 즉 **백본을 얼린 채 DPT 디코더만 학습**했는데 end-to-end 학습된 전용 깊이 모델을 넘었다.
- KITTI DPT 2.11은 SOTA 2.10과 사실상 동률, NYUd→SUN DPT 0.338은 SOTA 0.421을 크게 앞선다(실내 학습 → 다른 도메인 zero-shot 전이가 매우 잘 됨).
- OpenCLIP ViT-G(0.414)와 비교하면 NYUd DPT RMSE가 **약 1/3 감소**. 파라미터는 ViT-g(약 1.1B)가 ViT-G(약 1.8B)보다 오히려 적다.
- 스케일링 경향도 깨끗하다: DINOv2 ViT-S 0.356 → B 0.317 → L 0.293 → g 0.279. 그리고 **DINOv2 ViT-S/14(0.356)가 iBOT ViT-L/16(0.358)과 비슷**하다 — 14배 작은 모델로 iBOT ViT-L급 깊이 특징을 얻은 셈이라, "SSL을 제대로 스케일하면"이라는 논문의 주장이 여기서도 확인된다.

## 한 줄 정리

> 작은 iBOT ViT-L(0.3B, INet-22k)이 거대한 OpenCLIP ViT-G(1.8B, LAION-2B)를 깊이 추정에서 이긴 것은 **캡션 지도가 언어로 잘 표현되지 않는 픽셀 수준 기하 정보를 특징에 담지 못한다**는 서론의 가설을 뒷받침하는 증거이고, 그 SSL 계열을 제대로 스케일한 DINOv2 ViT-g/14는 frozen 백본만으로 NYUd DPT 0.279를 기록해 풀 파인튜닝 SOTA(0.330)까지 넘어섰다.
