# Semantic segmentation의 "+ms" 설정: linear와 무엇이 다른가

> **출처**: DINOv2 논문(arXiv 2304.07193v2) §7.4 *Dense Recognition Tasks* — "Semantic segmentation" 단락과 Table 10.

## 한 줄 요약

`+ms`는 **linear 설정을 강화(boost)한 버전**이다. 백본은 그대로 얼려 두고(frozen), 선형 분류기 위에 세 가지를 더한다.

| 항목 | **linear** | **+ms** |
|---|---|---|
| 입력 특징 | 마지막 층의 패치 토큰 (차원 D) | **마지막 4개 층의 패치 토큰을 concat** (차원 4D) |
| 이미지 해상도 | 512×512 | **640** |
| 테스트 시 추론 | 단일 스케일 | **multiscale test-time augmentation** |
| 예측기 | 선형 층 1개 | 선형 층 1개 (동일) |
| 백본 | frozen | frozen (동일) |

핵심은 "더 복잡한 디코더"가 아니라 **선형 층은 그대로 두고 입력과 추론 방식만 바꿨다**는 점이다.

---

## 1. 먼저 linear 설정을 정확히 이해하기

논문 원문:

> **Linear:** a linear layer is trained to predict class logits from a patch tokens. It is used to produce a low-resolution logit map (eg 32x32 for a model with patch size 16), which is then upsampled to full resolution (512x512) to obtain a segmentation map. This procedure is extremely simple but cannot easily produce high-resolution segmentations.

- ViT는 이미지를 패치 격자로 자르고, 각 패치마다 토큰 하나를 내보낸다.
- linear 설정은 **패치 토큰 하나 → 클래스 logit 하나**를 선형 층으로 맵핑한다. 그러면 패치 격자 크기의 저해상도 logit map(패치 16이면 512/16 = **32×32**, DINOv2의 패치 14면 약 36×36)이 나온다.
- 이 logit map을 512×512로 **단순 업샘플**해서 분할 마스크로 쓴다.
- 한계: 패치 하나가 화면의 16×16(또는 14×14) 픽셀을 통째로 대표하므로 **경계가 뭉개지고, 작은 물체는 놓치기 쉽다**. 논문도 "cannot easily produce high-resolution segmentations"라고 못 박는다.

![Fig. 7: frozen 특징 + 선형 분류기의 정성적 결과 (ADE20K 첫 행)](fig-1.jpeg)

위 그림(Fig. 7)은 linear 설정으로 얻은 정성적 결과다. 첫 행(ADE20K)에서 OpenCLIP-G 마스크는 자잘한 색 조각(artifact)과 끊어진 영역이 많고, DINOv2-g 마스크는 훨씬 깨끗하다. 그러나 DINOv2 결과도 건물·하늘·잔디의 경계가 패치 단위로 계단처럼 거칠게 보인다 — 이것이 저해상도 logit map을 업샘플한 linear 설정의 태생적 한계이고, `+ms`가 보완하려는 지점이다.

---

## 2. `+ms`의 세 가지 강화 요소와 각각이 도움이 되는 이유

논문 원문:

> **+ms:** a boosted version of the linear setup. We concatenate the patch tokens of the 4 last layers, use a larger image resolution of 640, and use multiscale test-time augmentations to improve the predictions.

### (1) 마지막 4개 층의 패치 토큰 concat — 특징 차원 D → 4D

- ViT의 층마다 패치 토큰이 담는 정보가 다르다. **저층(앞쪽 층)**은 위치·색·질감·경계 같은 국소적 정보를 잘 보존하고, **고층(마지막 층)**은 "이건 침대다/벽이다"라는 의미(semantic) 정보에 치우친다.
- 마지막 층만 쓰면 의미는 잘 맞히지만 "어디서 끝나는가"에 약하다. 마지막 4개 층을 이어 붙이면 선형 층이 **의미 정보와 경계·위치 정보를 동시에** 참고할 수 있다.
- 특징 차원은 4배가 된다: ViT-S 384→1536, ViT-B 768→3072, ViT-L 1024→4096, ViT-g 1536→6144. 선형 층의 입력이 커지므로 표현력이 늘지만, 여전히 **선형 층 하나**라는 점은 변하지 않는다.
- 이 아이디어는 DINO/iBOT 계열의 dense 평가에서 이미 관행이었고(마지막 n층 concat), UperNet 같은 디코더가 여러 스테이지의 특징을 합치는 것을 "선형 층 버전"으로 흉내 낸 것이라 볼 수 있다.

### (2) 이미지 해상도 512 → 640 — 패치 격자가 더 촘촘해짐

- 패치 크기는 고정이므로 입력 해상도를 키우면 **패치 개수가 늘어** logit map 해상도가 올라간다.
  - 패치 16 기준: 32×32 → **40×40**
  - DINOv2 패치 14 기준: 약 36×36 → 약 **45×45**
- 픽셀 수로는 (640/512)² ≈ **1.56배** 더 많은 위치에서 예측을 하므로 업샘플 배율이 줄고, 얇은 물체·경계가 더 세밀하게 잡힌다.
- ViT는 위치 임베딩을 보간하면 학습 때와 다른 해상도로도 동작하며, DINOv2는 학습 마지막 단계에서 518 해상도로 짧게 적응(high-resolution adaptation)을 했기 때문에 640 입력에도 특징 품질이 잘 유지된다.

### (3) Multiscale test-time augmentation — 여러 스케일 예측을 평균

- 테스트 시 같은 이미지를 여러 배율(예: 0.5×~1.75× 사이의 여러 값, 좌우 반전 포함)로 리사이즈해 각각 추론하고, 결과 logit을 원 해상도로 되돌려 **평균**한다.
- 큰 스케일에서는 작은 물체·세부 경계가, 작은 스케일에서는 큰 물체의 전체 문맥이 잘 잡히므로 평균하면 서로의 약점을 보완한다. 또한 여러 예측의 앙상블이므로 노이즈가 줄어 마스크가 매끄러워진다.
- 학습은 건드리지 않고 **추론 비용만 늘리는** 방법이다. semantic segmentation 벤치마크에서는 오래된 표준 트릭(mmsegmentation의 `aug_test`)이라 "+ms"라는 약어 자체가 이 관행에서 왔다.

---

## 3. Table 10에서 실제로 얼마나 오르나

Table 10 (frozen 특징, mIoU). 괄호 속은 각 데이터셋의 절대 SOTA.

| Method | Arch. | ADE20k (62.9) lin. | +ms | CityScapes (86.9) lin. | +ms | Pascal VOC (89.0) lin. | +ms |
|---|---|---|---|---|---|---|---|
| OpenCLIP | ViT-G/14 | 39.3 | 46.0 | 60.3 | 70.3 | 71.4 | 79.2 |
| MAE | ViT-H/14 | 33.3 | **30.7** | 58.4 | 61.0 | 67.6 | **63.3** |
| DINO | ViT-B/8 | 31.8 | 35.2 | 56.9 | 66.2 | 66.4 | 75.6 |
| iBOT | ViT-L/16 | 44.6 | 47.5 | 64.8 | 74.5 | 82.3 | 84.3 |
| DINOv2 | ViT-S/14 | 44.3 | 47.2 | 66.6 | 77.1 | 81.1 | 82.6 |
| DINOv2 | ViT-B/14 | 47.3 | 51.3 | 69.4 | 80.0 | 82.5 | 84.9 |
| DINOv2 | ViT-L/14 | 47.7 | 53.1 | 70.3 | 80.9 | 82.1 | 86.0 |
| DINOv2 | ViT-g/14 | 49.0 | 53.0 | 71.3 | 81.0 | 83.0 | 86.2 |

DINOv2에서 linear → +ms 상승폭:

| Arch. | ADE20k | CityScapes | Pascal VOC |
|---|---|---|---|
| ViT-S/14 | +2.9 | +10.5 | +1.5 |
| ViT-B/14 | +4.0 | +10.6 | +2.4 |
| ViT-L/14 | +5.4 | +10.6 | +3.9 |
| ViT-g/14 | +4.0 | +9.7 | +3.2 |

읽어낼 점:

- **CityScapes에서 상승폭이 가장 크다(약 +10)**. CityScapes는 원본이 1024×2048의 넓은 도로 장면이어서 신호등·기둥·보행자 같은 얇고 작은 클래스가 많다. 해상도 확대와 multiscale TTA가 정확히 이런 경우에 효과적이다.
- **MAE는 오히려 떨어진다** (ADE20k 33.3→30.7, VOC 67.6→63.3). MAE 특징은 파인튜닝 없이 선형 분리가 잘 안 되기로 유명하고, 여러 층을 concat해도 선형 층이 활용할 만한 의미 정보가 부족하기 때문이다. 즉 `+ms`는 만능이 아니라 **백본 특징이 이미 좋을 때** 그 잠재력을 끌어내는 도구다.
- ADE20k에서는 ViT-L +ms(53.1)가 ViT-g +ms(53.0)와 사실상 같다 — 이 설정에서는 백본 크기가 이미 포화에 가깁다.

---

## 4. MAE + UperNet 완전 파인튜닝(53.6)과의 비교

논문 원문:

> Interestingly, our evaluation using `+ms` is on par with fully finetuning MAE with an Upernet decoder (53.0 versus 53.6 mIoU). This is surprising because we use a significantly simpler predictor.

- **MAE + UperNet 53.6**: MAE 논문에서 ViT-L 백본 **전체를 파인튜닝**하고, 그 위에 다층 특징을 합치는 FPN 구조의 **UperNet 디코더**(수천만 파라미터)를 붙여 ADE20k에서 얻은 수치.
- **DINOv2 ViT-g +ms 53.0**: 백본은 **frozen**, 예측기는 **선형 층 하나**. 이 둘이 0.6 mIoU 차이밖에 안 난다.
- 의미: 얼린 DINOv2 특징 위에 선형 층만 얹어도, 다른 SSL 모델을 통째로 파인튜닝하고 무거운 디코더를 쓴 결과에 도달한다. 즉 **좋은 dense 특징은 복잡한 디코더의 역할 상당 부분을 대체할 수 있다**는 논문의 주장을 뒷받침하는 수치다.
- 같은 맥락으로, Pascal VOC에서 +ms 86.2는 완전 파인튜닝 기반 SOTA 89.0에 근접하고, ADE20k에서 frozen ViT-g에 Mask2Former + ViT-Adapter를 붙이면 60.2까지 올라가 SOTA 62.9에 가까워진다(Table 10 캡션).

---

## 5. 기억용 정리

- **linear** = 마지막 층 패치 토큰 → 선형 층 → 저해상도 logit map → 업샘플(512).
- **+ms** = linear에 세 가지를 더한 boost 버전:
  1. **마지막 4개 층 concat** (D → 4D): 저층의 경계·위치 + 고층의 의미.
  2. **해상도 640**: 패치 격자 32→40(패치 16 기준), 예측 위치 1.56배.
  3. **multiscale TTA**: 여러 스케일 예측을 평균해 작은 물체·큰 문맥을 모두 잡고 노이즈 감소.
- 예측기는 여전히 선형 층 하나, 백본은 여전히 frozen.
- 효과: DINOv2 ViT-g ADE20k 49.0 → 53.0, CityScapes 71.3 → 81.0, VOC 83.0 → 86.2. MAE는 반대로 하락.
- 53.0(frozen + 선형)은 MAE + UperNet **완전 파인튜닝** 53.6과 사실상 동급.
